# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local text-to-video inference backend (diffusers).

A deliberate sibling of ``DiffusionBackend`` rather than a mode of it: video
pipelines take frame/fps arguments, return frame stacks (plus synchronized audio
for LTX-2) instead of PIL images, and persist MP4s -- none of the image module's
img2img/inpaint/ControlNet/LoRA surface applies. The concurrency skeleton
(load token, per-generation cancel event, split status/generate locks) is copied
from the image backend so the two cannot diverge in lifecycle behaviour, and the
hardware/optimisation layers are IMPORTED from the image stack unchanged:
device/dtype resolution, memory planning + offload tiers, attention backends,
speed profiles (regional torch.compile), and FBCache step caching all operate on
``pipe.transformer`` generically.

Video-specific behaviour lives here:
- the runtime headroom estimate is frames-aware (``estimate_video_runtime_mib``):
  the VAE decode of a whole clip is the memory peak, not the denoise;
- VAE tiling is always enabled (decode of 100+ frames at 720p-class resolutions
  spikes far beyond the image case, and tiling's quality cost is negligible);
- generation snaps num_frames to the family's temporal lattice (k * step + 1)
  and width/height to its required multiple BEFORE latents are allocated;
- the result is encoded to MP4 (H.264) via diffusers' PyAV-backed exporter,
  muxing the audio track for families that produce one.

Loads are gated to trusted repos exactly like the image backend: unsloth/*, the
family's official base repos, or a local path the user explicitly picked.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from .diffusion_attention import apply_attention_backend, select_attention_backend
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_FBCACHE,
    apply_step_cache,
    maybe_toggle_step_cache,
    normalize_transformer_cache,
)
from .diffusion_device import resolve_diffusion_device_target
from .diffusion_memory import (
    apply_memory_plan,
    estimate_gguf_resident_mib,
    estimate_safetensors_dense_mib,
    estimate_video_runtime_mib,
    file_size_mib,
    normalize_memory_mode,
    plan_diffusion_memory,
    snapshot_device_memory,
)
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_OFF,
    apply_speed_optims,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import _QUANT_STEADY_FACTOR, build_resolved_record
from .diffusion_transformer_quant import (
    TQ_AUTO,
    dense_transformer_supported,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)
from .diffusion_precision import TE_QUANT_AUTO, normalize_te_quant, quantize_text_encoders
from .video_families import (
    VIDEO_CANCELLED_MSG,
    VIDEO_NOT_LOADED_MSG,
    VideoFamily,
    default_video_generation_params,
    detect_video_family,
    resolve_video_base_repo,
    snap_num_frames,
    snap_video_size,
    supported_video_family_names,
)
from utils.hardware import clear_gpu_cache

logger = get_logger(__name__)

# Load kinds, mirroring the image backend: "gguf" (single-file GGUF DiT +
# companion base repo), "single_file" (safetensors DiT, e.g. the fp8 LTX-2.3
# checkpoints), "pipeline" (a full diffusers repo via from_pretrained).
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})

# Official vendor base repos allowed to load as full (non-GGUF) artifacts even
# though they are not under unsloth/. Exact-match, lowercased, safetensors-only,
# no remote code -- same bar as the image backend's allowlist.
_TRUSTED_NON_GGUF_VIDEO_REPOS = frozenset(
    {
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "lightricks/ltx-2.3-fp8",
        # Wan2.2 official diffusers base repos (Wan-AI org): safetensors-only, no
        # remote code, so allowed as full (pipeline-kind) loads like the LTX-2 bases.
        "wan-ai/wan2.2-ti2v-5b-diffusers",
        "wan-ai/wan2.2-t2v-a14b-diffusers",
        # HunyuanVideo-1.5 community Diffusers repacks (tencent's own repo is the
        # original non-diffusers layout: config.json, no model_index.json, so it
        # cannot load through HunyuanVideo15Pipeline at all).
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-720p_t2v",
    }
)


def resolve_video_model_kind(gguf_filename: Optional[str], model_kind: Optional[str]) -> str:
    """Classify a load request; explicit model_kind wins, else the filename decides."""
    if model_kind:
        kind = model_kind.strip().lower()
        if kind not in _MODEL_KINDS:
            raise ValueError(
                f"Unknown model_kind '{model_kind}'. Expected one of {sorted(_MODEL_KINDS)}."
            )
        return kind
    if not gguf_filename:
        return "pipeline"
    return "gguf" if gguf_filename.strip().lower().endswith(".gguf") else "single_file"


def _is_trusted_video_repo(repo_id: str) -> bool:
    """Whether a NON-GGUF load may deserialise this repo (see the image twin)."""
    try:
        if Path(repo_id).expanduser().exists():
            return True
    except OSError:
        pass
    rid = repo_id.strip().lower()
    return rid.startswith("unsloth/") or rid in _TRUSTED_NON_GGUF_VIDEO_REPOS


def _picked_gguf_arch(repo_id: str, gguf_filename: str) -> Optional[str]:
    """``general.architecture`` of a picked GGUF, or None. The Video picker admits a GGUF by its
    arch (not its name) -- for a LOCAL dir (``repo_id`` is a directory) AND for a cached HUB repo
    (the cached-gguf listing tags it by arch too), so a renamed/opaquely-named file whose path
    carries no family token still shows up; reading the arch lets the loader resolve the same
    family the picker offered. Reads the local file when present, else the cached hub blob
    (network-free via try_to_load_from_cache). Header-only, bounds-checked."""
    try:
        from pathlib import Path

        path = Path(repo_id).expanduser() / gguf_filename
        if not path.is_file():
            # Not a local dir: resolve a cached HUB blob from the HF cache (no network). The
            # cached-gguf picker only offers already-downloaded repos, so the blob is on disk --
            # but that listing scans the active, legacy, AND default cache roots, so probe all
            # three here or a GGUF cached in a non-active root would be offered yet 400 on load.
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(repo_id, gguf_filename)
            if not isinstance(cached, str):
                from hub.utils.paths import hf_default_cache_dir, legacy_hf_cache_dir
                for root_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
                    try:
                        cached = try_to_load_from_cache(
                            repo_id, gguf_filename, cache_dir = str(root_fn())
                        )
                    except Exception:  # noqa: BLE001 -- a bad/absent root just falls through
                        cached = None
                    if isinstance(cached, str):
                        break
            if not isinstance(cached, str):
                return None
            path = Path(cached)
        from utils.models.gguf_metadata import read_gguf_general_metadata

        arch = (read_gguf_general_metadata(str(path)) or {}).get("general.architecture")
        return arch.strip() if isinstance(arch, str) and arch.strip() else None
    except Exception:  # noqa: BLE001 -- a header read glitch just falls through to name detection
        return None


class _VideoGenerationCancelled(Exception):
    """Unwinds a denoise loop that has no cooperative interrupt (no step callback);
    generate() maps it to the VIDEO_CANCELLED_MSG sentinel the routes 409 on."""


@contextlib.contextmanager
def _scheduler_step_progress(pipe: Any, on_step: Any):
    """Progress + cancellation for pipelines WITHOUT callback_on_step_end.

    HunyuanVideo15Pipeline exposes no per-step callback, but every denoise step
    makes exactly one ``scheduler.step`` call, so wrapping that method gives the
    same per-step tick the callback path gets. ``on_step`` receives the 1-based
    step count and may raise (_VideoGenerationCancelled) to abort the loop. The
    original method is always restored, even when the pipeline raises.
    """
    scheduler = pipe.scheduler
    original = scheduler.step
    count = {"n": 0}

    def _step(*args: Any, **kwargs: Any) -> Any:
        count["n"] += 1
        on_step(count["n"])
        return original(*args, **kwargs)

    scheduler.step = _step
    try:
        yield
    finally:
        scheduler.step = original


def _detect_load_family(
    repo_id: str, gguf_filename: Optional[str], family_override: Optional[str]
) -> Optional[VideoFamily]:
    """Family detection shared by validate_load_request and the load worker: the
    repo id first, then the picked filename -- a local directory or generically
    named repo often carries the family token only in the checkpoint filename,
    and the worker must resolve the same family the validator accepted."""
    fam = detect_video_family(repo_id, family_override) or (
        detect_video_family(f"{repo_id}/{gguf_filename}")
        if gguf_filename and not family_override
        else None
    )
    if fam is None and gguf_filename and not family_override:
        # The picker admits a GGUF (local dir OR cached hub repo) by its general.architecture, but
        # its path/name may carry no whole-segment family token (e.g. a renamed "model.gguf"), so
        # the name-based detection above misses it. Resolve the same family the picker offered by
        # reading the arch -- its string ("ltxv") is a family alias. A video arch with no backend
        # family (e.g. "wan") still yields None, so an unsupported pick 400s exactly as before.
        arch = _picked_gguf_arch(repo_id, gguf_filename)
        if arch:
            fam = detect_video_family(repo_id, override = arch)
    return fam


def _ensure_mp4_encoder_available() -> None:
    """Fail a load fast when PyAV is missing: the export otherwise dies AFTER a
    multi-minute denoise, which is the worst possible time to learn about it."""
    try:
        import av  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- any import failure means no encoder
        raise ValueError(
            "Video generation needs the 'av' package (PyAV) to encode MP4s. "
            "Install it with: pip install av"
        ) from exc


@dataclass(frozen = True)
class _VideoLoadState:
    """Everything about the currently-loaded video pipeline, swapped as one unit."""

    pipe: Any
    family: VideoFamily
    repo_id: str
    base_repo: str
    device: str
    dtype: str
    kind: str
    gguf_filename: Optional[str] = None
    offload_policy: str = "none"
    vae_tiling: bool = True
    memory_mode: str = "auto"
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    backend_flags: Optional[dict] = None
    attention_backend: Optional[str] = None
    transformer_cache: Optional[str] = None
    # True when the cache decision was AUTO on a cache-capable DiT: generate() then
    # re-checks the actual step count and toggles FBCache across FBCACHE_MIN_STEPS.
    # An explicit request (off / fbcache) is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Dense transformer quant actually engaged ("int8" | "fp8" | "nvfp4" | "mxfp8") or
    # None. Mirrors the image backend's _LoadState.transformer_quant: on a pipeline-kind
    # load the dense DiT(s) can be torchao-quantised in place onto the low-precision
    # tensor cores; None means they run at their loaded (bf16) precision.
    transformer_quant: Optional[str] = None
    # Text-encoder quant actually engaged ("fp8" | "fp8_dynamic" | "int8" | "nvfp4") or None.
    # The companion text encoder (UMT5 / Gemma3 / Qwen2.5-VL) loads dense bf16 and is often the
    # largest resident component; this shrinks it in place, mirroring the image backend.
    text_encoder_quant: Optional[str] = None
    resolved: Optional[dict] = None


@dataclass
class _VideoLoadingState:
    repo_id: str
    base_repo: str
    expected_bytes: Optional[int] = None
    error: Optional[str] = None


def _progress(phase: Optional[str], **extra: Any) -> dict[str, Any]:
    return {"phase": phase, **extra}


# ── dual-DiT (Wan2.2-A14B MoE) helpers ────────────────────────────────────────
#
# The imported optimisation helpers (apply_speed_optims / apply_attention_backend /
# apply_step_cache) and the dense quantiser all read ``pipe.transformer`` and act on
# that ONE denoiser -- correct for every single-DiT family (LTX-2, Wan2.2-TI2V-5B).
# Wan2.2-A14B is a dual-expert MoE: ``transformer`` handles the high-noise steps and
# ``transformer_2`` the low-noise steps (pipeline_wan.py routes by boundary_ratio), so
# an optimisation applied only to ``transformer`` would leave the second expert eager /
# unquantised / on the wrong attention kernel for half the schedule. Rather than fork
# each helper, present the second DiT to them AS ``pipe.transformer`` via a thin proxy
# and call the helper a second time, so the helpers stay untouched and single-DiT loads
# are bit-identical (the proxy is only built for is_moe families).


def _transformer_names(pipe: Any, fam: VideoFamily) -> tuple[str, ...]:
    """Attribute names of the denoiser(s) on ``pipe`` to optimise. Just
    ("transformer",) for a single-DiT family; also "transformer_2" for an MoE family
    whose second expert is actually present (a checkpoint may ship only the first)."""
    names = ["transformer"]
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        names.append("transformer_2")
    return tuple(names)


class _SecondDiTView:
    """A thin proxy that makes ``pipe.transformer_2`` look like ``pipe.transformer`` to a
    helper that hardcodes ``getattr(pipe, "transformer")``, while every other attribute
    (vae, components, __call__, ...) reads through to the real pipe unchanged.

    This lets the existing single-DiT helpers optimise the second expert without a fork:
    ``apply_speed_optims(_SecondDiTView(pipe), ...)`` compiles / caches / sets attention on
    ``transformer_2``. Only ever wrapped around an MoE pipe (guarded by fam.is_moe)."""

    def __init__(self, pipe: Any) -> None:
        # Store on the instance dict under a name __getattr__ never fires for.
        object.__setattr__(self, "_pipe", pipe)

    @property
    def transformer(self) -> Any:
        return self._pipe.transformer_2

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the instance/class (i.e. not
        # ``transformer`` / ``_pipe``), so everything else delegates to the real pipe.
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes must land on the real pipe, or a helper's side effect (for example
        # reassigning the transformer it optimised) would vanish with the view.
        # ``transformer`` mirrors the read property onto the second expert.
        pipe = object.__getattribute__(self, "_pipe")
        setattr(pipe, "transformer_2" if name == "transformer" else name, value)


def _views_for(pipe: Any, fam: VideoFamily) -> tuple[Any, ...]:
    """The pipe view(s) to pass through the ``getattr(pipe, "transformer")`` helpers so
    they cover every denoiser: the real pipe (its ``transformer``), plus a
    ``_SecondDiTView`` (its ``transformer_2``) for a dual-DiT MoE family. A single-DiT
    load returns just ``(pipe,)``, so its behaviour is unchanged."""
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        return (pipe, _SecondDiTView(pipe))
    return (pipe,)


class VideoBackend:
    """One loaded video pipeline; loads swap it atomically (same model as images)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._state: Optional[_VideoLoadState] = None
        self._loading: Optional[_VideoLoadingState] = None
        self._load_token = 0
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        # Generation progress, written by the step callback / phase transitions.
        self._gen: dict[str, Any] = {"active": False}

    # ── validation ───────────────────────────────────────────────────────────

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # A -GGUF repo picked without a quant filename resolves to the pipeline
        # kind and would only fail minutes later in from_pretrained (no
        # model_index.json), AFTER the route evicted the current GPU owner.
        # Reject it here, where failing is still free.
        if kind == "pipeline" and repo_id.strip().lower().rstrip("/").endswith("-gguf"):
            raise ValueError(
                f"'{repo_id}' is a GGUF repo: pick one of its .gguf files "
                "(gguf_filename) instead of loading it as a diffusers pipeline."
            )
        fam = _detect_load_family(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported text-to-video model. Supported families: "
                f"{', '.join(supported_video_family_names())}. If this is a variant of one "
                f"of them, pass family_override with that family name."
            )
        if kind != "gguf" and not _is_trusted_video_repo(repo_id):
            raise ValueError(
                f"Non-GGUF video loads are limited to unsloth/* repos, the official "
                f"family base repos, and local paths; '{repo_id}' is neither."
            )
        # The companions load with from_pretrained too, so an explicit base repo is
        # held to the same bar as a non-GGUF repo id: a GGUF pick must not smuggle
        # in an arbitrary remote base.
        if base_repo and (base_repo or "").strip() and not _is_trusted_video_repo(base_repo):
            raise ValueError(
                f"base_repo is limited to unsloth/* repos, the official family base "
                f"repos, and local paths; '{base_repo}' is neither."
            )
        # An existing LOCAL base_repo loads as a full pipeline (from_pretrained(base) / config=base),
        # which needs a model_index.json. The pipeline-kind shape check below covers only repo_id,
        # and an explicit base_repo is only meaningful for gguf/single_file kinds, so a non-pipeline
        # local base would otherwise pass here and fail deep in the background load AFTER the route
        # evicted the resident model. Shared helper, so image/video/training stay in sync.
        from core.inference.diffusion import _assert_local_base_is_pipeline

        _assert_local_base_is_pipeline(base_repo)
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        if kind in ("gguf", "single_file") and fam.is_moe:
            # A single checkpoint carries only one expert; the pipeline would then pull
            # the other expert dense bf16 from the base repo, outside the memory plan.
            raise ValueError(
                f"'{fam.name}' is a dual-expert model: a single {kind} file covers only "
                f"one of its two transformers. Load the diffusers pipeline repo "
                f"('{fam.base_repo}') instead."
            )
        # A local checkpoint that cannot exist must fail HERE, before the route evicts
        # a resident chat/image model for a load that dies at resolve time.
        if kind in ("gguf", "single_file"):
            # Fail a kind/extension mismatch before the GPU handoff instead of deep in the
            # background loader: a "gguf" load needs a .gguf file, a "single_file" load must not be
            # handed a .gguf and must name an actual .safetensors checkpoint. Mirrors the image
            # loader's kind/extension gate in diffusion.validate_load_request.
            is_gguf_name = (gguf_filename or "").lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            if kind == "single_file" and not (gguf_filename or "").lower().endswith(".safetensors"):
                raise ValueError(
                    f"'{gguf_filename}' is not a loadable single-file checkpoint "
                    f"(expected a .safetensors name; use a .gguf name for a GGUF load)."
                )
            root = Path(repo_id).expanduser()
            # POSIX path-shaped, a "."/".." prefix (covers ./ ../ and Windows .\ ..\), a Windows
            # separator anywhere (never in a bare "org/name" id), or an absolute path on this OS
            # (covers Windows C:\ / C:/). Mirrors the image loader so a missing Windows-shaped
            # local pick fails before the GPU handoff instead of being treated as a Hub repo.
            path_shaped = (
                repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or root.is_absolute()
            )
            if root.is_dir():
                from .diffusion_families import resolve_local_gguf_child
                try:
                    resolve_local_gguf_child(root, gguf_filename or "")
                except Exception as exc:  # noqa: BLE001 -- surface as client input error
                    raise ValueError(str(exc)) from exc
            elif root.is_file():
                # The loader hands a local FILE straight to the gguf/single_file loader
                # (_resolve_checkpoint_path returns the file itself, ignoring gguf_filename),
                # so the file's OWN suffix must match the kind. Otherwise a .gguf picked as
                # single_file (or a .safetensors picked as gguf) slips past the gguf_filename
                # checks above, evicts the resident model in the route, and only then fails
                # in from_single_file / the GGUF reader. Reject it here, before the handoff.
                suffix = root.suffix.lower()
                if kind == "gguf" and suffix != ".gguf":
                    raise ValueError(
                        f"Local checkpoint '{repo_id}' is not a .gguf file; a 'gguf' load "
                        f"needs a .gguf checkpoint."
                    )
                if kind == "single_file" and suffix != ".safetensors":
                    raise ValueError(
                        f"Local checkpoint '{repo_id}' is not a .safetensors file; a "
                        f"'single_file' load needs a .safetensors checkpoint."
                    )
            elif path_shaped:
                raise ValueError(f"Local model path '{repo_id}' does not exist.")
        # A local pipeline pick must be a real diffusers directory (model_index.json), or it
        # would only fail deep in from_pretrained AFTER the route evicted the resident model.
        # Mirrors the image loader's local-pipeline shape check in diffusion.validate_load_request.
        if kind == "pipeline":
            root = Path(repo_id).expanduser()
            # Gate on .exists() (not .is_dir()) so a local FILE picked as a pipeline is rejected
            # too: a bare .safetensors file is not a diffusers directory, so from_pretrained would
            # still fail in the background load after the eviction. Mirrors the image loader, which
            # uses .exists() here.
            if root.exists() and not (root.is_dir() and (root / "model_index.json").is_file()):
                raise ValueError(
                    f"Local pipeline path is not a diffusers directory "
                    f"(no model_index.json): {repo_id}"
                )
        # Reject a malformed transformer_quant scheme cheaply, before the GPU handoff
        # (normalize_transformer_quant raises ValueError on an unknown scheme). It applies
        # only on pipeline-kind loads (the dense DiT from the base repo); an ignored value
        # on a gguf/single_file load is left to the loader, matching the image backend.
        normalize_transformer_quant(transformer_quant)
        # Reject a malformed text_encoder_quant the same way (applies to any load kind: the dense
        # text encoder is resident for pipeline / gguf / single_file alike).
        normalize_te_quant(text_encoder_quant)
        _ensure_mp4_encoder_available()
        return fam

    # ── background load + progress ───────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        model_kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            base_repo = base_repo,
            family_override = family_override,
            model_kind = model_kind,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
        )
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A video load is already in progress.")
            self._load_token += 1
            token = self._load_token
            self._cancel_event.clear()
            self._loading = _VideoLoadingState(repo_id = repo_id, base_repo = fam.base_repo)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base_repo = base_repo,
                family_override = family_override,
                hf_token = hf_token,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                transformer_quant = transformer_quant,
                text_encoder_quant = text_encoder_quant,
                model_kind = model_kind,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            fam = _detect_load_family(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_video_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            base = (
                kwargs["repo_id"]
                if kind == "pipeline"
                else resolve_video_base_repo(fam, kwargs.get("base_repo"))
            )
            kwargs["base_repo"] = base
            expected = self._estimate_download_bytes(
                kwargs["repo_id"], kwargs.get("gguf_filename"), base, kwargs.get("hf_token"), kind
            )
            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.expected_bytes = expected
            # The GGUF/single-file checkpoint downloads outside the lock so an
            # unload/eviction can preempt the multi-GB pull; the pipeline
            # companions pre-download the same way (scoped file list, cancellable,
            # resumes from the cache so a cancelled pull costs nothing).
            checkpoint_local: Optional[Path] = None
            if kwargs.get("gguf_filename") and not Path(kwargs["repo_id"]).expanduser().exists():
                from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
                checkpoint_local = Path(
                    hf_hub_download_with_xet_fallback(
                        kwargs["repo_id"],
                        kwargs["gguf_filename"],
                        kwargs.get("hf_token"),
                        cancel_event = self._cancel_event,
                    )
                )
            # An LTX-2.3 checkpoint replaces the base VAEs/vocoder/connectors too, so
            # its base pull shrinks to scheduler + text encoder + tokenizer; the
            # estimate is recomputed to match (detectable only once the checkpoint
            # header is on disk, hence after the pull above).
            ltx23 = False
            if fam is not None and fam.name == "ltx-2" and kind != "pipeline":
                from .video_ltx2 import is_ltx23_checkpoint

                probe = checkpoint_local
                if probe is None:
                    # Local repos: a bare file, or a directory whose child the same
                    # resolver load_pipeline uses picks out. Unresolvable here means
                    # load_pipeline will surface the real error; keep the wide pull.
                    root = Path(kwargs["repo_id"]).expanduser()
                    if root.is_file():
                        probe = root
                    elif root.is_dir():
                        try:
                            probe = self._resolve_checkpoint_path(
                                kwargs["repo_id"],
                                kwargs.get("gguf_filename"),
                                kwargs.get("hf_token"),
                            )
                        except Exception:  # noqa: BLE001 -- surfaced by load_pipeline
                            probe = None
                ltx23 = probe is not None and is_ltx23_checkpoint(probe)
                if ltx23:
                    expected = self._estimate_download_bytes(
                        kwargs["repo_id"],
                        kwargs.get("gguf_filename"),
                        base,
                        kwargs.get("hf_token"),
                        kind,
                        ltx23 = True,
                    )
                    with self._lock:
                        if self._load_token == token and self._loading is not None:
                            self._loading.expected_bytes = expected
            base_local = self._predownload_base(base, kwargs.get("hf_token"), kind, ltx23 = ltx23)
            # The 2.3 assembly pulls per component from the hub id (its snapshot here
            # deliberately lacks the base VAEs), so it only gets the warmed cache; the
            # generic from_pretrained paths get the complete local snapshot.
            kwargs["_base_local_dir"] = None if ltx23 else base_local
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            # A failed or cancelled load never commits _VideoLoadState, so the
            # teardown path has no snapshot to restore: roll back the process-wide
            # speed globals here (token-scoped, so a superseded load cannot clobber
            # the globals a newer in-flight load now owns).
            self._rollback_precommit_globals(token)
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def _rollback_precommit_globals(self, token: Optional[int]) -> None:
        """Restore process-wide speed globals (cudnn.benchmark / TF32 / the compiled
        GGUF dequantizer) for a load that died BEFORE committing _VideoLoadState.
        _teardown_state only restores from the committed state's snapshot, so an
        uncommitted load would otherwise leak its profile into the next speed=off
        load. Token-scoped: when a newer load has already taken the snapshot slot,
        the stale worker must leave the globals alone."""
        stored = getattr(self, "_precommit_globals", None)
        if stored is None:
            return
        stored_token, flags = stored
        if token is not None and stored_token is not None and stored_token != token:
            return
        self._precommit_globals = None
        restore_backend_flags(flags)
        from . import diffusion_gguf_compile

        diffusion_gguf_compile.uninstall_all()

    # Base-repo subfolders an LTX-2.3 assembly reads: the checkpoint (plus the GGUF
    # repo's extras files) supplies the DiT, connectors, both VAEs and the vocoder,
    # so only the 2.0 base's scheduler / text encoder / tokenizer are pulled.
    _LTX23_BASE_PREFIXES = ("scheduler/", "text_encoder/", "tokenizer/")

    @staticmethod
    def _base_download_files(
        info: Any,
        kind: str,
        *,
        ltx23: bool = False,
    ) -> list[tuple[str, int]]:
        """The (rfilename, size) list a load actually needs from the base repo.

        Single source of truth for the progress estimate AND the scoped pre-download,
        so the two can never disagree. Excluded on purpose:
        - root-level packaged checkpoints (ComfyUI-style singles; 170 GB of the LTX-2
          repo) -- the diffusers pipeline only reads per-component subfolders;
        - the duplicate ``text_encoder/diffusion_pytorch_model*`` shard set (the LTX-2
          base repo ships its text encoder twice; transformers loads the ``model-*``
          naming via the shard index);
        - ``transformer/`` when a GGUF/single-file checkpoint replaces the DiT;
        - everything but scheduler / text encoder / tokenizer for an LTX-2.3
          checkpoint (``ltx23``), whose VAEs/vocoder/connectors come from the
          checkpoint and its extras, not the 2.0 base."""
        files: list[tuple[str, int]] = []
        for sibling in info.siblings or []:
            name, size = sibling.rfilename, sibling.size or 0
            # .jinja: tokenizer/chat_template.jinja ships as a standalone file in the
            # LTX-2 and HunyuanVideo-1.5 repos (not embedded in tokenizer_config.json)
            # and apply_chat_template needs it at generation time, so a snapshot
            # without it loads fine and then crashes the first generation.
            if not name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja")):
                continue
            if "/" not in name and name.endswith(".safetensors"):
                continue
            if kind != "pipeline" and name.startswith("transformer/"):
                continue
            if name.startswith("text_encoder/diffusion_pytorch_model"):
                continue
            if ltx23 and "/" in name and not name.startswith(VideoBackend._LTX23_BASE_PREFIXES):
                continue
            files.append((name, int(size)))
        return files

    def _estimate_download_bytes(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        hf_token: Optional[str],
        kind: str,
        ltx23: bool = False,
    ) -> Optional[int]:
        """Total bytes this load will pull (checkpoint + companions), or None."""
        try:
            from huggingface_hub import HfApi

            total = 0
            api = HfApi(token = hf_token or None)
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True)
                for sibling in info.siblings or []:
                    if sibling.rfilename == gguf_filename and sibling.size:
                        total += int(sibling.size)
            if base and not Path(base).expanduser().exists():
                info = api.model_info(base, files_metadata = True)
                total += sum(size for _, size in self._base_download_files(info, kind, ltx23 = ltx23))
            return total or None
        except Exception:  # noqa: BLE001 -- progress totals are best-effort only
            return None

    def _predownload_base(
        self,
        base: str,
        hf_token: Optional[str],
        kind: str,
        *,
        ltx23: bool = False,
    ) -> Optional[str]:
        """Pull exactly the base-repo files the load needs; return the local snapshot dir.

        A bare ``from_pretrained(repo_id)`` snapshot of Lightricks/LTX-2 downloads the
        whole 314 GB repo (root packaged checkpoints plus a second 50 GB text-encoder
        shard set) when ~93 GB is used. Downloading the scoped file list ourselves is
        also cancellable per file, and handing the local dir to from_pretrained skips
        diffusers' own expected-files sweep. None -> caller keeps the hub id (local
        path, non-diffusers layout, or any metadata failure: from_pretrained then
        resolves the repo exactly as before)."""
        try:
            if not base or Path(base).expanduser().exists():
                return None
            from huggingface_hub import HfApi

            info = HfApi(token = hf_token or None).model_info(base, files_metadata = True)
            files = self._base_download_files(info, kind, ltx23 = ltx23)
            if not any(name == "model_index.json" for name, _ in files):
                return None
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

            snapshot_root: Optional[Path] = None
            for name, _ in files:
                # Explicit per-file check: a fully-cached file returns without ever
                # consulting the event, so a warm-cache sweep would otherwise run to
                # completion after an unload already cancelled this load.
                if self._cancel_event.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                local = Path(
                    hf_hub_download_with_xet_fallback(
                        base, name, hf_token, cancel_event = self._cancel_event
                    )
                )
                if name == "model_index.json":
                    snapshot_root = local.parent
            return str(snapshot_root) if snapshot_root is not None else None
        except Exception as exc:  # noqa: BLE001 -- fall back to from_pretrained's own pull
            if self._cancel_event.is_set():
                raise
            logger.warning("video.predownload_fallback: %s", exc)
            return None

    def _cache_bytes(self, repo_id: Optional[str]) -> int:
        """Bytes of ``repo_id`` currently in the HF blob cache (progress polling)."""
        if not repo_id:
            return 0
        try:
            from huggingface_hub import scan_cache_dir
            rid = repo_id.strip()
            for repo in scan_cache_dir().repos:
                if repo.repo_id == rid:
                    return int(repo.size_on_disk)
        except Exception:  # noqa: BLE001 -- cache scan is best-effort
            return 0
        return 0

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        downloaded = self._cache_bytes(loading.repo_id)
        if loading.base_repo and loading.base_repo != loading.repo_id:
            downloaded += self._cache_bytes(loading.base_repo)
        expected = loading.expected_bytes
        phase = "downloading"
        if expected and downloaded >= expected:
            phase = "finalizing"
            # The cache scan counts every blob of the repo(s), including files a
            # previous (or broader) pull left behind that this load never reads, so
            # the raw counter can exceed the scoped estimate. Clamp: everything the
            # load needs is present, which is what the bar reports.
            downloaded = expected
        return _progress(
            phase,
            downloaded_bytes = int(downloaded),
            expected_bytes = int(expected) if expected else None,
        )

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base) would yank
        blobs and snapshot files from under the download/assembly. Mirrors the image
        backend's guard (DiffusionBackend.loading_repo_ids)."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            return tuple(r for r in (loading.repo_id, loading.base_repo) if r)

    # ── the load itself ──────────────────────────────────────────────────────

    def load_pipeline(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        model_kind: Optional[str] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        import diffusers
        import torch

        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            base_repo = base_repo,
            family_override = family_override,
            model_kind = model_kind,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # text_encoder_quant tri-state (mirrors the image backend + transformer_quant): UNSET
        # (None / "") -> auto (pick the best accurate TE scheme for this GPU + family); an explicit
        # "none"/"off" pins the encoder dense; a scheme forces it. So the shipped default is auto.
        if text_encoder_quant is None or str(text_encoder_quant).strip() == "":
            text_encoder_quant = TE_QUANT_AUTO
        base = repo_id if kind == "pipeline" else resolve_video_base_repo(fam, base_repo)

        with self._lock:
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Video load was cancelled or superseded.")
            # Signal only a generation from the PREVIOUS model; the token check
            # above already bailed a superseded worker before this point.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        # Wait for the signalled generation to actually exit before tearing the old
        # pipeline down: the denoise loop holds its own pipe reference until the
        # next step callback, and freeing/reallocating under it would put two
        # models in VRAM at once. generate() holds _generate_lock for its full
        # body, so a bare acquire is the exit barrier (never while holding _lock).
        with self._generate_lock:
            pass
        # The barrier wait can outlive this load: an unload or a newer load may
        # have superseded it while blocked, and tearing down now would destroy
        # the model that should remain current (or waste minutes building a
        # pipeline nobody wants). Recheck before touching shared state.
        if _load_token is not None and _load_token != self._load_token:
            raise RuntimeError("Video load was cancelled or superseded.")
        self._teardown_state()

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows them, so a resolved fp16
        # promotes to float32 (the same rule as the fp16-incompatible image
        # families). CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32
        # The size tables below are bf16 (2-byte) figures. When the promotion
        # above lands fp32 weights on an accelerator (a pre-bf16 GPU), every
        # dense estimate doubles; budgeting the 2-byte figure would let auto
        # pick a resident plan that OOMs inside from_pretrained. GGUF weights
        # stay quantised on disk and in memory, so only dense estimates scale.
        dtype_scale = 2.0 if device != "cpu" and dtype is torch.float32 else 1.0

        # Precision tri-state, mirroring the image backend: an UNSET request (or
        # "auto") hands the decision to the hardware ladder -- on a dense-capable
        # GPU the quantised DiT (int8 minimum, fp8 on data-center silicon) is
        # faster at the same resident-or-better footprint. An explicit
        # "none"/"off" pins dense bf16 and an explicit scheme pins that scheme.
        # Only the pipeline kind can engage it (gguf/single_file checkpoints
        # already carry their own precision), and the offload guard below still
        # skips it when the plan moves the DiT.
        if transformer_quant is None or str(transformer_quant).strip().lower() in (
            "",
            "auto",
        ):
            # An explicit Speed="off" (bit-exact) load must stay dense bf16: promoting the unset
            # precision to auto-quant here would engage int8/fp8 + regional compile and silently
            # break the user's bit-exact request (an auto DEFAULT overriding an EXPLICIT control),
            # and the quant path below would then also force effective_speed back to default.
            # Suppress the auto default when speed was explicitly pinned off, mirroring the image
            # backend (diffusion.py); otherwise auto (the dense-capable default) applies. "off"
            # normalizes to None (no dense quant), keeping the dense bf16 path.
            speed_off = speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
            transformer_quant = "off" if speed_off else TQ_AUTO

        # ── memory plan: family-table resident estimate + frames-aware headroom.
        device_memory = snapshot_device_memory(target)
        components = fam.bf16_components_gb
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        if kind == "pipeline":
            model_dense_mib = (
                int(sum(components) * mib_per_gb * dtype_scale) if components is not None else None
            )
            companion_mib = None
        else:
            checkpoint_path = self._resolve_checkpoint_path(repo_id, gguf_filename, hf_token)
            size_mib = file_size_mib(str(checkpoint_path))
            model_dense_mib = None
            if kind == "gguf":
                transformer_mib = estimate_gguf_resident_mib(size_mib)
            else:
                transformer_mib = estimate_safetensors_dense_mib(size_mib)
                if transformer_mib is not None:
                    transformer_mib = int(transformer_mib * dtype_scale)
            companion_mib = (
                int((components[1] + components[2]) * mib_per_gb * dtype_scale)
                if components is not None
                else None
            )
            # The resident check budgets ALL weights (the image backend's contract):
            # the companions stay resident even when only the transformer would fit,
            # so budgeting the transformer alone lets auto pick OFFLOAD_NONE and OOM
            # while from_pretrained loads the text encoder / VAEs.
            model_dense_mib = (
                transformer_mib + (companion_mib or 0) if transformer_mib is not None else None
            )
        runtime_mib = estimate_video_runtime_mib(
            width = fam.resolution_presets[0][0],
            height = fam.resolution_presets[0][1],
            num_frames = fam.default_num_frames,
        )
        plan = plan_diffusion_memory(
            target = target,
            device_memory = device_memory,
            model_dense_mib = model_dense_mib,
            runtime_headroom_mib = runtime_mib,
            companion_dense_mib = companion_mib,
            requested_mode = normalize_memory_mode(memory_mode),
        )
        # Parity with the image dense-quant path: the bf16-table plan can force offload
        # a quantised DiT would not need (offload also disables quant entirely). Re-plan
        # with the scheme's steady factor and keep the resident placement when it fits;
        # if quantisation later fails, the load falls back to this bf16 plan.
        bf16_plan = plan
        quant_replanned = False
        if (
            kind == "pipeline"
            and plan.offload_policy != "none"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and components is not None
        ):
            scheme_preview = select_transformer_quant_scheme(
                target, transformer_quant, family = fam.name
            )
            factor = _QUANT_STEADY_FACTOR.get(scheme_preview) if scheme_preview else None
            if factor is not None:
                quant_mib = int(
                    (components[0] * factor + components[1] + components[2]) * mib_per_gb
                )
                replanned = plan_diffusion_memory(
                    target = target,
                    device_memory = device_memory,
                    model_dense_mib = quant_mib,
                    runtime_headroom_mib = runtime_mib,
                    companion_dense_mib = None,
                    requested_mode = normalize_memory_mode(memory_mode),
                )
                if replanned.offload_policy == "none":
                    logger.info(
                        "video.transformer_quant: %s fits resident (%d MiB steady); "
                        "dropping the bf16 plan's '%s' offload",
                        scheme_preview,
                        quant_mib,
                        plan.offload_policy,
                    )
                    plan = replanned
                    quant_replanned = True

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if getattr(fam, "vae_force_fp32", False):
            # Wan's VAE must decode in float32, but a scalar torch_dtype casts EVERY component
            # (VAE included) to the pipe dtype during load -- AutoencoderKLWan has no
            # _keep_in_fp32_modules, so from_pretrained truncates its fp32 weights to bf16 and a
            # later .to(float32) only widens the already-lossy values (banding / black frames).
            # diffusers >= 0.39 takes a per-component dtype dict, so load the VAE at fp32 directly;
            # "default" MUST be set or unlisted components fall back to fp32 (over-widening the DiT).
            pipe_kwargs["torch_dtype"] = {"vae": torch.float32, "default": dtype}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if kind == "pipeline":
            # The pre-downloaded snapshot dir keeps from_pretrained off the hub (its
            # own snapshot sweep would also pull the repo's packaged root checkpoints
            # and duplicate text-encoder shards); hub id when pre-download was skipped.
            pipe = pipeline_cls.from_pretrained(_base_local_dir or repo_id, **pipe_kwargs)
        else:
            transformer_cls = getattr(diffusers, fam.transformer_class)
            # checkpoint_path was already resolved (and downloaded) by the memory
            # planning branch above for every non-pipeline kind.
            sf_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "config": base,
                "subfolder": "transformer",
                "token": hf_token,
            }
            if kind == "gguf":
                sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                    compute_dtype = dtype
                )
            from .video_ltx2 import is_ltx23_checkpoint, load_ltx23_pipeline

            if fam.name == "ltx-2" and is_ltx23_checkpoint(checkpoint_path):
                # 2.3 checkpoints need the full assembly: new transformer config
                # flags, key renames the stock converter lacks, and the 2.3
                # connectors/VAEs/vocoder the 2.0 base repo does not carry.
                pipe = load_ltx23_pipeline(
                    checkpoint_path,
                    base_repo = base,
                    torch_dtype = dtype,
                    is_gguf = kind == "gguf",
                    hf_token = hf_token,
                )
            else:
                transformer = transformer_cls.from_single_file(str(checkpoint_path), **sf_kwargs)
                pipe = pipeline_cls.from_pretrained(
                    _base_local_dir or base, transformer = transformer, **pipe_kwargs
                )

        # The per-component torch_dtype above already loads the Wan VAE at float32 (bf16_components_gb
        # budgets it at that fp32 size, so the memory plan stays consistent). Belt-and-suspenders for
        # any path that bypassed the dict (e.g. a passed-in vae=): re-pin an fp32-force VAE that came
        # back at a lower precision. This is a no-op on the primary path (the load already fp32'd it).
        if getattr(fam, "vae_force_fp32", False):
            vae = getattr(pipe, "vae", None)
            if vae is not None and getattr(vae, "dtype", None) is not torch.float32:
                vae.to(torch.float32)

        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")

        # For a dual-DiT MoE family (Wan2.2-A14B), every optimisation site below must
        # cover BOTH experts: ``views`` is (pipe, _SecondDiTView(pipe)) so a helper that
        # reads ``pipe.transformer`` runs once per denoiser. A single-DiT load resolves to
        # (pipe,), so it behaves exactly as before.
        views = _views_for(pipe, fam)

        # ── dense transformer quant (opt-in, pipeline-kind only): load the dense bf16
        # DiT from the base repo and torchao-quantise it in place onto the low-precision
        # tensor cores, mirroring the image backend's transformer_quant fast path. Only
        # the pipeline kind materialises the dense weights (gguf/single_file already carry
        # their own precision), and only on CUDA + bf16. Best-effort: any failure leaves
        # the DiT dense. Quant must precede compile (dynamic quant is ~30x slower eager),
        # so it runs before apply_speed_optims below -- same order as diffusion.py.
        transformer_quant_engaged: Optional[str] = None
        quant_skipped_for_offload = False
        if (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and plan.offload_policy != "none"
        ):
            # Offload hooks move modules with Module.to(), which torchao quantized
            # tensors reject (aten._has_compatible_shallow_copy_type is
            # unimplemented) -- observed as a hard crash on the Wan2.2-A14B gate
            # run, where the 114 GB dual DiT plans model offload. A dense DiT
            # under offload beats a crashed one, so quant is skipped, surfaced in
            # the resolved record, and the user can force it by pinning a
            # resident memory mode.
            logger.info(
                "video.transformer_quant: skipped (offload policy '%s' moves the "
                "DiT via Module.to(), unsupported for torchao quantized tensors); "
                "pin a resident memory mode to combine quant with this model",
                plan.offload_policy,
            )
            quant_skipped_for_offload = True
        elif (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
        ):
            engaged = []
            for view in views:
                # quantize_transformer reads ``pipe.transformer`` and returns the scheme it
                # engaged (or None); pass each expert's view so both DiTs are quantised with
                # the same arch-chosen scheme. The family name drives the per-family deny
                # table (_FAMILY_SCHEME_DENY) exactly as on the image side.
                scheme = quantize_transformer(
                    view,
                    target,
                    mode = transformer_quant,
                    family = fam.name,
                    logger = logger,
                )
                if scheme is not None:
                    engaged.append(scheme)
            # Quant must engage on every DiT or none: the first expert is mutated in
            # place, so a second-expert failure cannot fall back to dense (the schedule
            # would run at mismatched precision with quant reported off). Fail the load
            # cleanly instead; a full miss (nothing engaged) stays best-effort dense.
            if engaged and len(engaged) < len(views):
                del pipe
                clear_gpu_cache()
                raise RuntimeError(
                    f"transformer_quant={engaged[0]} engaged on only "
                    f"{len(engaged)}/{len(views)} experts; retry without quant."
                )
            if engaged:
                transformer_quant_engaged = engaged[0]
        # The quant-sized plan is only valid when quant actually engaged; a dense
        # fallback must keep the conservative bf16 placement.
        if quant_replanned and transformer_quant_engaged is None:
            plan = bf16_plan

        # ── dense text-encoder quant (opt-in): the DiT arrives quantised in a GGUF, but the
        # companion encoder (Gemma3 / UMT5 / Qwen2.5-VL) loads dense bf16 from the base repo and
        # is often the largest resident component. Quantise it in place, mirroring the image
        # backend (diffusion.py): applied for every kind (the encoder is dense regardless of how
        # the DiT was sourced) and before placement so the offload hooks move the smaller weights.
        # Best-effort: quantize_text_encoders leaves any encoder it can't cast dense. int8 needs a
        # per-family keep-bf16 schedule, so the family name is passed.
        text_encoder_quant_engaged = quantize_text_encoders(
            pipe,
            target,
            mode = text_encoder_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            logger = logger,
        )

        # ── optimisation layers, in the image backend's order: step cache FIRST
        # (compile keys its fullgraph decision off an active cache: FBCache hooks
        # graph-break, so compiling fullgraph before installing the cache crashes
        # the first cached generation), then attention, the speed profile, and
        # placement/offload last.
        # A clip denoise runs minutes, so even a dense (non-GGUF) load amortises the
        # one-time regional compile within a single generation: unset resolves to the
        # near-lossless `default` profile for every kind. Explicit values (incl.
        # "off") are honored verbatim, and `max` is never an auto choice.
        effective_speed = resolve_speed_mode(
            speed_mode, is_gguf = kind == "gguf", dense_default = SPEED_DEFAULT
        )
        # A torchao-quantised DiT must be compiled (eager dynamic quant is ~30x slower and
        # would lose to the bf16 it replaced), so force at least the regional-compile
        # profile when quant engaged and the effective speed was off, matching diffusion.py.
        if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
            logger.info(
                "video.transformer_quant: forcing speed_mode=default "
                "(quantized transformer must be compiled; eager is ~30x slower)"
            )
            effective_speed = SPEED_DEFAULT
        backend_flags = snapshot_backend_flags()
        # Until the state commit below transfers ownership to _teardown_state, a
        # failure or cancellation must restore these process-wide globals itself
        # (_run_load's error handler calls _rollback_precommit_globals with this
        # token). Registered BEFORE the first mutating call.
        self._precommit_globals = (_load_token, backend_flags)
        # Step cache tri-state, mirroring the image backend: unset / "auto" lets the
        # step-count policy decide (engage when this model's DEFAULT schedule reaches
        # FBCACHE_MIN_STEPS, re-checked against the actual step count per generation);
        # explicit "off" / "fbcache" are pinned and never toggled. Run it per expert
        # so both denoisers cache; the engaged mode is identical across experts.
        cache_request = normalize_transformer_cache(transformer_cache)
        cache_auto = transformer_cache is None or cache_request == TC_AUTO
        # GGUF checkpoints and torchao-quantised DiTs both need the higher quantised
        # threshold for the cache to still trigger over the quant noise.
        cache_quant_active = kind == "gguf" or transformer_quant_engaged is not None
        default_cache_steps: Optional[int] = None
        if cache_auto:
            default_cache_steps, _ = default_video_generation_params(gguf_filename, repo_id, base)
            cache_request = TC_FBCACHE if default_cache_steps >= FBCACHE_MIN_STEPS else None
        cache_engaged = None
        for view in views:
            engaged = apply_step_cache(
                view,
                mode = cache_request,
                threshold = transformer_cache_threshold,
                # A quantized transformer's block residuals are larger, so it needs the
                # higher FBCache trigger threshold to cache at all. Mirror the image path
                # (diffusion.py): both an engaged transformer_quant AND a GGUF checkpoint
                # (quantized weights) count as quant-active here (cache_quant_active, L1172).
                quant_active = cache_quant_active,
                logger = logger,
            )
            if view is pipe:
                cache_engaged = engaged
        # The auto decision can flip at generation time, but only on a DiT that
        # supports caching at all (a non-CacheMixin transformer can never engage).
        cache_may_toggle = cache_auto and callable(
            getattr(getattr(pipe, "transformer", None), "enable_cache", None)
        )
        if cache_auto:
            if cache_engaged:
                cache_reason = (
                    f"auto: {default_cache_steps}-step default schedule reaches "
                    f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                )
            elif cache_request is not None:
                cache_reason = "auto: model does not support step caching"
            else:
                cache_reason = (
                    f"auto: {default_cache_steps}-step default schedule is below "
                    f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                )
        else:
            cache_reason = "requested"
        attention_engaged = None
        speed_optims: tuple = ()
        for view in views:
            # apply_attention_backend / apply_speed_optims both act on ``view.transformer``;
            # calling them once per view sets the kernel and compiles each expert. The
            # engaged values match across experts (same device/family/mode), so record the
            # first pass; a dense torchao transformer on the pipeline path is not a GGUF one,
            # so is_gguf keys off the load kind (gguf) AND no quant having engaged.
            gguf_transformer = kind == "gguf" and transformer_quant_engaged is None
            engaged = apply_attention_backend(
                view,
                select_attention_backend(
                    target, attention_backend, speed_active = effective_speed != SPEED_OFF
                ),
                logger = logger,
            )
            applied = apply_speed_optims(
                view,
                target,
                is_gguf = gguf_transformer,
                family = fam,
                speed_mode = effective_speed,
                # An auto cache that could still engage mid-session also drops
                # fullgraph: enabling FBCache under a fullgraph-compiled DiT would
                # crash the first cached generation.
                cache_active = cache_engaged is not None or cache_may_toggle,
                offload_active = plan.offload_policy != "none",
            )
            if view is pipe:
                attention_engaged = engaged
                speed_optims = tuple(k for k, v in applied.items() if v)
        with self._generate_lock:
            # A cancelled/superseded load must not place weights on the GPU the arbiter
            # may already have handed to another backend; recheck right before placement
            # (the commit below still does the final locked check).
            if _load_token is not None and _load_token != self._load_token:
                del pipe
                clear_gpu_cache()
                raise RuntimeError("Video load was cancelled or superseded.")
            offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
            # A dual-DiT MoE pipe (Wan2.2-A14B) needs no extra per-expert offload pass here:
            # apply_memory_plan's group tier (_apply_group_offload) already block-streams every
            # DiT it finds on the pipe -- transformer AND transformer_2 -- and model/sequential
            # offload hook every top-level module, so the second expert is covered under all tiers.
            # A second _apply_group_offload on transformer_2 would re-register the group-offload
            # hooks it already carries, which diffusers rejects with a duplicate-hook ValueError.
            if not vae_tiling:
                # Decode of a whole clip is the video memory peak; tiling is near-free
                # in quality and keeps the decode bounded, so it is always on.
                try:
                    pipe.vae.enable_tiling()
                    vae_tiling = True
                except Exception as exc:  # noqa: BLE001 -- tiling is an optimisation only
                    logger.warning("video.vae_tiling_failed: %s", exc)

            resolved = build_resolved_record(
                {
                    "memory_mode": (
                        memory_mode,
                        plan.requested_mode,
                        f"planned '{plan.offload_policy}' offload from the family size table",
                    ),
                    "speed_mode": (
                        speed_mode,
                        effective_speed,
                        "quantized transformer requires compile"
                        if transformer_quant_engaged is not None
                        else "clip denoises amortise the one-time compile within a single run"
                        if speed_mode is None
                        else "requested",
                    ),
                    "attention_backend": (
                        attention_backend,
                        attention_engaged or "native",
                        "cuDNN fused attention on NVIDIA when a speed profile is active",
                    ),
                    "transformer_cache": (
                        None if cache_auto else transformer_cache,
                        cache_engaged or "off",
                        cache_reason,
                    ),
                    "transformer_quant": (
                        transformer_quant,
                        transformer_quant_engaged or "off",
                        "dense DiT(s) torchao-quantised onto the low-precision tensor cores"
                        if transformer_quant_engaged is not None
                        else (
                            "skipped: offload moves the DiT, unsupported for torchao "
                            "tensors; pin a resident memory mode to combine them"
                            if quant_skipped_for_offload
                            else "not engaged (dense bf16 DiT loaded)"
                        ),
                    ),
                    "text_encoder_quant": (
                        text_encoder_quant,
                        text_encoder_quant_engaged or "off",
                        "dense text encoder quantised in place"
                        if text_encoder_quant_engaged is not None
                        else "not engaged (dense bf16 text encoder loaded)",
                    ),
                }
            )

            with self._lock:
                if _load_token is not None and _load_token != self._load_token:
                    del pipe
                    clear_gpu_cache()
                    raise RuntimeError("Video load was cancelled or superseded.")
                self._state = _VideoLoadState(
                    pipe = pipe,
                    family = fam,
                    repo_id = repo_id,
                    base_repo = base,
                    device = device,
                    dtype = str(dtype).replace("torch.", ""),
                    kind = kind,
                    gguf_filename = gguf_filename,
                    offload_policy = offload_policy,
                    vae_tiling = vae_tiling,
                    memory_mode = plan.requested_mode,
                    speed_mode = effective_speed,
                    # Already filtered above to only the optimisations that engaged;
                    # apply_speed_optims returns every flag True/False and the view
                    # loop keeps just the True names.
                    speed_optims = speed_optims,
                    backend_flags = backend_flags,
                    attention_backend = attention_engaged,
                    transformer_cache = cache_engaged,
                    cache_auto = cache_may_toggle,
                    cache_quant_active = cache_quant_active,
                    cache_threshold = transformer_cache_threshold,
                    transformer_quant = transformer_quant_engaged,
                    text_encoder_quant = text_encoder_quant_engaged,
                    resolved = resolved,
                )
                # Ownership of the globals transferred to _state / _teardown_state.
                self._precommit_globals = None
        logger.info(
            "video.loaded: %s (%s, %s, offload=%s, speed=%s, quant=%s)",
            repo_id,
            fam.name,
            kind,
            offload_policy,
            effective_speed,
            transformer_quant_engaged or "off",
        )
        return self.status()

    @staticmethod
    def _resolve_checkpoint_path(
        repo_id: str, gguf_filename: Optional[str], hf_token: Optional[str]
    ) -> Path:
        """The local checkpoint file for a gguf/single_file load (downloads if hub)."""
        from .diffusion_families import resolve_local_gguf_child

        root = Path(repo_id).expanduser()
        if root.is_dir():
            return resolve_local_gguf_child(root, gguf_filename or "")
        if root.is_file():
            return root
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        return Path(hf_hub_download_with_xet_fallback(repo_id, gguf_filename or "", hf_token))

    # ── generation ───────────────────────────────────────────────────────────

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clear FBCache residuals on the resident DiT(s) before a generation.

        diffusers keys the residuals on the long-lived transformer and no pipeline
        resets them, so the next clip would compare against the previous request's
        state: a shape mismatch when the resolution changed, stale reuse otherwise.
        ``_reset_stateful_cache`` is the transformer-level entry point in diffusers
        0.39 (``reset_stateful_hooks`` lives only on the HookRegistry). Best-effort:
        an uncached transformer is a silent no-op."""
        for name in ("transformer", "transformer_2"):
            module = getattr(pipe, name, None)
            reset = getattr(module, "_reset_stateful_cache", None) or getattr(
                module, "reset_stateful_hooks", None
            )
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001 -- reset is best-effort, never fail a generation
                    pass

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        guidance_2: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        import torch
        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(VIDEO_NOT_LOADED_MSG)
                self._active_generate_cancel = cancel
            try:
                fam = state.family
                width, height = snap_video_size(
                    fam,
                    width or fam.resolution_presets[0][0],
                    height or fam.resolution_presets[0][1],
                )
                frames = snap_num_frames(fam, num_frames or fam.default_num_frames)
                out_fps = int(fps or fam.default_fps)
                default_steps, default_guidance = default_video_generation_params(
                    state.gguf_filename,
                    state.repo_id,
                    state.base_repo,
                    fallback = (fam.default_steps, fam.default_guidance),
                )
                steps = int(steps or default_steps)
                guidance = float(default_guidance if guidance is None else guidance)

                generator = torch.Generator(device = state.device)
                if seed is None:
                    seed = int(generator.seed()) % (2**53)
                generator = generator.manual_seed(int(seed))

                pipe = state.pipe
                call_params = inspect.signature(pipe.__call__).parameters
                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                    "generator": generator,
                }
                if fam.guidance_via_guider:
                    # HunyuanVideo-1.5: __call__ has no guidance kwarg at all; the
                    # CFG scale is a plain attribute on the pipeline's guider
                    # component, set per request. Near-1 scales auto-disable CFG
                    # inside the guider itself (_is_cfg_enabled's is_close check).
                    pipe.guider.guidance_scale = float(guidance)
                else:
                    kwargs[fam.cfg_kwarg] = guidance
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                # LTX-2 takes frame_rate (it shapes the audio track length); other
                # pipelines fix their own rate and fps only matters at export.
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(out_fps)
                # Dual-DiT MoE (Wan2.2-A14B): the low-noise expert (transformer_2) has its
                # own guidance kwarg (cfg2_kwarg = "guidance_scale_2"). Thread it only when
                # the loaded family declares one AND the pipeline signature accepts it (the
                # same inspect.signature gate frame_rate uses), so a single-DiT pipeline is
                # never handed a kwarg its check_inputs would reject. WanPipeline raises if
                # guidance_scale_2 is passed to a pipeline with boundary_ratio=None
                # (pipeline_wan.py:322), so the gate is BOTH the family flag and the
                # signature: TI2V-5B has no cfg2_kwarg, so it never reaches here. A None
                # request lets the pipeline default it (to guidance_scale) itself.
                if fam.cfg2_kwarg and fam.cfg2_kwarg in call_params and guidance_2 is not None:
                    kwargs[fam.cfg2_kwarg] = float(guidance_2)

                started = time.monotonic()
                self._gen = {
                    "active": True,
                    "phase": "denoise",
                    "step": 0,
                    "total": steps,
                    "started": started,
                    "eta_seconds": None,
                    "error": None,
                }

                def _tick(done: int) -> None:
                    elapsed = time.monotonic() - started
                    self._gen.update(
                        step = done,
                        eta_seconds = (elapsed / max(1, done)) * max(0, steps - done),
                    )

                def _on_step(p, step_index, timestep, callback_kwargs):
                    if cancel.is_set():
                        p._interrupt = True
                        return callback_kwargs
                    _tick(step_index + 1)
                    return callback_kwargs

                def _on_scheduler_step(done: int) -> None:
                    # No cooperative _interrupt here: without a callback the pipeline
                    # never checks it, so cancellation must unwind the denoise loop
                    # via an exception (mapped to the cancelled sentinel below).
                    if cancel.is_set():
                        raise _VideoGenerationCancelled()
                    _tick(done)

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    progress_ctx = contextlib.nullcontext()
                else:
                    # HunyuanVideo-1.5 has no step callback; every scheduler.step
                    # call is exactly one denoise step, so wrap it for progress +
                    # cancel and restore it afterwards.
                    progress_ctx = _scheduler_step_progress(pipe, _on_scheduler_step)

                # An AUTO cache decision is re-checked against the ACTUAL step count,
                # mirroring the image backend: a many-step request gains FBCache even
                # when the load's default schedule kept it off, and a few-step request
                # drops it. Explicit choices never toggle. Runs per view so a dual-DiT
                # MoE toggles both experts.
                if state.cache_auto:
                    toggled = state.transformer_cache
                    for view in _views_for(pipe, fam):
                        toggled = maybe_toggle_step_cache(
                            view,
                            steps = steps,
                            quant_active = state.cache_quant_active,
                            threshold = state.cache_threshold,
                            logger = logger,
                        )
                    if toggled != state.transformer_cache:
                        # _VideoLoadState is frozen (loads swap it as one unit); this
                        # tracks the pipe-level toggle that already happened so
                        # status() reports the true cache state.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                try:
                    with torch.inference_mode(), progress_ctx:
                        output = pipe(**kwargs)
                except _VideoGenerationCancelled:
                    # This cancel unwinds pipe.__call__ by exception (the scheduler
                    # wrapper has no cooperative _interrupt), skipping the pipeline's
                    # end-of-call maybe_free_model_hooks(); under model/group offload
                    # the currently-onloaded modules would otherwise stay on the GPU
                    # until the next request touches them.
                    free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
                    if callable(free_hooks):
                        try:
                            free_hooks()
                        except Exception:  # noqa: BLE001 -- cleanup is best-effort
                            pass
                    raise RuntimeError(VIDEO_CANCELLED_MSG) from None
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)

                self._gen.update(phase = "export", eta_seconds = None)
                video_frames = output.frames[0]
                audio = getattr(output, "audio", None)
                audio_track = audio[0] if fam.has_audio and audio is not None else None
                mp4_bytes = self._encode_mp4(
                    video_frames, out_fps, audio_track, pipe if fam.has_audio else None
                )
                duration_s = len(video_frames) / float(out_fps) if out_fps else 0.0
                self._gen = {"active": False}
                return {
                    "mp4_bytes": mp4_bytes,
                    "seed": int(seed),
                    "repo_id": state.repo_id,
                    "width": width,
                    "height": height,
                    "num_frames": len(video_frames),
                    "fps": out_fps,
                    "duration_s": duration_s,
                    "has_audio": bool(audio_track is not None),
                    "steps": steps,
                    "guidance": guidance,
                }
            except Exception:
                self._gen = {"active": False}
                raise
            finally:
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    @staticmethod
    def _encode_mp4(video_frames, fps: int, audio, pipe) -> bytes:
        """Encode frames (+ optional audio) to H.264 MP4 bytes via diffusers' PyAV
        exporter. A temp file bridges the exporter's path-based API; the bytes are
        what the gallery persists."""
        from diffusers.utils.export_utils import encode_video

        tmp = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
        tmp.close()
        try:
            encode_kwargs: dict[str, Any] = {}
            if audio is not None and pipe is not None:
                encode_kwargs["audio"] = audio
                sample_rate = getattr(
                    getattr(getattr(pipe, "vocoder", None), "config", None),
                    "output_sampling_rate",
                    None,
                )
                if sample_rate:
                    encode_kwargs["audio_sample_rate"] = int(sample_rate)
            encode_video(video_frames, fps, tmp.name, **encode_kwargs)
            return Path(tmp.name).read_bytes()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def generate_progress(self) -> dict[str, Any]:
        gen = dict(self._gen)
        gen.setdefault("active", False)
        return gen

    def cancel_generate(self) -> bool:
        """Signal the in-flight generation to stop at its next step callback."""
        with self._lock:
            cancel = self._active_generate_cancel
            if cancel is None:
                return False
            cancel.set()
            return True

    # ── teardown + status ────────────────────────────────────────────────────

    def _teardown_state(self) -> None:
        state = None
        with self._lock:
            state, self._state = self._state, None
        if state is not None:
            restore_backend_flags(state.backend_flags)
            # A GGUF video load may have installed the process-wide compiled GGUF
            # dequantizer; restore the stock kernels so a later load that asked for
            # speed_mode=off gets the bit-identical path (mirrors the image unload).
            from . import diffusion_gguf_compile

            diffusion_gguf_compile.uninstall_all()
            del state
            clear_gpu_cache()

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._load_token += 1
            self._cancel_event.set()
            self._loading = None
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        # Wait for the signalled generation to actually exit before freeing the
        # pipeline: the denoise loop holds its own pipe reference until the next
        # step callback, so tearing down under it would report the VRAM free (and
        # let the GPU arbiter start another multi-GB load) while this clip still
        # occupies it. generate() holds _generate_lock for its full body, so a
        # bare acquire is the exit barrier (never taken while holding _lock).
        with self._generate_lock:
            pass
        self._teardown_state()
        logger.info("video.unloaded")
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "model_kind": None,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "attention_backend": None,
                "transformer_cache": None,
                "transformer_quant": None,
                "text_encoder_quant": None,
                "has_audio": False,
                "defaults": None,
                "resolved": None,
            }
        fam = state.family
        default_steps, default_guidance = default_video_generation_params(
            state.gguf_filename,
            state.repo_id,
            state.base_repo,
            fallback = (fam.default_steps, fam.default_guidance),
        )
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": fam.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "model_kind": state.kind,
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": list(state.speed_optims),
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "transformer_quant": state.transformer_quant,
            "text_encoder_quant": state.text_encoder_quant,
            "has_audio": fam.has_audio,
            "defaults": {
                "steps": default_steps,
                "guidance": default_guidance,
                "num_frames": fam.default_num_frames,
                "fps": fam.default_fps,
                "frame_step": fam.frame_step,
                "resolution_multiple": fam.resolution_multiple,
                "resolution_presets": [list(p) for p in fam.resolution_presets],
            },
            "resolved": state.resolved,
        }


_backend: Optional[VideoBackend] = None
_backend_lock = threading.Lock()


def get_video_backend() -> VideoBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = VideoBackend()
        return _backend
