# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local diffusion (text-to-image) backend.

A torch-only singleton that loads one of three "kinds" (see ``resolve_model_kind``):
a single-file GGUF transformer dequantised on-device via ``GGUFQuantizationConfig``,
a single-file safetensors transformer (e.g. fp8), or a full diffusers pipeline via
``from_pretrained`` (which re-applies an embedded quant config such as bnb-4bit). The
single-file kinds pull the rest of the pipeline (VAE, text encoders, scheduler) from
the matching base repo; the pipeline kind pulls everything from the repo itself.
Non-GGUF kinds are gated to the ``unsloth/*`` org (or a local path) for safety.

torch/diffusers are imported lazily so this stays importable in a no-torch runtime.
``begin_load`` runs on a background thread; poll ``load_progress`` for the download
bar. GPU-handoff policy lives in the arbiter the routes call, not here.
"""

from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger
from utils.hardware import clear_gpu_cache

from .diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    default_generation_params,
    detect_family_for_pick,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
)
from .diffusion_device import (
    DiffusionDeviceTarget,
    diffusion_device_target_from_torch_device,
    resolve_diffusion_device_target,
)
from .diffusion_krea2 import KREA2_FAMILY_NAME, load_krea2_pipeline
from .diffusion_memory import (
    OFFLOAD_NONE,
    apply_memory_plan,
    estimate_gguf_resident_mib,
    estimate_image_runtime_mib,
    estimate_safetensors_dense_mib,
    file_size_mib,
    plan_diffusion_memory,
    snapshot_device_memory,
)
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    compile_eligible,
    normalize_speed_mode,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_attention import (
    apply_attention_backend,
    select_attention_backend,
)
from . import diffusion_compile_cache as compile_cache
from . import diffusion_gguf_compile as gguf_compile
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_FBCACHE,
    apply_step_cache,
    maybe_toggle_step_cache,
    normalize_transformer_cache,
)
from .diffusion_precision import quantize_text_encoders
from .diffusion_prequant import (
    load_prequantized_transformer,
    resolve_prequant_source,
)
from .diffusion_auto_policy import build_resolved_record, resolve_dense_quant_candidate
from .diffusion_transformer_quant import (
    DEFAULT_MIN_LINEAR_FEATURES,
    dense_transformer_supported,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)

logger = get_logger(__name__)


# A load resolves to exactly one of these "kinds", which decide how the transformer
# (and the rest of the pipeline) is built:
#   "gguf"        -- a single-file GGUF transformer dequantised on-device via
#                    GGUFQuantizationConfig; the VAE / text encoders / scheduler come
#                    from the companion base diffusers repo. The original behaviour.
#   "single_file" -- a single-file *.safetensors transformer loaded with from_single_file
#                    WITHOUT the GGUF dequant config (e.g. an fp8 checkpoint); companions
#                    still come from the base repo.
#   "pipeline"    -- a full diffusers repo loaded with pipeline_cls.from_pretrained(repo_id),
#                    which pulls every component (transformer included) and re-applies any
#                    embedded quantization_config (e.g. a bnb-4bit pipeline) automatically.
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})


def resolve_model_kind(gguf_filename: Optional[str], model_kind: Optional[str] = None) -> str:
    """Classify a load request into one of ``_MODEL_KINDS``.

    An explicit ``model_kind`` wins (validated). Otherwise the kind is inferred from
    the single-file name: a ``.gguf`` name is ``"gguf"``, any other single-file name is
    ``"single_file"``, and the absence of a name is a full ``"pipeline"`` load. Pure and
    network-free, so the route, validation, and load paths all agree on the kind."""
    if model_kind:
        kind = model_kind.strip().lower()
        if kind not in _MODEL_KINDS:
            raise ValueError(
                f"Unknown model_kind '{model_kind}'. Expected one of {sorted(_MODEL_KINDS)}."
            )
        return kind
    name = (gguf_filename or "").strip()
    if not name:
        return "pipeline"
    if name.lower().endswith(".gguf"):
        return "gguf"
    return "single_file"


def _decode_b64_image(data: str, *, mode: str = "RGB") -> Any:
    """Decode a base64 (optionally ``data:`` URL) image string to a PIL image.

    The image-conditioned workflows (img2img / inpaint / edit) transport the input
    image and mask as base64 in the JSON request, so this is the single decode path.
    A mask is decoded as single-channel ``L``; the source image as ``RGB``."""
    import base64
    import binascii
    import io

    from PIL import Image

    raw = data.strip()
    if raw.startswith("data:"):
        # data:[<mime>][;base64],<payload>
        _, _, raw = raw.partition(",")
    try:
        blob = base64.b64decode(raw, validate = False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    # Bound the decoded size. Every image-conditioned workflow (img2img / inpaint / upscale /
    # reference / edit) decodes through here, so this single guard protects init, mask, and
    # each reference image uniformly. PIL only WARNS in its 89-178MP "decompression bomb" soft
    # zone and still loads (~0.5 GB RGB each, times up to 4 with multi-reference); cap the side
    # well below that. 4096px covers txt2img's 2048 max, upscales, and normal outpaint canvases;
    # anything larger is rejected with a clear 400 instead of risking an OOM.
    max_side = 4096
    try:
        img = Image.open(io.BytesIO(blob))
        # Read the declared dimensions from the header (Image.open is lazy) and reject an
        # over-limit image BEFORE img.load() decompresses its pixels, so a crafted
        # small-payload/huge-dimension file can't spike memory before the guard runs.
        w, h = img.size
        if w > max_side or h > max_side:
            raise ValueError(f"Image is too large ({w}x{h}); maximum is {max_side}px per side.")
        img.load()
    except ValueError:
        raise  # the size guard's own message; don't wrap it as a decode error
    except Exception as exc:  # noqa: BLE001 — surfaced as a 400 to the client
        raise ValueError(f"Could not decode image: {exc}") from exc
    return img.convert(mode)


def _snap_to_multiple(img: Any, multiple: int = 16) -> Any:
    """Resize a PIL image so both sides are multiples of ``multiple`` (rounded to nearest,
    minimum one multiple), preserving content with a high-quality resample.

    Image-conditioned pipelines (Z-Image / Qwen / FLUX: 8x VAE downsample + 2x patch) reject
    sizes that are not divisible by 16. Rather than error on an odd-sized upload, snap it so
    the workflow just works; rounding to nearest keeps the rescale minimal/accurate."""
    from PIL import Image

    w, h = img.size
    nw = max(multiple, int(round(w / multiple)) * multiple)
    nh = max(multiple, int(round(h / multiple)) * multiple)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


# A small allowlist of well-known official base repos that may load as a full
# (non-GGUF) pipeline even though they are not under ``unsloth/``. These are
# safetensors-only checkpoints from their original publisher (no pickle, no remote
# code) that some architectures require: SDXL ships only as a full pipeline and has
# no unsloth-hosted GGUF, so without this its curated catalog entry could not load.
# Exact-match, lowercased, so it cannot be widened by a typo-squat. Extend
# deliberately, and never add a repo that carries pickled weights or remote code.
# The SDXL refiner is intentionally NOT here: it is an img2img-only refiner pipeline
# (StableDiffusionXLImg2ImgPipeline), but this backend loads every ``sdxl`` repo as the
# base txt2img StableDiffusionXLPipeline and advertises txt2img, so allowlisting the
# refiner would surface the wrong workflow and call it without its required input image.
_TRUSTED_NON_GGUF_REPOS = frozenset(
    {
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        # Official vendor, safetensors-only base repos allowlisted as LoRA TRAINING bases
        # (diffusion training loads the full pipeline from these). Same rule as above: no
        # pickled weights, no remote code, exact-match lowercased. FLUX.1-dev is gated on
        # the Hub (needs the user's token); the other two are open.
        "black-forest-labs/flux.1-dev",
        "tongyi-mai/z-image-turbo",
        "qwen/qwen-image",
        # Krea 2: official vendor repos, safetensors-only, no remote code. Loaded
        # per-component via core/inference/diffusion_krea2.py (no GGUF variant yet).
        # Turbo is the inference model; Raw is the undistilled base Krea recommends
        # training LoRAs on (train on Raw, run adapters on Turbo).
        "krea/krea-2-turbo",
        "krea/krea-2-raw",
    }
)


def _is_trusted_diffusion_repo(repo_id: str) -> bool:
    """Whether a NON-GGUF load is allowed for ``repo_id``.

    Making ``gguf_filename`` optional opens a ``from_pretrained`` / ``from_single_file``
    on an arbitrary repo, which fetches and deserialises third-party weights. So the
    non-GGUF paths are gated to the ``unsloth/*`` org (the curated safetensors models),
    a short allowlist of official safetensors-only base repos (``_TRUSTED_NON_GGUF_REPOS``,
    e.g. the SDXL base), and local paths the user explicitly pointed at (already on their
    disk). The GGUF path is unchanged and stays open to any repo, as before.

    A bare ``owner/name`` HF id is never a real filesystem path, and an id with invalid
    characters makes ``Path.exists()`` raise OSError; treat any such failure as "not a
    local path" so the trust decision falls through to the org/allowlist checks (the
    loader's validate_load_request raises the clear FileNotFoundError for a genuinely
    missing local pick)."""
    try:
        if Path(repo_id).expanduser().exists():
            return True
    except OSError:
        pass
    rid = repo_id.strip().lower()
    return rid.startswith("unsloth/") or rid in _TRUSTED_NON_GGUF_REPOS


@dataclass(frozen = True)
class _LoadState:
    """Everything about the currently-loaded pipeline, swapped as one unit."""

    pipe: Any
    family: Any
    repo_id: str
    base_repo: str
    device: str
    dtype: str
    cpu_offload: bool
    # The resolved memory profile (Phase 2A). Appended with defaults so older
    # positional constructions (and the back-compat status shape) keep working.
    offload_policy: str = OFFLOAD_NONE
    vae_tiling: bool = False
    memory_mode: str = "auto"
    # The resolved load kind: "gguf" | "single_file" | "pipeline". Surfaced in status so the
    # UI can gate GGUF-only controls (the dense transformer_quant fast path only engages on
    # the gguf kind; on single_file/pipeline it is a silent no-op).
    kind: str = "gguf"
    # The opt-in speed profile (Phase 3).
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    # Process-wide torch backend flags (TF32 / cudnn.benchmark) captured before the
    # speed layer mutated them, restored on unload so a later `off` load is not
    # contaminated by this one's globals. None when nothing was changed.
    backend_flags_before: Optional[dict] = None
    # Text-encoder quantisation actually engaged: "fp8" | "nvfp4" | None (Phase 2B/2C).
    text_encoder_quant: Optional[str] = None
    # Transformer quant actually engaged on the opt-in dense fast path: "int8" | "fp8"
    # | "nvfp4" | "mxfp8" | None. None means the default GGUF transformer was loaded.
    transformer_quant: Optional[str] = None
    # Attention backend engaged via the diffusers dispatcher (e.g. "_native_cudnn"), or
    # None for the default SDPA. Set before compile; orthogonal to the weight quant.
    attention_backend: Optional[str] = None
    # Step cache engaged ("fbcache") or None. Opt-in, for many-step models.
    transformer_cache: Optional[str] = None
    # True when the cache decision was AUTO on a cache-capable transformer: generate()
    # then re-checks the actual step count and toggles FBCache across FBCACHE_MIN_STEPS.
    # An explicit request (off / fbcache) is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Shared eager monkey-patches (diffusion_eager_patches) installed for this load (any
    # non-off speed tier). Uninstalled on unload so a later `off` load is bit-identical.
    eager_patched: bool = False
    # Pre-warmed torch.compile cache context (diffusion_compile_cache.CacheContext) when a
    # compiled tier ran, else None. Carries the per-key inductor dir + bundle for save/restore.
    compile_cache_ctx: Any = None
    # Token kept so LoRA adapters selected at generate time can be fetched from the Hub.
    hf_token: Optional[str] = None
    # Per-control provenance from the auto-policy: {control: {value, source, reason}}.
    # source is "auto" when the backend decided (request left unset / "auto") and
    # "explicit" when the caller pinned the value. Surfaced via status for the UI badges.
    resolved: Optional[dict] = None


@dataclass
class _LoadingState:
    """An in-flight background load, polled for download progress."""

    repo_id: str
    base_repo: str
    expected_bytes: int = 0
    error: Optional[str] = None


@dataclass
class _GenState:
    """An in-flight generation, updated per denoising step for the progress bar."""

    total_steps: int
    step: int = 0
    # Set when the first step finishes; the ETA rate is measured from there so the
    # slower first step (warmup) doesn't skew it.
    first_step_at: float = 0.0
    # Computed once per step (in the callback) so it's stable between polls.
    eta_seconds: Optional[float] = None


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    """Seconds remaining, from the average step time measured after the first step.
    None until at least one step has elapsed since the first."""
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def _resolve_diffusion_compute_dtype(fam: Optional[DiffusionFamily], dtype: Any) -> Any:
    """Promote float16 -> float32 for fp16-incompatible families (e.g. Z-Image),
    whose activations overflow float16's finite range and render a black image.
    Every other dtype/family passes through unchanged."""
    if fam is None or not getattr(fam, "fp16_incompatible", False):
        return dtype
    import torch

    return torch.float32 if dtype == torch.float16 else dtype


class DiffusionBackend:
    """Holds at most one loaded diffusers pipeline. All mutations are serialised."""

    def __init__(self) -> None:
        # _lock serialises the small state mutations (the load swap, _loading,
        # _load_token, _gen). status() / load_progress() / generate_progress()
        # read those references WITHOUT it, so polling never blocks a slow load.
        self._lock = threading.Lock()
        # _generate_lock serialises generations and is the ONLY lock the denoise
        # holds, so a long generation never blocks status()/unload()/a new load.
        self._generate_lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None
        # Bumped on every begin_load and unload so a worker whose load was
        # superseded (a new load) or cancelled (unload, incl. an arbiter eviction)
        # neither commits its pipeline nor stamps progress onto the current load.
        self._load_token = 0
        # Set by unload() to abort an in-flight download (which runs without the
        # lock, like the chat backend), so an eviction/unload can preempt a slow
        # load instead of blocking on the lock for the whole download.
        self._cancel_event = threading.Event()
        # The cancel Event of the generation currently in flight (or None). Set
        # under _lock by unload() / a superseding load to abort that specific
        # denoise (its step callback flips pipe._interrupt). Per-generation rather
        # than one shared flag the next generate would clear, so a cancel can't be
        # lost to a racing generate nor leak onto the wrong one.
        self._active_generate_cancel: Optional[threading.Event] = None
        # The callback mutates _gen and generate_progress() reads it, both lock-free,
        # so per-step progress polling stays live during a generation.
        self._gen: Optional[_GenState] = None
        # Cache of image-conditioned workflow pipelines (img2img / inpaint) built via
        # Pipeline.from_pipe around the loaded text-to-image pipe. They share its already
        # resident modules (no extra VRAM, no reload), so we build each once per load and
        # reuse it. Keyed by pipeline class name; cleared on unload with the base pipe.
        self._aux_pipes: dict[str, Any] = {}
        # Cache of loaded ControlNet models (id -> module) and the ControlNet workflow
        # pipelines built around them ((pipeline_class, cn_id) -> pipe). ControlNet models
        # are a small extra module loaded via from_pretrained; the pipeline is assembled via
        # Pipeline.from_pipe(base, controlnet=model), reusing the resident base modules (no
        # reload). Both are cleared on unload with the base pipe.
        self._cn_models: dict[str, Any] = {}
        self._cn_pipes: dict[tuple[str, str], Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def _pick_device_and_dtype(self) -> tuple[str, Any]:
        """(device, dtype) for the current host. Thin wrapper over the device
        policy module, kept as a method so tests can still monkeypatch it."""
        target = resolve_diffusion_device_target()
        return target.device, target.dtype

    def _resolve_device_target(self, fam: Optional[DiffusionFamily]) -> DiffusionDeviceTarget:
        """The device target with the family fp16 guard applied.

        Routes through _pick_device_and_dtype() (so a monkeypatched override still
        drives the result), then promotes float16 -> float32 for fp16-incompatible
        families (Z-Image), rebuilding the target so dtype + capability flags stay
        consistent with the effective dtype.
        """
        device, dtype = self._pick_device_and_dtype()
        effective = _resolve_diffusion_compute_dtype(fam, dtype)
        if effective is not dtype:
            logger.warning(
                "diffusion.dtype_promoted: family=%s float16 -> float32 (fp16-incompatible)",
                getattr(fam, "name", None),
            )
        return diffusion_device_target_from_torch_device(device, effective)

    def _resolve_gguf_path(self, repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> str:
        local_root = Path(repo_id).expanduser()
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id, gguf_filename, token = hf_token)

    def _dense_quant_prefetch_needed(self, fam: DiffusionFamily, kwargs: dict) -> bool:
        """True when ``load_pipeline`` may take the dense transformer-quant path, so
        the prefetch should also pull the base repo's ``transformer/`` shards.

        Those shards are excluded from the prefetch by default (the GGUF supplies
        the transformer), but ``_load_dense_quant_pipeline`` fetches them with
        ``from_pretrained(subfolder = "transformer")`` under the load lock during
        "finalizing", after the previous pipeline was already evicted, where
        unload/cancellation cannot preempt the download. Mirrors the dense-path
        gates in ``load_pipeline``: quant requested and supported for this device,
        and no pre-quantized checkpoint that would shortcut the dense build."""
        mode = normalize_transformer_quant(kwargs.get("transformer_quant"))
        if mode is None:
            return False
        try:
            target = self._resolve_device_target(fam)
            if not dense_transformer_supported(target):
                return False
            scheme = select_transformer_quant_scheme(target, mode)
            if scheme is None:
                return False
            source = resolve_prequant_source(
                fam, scheme, path_override = kwargs.get("transformer_prequant_path")
            )
            return source is None
        except Exception:  # noqa: BLE001 — widening the prefetch is best-effort only
            return False

    def _prefetch_files(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        base_files: list[str],
        hf_token: Optional[str],
    ) -> None:
        """Pre-download the GGUF + the given ``base_files`` into the HF cache,
        WITHOUT the lock and honoring ``_cancel_event``, so load_pipeline's
        from_single_file / from_pretrained hit the cache and the heavy download can
        be preempted by an unload/eviction. Raises ``RuntimeError("Cancelled")``."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # GGUF transformer (hub repos only; a local path is already on disk).
        if gguf_filename and not Path(repo_id).expanduser().exists():
            hf_hub_download_with_xet_fallback(
                repo_id, gguf_filename, hf_token, cancel_event = self._cancel_event
            )
        # Base repo (VAE / text-encoder / scheduler); list comes from the estimate.
        for rfilename in base_files:
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            hf_hub_download_with_xet_fallback(
                base, rfilename, hf_token, cancel_event = self._cancel_event
            )

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
    ) -> DiffusionFamily:
        """Cheap, network-free validation shared by the route (before it evicts the
        chat model) and the load paths, so an unloadable pick fails BEFORE the GPU
        handoff. Resolves the load kind (gguf / single_file / pipeline), then raises
        ValueError for a missing single-file name, a non-unsloth non-GGUF repo, or an
        undetectable family, and ValueError/FileNotFoundError for a bad local path.
        Touches no GPU, network, or state."""
        kind = resolve_model_kind(gguf_filename, model_kind)
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name. (Video models and image models "
                f"whose diffusers transformer has no single-file loader are not supported.)"
            )
        # A GGUF load builds a transformer-only file via the generic GGUF branch
        # (UNet2DConditionModel.from_single_file(subfolder="transformer", GGUFQuantizationConfig)).
        # Families whose single file IS the whole pipeline (SDXL) have no transformer-only
        # GGUF path, so reject GGUF here -- before the route evicts the current model and
        # the background load fails deep in from_single_file.
        if kind == "gguf" and fam.single_file_is_pipeline:
            raise ValueError(
                f"'{fam.name}' checkpoints are whole-pipeline single files and have no GGUF "
                f"transformer variant; load the .safetensors pipeline instead of a GGUF."
            )
        # Non-GGUF loads (a single-file safetensors transformer, or a full pipeline)
        # are gated to the unsloth org or a local path -- they fetch + deserialise
        # weights, so an arbitrary remote repo is rejected here, before any work.
        if kind != "gguf" and not _is_trusted_diffusion_repo(repo_id):
            raise ValueError(
                f"Non-GGUF diffusion loads are restricted to unsloth/* repos (or a local "
                f"path); got '{repo_id}'. Pass a gguf_filename to load a GGUF instead."
            )
        # Reject a bad LOCAL pick now (the same checks the load would hit later), so
        # the route never evicts a working chat model for a request that can't load.
        # A path-shaped repo_id (absolute / ~ / ./ / ..) is meant to be on disk, so a
        # missing one is an error here; a bare "org/name" id is a remote HF repo and
        # is left for the background load to resolve.
        local_root = Path(repo_id).expanduser()
        # POSIX path-shaped, a "."/".." prefix (covers ./ ../ and their Windows .\ ..\
        # forms), a Windows separator anywhere (never present in a bare "org/name" HF
        # id), or an absolute path on this OS.
        path_shaped = (
            repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or local_root.is_absolute()
        )
        if kind in ("gguf", "single_file"):
            if not gguf_filename:
                raise ValueError(f"a single-file checkpoint name is required for a '{kind}' load.")
            # Fail a kind/extension mismatch here (before the route evicts chat and grabs the
            # GPU), instead of deep in the background from_single_file: a "gguf" load needs a
            # .gguf file, and a "single_file" load must not be handed a .gguf.
            is_gguf_name = gguf_filename.lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            # A single-file load must name an actual checkpoint: an arbitrary repo file
            # (README.md, config.json) would pass preflight, evict the chat model, and
            # only fail in the background from_single_file -- the eviction this
            # validation exists to prevent.
            if kind == "single_file" and not gguf_filename.lower().endswith(".safetensors"):
                raise ValueError(
                    f"'{gguf_filename}' is not a loadable single-file checkpoint "
                    f"(expected a .safetensors name; use a .gguf name for a GGUF load)."
                )
            if local_root.exists():
                resolve_local_gguf_child(local_root, gguf_filename)
            elif path_shaped:
                raise FileNotFoundError(f"Local model path does not exist: {repo_id}")
        else:  # pipeline
            if gguf_filename:
                raise ValueError(
                    "a 'pipeline' load takes a full diffusers repo, not a single-file name."
                )
            if local_root.exists():
                if not (local_root / "model_index.json").exists():
                    raise FileNotFoundError(
                        f"Local pipeline directory has no model_index.json: {repo_id}"
                    )
            elif path_shaped:
                raise FileNotFoundError(f"Local model path does not exist: {repo_id}")
            elif repo_id.upper().endswith("-GGUF"):
                # A remote "*-GGUF" id is a single-file GGUF repo, not a full diffusers
                # pipeline: loading it as a pipeline passes the trusted-repo check, evicts
                # chat, then fails in the background when from_pretrained finds no
                # model_index.json. Reject the certain case here (no network round-trip)
                # so the bad pick fails before the GPU handoff, as the route expects.
                raise ValueError(
                    f"'{repo_id}' is a single-file GGUF repo; load it with model_kind 'gguf' "
                    f"and a .gguf filename, not as a full pipeline."
                )
        return fam

    # ── Background load + progress ─────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        # A blank token (the Studio default when none is configured) must mean
        # "anonymous", not an explicit empty credential the Hub rejects with 401.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
        )

        with self._lock:
            # Allow starting over a previously-failed load, but not over a live one.
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            self._load_token += 1
            token = self._load_token
            # Best-effort download preemption only; the token (not this event) is
            # the real guard that a superseded worker can't commit its pipeline.
            self._cancel_event.clear()
            # Seed with the family fallback; the worker resolves the real base
            # (a network lookup) and updates this, so begin_load never blocks.
            self._loading = _LoadingState(repo_id = repo_id, base_repo = fam.base_repo)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base_repo = base_repo,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                text_encoder_quant = text_encoder_quant,
                transformer_quant = transformer_quant,
                transformer_quant_fast_accum = transformer_quant_fast_accum,
                transformer_prequant_path = transformer_prequant_path,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                model_kind = model_kind,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            # Resolve the base repo and estimate sizes on this thread (both network
            # calls) so begin_load returns instantly; the bar shows raw bytes until
            # the total lands. This is the only writer of _loading's fields here.
            fam = detect_family_for_pick(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            if kind == "pipeline":
                # The full pipeline IS the repo: from_pretrained pulls every component
                # (transformer included) from it, so the base repo is the repo itself.
                base = kwargs["repo_id"]
            else:
                base = _resolve_base_repo(
                    kwargs["repo_id"], kwargs.get("base_repo"), fam, kwargs.get("hf_token")
                )
            kwargs["base_repo"] = base
            expected, base_files = self._estimate_download_bytes(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                kwargs.get("hf_token"),
                kind = kind,
                single_file_is_pipeline = bool(fam and fam.single_file_is_pipeline),
                # The dense transformer-quant path downloads the base repo's
                # transformer/ shards via from_pretrained(subfolder="transformer")
                # INSIDE the locked finalize phase, where unload/cancellation cannot
                # preempt the multi-GB pull. When that path can actually run, pull the
                # shards here in the preemptible prefetch instead. (Pipeline loads
                # already include transformer/ via their own filter.)
                include_transformer = kind == "gguf"
                and self._dense_quant_prefetch_needed(fam, kwargs),
            )
            with self._lock:
                # Stamp progress only if this load is still current; a superseding
                # load (or unload) has its own token and its own _LoadingState.
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.expected_bytes = expected
            # Download outside the lock so unload()/an eviction can preempt the
            # multi-GB pull; load_pipeline below then assembles from the cache.
            self._prefetch_files(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                base_files,
                kwargs.get("hf_token"),
            )
            self.load_pipeline(**kwargs)
            with self._lock:
                # Only clear the marker if this load is still the current one; a
                # newer begin_load (or an unload) has its own token.
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 — surfaced to the client via load_progress
            # A cancelled/superseded load raised below; don't log it as a failure
            # or stamp its error onto whatever load is current now.
            if self._load_token != token:
                return
            logger.error("diffusion.load_failed: %s", exc)
            # Redact native paths: this error is surfaced verbatim via the
            # load-progress poll, and Studio can run as a shared server.
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)

        # Sum the checkpoint repo + companion base cache. For a full-pipeline load the
        # base IS the repo, so count it once (else the bar double-counts to "finalizing").
        downloaded = self._cache_bytes(loading.repo_id)
        if loading.base_repo and loading.base_repo != loading.repo_id:
            downloaded += self._cache_bytes(loading.base_repo)
        expected = loading.expected_bytes
        # Downloads done but pipeline still dequantising / moving to GPU. The cache
        # scan can slightly exceed the estimate (extra cached quants, blob padding),
        # so clamp the reported bytes/fraction so the bar never overshoots 100%.
        if expected > 0 and downloaded >= expected * 0.999:
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        fraction = min(downloaded / expected, 1.0) if expected > 0 else 0.0
        return _progress("downloading", downloaded, expected, fraction)

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base) would yank
        blobs and snapshot files from under the download/assembly."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            return tuple(r for r in (loading.repo_id, loading.base_repo) if r)

    @staticmethod
    def _estimate_download_bytes(
        repo_id: str,
        gguf_filename: Optional[str],
        base_repo: str,
        hf_token: Optional[str],
        *,
        kind: str = "gguf",
        single_file_is_pipeline: bool = False,
        include_transformer: bool = False,
    ) -> tuple[int, list[str]]:
        """Total download size for the progress bar, plus the base-repo files to
        fetch (the prefetch reuses this list, so the base is listed only once).

        For a ``pipeline`` load the whole repo IS the pipeline (``base_repo`` is the
        repo itself), so the transformer/ subfolder is INCLUDED -- unlike the GGUF /
        single-file paths, where the transformer is the single file and the base repo
        supplies only the companions. For a ``single_file_is_pipeline`` family (SDXL) the
        single file is the WHOLE pipeline, so the base repo supplies only config/tokenizer
        (no weights) and its weight files are skipped."""
        from huggingface_hub import HfApi

        api = HfApi()
        total = 0
        base_files: list[str] = []
        try:
            if kind == "pipeline":
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                picked = [s for s in info.siblings if _pipeline_file_downloaded(s.rfilename)]
                # diffusers prefers safetensors per component: drop a .bin whose
                # directory also carries a picked .safetensors weight.
                st_dirs = {
                    s.rfilename.rsplit("/", 1)[0]
                    for s in picked
                    if s.rfilename.endswith(".safetensors")
                }
                for s in picked:
                    if s.rfilename.endswith(".bin") and s.rfilename.rsplit("/", 1)[0] in st_dirs:
                        continue
                    base_files.append(s.rfilename)
                    total += s.size or 0
                return total, base_files
            # Skip the Hub size lookup for a LOCAL gguf path: model_info(repo_id) would
            # raise on a filesystem path and (caught below) skip the base-repo lookup too,
            # so the companion VAE/text-encoder files would never be prefetched and would
            # instead download synchronously under the load lock.
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                total += sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
            # A whole-pipeline single file (SDXL) needs only the base repo's config/tokenizer,
            # not its (unused, multi-GB) weight files.
            if kind == "single_file" and single_file_is_pipeline:
                base_filter = _base_config_file_downloaded
            else:

                def base_filter(rfilename: str) -> bool:
                    return _base_file_downloaded(rfilename, include_transformer = include_transformer)

            base_info = api.model_info(base_repo, files_metadata = True, token = hf_token)
            for s in base_info.siblings:
                if base_filter(s.rfilename):
                    base_files.append(s.rfilename)
                    total += s.size or 0
        except Exception as exc:  # noqa: BLE001 — estimate is best-effort
            logger.warning("diffusion.size_estimate_failed: %s", exc)
        return total, base_files

    @staticmethod
    def _cache_bytes(repo_id: str) -> int:
        from huggingface_hub import constants

        blobs = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "blobs"
        total = 0
        try:
            for entry in blobs.iterdir():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue  # broken symlink / unreadable
        except OSError:
            return 0  # repo not in cache yet
        return total

    @staticmethod
    def _local_dir_weight_bytes(path: Path, *, exclude_transformer: bool) -> int:
        """Sum the on-disk weight files under a local diffusers directory. The HF blob
        cache is empty for a local path, so this is the only size signal for auto memory
        planning; without it a large local model folds to zero and the planner skips
        offload and OOMs. ``exclude_transformer`` drops the ``transformer/`` subfolder
        for GGUF/single-file loads (their transformer is the single file, not resident
        here); a full pipeline load keeps it (the whole repo is resident)."""
        total = 0
        for f in path.rglob("*"):
            if f.suffix.lower() not in (".safetensors", ".bin", ".pt", ".ckpt"):
                continue
            try:
                rel = f.relative_to(path)
            except ValueError:
                continue
            if exclude_transformer and rel.parts and rel.parts[0] == "transformer":
                continue
            try:
                total += f.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def _companion_cache_bytes(base: str) -> int:
        """Resident companion (VAE + text-encoder) size for the memory plan.

        For a hub base repo this is the cached blob total (``_cache_bytes``). For a
        LOCAL diffusers base directory the blob cache is empty, so sum the on-disk
        component weights instead, excluding ``transformer/`` (the GGUF supplies the
        transformer). Without this a local base folds its multi-GB VAE / text-encoder
        weights to zero and auto planning can pick a resident placement that OOMs."""
        local = Path(base).expanduser()
        if local.is_dir():
            return DiffusionBackend._local_dir_weight_bytes(local, exclude_transformer = True)
        return DiffusionBackend._cache_bytes(base)

    # ── Synchronous load / generate / unload ───────────────────────────────

    def load_pipeline(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        _load_token: Optional[int] = None,
    ) -> dict[str, Any]:
        # A blank / whitespace-only token must degrade to anonymous access, not be passed
        # as an explicit credential (from_single_file / from_pretrained / the Hub client
        # can error on a malformed token instead of falling back). Normalize once here so
        # every load branch and the size estimate below use a real token or None.
        hf_token = hf_token.strip() if isinstance(hf_token, str) else hf_token
        hf_token = hf_token or None

        # Validate first (cheap, no torch/diffusers) so a direct call with a bad
        # family fails with ValueError even in a no-diffusers runtime. Sanitize the
        # token here too (direct callers bypass begin_load): a blank string must
        # load anonymously, not 401 as an explicit empty credential.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
        )
        kind = resolve_model_kind(gguf_filename, model_kind)
        # For a full pipeline the repo itself supplies every component, so it is its
        # own base; the single-file kinds resolve the companion base diffusers repo.
        base = (
            repo_id if kind == "pipeline" else _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        )
        target = self._resolve_device_target(fam)
        device, dtype = target.device, target.dtype

        import diffusers

        # Signal an in-flight denoise to abort, then take _generate_lock to WAIT for
        # it to actually exit before allocating the replacement: a load is about to
        # claim VRAM, so unlike unload() it must not overlap a still-live pipeline.
        # The cancel makes that wait ~one step (or the rest of the denoise for a
        # pipeline that ignores the step callback).
        with self._lock:
            # Bail BEFORE signalling any cancel if this load was already superseded (an
            # unload/eviction or a newer load bumped the token while we were resolving /
            # downloading). Otherwise a stale worker would abort an unrelated, still-live
            # generation from the CURRENT model and only then discover it has nothing to do.
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Diffusion load was cancelled.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        with self._generate_lock:
            with self._lock:
                # Re-check under the generate lock: a newer load/unload may have superseded
                # this one while we waited for the in-flight denoise to exit.
                if _load_token is not None and _load_token != self._load_token:
                    raise RuntimeError("Diffusion load was cancelled.")

                # Free the old pipeline before allocating the new one so two
                # checkpoints never sit in VRAM at once.
                self._unload_locked()

                # The single-file kinds resolve a checkpoint path (GGUF or safetensors);
                # the pipeline kind has none (from_pretrained pulls the repo directly).
                single_file_path = (
                    self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
                    if kind in ("gguf", "single_file")
                    else None
                )
                transformer_cls = getattr(diffusers, fam.transformer_class)
                pipeline_cls = getattr(diffusers, fam.pipeline_class)

                # Decide placement up front (the weights are still on CPU, so free VRAM is
                # the real budget) -- this also doubles as the dense-quant preflight: the
                # dense bf16 transformer must fit resident, so the fast path is offered only
                # when the plan is `none`.
                plan = self._plan_memory(
                    target,
                    single_file_path,
                    base,
                    fam,
                    memory_mode,
                    cpu_offload,
                    kind = kind,
                    repo_id = repo_id,
                )

                # Opt-in fast path: load the DENSE bf16 transformer and torchao-quantise it
                # (int8 / fp8 / fp4 tensor cores), which beats GGUF's bf16-rate per-matmul
                # dequant on both speed and quality, at the cost of a higher-memory dense
                # load. Gated on CUDA + bf16 + a resident fit; ANY failure (unsupported arch
                # / scheme, OOM, partial quant) falls back to the GGUF build below. Only the
                # GGUF kind offers it: it materialises the dense bf16 transformer from the
                # base repo, which the safetensors kinds (a single-file or already-quantized
                # pipeline) do not have.
                pipe = None
                transformer_quant_engaged = None
                quant_plan = None
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and plan.offload_policy != OFFLOAD_NONE
                ):
                    # The GGUF-size plan picked offload, but the dense-quant artifact has
                    # a DIFFERENT footprint: int8/fp8 weights are ~half the bf16 bytes, and
                    # a pre-quantized checkpoint never materialises dense bf16 at all. Ask
                    # the auto-policy for the candidate's estimate and re-plan against it:
                    # a resident quantised build beats an offloaded GGUF on speed AND
                    # quality, so it must be attempted before settling for offload.
                    candidate = resolve_dense_quant_candidate(
                        fam = fam,
                        target = target,
                        requested = transformer_quant,
                        base_repo = base,
                        prequant_path = transformer_prequant_path,
                        logger = logger,
                    )
                    if candidate is not None:
                        replanned = self._plan_memory(
                            target,
                            single_file_path,
                            base,
                            fam,
                            memory_mode,
                            cpu_offload,
                            kind = kind,
                            repo_id = repo_id,
                            transformer_resident_override_mib = (candidate.transient_transformer_mib),
                        )
                        if replanned.offload_policy == OFFLOAD_NONE:
                            quant_plan = replanned
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and (plan.offload_policy == OFFLOAD_NONE or quant_plan is not None)
                ):
                    try:
                        pipe, transformer_quant_engaged = self._load_dense_quant_pipeline(
                            transformer_cls,
                            pipeline_cls,
                            base,
                            device,
                            dtype,
                            hf_token,
                            target,
                            transformer_quant,
                            transformer_quant_fast_accum,
                            fam = fam,
                            prequant_path = transformer_prequant_path,
                        )
                    except Exception as exc:  # noqa: BLE001 — fall back to the GGUF build
                        logger.warning(
                            "diffusion.transformer_quant_fallback: %s (loading GGUF)", exc
                        )
                        pipe = None
                        transformer_quant_engaged = None
                        # Drop the exception (and its traceback) BEFORE clearing the cache:
                        # exc.__traceback__ keeps _load_dense_quant_pipeline's frame -- and
                        # thus its partially-built dense bf16 transformer/pipe -- alive, so
                        # clear_gpu_cache() could not otherwise reclaim that VRAM before the
                        # GGUF build (the OOM-fallback path this cleanup exists for).
                        del exc
                        clear_gpu_cache()
                if transformer_quant_engaged is not None and quant_plan is not None:
                    # The re-planned resident placement is the one the engaged dense build
                    # actually uses; the GGUF-size plan stays in force for the fallback.
                    plan = quant_plan

                if pipe is None:
                    if kind == "pipeline":
                        # Full diffusers repo: from_pretrained pulls every component
                        # (transformer + VAE + text encoders + scheduler) from the repo
                        # and re-applies any embedded quantization_config (e.g. bnb-4bit),
                        # so a pre-quantized pipeline reloads quantized with no extra config.
                        if fam.name == KREA2_FAMILY_NAME:
                            # The krea repo ships transformers-5.x style configs the 4.x
                            # line cannot parse; assemble the pipeline per-component
                            # (see diffusion_krea2.py for the exact compat story).
                            pipe = load_krea2_pipeline(repo_id, dtype, hf_token = hf_token)
                        else:
                            pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
                            if hf_token:
                                pipe_kwargs["token"] = hf_token
                            pipe = pipeline_cls.from_pretrained(repo_id, **pipe_kwargs)
                    elif kind == "single_file" and fam.single_file_is_pipeline:
                        # A single-file SDXL-style checkpoint is the WHOLE pipeline
                        # (U-Net + VAE + both text encoders), not a transformer-only file,
                        # so load it through the pipeline class. ``config`` points at the
                        # base repo so diffusers builds the correct structure/scheduler
                        # around the single-file weights instead of guessing from the file.
                        sf_pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "config": base}
                        if hf_token:
                            sf_pipe_kwargs["token"] = hf_token
                        pipe = pipeline_cls.from_single_file(single_file_path, **sf_pipe_kwargs)
                    else:
                        # Single-file transformer; the VAE / text-encoder / scheduler come
                        # from the base diffusers repo (the single file is transformer-only).
                        sf_kwargs: dict[str, Any] = {
                            "torch_dtype": dtype,
                            "config": base,
                            "subfolder": "transformer",
                            # Forward the token: the config is fetched from the (possibly
                            # gated) base repo before from_pretrained can authenticate.
                            "token": hf_token,
                        }
                        if kind == "gguf":
                            # Dequantise the GGUF transformer on-device at the compute dtype.
                            sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                                compute_dtype = dtype
                            )
                        # A safetensors single-file (e.g. fp8) carries its own dtype, so no
                        # GGUF dequant config is passed.
                        transformer = transformer_cls.from_single_file(
                            single_file_path, **sf_kwargs
                        )

                        if fam.name == KREA2_FAMILY_NAME:
                            pipe = load_krea2_pipeline(
                                base, dtype, hf_token = hf_token, transformer = transformer
                            )
                        else:
                            pipe_kwargs = {"torch_dtype": dtype, "transformer": transformer}
                            if hf_token:
                                pipe_kwargs["token"] = hf_token
                            pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)

                # Resolve the effective speed mode: GGUF models default to the
                # near-lossless `default` profile (compile is ~2.2x and sits below
                # the quant noise floor), dense models stay bit-identical `off`. An
                # explicit speed_mode (incl. "off") is honored verbatim.
                effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
                # A torchao-quantized dense transformer runs its matmuls through the
                # regional torch.compile; UNcompiled (eager) it is ~30x slower and would
                # lose to the GGUF fallback. A dense model otherwise resolves to `off`, so
                # force at least `default` (regional compile) whenever the quant engaged,
                # or the opt-in "fast" path silently commits an eager, pathologically slow
                # pipeline.
                if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
                    logger.info(
                        "diffusion.transformer_quant: forcing speed_mode=default "
                        "(quantized transformer must be compiled; eager is ~30x slower)"
                    )
                    effective_speed = SPEED_DEFAULT
                # Opt-in speed optims run BEFORE placement (channels_last / compile
                # must precede CPU offload). Snapshot the process-wide backend flags
                # first so unload can restore them: TF32 / cudnn.benchmark are global,
                # and a later `off` load must not inherit this load's settings.
                backend_flags_before = snapshot_backend_flags()
                # Pick the attention kernel BEFORE compile (compile traces attention). auto
                # upgrades to cuDNN fused attention on NVIDIA when a speed profile is active
                # (~1.18x, near-lossless); an explicit backend is honored, falling back to
                # the diffusers default if its kernel is unavailable. Orthogonal to the
                # weight quant -- it speeds the QK/PV matmuls torchao does not touch.
                attention_engaged = apply_attention_backend(
                    pipe,
                    select_attention_backend(
                        target, attention_backend, speed_active = effective_speed != SPEED_OFF
                    ),
                    logger = logger,
                )
                # Step caching (First-Block-Cache), also before compile. For many-step
                # models it reuses the transformer tail across steps (~1.4x on Flux at
                # LPIPS ~0.08). When engaged, compile must drop fullgraph (the cache's
                # per-step decision is a graph break), so pass it through.
                # Tri-state request: unset / "auto" lets the step-count policy decide
                # (engage when this model's DEFAULT schedule reaches FBCACHE_MIN_STEPS,
                # then re-check against the actual step count on every generation);
                # explicit "off" / "fbcache" are pinned and never toggled.
                cache_request = normalize_transformer_cache(transformer_cache)
                cache_auto = transformer_cache is None or cache_request == TC_AUTO
                cache_quant_active = (
                    transformer_quant_engaged is not None or bool(gguf_filename)
                )
                default_steps: Optional[int] = None
                if cache_auto:
                    default_steps, _ = default_generation_params(
                        gguf_filename, repo_id, base, fam.name
                    )
                    cache_request = (
                        TC_FBCACHE if default_steps >= FBCACHE_MIN_STEPS else None
                    )
                cache_engaged = apply_step_cache(
                    pipe,
                    mode = cache_request,
                    threshold = transformer_cache_threshold,
                    # GGUF transformers are quantized too (the default Studio path), so the
                    # cache needs the higher quantized threshold to still trigger -- not just
                    # the dense-quant fast path.
                    quant_active = cache_quant_active,
                    logger = logger,
                )
                # An auto decision can flip at generation time, but only on a transformer
                # that supports caching at all; a non-CacheMixin transformer (e.g.
                # Z-Image) can never engage, so compile keeps fullgraph there.
                cache_may_toggle = cache_auto and callable(
                    getattr(getattr(pipe, "transformer", None), "enable_cache", None)
                )
                if cache_auto:
                    if cache_engaged:
                        cache_reason = (
                            f"auto: {default_steps}-step default schedule reaches "
                            f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                        )
                    elif cache_request is not None:
                        cache_reason = "auto: model does not support step caching"
                    else:
                        cache_reason = (
                            f"auto: {default_steps}-step default schedule is below "
                            f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                        )
                else:
                    cache_reason = "requested"
                # Install the shared compile-safe eager patches (fused RMSNorm /
                # AdaLayerNorm) for any active speed tier. They are class-level, idempotent
                # and math-equivalent (FMA / fused -> neutral under compile, equal-or-more
                # accurate), so they help eager AND compiled runs. The bit-identical `off`
                # reference path must run with them UNINSTALLED, so uninstall there.
                #
                # Everything from here to the _LoadState commit mutates PROCESS-WIDE state
                # (class patches, TORCHINDUCTOR_CACHE_DIR, backend flags). _unload_locked only
                # reverses it via _state, so a failure BEFORE the commit would leak it (and
                # break the next `off` load's bit-identity). Guard the whole block: on any
                # pre-commit failure, restore everything; on success the commit transfers
                # ownership to _state and _unload_locked takes over.
                # The GGUF-specific speed lever (compiled dequant) applies only when the
                # GGUF transformer was ACTUALLY loaded. On the dense torchao-quant
                # fast path (fp8 / int8 / fp4) `gguf_filename` is still set as the fallback,
                # but `pipe.transformer` is dense (no GGUFLinear), and those schemes need the
                # REGIONAL block compile (dynamic quant is ~30x slower eager), not the GGUF
                # dequant compile -- so treat the transformer as non-GGUF here. The
                # safetensors kinds (single_file / pipeline) likewise have no GGUFLinear.
                gguf_transformer = kind == "gguf" and transformer_quant_engaged is None

                eager_patched = False
                compile_ctx = None
                state_committed = False
                # Lazy import: these patch modules import torch at module level, so
                # importing them here (not at module load) keeps diffusion.py torch-free
                # to import, letting get_diffusion_backend() run on a torchless native install.
                from .diffusion_eager_patches import (
                    install_compile_safe_patches,
                    uninstall_patches,
                )
                from .diffusion_arch_patches import (
                    install_arch_patches,
                    uninstall_arch_patches,
                )

                try:
                    if effective_speed != SPEED_OFF:
                        install_compile_safe_patches()
                        # Per-arch compile-safe fusions (qwen _modulate / z-image residual
                        # addcmul, etc.). Also neutral under compile, so on for every active
                        # tier; tracked by the same eager_patched flag for uninstall.
                        install_arch_patches()
                        eager_patched = True
                    else:
                        uninstall_patches()
                        uninstall_arch_patches()

                    # Pre-warmed torch.compile cache (Mega-cache): when a compiled tier will
                    # run, point inductor at a per-fingerprint dir and load a matching bundle
                    # BEFORE the first compiled forward, so the one-time 25-58s compile can be
                    # paid once (by us / a first run) and reused. A miss is silent -> local
                    # compile, exactly as today.
                    if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and compile_eligible(
                        target, is_gguf = gguf_transformer, family = fam
                    ):
                        compile_ctx = compile_cache.begin(
                            family = fam.name,
                            transformer = getattr(pipe, "transformer", None),
                            dtype = getattr(target, "dtype", None),
                            quant = transformer_quant_engaged,
                            attention_backend = attention_engaged,
                            compile_kwargs = {
                                # Mirrors apply_speed_optims' fullgraph decision: an active
                                # step cache OR a planned offload graph-breaks, so the cached
                                # bundle must be keyed on the same fullgraph setting. An auto
                                # cache that could still engage mid-session also drops
                                # fullgraph: enabling FBCache under a fullgraph-compiled
                                # transformer would crash the first cached generation.
                                "fullgraph": cache_engaged is None
                                and not cache_may_toggle
                                and plan.offload_policy == OFFLOAD_NONE,
                                "dynamic": effective_speed != SPEED_MAX,
                                "mode": "max-autotune-no-cudagraphs"
                                if effective_speed == SPEED_MAX
                                else "default",
                            },
                            logger = logger,
                        )

                    speed_applied = apply_speed_optims(
                        pipe,
                        target,
                        is_gguf = gguf_transformer,
                        family = fam,
                        speed_mode = effective_speed,
                        cache_active = cache_engaged is not None or cache_may_toggle,
                        # The planned offload policy: group/model/sequential offload installs
                        # compiler-disabled onload hooks, so compile must drop fullgraph.
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )
                    if transformer_quant_engaged is not None and not speed_applied.get("compiled"):
                        # Promotion above could not engage compile (e.g. the family is not
                        # compile-friendly, or compile_repeated_blocks failed): the quantized
                        # transformer is now running eager, which is far slower than the GGUF
                        # path it replaced. Surface it loudly rather than hiding the regression.
                        logger.warning(
                            "diffusion.transformer_quant: %s engaged but the transformer is NOT "
                            "compiled; eager torchao quant is ~30x slower than GGUF here",
                            transformer_quant_engaged,
                        )
                    # Quantise the dense companion text encoder(s) (opt-in fp8 / nvfp4),
                    # also before placement so the offload hooks move the smaller weights.
                    te_quant = quantize_text_encoders(
                        pipe,
                        target,
                        mode = text_encoder_quant,
                        logger = logger,
                    )

                    # Apply the placement planned above (from MEASURED free device memory vs
                    # the model's estimated resident size). apply_memory_plan returns the
                    # (policy, tiling) ACTUALLY engaged (it may fall back to whole-module
                    # offload, and tiling is a no-op on a pipeline with no tiling control), so
                    # status stays honest. The dense fast path already placed the pipe
                    # resident; for the `none` policy this is an idempotent re-placement.
                    effective_policy, effective_tiling = apply_memory_plan(
                        pipe, plan, device = device, logger = logger
                    )

                    # Per-control provenance for status: what engaged and who decided it
                    # (the caller, or this backend's auto resolution). cpu_offload=False is
                    # the unset default, so only True counts as an explicit request.
                    resolved = build_resolved_record(
                        {
                            "speed_mode": (
                                speed_mode,
                                effective_speed,
                                "quantized transformer requires compile"
                                if transformer_quant_engaged is not None
                                and normalize_speed_mode(speed_mode) in (None, SPEED_OFF)
                                else "per-kind default"
                                if speed_mode is None
                                else "requested",
                            ),
                            "transformer_quant": (
                                transformer_quant,
                                transformer_quant_engaged or "off",
                                "not engaged (GGUF transformer loaded)"
                                if transformer_quant_engaged is None
                                else "re-planned resident for the quantised artifact"
                                if quant_plan is not None
                                else "engaged on the dense fast path",
                            ),
                            "attention_backend": (
                                attention_backend,
                                attention_engaged or "native",
                                "cuDNN fused attention upgrade"
                                if attention_engaged and attention_backend is None
                                else "diffusers default"
                                if attention_engaged is None
                                else "requested",
                            ),
                            "memory_mode": (
                                memory_mode,
                                effective_policy,
                                "planned from measured free VRAM vs estimated footprint",
                            ),
                            "transformer_cache": (
                                None if cache_auto else transformer_cache,
                                cache_engaged or "off",
                                cache_reason,
                            ),
                            "cpu_offload": (
                                True if cpu_offload else None,
                                effective_policy != OFFLOAD_NONE,
                                "legacy flag" if cpu_offload else "from the memory plan",
                            ),
                        }
                    )

                    self._state = _LoadState(
                        pipe = pipe,
                        family = fam,
                        repo_id = repo_id,
                        base_repo = base,
                        device = device,
                        dtype = str(dtype).replace("torch.", ""),
                        kind = kind,
                        cpu_offload = effective_policy != OFFLOAD_NONE,
                        offload_policy = effective_policy,
                        vae_tiling = effective_tiling,
                        memory_mode = plan.requested_mode,
                        speed_mode = effective_speed,
                        speed_optims = tuple(k for k, v in speed_applied.items() if v),
                        backend_flags_before = backend_flags_before,
                        text_encoder_quant = te_quant,
                        transformer_quant = transformer_quant_engaged,
                        attention_backend = attention_engaged,
                        transformer_cache = cache_engaged,
                        cache_auto = cache_may_toggle,
                        cache_quant_active = cache_quant_active,
                        cache_threshold = transformer_cache_threshold,
                        eager_patched = eager_patched,
                        compile_cache_ctx = compile_ctx,
                        hf_token = hf_token,
                        resolved = resolved,
                    )
                    state_committed = True
                finally:
                    # Pre-commit failure: nothing owns the process-wide mutations yet, so
                    # roll them back here (symmetric with _unload_locked).
                    if not state_committed:
                        restore_backend_flags(backend_flags_before)
                        compile_cache.restore(compile_ctx)
                        # apply_speed_optims may have installed the compiled GGUF dequant
                        # before a later step failed; uninstall is idempotent.
                        gguf_compile.uninstall_all()
                        if eager_patched:
                            uninstall_patches()
                            uninstall_arch_patches()
                        # Also free the half-built pipe's VRAM: the failed load never
                        # commits _state, so nothing else reclaims it until the next unload.
                        clear_gpu_cache()

        logger.info(
            "diffusion.loaded: repo=%s base=%s device=%s offload=%s tiling=%s reasons=%s",
            repo_id,
            base,
            device,
            effective_policy,
            effective_tiling,
            "; ".join(plan.reasons),
        )
        return self.status()

    def _load_dense_quant_pipeline(
        self,
        transformer_cls: Any,
        pipeline_cls: Any,
        base: str,
        device: str,
        dtype: Any,
        hf_token: Optional[str],
        target: DiffusionDeviceTarget,
        mode: Optional[str],
        fast_accum: Optional[bool] = None,
        *,
        fam: Optional[DiffusionFamily] = None,
        prequant_path: Optional[str] = None,
    ) -> tuple[Any, str]:
        """Build the opt-in fast pipeline and return ``(pipe, engaged_scheme)``.

        Two ways to get the quantized transformer, in order:

        1. Pre-quantized: if a checkpoint is configured for the chosen scheme (an explicit
           ``prequant_path`` or the family's hosted repo), load the already-quantized
           weights onto the meta device and assign them in -- the dense bf16 never lands on
           the GPU, so the load peak is ~half and the download is smaller.
        2. Dense + quantise (fallback): load the DENSE bf16 transformer from the base repo,
           place it on the device, and torchao-quantise it in place.

        Raises if the scheme is unsupported or quantisation fails, so ``load_pipeline``
        catches it and falls back to the GGUF build. Quantisation runs ON the device and
        BEFORE the loader compiles the repeated block, so the order stays quantize ->
        compile -> placement."""
        # 1. Pre-quantized checkpoint, when one is configured for the resolved scheme.
        scheme = select_transformer_quant_scheme(target, mode)
        if scheme is None:
            # Bail BEFORE the (multi-GB) dense download: an explicit unsupported scheme
            # (e.g. fp8 on Ampere, nvfp4 off Blackwell) would otherwise materialise the
            # dense transformer and move the pipe to CUDA only to fail at quantize below --
            # a long finalization under the load lock after the old model was already
            # evicted. load_pipeline catches this and builds the GGUF pipeline instead.
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None:
            source = resolve_prequant_source(fam, scheme, path_override = prequant_path)
            if source is not None:
                transformer = load_prequantized_transformer(
                    transformer_cls,
                    base,
                    source,
                    device = device,
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint built with a different Linear filter than the
                    # dense path uses, so the prequant and runtime-quant models match.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Only enforced when the caller forces fp8 fast-accum: a checkpoint that
                    # baked the other choice would ignore the request, so fall to the dense
                    # path (which applies it) instead of silently using the baked kernels.
                    fast_accum = fast_accum,
                    logger = logger,
                )
                if transformer is not None:
                    pipe = self._assemble_pipe(
                        pipeline_cls, base, transformer, dtype, hf_token, device
                    )
                    return pipe, scheme

        # 2. Fallback: materialise the dense bf16 transformer and quantise it on-device.
        transformer = transformer_cls.from_pretrained(
            base, subfolder = "transformer", torch_dtype = dtype, token = hf_token
        )
        pipe = self._assemble_pipe(pipeline_cls, base, transformer, dtype, hf_token, device)
        scheme = quantize_transformer(pipe, target, mode = mode, fast_accum = fast_accum, logger = logger)
        if scheme is None:
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        return pipe, scheme

    @staticmethod
    def _assemble_pipe(
        pipeline_cls: Any,
        base: str,
        transformer: Any,
        dtype: Any,
        hf_token: Optional[str],
        device: str,
    ) -> Any:
        """Assemble the diffusers pipeline around ``transformer`` and place it on ``device``
        (a no-op for an already-placed pre-quantized transformer; it moves the companions)."""
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": transformer}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)
        pipe.to(device)
        return pipe

    def _plan_memory(
        self,
        target: DiffusionDeviceTarget,
        single_file_path: Optional[str],
        base: str,
        fam: DiffusionFamily,
        memory_mode: Optional[str],
        cpu_offload: bool,
        *,
        kind: str = "gguf",
        repo_id: Optional[str] = None,
        transformer_resident_override_mib: Optional[int] = None,
    ):
        """Build the memory plan for this load: snapshot free device memory and
        estimate the model's resident footprint, then let the planner pick an
        offload policy + VAE memory savers. Kept on the backend so the cached base
        repo (companion text-encoder / VAE) feeds the size estimate.

        The size estimate is per-kind: diffusers keeps GGUF weights packed (per-matmul
        transient dequant), so a GGUF loads near its on-disk size; a safetensors
        single-file loads near its on-disk size (it carries its dtype); and a full
        pipeline is one cached download (transformer + companions), already compressed.
        ``transformer_resident_override_mib`` replaces the file-size transformer estimate
        when the loader is planning for a DIFFERENT artifact than the file on disk (the
        dense transformer-quant candidate, whose footprint the auto-policy estimates)."""
        device_memory = snapshot_device_memory(target)
        if kind == "pipeline":
            # The whole repo (transformer + companions) is one cached download; the
            # cached bytes are the resident estimate (bnb-4bit / fp8 stay compressed).
            # A LOCAL pipeline path isn't in the HF blob cache, so sum its on-disk weights
            # (transformer included) instead of folding to zero and skipping offload.
            local_repo = Path(repo_id).expanduser() if repo_id else None
            if local_repo is not None and local_repo.is_dir():
                cached = self._local_dir_weight_bytes(local_repo, exclude_transformer = False)
            else:
                cached = self._cache_bytes(repo_id) if repo_id else 0
            cached_mib = int(cached // (1024 * 1024)) if cached else None
            model_dense_mib = estimate_safetensors_dense_mib(cached_mib)
            companion_mib = None
        else:
            if transformer_resident_override_mib is not None:
                # Planning for a different artifact than the file on disk (the dense
                # transformer-quant candidate): the auto-policy's estimate replaces the
                # file-size derivation; companions below stay measured from the cache.
                transformer_resident = transformer_resident_override_mib
            elif kind == "single_file":
                # Safetensors single-file: no dequant expansion (it carries its dtype).
                transformer_resident = estimate_safetensors_dense_mib(
                    file_size_mib(single_file_path)
                )
            else:
                transformer_resident = estimate_gguf_resident_mib(file_size_mib(single_file_path))
            # The companion components (VAE + text encoders) load near their on-disk
            # size; sum whatever the prefetch placed in the base-repo cache, or -- for a
            # LOCAL diffusers base -- the on-disk component weights (the blob cache is
            # empty for a local path, which would otherwise fold multi-GB companions to 0
            # and let auto planning pick a resident placement that OOMs).
            companion = self._companion_cache_bytes(base)
            companion_mib = int(companion // (1024 * 1024)) if companion else None
            model_dense_mib = None
            if transformer_resident is not None:
                model_dense_mib = transformer_resident + (companion_mib or 0)
        # Feed the variant hint (single-file basename + base/repo) next to the family name
        # so estimate_image_runtime_mib sees distilled markers ("turbo"/"schnell") that
        # detect_family normalizes out of fam.name -- distilled models need ~15% less
        # activation headroom, and over-reserving can force needless offload / tiling.
        variant_hint = " ".join(
            p
            for p in (
                fam.name,
                Path(single_file_path).name if single_file_path else "",
                repo_id or base or "",
            )
            if p
        )
        runtime_headroom = estimate_image_runtime_mib(width = None, height = None, family = variant_hint)
        return plan_diffusion_memory(
            target = target,
            device_memory = device_memory,
            model_dense_mib = model_dense_mib,
            companion_dense_mib = companion_mib,
            runtime_headroom_mib = runtime_headroom,
            requested_mode = memory_mode,
            explicit_offload = cpu_offload,
        )

    def _workflow_pipe(self, state: _LoadState, class_name: Optional[str], workflow: str) -> Any:
        """The diffusers pipeline for an image-conditioned ``workflow``, built once and
        cached. ``Pipeline.from_pipe`` re-wires the loaded text-to-image pipe's resident
        modules (transformer/VAE/text-encoder, incl. any compiled/quantised state) into
        the workflow pipeline class, so there is no extra VRAM and no reload. Raises a
        clear ValueError when the family does not support the workflow."""
        if not class_name:
            raise ValueError(
                f"{workflow} is not supported for the '{state.family.name}' model family."
            )
        cached = self._aux_pipes.get(class_name)
        if cached is not None:
            return cached
        import diffusers

        # torch_dtype=None is load-bearing: diffusers' from_pipe defaults torch_dtype to
        # torch.float32 and then runs new_pipeline.to(dtype=float32) over EVERY component.
        # That recast (a) needlessly upcasts the reused bf16 modules and (b) hard-crashes
        # on the dense-quant fast path -- a torchao-quantized + torch.compiled transformer
        # has tensor-subclass Linear weights that torch.nn.Module._apply cannot swap_tensors
        # ("Couldn't swap Linear.weight"). Passing None makes from_pipe skip the cast and
        # reuse the resident modules AT THEIR LOADED dtype, which is the whole point of
        # from_pipe (component reuse, no reload, no extra VRAM).
        pipe = getattr(diffusers, class_name).from_pipe(state.pipe, torch_dtype = None)
        # Only publish to the shared aux cache if THIS load is still current. from_pipe runs
        # under _generate_lock but NOT _lock, so an unload()/superseding load can clear
        # _aux_pipes and null _state while it builds; caching unconditionally would re-insert
        # a wrapper over now-stale modules that a later same-workflow load would reuse (or
        # keep the old VRAM pinned). This generation still uses the returned pipe.
        with self._lock:
            if self._state is state:
                self._aux_pipes[class_name] = pipe
        return pipe

    def _controlnet_pipe(self, state: _LoadState, resolved_cn: Any, cancel: threading.Event) -> Any:
        """Build (once, cached) the family's diffusers ControlNet pipeline around the requested
        ControlNet model. The ControlNet model is a small extra module loaded via from_pretrained
        and cached by id; the pipeline is assembled with ``Pipeline.from_pipe(base,
        controlnet=model)`` -- reusing the resident base modules at their loaded dtype (no reload,
        no recast; torch_dtype=None for the same reason as _workflow_pipe). Raises a clear
        ValueError when the family declares no ControlNet classes."""
        fam = state.family
        pipe_cls_name = getattr(fam, "controlnet_pipeline_class", None)
        model_cls_name = getattr(fam, "controlnet_model_class", None)
        if not pipe_cls_name or not model_cls_name:
            raise ValueError(f"ControlNet is not supported for the '{fam.name}' model family.")
        import diffusers

        cn_model = self._cn_models.get(resolved_cn.id)
        if cn_model is None:
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            import torch

            # state.dtype is the display string saved at load ("bfloat16"), NOT a
            # torch.dtype; pass the real dtype so diffusers loads the ControlNet at the
            # base compute dtype instead of silently defaulting to float32 (extra VRAM).
            cn_dtype = getattr(torch, str(state.dtype).replace("torch.", ""), None)
            cn_model = getattr(diffusers, model_cls_name).from_pretrained(
                resolved_cn.path,
                torch_dtype = cn_dtype,
                # An empty / malformed token means anonymous access; the HF client can
                # raise on a blank credential instead of falling back, so coerce to None.
                token = state.hf_token or None,
            )
            if cancel.is_set():
                # An unload/eviction raced the blocking download above and may have already
                # cleared the load. Bail BEFORE any device placement so we don't allocate
                # several GB onto the GPU after _unload_locked() freed it (which would OOM
                # or make the unload appear to free memory only to repopulate it).
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            # Placement must follow the base model's offload policy. A resident base moves
            # the ControlNet resident too; an offloaded (low-VRAM) base streams it through
            # the device with group offloading instead of forcing the whole module onto the
            # GPU, which would defeat the offload and risk an OOM. Best-effort: any failure
            # falls back to the resident placement (the prior behaviour).
            if getattr(state, "offload_policy", OFFLOAD_NONE) != OFFLOAD_NONE and (
                _offload_controlnet_module(cn_model, state.device, logger)
            ):
                pass
            else:
                cn_model = cn_model.to(state.device)
            if cancel.is_set():
                # An unload raced the blocking download above and already cleared the
                # ControlNet caches; caching now would pin the module past the unload.
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            self._cn_models[resolved_cn.id] = cn_model
        key = (pipe_cls_name, resolved_cn.id)
        pipe = self._cn_pipes.get(key)
        if pipe is None:
            pipe = getattr(diffusers, pipe_cls_name).from_pipe(
                state.pipe, controlnet = cn_model, torch_dtype = None
            )
            self._cn_pipes[key] = pipe
        return pipe

    @staticmethod
    def _align_vae_dtype(pipe: Any, denoiser_attr: str = "transformer") -> None:
        """Cast the VAE to the denoiser's compute dtype before an image-conditioned
        call. The img2img/inpaint pipelines VAE-encode the input image at the text-
        encoder dtype (bf16), but a prior txt2img DECODE may have left the shared VAE
        upcast to fp32 (its ``force_upcast`` path), so the encode would mismatch
        (bf16 image vs fp32 VAE). Re-aligning here is safe: our families run bf16 or
        fp32 only (the fp16 guard promotes fp16), and a later txt2img decode re-upcasts
        as needed. ``denoiser_attr`` is ``pipe.transformer`` for DiT families and
        ``pipe.unet`` for SDXL. Best-effort; a no-op when already aligned."""
        denoiser = getattr(pipe, denoiser_attr, None)
        vae = getattr(pipe, "vae", None)
        if denoiser is None or vae is None:
            return
        try:
            # Read the dtype from the parameters (not denoiser.dtype): a plain nn.Module
            # has no .dtype, and a torch.compile'd / wrapped denoiser can obscure it. Take
            # the first FLOATING dtype: a GGUF-quantized transformer's leading params are
            # packed uint8 storage, and nn.Module.to() rejects integer dtypes outright.
            target_dtype = next(
                (p.dtype for p in denoiser.parameters() if p.dtype.is_floating_point),
                None,
            )
            if target_dtype is None:
                return
            if next(vae.parameters()).dtype != target_dtype:
                vae.to(dtype = target_dtype)
        except (StopIteration, AttributeError, RuntimeError, TypeError):
            pass

    def _apply_loras(
        self, state: Any, loras: Optional[list[tuple[str, float]]], cancel: threading.Event
    ) -> None:
        """Load + activate requested LoRA adapters on ``state.pipe`` (non-fused), or clear
        them when none are requested.

        The applied set is recorded on the pipe object, so an unchanged selection is a no-op
        and a model swap (a fresh pipe with no marker) resets naturally. Never fuses: fusing
        breaks on quantized (bnb-4bit / torchao) transformers and blocks live weight tweaks.
        """
        from core.inference import diffusion_lora

        pipe = state.pipe
        current = getattr(pipe, "_unsloth_loras", ())
        specs = [(i, w) for (i, w) in (loras or []) if w != 0]

        if not specs:
            if current:
                try:
                    pipe.unload_lora_weights()
                except Exception:  # noqa: BLE001 -- best-effort clear
                    pass
                pipe._unsloth_loras = ()
            return

        if not diffusion_lora.supports_lora(
            engine = "diffusers",
            family = getattr(state.family, "name", None),
            model_kind = state.kind,
            transformer_quant = state.transformer_quant,
            compiled = "compiled" in (getattr(state, "speed_optims", ()) or ()),
        ):
            raise ValueError(
                "LoRA is not supported for this model/quantisation on the diffusers engine "
                "(GGUF-via-diffusers, torchao fp8/int8, or a torch.compile'd Speed=default/max "
                "load). Use a bf16 or bnb-4bit load at Speed=off/eager, or the native engine "
                "for GGUF models."
            )

        resolved = diffusion_lora.resolve_specs(specs, hf_token = state.hf_token, cancel_event = cancel)
        # The shared catalog scans both .safetensors and .gguf, but diffusers'
        # load_lora_weights only takes safetensors; a .gguf adapter would otherwise fail
        # deep in generation. Reject it here as a clean 400 before touching the pipe.
        bad = [r.id for r in resolved if r.fmt != "safetensors"]
        if bad:
            raise ValueError(
                "GGUF LoRA adapters are not supported on the diffusers engine "
                f"({', '.join(bad)}); use a .safetensors adapter, or the native engine."
            )
        # Unique adapter names (diffusers requires distinct names; sanitized stems can collide).
        uniq: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for r in resolved:
            name = r.alias
            n = 1
            while name in seen:
                n += 1
                name = f"{r.alias}_{n}"
            seen.add(name)
            uniq.append((name, r.path, r.weight))

        desired = tuple(uniq)
        if desired == current:
            return
        try:
            if current:
                pipe.unload_lora_weights()
            for name, path, _weight in uniq:
                pipe.load_lora_weights(path, adapter_name = name)
            pipe.set_adapters(
                [name for name, _p, _w in uniq], adapter_weights = [w for _n, _p, w in uniq]
            )
        except Exception as exc:  # noqa: BLE001 -- surface as a clean 400
            try:
                pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            pipe._unsloth_loras = ()
            raise ValueError(f"Failed to apply LoRA: {exc}") from exc
        pipe._unsloth_loras = desired

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clear the transformer's stateful step cache (FBCache) before a generation.

        diffusers keys FBCache residuals by cache context ("cond"/"uncond") on the
        long-lived transformer, and neither the pipeline nor the context exit resets
        them (``StateManager`` only clears via ``reset_stateful_hooks``, which no
        pipeline calls). This backend reuses one resident pipe across generations, so
        without a reset the next generation's first step compares its first-block
        residual against the PREVIOUS request's -- a tensor-shape mismatch when the
        resolution/batch changed, or a stale-cache reuse otherwise. Best-effort: a
        transformer without the hook (uncached load) is a silent no-op."""
        transformer = getattr(pipe, "transformer", None)
        reset = getattr(transformer, "reset_stateful_hooks", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 — reset is best-effort, never fail a generation
                pass

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        # Fallbacks for a caller that passes nothing; the route always sends the
        # per-model values the UI seeds (few steps / no CFG for distilled models,
        # more steps / real CFG for full ones).
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        # Image-conditioned workflows (base64 / data-URL): an init image alone selects
        # img2img; an init image + mask selects inpaint. ``strength`` is the img2img/
        # inpaint denoise strength (0 = keep source, 1 = full redraw). None = txt2img.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        # Upscale (hires fix): a factor > 1 with an init image enlarges the input and
        # re-denoises it at low strength to paint detail at the higher resolution.
        upscale: Optional[float] = None,
        # Reference workflow (FLUX.2): ADDITIONAL reference images beyond ``init_image``. The
        # pipeline accepts a list, so multiple references can be combined (subject + style,
        # character + scene). Ignored by non-reference workflows.
        reference_images: Optional[list[str]] = None,
        # LoRA adapters as (id, weight) pairs; loaded onto the pipe (non-fused) and activated
        # with set_adapters for this generation. None/empty = no LoRA (adapters cleared).
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet as (id, control_image_b64, control_type, strength, guidance_start,
        # guidance_end); conditions the text-to-image path on a spatial control map. None = off.
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
    ) -> dict[str, Any]:
        import torch
        from PIL import Image

        # A per-generation cancel Event: unload()/a superseding load set THIS event
        # (registered under _lock below) to abort just this denoise. _generate_lock
        # serialises generations and is the only lock the denoise holds, so a slow
        # generation never blocks status()/unload()/a new load.
        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # Register under _lock so unload()/a load can signal THIS generation.
                # A cancel that arrived before now either nulled _state (we raised
                # above) or targets an older generation, so nothing is lost.
                self._active_generate_cancel = cancel
            try:
                # Snapshot taken: the local `state` ref keeps the pipe alive even if
                # unload() nulls _state mid-denoise, so the call below needs no _lock.
                generator = torch.Generator(device = state.device)
                if seed is None:
                    # Draw a fresh random seed but keep it within JS's safe-integer
                    # range (< 2**53), so the reported seed round-trips through JSON
                    # and actually reproduces the image (a raw 64-bit seed would lose
                    # precision in the browser and the recipe couldn't be replayed).
                    seed = generator.seed() & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                generator.manual_seed(seed)

                # Apply/adjust LoRA adapters on the resident pipe (non-fused) before picking
                # the workflow pipe; from_pipe pipes share the transformer, so it propagates.
                self._apply_loras(state, loras, cancel)

                # Select the pipeline for this workflow. txt2img uses the loaded pipe;
                # img2img/inpaint reuse its resident modules via from_pipe (no reload);
                # an edit model's OWN loaded pipe is already the edit pipeline.
                pipe = state.pipe
                init_pil = mask_pil = None
                control_pil = None
                cn_scale = cn_gstart = cn_gend = cn_mode = None
                ref_extra: list = []
                # Validate parameter dependencies up front: mask / upscale / reference all
                # need an input image, and reference conditioning needs a family that
                # supports it. Without these guards an unsupported combination would be
                # silently ignored and quietly fall back to txt2img / img2img.
                if init_image is None:
                    if mask_image is not None:
                        raise ValueError("mask_image requires an input image (init_image).")
                    if upscale is not None and upscale > 1.0:
                        raise ValueError("upscale requires an input image (init_image).")
                    if reference_images:
                        raise ValueError("reference_images require an input image (init_image).")
                if reference_images and not getattr(state.family, "reference", False):
                    raise ValueError(
                        f"Reference images are not supported for the '{state.family.name}' "
                        "model family."
                    )
                if getattr(state.family, "edit", False):
                    # Instruction editing: the loaded pipe is the edit pipeline. It always
                    # needs an input image; the prompt is the edit instruction. No mask, no
                    # from_pipe (the model has no plain text-to-image mode).
                    if init_image is None:
                        raise ValueError(
                            f"{state.family.name} is an image-editing model: provide an input image."
                        )
                    workflow = "edit"
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                elif mask_image is not None and init_image is not None:
                    workflow = "inpaint"
                    pipe = self._workflow_pipe(state, state.family.inpaint_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    mask_pil = _decode_b64_image(mask_image, mode = "L")
                elif init_image is not None and upscale is not None and upscale > 1.0:
                    # Upscale (hires fix): enlarge the input with Lanczos, then re-run the
                    # img2img pipeline on it at a low denoise strength so the transformer
                    # adds high-frequency detail without redrawing the content. Shares the
                    # img2img pipeline/modules via from_pipe (no extra VRAM, no reload).
                    workflow = "upscale"
                    pipe = self._workflow_pipe(state, state.family.img2img_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    iw, ih = init_pil.size
                    # Cap the factor, THEN cap the absolute output: a large input times the
                    # factor (e.g. 1024 at 4x = 4096, or a big upload) would otherwise OOM the
                    # VAE/transformer. Bound the longest side to 2048 (txt2img's own max),
                    # scaling both dims to keep the aspect ratio; round to a multiple of 16
                    # (VAE downsample + patch size require it for our families).
                    factor = max(1.0, min(float(upscale), 4.0))
                    tw_f, th_f = iw * factor, ih * factor
                    max_side = 2048
                    fit = min(1.0, max_side / max(tw_f, th_f))
                    tw = max(16, int(round(tw_f * fit / 16.0)) * 16)
                    th = max(16, int(round(th_f * fit / 16.0)) * 16)
                    # After the absolute cap, the target must still exceed the input, or
                    # "upscale" would shrink it (e.g. a 3000px source at 2x clamps to 2048).
                    # Reject rather than silently return a smaller image than uploaded.
                    if max(tw, th) <= max(iw, ih):
                        raise ValueError(
                            f"Upscale would not enlarge this image: its longest side "
                            f"({max(iw, ih)}px) already meets the {max_side}px output limit. "
                            f"Use a smaller source image."
                        )
                    init_pil = init_pil.resize((tw, th), Image.LANCZOS)
                    if strength is None:
                        # Hires-fix default: low enough to preserve content, high enough to
                        # synthesise new detail at the higher resolution.
                        strength = 0.35
                elif getattr(state.family, "reference", False) and init_image is not None:
                    # FLUX.2-style reference conditioning: the loaded pipe (Flux2KleinPipeline)
                    # takes the reference image directly via its `image` arg and generates a
                    # fresh image at the REQUESTED size, guided by both the prompt and the
                    # reference. No from_pipe (the loaded pipe already supports it), no strength
                    # (reference-conditioning, not a denoise blend), and the output size comes
                    # from the sliders (the pipeline resizes the reference to ~1MP itself).
                    # Checked AFTER inpaint/upscale so a mask/upscale request on a reference
                    # family (FLUX.2-klein also has an inpaint pipeline) still routes correctly.
                    workflow = "reference"
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    # Additional references (FLUX.2 accepts a list): decode them so the
                    # conditioning combines all of them. Capped to keep VRAM bounded.
                    ref_extra = [
                        _decode_b64_image(x, mode = "RGB") for x in (reference_images or [])[:3]
                    ]
                elif init_image is not None:
                    workflow = "img2img"
                    pipe = self._workflow_pipe(state, state.family.img2img_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                else:
                    workflow = "txt2img"

                # ControlNet conditioning (diffusers): applies to the plain text-to-image path.
                # Builds the family's ControlNet pipeline around the resident modules (no reload)
                # and passes a control map. v1 conditions txt2img only (not img2img/inpaint/edit).
                if controlnet is not None:
                    from core.inference import diffusion_controlnet
                    cn_id, cn_image_b64, cn_type, cn_strength, cn_gs, cn_ge = controlnet
                    # strength 0 disables ControlNet (documented on the request model, and the
                    # frontend slider allows it): skip the whole path so a no-op selection never
                    # pays the multi-GB ControlNet download / VRAM cost.
                    if cn_strength in (None, 0, 0.0):
                        controlnet = None
                    else:
                        if workflow != "txt2img":
                            raise ValueError(
                                "ControlNet currently combines with plain text-to-image only, not "
                                f"the {workflow} workflow."
                            )
                        if not diffusion_controlnet.supports_controlnet(
                            engine = "diffusers",
                            family = state.family.name,
                            has_controlnet_pipeline = bool(
                                getattr(state.family, "controlnet_pipeline_class", None)
                            ),
                            model_kind = state.kind,
                            transformer_quant = state.transformer_quant,
                        ):
                            raise ValueError(
                                "ControlNet is not supported for this model/quantisation on the "
                                "diffusers engine (needs a bf16 or bnb-4bit load of a family with a "
                                "ControlNet pipeline; not GGUF-via-diffusers or torchao fp8/int8)."
                            )
                        # Decode + preprocess the control image FIRST so a malformed / unsupported
                        # image fails as a clean 400 BEFORE any ControlNet download or pipe build,
                        # rather than after paying that cost. Control map at the OUTPUT size so it
                        # aligns with the generated latents.
                        src = _decode_b64_image(cn_image_b64, mode = "RGB")
                        control_pil = diffusion_controlnet.preprocess_control(src, cn_type).resize(
                            (width, height), Image.LANCZOS
                        )
                        try:
                            resolved_cn = diffusion_controlnet.resolve_controlnet(
                                cn_id, family = state.family.name
                            )
                        except FileNotFoundError as exc:
                            # An unknown / missing ControlNet id is a bad selection -> 400, not a
                            # generic 500 (the route maps ValueError, not FileNotFoundError).
                            raise ValueError(str(exc)) from exc
                        pipe = self._controlnet_pipe(state, resolved_cn, cancel)
                        workflow = "controlnet"
                        cn_scale, cn_gstart, cn_gend = cn_strength, cn_gs, cn_ge
                        # Flux Union ControlNet selects the active mode by an integer
                        # ``control_mode`` (canny/depth/pose/...); map the chosen control type so
                        # the union model applies the right head instead of a default/wrong one.
                        cn_mode = diffusion_controlnet.union_control_mode(cn_id, cn_type)
                # Auto-resize odd-sized inputs to a multiple of 16 for the workflows whose
                # OUTPUT size is taken from the input image (img2img / inpaint / extend / edit),
                # so an upload like 186px tall no longer fails the pipeline's divisibility check.
                # txt2img/reference use the validated slider size; upscale already produced a /16
                # target. The mask is matched to the snapped image so inpaint stays aligned.
                if init_pil is not None and workflow in ("img2img", "inpaint", "edit"):
                    init_pil = _snap_to_multiple(init_pil, 16)
                    if mask_pil is not None and mask_pil.size != init_pil.size:
                        from PIL import Image as _PILImage
                        mask_pil = mask_pil.resize(init_pil.size, _PILImage.NEAREST)
                if init_pil is not None:
                    # Keep the VAE encode dtype consistent with the input image.
                    # state.family is always a DiffusionFamily, which defines denoiser_attr.
                    self._align_vae_dtype(pipe, state.family.denoiser_attr)

                # Pipelines vary in which kwargs they accept (img2img derives size from the
                # input image and may reject width/height; a distilled pipe may take no
                # negative prompt or step callback), so gate every optional kwarg on the
                # actual signature.
                call_params = inspect.signature(pipe.__call__).parameters

                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    # Most pipelines take guidance via "guidance_scale"; Qwen-Image
                    # uses "true_cfg_scale" (its distilled guidance is off).
                    state.family.cfg_kwarg: guidance,
                    "generator": generator,
                    # Generate the whole batch in one forward pass (VRAM-heavy). All
                    # share this call's seed, drawn sequentially from one generator.
                    "num_images_per_prompt": batch_size,
                }
                if init_pil is not None:
                    # Reference with extra images passes the whole list (FLUX.2 combines them);
                    # every other workflow takes the single image.
                    kwargs["image"] = [init_pil, *ref_extra] if ref_extra else init_pil
                    if mask_pil is not None and "mask_image" in call_params:
                        kwargs["mask_image"] = mask_pil
                    if strength is not None and "strength" in call_params:
                        kwargs["strength"] = strength
                # width/height. txt2img uses the requested slider size. Image-conditioned
                # pipes must use the INPUT IMAGE's own size, NOT the slider: the output is
                # the redrawn/extended input, and the denoise builds latents from the image,
                # so a slider size that differs from the image mismatches (e.g. a 1536px
                # outpaint vs a 1024 slider -> "tensor a (128) must match tensor b (192)").
                # Many img2img/inpaint pipelines drop width/height entirely; pass them only
                # when accepted, derived from the image so they are always consistent.
                if workflow in ("txt2img", "reference", "controlnet"):
                    # txt2img, FLUX.2 reference, and ControlNet all generate at the REQUESTED
                    # size; the reference/control image is resized to match, so it must not be
                    # pinned to an input image's size like img2img/inpaint/upscale are.
                    kwargs["width"] = width
                    kwargs["height"] = height
                elif init_pil is not None:
                    iw, ih = init_pil.size
                    if "width" in call_params:
                        kwargs["width"] = iw
                    if "height" in call_params:
                        kwargs["height"] = ih
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                if workflow == "controlnet" and control_pil is not None:
                    # The ControlNet pipeline takes the control map + its conditioning scale;
                    # guidance start/end bound the step range it acts over. Every kwarg is gated
                    # on the pipe signature so a family whose CN pipe omits one still runs.
                    if "control_image" in call_params:
                        kwargs["control_image"] = control_pil
                    elif "image" in call_params:  # some CN pipelines name it "image"
                        kwargs["image"] = control_pil
                    if "controlnet_conditioning_scale" in call_params and cn_scale is not None:
                        kwargs["controlnet_conditioning_scale"] = cn_scale
                    if "control_guidance_start" in call_params and cn_gstart is not None:
                        kwargs["control_guidance_start"] = cn_gstart
                    if "control_guidance_end" in call_params and cn_gend is not None:
                        kwargs["control_guidance_end"] = cn_gend
                    # Union ControlNet mode index (Flux); only when the pipe accepts it and the
                    # selected control type maps to a known mode.
                    if "control_mode" in call_params and cn_mode is not None:
                        kwargs["control_mode"] = cn_mode

                gen = _GenState(total_steps = steps)

                def _on_step(pipe, step_index, timestep, callback_kwargs):
                    now = time.time()
                    gen.step = step_index + 1
                    if gen.first_step_at == 0.0:
                        gen.first_step_at = now
                    gen.eta_seconds = _estimate_eta(
                        gen.total_steps, gen.step, gen.first_step_at, now
                    )
                    # Preempt a long denoise on unload/eviction or a superseding load:
                    # diffusers checks pipe._interrupt and stops after the current step.
                    if cancel.is_set():
                        pipe._interrupt = True
                    return callback_kwargs

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step

                # An AUTO cache decision is re-checked against the ACTUAL step count:
                # a 28-step dev-style request gains FBCache even when the load's default
                # schedule kept it off, and a few-step turbo request drops it (skipping
                # a step there is a large quality hit). Explicit choices never toggle.
                if state.cache_auto:
                    toggled = maybe_toggle_step_cache(
                        state.pipe,
                        steps = steps,
                        quant_active = state.cache_quant_active,
                        threshold = state.cache_threshold,
                        logger = logger,
                    )
                    if toggled != state.transformer_cache:
                        # _LoadState is frozen (loads swap it as one unit); this is the
                        # one deliberate in-place update, tracking the pipe-level toggle
                        # that already happened so status() reports the true cache state.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                # Start each generation from a clean step cache: FBCache residuals from
                # a prior request on this resident pipe would otherwise be compared
                # against this generation's first step (shape mismatch on a resolution/
                # batch change, or stale reuse). No-op when no cache is engaged.
                if state.transformer_cache:
                    self._reset_step_cache(state.pipe)

                self._gen = gen
                try:
                    # inference_mode is strictly faster than the no_grad diffusers
                    # uses internally and numerically identical for inference.
                    with torch.inference_mode():
                        images = pipe(**kwargs).images
                finally:
                    self._gen = None
                # A cancelled denoise returns early with a partial/garbage image;
                # don't hand it back to be persisted.
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # The first compiled generation just paid the compile cost; persist the
                # warm torch.compile cache bundle when saving is enabled (distributor /
                # first-run warm). Idempotent + best-effort -- never fails a generation.
                try:
                    compile_cache.save(state.compile_cache_ctx, logger = logger)
                except Exception:  # noqa: BLE001 — cache persistence is best-effort
                    pass
                # Return the PIL images (not yet encoded): the route embeds each
                # image's recipe and persists it via the gallery.
                return {"images": list(images), "seed": int(seed), "repo_id": state.repo_id}
            finally:
                # Deregister so a later unload/load can't poke a finished generation
                # (only if still ours — a newer generation may have replaced it).
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    def generate_progress(self) -> dict[str, Any]:
        """Live per-step progress for an in-flight generation (lock-free read)."""
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
            }
        return {
            "active": True,
            "step": gen.step,
            "total_steps": gen.total_steps,
            "fraction": gen.step / gen.total_steps,  # step is 1..total, never over 1.0
            "eta_seconds": gen.eta_seconds,
        }

    def unload(self) -> dict[str, Any]:
        # Abort an in-flight download so unload/an eviction returns promptly instead
        # of waiting it out (the download runs without _lock and checks this event).
        self._cancel_event.set()
        with self._lock:
            # Abort an in-flight denoise too by setting ITS cancel event, so the step
            # callback stops it. unload does NOT take _generate_lock — it must return
            # promptly; the running generate keeps its own pipe reference, so freeing
            # _state here can't crash it, and its VRAM is reclaimed when it returns
            # (within ~one step thanks to the cancel).
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._unload_locked()
            # Cancel any in-flight load (its worker checks this token before
            # committing) and drop the marker so the next load starts clean.
            self._load_token += 1
            self._loading = None
        return self.status()

    def _unload_locked(self) -> None:
        state = self._state
        if state is None:
            return
        # Restore the process-wide backend flags (TF32 / cudnn.benchmark) this load
        # may have flipped, so the next `off` load is bit-identical again.
        restore_backend_flags(state.backend_flags_before)
        # Restore TORCHINDUCTOR_CACHE_DIR and uninstall the shared eager patches, so a
        # later `off` load runs the bit-identical reference path. Both are idempotent.
        compile_cache.restore(state.compile_cache_ctx)
        # Uninstall the GGUF dequant accelerators (compiled dequant / global weight
        # buffer) this load may have installed, so a later `off` load runs the stock,
        # bit-identical dequant. Idempotent.
        gguf_compile.uninstall_all()
        if state.eager_patched:
            # Lazy import (torch at module level) to keep diffusion.py torch-free to import.
            from .diffusion_eager_patches import uninstall_patches
            from .diffusion_arch_patches import uninstall_arch_patches

            uninstall_patches()
            uninstall_arch_patches()
        # NOTE: we deliberately do NOT call state.pipe.unload_lora_weights() here. unload()
        # sets the cancel event but does not take _generate_lock, so a LoRA-backed denoise
        # can still be running on this same pipe for up to one more callback; mutating its
        # adapter layers now would race that in-flight generation. The whole pipe is dropped
        # just below (self._state = None; del state; clear_gpu_cache()), so the adapter
        # tensors are freed with it -- no explicit unload is needed for memory or for a
        # later load (which builds a fresh pipe).
        # Drop the workflow pipes built around this load's modules so they don't pin the
        # freed pipeline (they only re-wire its components, but holding the wrappers
        # would keep the modules alive past unload).
        self._aux_pipes.clear()
        # Drop any ControlNet models + pipelines so the freed load carries no extra modules.
        self._cn_pipes.clear()
        self._cn_models.clear()
        self._state = None
        del state
        clear_gpu_cache()

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
                "cpu_offload": False,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "text_encoder_quant": None,
                "transformer_quant": None,
                "attention_backend": None,
                "transformer_cache": None,
                "workflows": [],
                "supports_lora": False,
                "supports_controlnet": False,
                "resolved": None,
            }
        from core.inference import diffusion_controlnet, diffusion_lora

        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "model_kind": state.kind,
            "cpu_offload": state.cpu_offload,
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": list(state.speed_optims),
            "text_encoder_quant": state.text_encoder_quant,
            "transformer_quant": state.transformer_quant,
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "resolved": state.resolved,
            # Image-conditioned workflows the loaded family supports, so the UI can gate
            # its tabs. txt2img is always available on the diffusers engine.
            "workflows": _family_workflows(state.family),
            "supports_lora": diffusion_lora.supports_lora(
                engine = "diffusers",
                family = state.family.name,
                model_kind = state.kind,
                transformer_quant = state.transformer_quant,
                compiled = "compiled" in (getattr(state, "speed_optims", ()) or ()),
            ),
            "supports_controlnet": diffusion_controlnet.supports_controlnet(
                engine = "diffusers",
                family = state.family.name,
                has_controlnet_pipeline = bool(
                    getattr(state.family, "controlnet_pipeline_class", None)
                ),
                model_kind = state.kind,
                transformer_quant = state.transformer_quant,
            ),
        }


def _family_workflows(fam: DiffusionFamily) -> list[str]:
    """The workflow ids the diffusers engine can run for ``fam`` (drives UI gating)."""
    # Instruction-editing families have no plain text-to-image mode: their pipeline always
    # takes an input image + instruction, so they expose only the "edit" workflow.
    if getattr(fam, "edit", False):
        return ["edit"]
    workflows = ["txt2img"]
    # Reference families (FLUX.2) keep txt2img and add reference conditioning via their own
    # pipeline's optional image arg (no img2img/inpaint classes needed).
    if getattr(fam, "reference", False):
        workflows.append("reference")
    if getattr(fam, "img2img_pipeline_class", None):
        # Upscale (hires fix) runs on the img2img pipeline, so it is available exactly
        # when img2img is.
        workflows.append("img2img")
        workflows.append("upscale")
    if getattr(fam, "inpaint_pipeline_class", None):
        workflows.append("inpaint")
        # Outpaint (extend) reuses the inpaint pipeline with a padded canvas + border mask,
        # so it needs an inpaint pipeline that preserves the (larger) canvas size.
        if getattr(fam, "inpaint_preserves_size", True):
            workflows.append("outpaint")
    return workflows


def _resolve_base_repo(
    repo_id: str, base_repo: Optional[str], fam: DiffusionFamily, hf_token: Optional[str]
) -> str:
    """The companion diffusers repo: caller's base, else the GGUF repo's own
    ``base_model`` tag, else the family fallback. Shared by both load paths so a
    direct ``load_pipeline`` call resolves the variant base the same way."""
    return resolve_base_repo(fam, (base_repo or "").strip() or _hf_base_model(repo_id, hf_token))


def _hf_base_model(repo_id: str, hf_token: Optional[str]) -> Optional[str]:
    """The diffusers base repo from a GGUF repo's ``base_model`` tag, or None.

    Lets one family entry cover every variant (Turbo/full, schnell/dev, the
    2512 Qwen revision). Skipped for local paths; None on any lookup failure.
    """
    if Path(repo_id).expanduser().exists():
        return None
    try:
        from huggingface_hub import HfApi
        meta = HfApi().model_info(repo_id, token = hf_token).cardData or {}
    except Exception:  # noqa: BLE001 — best-effort; fall back to the family default
        return None
    base = meta.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    return base if isinstance(base, str) and base.strip() else None


def _offload_controlnet_module(cn_model: Any, device: str, logger: Any) -> bool:
    """Stream a ControlNet module through ``device`` via diffusers group offloading.

    Used when the base model was loaded with an offload policy: forcing the ControlNet
    fully resident with ``.to(device)`` would defeat that low-VRAM placement and can OOM.
    Group offloading is applied to this single module (it does not touch the base pipe's
    existing hooks), so it is isolated and reversible. Returns True on success; on any
    failure the caller falls back to a resident placement, so this never blocks a load."""
    try:
        import torch
        from diffusers.hooks import apply_group_offloading

        onload = torch.device(device)
        apply_group_offloading(
            cn_model,
            onload_device = onload,
            offload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = 1,
            use_stream = onload.type == "cuda",
        )
        return True
    except Exception as exc:  # noqa: BLE001 — offload is best-effort; resident is the fallback
        if logger is not None:
            logger.warning("diffusion.controlnet: group offload failed (%s); loading resident", exc)
        return False


def _base_file_downloaded(rfilename: str, *, include_transformer: bool = False) -> bool:
    """True for base-repo files ``from_pretrained`` actually fetches.

    The transformer is supplied by the GGUF, and repo docs (``assets/``, the
    top-level README/PDF/images) are never downloaded — counting them would peg
    the progress estimate above what lands on disk, so the bar would sit short of
    100% for the whole pipeline-load phase instead of advancing to "finalizing".
    ``include_transformer`` admits the ``transformer/`` shards for loads where the
    dense transformer-quant path will fetch them anyway (see
    ``_dense_quant_prefetch_needed``)."""
    if rfilename.startswith("transformer/"):
        return include_transformer
    if "/" not in rfilename:  # top-level: only the pipeline manifest is fetched
        return rfilename == "model_index.json"
    return not rfilename.startswith("assets/")


# Weight file extensions the base repo need NOT supply when the single file is the whole
# pipeline (SDXL): from_single_file(config=base) reads only the base repo's structure
# (config/tokenizer/scheduler) and takes the weights from the single file.
_BASE_WEIGHT_EXTS = (
    ".safetensors",
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
    ".onnx_data",
    ".msgpack",
    ".h5",
    ".pb",
)


def _base_config_file_downloaded(rfilename: str) -> bool:
    """True for base-repo files needed to BUILD a pipeline structure around a whole-pipeline
    single file WITHOUT its weights: config / tokenizer / scheduler JSON, but no weight
    tensors (the single file supplies those). Used for ``single_file_is_pipeline`` families."""
    if not _base_file_downloaded(rfilename):
        return False
    return not rfilename.lower().endswith(_BASE_WEIGHT_EXTS)


def _pipeline_file_downloaded(rfilename: str) -> bool:
    """True for files a full-pipeline ``from_pretrained`` fetches.

    Like ``_base_file_downloaded`` but for the ``pipeline`` kind, where the repo
    supplies its OWN transformer weights, so the ``transformer/`` subfolder is kept.
    Top-level docs (README/PDF/images) and ``assets/`` are skipped, and so are
    artifacts the torch loader never touches -- ONNX / OpenVINO / Flax exports and
    dtype-variant twins (``*.fp16.safetensors``: the loader requests the default
    variant) -- so an official repo that ships many formats (e.g. SDXL Base) does
    not prefetch tens of GB it will not load.
    """
    if "/" not in rfilename:  # top-level: only the pipeline manifest is fetched
        return rfilename == "model_index.json"
    lower = rfilename.lower()
    if lower.startswith(("assets/", "onnx/", "openvino/")):
        return False
    name = lower.rsplit("/", 1)[1]
    if name.startswith(("openvino_", "flax_")):
        return False
    if name.endswith((".onnx", ".onnx_data", ".pb", ".msgpack", ".h5", ".ckpt")):
        return False
    if ".fp16." in name or ".bf16." in name or ".non_ema." in name:
        return False
    return True


def _progress(
    phase: Optional[str],
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    fraction: float = 0.0,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "fraction": fraction,
        "error": error,
    }


_diffusion_backend: Optional[DiffusionBackend] = None


def get_diffusion_backend() -> DiffusionBackend:
    global _diffusion_backend
    if _diffusion_backend is None:
        _diffusion_backend = DiffusionBackend()
    return _diffusion_backend
