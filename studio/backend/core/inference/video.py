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
from .diffusion_cache import apply_step_cache, normalize_transformer_cache
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
    SPEED_OFF,
    apply_speed_optims,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import build_resolved_record
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
    resolved: Optional[dict] = None


@dataclass
class _VideoLoadingState:
    repo_id: str
    base_repo: str
    expected_bytes: Optional[int] = None
    error: Optional[str] = None


def _progress(phase: Optional[str], **extra: Any) -> dict[str, Any]:
    return {"phase": phase, **extra}


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
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        fam = detect_video_family(repo_id, family_override) or (
            detect_video_family(f"{repo_id}/{gguf_filename}")
            if gguf_filename and not family_override
            else None
        )
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
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        # A local checkpoint that cannot exist must fail HERE, before the route evicts
        # a resident chat/image model for a load that dies at resolve time.
        if kind in ("gguf", "single_file"):
            root = Path(repo_id).expanduser()
            if root.is_dir():
                from .diffusion_families import resolve_local_gguf_child
                try:
                    resolve_local_gguf_child(root, gguf_filename or "")
                except Exception as exc:  # noqa: BLE001 -- surface as client input error
                    raise ValueError(str(exc)) from exc
            elif repo_id.startswith(("/", "~", "./", "../")) and not root.is_file():
                raise ValueError(f"Local model path '{repo_id}' does not exist.")
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
                model_kind = model_kind,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            fam = detect_video_family(kwargs["repo_id"], kwargs.get("family_override"))
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
                    root = Path(kwargs["repo_id"]).expanduser()
                    probe = root if root.is_file() else None
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
            base_local = self._predownload_base(
                base, kwargs.get("hf_token"), kind, ltx23 = ltx23
            )
            # The 2.3 assembly pulls per component from the hub id (its snapshot here
            # deliberately lacks the base VAEs), so it only gets the warmed cache; the
            # generic from_pretrained paths get the complete local snapshot.
            kwargs["_base_local_dir"] = None if ltx23 else base_local
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    # Base-repo subfolders an LTX-2.3 assembly reads: the checkpoint (plus the GGUF
    # repo's extras files) supplies the DiT, connectors, both VAEs and the vocoder,
    # so only the 2.0 base's scheduler / text encoder / tokenizer are pulled.
    _LTX23_BASE_PREFIXES = ("scheduler/", "text_encoder/", "tokenizer/")

    @staticmethod
    def _base_download_files(info: Any, kind: str, *, ltx23: bool = False) -> list[tuple[str, int]]:
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
            if not name.endswith((".safetensors", ".json", ".model", ".txt")):
                continue
            if "/" not in name and name.endswith(".safetensors"):
                continue
            if kind != "pipeline" and name.startswith("transformer/"):
                continue
            if name.startswith("text_encoder/diffusion_pytorch_model"):
                continue
            if (
                ltx23
                and "/" in name
                and not name.startswith(VideoBackend._LTX23_BASE_PREFIXES)
            ):
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
        self, base: str, hf_token: Optional[str], kind: str, *, ltx23: bool = False
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
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
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
        self._teardown_state()

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows them, so a resolved fp16
        # promotes to float32 (the same rule as the fp16-incompatible image
        # families). CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32

        # ── memory plan: family-table resident estimate + frames-aware headroom.
        device_memory = snapshot_device_memory(target)
        components = fam.bf16_components_gb
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        if kind == "pipeline":
            model_dense_mib = int(sum(components) * mib_per_gb) if components is not None else None
            companion_mib = None
        else:
            checkpoint_path = self._resolve_checkpoint_path(repo_id, gguf_filename, hf_token)
            size_mib = file_size_mib(str(checkpoint_path))
            model_dense_mib = None
            if kind == "gguf":
                transformer_mib = estimate_gguf_resident_mib(size_mib)
            else:
                transformer_mib = estimate_safetensors_dense_mib(size_mib)
            companion_mib = (
                int((components[1] + components[2]) * mib_per_gb)
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

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
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

        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")

        # ── optimisation layers, in the image backend's order: step cache FIRST
        # (compile keys its fullgraph decision off an active cache: FBCache hooks
        # graph-break, so compiling fullgraph before installing the cache crashes
        # the first cached generation), then attention, the speed profile, and
        # placement/offload last.
        effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
        backend_flags = snapshot_backend_flags()
        cache_engaged = apply_step_cache(
            pipe,
            mode = normalize_transformer_cache(transformer_cache),
            threshold = transformer_cache_threshold,
            logger = logger,
        )
        attention_engaged = apply_attention_backend(
            pipe,
            select_attention_backend(
                target, attention_backend, speed_active = effective_speed != SPEED_OFF
            ),
            logger = logger,
        )
        speed_optims = apply_speed_optims(
            pipe,
            target,
            is_gguf = kind == "gguf",
            family = fam,
            speed_mode = effective_speed,
            cache_active = cache_engaged is not None,
            offload_active = plan.offload_policy != "none",
        )
        # A cancelled/superseded load must not place weights on the GPU the arbiter
        # may already have handed to another backend; recheck right before placement
        # (the commit below still does the final locked check).
        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")
        offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
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
                    "GGUF video loads default to the near-lossless compile profile",
                ),
                "attention_backend": (
                    attention_backend,
                    attention_engaged or "native",
                    "cuDNN fused attention on NVIDIA when a speed profile is active",
                ),
                "transformer_cache": (
                    transformer_cache,
                    cache_engaged or "off",
                    "step cache engages on many-step schedules only",
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
                # Only the optimisations that actually engaged: apply_speed_optims
                # returns every flag with True/False, and iterating the dict raw
                # would report disabled ones as active in /video/status.
                speed_optims = tuple(k for k, v in (speed_optims or {}).items() if v),
                backend_flags = backend_flags,
                attention_backend = attention_engaged,
                transformer_cache = cache_engaged,
                resolved = resolved,
            )
        logger.info(
            "video.loaded: %s (%s, %s, offload=%s, speed=%s)",
            repo_id,
            fam.name,
            kind,
            offload_policy,
            effective_speed,
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
                    state.gguf_filename, state.repo_id, state.base_repo
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
                    fam.cfg_kwarg: guidance,
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                    "generator": generator,
                }
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                # LTX-2 takes frame_rate (it shapes the audio track length); other
                # pipelines fix their own rate and fps only matters at export.
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(out_fps)

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

                def _on_step(p, step_index, timestep, callback_kwargs):
                    if cancel.is_set():
                        p._interrupt = True
                        return callback_kwargs
                    done = step_index + 1
                    elapsed = time.monotonic() - started
                    self._gen.update(
                        step = done,
                        eta_seconds = (elapsed / max(1, done)) * max(0, steps - done),
                    )
                    return callback_kwargs

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step

                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                with torch.inference_mode():
                    output = pipe(**kwargs)
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
                "has_audio": False,
                "defaults": None,
                "resolved": None,
            }
        fam = state.family
        default_steps, default_guidance = default_video_generation_params(
            state.gguf_filename, state.repo_id, state.base_repo
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
