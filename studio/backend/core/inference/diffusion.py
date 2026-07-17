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
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger
from utils.hardware import clear_gpu_cache

from .diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    IDEOGRAM4_FAMILY_NAME,
    LUMINA2_FAMILY_NAME,
    DiffusionFamily,
    default_generation_params,
    detect_family_for_pick,
    excluded_model_reason,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
)
from .diffusion_device import (
    DiffusionDeviceTarget,
    diffusion_device_target_from_torch_device,
    resolve_diffusion_device_target,
)
from .diffusion_ideogram4 import ideogram4_repo_is_fp8, load_ideogram4_pipeline
from .diffusion_krea2 import KREA2_FAMILY_NAME, load_krea2_pipeline
from .diffusion_memory import (
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
    OFFLOAD_NONE,
    apply_memory_plan,
    estimate_gguf_resident_mib,
    estimate_image_runtime_mib,
    estimate_safetensors_dense_mib,
    file_size_mib,
    normalize_memory_mode,
    plan_diffusion_memory,
    plan_fits_total_capacity,
    settled_snapshot_device_memory,
)
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    compile_eligible,
    compiled_shapes_are_static,
    normalize_speed_mode,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_attention import (
    apply_attention_backend,
    normalize_attention_backend,
    select_attention_backend,
    _ensure_attention_backend_installed,
)
from . import diffusion_compile_cache as compile_cache
from . import diffusion_gguf_compile as gguf_compile
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_FBCACHE,
    apply_step_cache,
    effective_denoise_steps,
    effective_request_strength,
    maybe_toggle_step_cache,
    normalize_transformer_cache,
)
from .diffusion_precision import TE_QUANT_AUTO, normalize_te_quant, quantize_text_encoders
from .diffusion_vae_quant import VAE_QUANT_AUTO, normalize_vae_quant, quantize_vae
from .diffusion_prequant import (
    load_prequantized_transformer,
    resolve_prequant_source,
    usable_prequant_source,
)
from .diffusion_auto_policy import (
    build_resolved_record,
    family_bf16_components_gb,
    resolve_dense_quant_candidate,
)
from .diffusion_transformer_quant import (
    TQ_AUTO,
    DEFAULT_MIN_LINEAR_FEATURES,
    dense_transformer_supported,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)

logger = get_logger(__name__)


# A load resolves to one "kind", deciding how the transformer + pipeline is built:
#   "gguf"        -- single-file GGUF transformer dequantised on-device; companions from base repo.
#   "single_file" -- single-file *.safetensors transformer (e.g. fp8), no GGUF dequant; companions from base.
#   "pipeline"    -- full diffusers repo via from_pretrained, re-applying any embedded quant config.
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


def resolve_local_single_file(model_path: str) -> Optional[str]:
    """The sole single-file checkpoint basename in a local ``model_path`` directory that is NOT a
    diffusers pipeline (no ``model_index.json``) and holds exactly one ``.safetensors`` file, else
    None.

    The On-Device scanner advertises a bare single-file safetensors directory as a text-to-image
    model (it matches a known family by name), but the local picker starts it as a ``pipeline``
    with no filename, so a pipeline load 400s on the missing ``model_index.json`` and the
    advertised model is unusable. The images load route uses this to reinterpret such a pick as a
    ``single_file`` load of the sole checkpoint. A real pipeline dir (has ``model_index.json``) or
    an ambiguous one (0 or more than 1 ``.safetensors``, e.g. a sharded pipeline) returns None and
    loads unchanged. A PEFT LoRA adapter folder is also skipped (see below). Never raises."""
    try:
        root = Path(model_path).expanduser()
        if not root.is_dir() or (root / "model_index.json").is_file():
            return None
        # A PEFT LoRA adapter folder (adapter_config.json + adapter_model.safetensors) is not a
        # base checkpoint: from_single_file would fail on the adapter weights AFTER the route
        # evicted the resident GPU model. Skip it so the pick stays a pipeline load and 400s in
        # validation, before the GPU handoff. Also drop a bare adapter_model.safetensors so a
        # config-less adapter export is never reinterpreted as the sole checkpoint.
        if (root / "adapter_config.json").is_file():
            return None
        checkpoints = [
            p.name
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".safetensors"
            and p.stem.lower() != "adapter_model"
        ]
    except OSError:
        return None
    return checkpoints[0] if len(checkpoints) == 1 else None


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
    # Bound the decoded size (this is the single decode path for every image-conditioned
    # workflow). 4096px covers txt2img's 2048 max, upscales, and outpaint canvases; larger 400s.
    max_side = 4096
    try:
        img = Image.open(io.BytesIO(blob))
        # Reject an over-limit image from the header BEFORE img.load() decompresses pixels, so a
        # crafted small-payload/huge-dimension file can't spike memory first.
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


def _clamp_max_side(img: Any, max_side: int) -> Any:
    """Downscale a PIL image so its longest side is <= ``max_side``, preserving aspect ratio
    (high-quality resample); a no-op when it already fits.

    img2img / inpaint take their OUTPUT size from the uploaded image, so without a bound an
    oversized upload (up to the 4096/side decode cap -- 4x the txt2img 2048 ceiling, ~16x the
    area) drives a proportionally larger latent and O(n^2) attention that OOMs the transformer/
    VAE on a normal card, surfacing only as an opaque 500. Clamping the longest side to the same
    2048 ceiling txt2img enforces (and upscale caps to) keeps these workflows bounded."""
    from PIL import Image

    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def _compile_shape_dims(workflow: str, init_pil: Any, width: int, height: int) -> tuple[int, int]:
    """The (width, height) a generation's forward ACTUALLY runs at, for static
    compile-cache shape registration.

    txt2img / reference / controlnet generate at the requested slider size, but the
    image-conditioned workflows (img2img / inpaint / upscale / edit) derive the output
    from the (resized/snapped) input image -- registering the slider values there would
    mark a shape covered that was never compiled, so the truly-used shape never
    re-dirties the bundle and warm restarts keep paying its compile. Mirrors the
    width/height kwarg derivation in generate()."""
    if workflow in ("txt2img", "reference", "controlnet") or init_pil is None:
        return int(width), int(height)
    iw, ih = init_pil.size
    return int(iw), int(ih)


# Official base repos that may load as a full (non-GGUF) pipeline despite not being under
# unsloth/. Safetensors-only, no pickle/remote code; exact-match lowercased (typo-squat safe).
# Extend deliberately; never add pickled weights or remote code. The SDXL refiner is
# intentionally NOT here (img2img-only; this backend loads every sdxl repo as base txt2img).
_TRUSTED_NON_GGUF_REPOS = frozenset(
    {
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        # Vendor safetensors-only bases: LoRA TRAINING bases + the BF16 artifact behind each
        # catalog group. FLUX.1 repos are Hub-gated (need the user's token); Qwen/Z-Image are open.
        "black-forest-labs/flux.1-dev",
        "black-forest-labs/flux.1-schnell",
        "black-forest-labs/flux.1-kontext-dev",
        # Krea's guidance-distilled FLUX.1-dev finetune: same arch/layout as dev (FluxPipeline,
        # CLIP+T5+ae), gated like dev. Detected as the flux.1 family via the "flux.1" token.
        "black-forest-labs/flux.1-krea-dev",
        "tongyi-mai/z-image-turbo",
        "qwen/qwen-image",
        "qwen/qwen-image-2512",
        "qwen/qwen-image-edit-2511",
        # Krea 2: assembled per-component (diffusion_krea2.py). Turbo = inference; Raw = the
        # undistilled base to train LoRAs on (train on Raw, run adapters on Turbo).
        "krea/krea-2-turbo",
        "krea/krea-2-raw",
        # Lumina Image 2.0: standard diffusers layout (Gemma2-2B encoder), safetensors-only,
        # loads through the generic from_pretrained pipeline path.
        "alpha-vllm/lumina-image-2.0",
        # Ideogram 4: no bf16 ships. -fp8 stores the two DiTs as raw float8 (the family base);
        # the two nf4 repos are identical bnb-4bit exports (both listed so either id loads).
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "ideogram-ai/ideogram-4-nf4-diffusers",
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


def _assert_local_base_is_pipeline(base_repo: str) -> None:
    """A companion ``base_repo`` fed to ``from_pretrained(base)`` (or ``config=base``) must be a
    diffusers PIPELINE directory (has ``model_index.json``). ``_is_trusted_diffusion_repo`` accepts
    ANY existing local path, so without this a local base that is not a pipeline dir would pass the
    preflight, let the route evict the resident GPU model, then fail deep in the background load --
    the eviction this validation exists to prevent. A non-existent local base is already rejected
    by the trust check (it is neither an existing path nor an unsloth/*/allowlisted repo); a bare
    remote id is left for the loader to resolve. Shared by the image, video, and training preflights
    so their local-base shape check stays in sync. Never evicts; raises ValueError on a bad local
    base."""
    base = (base_repo or "").strip()
    if not base:
        return
    try:
        root = Path(base).expanduser()
        exists = root.exists()
    except OSError:
        return  # invalid path characters -> a remote id, not a local path
    if not exists:
        return
    if not root.is_dir() or not (root / "model_index.json").is_file():
        raise ValueError(
            f"Local base_repo is not a diffusers pipeline directory (no model_index.json): {base}"
        )


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
    # Resolved memory profile; defaulted so older positional constructions keep working.
    offload_policy: str = OFFLOAD_NONE
    vae_tiling: bool = False
    memory_mode: str = "auto"
    # Resolved load kind ("gguf"|"single_file"|"pipeline"); surfaced so the UI can gate
    # GGUF-only controls (the dense transformer_quant fast path engages only on gguf).
    kind: str = "gguf"
    # The opt-in speed profile.
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    # Process-wide torch backend flags (TF32 / cudnn.benchmark) captured before the speed
    # layer mutated them; restored on unload so a later `off` load isn't contaminated.
    backend_flags_before: Optional[dict] = None
    # Text-encoder quant engaged: "fp8" | "nvfp4" | None.
    text_encoder_quant: Optional[str] = None
    # VAE quant engaged: "fp8" (layerwise storage) | "fp8_dynamic" (torchao conv) | None. When set,
    # the VAE holds fp8 tensor subclasses, so the img2img/inpaint _align_vae_dtype re-cast is
    # skipped (it would corrupt them).
    vae_quant: Optional[str] = None
    # Transformer quant actually engaged on the opt-in dense fast path: "int8" | "fp8"
    # | "nvfp4" | "mxfp8" | None. None means the default GGUF transformer was loaded.
    transformer_quant: Optional[str] = None
    # Attention backend engaged via the diffusers dispatcher, or None for default SDPA.
    attention_backend: Optional[str] = None
    # The caller's ORIGINAL attention request, carried so the deferred-speed engagement
    # re-runs the SAME selection instead of forcing the auto cuDNN upgrade over an explicit pin.
    attention_request: Optional[str] = None
    # Step cache engaged ("fbcache") or None. Opt-in, for many-step models.
    transformer_cache: Optional[str] = None
    # AUTO on a cache-capable transformer: generate() re-checks the step count and toggles
    # FBCache across FBCACHE_MIN_STEPS. An explicit request (off / fbcache) is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Shared eager patches installed for this load (any non-off tier); uninstalled on unload.
    eager_patched: bool = False
    # Deferred speed auto: the load stays eager/bit-identical; generate() engages the `default`
    # compile profile at the 3rd generation this session. Cleared once engaged (or failed).
    speed_deferred: bool = False
    # Successful generations on this load; drives the deferred engagement above.
    generation_count: int = 0
    # Pre-warmed torch.compile cache context when a compiled tier ran, else None.
    compile_cache_ctx: Any = None
    # Token kept so LoRA adapters selected at generate time can be fetched from the Hub.
    hf_token: Optional[str] = None
    # Per-control provenance {control: {value, source, reason}} (source auto/explicit), for status badges.
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
        # _lock serialises the small state mutations; status()/load_progress()/
        # generate_progress() read lock-free so polling never blocks a slow load.
        self._lock = threading.Lock()
        # _generate_lock serialises generations and is the ONLY lock the denoise holds.
        self._generate_lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None
        # Bumped on every begin_load/unload so a superseded/cancelled worker neither
        # commits its pipeline nor stamps progress onto the current load.
        self._load_token = 0
        # Set by unload() to abort an in-flight (lock-free) download so an eviction preempts it.
        self._cancel_event = threading.Event()
        # Cancel Event of the in-flight generation (or None), set under _lock to abort THAT
        # denoise. Per-generation so a cancel can't be lost to a racing generate nor leak.
        self._active_generate_cancel: Optional[threading.Event] = None
        # Written by the callback, read lock-free by generate_progress().
        self._gen: Optional[_GenState] = None
        # Image-conditioned workflow pipes (img2img/inpaint) built via from_pipe around the
        # loaded pipe (shared modules, no extra VRAM). Keyed by class name; cleared on unload.
        self._aux_pipes: dict[str, Any] = {}
        # Loaded ControlNet models (id -> module) and their from_pipe pipelines
        # ((class, cn_id) -> pipe), reusing resident base modules; cleared on unload.
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
        raw = kwargs.get("transformer_quant")
        # Unset defaults to the hardware ladder (mirrors load_pipeline's tri-state).
        if raw is None or str(raw).strip().lower() in ("", "auto"):
            mode = TQ_AUTO
        else:
            mode = normalize_transformer_quant(raw)
        if mode is None:
            return False
        # An explicit Speed="off" load stays GGUF-as-is (dense path never runs); don't widen the prefetch.
        speed = kwargs.get("speed_mode")
        if speed is not None and str(speed).strip().lower() == SPEED_OFF:
            return False
        try:
            # A definite-offload policy forces load_pipeline onto offload, so the dense build
            # never runs and never touches the base transformer/ shards. Widening the prefetch
            # would only waste a multi-GB pull -- and a disk-full here has NO GGUF fallback
            # (unlike an in-load_pipeline dense failure). Mirror those offload gates.
            mm = normalize_memory_mode(kwargs.get("memory_mode"))
            if mm in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM):
                return False
            if mm is None and kwargs.get("cpu_offload"):
                return False
            target = self._resolve_device_target(fam)
            # Only widen when the loader would actually take the dense path: resolve the SAME
            # candidate load_pipeline re-plans against (which also checks the cache has disk room).
            # When disk/scheme/support/a prequant rule it out, don't eagerly pull the shards.
            candidate = resolve_dense_quant_candidate(
                fam = fam,
                target = target,
                requested = mode,
                base_repo = kwargs.get("base_repo"),
                prequant_path = kwargs.get("transformer_prequant_path"),
                force_dense = bool(kwargs.get("loras")),
                logger = None,
            )
            # A prequant candidate loads a small checkpoint, not the dense transformer/ shards,
            # so widening for it defeats the savings and can disk-full the fallback-less begin_load pull.
            return candidate is not None and not candidate.prequant
        except Exception:  # noqa: BLE001 — widening the prefetch is best-effort only
            return False

    def _prefetch_files(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        base_files: list[str],
        hf_token: Optional[str],
    ) -> Optional[str]:
        """Pre-download the GGUF + the given ``base_files`` into the HF cache,
        WITHOUT the lock and honoring ``_cancel_event``, so load_pipeline's
        from_single_file / from_pretrained hit the cache and the heavy download can
        be preempted by an unload/eviction. Raises ``RuntimeError("Cancelled")``.

        Returns the base repo's local snapshot dir when the prefetched set includes
        the pipeline manifest, so from_pretrained can load from disk instead of
        re-sweeping the hub (its own sweep also pulls files the scoped list skips,
        e.g. the 24 GB packaged root singles in each FLUX.1 repo); None otherwise
        (estimate failure, config-only base, local repo) -> hub id as before."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # GGUF transformer (hub repos only; a local path is already on disk).
        if gguf_filename and not Path(repo_id).expanduser().exists():
            hf_hub_download_with_xet_fallback(
                repo_id, gguf_filename, hf_token, cancel_event = self._cancel_event
            )
        # Base repo (VAE / text-encoder / scheduler); list comes from the estimate.
        snapshot_root: Optional[str] = None
        for rfilename in base_files:
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            local = hf_hub_download_with_xet_fallback(
                base, rfilename, hf_token, cancel_event = self._cancel_event
            )
            if rfilename == "model_index.json":
                snapshot_root = str(Path(local).parent)
        return snapshot_root

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
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
            # A deliberately-excluded model gets its stated reason, not the generic
            # unknown-family message (which reads like a detection gap and invites a
            # family_override retry that would fail deeper and less clearly).
            excluded = excluded_model_reason(repo_id)
            if excluded:
                raise ValueError(f"'{repo_id}' cannot be loaded: {excluded}")
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name. (Video models and image models "
                f"whose diffusers transformer has no single-file loader are not supported.)"
            )
        # Families whose single file IS the whole pipeline (SDXL) have no transformer-only
        # GGUF path; reject GGUF here, before the route evicts the current model.
        if kind == "gguf" and fam.single_file_is_pipeline:
            raise ValueError(
                f"'{fam.name}' checkpoints are whole-pipeline single files and have no GGUF "
                f"transformer variant; load the .safetensors pipeline instead of a GGUF."
            )
        # A multi-denoiser family (Ideogram 4's dual DiTs) has no transformer-only path;
        # a single-file/GGUF load would miss its second DiT. Reject here, before eviction.
        if kind in ("gguf", "single_file") and fam.pipeline_only:
            raise ValueError(
                f"'{fam.name}' loads only as a full diffusers pipeline (it assembles "
                f"multiple transformers), not from a single-file or GGUF checkpoint; "
                f"select the pipeline repo."
            )
        # Non-GGUF loads fetch + deserialise weights, so gate to unsloth/ or a local path.
        if kind != "gguf" and not _is_trusted_diffusion_repo(repo_id):
            raise ValueError(
                f"Non-GGUF diffusion loads are restricted to unsloth/* repos (or a local "
                f"path); got '{repo_id}'. Pass a gguf_filename to load a GGUF instead."
            )
        # A companion base repo also loads via from_pretrained, so it must clear the same
        # trust bar (else a GGUF pick could smuggle in an arbitrary remote base). Gate here.
        if base_repo and base_repo.strip() and not _is_trusted_diffusion_repo(base_repo):
            raise ValueError(
                f"base_repo is restricted to unsloth/* repos (or a local path); got "
                f"'{base_repo}'."
            )
        # A local base_repo loads as a full pipeline (needs model_index.json); reject a
        # non-pipeline local base here, before eviction.
        _assert_local_base_is_pipeline(base_repo)
        # Reject a bad LOCAL pick now so the route never evicts chat for an unloadable request.
        # A path-shaped repo_id is meant to be on disk; a bare "org/name" is a remote HF repo.
        local_root = Path(repo_id).expanduser()
        # Path-shaped: "."/".." prefix, a backslash (never in "org/name"), or an absolute path.
        path_shaped = (
            repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or local_root.is_absolute()
        )
        if kind in ("gguf", "single_file"):
            if not gguf_filename:
                raise ValueError(f"a single-file checkpoint name is required for a '{kind}' load.")
            # Fail a kind/extension mismatch here, before the handoff: gguf needs .gguf,
            # single_file must not be handed a .gguf.
            is_gguf_name = gguf_filename.lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            # A single-file load must name an actual .safetensors checkpoint (else it evicts
            # chat and only fails in the background from_single_file).
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
                # A remote "*-GGUF" id is a GGUF repo, not a pipeline; loading it as a pipeline
                # would evict chat then fail on the missing model_index.json. Reject here.
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
        vae_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        # A blank token must mean "anonymous", not an empty credential the Hub 401s.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        # base_repo is already gated at the route's pre-eviction validate; this re-validation
        # is a cheap-fail guard for the resolved repo/family and does not re-gate base_repo.
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
            # Best-effort download preemption; the token is the real commit guard.
            self._cancel_event.clear()
            # Seed with the family fallback; the worker resolves the real base and updates this.
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
                vae_quant = vae_quant,
                transformer_quant = transformer_quant,
                transformer_quant_fast_accum = transformer_quant_fast_accum,
                transformer_prequant_path = transformer_prequant_path,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                model_kind = model_kind,
                loras = loras,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            # Resolve the base repo and estimate sizes on this thread (both network calls) so
            # begin_load returns instantly. Only writer of _loading's fields here.
            fam = detect_family_for_pick(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            if kind == "pipeline":
                # The full pipeline IS the repo, so the base repo is the repo itself.
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
                # The dense-quant path otherwise pulls the base transformer/ shards inside the
                # locked finalize (unpreemptable); when it can run, pull them in the prefetch here.
                include_transformer = kind == "gguf"
                and self._dense_quant_prefetch_needed(fam, kwargs),
            )
            with self._lock:
                # Stamp progress only if this load is still current (a superseder has its own token).
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.expected_bytes = expected
            # Download outside the lock so unload/an eviction can preempt the pull.
            kwargs["_base_local_dir"] = self._prefetch_files(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                base_files,
                kwargs.get("hf_token"),
            )
            self.load_pipeline(**kwargs)
            with self._lock:
                # Only clear the marker if this load is still current (a superseder has its own token).
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 — surfaced to the client via load_progress
            # A cancelled/superseded load raised below; don't log/stamp it onto the current load.
            if self._load_token != token:
                return
            logger.error("diffusion.load_failed: %s", exc)
            # Free the debris of a failed construction (uncommitted _state, so nothing else
            # reclaims the VRAM). Guarded so a sticky CUDA error can't skip stamping the real error.
            try:
                clear_gpu_cache()
            except Exception:  # noqa: BLE001
                pass
            # Redact native paths: this error is surfaced verbatim and Studio can be shared.
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

        # Sum checkpoint + companion base cache; for a full-pipeline load base IS the repo,
        # so count it once (else the bar double-counts).
        downloaded = self._cache_bytes(loading.repo_id)
        if loading.base_repo and loading.base_repo != loading.repo_id:
            downloaded += self._cache_bytes(loading.base_repo)
        expected = loading.expected_bytes
        # Downloads done, still finalizing. The cache scan can exceed the estimate, so clamp to 100%.
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
                # diffusers prefers safetensors: drop a .bin whose dir also has a picked .safetensors.
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
            # Skip the Hub size lookup for a LOCAL gguf path: model_info would raise on a
            # filesystem path and (caught below) skip the base lookup, forcing a synchronous companion pull.
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                total += sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
            # A whole-pipeline single file (SDXL) needs only the base's config/tokenizer, not its weights.
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
    def _hub_cache_repo_dir(repo_id: str) -> Path:
        """Local HF hub cache dir for ``repo_id``."""
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"

    @staticmethod
    def _cache_bytes(repo_id: str) -> int:
        blobs = DiffusionBackend._hub_cache_repo_dir(repo_id) / "blobs"
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
    def _max_over_cached_revs(base: str, fn: Callable[[Path], int]) -> int:
        """Apply ``fn`` to a LOCAL diffusers dir, or to the fullest cached hub snapshot
        revision (the active one is the fullest), returning that count. 0 when nothing is
        cached. Multiple revisions may be cached, so take the max."""
        local = Path(base).expanduser()
        if local.is_dir():
            return fn(local)
        snapshots = DiffusionBackend._hub_cache_repo_dir(base) / "snapshots"
        if not snapshots.is_dir():
            return 0
        return max((fn(rev) for rev in snapshots.iterdir() if rev.is_dir()), default = 0)

    @staticmethod
    def _companion_cache_bytes(base: str) -> int:
        """Resident companion (VAE + text-encoder) size for the memory plan.

        Excludes ``transformer/`` (supplied by the GGUF/single file, not resident here) --
        otherwise the dense-quant prefetch's cached transformer shards would inflate this
        and wrongly force offload. Walks the snapshot dir, not the flat ``blobs/`` cache,
        since only the snapshot preserves the subfolder split needed to exclude it."""
        return DiffusionBackend._max_over_cached_revs(
            base, lambda d: DiffusionBackend._local_dir_weight_bytes(d, exclude_transformer = True)
        )

    @staticmethod
    def _safetensors_param_count(path: Path) -> int:
        """Total tensor elements in a safetensors file, read from its JSON header without
        touching the tensor data. 0 on any read/parse failure."""
        try:
            with open(path, "rb") as fh:
                header_len = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(header_len))
            total = 0
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                numel = 1
                for dim in meta.get("shape", []):
                    numel *= dim
                total += numel
            return total
        except Exception:  # noqa: BLE001 — corrupt/crafted shard degrades to 0, never crashes the load
            return 0

    @staticmethod
    def _dense_transformer_resident_bytes(base: str) -> int:
        """Resident bf16 size of the base repo's dense ``transformer/`` for the dense-quant
        preflight. That fast path loads the transformer at the compute dtype (bf16, 2
        bytes/param) before quantizing, so budget num_params * 2 -- NOT the on-disk bytes,
        which for an F32 base (e.g. Z-Image) are ~2x the resident size. Read from the
        safetensors shard headers. Returns 0 when no ``transformer/*.safetensors`` shards
        are present (an uncached base, or a .bin-only transformer); the caller then gates
        the fast path on the plain plan."""

        def _params(d: Path) -> int:
            tdir = d / "transformer"
            if not tdir.is_dir():
                return 0
            return sum(
                DiffusionBackend._safetensors_param_count(s) for s in tdir.glob("*.safetensors")
            )

        return DiffusionBackend._max_over_cached_revs(base, _params) * 2  # bf16: 2 bytes/param

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
        vae_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        # LoRA adapters to BAKE into a torchao int8/fp8 build (attached on the dense
        # transformer before quantize_ + compile). Ignored by every other load kind: bf16 /
        # bnb loads take adapters at generation time, GGUF-as-is has no dense transformer.
        loras: Optional[list[tuple[str, float]]] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        # A blank/whitespace token must degrade to anonymous, not be passed as a credential
        # the Hub client can error on. Normalize once for every branch below.
        hf_token = hf_token.strip() if isinstance(hf_token, str) else hf_token
        hf_token = hf_token or None

        # Validate first (cheap, no torch/diffusers) so a bad family fails even in a no-diffusers
        # runtime. Re-sanitize the token (direct callers bypass begin_load).
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        # base_repo is gated at the route before eviction; this re-validation cheap-fails the
        # resolved repo/family and does not re-gate an already-validated base_repo.
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
        )
        kind = resolve_model_kind(gguf_filename, model_kind)
        # Validate every mode string that can raise NOW, before this load evicts the previous
        # pipeline. Validate-only for transformer_quant (keep the unset/auto vs explicit-off tri-state).
        normalize_transformer_quant(transformer_quant)
        normalize_speed_mode(speed_mode)
        normalize_attention_backend(attention_backend)
        normalize_transformer_cache(transformer_cache)
        normalize_te_quant(text_encoder_quant)
        normalize_vae_quant(vae_quant)
        # An explicit Speed="off" (bit-exact) load pins the companions dense too: promoting an
        # UNSET or "auto" TE/VAE to auto-quant would silently fp8/int8 them and break the request.
        # auto is backend-owned, so both UNSET and "auto" go dense under off; only an explicit
        # CONCRETE scheme still forces quant under off.
        speed_off = speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
        # text_encoder_quant tri-state (mirrors transformer_quant): UNSET ("" / None) or "auto" ->
        # auto (best accurate TE scheme, else dense); "none"/"off" -> dense; a concrete scheme forces
        # it. Default is auto (dense under off).
        if text_encoder_quant is None or str(text_encoder_quant).strip().lower() in ("", "auto"):
            text_encoder_quant = "off" if speed_off else TE_QUANT_AUTO
        # vae_quant tri-state, same contract: auto -> layerwise fp8 where the family qualifies, else
        # dense (fp8_dynamic is explicit-only); none/off -> dense. Also dense under Speed="off".
        if vae_quant is None or str(vae_quant).strip().lower() in ("", "auto"):
            vae_quant = "off" if speed_off else VAE_QUANT_AUTO
        # For a full pipeline the repo itself supplies every component, so it is its
        # own base; the single-file kinds resolve the companion base diffusers repo.
        base = (
            repo_id if kind == "pipeline" else _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        )
        target = self._resolve_device_target(fam)
        device, dtype = target.device, target.dtype

        import diffusers

        # Pre-install the optional attention kernel BEFORE the load locks: the wheel-only pip
        # install can run up to 600s, and doing it under the lock would block unload/cancel that
        # whole window. Only an explicit backend pulls a package (auto uses cuDNN/native from
        # torch). Best-effort; the authoritative resolve + set under the lock is then a no-op.
        try:
            preinstall_backend = select_attention_backend(
                target, attention_backend, speed_active = True
            )
            if preinstall_backend is not None:
                _ensure_attention_backend_installed(preinstall_backend, logger)
        except Exception:  # noqa: BLE001 — the locked path re-resolves and validates
            pass

        # Signal an in-flight denoise to abort, then take _generate_lock to WAIT for it to exit
        # before allocating the replacement (a load claims VRAM, so it must not overlap a live pipe).
        with self._lock:
            # Bail BEFORE signalling a cancel if this load was already superseded, else a stale
            # worker would abort an unrelated live generation from the CURRENT model.
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Diffusion load was cancelled.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        with self._generate_lock:
            with self._lock:
                # Re-check: a newer load/unload may have superseded this one while we waited.
                if _load_token is not None and _load_token != self._load_token:
                    raise RuntimeError("Diffusion load was cancelled.")

                # Free the old pipeline before allocating the new one (never two in VRAM).
                self._unload_locked()

                # Single-file kinds resolve a checkpoint path; the pipeline kind has none.
                single_file_path = (
                    self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
                    if kind in ("gguf", "single_file")
                    else None
                )
                transformer_cls = getattr(diffusers, fam.transformer_class)
                pipeline_cls = getattr(diffusers, fam.pipeline_class)

                # Decide placement up front (weights still on CPU, so free VRAM is the budget).
                # Budgets the GGUF file; the dense-quant fast path is preflighted separately below.
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

                # Dtype tri-state: unset/"auto" -> hardware ladder picks a quantised build (int8
                # min, fp8 on datacenter silicon) over GGUF-as-is; "none"/"off" pins GGUF-as-is;
                # an explicit scheme pins it. An overwritten "auto" still records source=auto.
                if transformer_quant is None or str(transformer_quant).strip().lower() in (
                    "",
                    "auto",
                ):
                    # An explicit Speed="off" load must stay GGUF-as-is: auto-quant would engage
                    # int8/fp8 + compile and break the bit-exact request. "off" -> None (GGUF-as-is).
                    speed_off = (
                        speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
                    )
                    transformer_quant = "off" if speed_off else TQ_AUTO

                # Default-on fast path: load the DENSE bf16 transformer and torchao-quantise it
                # (int8/fp8/fp4 tensor cores), which beats GGUF's per-matmul dequant on speed AND
                # quality at a higher-memory dense load. CUDA + bf16 + resident fit; ANY failure
                # falls back to the GGUF build. GGUF kind only (it has the dense bf16 to materialise).
                pipe = None
                transformer_quant_engaged = None
                quant_plan = None
                # The GGUF-size `plan` can mis-budget the fast path two ways, so preflight the real
                # footprint BEFORE eviction; both branches need the base repo + a resolved scheme.
                dense_declined = False
                # False when the memory plan only holds a PREQUANT-sized build: if the prequant
                # load then fails, the loader must raise to GGUF instead of materialising the
                # dense bf16 transformer the plan never budgeted for.
                dense_fallback_allowed = True
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                ):
                    if plan.offload_policy != OFFLOAD_NONE:
                        # The GGUF plan picked offload, but the quantised artifact is smaller
                        # (int8/fp8 ~half bf16; a prequant never materialises dense). Re-plan
                        # against the candidate's estimate -- a resident quant build beats an offloaded GGUF.
                        candidate = resolve_dense_quant_candidate(
                            fam = fam,
                            target = target,
                            requested = transformer_quant,
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                            # A LoRA bake skips the prequant shortcut, so size the candidate
                            # for the dense build it will actually run.
                            force_dense = bool(loras),
                            logger = logger,
                        )
                        if candidate is not None:
                            def _replan_candidate():
                                return self._plan_memory(
                                    target,
                                    single_file_path,
                                    base,
                                    fam,
                                    memory_mode,
                                    cpu_offload,
                                    kind = kind,
                                    repo_id = repo_id,
                                    transformer_resident_override_mib = (
                                        candidate.transient_transformer_mib
                                    ),
                                    # Pass the auto-policy's companion estimate so the prefetched base
                                    # transformer/ shards in the cache aren't double-counted.
                                    companion_override_mib = candidate.companions_mib,
                                )

                            replanned = _replan_candidate()
                            if (
                                replanned.offload_policy != OFFLOAD_NONE
                                # Explicit balanced/low_vram picks offload BY MODE; a fresh
                                # snapshot cannot change that, so don't waste a retry.
                                and normalize_memory_mode(memory_mode)
                                not in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM)
                                and plan_fits_total_capacity(replanned)
                            ):
                                # The candidate fits TOTAL device capacity with the standard
                                # reserve + resident margin, yet the instantaneous free reading
                                # said no: a transient foreign allocation (measured on B200:
                                # ~100 GB held for under a minute on an idle card) must not
                                # force the GGUF fallback. Re-snapshot (settled) and replan
                                # once before declining.
                                replanned = _replan_candidate()
                            if replanned.offload_policy != OFFLOAD_NONE:
                                logger.info(
                                    "diffusion.transformer_quant_declined: required=%s MiB "
                                    "budget=%s MiB free=%s MiB policy=%s (%s)",
                                    replanned.estimates.get("resident_required_mib"),
                                    replanned.estimates.get("safe_device_budget_mib"),
                                    getattr(replanned.device_memory, "free_mib", None),
                                    replanned.offload_policy,
                                    "; ".join(replanned.reasons),
                                )
                            if replanned.offload_policy == OFFLOAD_NONE:
                                quant_plan = replanned
                                # The GGUF plan already declined resident; a prequant-sized
                                # replan says nothing about the (larger) dense transformer.
                                if candidate.prequant:
                                    dense_fallback_allowed = False
                    else:
                        # The GGUF fits resident, but this path first materialises the base's dense
                        # bf16 transformer (bigger), so re-check the fit against THAT -- a card that
                        # fits the GGUF but not the dense must skip the fast path up front, not OOM
                        # after eviction. A prequant loads a small file (no dense), so skip the
                        # re-check there. _dense_transformer_resident_bytes returns 0 if shards are absent.
                        scheme = select_transformer_quant_scheme(
                            target,
                            transformer_quant,  # normalized above
                            family = getattr(fam, "name", None),
                        )
                        # usable_prequant_source (not resolve_): a missing/non-allowlisted local
                        # path must NOT count as prequant here, or it skips the dense-fit re-check
                        # and OOMs materialising the dense transformer after eviction.
                        prequant = (
                            # A LoRA bake skips the prequant shortcut (adapters attach on the
                            # dense transformer), so the dense-fit re-check must gate the fast
                            # path exactly as if no prequant source existed.
                            None
                            if loras
                            else usable_prequant_source(
                                fam,
                                scheme,
                                path_override = transformer_prequant_path,
                                base_repo = base,
                            )
                            if scheme is not None
                            else None
                        )
                        dense_mib = int(
                            self._dense_transformer_resident_bytes(base) // (1024 * 1024)
                        )
                        if dense_mib > 0:
                            dense_plan = self._plan_memory(
                                target,
                                single_file_path,
                                base,
                                fam,
                                memory_mode,
                                cpu_offload,
                                kind = kind,
                                repo_id = repo_id,
                                transformer_resident_override_mib = dense_mib,
                            )
                            if dense_plan.offload_policy != OFFLOAD_NONE:
                                dense_fallback_allowed = False
                                # Without a prequant source the dense build is the ONLY path,
                                # so a dense misfit skips the fast path entirely (as before); with
                                # one, the small prequant load proceeds and only the dense
                                # fallback is forbidden.
                                if prequant is None:
                                    dense_declined = True
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and not dense_declined
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
                            base_local_dir = _base_local_dir,
                            prequant_path = transformer_prequant_path,
                            allow_dense_fallback = dense_fallback_allowed,
                            lora_specs = loras,
                        )
                    except Exception as exc:  # noqa: BLE001 — fall back to the GGUF build
                        logger.warning(
                            "diffusion.transformer_quant_fallback: %s (loading GGUF)", exc
                        )
                        pipe = None
                        transformer_quant_engaged = None
                        # Drop the exception BEFORE clearing the cache: its traceback keeps the
                        # partially-built dense transformer/pipe alive, blocking VRAM reclaim.
                        del exc
                        # Guarded: a sticky CUDA error can raise; the fallback must reach the GGUF build.
                        try:
                            clear_gpu_cache()
                        except Exception:  # noqa: BLE001
                            pass
                if transformer_quant_engaged is not None and quant_plan is not None:
                    # The engaged dense build uses the re-planned placement; the GGUF-size plan stays for fallback.
                    plan = quant_plan

                if pipe is None:
                    if kind == "pipeline":
                        # Full diffusers repo: from_pretrained pulls every component and re-applies
                        # any embedded quantization_config (e.g. bnb-4bit).
                        if fam.name == KREA2_FAMILY_NAME:
                            # krea ships transformers-5.x configs the 4.x line can't parse; assemble
                            # per-component (see diffusion_krea2.py).
                            pipe = load_krea2_pipeline(repo_id, dtype, hf_token = hf_token)
                        elif fam.name == IDEOGRAM4_FAMILY_NAME:
                            # ideogram ships the same transformers-5.x Qwen stack as krea; assemble
                            # per-component too (see diffusion_ideogram4.py).
                            pipe = load_ideogram4_pipeline(repo_id, dtype, hf_token = hf_token)
                        else:
                            pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
                            if hf_token:
                                pipe_kwargs["token"] = hf_token
                            # The prefetched snapshot dir keeps from_pretrained off the hub (its
                            # sweep re-pulls files the scoped prefetch skipped: 24 GB per FLUX.1).
                            pipe = pipeline_cls.from_pretrained(
                                _base_local_dir or repo_id, **pipe_kwargs
                            )
                    elif kind == "single_file" and fam.single_file_is_pipeline:
                        # A single-file SDXL-style checkpoint is the WHOLE pipeline, so load it
                        # through the pipeline class; ``config`` points at the base repo so diffusers
                        # builds the correct structure around the single-file weights.
                        sf_pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "config": base}
                        if hf_token:
                            sf_pipe_kwargs["token"] = hf_token
                        pipe = pipeline_cls.from_single_file(single_file_path, **sf_pipe_kwargs)
                    else:
                        # Transformer-only single file; VAE/text-encoder/scheduler come from the base repo.
                        sf_kwargs: dict[str, Any] = {
                            "torch_dtype": dtype,
                            "config": base,
                            "subfolder": "transformer",
                            # Config is fetched from the (possibly gated) base before auth.
                            "token": hf_token,
                        }
                        if kind == "gguf":
                            # Dequantise the GGUF transformer on-device at the compute dtype.
                            sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                                compute_dtype = dtype
                            )
                        # A safetensors single-file (fp8) carries its own dtype: no GGUF dequant config.
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
                            pipe = pipeline_cls.from_pretrained(
                                _base_local_dir or base, **pipe_kwargs
                            )

                # Effective speed: GGUF defaults to near-lossless `default` (compile ~2.2x, below
                # the quant noise floor); dense stays bit-identical `off`. Explicit is honored.
                effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
                # A torchao-quantized dense transformer must be compiled (eager is ~30x slower and
                # loses to GGUF), so force at least `default` when quant engaged.
                if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
                    logger.info(
                        "diffusion.transformer_quant: forcing speed_mode=default "
                        "(quantized transformer must be compiled; eager is ~30x slower)"
                    )
                    effective_speed = SPEED_DEFAULT
                # Deferred speed auto for dense models: stay eager (a one-off image shouldn't pay
                # the 25-60s compile), but generate() engages `default` on the 3rd image, where
                # repeated use amortises it. Only when speed was unset, nothing forced compile, and this device can compile.
                speed_deferred = (
                    speed_mode is None
                    and effective_speed == SPEED_OFF
                    and transformer_quant_engaged is None
                    and compile_eligible(target, is_gguf = False, family = fam)
                )
                # Speed optims run BEFORE placement (channels_last/compile precede offload).
                # Snapshot the global backend flags (TF32/cudnn.benchmark) first for unload restore.
                backend_flags_before = snapshot_backend_flags()
                # Pick the attention kernel BEFORE compile. auto upgrades to cuDNN fused attention
                # on NVIDIA when a speed profile is active (~1.18x); explicit is honored.
                attention_engaged = apply_attention_backend(
                    pipe,
                    select_attention_backend(
                        target, attention_backend, speed_active = effective_speed != SPEED_OFF
                    ),
                    logger = logger,
                )
                # Step caching (First-Block-Cache), also before compile: reuses the transformer
                # tail across steps (~1.4x on Flux at LPIPS ~0.08); when engaged, compile drops
                # fullgraph (graph break). Tri-state: unset/"auto" -> step-count policy decides
                # (engage when the DEFAULT schedule reaches FBCACHE_MIN_STEPS); "off"/"fbcache" pinned.
                cache_request = normalize_transformer_cache(transformer_cache)
                cache_auto = transformer_cache is None or cache_request == TC_AUTO
                cache_quant_active = transformer_quant_engaged is not None or bool(gguf_filename)
                default_steps: Optional[int] = None
                if cache_auto:
                    default_steps, _ = default_generation_params(
                        gguf_filename, repo_id, base, fam.name
                    )
                    cache_request = TC_FBCACHE if default_steps >= FBCACHE_MIN_STEPS else None
                cache_engaged = apply_step_cache(
                    pipe,
                    mode = cache_request,
                    threshold = transformer_cache_threshold,
                    # GGUF transformers are quantized too, so the cache needs the higher threshold.
                    quant_active = cache_quant_active,
                    logger = logger,
                )
                # An auto decision can flip at generation time, but only on a cache-capable
                # transformer (a non-CacheMixin one keeps fullgraph).
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
                # Everything from here to the _LoadState commit mutates PROCESS-WIDE state (class
                # patches, TORCHINDUCTOR_CACHE_DIR, backend flags). _unload_locked reverses it via
                # _state, so a pre-commit failure would leak it; the try/finally below restores on failure.
                # gguf_transformer: the GGUF-specific compiled dequant applies only when the GGUF
                # was actually loaded. On the dense fast path gguf_filename is still set (fallback)
                # but pipe.transformer is dense (needs REGIONAL block compile), so treat it non-GGUF.
                gguf_transformer = kind == "gguf" and transformer_quant_engaged is None

                eager_patched = False
                compile_ctx = None
                state_committed = False
                # Lazy import (these modules import torch) keeps diffusion.py torch-free to import.
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
                        # Per-arch compile-safe fusions (qwen _modulate / z-image residual, etc.);
                        # neutral under compile, tracked by the same eager_patched flag.
                        install_arch_patches()
                        eager_patched = True
                    else:
                        uninstall_patches()
                        uninstall_arch_patches()

                    # Pre-warmed torch.compile cache: point inductor at a per-fingerprint dir and
                    # load a matching bundle before the first compiled forward, so the 25-58s
                    # compile is paid once and reused. A miss is silent -> local compile.
                    if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and compile_eligible(
                        target, is_gguf = gguf_transformer, family = fam
                    ):
                        compile_ctx = compile_cache.begin(
                            family = fam.name,
                            # U-Net families (SDXL) carry the denoiser as pipe.unet.
                            transformer = getattr(pipe, "transformer", None)
                            or getattr(pipe, "unet", None),
                            dtype = getattr(target, "dtype", None),
                            quant = transformer_quant_engaged,
                            attention_backend = attention_engaged,
                            compile_kwargs = {
                                # Mirrors apply_speed_optims' fullgraph decision: an active or
                                # still-toggleable step cache OR a planned offload graph-breaks,
                                # so the cached bundle must key on the same fullgraph setting.
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
                        # Offload installs compiler-disabled onload hooks, so compile drops fullgraph.
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )
                    if transformer_quant_engaged is not None and not speed_applied.get("compiled"):
                        # Compile couldn't engage: the quantized transformer runs eager, far slower
                        # than the GGUF it replaced. Surface it loudly.
                        logger.warning(
                            "diffusion.transformer_quant: %s engaged but the transformer is NOT "
                            "compiled; eager torchao quant is ~30x slower than GGUF here",
                            transformer_quant_engaged,
                        )
                    # Quantise the dense companion text encoder(s) (opt-in), before placement so
                    # offload moves the smaller weights. Family drives int8's keep-bf16 schedule.
                    te_quant = quantize_text_encoders(
                        pipe,
                        target,
                        mode = text_encoder_quant,
                        family = fam.name,
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )
                    # Quantise the dense VAE (opt-in fp8 layerwise / fp8_dynamic torchao conv),
                    # before placement so offload moves the smaller weights. Image families never
                    # force-fp32 their VAE; auto skips torchao under offload.
                    vae_quant_engaged = quantize_vae(
                        pipe,
                        target,
                        mode = vae_quant,
                        family = fam.name,
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        force_fp32 = False,
                        logger = logger,
                    )

                    # Apply the planned placement; apply_memory_plan returns the (policy, tiling)
                    # ACTUALLY engaged so status stays honest. Idempotent for the `none` policy.
                    effective_policy, effective_tiling = apply_memory_plan(
                        pipe, plan, device = device, logger = logger
                    )

                    # Per-control provenance for status. cpu_offload=False is the unset default,
                    # so only True is an explicit request.
                    resolved = build_resolved_record(
                        {
                            "speed_mode": (
                                speed_mode,
                                "deferred" if speed_deferred else effective_speed,
                                "quantized transformer requires compile"
                                if transformer_quant_engaged is not None
                                and normalize_speed_mode(speed_mode) in (None, SPEED_OFF)
                                else "auto: exact eager for the first two images; "
                                "the compile profile engages on the 3rd"
                                if speed_deferred
                                else "per-kind default"
                                if speed_mode is None
                                else "requested",
                            ),
                            "transformer_quant": (
                                transformer_quant,
                                transformer_quant_engaged or "off",
                                # The None reason matches the load kind (GGUF loaded vs dense kept).
                                (
                                    "not engaged (GGUF transformer loaded)"
                                    if kind == "gguf"
                                    else "dense transformer kept unquantized"
                                )
                                if transformer_quant_engaged is None
                                else "re-planned resident for the quantised artifact"
                                if quant_plan is not None
                                else "engaged on the dense fast path",
                            ),
                            "text_encoder_quant": (
                                text_encoder_quant,
                                te_quant or "off",
                                "dense (no accurate scheme for this GPU / disabled)"
                                if te_quant is None
                                else "auto-selected for this GPU + family"
                                if text_encoder_quant == TE_QUANT_AUTO
                                else "requested",
                            ),
                            "vae_quant": (
                                vae_quant,
                                vae_quant_engaged or "off",
                                "dense (no accurate scheme for this GPU / disabled)"
                                if vae_quant_engaged is None
                                else "auto-selected for this GPU + family"
                                if vae_quant == VAE_QUANT_AUTO
                                else "requested",
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
                                "everything fits on the GPU, no offload needed"
                                if effective_policy == OFFLOAD_NONE
                                else "planned from measured free VRAM vs estimated footprint",
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
                        vae_quant = vae_quant_engaged,
                        transformer_quant = transformer_quant_engaged,
                        attention_backend = attention_engaged,
                        attention_request = attention_backend,
                        transformer_cache = cache_engaged,
                        cache_auto = cache_may_toggle,
                        cache_quant_active = cache_quant_active,
                        cache_threshold = transformer_cache_threshold,
                        eager_patched = eager_patched,
                        speed_deferred = speed_deferred,
                        compile_cache_ctx = compile_ctx,
                        hf_token = hf_token,
                        resolved = resolved,
                    )
                    state_committed = True
                finally:
                    # Pre-commit failure: roll back the process-wide mutations (symmetric with _unload_locked).
                    if not state_committed:
                        restore_backend_flags(backend_flags_before)
                        compile_cache.restore(compile_ctx)
                        gguf_compile.uninstall_all()  # idempotent
                        if eager_patched:
                            uninstall_patches()
                            uninstall_arch_patches()
                        # Free the half-built pipe's VRAM (uncommitted _state -> nothing else reclaims it).
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
        base_local_dir: Optional[str] = None,
        allow_dense_fallback: bool = True,
        lora_specs: Optional[list[tuple[str, float]]] = None,
    ) -> tuple[Any, str]:
        """Build the opt-in fast pipeline and return ``(pipe, engaged_scheme)``.

        Two ways to get the quantized transformer, in order:

        1. Pre-quantized: if a checkpoint is configured for the chosen scheme (an explicit
           ``prequant_path`` or the family's hosted repo), load the already-quantized
           weights onto the meta device and assign them in -- the dense bf16 never lands on
           the GPU, so the load peak is ~half and the download is smaller.
        2. Dense + quantise (fallback): load the DENSE bf16 transformer from the base repo,
           place it on the device, and torchao-quantise it in place.

        ``lora_specs`` bakes LoRA adapters into the build: they attach on the DENSE
        transformer (peft's post-quant torchao dispatch needs quantizer metadata a manual
        quantize_ never has), then quantize_ converts only the frozen base linears (the
        ``lora_`` side path is excluded by name), then the loader compiles. That forces the
        dense path -- the prequant shortcut is skipped -- so a baked-LoRA load pays the dense
        peak. Verified on the Studio stack: scale 0 reproduces the quantized base exactly.

        Raises if the scheme is unsupported or quantisation fails, so ``load_pipeline``
        catches it and falls back to the GGUF build. Quantisation runs ON the device and
        BEFORE the loader compiles the repeated block, so the order stays quantize ->
        compile -> placement."""
        # 1. Pre-quantized checkpoint, when one is configured for the resolved scheme.
        scheme = select_transformer_quant_scheme(target, mode, family = getattr(fam, "name", None))
        if scheme is None:
            # Bail BEFORE the multi-GB dense download: an unsupported scheme (fp8 on Ampere,
            # nvfp4 off Blackwell) would otherwise materialise the transformer only to fail at
            # quantize, after eviction. load_pipeline catches this and builds the GGUF pipeline.
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None and not lora_specs:
            # A LoRA bake needs the DENSE transformer (adapters attach before quantize_), so
            # the prequant shortcut is skipped when adapters were requested.
            source = resolve_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base
            )
            if source is not None:
                transformer = load_prequantized_transformer(
                    transformer_cls,
                    base,
                    source,
                    device = device,
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint with a different Linear filter so prequant matches runtime-quant.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Only enforced when the caller forces fp8 fast-accum; a checkpoint that baked
                    # the other choice falls to the dense path instead of using the baked kernels.
                    fast_accum = fast_accum,
                    logger = logger,
                )
                if transformer is not None:
                    pipe = self._assemble_pipe(
                        pipeline_cls, base, transformer, dtype, hf_token, device, base_local_dir,
                        fam = fam,
                    )
                    return pipe, scheme

        # 2. Fallback: materialise the dense bf16 transformer and quantise it on-device.
        if not allow_dense_fallback:
            # The memory plan only budgeted the prequant-sized build; materialising the dense
            # bf16 transformer here would exceed it after eviction. Raise to the GGUF build.
            raise RuntimeError(
                "prequant checkpoint unavailable and the dense transformer does not fit resident"
            )
        transformer = transformer_cls.from_pretrained(
            base, subfolder = "transformer", torch_dtype = dtype, token = hf_token
        )
        pipe = self._assemble_pipe(
            pipeline_cls, base, transformer, dtype, hf_token, device, base_local_dir, fam = fam
        )
        if lora_specs:
            # Bake the adapters BEFORE quantize_: peft injects its wrappers on the dense
            # Linears (the post-quant torchao dispatch would TypeError on a manually
            # quantized module), then quantize_ converts only each wrapper's frozen
            # base_layer while the "lora_" side path stays high precision.
            baked = self._resolve_lora_set(
                [(i, w) for (i, w) in lora_specs if w != 0],
                family = getattr(fam, "name", None),
                hf_token = hf_token,
            )
            for name, path, _weight in baked:
                pipe.load_lora_weights(path, adapter_name = name)
            pipe.set_adapters(
                [n for (n, _p, _w) in baked],
                adapter_weights = [w for (_n, _p, w) in baked],
            )
            pipe._unsloth_loras = baked
            pipe._unsloth_loras_baked = True
            logger.info(
                "diffusion.lora_bake: %d adapter(s) attached before %s quantize",
                len(baked),
                scheme,
            )
        scheme = quantize_transformer(
            pipe,
            target,
            mode = mode,
            family = getattr(fam, "name", None),
            fast_accum = fast_accum,
            logger = logger,
        )
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
        base_local_dir: Optional[str] = None,
        fam: Optional[DiffusionFamily] = None,
    ) -> Any:
        """Assemble the diffusers pipeline around ``transformer`` and place it on ``device``
        (a no-op for an already-placed pre-quantized transformer; it moves the companions)."""
        if getattr(fam, "name", None) == KREA2_FAMILY_NAME:
            # krea ships transformers-5.x configs and no top-level tokenizer files, so
            # Pipeline.from_pretrained dies in the tokenizer (vocab_file = None); assemble
            # per-component like every other krea load path (see diffusion_krea2.py).
            pipe = load_krea2_pipeline(
                base_local_dir or base, dtype, hf_token = hf_token, transformer = transformer
            )
            pipe.to(device)
            return pipe
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": transformer}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        pipe = pipeline_cls.from_pretrained(base_local_dir or base, **pipe_kwargs)
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
        companion_override_mib: Optional[int] = None,
    ):
        """Build the memory plan for this load: snapshot free device memory and
        estimate the model's resident footprint, then let the planner pick an
        offload policy + VAE memory savers. Kept on the backend so the cached base
        repo (companion text-encoder / VAE) feeds the size estimate.

        The size estimate is per-kind: diffusers keeps GGUF weights packed (per-matmul
        transient dequant), so a GGUF loads near its on-disk size; a safetensors
        single-file loads near its on-disk size (it carries its dtype), except an fp8
        transformer file that gets upcast to bf16 on load (~2x resident); and a full
        pipeline is one cached download (transformer + companions), already compressed.
        ``transformer_resident_override_mib`` replaces the file-size transformer estimate
        when the loader is planning for a DIFFERENT artifact than the file on disk (the
        dense transformer-quant candidate, whose footprint the auto-policy estimates);
        ``companion_override_mib`` likewise replaces the cached companion total on that
        re-plan, so the base repo's PREFETCHED transformer/ shards -- which land in the
        same blob cache _companion_cache_bytes sums -- are not counted as companions on
        top of transformer_resident_override_mib (a double-count of the transformer)."""
        # Settled (max-over-reads) on cuda: a transient foreign allocation at the wrong instant
        # otherwise makes an empty card look full and silently declines the resident/quant fast
        # path (see settled_snapshot_device_memory).
        device_memory = settled_snapshot_device_memory(target)
        if kind == "pipeline":
            # The whole repo is one cached download; cached bytes are the resident estimate
            # (bnb-4bit/fp8 stay compressed). A LOCAL path isn't cached, so sum its on-disk weights.
            local_repo = Path(repo_id).expanduser() if repo_id else None
            if local_repo is not None and local_repo.is_dir():
                cached = self._local_dir_weight_bytes(local_repo, exclude_transformer = False)
            else:
                cached = self._cache_bytes(repo_id) if repo_id else 0
            cached_mib = int(cached // (1024 * 1024)) if cached else None
            model_dense_mib = estimate_safetensors_dense_mib(cached_mib)
            # A repo can store weights NARROWER than the loaded dtype: ideogram-4's base ships its
            # two DiTs as raw float8, so cached bytes undershoot the bf16 footprint ~2x and auto
            # would OOM. When the size table knows the bf16 total for THIS repo, plan against the larger.
            is_narrow_base = bool(repo_id) and repo_id.strip().lower() == fam.base_repo.lower()
            if (
                not is_narrow_base
                and fam.name == IDEOGRAM4_FAMILY_NAME
                and local_repo is not None
                and local_repo.is_dir()
            ):
                # A local fp8 mirror never string-matches base_repo, so detect fp8 from the shard
                # headers and reserve the bf16 footprint (a local nf4 mirror stays compressed).
                is_narrow_base = ideogram4_repo_is_fp8(repo_id)
            if is_narrow_base:
                table = family_bf16_components_gb(fam, fam.base_repo)
                if table is not None:
                    # Reserve the bf16 footprint from this network-free constant even when the
                    # cache estimate is absent; else model_dense_mib stays None ("size unknown ->
                    # resident") and the ~54 GB fp8 pipeline OOMs a card offload would have fit.
                    table_mib = int(sum(table) * (1000.0**3) / (1024.0 * 1024.0))
                    model_dense_mib = (
                        table_mib if model_dense_mib is None else max(model_dense_mib, table_mib)
                    )
            companion_mib = None
        else:
            if transformer_resident_override_mib is not None:
                # Planning for the dense-quant candidate (not the file on disk): the auto-policy's
                # estimate replaces the file-size derivation; companions stay measured from cache.
                transformer_resident = transformer_resident_override_mib
            elif kind == "single_file":
                # An fp8 checkpoint upcasts to bf16 on load (~2x resident); detect from the
                # basename. Excludes the SDXL (single_file_is_pipeline) case (already bf16).
                fp8_upcast = not getattr(fam, "single_file_is_pipeline", False) and (
                    "fp8" in Path(single_file_path).name.lower() if single_file_path else False
                )
                transformer_resident = estimate_safetensors_dense_mib(
                    file_size_mib(single_file_path), fp8_upcast = fp8_upcast
                )
            else:
                transformer_resident = estimate_gguf_resident_mib(file_size_mib(single_file_path))
            # Companions (VAE + text encoders) load near on-disk size; sum the base-repo cache,
            # or a LOCAL base's on-disk weights (the blob cache is empty for a local path).
            if companion_override_mib is not None:
                # Re-planning the dense candidate: the prefetched transformer/ shards land in the
                # SAME cache _companion_cache_bytes sums, so use the auto-policy's companion estimate
                # instead of double-counting the transformer.
                companion_mib = companion_override_mib
            else:
                companion = self._companion_cache_bytes(base)
                companion_mib = int(companion // (1024 * 1024)) if companion else None
            model_dense_mib = None
            if transformer_resident is not None:
                model_dense_mib = transformer_resident + (companion_mib or 0)
        # Feed the variant hint (basename + repo) so estimate_image_runtime_mib sees distilled
        # markers ("turbo"/"schnell") normalized out of fam.name (distilled needs ~15% less headroom).
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

        # torch_dtype=None is load-bearing: from_pipe otherwise recasts EVERY component to fp32,
        # which upcasts the reused bf16 modules and hard-crashes the dense-quant path (a
        # torchao+compiled transformer's tensor-subclass weights can't swap_tensors). None reuses
        # the resident modules at their loaded dtype (the point of from_pipe).
        pipe = getattr(diffusers, class_name).from_pipe(state.pipe, torch_dtype = None)
        # Publish to the shared aux cache only if THIS load is still current: from_pipe runs under
        # _generate_lock but NOT _lock, so an unload can null _state while it builds; caching then
        # would hand a wrapper over stale modules to a later load.
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
            # resolve_controlnet accepts a bare owner/name without the base trust gate, and
            # from_pretrained deserializes it (a malicious pickle would execute), so run the same
            # Hub malware preflight the chat/export loaders use. A local dir is exempt (fail-open).
            if not getattr(resolved_cn, "is_local", False):
                from utils.security import evaluate_file_security
                _cn_fs = evaluate_file_security(resolved_cn.path, hf_token = state.hf_token or None)
                if _cn_fs.blocked:
                    raise ValueError(_cn_fs.reason)
            # Keep at most one ControlNet resident: evict the previous module + wrapper first,
            # or swapping ControlNets within a load accumulates until OOM.
            if self._cn_models or self._cn_pipes:
                self._cn_models.clear()
                self._cn_pipes.clear()
                clear_gpu_cache()
            import torch

            # state.dtype is the display string ("bfloat16"), not a torch.dtype; pass the real
            # dtype so diffusers loads at the base compute dtype, not float32 (extra VRAM).
            cn_dtype = getattr(torch, str(state.dtype).replace("torch.", ""), None)
            cn_model = getattr(diffusers, model_cls_name).from_pretrained(
                resolved_cn.path,
                torch_dtype = cn_dtype,
                token = state.hf_token or None,  # blank -> anonymous
            )
            if cancel.is_set():
                # An unload raced the blocking download; bail BEFORE placement so we don't
                # allocate onto a GPU _unload_locked() just freed.
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            # Placement follows the base's offload policy: a resident base places it resident, an
            # offloaded base streams it via group offloading. Best-effort; failure -> resident.
            if getattr(state, "offload_policy", OFFLOAD_NONE) != OFFLOAD_NONE and (
                _offload_controlnet_module(cn_model, state.device, logger)
            ):
                pass
            else:
                cn_model = cn_model.to(state.device)
            if cancel.is_set():
                # An unload raced the download and cleared the caches; caching now would pin it.
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            self._cn_models[resolved_cn.id] = cn_model
        key = (pipe_cls_name, resolved_cn.id)
        pipe = self._cn_pipes.get(key)
        if pipe is None:
            pipe = getattr(diffusers, pipe_cls_name).from_pipe(
                state.pipe, controlnet = cn_model, torch_dtype = None
            )
            with self._lock:
                # Same race as the model cache: an unload may have cleared _cn_pipes while
                # from_pipe ran; caching now would pin a pipeline over the unloaded base.
                if cancel.is_set() or self._state is not state:
                    del pipe
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                self._cn_pipes[key] = pipe
        return pipe

    @staticmethod
    def _align_vae_dtype(
        pipe: Any,
        denoiser_attr: str = "transformer",
        vae_quant: Optional[str] = None,
    ) -> None:
        """Cast the VAE to the denoiser's compute dtype before an image-conditioned
        call. The img2img/inpaint pipelines VAE-encode the input image at the text-
        encoder dtype (bf16), but a prior txt2img DECODE may have left the shared VAE
        upcast to fp32 (its ``force_upcast`` path), so the encode would mismatch
        (bf16 image vs fp32 VAE). Re-aligning here is safe: our families run bf16 or
        fp32 only (the fp16 guard promotes fp16), and a later txt2img decode re-upcasts
        as needed. ``denoiser_attr`` is ``pipe.transformer`` for DiT families and
        ``pipe.unet`` for SDXL. Best-effort; a no-op when already aligned.

        Skipped when ``vae_quant`` engaged: the fp8 tensor subclasses mishandle
        ``.to(dtype=...)`` (torchao rejects it), and the VAE already runs at the compute
        dtype under fp8, so the re-align is both harmful and unnecessary."""
        if vae_quant is not None:
            return
        denoiser = getattr(pipe, denoiser_attr, None)
        vae = getattr(pipe, "vae", None)
        if denoiser is None or vae is None:
            return
        try:
            # Read the dtype from the parameters (a plain/compiled nn.Module may hide .dtype).
            # Take the first FLOATING dtype (a GGUF transformer's leading params are packed uint8).
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

    @staticmethod
    def _resolve_lora_set(
        specs: list[tuple[str, float]],
        *,
        family: Optional[str],
        hf_token: Optional[str],
        cancel: Optional[threading.Event] = None,
    ) -> tuple[tuple[str, str, float], ...]:
        """Resolve (id, weight) specs to a ``(name, path, weight)`` tuple set for diffusers.

        Shared by the generation-time apply path and the quant load-time bake so both produce
        IDENTICAL tuples for the same request (the no-op / weight-only comparisons depend on it).
        """
        from core.inference import diffusion_lora

        # This branch's resolve_specs has no family-scoped catalog; the parameter is kept so
        # the caller code stays identical across branches.
        del family
        resolved = diffusion_lora.resolve_specs(
            specs,
            hf_token = hf_token,
            cancel_event = cancel,
        )
        # diffusers load_lora_weights takes safetensors only; reject a .gguf adapter as a clean 400.
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
        return tuple(uniq)

    def _apply_loras(
        self, state: Any, loras: Optional[list[tuple[str, float]]], cancel: threading.Event
    ) -> None:
        """Load + activate requested LoRA adapters on ``state.pipe`` (non-fused), or clear
        them when none are requested.

        The applied set is recorded on the pipe object, so an unchanged selection is a no-op
        and a model swap (a fresh pipe with no marker) resets naturally. Never fuses: fusing
        breaks on quantized (bnb-4bit / torchao) transformers and blocks live weight tweaks.

        A torchao int8/fp8 pipe carries its adapters from the load-time BAKE (attached before
        quantize_ + compile). Its module topology is frozen: weight-only changes go through
        set_adapters (value-level, compile-guard safe); adding/removing adapters needs a reload
        with the new selection, surfaced as a clean 400 here.
        """
        from core.inference import diffusion_lora

        pipe = state.pipe
        current = getattr(pipe, "_unsloth_loras", ())
        specs = [(i, w) for (i, w) in (loras or []) if w != 0]

        quant_baked = bool(getattr(pipe, "_unsloth_loras_baked", False))
        quant = (state.transformer_quant or "").lower()
        if quant in ("int8", "fp8", "nvfp4", "mxfp8"):
            self._adjust_baked_loras(state, pipe, specs, current, quant_baked, cancel)
            return

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
                "(GGUF-via-diffusers, or a torch.compile'd Speed=default/max load). Use a bf16 "
                "or bnb-4bit load at Speed=off/eager, or the native engine for GGUF models."
            )

        desired = self._resolve_lora_set(
            specs,
            family = getattr(state.family, "name", None),
            hf_token = state.hf_token,
            cancel = cancel,
        )
        uniq = list(desired)
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

    def _adjust_baked_loras(
        self,
        state: Any,
        pipe: Any,
        specs: list[tuple[str, float]],
        current: tuple,
        quant_baked: bool,
        cancel: threading.Event,
    ) -> None:
        """Generation-time LoRA handling for a torchao-quantized pipe.

        The adapters (if any) were baked at load time, before quantize_ + compile, so the
        module topology is immutable here. Allowed without a reload: weight tweaks on the
        baked set and disabling everything (scale 0 reproduces the quantized base exactly;
        set_adapters is value-level, so torch.compile guards absorb it). Anything that would
        change topology (adding adapters to a bake-less load, or a different adapter set)
        raises a clean 400 telling the client to reload with the new selection.
        """
        if not quant_baked:
            if not specs:
                return  # no adapters baked, none requested
            raise ValueError(
                "This quantized (int8/fp8) load was built without LoRA adapters. Reload the "
                "model with the adapter selection to bake it into the quantized transformer."
            )
        if not specs:
            # Disable every baked adapter: scale 0 reproduces the quantized base exactly.
            names = [n for (n, _p, _w) in current]
            if any(w != 0 for (_n, _p, w) in current):
                pipe.set_adapters(names, adapter_weights = [0.0] * len(names))
                pipe._unsloth_loras = tuple((n, p, 0.0) for (n, p, _w) in current)
            return
        desired = self._resolve_lora_set(
            specs,
            family = getattr(state.family, "name", None),
            hf_token = state.hf_token,
            cancel = cancel,
        )
        if desired == current:
            return
        if [(n, p) for (n, p, _w) in desired] == [(n, p) for (n, p, _w) in current]:
            # Same adapters, new weights: value-level change on the baked topology.
            pipe.set_adapters(
                [n for (n, _p, _w) in desired],
                adapter_weights = [w for (_n, _p, w) in desired],
            )
            pipe._unsloth_loras = desired
            return
        raise ValueError(
            "The LoRA selection changed, but a quantized (int8/fp8) transformer bakes its "
            "adapters at load time. Reload the model with the new adapter selection."
        )

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clear the transformer's stateful step cache (FBCache) before a generation.

        diffusers keys FBCache residuals by cache context ("cond"/"uncond") on the
        long-lived transformer, and neither the pipeline nor the context exit resets
        them. The transformer-level reset entry point is ``_reset_stateful_cache`` in
        diffusers 0.39 (``reset_stateful_hooks`` lives only on the HookRegistry, so a
        getattr for it on the transformer is a silent no-op), and no pipeline calls it.
        This backend reuses one resident pipe across generations, so without a reset the
        next generation's first step compares its first-block residual against the
        PREVIOUS request's -- a tensor-shape mismatch when the resolution/batch changed,
        or a stale-cache reuse otherwise. Best-effort: a transformer without the hook
        (uncached load) is a silent no-op."""
        transformer = getattr(pipe, "transformer", None)
        reset = getattr(transformer, "_reset_stateful_cache", None) or getattr(
            transformer, "reset_stateful_hooks", None
        )
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 — reset is best-effort, never fail a generation
                pass

    def _engage_deferred_speed(self, state: _LoadState) -> None:
        """Engage the deferred `default` speed profile at the start of the 3rd
        generation this session.

        The load left the pipe fully eager (bit-identical reference); by the 3rd
        image repeated use is established, so pay the one-time compile now: eager
        patches + attention auto upgrade + regional compile -- exactly what an
        unset-speed GGUF load gets at load time. Runs under _generate_lock (the
        caller), so no denoise can race the mutation. The flag is cleared FIRST so
        a failure never retries per generation; unload cleans everything up via the
        same state fields the load-time path uses (backend flags were snapshotted
        at load, before any speed layer could mutate them)."""
        object.__setattr__(state, "speed_deferred", False)
        from .diffusion_eager_patches import install_compile_safe_patches
        from .diffusion_arch_patches import install_arch_patches

        target = self._resolve_device_target(state.family)
        install_compile_safe_patches()
        install_arch_patches()
        object.__setattr__(state, "eager_patched", True)
        # Re-run the load-time selection with the caller's ORIGINAL request (not a bare
        # None): an explicit backend must survive the deferred upgrade. auto still upgrades
        # to cuDNN here (speed_active=True), but an explicit "native"/"sage"/"flash" is
        # honored verbatim rather than silently replaced by the auto cuDNN choice.
        attention_engaged = apply_attention_backend(
            state.pipe,
            select_attention_backend(target, state.attention_request, speed_active = True),
            logger = logger,
        )
        object.__setattr__(state, "attention_backend", attention_engaged)
        gguf_transformer = state.kind == "gguf" and state.transformer_quant is None
        if compile_eligible(target, is_gguf = gguf_transformer, family = state.family):
            compile_ctx = compile_cache.begin(
                family = state.family.name,
                # U-Net families (SDXL) carry the denoiser as pipe.unet.
                transformer = getattr(state.pipe, "transformer", None)
                or getattr(state.pipe, "unet", None),
                dtype = getattr(target, "dtype", None),
                quant = state.transformer_quant,
                attention_backend = attention_engaged,
                compile_kwargs = {
                    # Mirrors the load-time fullgraph decision: an engaged or
                    # still-toggleable step cache OR an offload graph-breaks.
                    "fullgraph": state.transformer_cache is None
                    and not state.cache_auto
                    and state.offload_policy == OFFLOAD_NONE,
                    "dynamic": True,
                    "mode": "default",
                },
                logger = logger,
            )
            object.__setattr__(state, "compile_cache_ctx", compile_ctx)
        speed_applied = apply_speed_optims(
            state.pipe,
            target,
            is_gguf = gguf_transformer,
            family = state.family,
            speed_mode = SPEED_DEFAULT,
            cache_active = state.transformer_cache is not None or state.cache_auto,
            offload_active = state.offload_policy != OFFLOAD_NONE,
            logger = logger,
        )
        object.__setattr__(state, "speed_mode", SPEED_DEFAULT)
        object.__setattr__(state, "speed_optims", tuple(k for k, v in speed_applied.items() if v))
        entry = (state.resolved or {}).get("speed_mode")
        if isinstance(entry, dict):
            entry["value"] = SPEED_DEFAULT
            entry["reason"] = (
                "auto: compiled on the 3rd image this session "
                "(repeated use amortises the one-time compile)"
            )
        att = (state.resolved or {}).get("attention_backend")
        if isinstance(att, dict) and att.get("source") == "auto":
            att["value"] = attention_engaged or "native"
            att["reason"] = (
                "cuDNN fused attention upgrade" if attention_engaged else "diffusers default"
            )
        logger.info(
            "diffusion.speed: deferred profile engaged on generation 3 "
            "(optims=%s, attention=%s)",
            ",".join(state.speed_optims) or "none",
            attention_engaged or "native",
        )

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        # Fallbacks; the route always sends the per-model values the UI seeds.
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        # Image-conditioned (base64/data-URL): init alone = img2img; init + mask = inpaint.
        # ``strength`` is the denoise strength (0 = keep source, 1 = full redraw). None = txt2img.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        # Upscale (hires fix): factor > 1 with an init image enlarges then re-denoises at low strength.
        upscale: Optional[float] = None,
        # Reference (FLUX.2): additional reference images beyond init_image (a list). Ignored elsewhere.
        reference_images: Optional[list[str]] = None,
        # LoRA (id, weight) pairs; loaded non-fused and activated for this generation. None/empty clears.
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet (id, control_image_b64, control_type, strength, guidance_start, guidance_end). None = off.
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
    ) -> dict[str, Any]:
        import torch
        from PIL import Image

        # Per-generation cancel Event that unload()/a superseding load set (registered under
        # _lock below) to abort just this denoise. _generate_lock is the only lock the denoise holds.
        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # Register under _lock so unload()/a load can signal THIS generation.
                self._active_generate_cancel = cancel
                # Publish an active (step 0) state now, before the slow pre-denoise setup (deferred
                # compile, LoRA resolution, ControlNet build), so a reload's mount probe doesn't read
                # idle while this generation holds _generate_lock and let a second generate queue
                # behind it. The per-step callback swaps in its own _GenState at denoise start.
                self._gen = _GenState(total_steps = steps)
            try:
                # The local `state` ref keeps the pipe alive even if unload() nulls _state.
                generator = torch.Generator(device = state.device)
                if seed is None:
                    # Keep the seed in JS's safe-integer range (< 2**53) so it round-trips
                    # through JSON and reproduces the image (a raw 64-bit seed loses precision).
                    seed = generator.seed() & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                generator.manual_seed(seed)

                # Deferred speed auto: engage the compile profile on the 3rd image, before the LoRA/
                # workflow wiring (load-time ordering). Best-effort; a failure stays eager and never retries.
                # NOT when a LoRA is requested: a compiled transformer rejects LoRA, so compiling
                # here would permanently break every LoRA generation on this load.
                lora_requested = any(w != 0 for (_id, w) in (loras or []))
                # Also stay eager while a PRIOR generation's adapters are attached: _apply_loras runs
                # AFTER the engage below, so compiling would bake the adapter in and the swallowed
                # unload_lora_weights() would leave it active forever. Defer until a later gen.
                loras_attached = bool(getattr(state.pipe, "_unsloth_loras", ()))
                if (
                    state.speed_deferred
                    and state.generation_count >= 2
                    and not lora_requested
                    and not loras_attached
                ):
                    try:
                        self._engage_deferred_speed(state)
                    except Exception as exc:  # noqa: BLE001 — speed is best-effort
                        logger.warning(
                            "diffusion.speed: deferred engagement failed, staying eager: %s",
                            exc,
                        )

                # Apply/adjust LoRA before picking the workflow pipe; from_pipe pipes share the transformer.
                self._apply_loras(state, loras, cancel)

                # Select the workflow pipeline: txt2img uses the loaded pipe; img2img/inpaint reuse
                # its modules via from_pipe; an edit model's own pipe is already the edit pipeline.
                pipe = state.pipe
                init_pil = mask_pil = None
                control_pil = None
                cn_scale = cn_gstart = cn_gend = cn_mode = None
                ref_extra: list = []
                # Validate dependencies up front: mask/upscale/reference need an input image, and
                # reference needs a supporting family (else the combo silently falls back to txt2img).
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
                    # Instruction editing: the loaded pipe IS the edit pipeline; always needs an
                    # input image, prompt is the instruction. No mask, no from_pipe.
                    if init_image is None:
                        raise ValueError(
                            f"{state.family.name} is an image-editing model: provide an input image."
                        )
                    if mask_image is not None:
                        # The edit family has no inpaint pipeline; a mask would be silently dropped.
                        raise ValueError(
                            f"{state.family.name} is an image-editing model and does not "
                            "support masks (mask_image)."
                        )
                    workflow = "edit"
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                elif mask_image is not None and init_image is not None:
                    workflow = "inpaint"
                    pipe = self._workflow_pipe(state, state.family.inpaint_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    mask_pil = _decode_b64_image(mask_image, mode = "L")
                elif init_image is not None and upscale is not None and upscale > 1.0:
                    # Upscale (hires fix): enlarge with Lanczos, then re-run img2img at low strength
                    # to add detail without redrawing. Shares the img2img pipeline via from_pipe.
                    workflow = "upscale"
                    pipe = self._workflow_pipe(state, state.family.img2img_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    iw, ih = init_pil.size
                    # Cap the factor, then the absolute output (longest side 2048, txt2img's max)
                    # to avoid an OOM-scale latent; round to a multiple of 16 (VAE downsample + patch).
                    factor = max(1.0, min(float(upscale), 4.0))
                    tw_f, th_f = iw * factor, ih * factor
                    max_side = 2048
                    fit = min(1.0, max_side / max(tw_f, th_f))
                    tw = max(16, int(round(tw_f * fit / 16.0)) * 16)
                    th = max(16, int(round(th_f * fit / 16.0)) * 16)
                    # After the cap, the target must still exceed the input (else upscale shrinks it).
                    if max(tw, th) <= max(iw, ih):
                        raise ValueError(
                            f"Upscale would not enlarge this image: its longest side "
                            f"({max(iw, ih)}px) already meets the {max_side}px output limit. "
                            f"Use a smaller source image."
                        )
                    init_pil = init_pil.resize((tw, th), Image.LANCZOS)
                    if strength is None:
                        strength = 0.35  # hires-fix default: preserve content, add detail
                elif getattr(state.family, "reference", False) and init_image is not None:
                    # FLUX.2 reference conditioning: the loaded pipe takes the reference via `image`
                    # and generates at the REQUESTED size. No from_pipe, no strength; output size
                    # from the sliders. After inpaint/upscale so a mask/upscale on a reference family routes right.
                    workflow = "reference"
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    # Additional references (FLUX.2 combines a list); capped to bound VRAM.
                    ref_extra = [
                        _decode_b64_image(x, mode = "RGB") for x in (reference_images or [])[:3]
                    ]
                elif init_image is not None:
                    workflow = "img2img"
                    pipe = self._workflow_pipe(state, state.family.img2img_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                else:
                    workflow = "txt2img"

                # ControlNet (diffusers): txt2img only (not img2img/inpaint/edit). Builds the
                # family's CN pipeline around resident modules and passes a control map.
                if controlnet is not None:
                    from core.inference import diffusion_controlnet
                    cn_id, cn_image_b64, cn_type, cn_strength, cn_gs, cn_ge = controlnet
                    # strength 0 disables CN: skip the whole path so a no-op never pays the download/VRAM.
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
                        # Decode + preprocess the control image FIRST so a bad image 400s before
                        # any CN download/build. Control map at the OUTPUT size to align with latents.
                        src = _decode_b64_image(cn_image_b64, mode = "RGB")
                        control_pil = diffusion_controlnet.preprocess_control(src, cn_type).resize(
                            (width, height), Image.LANCZOS
                        )
                        try:
                            resolved_cn = diffusion_controlnet.resolve_controlnet(
                                cn_id, family = state.family.name
                            )
                        except FileNotFoundError as exc:
                            # An unknown CN id -> 400, not 500 (the route maps ValueError).
                            raise ValueError(str(exc)) from exc
                        pipe = self._controlnet_pipe(state, resolved_cn, cancel)
                        workflow = "controlnet"
                        cn_scale, cn_gstart, cn_gend = cn_strength, cn_gs, cn_ge
                        # Flux Union CN selects its head by an integer control_mode; map the type.
                        cn_mode = diffusion_controlnet.union_control_mode(cn_id, cn_type)
                # Snap odd-sized inputs to a multiple of 16 for workflows whose OUTPUT size comes
                # from the input image (img2img/inpaint/edit); txt2img/reference use the slider,
                # upscale already produced a /16 target. Mask is matched to the snapped image.
                if init_pil is not None and workflow in ("img2img", "inpaint", "edit"):
                    # img2img/inpaint take output size from the upload, so bound the longest side to
                    # 2048 first (else a phone photo drives an OOM-scale latent). edit resizes internally.
                    if workflow in ("img2img", "inpaint"):
                        init_pil = _clamp_max_side(init_pil, 2048)
                    init_pil = _snap_to_multiple(init_pil, 16)
                    if mask_pil is not None and mask_pil.size != init_pil.size:
                        from PIL import Image as _PILImage
                        mask_pil = mask_pil.resize(init_pil.size, _PILImage.NEAREST)
                if init_pil is not None:
                    # Keep the VAE encode dtype consistent with the input image. Pass the engaged
                    # vae_quant so a quantised (fp8) VAE skips the re-align (must not be re-cast).
                    self._align_vae_dtype(pipe, state.family.denoiser_attr, state.vae_quant)

                # Pipelines vary in accepted kwargs, so gate every optional one on the signature.
                call_params = inspect.signature(pipe.__call__).parameters

                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    # Most pipelines use "guidance_scale"; Qwen-Image uses "true_cfg_scale".
                    state.family.cfg_kwarg: guidance,
                    "generator": generator,
                    # Whole batch in one forward pass; all share this call's seed.
                    "num_images_per_prompt": batch_size,
                }
                if state.family.name == IDEOGRAM4_FAMILY_NAME:
                    # Ideogram 4 drives CFG via EITHER a constant guidance_scale OR a per-step
                    # guidance_schedule (check_inputs rejects both). At the advertised defaults drop
                    # the constant so the recommended 48-step taper engages; else null the schedule.
                    if steps == 48 and abs(float(guidance) - 7.0) < 1e-6:
                        kwargs.pop(state.family.cfg_kwarg, None)
                    else:
                        kwargs["guidance_schedule"] = None
                if state.family.name == LUMINA2_FAMILY_NAME and "cfg_trunc_ratio" in call_params:
                    # Lumina 2's card recipe runs the CFG double-forward only over the FIRST
                    # quarter of the trajectory (cfg_trunc_ratio=0.25); the pipeline default (1.0)
                    # applies it everywhere, visibly oversaturating output. Constant card value.
                    kwargs["cfg_trunc_ratio"] = 0.25
                if init_pil is not None:
                    # Reference passes the whole list (FLUX.2 combines); others take the single image.
                    kwargs["image"] = [init_pil, *ref_extra] if ref_extra else init_pil
                    if mask_pil is not None and "mask_image" in call_params:
                        kwargs["mask_image"] = mask_pil
                    if strength is not None and "strength" in call_params:
                        kwargs["strength"] = strength
                # width/height: txt2img uses the slider; image-conditioned pipes must use the INPUT
                # IMAGE's own size (a differing slider mismatches the latents). Many img2img/inpaint
                # pipes drop them entirely, so pass only when accepted, derived from the image.
                if workflow in ("txt2img", "reference", "controlnet"):
                    # These generate at the REQUESTED size (reference/control image resized to match).
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
                    # CN pipeline takes the control map + scale; guidance start/end bound its step
                    # range. Every kwarg is signature-gated so a family that omits one still runs.
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
                    # Union CN mode index (Flux); only when accepted and the type maps to a mode.
                    if "control_mode" in call_params and cn_mode is not None:
                        kwargs["control_mode"] = cn_mode

                gen = _GenState(total_steps = steps)

                def _on_step(pipe, step_index, timestep, callback_kwargs):
                    # Monotonic: a wall-clock adjustment (NTP) mid-denoise would skew the ETA.
                    now = time.monotonic()
                    gen.step = step_index + 1
                    if gen.first_step_at == 0.0:
                        gen.first_step_at = now
                    gen.eta_seconds = _estimate_eta(
                        gen.total_steps, gen.step, gen.first_step_at, now
                    )
                    # Preempt a long denoise on unload/superseding load (diffusers checks _interrupt).
                    if cancel.is_set():
                        pipe._interrupt = True
                    return callback_kwargs

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step

                # Re-check an AUTO cache decision against the ACTUAL step count (a 28-step request
                # gains FBCache, a few-step turbo drops it); explicit choices never toggle.
                if state.cache_auto:
                    # Key on the EFFECTIVE denoise steps: an img2img/upscale request at strength < 1
                    # only denoises a fraction of `steps`, so folding in `strength` (only when
                    # actually applied) keeps FBCache off the short trajectory. When strength is
                    # omitted the pipe's own default (< 1) still applies, so key on that too.
                    strength_applied = effective_request_strength(
                        strength,
                        init_pil is not None,
                        "strength" in call_params,
                        call_params["strength"].default if "strength" in call_params else None,
                    )
                    denoise_steps = effective_denoise_steps(steps, strength_applied)
                    toggled = maybe_toggle_step_cache(
                        state.pipe,
                        steps = denoise_steps,
                        quant_active = state.cache_quant_active,
                        threshold = state.cache_threshold,
                        logger = logger,
                    )
                    if toggled != state.transformer_cache:
                        # _LoadState is frozen; the one deliberate in-place update, tracking the
                        # pipe-level toggle so status() is truthful.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {denoise_steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                # Start each generation from a clean step cache: prior FBCache residuals would
                # otherwise be compared against this first step (shape mismatch / stale reuse).
                if state.transformer_cache:
                    self._reset_step_cache(state.pipe)

                self._gen = gen
                try:
                    # inference_mode is faster than no_grad and numerically identical here.
                    with torch.inference_mode():
                        images = pipe(**kwargs).images
                finally:
                    self._gen = None
                # A cancelled denoise returns a partial/garbage image; don't persist it.
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # Persist the warm torch.compile bundle after the first compiled generation. A
                # STATIC compile makes new artifacts per (w,h,batch), so register this shape first
                # (an uncovered shape re-dirties the context and the save rewrites the bundle).
                # Idempotent + best-effort.
                try:
                    # Register the dims the forward ACTUALLY compiled with (image-conditioned
                    # workflows run at the input image's size, not the slider; see _compile_shape_dims).
                    reg_width, reg_height = _compile_shape_dims(workflow, init_pil, width, height)
                    compile_cache.register_shape(
                        state.compile_cache_ctx,
                        (reg_width, reg_height, int(batch_size)),
                        static = "compiled" in (state.speed_optims or ())
                        and compiled_shapes_are_static(state.pipe, state.speed_mode),
                    )
                    compile_cache.save(state.compile_cache_ctx, logger = logger)
                except Exception:  # noqa: BLE001 — cache persistence is best-effort
                    pass
                # Count the finished generation (drives deferred speed); a batch is one generation.
                object.__setattr__(state, "generation_count", state.generation_count + 1)
                # Return the PIL images (unencoded); the route embeds recipes and persists them.
                return {"images": list(images), "seed": int(seed), "repo_id": state.repo_id}
            finally:
                # Deregister so a later unload/load can't poke a finished generation (if still ours).
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                    # Drop the published progress state, covering a setup-time error that skips
                    # the inner finally. Safe under _generate_lock.
                    self._gen = None

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
        # Abort an in-flight (lock-free) download so unload/eviction returns promptly.
        self._cancel_event.set()
        with self._lock:
            # Abort an in-flight denoise via ITS cancel event.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            # Cancel any in-flight load (its worker checks this token) and drop the marker.
            self._load_token += 1
            self._loading = None
        # Wait for the signalled denoise to exit BEFORE tearing down: _unload_locked uninstalls
        # process-wide state (attention patches, GGUF compile hooks, backend flags, compile cache)
        # the denoise still depends on but its pipe ref doesn't pin. The denoise holds _generate_lock
        # for its whole body, so acquiring it here blocks until it has finished. Mirrors begin_load.
        with self._generate_lock:
            with self._lock:
                self._unload_locked()
        return self.status()

    def _unload_locked(self) -> None:
        state = self._state
        if state is None:
            return
        # Restore the process-wide backend flags this load flipped, so the next `off` load is
        # bit-identical. compile_cache.restore + gguf_compile.uninstall_all likewise; all idempotent.
        restore_backend_flags(state.backend_flags_before)
        compile_cache.restore(state.compile_cache_ctx)
        gguf_compile.uninstall_all()
        if state.eager_patched:
            # Lazy import to keep diffusion.py torch-free to import.
            from .diffusion_eager_patches import uninstall_patches
            from .diffusion_arch_patches import uninstall_arch_patches

            uninstall_patches()
            uninstall_arch_patches()
        # Deliberately NOT unload_lora_weights() here: the whole pipe is dropped below, freeing any
        # LoRA adapters with it. Both callers hold _generate_lock across this teardown, so no denoise
        # is in flight.
        # Drop the workflow pipes so they don't pin the freed pipeline's modules past unload.
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
                "vae_quant": None,
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
            "vae_quant": state.vae_quant,
            "transformer_quant": state.transformer_quant,
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "resolved": state.resolved,
            # Workflows the loaded family supports, so the UI can gate its tabs.
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
    # Instruction-editing families have no txt2img mode, so expose only "edit".
    if getattr(fam, "edit", False):
        return ["edit"]
    workflows = ["txt2img"]
    # Reference families (FLUX.2) add reference conditioning via their pipeline's image arg.
    if getattr(fam, "reference", False):
        workflows.append("reference")
    if getattr(fam, "img2img_pipeline_class", None):
        # Upscale runs on the img2img pipeline, so available exactly when img2img is.
        workflows.append("img2img")
        workflows.append("upscale")
    if getattr(fam, "inpaint_pipeline_class", None):
        workflows.append("inpaint")
        # Outpaint reuses the inpaint pipeline with a padded canvas, so needs one that preserves size.
        if getattr(fam, "inpaint_preserves_size", True):
            workflows.append("outpaint")
    return workflows


def _resolve_base_repo(
    repo_id: str, base_repo: Optional[str], fam: DiffusionFamily, hf_token: Optional[str]
) -> str:
    """The companion diffusers repo: caller's base, else the GGUF repo's own
    ``base_model`` tag, else the family fallback. Shared by both load paths so a
    direct ``load_pipeline`` call resolves the variant base the same way.

    The base loads via ``from_pretrained``, so it must be trusted -- an explicit
    base_repo is already gated at ``validate_load_request``, but the ``base_model``
    card tag is attacker-controlled metadata on any remote GGUF repo, so a tag that
    is not unsloth/allowlisted/local is dropped in favour of the curated family
    default (never fed to ``from_pretrained``), closing the pickle-deserialisation
    vector the ControlNet path already guards with evaluate_file_security."""
    base = (base_repo or "").strip()
    if not base:
        tag = _hf_base_model(repo_id, hf_token)
        if tag and _is_trusted_diffusion_repo(tag):
            base = tag
    return resolve_base_repo(fam, base)


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
