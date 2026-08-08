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
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core._torchao_stub import (
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
)
from loggers import get_logger
from utils.hardware import clear_gpu_cache

from .diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    IDEOGRAM4_FAMILY_NAME,
    LUMINA2_FAMILY_NAME,
    DiffusionFamily,
    assert_flux2_gguf_matches_base,
    assert_pipeline_class_available,
    canonical_base,
    default_generation_params,
    detect_family_for_pick,
    excluded_model_reason,
    prefer_ungated_mirror,
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
from .diffusion_hidream import HIDREAM_FAMILY_NAME, hidream_te4_kwargs
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
from . import diffusion_cond_cache as cond_cache
from . import diffusion_gguf_compile as gguf_compile
from .diffusion_batched import (
    chunk_jobs,
    is_oom_error,
    resolve_batch_jobs,
    split_chunk,
    uniform_prompt,
)
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
from .diffusion_precision import normalize_te_quant, quantize_text_encoders
from .diffusion_te_prequant import te_prequant_pipe_kwargs
from .diffusion_prequant import (
    load_prequantized_transformer,
    prequant_checkpoint_cached,
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

# Every `import diffusers` below is lazy, so this runs first. On Windows ROCm both reach an absent
# distributed backend: diffusers imports xformers on sight, its quantizers torchao.
install_xformers_windows_rocm_stub()
install_torchao_windows_rocm_stub()


# "gguf" and "single_file" take companions from the base repo; "pipeline" is a full diffusers repo.
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})


def _record_sha(shas_out: Optional[dict[str, str]], repo_id: str, info: Any) -> None:
    """Note the commit ``info`` describes so a cache probe can pin to it.

    No ``sha`` just means the plan cannot pin, and stages as before."""
    if shas_out is None:
        return
    sha = getattr(info, "sha", None)
    if isinstance(sha, str) and sha:
        shas_out[repo_id] = sha


def hub_cache_dir() -> str:
    """The cache root every loader call must be pinned to.

    diffusers resolves an unset cache_dir through huggingface_hub's import-time constant,
    which a mid-session cache-folder change does not update. The prefetch reads the live
    setting, so without this a single load could split across two roots."""
    from utils.hf_cache_settings import active_hf_hub_cache
    return active_hf_hub_cache()


# Repo id out of any hub URL in an error. A gated PUBLIC repo raises a download URL,
# ".../huggingface.co/<owner>/<name>/resolve/..."; auth_check, and model_info on a gated PRIVATE
# repo, raise ".../huggingface.co/api/models/<owner>/<name>", which without the prefix branch would yield "api/models".
_HUB_REPO_RE = re.compile(
    r"huggingface\.co/(?:api/(?:models|datasets|spaces)/)?([\w.\-]+/[\w.\-]+)"
)


def _gated_in_chain(exc: BaseException) -> Optional[BaseException]:
    """The GatedRepoError in ``exc``'s cause/context chain, or None. Transformers loads re-raise
    the 403 wrapped in an OSError, so the outermost error alone misses the case this exists for.
    Screened by class name across the MRO, so no hub import."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(cls.__name__ == "GatedRepoError" for cls in type(exc).__mro__):
            return exc
        # `raise ... from None` means the raiser already wrote a better message (the base-repo preflight does), so stop.
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return None


def _hf_token_in_play(hf_token: Optional[str]) -> bool:
    """Whether the failing Hub call carried ANY credential, not just Studio's own: with token=None
    huggingface_hub still falls back to HF_TOKEN or the cached CLI login, so keying off the request
    token alone loops an already-authenticated user."""
    if hf_token:
        return True
    try:
        # What build_hf_headers calls: under HF_HUB_DISABLE_IMPLICIT_TOKEN get_token() still answers while the request goes out anonymous.
        from huggingface_hub.utils import get_token_to_send
        return bool(get_token_to_send(None))
    except Exception:  # noqa: BLE001 -- assume none; at worst the message says "add a token"
        return False


def hub_access_message(exc: BaseException, *, had_token: bool) -> Optional[str]:
    """Rewrite a gated-repo failure into the step that actually unblocks the user, else None so an
    unrelated load error keeps its own text. Only the toast is affected; the raw exception, request
    id and resolve URL still reach the log."""
    gated = _gated_in_chain(exc)
    if gated is None:
        return None
    found = _HUB_REPO_RE.search(str(gated))
    # An API URL ends at the repo, so a trailing full stop lands inside the name (dots are legal mid-name, but a name never ends in one).
    repo = found.group(1).rstrip(".") if found else None
    # Any other /api/<endpoint> URL (whoami-v2, ...) would parse as the repo "api/<endpoint>".
    if repo and repo.split("/", 1)[0] == "api":
        repo = None
    where = f"https://huggingface.co/{repo}" if repo else "its Hugging Face page"
    subject = repo or "This model"
    if had_token:
        # A token was sent and still bounced, so the account itself lacks access.
        return f"{subject} is gated and this Hugging Face account is not on its access list. Request access at {where}, then load again."
    return f"{subject} is gated. Request access at {where}, then add a Hugging Face token in Settings and load again."


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


def _active_lora_pairs(pipe: Any) -> list:
    """``[(name, weight)]`` for the adapters actually attached to ``pipe``, zero-weight ones
    dropped.

    Reads the ``_unsloth_loras`` marker, which the LoRA paths write as ``(name, path, weight)``.
    Shape is tolerated rather than assumed: this runs inside the generate result and an unpacking
    error here would sink a finished generation whose images are already in hand."""
    pairs = []
    for entry in getattr(pipe, "_unsloth_loras", ()) or ():
        try:
            if len(entry) == 3:
                name, _path, weight = entry
            elif len(entry) == 2:
                name, weight = entry
            else:
                continue
            weight = float(weight)
        except Exception:  # noqa: BLE001 — an unrecognised marker records no adapter
            continue
        if weight:
            pairs.append((name, weight))
    return pairs


def _baked_lora_names(pipe: Any) -> list:
    """Names of the adapters BAKED INTO ``pipe`` at load time, whatever their current scale.

    A torchao load attaches its adapters before ``quantize_`` + compile, so they are part of the
    BUILD, not of any one generation: peft rewraps each targeted Linear as ``lora.Linear`` and the
    quantiser then converts ``base_layer`` while the ``lora_`` side path stays high precision (see
    ``diffusion_transformer_quant``). Disabling them at generate time sets their scale to 0, which
    is not the same pipeline as one built without them -- so the recipe has to record the bake to
    describe the build that made the image. ``_active_lora_pairs`` deliberately drops the
    zero-weight entries, which is why the applied set cannot carry this.

    Shape is tolerated rather than assumed, for the same reason as ``_active_lora_pairs``: this
    runs inside the generate result and must never sink a finished generation."""
    if not getattr(pipe, "_unsloth_loras_baked", False):
        return []
    names = []
    for entry in getattr(pipe, "_unsloth_loras", ()) or ():
        try:
            name = entry[0]
        except Exception:  # noqa: BLE001 — an unrecognised marker records no adapter
            continue
        if name:
            names.append(str(name))
    return names


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
        # A PEFT adapter folder is not a base checkpoint; skip it so validation 400s before eviction.
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
    # Bound the decoded size: 4096px covers txt2img 2048, upscales and outpaint canvases.
    max_side = 4096
    try:
        img = Image.open(io.BytesIO(blob))
        # Reject from the header before img.load() so a huge-dimension file cannot spike memory.
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


# Official non-unsloth pipeline bases: safetensors-only, no remote code, exact lowercased match.
_TRUSTED_NON_GGUF_REPOS = frozenset(
    {
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        # Vendor safetensors-only bases: LoRA training bases + the BF16 artifact per group. FLUX.1 is Hub-gated; Qwen/Z-Image open.
        "black-forest-labs/flux.1-dev",
        "black-forest-labs/flux.1-schnell",
        "black-forest-labs/flux.1-kontext-dev",
        # Krea: guidance-distilled FLUX.1-dev finetune, same arch and gating; detected via "flux.1".
        "black-forest-labs/flux.1-krea-dev",
        # FLUX.2 LoRA training bases (dev gated, klein-4B open); reloaded as a pipeline by "Deploy to Create".
        "black-forest-labs/flux.2-dev",
        "black-forest-labs/flux.2-klein-4b",
        # The other klein sizes and variants. One family covers both sizes and defaults to 4B, on
        # the understanding that the real base comes from the base_model card tag. Leaving these
        # out made that tag untrusted, so a klein-9B GGUF silently resolved to the 4B config and
        # died in the GGUF quantizer with a bare shape mismatch (24576x4096 vs 18432x3072).
        "black-forest-labs/flux.2-klein-9b",
        "black-forest-labs/flux.2-klein-base-4b",
        "black-forest-labs/flux.2-klein-base-9b",
        "tongyi-mai/z-image-turbo",
        "qwen/qwen-image",
        "qwen/qwen-image-2512",
        "qwen/qwen-image-edit-2511",
        # Krea 2: assembled per-component. Turbo = inference; Raw = the LoRA training base.
        "krea/krea-2-turbo",
        "krea/krea-2-raw",
        # Lumina Image 2.0: standard diffusers layout, generic from_pretrained path.
        "alpha-vllm/lumina-image-2.0",
        # HunyuanImage 2.1: open community diffusers mirror, safetensors-only.
        "hunyuanvideo-community/hunyuanimage-2.1-diffusers",
        # HiDream-I1: open MIT weights, all three variants one family; Llama TE from the unsloth mirror.
        "hidream-ai/hidream-i1-full",
        "hidream-ai/hidream-i1-dev",
        "hidream-ai/hidream-i1-fast",
        # Ideogram 4: no bf16 ships. -fp8 stores both DiTs as raw float8 (the family base); both nf4 repos are identical.
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


def _repo_access_message(repo: str, *, gated: bool) -> str:
    """The repo id AND its licence page: the worker's 401/403 names neither, and the base comes from a
    card tag, so the user never saw which repo it is."""
    url = f"https://huggingface.co/{repo}"
    if gated:
        return (
            f"'{repo}' is gated on Hugging Face and this model cannot be downloaded without it. "
            f"Accept its licence at {url}, then add a Hugging Face token that has access in "
            "Studio settings and try again."
        )
    return (
        f"'{repo}' could not be read from Hugging Face (private, renamed or removed) and this "
        f"model cannot be downloaded without it. Check {url}, then add a Hugging Face token that "
        "has access in Studio settings and try again."
    )


def _assert_base_repo_accessible(
    base_repo: str,
    hf_token: Optional[str],
    probe_file: str = "model_index.json",
) -> Optional[str]:
    """Fail up front, with the licence URL, when a companion base cannot be read.

    The Hub gates the BYTE endpoint only, so ``model_info`` answers anonymously for gated repos and
    a plan built from it dies mid-download on a bare token error. Probes ``probe_file`` (fetched by
    the load anyway) only when ``gated`` is set, so an open repo costs one metadata call; the native
    plan passes its own asset name, as a repo it reads only for a VAE has no manifest cached. Fails
    open on any non-access error: offline/transient must not block a load.

    Returns the base's snapshot dir when an ACCESS verdict was excused by a copy living only under
    huggingface_hub's import-time root, so the caller can load off disk: ``from_pretrained`` is
    pinned to ``hub_cache_dir()`` and cannot see it, and the failure that earned the escape also
    empties the size estimate. None otherwise."""
    repo = (base_repo or "").strip()
    # Only a remote 'org/name' can be gated; a local base is already on disk.
    if not repo or repo.count("/") != 1:
        return None
    # Blank to None, as load_pipeline does: build_hf_headers sends "" as a literal "Bearer " that
    # 401s, which the handling below would read as an open base being unreadable.
    hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
    try:
        if Path(repo).expanduser().exists():
            return None
    # OSError: invalid path characters -> a remote id. RuntimeError: pathlib raises it, not OSError,
    # when expanduser() cannot resolve the home dir ('~other/models', or '~/models' with no HOME).
    # Both carry one slash, so they reach here, and an escape would 500 this fail-open probe.
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception:  # noqa: BLE001 — an unexpected hub layout leaves today's behaviour
        return None

    # Set when only the import-time root holds the excusing copy, which the pinned from_pretrained
    # would miss.
    other_root_snapshot: Optional[str] = None

    # A base already on disk needs no Hub access: hf_hub_download catches the gated/401 HEAD and
    # serves the cached pointer (file_download._get_metadata_or_catch_error), which is how a
    # downloaded gated base still loads once the token is cleared or expires. Excuses an ACCESS
    # verdict only, never a 404: a renamed or removed repo cannot be un-renamed by a stale copy.
    def _already_downloaded() -> bool:
        """True when ``probe_file`` is on disk under EITHER root (Studio pins its live setting, the
        prefetch writes under huggingface_hub's import-time constant). Never raises. Exact for the
        native plan, which probes an asset it stages; a proxy for the diffusers plan, which probes
        the manifest, so a manifest-cached base with missing shards still dies mid-download on the
        bare token error -- the pre-preflight behaviour, never a blocked load."""
        nonlocal other_root_snapshot
        try:
            from huggingface_hub import try_to_load_from_cache
            for root in (hub_cache_dir(), None):
                # Only a str is a cached path; a miss is None and an absent file is a sentinel.
                hit = try_to_load_from_cache(repo, probe_file, cache_dir = root)
                if isinstance(hit, str):
                    # The live root is tried first, so root None means only the import-time one has
                    # it. Its parent is the snapshot dir only for a top-level probe.
                    if root is None and "/" not in probe_file:
                        other_root_snapshot = str(Path(hit).parent)
                    return True
        except Exception:  # noqa: BLE001 — a cache we cannot read is not an access verdict
            pass
        return False

    def _is_auth_error(exc: Any) -> bool:
        """A 401/403 that hf_raise_for_status did not classify: an expired token 401s "Invalid
        credentials in Authorization header", which _http.py excludes from its RepoNotFound branch
        by name, and a token missing a permission 403s. Both arrive as plain HfHubHTTPError, so
        catching only the classified errors fails open on the very case this probe exists to catch."""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (401, 403)

    try:
        gated = getattr(HfApi().model_info(repo, token = hf_token), "gated", None)
    except GatedRepoError:  # a gated repo can also withhold its metadata
        if _already_downloaded():
            return other_root_snapshot
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except RepositoryNotFoundError as exc:
        # 401 and 404 both land here: hf_raise_for_status folds unauthenticated private/gated in
        # with a missing repo because "401 is misleading" (_http.py), so re-read the status. A
        # 401/403 earns the cache escape; a genuine 404, or an error carrying no response, raises.
        if _is_auth_error(exc) and _already_downloaded():
            return other_root_snapshot
        raise ValueError(_repo_access_message(repo, gated = False)) from None
    except HfHubHTTPError as exc:
        if not _is_auth_error(exc):
            return None  # a 5xx or rate limit is not an access verdict
        if _already_downloaded():
            return other_root_snapshot
        raise ValueError(_repo_access_message(repo, gated = False)) from None
    except Exception:  # noqa: BLE001 — offline / transient: the download surfaces any real error
        return None
    # Nothing to carry: model_info answered, so the size estimate lists the base files and the
    # prefetch resolves each through whichever root holds it.
    if not gated or _already_downloaded():
        return None
    try:
        # A metadata HEAD, never hf_hub_download: a cached manifest makes the download return the
        # cached pointer, so a stale token would pass the probe and 401 again mid-prefetch.
        get_hf_file_metadata(hf_hub_url(repo, probe_file), token = hf_token)
    except GatedRepoError:
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except HfHubHTTPError as exc:
        if not _is_auth_error(exc):
            return None
        raise ValueError(_repo_access_message(repo, gated = True)) from None
    except Exception:  # noqa: BLE001 — no manifest / offline / transient is not an access verdict
        return None
    return None


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
    # Defaulted so older positional constructions keep working.
    offload_policy: str = OFFLOAD_NONE
    vae_tiling: bool = False
    memory_mode: str = "auto"
    # Resolved load kind ("gguf"|"single_file"|"pipeline"); lets the UI gate GGUF-only controls.
    kind: str = "gguf"
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    # Torch backend flags (TF32 / cudnn.benchmark) captured before the speed layer mutated them.
    backend_flags_before: Optional[dict] = None
    # Text-encoder quant engaged: "fp8" | "nvfp4" | None.
    text_encoder_quant: Optional[str] = None
    # Transformer quant on the dense fast path ("int8"|"fp8"|"nvfp4"|"mxfp8"), or None when GGUF loaded.
    transformer_quant: Optional[str] = None
    # Attention backend via the diffusers dispatcher, or None for default SDPA.
    attention_backend: Optional[str] = None
    # Caller original attention request, so deferred engagement re-runs the same selection.
    attention_request: Optional[str] = None
    # Step cache engaged ("fbcache") or None. Opt-in, for many-step models.
    transformer_cache: Optional[str] = None
    # AUTO: generate() toggles FBCache across FBCACHE_MIN_STEPS; an explicit request never toggles.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Shared eager patches installed for this load; uninstalled on unload.
    eager_patched: bool = False
    # Deferred speed auto: the load stays eager, generate() engages `default` at the 3rd generation.
    speed_deferred: bool = False
    # Successful generations on this load; drives the deferred engagement above.
    generation_count: int = 0
    # Pre-warmed torch.compile cache context when a compiled tier ran, else None.
    compile_cache_ctx: Any = None
    # Token kept so LoRA adapters selected at generate time can be fetched.
    hf_token: Optional[str] = None
    # Per-control provenance {control: {value, source, reason}}, for status badges.
    resolved: Optional[dict] = None
    # The single-file checkpoint basename this load committed (None for a pipeline). Part of the build identity.
    gguf_filename: Optional[str] = None


@dataclass
class _LoadingState:
    """An in-flight background load, polled for download progress."""

    repo_id: str
    base_repo: str
    expected_bytes: int = 0
    error: Optional[str] = None
    # Where the companion BYTES land: ``base_repo``, or its mirror when one was swapped in. Never
    # surfaced (``base_repo`` stays the id status() reports), but the cache scan and the delete
    # guard must look here.
    fetch_repo: Optional[str] = None


@dataclass
class _GenState:
    """An in-flight generation, updated per denoising step for the progress bar."""

    total_steps: int
    step: int = 0
    # Set when the first step finishes, so the slower warmup step does not skew the ETA rate.
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


def _install_gguf_prefix_strip(transformer_cls: Any, logger: Any) -> None:
    """Wrap the class's diffusers single-file converter to strip the
    ``model.diffusion_model.`` container prefix that sd.cpp-converted GGUFs
    carry on every tensor.

    diffusers (<= 0.39) handles the prefix inconsistently: the FLUX.1 converter
    strips it natively, but the FLUX.2 converter never does and KeyErrors on the
    prefixed keys (``'double_blocks.0.img_attn.norm.key_norm'`` misparse in
    ``convert_flux2_transformer_checkpoint_to_diffusers``), and the Qwen-Image
    mapping fn is an identity, so every prefixed tensor is reported "not used",
    the model stays on meta, and ``.to(cuda)`` dies with "Cannot copy out of
    meta tensor". Stripping the prefix when present is a no-op for already-clean
    checkpoints, so the shim applies to every GGUF transformer class uniformly.
    Idempotent (the wrapper is marked) and best-effort."""
    try:
        from diffusers.loaders import single_file_model as sfm

        entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(
            getattr(transformer_cls, "__name__", str(transformer_cls))
        )
        if not isinstance(entry, dict):
            return
        original = entry.get("checkpoint_mapping_fn")
        if not callable(original) or getattr(original, "_unsloth_prefix_strip", False):
            return
        prefix = "model.diffusion_model."

        def _stripped_mapping_fn(checkpoint = None, **kwargs):
            checkpoint = {
                (key[len(prefix) :] if key.startswith(prefix) else key): value
                for key, value in (checkpoint or {}).items()
            }
            return original(checkpoint = checkpoint, **kwargs)

        _stripped_mapping_fn._unsloth_prefix_strip = True
        entry["checkpoint_mapping_fn"] = _stripped_mapping_fn
    except Exception as exc:  # noqa: BLE001 — loader-compat shim only, never fail the load
        logger.warning("diffusion.gguf: prefix-strip shim not installed: %s", exc)


_NO_WEIGHT = object()


def _has_active_lora(loras: Any) -> bool:
    """True when any adapter would actually be baked, for either shape the callers pass.

    Weight 0 means disabled, but /images/load passes ``(id, weight)`` tuples while
    /images/download-plan passes ``LoraSpec`` models, whose unpacking yields ``(field, value)``
    pairs -- so ``(_lid, w)`` would bind ``w`` to ``("weight", 0.0)`` and read a disabled adapter
    as active. Reading the attribute first covers both."""
    for entry in loras or ():
        weight = getattr(entry, "weight", _NO_WEIGHT)
        if weight is _NO_WEIGHT:
            try:
                _lid, weight = entry
            except (TypeError, ValueError):  # an unrecognised shape is not a disabled adapter
                return True
        try:
            if float(weight) != 0:
                return True
        except (TypeError, ValueError):  # likewise: an unreadable weight must not skip the bake
            return True
    return False


def _uncached_prequant_repo(
    fam: Optional[DiffusionFamily],
    target: Any,
    requested: Optional[str],
    *,
    base_repo: Optional[str],
    prequant_path: Optional[str],
) -> Optional[str]:
    """The hosted pre-quant repo an AUTO-derived quant would have to DOWNLOAD for this pick, or None
    when it costs no extra bytes (no hosted source, a local override, or already cached).

    Otherwise an auto GGUF pick fetches the GGUF and then a second multi-GB denoiser it uses
    instead. Shared by ``load_pipeline`` and ``_dense_quant_prefetch_needed`` so the load and the
    download plan decline together. Cheap (a refs read + stat) and never raises."""
    try:
        scheme = select_transformer_quant_scheme(
            target, requested, family = getattr(fam, "name", None)
        )
        if scheme is None:
            return None
        source = usable_prequant_source(
            fam, scheme, path_override = prequant_path, base_repo = base_repo
        )
        # A local override is the operator's own file, so it never downloads.
        if source is None or source.kind != "repo":
            return None
        if prequant_checkpoint_cached(source, cache_dir = hub_cache_dir()):
            return None
        return source.location
    except Exception:  # noqa: BLE001 — a probe that cannot answer keeps the prequant shortcut
        return None


class DiffusionBackend:
    """Holds at most one loaded diffusers pipeline. All mutations are serialised."""

    def __init__(self) -> None:
        # _lock serialises the small state mutations; the status/progress readers stay lock-free.
        self._lock = threading.Lock()
        # _generate_lock serialises generations and is the ONLY lock the denoise holds.
        self._generate_lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None
        # Bumped on begin_load/unload so a superseded worker neither commits nor stamps progress.
        self._load_token = 0
        # Set by unload() to abort an in-flight download. Replaced, never cleared, so a cancelled worker stays cancelled.
        self._cancel_event = threading.Event()
        # Cancel Event of the in-flight generation; per-generation so a cancel can't be lost or leak.
        self._active_generate_cancel: Optional[threading.Event] = None
        # Unloads / superseding loads waiting on _generate_lock to free this pipeline. A generation queued behind the active
        # one holds no cancel event yet, so without this fence it could win the lock after an eject and denoise anyway. A
        # count, not a flag, so concurrent teardowns each own their own release.
        self._teardown_waiters = 0
        # Written by the callback, read lock-free by generate_progress().
        self._gen: Optional[_GenState] = None
        # img2img/inpaint pipes built via from_pipe (shared modules, no extra VRAM); cleared on unload.
        self._aux_pipes: dict[str, Any] = {}
        # Loaded ControlNets and their from_pipe pipelines, reusing resident modules; cleared on unload.
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
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        # Pin the LIVE root, as the prefetch does: unpinned, a mid-session cache change misses the
        # staged GGUF and re-pulls the whole multi-GB file inside the load lock, after eviction,
        # where unload cannot preempt it and progress already reported 100%.
        cache_dir = hub_cache_dir()
        # Best-effort: an unreadable cache raises here, and letting that escape would abort a load
        # that only had to download. Any failure falls through to the pinned download below.
        try:
            if not isinstance(
                try_to_load_from_cache(repo_id, gguf_filename, cache_dir = cache_dir), str
            ):
                # Same other-root reuse as the pre-quant checkpoint, reached THROUGH that root
                # rather than returned raw, so the ref still resolves and a republished GGUF is
                # picked up; the blob is reused, and offline the cached pointer comes back anyway.
                elsewhere = try_to_load_from_cache(repo_id, gguf_filename, cache_dir = None)
                if isinstance(elsewhere, str) and Path(elsewhere).is_file():
                    try:
                        return hf_hub_download(
                            repo_id, gguf_filename, token = hf_token, cache_dir = None
                        )
                    except Exception:  # noqa: BLE001 — revalidation is a bonus, never a new failure
                        return elsewhere
        except Exception:  # noqa: BLE001 — an unreadable cache is not a verdict, just download
            pass
        return hf_hub_download(repo_id, gguf_filename, token = hf_token, cache_dir = cache_dir)

    def _dense_quant_prefetch_needed(self, fam: DiffusionFamily, kwargs: dict) -> bool:
        """True when ``load_pipeline`` may take the dense transformer-quant path, so
        the prefetch should also pull the base repo's ``transformer/`` shards.

        Those shards are excluded from the prefetch by default (the GGUF supplies
        the transformer), but ``_load_dense_quant_pipeline`` fetches them with
        ``from_pretrained(subfolder = "transformer")`` under the load lock during
        "finalizing", after the previous pipeline was already evicted, where
        unload/cancellation cannot preempt the download. Mirrors the dense-path
        gates in ``load_pipeline``: quant requested and supported for this device,
        and no pre-quantized checkpoint that would shortcut the dense build. Callers only ask for
        a ``kind == "gguf"`` pick, so the auto-quant decline below applies as written."""
        raw = kwargs.get("transformer_quant")
        # Unset defaults to the hardware ladder (mirrors load_pipeline's tri-state).
        auto = raw is None or str(raw).strip().lower() in ("", "auto")
        if auto:
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
            # A definite-offload policy skips the dense build, so widening wastes a multi-GB pull with no GGUF fallback.
            mm = normalize_memory_mode(kwargs.get("memory_mode"))
            if mm in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM):
                return False
            if mm is None and kwargs.get("cpu_offload"):
                return False
            target = self._resolve_device_target(fam)
            # Same decline as load_pipeline: an uncached hosted pre-quant keeps the GGUF, so the
            # plan must not stage the base transformer/ shards either.
            if (
                auto
                # Active weights only: disagreeing with the load stages shards it never reads.
                and not _has_active_lora(kwargs.get("loras"))
                and _uncached_prequant_repo(
                    fam,
                    target,
                    mode,
                    base_repo = kwargs.get("base_repo"),
                    prequant_path = kwargs.get("transformer_prequant_path"),
                )
                is not None
            ):
                return False
            # Only widen when the loader would take the dense path; same candidate load_pipeline re-plans against.
            candidate = resolve_dense_quant_candidate(
                fam = fam,
                target = target,
                requested = mode,
                base_repo = kwargs.get("base_repo"),
                prequant_path = kwargs.get("transformer_prequant_path"),
                # An all-zero list bakes nothing; sizing it dense stages shards the load never reads.
                force_dense = _has_active_lora(kwargs.get("loras")),
                logger = None,
            )
            # A prequant loads a small checkpoint, so widening defeats the savings and can disk-full.
            if candidate is None or candidate.prequant:
                return False
            # Capacity gate: mirror plan_fits_total_capacity against TOTAL capacity, else load_pipeline declines the dense path anyway.
            from .diffusion_memory import (
                _reserve_mib,
                snapshot_device_memory,
            )

            memory = snapshot_device_memory(target)
            total = memory.total_mib
            steady = getattr(candidate, "steady_total_mib", None)
            if total is not None and steady is not None:
                budget = int((int(total) - _reserve_mib(memory.memory_kind, int(total))) * 0.85)
                if int(steady) > budget:
                    return False
            return True
        except Exception:  # noqa: BLE001 — widening the prefetch is best-effort only
            return False

    def _prefetch_files(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        base_files: list[str],
        hf_token: Optional[str],
        cancel_event: Optional[threading.Event] = None,
        fetch_base: Optional[str] = None,
    ) -> Optional[str]:
        """Pre-download the GGUF + the given ``base_files`` into the HF cache,
        WITHOUT the lock and honoring ``cancel_event`` (this load's own event, so a
        replacement load cannot un-cancel this one), so load_pipeline's
        from_single_file / from_pretrained hit the cache and the heavy download can
        be preempted by an unload/eviction. Raises ``RuntimeError("Cancelled")``.

        Returns the base repo's local snapshot dir when the prefetched set includes
        the pipeline manifest, so from_pretrained can load from disk instead of
        re-sweeping the hub (its own sweep also pulls files the scoped list skips,
        e.g. the 24 GB packaged root singles in each FLUX.1 repo); None otherwise
        (estimate failure, config-only base, local repo) -> hub id as before."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Only the BYTES move; the caller keeps the upstream id. ``fetch_base`` is the load-wide
        # decision when one was taken, so every later fetch agrees with what was staged here.
        base = fetch_base or prefer_ungated_mirror(base, hf_token, files = base_files)
        # Callers without a per-load event (tests, direct use) fall back to the current one.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        # A file cached only under huggingface_hub's import-time root resolves through that root:
        # the preflight clears a base found under EITHER root, so without this a cache-folder change
        # turns an already-downloaded gated base into a mid-prefetch Hub token error, and every
        # other moved asset into a needless re-download.
        # GGUF transformer (hub repos only; a local path is already on disk).
        if gguf_filename and not Path(repo_id).expanduser().exists():
            hf_hub_download_with_xet_fallback(
                repo_id,
                gguf_filename,
                hf_token,
                cancel_event = cancel,
                reuse_other_cache_root = True,
            )
        # Base repo (VAE / text-encoder / scheduler); list comes from the estimate.
        snapshot_root: Optional[str] = None
        # reuse_other_cache_root resolves EACH file through whichever root holds it, so a moved
        # cache can serve the manifest from the old root while companions land in the live one, and
        # returning that snapshot would point from_pretrained at a tree missing the rest.
        roots: set[str] = set()
        for rfilename in base_files:
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            local = hf_hub_download_with_xet_fallback(
                base, rfilename, hf_token, cancel_event = cancel, reuse_other_cache_root = True
            )
            # The resolved path minus the file's own relative path, so a subfolder entry yields the
            # same root as a top-level one. Not resolve()d: that follows the symlink into blobs/.
            try:
                root = str(Path(local).parents[len(Path(rfilename).parts) - 1])
            except (IndexError, ValueError, OSError):
                root = ""
            roots.add(root)
            if rfilename == "model_index.json":
                snapshot_root = root or None
        # Only hand back a snapshot the whole set lives in; otherwise the hub id resolves each file
        # through its own root as before.
        if snapshot_root is not None and roots != {snapshot_root}:
            return None
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
            # An excluded model gets its stated reason, not the unknown-family message that invites a doomed retry.
            excluded = excluded_model_reason(repo_id)
            if excluded:
                raise ValueError(f"'{repo_id}' cannot be loaded: {excluded}")
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name. (Video models and image models "
                f"whose diffusers transformer has no single-file loader are not supported.)"
            )
        # Refuse a too-old diffusers here, not deep in the load, but only when this load builds the diffusers pipeline: a
        # GGUF this host routes to native sd.cpp never instantiates the class. The picker gate reads the same predicate.
        # Imported here, not at module import, because the router imports this module's siblings.
        from .diffusion_engine_router import family_buildable_here

        if not family_buildable_here(fam, model_kind = kind):
            assert_pipeline_class_available(fam.pipeline_class, fam.name)
        # Families whose single file IS the whole pipeline have no GGUF path; reject before eviction.
        if kind == "gguf" and fam.single_file_is_pipeline:
            raise ValueError(
                f"'{fam.name}' checkpoints are whole-pipeline single files and have no GGUF "
                f"transformer variant; load the .safetensors pipeline instead of a GGUF."
            )
        # A multi-denoiser family (Ideogram 4) has no transformer-only path; reject before eviction.
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
        # The companion base repo also loads via from_pretrained, so it must clear the same trust bar.
        if base_repo and base_repo.strip() and not _is_trusted_diffusion_repo(base_repo):
            raise ValueError(
                f"base_repo is restricted to unsloth/* repos (or a local path); got "
                f"'{base_repo}'."
            )
        # A local base_repo loads as a full pipeline; reject a non-pipeline one before eviction.
        _assert_local_base_is_pipeline(base_repo)
        # Reject a bad LOCAL pick before the route evicts chat: a path-shaped repo_id must be on disk.
        local_root = Path(repo_id).expanduser()
        # Path-shaped: "."/".." prefix, a backslash (never in "org/name"), or an absolute path.
        path_shaped = (
            repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or local_root.is_absolute()
        )
        if kind in ("gguf", "single_file"):
            if not gguf_filename:
                raise ValueError(f"a single-file checkpoint name is required for a '{kind}' load.")
            # Fail a kind/extension mismatch before the handoff: gguf needs .gguf, single_file must not.
            is_gguf_name = gguf_filename.lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            # A single-file load must name a real .safetensors, else it evicts chat then fails in background.
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
                # A remote "*-GGUF" id is not a pipeline; reject here instead of evicting chat then failing.
                raise ValueError(
                    f"'{repo_id}' is a single-file GGUF repo; load it with model_kind 'gguf' "
                    f"and a .gguf filename, not as a full pipeline."
                )
        return fam

    def preflight_base_access(
        self,
        repo_id: str,
        fam: Optional[DiffusionFamily],
        *,
        gguf_filename: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        """The gated/unreadable-base refusal, run by the route BEFORE it takes the GPU.

        ``_run_load`` and ``download_plan`` already make it, but ``_run_load`` runs on the load
        thread, after the route evicted chat, and the plan's 400 does not stop the load: the images
        page falls back to /images/load on ANY plan failure. Resolves the base exactly as those two
        do, so all three agree. The one deliberate network step on the pre-eviction path
        (``validate_load_request`` stays network-free): two metadata calls for a remote pick, none
        for a local one, both already made by ``_run_load`` moments later. Fails open on
        offline/transient, so it can refuse a load but never block one that would have worked."""
        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind == "pipeline":
            base = repo_id  # the full pipeline IS the repo
        else:
            base = _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        # Probe the repo the load will FETCH from: refusing the upstream id would reject the gated
        # picks the ungated mirror rescues, and the swap is pure, so it decides the same here as on
        # the load thread. Only the raise matters; _run_load recomputes the excused snapshot.
        _assert_base_repo_accessible(prefer_ungated_mirror(base, hf_token), hf_token)

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
        loras: Optional[list[tuple[str, float]]] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        # A blank token must mean "anonymous", not an empty credential the Hub 401s.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        # base_repo is gated at the route pre-eviction; this only cheap-fails the resolved repo/family.
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
            # A NEW event per load, never a clear() of the shared one: unload() sets the event the running worker holds but also
            # drops _loading, so clearing here would un-cancel that worker. Download preemption is best-effort; the token is the
            # real commit guard.
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
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
                transformer_quant = transformer_quant,
                transformer_quant_fast_accum = transformer_quant_fast_accum,
                transformer_prequant_path = transformer_prequant_path,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                model_kind = model_kind,
                loras = loras,
                _load_token = token,
                _cancel_event = cancel_event,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = kwargs.pop("_cancel_event", None) or self._cancel_event
        try:
            # Resolve the base repo and estimate sizes here (both network) so begin_load returns instantly.
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
            # The pre-cast encoder replaces these weights, so skip their dense shards. Same resolver as the injection.
            te_prequant_files = self._te_prequant_plan_files(
                fam, kwargs.get("text_encoder_quant"), kwargs.get("hf_token")
            )
            expected, base_files = self._estimate_download_bytes(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                kwargs.get("hf_token"),
                kind = kind,
                single_file_is_pipeline = bool(fam and fam.single_file_is_pipeline),
                # Pull the base shards here rather than inside the locked, unpreemptable finalize.
                include_transformer = kind == "gguf"
                and self._dense_quant_prefetch_needed(fam, kwargs),
                skip_te_components = tuple(te_prequant_files),
            )
            # Only shards this prefetch staged may be materialised by the dense fallback, so read it off the staged list: a failed size estimate drops every base file too.
            kwargs["_transformer_prefetched"] = any(
                f.startswith("transformer/") for f in base_files
            )
            # ONE mirror decision per load, taken with the staged file list in hand and carried into
            # load_pipeline: per-call-site, one repo could be staged and the other assembled from.
            fetch_base = prefer_ungated_mirror(base, kwargs.get("hf_token"), files = base_files)
            kwargs["_fetch_base"] = fetch_base
            # Same preflight the plan runs: catch a gated base here, not 15 GiB into the prefetch.
            # Runs on ``fetch_base``, once it is decided, so it probes the repo the pull will read:
            # refusing the upstream id would reject the gated picks the ungated mirror rescues.
            # Everything above is metadata, so no byte has moved yet. Returns a snapshot dir when it
            # excused the base off a copy only the import-time root holds.
            base_snapshot = _assert_base_repo_accessible(fetch_base, kwargs.get("hf_token"))
            with self._lock:
                # Stamp progress only if this load is still current (a superseder has its own token).
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.fetch_repo = fetch_base
                    self._loading.expected_bytes = expected
            # Download outside the lock so unload/an eviction can preempt the pull.
            # The carried snapshot is the fallback, never the override: it fires only when the
            # estimate came back empty, since the metadata that fills it is the same call whose
            # failure earned the escape. Without it the load 401s with every byte already on disk.
            kwargs["_base_local_dir"] = (
                self._prefetch_files(
                    kwargs["repo_id"],
                    kwargs.get("gguf_filename"),
                    base,
                    base_files,
                    kwargs.get("hf_token"),
                    cancel_event = cancel_event,
                    fetch_base = fetch_base,
                )
                or base_snapshot
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
            # Free the debris of a failed construction; guarded so a sticky CUDA error still stamps the error.
            try:
                clear_gpu_cache()
            except Exception:  # noqa: BLE001
                pass
            # Rewrite a gated-repo 403 into the step that unblocks the user, then redact native paths:
            # this text is surfaced verbatim and Studio can be shared. Guarded because on this daemon
            # thread anything escaping leaves _loading.error unset and load_progress() stuck forever.
            from utils.native_path_leases import redact_native_paths

            try:
                text = hub_access_message(
                    exc, had_token = _hf_token_in_play(kwargs.get("hf_token"))
                ) or str(exc)
            except Exception:  # noqa: BLE001
                text = str(exc)
            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(text)

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)

        # Sum checkpoint + companion cache, scanning the repo the bytes LAND in (the mirror when one
        # was swapped in), else a mirrored companion download reads as zero and the bar sits still.
        companion = loading.fetch_repo or loading.base_repo
        if loading.base_repo and loading.base_repo == loading.repo_id:
            # Full pipeline: base IS the repo, so count it once. Summing would add the upstream's
            # stale partial blobs -- the very thing that selects the mirror -- to the mirror's live
            # bytes, pushing the bar to 100% / finalizing mid-download.
            downloaded = self._cache_bytes(companion or loading.repo_id)
        else:
            downloaded = self._cache_bytes(loading.repo_id)
            if companion and companion != loading.repo_id:
                downloaded += self._cache_bytes(companion)
        expected = loading.expected_bytes
        # Downloads done, still finalizing. The cache scan can exceed the estimate, so clamp to 100%.
        if expected > 0 and downloaded >= expected * 0.999:
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        if expected <= 0:
            # No size estimate: report the phase with no byte claim, since `downloaded` scans what is PRESENT.
            return _progress("downloading")
        return _progress("downloading", downloaded, expected, min(downloaded / expected, 1.0))

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base) would yank
        blobs and snapshot files from under the download/assembly. Includes the mirror
        when one was swapped in: that is where the companion bytes land, so guarding only
        the upstream id would leave the live download deletable."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (loading.repo_id, loading.base_repo, loading.fetch_repo)
            return tuple(dict.fromkeys(r for r in ids if r))

    @staticmethod
    def _te_prequant_plan_files(
        fam: Any, text_encoder_quant: Optional[str], hf_token: Optional[str]
    ) -> dict[str, tuple[str, list[tuple[str, int]]]]:
        """``{component: (repo_id, [(rfilename, size)])}`` for the text encoders this pick will
        take PRE-CAST from a hosted checkpoint instead of the base repo's dense weights.

        Empty unless the request asked for a scheme with a hosted artifact AND that artifact
        really resolves, so a plan can never drop a dense encoder the load still wants."""
        try:
            from huggingface_hub import HfApi

            from .diffusion_te_prequant import te_prequant_hub_files, te_prequant_sources

            sources = te_prequant_sources(
                fam,
                te_quant_mode = text_encoder_quant,
                target = resolve_diffusion_device_target(),
            )
            if not sources:
                return {}
            files = te_prequant_hub_files(sources, HfApi(token = hf_token or None), logger)
            return {c: (sources[c].location, f) for c, f in files.items()}
        except Exception as exc:  # noqa: BLE001 -- an unresolvable pre-cast keeps the dense encoder
            logger.warning("diffusion.te_prequant_plan_failed: %s", exc)
            return {}

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
        sizes_out: Optional[dict[str, int]] = None,
        file_sizes_out: Optional[dict[tuple[str, str], int]] = None,
        shas_out: Optional[dict[str, str]] = None,
        skip_te_components: tuple[str, ...] = (),
    ) -> tuple[int, list[str]]:
        """Total download size for the progress bar, plus the base-repo files to
        fetch (the prefetch reuses this list, so the base is listed only once).

        ``sizes_out``, when given, is filled with per-repo byte totals so the download
        plan can size one job per repo off this same single pair of Hub lookups.
        ``file_sizes_out`` is the per (repo, filename) breakdown, so the plan can drop cached
        files and size the rest. ``shas_out`` is each repo's CURRENT commit, so a cache hit
        counts only in that revision.

        For a ``pipeline`` load the whole repo IS the pipeline (``base_repo`` is the
        repo itself), so the transformer/ subfolder is INCLUDED -- unlike the GGUF /
        single-file paths, where the transformer is the single file and the base repo
        supplies only the companions. For a ``single_file_is_pipeline`` family (SDXL) the
        single file is the WHOLE pipeline, so the base repo supplies only config/tokenizer
        (no weights) and its weight files are skipped.

        ``skip_te_components`` names the text encoders this pick loads PRE-CAST from a hosted
        checkpoint, so their dense weight shards are not counted or fetched: staging the dense
        encoder for a pre-cast load wastes tens of GB (FLUX.2-dev's Mistral-24B is ~48 GB,
        Qwen-Image's Qwen2.5-VL ~16.6 GB) and nothing ever opens them. Everything else in the
        component folder (config, shard index, tokenizer) is kept -- the pre-cast loader
        meta-inits the encoder from the base repo's config."""
        from huggingface_hub import HfApi

        # No swap on purpose: the Hub gates only the BYTE endpoint, so model_info answers
        # anonymously and the mirror lists the same names. _prefetch_files swaps when it pulls.
        from .diffusion_te_prequant import is_prequant_covered_weight

        api = HfApi()
        total = 0
        base_files: list[str] = []

        def _dense_te_shard(rfilename: str) -> bool:
            return bool(skip_te_components) and is_prequant_covered_weight(
                rfilename, skip_te_components
            )

        try:
            if kind == "pipeline":
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                _record_sha(shas_out, repo_id, info)
                picked = [
                    s
                    for s in info.siblings
                    if _pipeline_file_downloaded(s.rfilename) and not _dense_te_shard(s.rfilename)
                ]
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
                    if file_sizes_out is not None:
                        file_sizes_out[(repo_id, s.rfilename)] = s.size or 0
                if sizes_out is not None:
                    sizes_out[repo_id] = total
                return total, base_files
            # Skip the Hub size lookup for a LOCAL gguf path: model_info raises on a filesystem path.
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                _record_sha(shas_out, repo_id, info)
                gguf_bytes = sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
                total += gguf_bytes
                if sizes_out is not None:
                    sizes_out[repo_id] = gguf_bytes
                if file_sizes_out is not None:
                    file_sizes_out[(repo_id, gguf_filename)] = gguf_bytes
            # A whole-pipeline single file (SDXL) needs only the base's config/tokenizer, not its weights.
            if kind == "single_file" and single_file_is_pipeline:
                base_filter = _base_config_file_downloaded
            else:

                def base_filter(rfilename: str) -> bool:
                    return _base_file_downloaded(rfilename, include_transformer = include_transformer)

            base_info = api.model_info(base_repo, files_metadata = True, token = hf_token)
            _record_sha(shas_out, base_repo, base_info)
            base_bytes = 0
            for s in base_info.siblings:
                if base_filter(s.rfilename) and not _dense_te_shard(s.rfilename):
                    base_files.append(s.rfilename)
                    base_bytes += s.size or 0
                    if file_sizes_out is not None:
                        file_sizes_out[(base_repo, s.rfilename)] = s.size or 0
            total += base_bytes
            if sizes_out is not None:
                sizes_out[base_repo] = base_bytes
        except Exception as exc:  # noqa: BLE001 — estimate is best-effort
            logger.warning("diffusion.size_estimate_failed: %s", exc)
        return total, base_files

    @staticmethod
    def _files_already_cached(
        repo_id: str,
        files: list[str],
        revision: Optional[str] = None,
    ) -> set[str]:
        """Of ``files``, the names ONE root serves whole at ``revision``: all of them, or none.

        Both roots are searched -- Studio's live setting, and ``cache_dir = None`` for
        huggingface_hub's constant -- but hits are never UNIONED. A set split between the roots is
        complete in neither: ``_prefetch_files`` hands back a snapshot only when every file came
        from the SAME root, else None, and ``from_pretrained`` is then pinned to ``hub_cache_dir()``
        and cannot see the other root, so dropping the repo moves its refetch inline, outside the
        panel's progress and disk preflight. One root holding all of it still drops.

        ``revision`` is REQUIRED to drop anything: defaulted, ``try_to_load_from_cache`` reads the
        cache's OWN ``refs/main``, which a republished repo leaves on the superseded commit, so the
        stale blob would leave the plan and the loader would pull the new one inline.

        Only a str is a cached path. Never raises: an unreadable cache means "stage it"."""
        if not revision or not files:
            return set()
        try:
            from huggingface_hub import try_to_load_from_cache

            # Once, inside the try: hub_cache_dir is a SQLite read plus Path.home(), which raises
            # with no home -- per file and outside, it was the one way this could 500.
            roots = (hub_cache_dir(), None)
        except Exception:  # noqa: BLE001 — no hub package / unreadable settings: stage everything
            return set()
        wanted = set(files)
        live_root, fallback_root = roots

        def _hits(root: Optional[str]) -> set[str]:
            found: set[str] = set()
            for name in wanted:
                try:
                    hit = try_to_load_from_cache(repo_id, name, cache_dir = root, revision = revision)
                except Exception:  # noqa: BLE001 — a cache we cannot read is not a verdict
                    continue
                # is_file() is belt and braces: hub already ends on os.path.isfile today.
                if isinstance(hit, str) and Path(hit).is_file():
                    found.add(name)
            return found

        live = _hits(live_root)
        if live == wanted:
            return wanted
        # A partial live hit IS the split: no single-root snapshot, so the pinned load refetches
        # the fallback's share.
        if live:
            return set()
        return wanted if _hits(fallback_root) == wanted else set()

    @staticmethod
    def _current_sha(repo_id: str, hf_token: Optional[str]) -> Optional[str]:
        """``repo_id``'s current commit, or None when the Hub does not say.

        Only for the MIRROR: it is a separate repo with its own history, so the vendor sha
        ``_estimate_download_bytes`` recorded would never hit."""
        try:
            from huggingface_hub import HfApi
            return getattr(HfApi().model_info(repo_id, token = hf_token), "sha", None) or None
        except Exception:  # noqa: BLE001 — no sha means no skip, never a failed plan
            return None

    def download_plan(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        hf_token: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        **load_kwargs: Any,
    ) -> dict[str, Any]:
        """The repos + exact files this pick needs, so the Hub download manager can fetch
        them with the same file scope the loader would.

        A plain snapshot_download would also pull what the loader deliberately skips (the
        packaged root single, transformer/ shards, fp16 twins) -- tens of GB per FLUX repo.
        Resolves family/kind/base exactly as ``_run_load`` does, so the plan and the load
        agree. Local paths are already on disk and yield no entries.

        ``text_encoder_quant`` is read for the same reason as the DiT quant: an fp8 request
        loads a hosted PRE-CAST encoder, so the base repo's dense encoder shards must not be
        staged and the pre-cast checkpoint must be. Without it the manager stages the dense
        encoder (tens of GB the load never opens) and the load then pulls the pre-cast file
        inline, outside the manager's progress and disk preflight."""
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        kind = resolve_model_kind(gguf_filename, model_kind)
        if kind == "pipeline":
            base = repo_id  # the full pipeline IS the repo
        else:
            base = _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        # Only a checkpoint that really resolves on the Hub earns the right to drop dense shards.
        te_files = self._te_prequant_plan_files(fam, text_encoder_quant, hf_token)
        sizes: dict[str, int] = {}
        file_sizes: dict[tuple[str, str], int] = {}
        shas: dict[str, str] = {}
        # Total discarded: it counts every file the pick needs, not what is left to fetch.
        _estimate_total, base_files = self._estimate_download_bytes(
            repo_id,
            gguf_filename,
            base,
            hf_token,
            kind = kind,
            single_file_is_pipeline = bool(fam and fam.single_file_is_pipeline),
            # The RESOLVED base, as the load passes it: a variant base picks its own pre-quant repo.
            include_transformer = kind == "gguf"
            and self._dense_quant_prefetch_needed(fam, {**load_kwargs, "base_repo": base}),
            sizes_out = sizes,
            file_sizes_out = file_sizes,
            shas_out = shas,
            skip_te_components = tuple(te_files),
        )
        # Decided once, from the staged file list, and both probed and reported: a gated base
        # answers model_info anonymously, so the plan would otherwise be confident and the 401 land
        # mid-download. Probing any id but the one staged would refuse a load that works. The route
        # maps ValueError -> 400. Nothing above downloads, so the reorder costs nothing.
        fetch_base = prefer_ungated_mirror(base, hf_token, files = base_files)
        _assert_base_repo_accessible(fetch_base, hf_token)
        entries: list[dict[str, Any]] = []

        def _stage(
            repo: str,
            files: list[str],
            sized: dict[str, int],
            gguf: Optional[str],
            revision: Optional[str] = None,
        ) -> None:
            """Queue the files the cache is missing at ``revision``.

            Entries are what the Downloads panel lists, so an all-cached pick yields none rather
            than reading as a re-download of a model the user already has. No revision drops
            nothing, so an unpinnable repo stages in full as it always did.

            All or nothing per repo, never a subset: every diffusion entry for a repo rides the one
            "@diffusion" scope slot, and download_registry refuses a claim whose scoped_files differ
            from the live job's, so a shrinking list would 409 a second pick sharing this base where
            it used to adopt the running job. Staging a cached file costs nothing anyway, since
            hf_hub_download returns the pointer without a transfer.

            "Cached" is per ROOT, never a union across the two; see ``_files_already_cached``."""
            cached = self._files_already_cached(repo, files, revision)
            if files and cached.issuperset(files):
                return
            entries.append(
                {
                    "repo_id": repo,
                    "files": files,
                    "bytes": int(sum(sized.get(name, 0) for name in files)),
                    "gguf_filename": gguf,
                }
            )

        for repo, files in te_files.values():
            # No revision: te_prequant_hub_files reports names and sizes only, so a pre-cast
            # checkpoint stages whole, not off a stale refs/main. Worth surfacing its info.sha:
            # these replace tens of GB.
            _stage(repo, [n for n, _s in files], dict(files), None)
        if gguf_filename and not Path(repo_id).expanduser().exists():
            _stage(
                repo_id,
                [gguf_filename],
                # file_sizes, not sizes: a base repo equal to this one overwrote sizes[repo_id]
                # with the base total, mis-sizing this row.
                {gguf_filename: int(file_sizes.get((repo_id, gguf_filename), 0))},
                gguf_filename,
                shas.get(repo_id),
            )
        if base_files and not Path(base).expanduser().exists():
            # STAGED before the loader runs, so it must name the MIRROR: a gated upstream here 401s
            # an anonymous user at staging and the swap downstream is never reached. status(), the
            # API base repo, saved configs and LoRA tags keep the vendor id; sizes key on it too.
            # The commit follows the id: the sizes came from the vendor, the probe must not.
            base_rev = (
                shas.get(base) if fetch_base == base else self._current_sha(fetch_base, hf_token)
            )
            _stage(
                fetch_base,
                base_files,
                {name: file_sizes.get((base, name), 0) for name in base_files},
                None,
                base_rev,
            )
        # Derived, never decremented: a carried total can only drift from the rows the panel shows.
        return {
            "entries": entries,
            "total_bytes": int(sum(e["bytes"] for e in entries)),
        }

    @staticmethod
    def _hub_cache_repo_dir(repo_id: str) -> Path:
        """Local HF hub cache dir for ``repo_id``.

        Reads the live setting, not huggingface_hub's import-time constant: changing the
        cache folder does not update the constant, so the old one would count bytes in a
        root the download no longer writes to (progress stuck at 0 for the whole pull)."""
        return Path(hub_cache_dir()) / f"models--{repo_id.replace('/', '--')}"

    @staticmethod
    def _hub_cache_repo_dirs(repo_id: str) -> list[Path]:
        """``repo_id``'s cache dir under EVERY root a load can resolve it through, live root first.

        The loader reuses a file cached only under huggingface_hub's import-time root
        (``reuse_other_cache_root``, and the gated preflight excuses a base off it), so after a
        cache-folder change the bytes the load reads are not in the live root at all and sizing that
        root alone reads zero. The constant is read HERE, never bound at import: it is what
        ``cache_dir = None`` resolves to, and tests move it."""
        # One read of the live root: it is a setting, and load_progress polls this from another
        # thread, so a second read could compare the new root against a `live` built from the old.
        live_root = hub_cache_dir()
        folder = f"models--{repo_id.replace('/', '--')}"
        live = Path(live_root) / folder
        try:
            from huggingface_hub import constants
            other = str(constants.HF_HUB_CACHE or "").strip()
        except Exception:  # noqa: BLE001 — no hub package: the live root still stands
            return [live]
        if not other:
            return [live]
        # normcase/normpath before comparing: on Windows the two roots can differ by case alone and
        # every caller below would walk the same tree twice. Best-effort; both readers are idempotent.
        if os.path.normcase(os.path.normpath(other)) == os.path.normcase(
            os.path.normpath(live_root)
        ):
            return [live]
        return [live, Path(other) / folder]

    @staticmethod
    def _live_snapshot_dir(repo_dir: Path) -> Optional[Path]:
        """The snapshot ``refs/main`` names: the ONE revision a load reads out of this root.

        ``try_to_load_from_cache`` (what the per-file root reuse probes with) defaults to ``main``
        and resolves it through ``refs/main``, so this is exactly the tree the loader will read.
        None when the cache does not say -- no ref file (a revision pinned to a commit hash never
        writes one), unreadable, or the snapshot it names is gone. Callers then read the whole root:
        a cache we cannot scope must still report its bytes, never zero."""
        try:
            rev = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
        except (OSError, ValueError):
            return None
        # A ref file holds a bare commit hash; a separator would join out of the cache.
        if not rev or rev in (".", "..") or "/" in rev or "\\" in rev:
            return None
        snapshot = repo_dir / "snapshots" / rev
        return snapshot if snapshot.is_dir() else None

    @staticmethod
    def _cache_bytes(repo_id: str) -> int:
        """Bytes of ``repo_id`` on disk across every cache root, for progress and the pipeline plan.

        Scoped per root to the revision that root serves (``refs/main``), because ``blobs/`` is
        append-only: a republished repo keeps the superseded revision's blobs forever under
        different etags, so summing the whole dir counts a stale full copy on top of the live
        partial one and ``load_progress`` reports "finalizing" through the rest of a multi-GB pull.
        The in-flight ``blobs/*.incomplete`` still counts: those are this download's own bytes, and
        dropping them would freeze the bar for the length of each shard.

        Keyed by blob filename, not summed per root: a blob is named after the file's etag, so a
        copy present in both roots counts once. Scanning only the live root reports 0 for a load a
        moved cache serves entirely off disk, pinning the progress bar near 0%."""
        sizes: dict[str, int] = {}

        def _add(key: str, path: Path) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                return  # broken symlink / unreadable
            sizes[key] = max(sizes.get(key, 0), size)

        for repo_dir in DiffusionBackend._hub_cache_repo_dirs(repo_id):
            snapshot = DiffusionBackend._live_snapshot_dir(repo_dir)
            try:
                blobs = list((repo_dir / "blobs").iterdir())
            except OSError:
                blobs = []  # repo not in this root yet / unreadable / no-symlink cache mode
            if snapshot is None:
                # Unscopeable root: read every blob, as before.
                for entry in blobs:
                    _add(entry.name, entry)
                continue
            for entry in blobs:
                if entry.name.endswith(".incomplete"):
                    _add(entry.name, entry)
            try:
                for f in snapshot.rglob("*"):
                    if not f.is_file():
                        continue
                    # Dedupe on the path INSIDE the snapshot, so one logical file counts once
                    # however many roots hold it. The etag would not: each root serves its own
                    # revision, so a republished file summed both copies and could report
                    # finalizing at half the real progress. Works for a no-symlink cache too, which
                    # stores the file in the snapshot rather than linking to blobs/.
                    _add(f"path:{f.relative_to(snapshot).as_posix()}", f)
            except OSError:
                pass  # the download is writing this tree under the walk; report what we counted
        return sum(sizes.values())

    @staticmethod
    def _local_dir_weight_sizes(path: Path, *, exclude_transformer: bool) -> dict[str, int]:
        """``{relative path: on-disk bytes}`` for the weight files under a diffusers directory.
        Per file, not a total, so callers merging several trees can dedupe by path. See
        ``_local_dir_weight_bytes`` for what the filter is for."""
        sizes: dict[str, int] = {}
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
                sizes[rel.as_posix()] = f.stat().st_size
            except OSError:
                continue
        return sizes

    @staticmethod
    def _local_dir_weight_bytes(path: Path, *, exclude_transformer: bool) -> int:
        """Sum the on-disk weight files under a local diffusers directory. The HF blob
        cache is empty for a local path, so this is the only size signal for auto memory
        planning; without it a large local model folds to zero and the planner skips
        offload and OOMs. ``exclude_transformer`` drops the ``transformer/`` subfolder
        for GGUF/single-file loads (their transformer is the single file, not resident
        here); a full pipeline load keeps it (the whole repo is resident)."""
        return sum(
            DiffusionBackend._local_dir_weight_sizes(
                path, exclude_transformer = exclude_transformer
            ).values()
        )

    @staticmethod
    def _union_over_cached_revs(
        base: str,
        fn: Callable[[Path], dict[str, int]],
        staged_dir: Optional[str] = None,
    ) -> int:
        """Total ``fn`` over a LOCAL diffusers dir, or over the UNION of every tree this load could
        read ``base`` from. 0 when nothing is cached.

        ``fn`` maps a directory to ``{relative path: count}`` so the merge happens per FILE. The
        candidates are every cached snapshot revision under EVERY cache root, plus ``staged_dir``
        (the snapshot the prefetch/preflight handed back). Both roots, because the loader's per-file
        root reuse makes them disjoint PARTS of one repo rather than copies: a text encoder left in
        the old snapshot and a VAE prefetched into the live one both load, and taking the larger
        total would budget only one -- under-counting leaves an auto plan resident and OOMing on
        weights it never counted. Keying on the relative path keeps genuine copies safe too (a file
        mirrored in two roots still counts once, at its largest) and makes the staged dir purely
        additive, so a partial snapshot cannot erase what a root does hold."""
        local = Path(base).expanduser()
        if local.is_dir():
            return sum(fn(local).values())
        candidates: list[Path] = [Path(staged_dir).expanduser()] if staged_dir else []
        for repo_dir in DiffusionBackend._hub_cache_repo_dirs(base):
            # One revision per root, the one it serves: the merge is per FILE, so a superseded
            # revision whose shards were repacked would count both namings and force offload on a
            # base that fits. A root naming no revision falls back to all of them, where
            # over-counting still beats reading zero.
            live_rev = DiffusionBackend._live_snapshot_dir(repo_dir)
            if live_rev is not None:
                candidates.append(live_rev)
                continue
            try:
                candidates.extend(rev for rev in (repo_dir / "snapshots").iterdir() if rev.is_dir())
            except OSError:
                continue  # a root with no copy of this repo contributes nothing
        merged: dict[str, int] = {}
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            for rel, count in fn(candidate).items():
                merged[rel] = max(merged.get(rel, 0), count)
        return sum(merged.values())

    @staticmethod
    def _companion_cache_bytes(base: str, staged_dir: Optional[str] = None) -> int:
        """Resident companion (VAE + text-encoder) size for the memory plan.

        Excludes ``transformer/`` (supplied by the GGUF/single file, not resident here) --
        otherwise the dense-quant prefetch's cached transformer shards would inflate this
        and wrongly force offload. Walks the snapshot dir, not the flat ``blobs/`` cache,
        since only the snapshot preserves the subfolder split needed to exclude it."""
        return DiffusionBackend._union_over_cached_revs(
            base,
            lambda d: DiffusionBackend._local_dir_weight_sizes(d, exclude_transformer = True),
            staged_dir,
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
    def _dense_transformer_resident_bytes(base: str, staged_dir: Optional[str] = None) -> int:
        """Resident bf16 size of the base repo's dense ``transformer/`` for the dense-quant
        preflight. That fast path loads the transformer at the compute dtype (bf16, 2
        bytes/param) before quantizing, so budget num_params * 2 -- NOT the on-disk bytes,
        which for an F32 base (e.g. Z-Image) are ~2x the resident size. Read from the
        safetensors shard headers. Returns 0 when no ``transformer/*.safetensors`` shards
        are present (an uncached base, or a .bin-only transformer); the caller then gates
        the fast path on the plain plan -- so a base whose shards this misses skips the fit
        check entirely and lets the dense build OOM under a plan sized for the GGUF."""

        def _params(d: Path) -> dict[str, int]:
            tdir = d / "transformer"
            if not tdir.is_dir():
                return {}
            return {
                f"transformer/{s.name}": DiffusionBackend._safetensors_param_count(s)
                for s in tdir.glob("*.safetensors")
            }

        # bf16: 2 bytes/param
        return DiffusionBackend._union_over_cached_revs(base, _params, staged_dir) * 2

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
        # LoRA adapters to BAKE into a torchao int8/fp8 build. Ignored elsewhere (bf16/bnb take them at generate time).
        loras: Optional[list[tuple[str, float]]] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
        # True when the prefetch staged the base repo's ``transformer/`` shards, the only condition under which the dense-quant
        # fallback may materialise them. Defaults True for a direct call, which has no prefetch phase.
        _transformer_prefetched: bool = True,
        # The repo the background load staged the companions from; re-derived below for a direct
        # call, which has no prefetch phase.
        _fetch_base: Optional[str] = None,
    ) -> dict[str, Any]:
        # A blank token must degrade to anonymous, not be passed as a credential. Normalize once.
        hf_token = hf_token.strip() if isinstance(hf_token, str) else hf_token
        hf_token = hf_token or None

        # Validate first (no torch/diffusers) so a bad family fails even in a no-diffusers runtime.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        # base_repo is gated at the route pre-eviction; this only cheap-fails the resolved repo/family.
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            family_override = family_override,
            model_kind = model_kind,
        )
        kind = resolve_model_kind(gguf_filename, model_kind)
        # Validate every mode string that can raise BEFORE this load evicts the previous pipeline.
        normalize_transformer_quant(transformer_quant)
        normalize_speed_mode(speed_mode)
        normalize_attention_backend(attention_backend)
        normalize_transformer_cache(transformer_cache)
        normalize_te_quant(text_encoder_quant)
        # A full pipeline is its own base; single-file kinds resolve the companion base repo.
        base = (
            repo_id if kind == "pipeline" else _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        )
        # ``base`` stays the UPSTREAM id every report and table lookup keys on; ``fetch_base`` is
        # the only id handed to something that downloads. One decision per load (the background
        # path took it before staging), so nothing stages one repo and assembles from the other.
        fetch_base = _fetch_base or prefer_ungated_mirror(base, hf_token)
        target = self._resolve_device_target(fam)
        device, dtype = target.device, target.dtype

        import diffusers

        # Pre-install the optional attention kernel before the load locks: the pip install can block unload for 600s.
        try:
            preinstall_backend = select_attention_backend(
                target, attention_backend, speed_active = True
            )
            if preinstall_backend is not None:
                _ensure_attention_backend_installed(preinstall_backend, logger)
        except Exception:  # noqa: BLE001 — the locked path re-resolves and validates
            pass

        # Abort an in-flight denoise and wait for it to exit: a load claims VRAM, so it must not overlap.
        with self._lock:
            # Bail before signalling if this load was superseded, else a stale worker aborts a live one.
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Diffusion load was cancelled.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            # Same fence unload() takes: a queued generation must not run on the pipeline this load is about to free.
            self._teardown_waiters += 1
        with self._generate_lock:
            with self._lock:
                try:
                    # Re-check: a newer load/unload may have superseded this one while we waited.
                    if _load_token is not None and _load_token != self._load_token:
                        raise RuntimeError("Diffusion load was cancelled.")

                    # Free the old pipeline before allocating the new one (never two in VRAM).
                    self._unload_locked()
                finally:
                    # Released here, not at the end of the load: the old pipe is gone and the rest of the load holds _generate_lock.
                    self._teardown_waiters -= 1

                # Single-file kinds resolve a checkpoint path; the pipeline kind has none.
                single_file_path = (
                    self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
                    if kind in ("gguf", "single_file")
                    else None
                )
                # A renamed or hand-picked FLUX.2 GGUF can still land on a different-size base, and
                # no name-based rule catches that. Say so here, naming the file and the repo,
                # rather than letting the GGUF quantizer raise a bare shape mismatch.
                assert_flux2_gguf_matches_base(fam, base, single_file_path)
                transformer_cls = getattr(diffusers, fam.transformer_class)
                pipeline_cls = getattr(diffusers, fam.pipeline_class)

                # Decide placement up front (weights still on CPU). Budgets the GGUF; dense is preflighted below.
                plan = self._plan_memory(
                    target,
                    single_file_path,
                    base,
                    fam,
                    memory_mode,
                    cpu_offload,
                    kind = kind,
                    repo_id = repo_id,
                    # The base may be resolved off the OTHER cache root, which the plan's live-root
                    # scans read as zero companions.
                    base_local_dir = _base_local_dir,
                    fetch_base = fetch_base,
                )

                # Dtype tri-state: unset/"auto" -> hardware ladder; "none"/"off" pins GGUF-as-is; an explicit scheme pins it.
                transformer_quant_auto = transformer_quant is None or str(
                    transformer_quant
                ).strip().lower() in ("", "auto")
                if transformer_quant_auto:
                    # An explicit Speed="off" stays GGUF-as-is: auto-quant would break the bit-exact request.
                    speed_off = (
                        speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
                    )
                    transformer_quant = "off" if speed_off else TQ_AUTO

                # Default-on fast path (GGUF kind only): dense bf16 + torchao quant beats GGUF per-matmul dequant on speed AND quality. Needs CUDA + bf16 + a resident fit.
                pipe = None
                transformer_quant_engaged = None
                quant_plan = None
                # The GGUF-size plan can mis-budget the fast path, so preflight the real footprint pre-eviction. Also set below when an auto quant declines a hosted pre-quant.
                dense_declined = False
                # False when the plan only holds a PREQUANT-sized build: a failed prequant load must raise, not materialise the
                # unbudgeted dense bf16 transformer. Also false when the prefetch skipped the base repo's transformer/ shards, since
                # the fallback would pull them HERE, inside the load lock, after eviction, where unload cannot preempt it and
                # progress already reported 100%.
                dense_fallback_allowed = bool(_transformer_prefetched)
                # A GGUF pick with the scheme left to us: the hosted pre-quant is free only while
                # already cached, since fetching it means a SECOND multi-GB denoiser and the GGUF
                # never runs. An explicit scheme, a LoRA bake and every non-GGUF kind keep today's
                # behaviour, where the prequant IS the point. An all-zero-weight list is not a bake.
                baking_loras = _has_active_lora(loras)
                if kind == "gguf" and transformer_quant_auto and not baking_loras:
                    uncached_prequant = _uncached_prequant_repo(
                        fam,
                        target,
                        transformer_quant,
                        base_repo = base,
                        prequant_path = transformer_prequant_path,
                    )
                    if uncached_prequant is not None:
                        logger.info(
                            "diffusion.transformer_quant_declined: pre-quant checkpoint in %s is "
                            "not cached (an auto quant never downloads a second transformer for a "
                            "GGUF pick); loading the GGUF",
                            uncached_prequant,
                        )
                        dense_declined = True
                if (
                    kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and not dense_declined
                ):
                    if plan.offload_policy != OFFLOAD_NONE:
                        # The GGUF plan picked offload but the quantised artifact is smaller, so re-plan: a resident quant build wins.
                        candidate = resolve_dense_quant_candidate(
                            fam = fam,
                            target = target,
                            requested = transformer_quant,
                            base_repo = base,
                            prequant_path = transformer_prequant_path,
                            # A LoRA bake skips the prequant shortcut, so size for the dense build it will run.
                            force_dense = _has_active_lora(loras),
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
                                    fetch_base = fetch_base,
                                    transformer_resident_override_mib = (
                                        candidate.transient_transformer_mib
                                    ),
                                    # Pass the companion estimate so prefetched base shards aren't double-counted.
                                    companion_override_mib = candidate.companions_mib,
                                )

                            replanned = _replan_candidate()
                            if (
                                replanned.offload_policy != OFFLOAD_NONE
                                # Explicit balanced/low_vram picks offload BY MODE, so a fresh snapshot cannot change it.
                                and normalize_memory_mode(memory_mode)
                                not in (MEMORY_MODE_BALANCED, MEMORY_MODE_LOW_VRAM)
                                and plan_fits_total_capacity(replanned)
                            ):
                                # The candidate fits TOTAL capacity but instantaneous free said no, so re-snapshot (settled) and replan once rather than letting a transient foreign allocation force the GGUF fallback.
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
                                # The GGUF plan declined resident; a prequant-sized replan says nothing about the dense build.
                                if candidate.prequant:
                                    dense_fallback_allowed = False
                    else:
                        # This materialises the dense bf16 transformer, so re-check the fit rather than OOMing after eviction (skipped for a prequant).
                        scheme = select_transformer_quant_scheme(
                            target,
                            transformer_quant,  # normalized above
                            family = getattr(fam, "name", None),
                        )
                        # usable_prequant_source (not resolve_): a missing/non-allowlisted local path must not skip the dense-fit re-check.
                        prequant = (
                            # A LoRA bake skips the prequant shortcut, so gate the fast path as if no prequant existed.
                            None
                            if _has_active_lora(loras)
                            else usable_prequant_source(
                                fam,
                                scheme,
                                path_override = transformer_prequant_path,
                                base_repo = base,
                            )
                            if scheme is not None
                            else None
                        )
                        # Reads the cache, so it looks where the prefetch WROTE (the mirror when one
                        # was swapped in) and carries the staged snapshot: a base served wholly from
                        # the import-time root sizes as 0 on the hub id alone, and a 0 skips this
                        # fit check, landing the dense build under a GGUF-sized plan.
                        dense_mib = int(
                            self._dense_transformer_resident_bytes(fetch_base, _base_local_dir)
                            // (1024 * 1024)
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
                                fetch_base = fetch_base,
                                transformer_resident_override_mib = dense_mib,
                                # No companion_override here, so this one reads the cache: point it
                                # at the snapshot the load will read, not the live root alone.
                                base_local_dir = _base_local_dir,
                            )
                            if dense_plan.offload_policy != OFFLOAD_NONE:
                                dense_fallback_allowed = False
                                # No prequant source: a dense misfit skips the fast path; with one, only the dense fallback is forbidden.
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
                            text_encoder_quant = text_encoder_quant,
                            fetch_base = fetch_base,
                        )
                    except Exception as exc:  # noqa: BLE001 — fall back to the GGUF build
                        logger.warning(
                            "diffusion.transformer_quant_fallback: %s (loading GGUF)", exc
                        )
                        pipe = None
                        transformer_quant_engaged = None
                        # Drop the exception before clearing the cache: its traceback pins the dense transformer's VRAM.
                        del exc
                        # Guarded: a sticky CUDA error can raise; the fallback must reach the GGUF build.
                        try:
                            clear_gpu_cache()
                        except Exception:  # noqa: BLE001
                            pass
                if transformer_quant_engaged is not None and quant_plan is not None:
                    # The engaged dense build uses the re-planned placement; the GGUF-size plan stays for fallback.
                    plan = quant_plan

                if (
                    pipe is None
                    and kind == "gguf"
                    and normalize_transformer_quant(transformer_quant) is not None
                    and _has_active_lora(loras)
                ):
                    # Adapters were requested BAKED but that build failed, and the GGUF fallback cannot carry them; fail loudly.
                    raise RuntimeError(
                        "The requested LoRA adapters could not be applied: baking adapters "
                        "requires the quantized (int8/fp8) transformer build, which was "
                        "declined or failed on this device (see the server log), and the "
                        "GGUF fallback cannot carry them. Retry without transformer_quant "
                        "adapters, free VRAM, or pick a smaller model."
                    )

                if pipe is None:
                    if kind == "pipeline":
                        # Full diffusers repo: from_pretrained pulls every component and re-applies embedded quant config.
                        if fam.name == KREA2_FAMILY_NAME:
                            # krea ships transformers-5.x configs the 4.x line cannot parse, so assemble per-component; that path never sees pipe_kwargs, so pass the pre-cast TE.
                            # Fetches EVERY component from the id given, so it must get the mirror.
                            pipe = load_krea2_pipeline(
                                fetch_base,
                                dtype,
                                hf_token = hf_token,
                                text_encoder = te_prequant_pipe_kwargs(
                                    fam,
                                    fetch_base,
                                    te_quant_mode = text_encoder_quant,
                                    target = target,
                                    dtype = dtype,
                                    hf_token = hf_token,
                                    logger = logger,
                                ).get("text_encoder"),
                            )
                        elif fam.name == IDEOGRAM4_FAMILY_NAME:
                            # ideogram ships the same transformers-5.x Qwen stack as krea; assemble per-component too, from the mirror for the same reason.
                            pipe = load_ideogram4_pipeline(fetch_base, dtype, hf_token = hf_token)
                        else:
                            pipe_kwargs: dict[str, Any] = {
                                "torch_dtype": dtype,
                                "cache_dir": hub_cache_dir(),
                            }
                            if hf_token:
                                pipe_kwargs["token"] = hf_token
                            if fam.name == HIDREAM_FAMILY_NAME:
                                # The repo names a Llama text_encoder_4 it does not ship; supply it from the open mirror.
                                pipe_kwargs.update(
                                    hidream_te4_kwargs(
                                        dtype,
                                        hf_token,
                                        fam = fam,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                    )
                                )
                            # A hosted pre-cast fp8 text encoder skips the dense TE download; the cast re-applies idempotently.
                            pipe_kwargs.update(
                                te_prequant_pipe_kwargs(
                                    fam,
                                    fetch_base,
                                    te_quant_mode = text_encoder_quant,
                                    target = target,
                                    dtype = dtype,
                                    hf_token = hf_token,
                                    logger = logger,
                                )
                            )
                            # The prefetched snapshot dir keeps from_pretrained off the hub (24 GB per FLUX.1 otherwise).
                            pipe = pipeline_cls.from_pretrained(
                                _base_local_dir or fetch_base, **pipe_kwargs
                            )
                    elif kind == "single_file" and fam.single_file_is_pipeline:
                        # A single-file SDXL-style checkpoint is the WHOLE pipeline: load it through the pipeline class with ``config`` on the base repo.
                        sf_pipe_kwargs: dict[str, Any] = {
                            "torch_dtype": dtype,
                            # ``config`` is a REPO FETCH ahead of the mirrored load, so a gated id
                            # would 401 here first.
                            "config": fetch_base,
                            "cache_dir": hub_cache_dir(),
                        }
                        if hf_token:
                            sf_pipe_kwargs["token"] = hf_token
                        pipe = pipeline_cls.from_single_file(single_file_path, **sf_pipe_kwargs)
                    else:
                        # Transformer-only single file; VAE/text-encoder/scheduler come from the base repo.
                        sf_kwargs: dict[str, Any] = {
                            "torch_dtype": dtype,
                            # Fetched before auth, same ordering as the whole-pipeline branch above.
                            "config": fetch_base,
                            "subfolder": "transformer",
                            "token": hf_token,
                            "cache_dir": hub_cache_dir(),
                        }
                        if kind == "gguf":
                            # Dequantise the GGUF transformer on-device at the compute dtype.
                            sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                                compute_dtype = dtype
                            )
                            # sd.cpp GGUFs prefix tensors with model.diffusion_model.; the FLUX.2 / Qwen converters choke.
                            _install_gguf_prefix_strip(transformer_cls, logger)
                        # A safetensors single-file (fp8) carries its own dtype: no GGUF dequant config.
                        transformer = transformer_cls.from_single_file(
                            single_file_path, **sf_kwargs
                        )

                        if fam.name == KREA2_FAMILY_NAME:
                            pipe = load_krea2_pipeline(
                                fetch_base,
                                dtype,
                                hf_token = hf_token,
                                transformer = transformer,
                                # Same pre-cast TE hand-in as the full-pipeline branch.
                                text_encoder = te_prequant_pipe_kwargs(
                                    fam,
                                    fetch_base,
                                    te_quant_mode = text_encoder_quant,
                                    target = target,
                                    dtype = dtype,
                                    hf_token = hf_token,
                                    logger = logger,
                                ).get("text_encoder"),
                            )
                        else:
                            pipe_kwargs = {
                                "torch_dtype": dtype,
                                "transformer": transformer,
                                "cache_dir": hub_cache_dir(),
                            }
                            if hf_token:
                                pipe_kwargs["token"] = hf_token
                            if fam.name == HIDREAM_FAMILY_NAME:
                                # Same Llama TE4 assembly as the full-pipeline branch above.
                                pipe_kwargs.update(
                                    hidream_te4_kwargs(
                                        dtype,
                                        hf_token,
                                        fam = fam,
                                        te_quant_mode = text_encoder_quant,
                                        target = target,
                                    )
                                )
                            # Same pre-cast TE injection as above: the GGUF supplies the transformer, so the TE is the big download.
                            pipe_kwargs.update(
                                te_prequant_pipe_kwargs(
                                    fam,
                                    fetch_base,
                                    te_quant_mode = text_encoder_quant,
                                    target = target,
                                    dtype = dtype,
                                    hf_token = hf_token,
                                    logger = logger,
                                )
                            )
                            pipe = pipeline_cls.from_pretrained(
                                _base_local_dir or fetch_base, **pipe_kwargs
                            )

                # Effective speed: GGUF defaults to `default` (~2.2x, below the quant noise floor); dense stays bit-identical `off`.
                effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
                # A torchao-quantized dense transformer must be compiled (eager is ~30x slower, losing to GGUF).
                if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
                    logger.info(
                        "diffusion.transformer_quant: forcing speed_mode=default "
                        "(quantized transformer must be compiled; eager is ~30x slower)"
                    )
                    effective_speed = SPEED_DEFAULT
                # Deferred speed auto for dense: stay eager and engage `default` on the 3rd image, where compile amortises. Only when speed was unset.
                speed_deferred = (
                    speed_mode is None
                    and effective_speed == SPEED_OFF
                    and transformer_quant_engaged is None
                    and compile_eligible(target, is_gguf = False, family = fam)
                )
                # Speed optims run BEFORE placement, so snapshot the global backend flags first for unload restore.
                backend_flags_before = snapshot_backend_flags()
                # Pick the attention kernel BEFORE compile: auto upgrades to cuDNN fused attention on NVIDIA (~1.18x).
                attention_engaged = apply_attention_backend(
                    pipe,
                    select_attention_backend(
                        target, attention_backend, speed_active = effective_speed != SPEED_OFF
                    ),
                    logger = logger,
                )
                # Step caching (First-Block-Cache), also before compile: reuses the transformer tail across steps (~1.4x on Flux,
                # LPIPS ~0.08) and drops compile fullgraph. Tri-state: unset/auto -> FBCACHE_MIN_STEPS policy; off/fbcache pinned.
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
                # An auto decision can flip at generation time, but only on a cache-capable transformer.
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
                # Everything to the _LoadState commit mutates PROCESS-WIDE state; the try/finally below restores it on failure.
                # gguf_transformer: the dense fast path still sets gguf_filename, but pipe.transformer is dense (REGIONAL compile).
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
                        # Per-arch compile-safe fusions; neutral under compile, tracked by the same eager_patched flag.
                        install_arch_patches()
                        eager_patched = True
                    else:
                        uninstall_patches()
                        uninstall_arch_patches()

                    # Pre-warmed torch.compile cache: a per-fingerprint inductor dir plus a bundle loaded before the first compiled forward pays the 25-58s compile once. A miss is silent.
                    if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and compile_eligible(
                        target, is_gguf = gguf_transformer, family = fam
                    ):
                        compile_ctx = compile_cache.begin(
                            family = fam.name,
                            # U-Net families (SDXL) carry the denoiser as pipe.unet.
                            transformer = getattr(pipe, "transformer", None)
                            or getattr(pipe, "unet", None),
                            dtype = getattr(target, "dtype", None),
                            # A GGUF transformer compiles a DIFFERENT graph than a dense load, so it keys its own bundles.
                            quant = "gguf" if gguf_transformer else transformer_quant_engaged,
                            attention_backend = attention_engaged,
                            compile_kwargs = {
                                # Mirrors apply_speed_optims' fullgraph decision (a step cache or planned offload graph-breaks), so the bundle keys on the same setting.
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
                        # Compile could not engage: the quantized transformer runs eager, far slower than the GGUF it replaced.
                        logger.warning(
                            "diffusion.transformer_quant: %s engaged but the transformer is NOT "
                            "compiled; eager torchao quant is ~30x slower than GGUF here",
                            transformer_quant_engaged,
                        )
                    # Quantise the dense companion text encoder(s) before placement so offload moves the smaller weights.
                    te_quant = quantize_text_encoders(
                        pipe,
                        target,
                        mode = text_encoder_quant,
                        family = fam.name,
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )

                    # Persistent conditioning cache (UNSLOTH_DIFFUSION_COND_CACHE_DIR): repeated prompts skip the text-encoder
                    # forward. After the TE quant so the key reflects the encoders that run; ``base`` keys the companion repo.
                    cond_cache.install(
                        pipe,
                        family = fam.name,
                        repo_id = repo_id,
                        base_repo = base,
                        dtype = dtype,
                        te_quant = te_quant,
                        logger = logger,
                    )

                    # Apply the planned placement; apply_memory_plan returns what ACTUALLY engaged so status stays honest.
                    effective_policy, effective_tiling = apply_memory_plan(
                        pipe, plan, device = device, logger = logger
                    )

                    # Per-control provenance for status. cpu_offload=False is the unset default, so only True is explicit.
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
                        gguf_filename = gguf_filename,
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
        text_encoder_quant: Optional[str] = None,
        fetch_base: Optional[str] = None,
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
        compile -> placement.

        ``base`` keeps the UPSTREAM id for the prequant table and base_model_id checks; every
        download uses ``fetch_base``. A gated base 401s on both the prequant config read and the
        dense pull, and a nonzero baked LoRA refuses the GGUF fallback, so that 401 fails the load.
        """
        fetch_base = fetch_base or prefer_ungated_mirror(base, hf_token)
        # 1. Pre-quantized checkpoint, when one is configured for the resolved scheme.
        scheme = select_transformer_quant_scheme(target, mode, family = getattr(fam, "name", None))
        if scheme is None:
            # Bail BEFORE the multi-GB dense download: an unsupported scheme (fp8 on Ampere, nvfp4 off Blackwell) would
            # materialise the transformer only to fail at quantize, after eviction. load_pipeline falls back to GGUF.
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None and not _has_active_lora(lora_specs):
            # A LoRA bake needs the DENSE transformer (adapters attach before quantize_), so skip
            # prequant. Active weights only: an all-zero list bakes nothing.
            source = resolve_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base
            )
            if source is not None:
                transformer = load_prequantized_transformer(
                    transformer_cls,
                    fetch_base,
                    source,
                    device = device,
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint with a different Linear filter so prequant matches runtime-quant.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Only enforced when the caller forces fp8 fast-accum; a checkpoint that baked the other choice falls to the dense path.
                    fast_accum = fast_accum,
                    # The root _uncached_prequant_repo cleared this load against, so its hit is
                    # reused rather than re-downloaded under the import-time constant.
                    cache_dir = hub_cache_dir(),
                    logger = logger,
                )
                if transformer is not None:
                    pipe = self._assemble_pipe(
                        pipeline_cls,
                        base,
                        transformer,
                        dtype,
                        hf_token,
                        device,
                        base_local_dir,
                        fam = fam,
                        te_quant_mode = text_encoder_quant,
                        target = target,
                        fetch_base = fetch_base,
                    )
                    return pipe, scheme

        # 2. Fallback: materialise the dense bf16 transformer and quantise it on-device.
        if not allow_dense_fallback:
            # The plan only budgeted the prequant-sized build, so the dense bf16 transformer would exceed it after eviction.
            raise RuntimeError(
                "prequant checkpoint unavailable and the dense transformer does not fit resident"
            )
        # Deliberately the hub id, not base_local_dir: diffusers treats a local directory as
        # terminal (_get_model_file raises rather than falling back to the hub) and a sharded load
        # raises per missing shard, so a partial snapshot would drop a build the hub id completes.
        # Off the hub id a base only the other root holds costs a re-download, or 401s into the
        # GGUF fallback, which is what main does today.
        transformer = transformer_cls.from_pretrained(
            fetch_base,
            subfolder = "transformer",
            torch_dtype = dtype,
            token = hf_token,
            cache_dir = hub_cache_dir(),
        )
        pipe = self._assemble_pipe(
            pipeline_cls,
            base,
            transformer,
            dtype,
            hf_token,
            device,
            base_local_dir,
            fam = fam,
            te_quant_mode = text_encoder_quant,
            target = target,
            fetch_base = fetch_base,
        )
        if _has_active_lora(lora_specs):
            # Bake the adapters BEFORE quantize_: peft wraps the dense Linears (post-quant torchao dispatch would TypeError),
            # then quantize_ converts only each wrapper's frozen base_layer while the "lora_" side path stays high precision.
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
        te_quant_mode: Optional[str] = None,
        target: Any = None,
        fetch_base: Optional[str] = None,
    ) -> Any:
        """Assemble the diffusers pipeline around ``transformer`` and place it on ``device``
        (a no-op for an already-placed pre-quantized transformer; it moves the companions).

        Everything below reads the base only to FETCH, so it uses ``fetch_base`` (the load-wide
        decision, re-derived for a direct call). Matters when ``base_local_dir`` is None: nothing
        was staged, so a gated upstream would 401 here."""
        base = fetch_base or prefer_ungated_mirror(base, hf_token)
        if getattr(fam, "name", None) == KREA2_FAMILY_NAME:
            # krea ships transformers-5.x configs and no top-level tokenizer files, so assemble per-component.
            krea_te = None
            if target is not None:
                krea_te = te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                ).get("text_encoder")
            pipe = load_krea2_pipeline(
                base_local_dir or base,
                dtype,
                hf_token = hf_token,
                transformer = transformer,
                text_encoder = krea_te,
            )
            pipe.to(device)
            return pipe
        pipe_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "transformer": transformer,
            "cache_dir": hub_cache_dir(),
        }
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if getattr(fam, "name", None) == HIDREAM_FAMILY_NAME:
            # The repo ships no Llama text_encoder_4; assemble it from the open mirror, as above.
            pipe_kwargs.update(
                hidream_te4_kwargs(
                    dtype,
                    hf_token,
                    fam = fam,
                    te_quant_mode = te_quant_mode,
                    target = target,
                )
            )
        # Same pre-cast TE injection as the other branches: the dense path supplies only the transformer.
        if target is not None:
            pipe_kwargs.update(
                te_prequant_pipe_kwargs(
                    fam,
                    base,
                    te_quant_mode = te_quant_mode,
                    target = target,
                    dtype = dtype,
                    hf_token = hf_token,
                    logger = logger,
                )
            )
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
        base_local_dir: Optional[str] = None,
        fetch_base: Optional[str] = None,
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
        top of transformer_resident_override_mib (a double-count of the transformer).

        ``base_local_dir`` is the snapshot the load will actually read, carried into the size lookups
        as an extra source alongside the cache roots. A prefetch split across roots hands back no
        snapshot at all and a preflight-excused one can hold less than the repo, so it is additive
        and never a replacement: under-counting here leaves an auto plan resident and OOMing on
        weights it never budgeted.

        ``fetch_base`` is the repo the bytes were staged from, so every cache scan below reads it:
        sizing an upstream id whose cache is empty folds the VAE/text-encoder to zero and wrongly
        picks resident placement. ``base`` and ``repo_id`` keep the upstream identity for the
        family/variant checks."""
        # Settled (max-over-reads) on cuda: a transient foreign allocation would make an empty card look full.
        device_memory = settled_snapshot_device_memory(target)
        if kind == "pipeline":
            # The whole repo is one cached download, so cached bytes are the resident estimate; a LOCAL path is not cached, so sum its on-disk weights.
            local_repo = Path(repo_id).expanduser() if repo_id else None
            staged_repo = Path(base_local_dir).expanduser() if base_local_dir else None
            # The bytes land under the repo they were STAGED from, so scan the mirror when there is
            # one: an upstream id whose cache is empty folds the whole pipeline to zero.
            cache_repo = fetch_base or repo_id
            # Largest of every source the load can read this repo from: a staged snapshot can sit
            # under the OTHER root, and a prefetch that split the repo across roots hands back none
            # at all, so the smaller answer would plan a multi-GB pipeline as unknown.
            cached = max(
                self._local_dir_weight_bytes(local_repo, exclude_transformer = False)
                if local_repo is not None and local_repo.is_dir()
                else 0,
                self._local_dir_weight_bytes(staged_repo, exclude_transformer = False)
                if staged_repo is not None and staged_repo.is_dir()
                else 0,
                self._cache_bytes(cache_repo) if cache_repo else 0,
            )
            cached_mib = int(cached // (1024 * 1024)) if cached else None
            model_dense_mib = estimate_safetensors_dense_mib(cached_mib)
            # A repo can store weights NARROWER than the loaded dtype (ideogram-4 ships raw float8), so cached bytes
            # undershoot the bf16 footprint ~2x. Plan against the size table's bf16 total when it knows this repo.
            is_narrow_base = bool(repo_id) and repo_id.strip().lower() == fam.base_repo.lower()
            if (
                not is_narrow_base
                and fam.name == IDEOGRAM4_FAMILY_NAME
                and local_repo is not None
                and local_repo.is_dir()
            ):
                # A local fp8 mirror never string-matches base_repo, so detect fp8 from the shard headers (a local nf4 mirror stays compressed).
                is_narrow_base = ideogram4_repo_is_fp8(repo_id)
            if is_narrow_base:
                table = family_bf16_components_gb(fam, fam.base_repo)
                if table is not None:
                    # Reserve the bf16 footprint from this network-free constant, else model_dense_mib stays None ("unknown -> resident") and the ~54 GB pipeline OOMs.
                    table_mib = int(sum(table) * (1000.0**3) / (1024.0 * 1024.0))
                    model_dense_mib = (
                        table_mib if model_dense_mib is None else max(model_dense_mib, table_mib)
                    )
            companion_mib = None
        else:
            if transformer_resident_override_mib is not None:
                # Planning the dense-quant candidate: the auto-policy estimate replaces the file-size derivation.
                transformer_resident = transformer_resident_override_mib
            elif kind == "single_file":
                # An fp8 checkpoint upcasts to bf16 on load (~2x resident); detect from the basename. SDXL single-file is already bf16.
                fp8_upcast = not getattr(fam, "single_file_is_pipeline", False) and (
                    "fp8" in Path(single_file_path).name.lower() if single_file_path else False
                )
                transformer_resident = estimate_safetensors_dense_mib(
                    file_size_mib(single_file_path), fp8_upcast = fp8_upcast
                )
            else:
                transformer_resident = estimate_gguf_resident_mib(file_size_mib(single_file_path))
            # Companions (VAE + text encoders) load near on-disk size; sum the base-repo cache, or a LOCAL base's on-disk weights.
            if companion_override_mib is not None:
                # Re-planning the dense candidate: the prefetched transformer/ shards land in the same cache, so use the estimate instead of double-counting.
                companion_mib = companion_override_mib
            else:
                # Scan the repo the bytes were staged from, and hand over the staged snapshot too: a
                # base served from the import-time root is invisible to a hub-id scan of the live
                # one, so the VAE + text encoders would load unbudgeted.
                companion = self._companion_cache_bytes(fetch_base or base, base_local_dir)
                companion_mib = int(companion // (1024 * 1024)) if companion else None
            model_dense_mib = None
            if transformer_resident is not None:
                model_dense_mib = transformer_resident + (companion_mib or 0)
        # Feed the variant hint so estimate_image_runtime_mib sees distilled markers (distilled needs ~15% less headroom).
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

        # torch_dtype=None is load-bearing: from_pipe otherwise recasts EVERY component to fp32, which hard-crashes the
        # dense-quant path (torchao subclasses cannot swap_tensors). None reuses resident modules at their loaded dtype.
        pipe = getattr(diffusers, class_name).from_pipe(state.pipe, torch_dtype = None)
        # Publish to the shared aux cache only if THIS load is still current: from_pipe runs without _lock, so an unload can null _state and caching would hand out stale modules.
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
            # resolve_controlnet accepts a bare owner/name without the trust gate and from_pretrained would execute a malicious
            # pickle, so run the same Hub malware preflight. It fails OPEN, so a remote repo also forces safetensors below.
            remote_cn = not getattr(resolved_cn, "is_local", False)
            if remote_cn:
                from utils.security import evaluate_file_security
                _cn_fs = evaluate_file_security(resolved_cn.path, hf_token = state.hf_token or None)
                if _cn_fs.blocked:
                    raise ValueError(_cn_fs.reason)
            # Keep at most one ControlNet resident, else swapping ControlNets accumulates until OOM.
            if self._cn_models or self._cn_pipes:
                self._cn_models.clear()
                self._cn_pipes.clear()
                clear_gpu_cache()
            import torch

            # state.dtype is the display string ("bfloat16"), so pass the real dtype and avoid a float32 load.
            cn_dtype = getattr(torch, str(state.dtype).replace("torch.", ""), None)
            # Force safetensors for an untrusted remote repo: if the Hub scan failed open, an embedded pickle would still deserialize.
            cn_from_pretrained_kwargs: dict[str, Any] = {"cache_dir": hub_cache_dir()}
            if remote_cn:
                cn_from_pretrained_kwargs["use_safetensors"] = True
            cn_model = getattr(diffusers, model_cls_name).from_pretrained(
                resolved_cn.path,
                torch_dtype = cn_dtype,
                token = state.hf_token or None,  # blank -> anonymous
                **cn_from_pretrained_kwargs,
            )
            if cancel.is_set():
                # An unload raced the blocking download; bail BEFORE placement so we do not allocate onto a just-freed GPU.
                del cn_model
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            # Placement follows the base offload policy (resident base -> resident, offloaded -> group offload). Best-effort.
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
                # Same race as the model cache: an unload may have cleared _cn_pipes while from_pipe ran.
                if cancel.is_set() or self._state is not state:
                    del pipe
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
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
            # Read the dtype from the parameters (a compiled nn.Module may hide .dtype), taking the first FLOATING one.
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

        resolved = diffusion_lora.resolve_specs(
            specs,
            family = family,
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
            # Name only routes the user can actually reach: a GPU host never selects the native engine on its own.
            raise ValueError(
                "LoRA is not supported for this model/quantisation on the diffusers engine "
                "(GGUF-via-diffusers, or a torch.compile'd Speed=default/max load). Reload with "
                "transformer_quant int8 or fp8, which rebuilds the GGUF into a LoRA-capable dense "
                "transformer, or use a bf16 / bnb-4bit load at Speed=off/eager. To keep the GGUF "
                "weights themselves, run the native engine, which a GPU host selects only when "
                "UNSLOTH_DIFFUSION_ENGINE=sd_cpp is set."
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
        """Clear the transformer's stateful step cache (FBCache) before a forward.

        diffusers keys FBCache residuals by cache context ("cond"/"uncond") on the
        long-lived transformer. The context exit does NOT reset them; the end of a
        pipeline ``__call__`` does, via ``maybe_free_model_hooks()`` -- but only when
        the call RETURNS. A call that raised (an OOM this generate() backs off from, a
        cancelled denoise, a failed prior request) leaves its own batch's residual on
        the resident transformer, and the next forward's first step then compares
        against it: a tensor-shape mismatch when the resolution/batch changed, or a
        stale-cache reuse otherwise. The transformer-level reset entry point is
        ``_reset_stateful_cache`` in diffusers 0.39 (``reset_stateful_hooks`` lives only
        on the HookRegistry, so a getattr for it on the transformer is a silent no-op).
        Best-effort: a transformer without the hook (uncached load) is a silent no-op."""
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
        # Re-run the load-time selection with the caller's ORIGINAL request: auto still upgrades to cuDNN here.
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
                # Same GGUF-vs-dense graph distinction as the load-time begin().
                quant = "gguf" if gguf_transformer else state.transformer_quant,
                attention_backend = attention_engaged,
                compile_kwargs = {
                    # Mirrors the load-time fullgraph decision: a step cache or an offload graph-breaks.
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
        # Batched multi-image generation: ``prompts`` renders one image per prompt (txt2img only), ``seeds`` one per seed,
        # ``batch_size`` alone derives base..base+n-1. Each image gets its OWN Generator, so same-seed repeats are bit-identical.
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        # Image-conditioned (base64/data-URL): init alone = img2img, init + mask = inpaint. ``strength`` 0 = keep source, 1 = full redraw.
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

        # Per-generation cancel Event that unload()/a superseding load set (under _lock) to abort just this denoise.
        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                # A teardown is waiting for this lock and Python locks are not FIFO, so refuse rather than start a denoise on a pipeline that is already being torn down.
                if self._teardown_waiters:
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # Register under _lock so unload()/a load can signal THIS generation.
                self._active_generate_cancel = cancel
                # Publish an active (step 0) state before the slow pre-denoise setup so a reload mount probe does not read idle.
                self._gen = _GenState(total_steps = steps)
            try:
                # The local `state` ref keeps the pipe alive even if unload() nulls _state. Resolve the per-image (prompt, seed) jobs
                # up front: N prompts, one prompt x N seeds, or one prompt deriving base..base+batch_size-1 (as the native engine does).
                jobs, seed = resolve_batch_jobs(
                    prompt = prompt,
                    prompts = prompts,
                    seed = seed,
                    seeds = seeds,
                    batch_size = batch_size,
                    draw_seed = torch.Generator(device = state.device).seed,
                )

                # Deferred speed auto: engage the compile profile on the 3rd image. Best-effort; a failure stays eager and never retries.
                # NOT when a LoRA is requested: a compiled transformer rejects LoRA, breaking every LoRA generation on this load.
                lora_requested = _has_active_lora(loras)
                # Also stay eager while a PRIOR generation's adapters are attached: compiling would bake them in permanently.
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

                # Select the workflow pipe: txt2img uses the loaded pipe; img2img/inpaint reuse its modules via from_pipe.
                pipe = state.pipe
                init_pil = mask_pil = None
                control_pil = None
                cn_scale = cn_gstart = cn_gend = cn_mode = None
                ref_extra: list = []
                # Validate dependencies up front: mask/upscale/reference need an input image, and reference needs a supporting family.
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
                    # Instruction editing: the loaded pipe IS the edit pipeline and always needs an input image; the prompt is the instruction.
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
                    # Upscale (hires fix): enlarge with Lanczos, then re-run img2img at low strength to add detail.
                    workflow = "upscale"
                    pipe = self._workflow_pipe(state, state.family.img2img_pipeline_class, workflow)
                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    iw, ih = init_pil.size
                    # Cap the factor, then the absolute output (longest side 2048); round to a multiple of 16 (VAE downsample + patch).
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
                    # FLUX.2 reference conditioning: the loaded pipe takes the reference via `image` and generates at the REQUESTED size.
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

                # ControlNet (diffusers): txt2img only. Builds the family CN pipeline around resident modules.
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
                        # Decode + preprocess the control image FIRST so a bad image 400s before any CN download, at the OUTPUT size.
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
                # A prompt LIST batches plain text-to-image only: conditioned workflows take one image per call and a silent broadcast would pair every prompt with it.
                if uniform_prompt(jobs) is None and workflow != "txt2img":
                    raise ValueError(
                        "A prompts list is supported for plain text-to-image only; the "
                        f"{workflow} workflow takes one prompt per call (seed lists still work)."
                    )
                # Snap odd-sized inputs (and the mask) to a multiple of 16 where the OUTPUT size comes from the input image.
                if init_pil is not None and workflow in ("img2img", "inpaint", "edit"):
                    # img2img/inpaint take output size from the upload, so bound the longest side to 2048 (a phone photo would OOM).
                    if workflow in ("img2img", "inpaint"):
                        init_pil = _clamp_max_side(init_pil, 2048)
                    init_pil = _snap_to_multiple(init_pil, 16)
                    if mask_pil is not None and mask_pil.size != init_pil.size:
                        from PIL import Image as _PILImage
                        mask_pil = mask_pil.resize(init_pil.size, _PILImage.NEAREST)
                if init_pil is not None:
                    # Keep the VAE encode dtype consistent with the input image.
                    self._align_vae_dtype(pipe, state.family.denoiser_attr)

                # Pipelines vary in accepted kwargs, so gate every optional one on the signature.
                call_params = inspect.signature(pipe.__call__).parameters

                kwargs: dict[str, Any] = {
                    # prompt / generator / num_images_per_prompt are set per chunk below, one torch.Generator PER IMAGE.
                    "num_inference_steps": steps,
                    # Most pipelines use "guidance_scale"; Qwen-Image uses "true_cfg_scale".
                    state.family.cfg_kwarg: guidance,
                }
                if state.family.name == IDEOGRAM4_FAMILY_NAME:
                    # Ideogram 4 drives CFG via EITHER a constant guidance_scale OR a per-step guidance_schedule, never both.
                    # At the advertised defaults drop the constant so the recommended 48-step taper engages; else null the schedule.
                    if steps == 48 and abs(float(guidance) - 7.0) < 1e-6:
                        kwargs.pop(state.family.cfg_kwarg, None)
                    else:
                        kwargs["guidance_schedule"] = None
                if state.family.name == LUMINA2_FAMILY_NAME and "cfg_trunc_ratio" in call_params:
                    # Lumina 2's card recipe truncates the CFG double-forward to the first quarter (cfg_trunc_ratio=0.25); the 1.0 default oversaturates.
                    kwargs["cfg_trunc_ratio"] = 0.25
                if init_pil is not None:
                    # Reference passes the whole list (FLUX.2 combines); others take the single image.
                    kwargs["image"] = [init_pil, *ref_extra] if ref_extra else init_pil
                    if mask_pil is not None and "mask_image" in call_params:
                        kwargs["mask_image"] = mask_pil
                    if strength is not None and "strength" in call_params:
                        kwargs["strength"] = strength
                # width/height: txt2img uses the slider; image-conditioned pipes must use the INPUT IMAGE's size or the latents mismatch, and many drop them, so pass only when accepted.
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
                    # CN pipeline takes the control map + scale; guidance start/end bound its step range. Every kwarg is signature-gated.
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

                # Per-forward chunks: the whole job list in ONE forward by default (batch 32 on 4-step models is ~10-22x over
                # serial engines), bounded by an explicit batch_size cap; the OOM backoff below halves a failed chunk.
                chunks = chunk_jobs(jobs, batch_size)
                gen = _GenState(total_steps = steps * len(chunks))
                # Steps completed by FINISHED chunks, so the bar spans the whole multi-chunk call (mutable cell for _on_step).
                steps_done = [0]

                def _on_step(pipe, step_index, timestep, callback_kwargs):
                    # Monotonic: a wall-clock adjustment (NTP) mid-denoise would skew the ETA.
                    now = time.monotonic()
                    gen.step = steps_done[0] + step_index + 1
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

                # Re-check an AUTO cache decision against the ACTUAL step count; explicit choices never toggle.
                if state.cache_auto:
                    # Key on the EFFECTIVE denoise steps: img2img at strength < 1 denoises a fraction of `steps`, so fold it in to keep FBCache off short trajectories.
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
                        # _LoadState is frozen; the one deliberate in-place update, so status() reports the pipe-level toggle.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {denoise_steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                self._gen = gen
                images: list[Any] = []
                per_image_seeds: list[int] = []
                chunk_shapes: list[int] = []
                pending = list(chunks)
                while pending:
                    chunk = pending.pop(0)
                    chunk_kwargs = dict(kwargs)
                    shared = uniform_prompt(chunk)
                    generators = [
                        torch.Generator(device = state.device).manual_seed(s) for _, s in chunk
                    ]
                    if len(jobs) == 1:
                        # Single image: scalar prompt + generator, exactly the pre-batching call shape (checked bit-identical).
                        chunk_kwargs["prompt"] = shared
                        chunk_kwargs["generator"] = generators[0]
                        chunk_kwargs["num_images_per_prompt"] = 1
                    elif shared is not None:
                        # Uniform prompt: encode it ONCE and fan out per-image generators.
                        chunk_kwargs["prompt"] = shared
                        chunk_kwargs["generator"] = generators
                        chunk_kwargs["num_images_per_prompt"] = len(chunk)
                    else:
                        # Distinct prompts: one image per prompt in a single forward. The negative prompt must be broadcast to match, else the pipeline asserts or fails in the txt/img concat.
                        chunk_kwargs["prompt"] = [p for p, _ in chunk]
                        chunk_kwargs["generator"] = generators
                        chunk_kwargs["num_images_per_prompt"] = 1
                        if isinstance(chunk_kwargs.get("negative_prompt"), str):
                            chunk_kwargs["negative_prompt"] = [
                                chunk_kwargs["negative_prompt"]
                            ] * len(chunk)
                    # Start every forward from a clean step cache: diffusers only resets FBCache after a SUCCESSFUL __call__, so a raised call leaves a residual the next forward trips over.
                    if state.transformer_cache:
                        self._reset_step_cache(state.pipe)
                    try:
                        # inference_mode is faster than no_grad and numerically identical here.
                        with torch.inference_mode():
                            out = pipe(**chunk_kwargs).images
                    except Exception as exc:  # noqa: BLE001 — reraised unless a splittable OOM
                        if len(chunk) < 2 or not is_oom_error(exc):
                            raise
                        # OOM backoff: halve the failed chunk and retry; per-image seeds keep every retry reproducible.
                        empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
                        if callable(empty_cache):
                            empty_cache()
                        first_half, second_half = split_chunk(chunk)
                        pending[:0] = [first_half, second_half]
                        gen.total_steps += steps  # one extra chunk to run
                        logger.warning(
                            "diffusion.generate: batch of %d hit OOM; retrying as %d + %d",
                            len(chunk),
                            len(first_half),
                            len(second_half),
                        )
                        continue
                    # A cancelled denoise returns a partial image; don't persist it, nor run remaining chunks.
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    images.extend(out)
                    per_image_seeds.extend(s for _, s in chunk)
                    chunk_shapes.append(len(chunk))
                    steps_done[0] += steps
                # Keep progress ACTIVE through the post-denoise work: the route persists the image after this returns, so a mount probe reading idle would refresh the gallery too early.
                # Persist the warm compile bundle; a STATIC compile makes new artifacts per (w,h,batch), so register this shape.
                try:
                    # Register the dims the forward ACTUALLY compiled with, and every distinct chunk size (a static compile makes one artifact per batch size too).
                    reg_width, reg_height = _compile_shape_dims(workflow, init_pil, width, height)
                    static_shapes = "compiled" in (
                        state.speed_optims or ()
                    ) and compiled_shapes_are_static(state.pipe, state.speed_mode)
                    for chunk_batch in sorted(set(chunk_shapes)):
                        compile_cache.register_shape(
                            state.compile_cache_ctx,
                            (reg_width, reg_height, int(chunk_batch)),
                            static = static_shapes,
                        )
                    compile_cache.save(state.compile_cache_ctx, logger = logger)
                except Exception:  # noqa: BLE001 — cache persistence is best-effort
                    pass
                # Count the finished generation (drives deferred speed); a batch is one generation.
                object.__setattr__(state, "generation_count", state.generation_count + 1)
                # Return the PIL images unencoded; the route embeds recipes and persists them. ``seeds`` records each image's own seed.
                return {
                    "images": list(images),
                    "seed": int(seed),
                    "seeds": [int(s) for s in per_image_seeds],
                    "repo_id": state.repo_id,
                    # The BUILD this ran on, not just the repo id: a GGUF quant and a torchao scheme each change the pixels.
                    "model_kind": state.kind,
                    "gguf_filename": state.gguf_filename,
                    "transformer_quant": state.transformer_quant,
                    # Adapters baked in at LOAD time: disabling them at generate time is not the same build, so they belong to the build record.
                    "baked_loras": _baked_lora_names(state.pipe),
                    # The adapters ACTUALLY attached, at non-zero weight, for this generation.
                    "active_loras": _active_lora_pairs(state.pipe),
                    # The workflow this generation ACTUALLY ran, so a conditioned image is not replayed as a plain Create recipe.
                    "workflow": workflow,
                }
            finally:
                # Deregister so a later unload/load can't poke a finished generation (if still ours).
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                    # Sole clear of the published progress state, on every exit, so a crashed generation never leaves the UI stuck.
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
        with self._lock:
            # Abort an in-flight (lock-free) download so unload returns promptly. Under the lock, like video.py: begin_load
            # rebinds this attribute, so an unlocked read could set an event the current load no longer watches.
            self._cancel_event.set()
            # Abort an in-flight denoise via ITS cancel event.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            # Fence queued generations too: they hold no cancel event yet, so the signal above cannot reach them.
            self._teardown_waiters += 1
            # Cancel any in-flight load (its worker checks this token) and drop the marker.
            self._load_token += 1
            self._loading = None
        # Wait for the signalled denoise to exit BEFORE tearing down: _unload_locked uninstalls process-wide state
        # (attention patches, GGUF compile hooks, backend flags, compile cache) the denoise still depends on.
        with self._generate_lock:
            with self._lock:
                try:
                    self._unload_locked()
                finally:
                    # Released in a finally, exactly like begin_load: _unload_locked ends in clear_gpu_cache(), which raises on a
                    # sticky CUDA fault, and an un-drained fence would refuse every later generation for the life of the process.
                    self._teardown_waiters -= 1
        return self.status()

    def _unload_locked(self) -> None:
        state = self._state
        if state is None:
            return
        # Restore the process-wide backend flags this load flipped so the next `off` load is bit-identical. All idempotent.
        restore_backend_flags(state.backend_flags_before)
        compile_cache.restore(state.compile_cache_ctx)
        gguf_compile.uninstall_all()
        if state.eager_patched:
            # Lazy import to keep diffusion.py torch-free to import.
            from .diffusion_eager_patches import uninstall_patches
            from .diffusion_arch_patches import uninstall_arch_patches

            uninstall_patches()
            uninstall_arch_patches()
        # Deliberately NOT unload_lora_weights(): the whole pipe is dropped below, freeing any adapters with it.
        # Drop the workflow pipes so they do not pin the freed pipeline modules past unload.
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
            # A card tag can now name a mirror (they are real unsloth/* repos) and clear the trust
            # bar. Map it back: this becomes status()["base_repo"] and a trained adapter's default
            # base_model, both of which must stay the vendor id. An EXPLICIT base_repo is verbatim.
            base = canonical_base(tag)
    # Returns the UPSTREAM id; the swap happens at the fetch sites only.
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


# Weights the base repo need NOT supply when the single file is the whole pipeline (SDXL): from_single_file(config=base) reads only the structure.
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
