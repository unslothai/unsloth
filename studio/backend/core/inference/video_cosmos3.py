# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3 pipeline; load refuses a missing guardrail, opt out with UNSLOTH_COSMOS3_SAFETY_CHECKER=0."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

SAFETY_CHECKER_ENV = "UNSLOTH_COSMOS3_SAFETY_CHECKER"
# Passed as None to from_pretrained: Studio renders video only.
NONE_COMPONENTS = ("sound_tokenizer",)
GUARDRAIL_PACKAGE = "cosmos_guardrail"
GUARDRAIL_REPO = "nvidia/Cosmos-1.0-Guardrail"
_GUARDRAIL_STEPS = (
    f"accept its licence at https://huggingface.co/{GUARDRAIL_REPO}, add that account's Hugging Face token in "
    f"Settings, and load again. Or set {SAFETY_CHECKER_ENV}=0 to run without it at your own responsibility."
)
# Fallback when the index is not on disk yet (validation is network-free).
_BASE_SAFETY_CHECKER = {
    "nvidia/cosmos3-nano": True,
    "nvidia/cosmos3-edge": False,
}


def is_cosmos3_listing(names: Iterable[str]) -> bool:
    """Whether a Diffusers repo listing has Cosmos3's layout: a ``text_tokenizer/`` and no ``text_encoder/``."""
    names = list(names)
    return any(n.startswith("text_tokenizer/") for n in names) and not any(
        n.startswith("text_encoder/") for n in names
    )


def safety_checker_opted_out() -> bool:
    return os.environ.get(SAFETY_CHECKER_ENV, "").strip().lower() in ("0", "false", "no", "off")


def guardrail_installed() -> bool:
    try:
        return importlib.util.find_spec(GUARDRAIL_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def _cached_model_index(repo_id: str) -> Optional[dict]:
    """The pipeline index from a local folder or the Hub cache; never the network."""
    path: Optional[Path] = None
    local = Path(repo_id).expanduser()
    if local.is_dir():
        path = local / "model_index.json"
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
            found = try_to_load_from_cache(repo_id, "model_index.json")
            path = Path(found) if isinstance(found, str) else None
        except Exception:  # noqa: BLE001 -- a cache miss falls back to the base repo's known setting
            path = None
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def checkpoint_enables_safety_checker(repo_id: str, index: Optional[dict] = None) -> bool:
    """``enable_safety_checker`` from the model_index.json, else the base repo's shipped value; on when unknown."""
    if index is None:
        index = _cached_model_index(repo_id)
    if index is not None:
        value = index.get("enable_safety_checker", True)
        return bool(value) if isinstance(value, bool) else True
    return _BASE_SAFETY_CHECKER.get(str(repo_id or "").strip().lower(), True)


_TE_QUANT_UNSET = frozenset({"", "auto", "none", "off"})


def load_request_refusal(
    family: str, kind: str, text_encoder_quant: Optional[str]
) -> Optional[str]:
    """Refuse non-pipeline picks (no ``from_single_file``) and an explicit text-encoder precision (none exists)."""
    if kind != "pipeline":
        return (
            f"{family} loads only as its full Diffusers pipeline (nvidia/Cosmos3-Nano or nvidia/Cosmos3-Edge): "
            "Cosmos3OmniTransformer has no single-file loader, so GGUF and single .safetensors checkpoints cannot run."
        )
    scheme = str(text_encoder_quant or "").strip().lower()
    if scheme not in _TE_QUANT_UNSET:
        return (
            f"text_encoder_quant '{text_encoder_quant}' does not apply to {family}: Cosmos3 has no separate text "
            "encoder (its transformer reads the prompt tokens itself). Leave text_encoder_quant unset."
        )
    return None


def guardrail_refusal(repo_id: str, index: Optional[dict] = None) -> Optional[str]:
    """Why this load cannot start, or None: the checkpoint enables the guardrail and the package is missing."""
    if safety_checker_opted_out() or guardrail_installed():
        return None
    if not checkpoint_enables_safety_checker(repo_id, index):
        return None
    return f"'{repo_id}' needs NVIDIA's gated Cosmos guardrail: pip install cosmos-guardrail, restart, {_GUARDRAIL_STEPS}"


# Small files the guardrail fetches on its first build (blocklist/*, face_blur_filter/*); all of them gated.
GUARDRAIL_PROBE_FILES = ("blocklist/custom/branding", "face_blur_filter/Resnet50_Final.pth")
_PROBE_TIMEOUT_S = 10.0
# A granted probe is remembered per token, so the route's check and the load's own one send one request.
_PROBE_OK_TTL_S = 60.0
_probe_ok: dict[str, float] = {}
_probe_lock = threading.Lock()


def _guardrail_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    for name in GUARDRAIL_PROBE_FILES:
        try:
            if not isinstance(try_to_load_from_cache(GUARDRAIL_REPO, name), str):
                return False
        except Exception:  # noqa: BLE001 -- an unreadable cache is a miss
            return False
    return True


def _hub_offline() -> bool:
    try:
        from huggingface_hub import constants
        if getattr(constants, "HF_HUB_OFFLINE", False):
            return True
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _probe_key(token: Any) -> str:
    import hashlib
    return hashlib.sha256(repr(token).encode("utf-8")).hexdigest()


def _remote_model_index(repo_id: str, hf_token: Any) -> Optional[dict]:
    """The checkpoint's few-KB model_index.json from the Hub, or None on any failure."""
    if Path(repo_id).expanduser().exists():
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, "model_index.json", token = hf_token)
        data = json.loads(Path(path).read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 -- unknown stays unknown; the family default decides
        return None
    return data if isinstance(data, dict) else None


def guardrail_access_preflight(
    repo_id: str,
    hf_token: Any = None,
    *,
    local_files_only: bool = False,
    index: Optional[dict] = None,
    default_enabled: bool = True,
) -> Optional[str]:
    """Refusal for a load whose guardrail weights this token cannot fetch, checked before any model download.

    One metadata request on a small gated guardrail file with the load's token (None = the ambient login). Only a
    401 / 403 refuses: a cached guardrail, offline mode and network errors all let the load go on as before. A
    checkpoint that is neither cached nor a base repo is judged by its own model_index.json (a few KB), else by
    ``default_enabled`` (the family's: Edge ships the guardrail off)."""
    if safety_checker_opted_out() or not guardrail_installed():
        return None
    if local_files_only or _hub_offline() or _guardrail_cached():
        return None
    if isinstance(hf_token, str):
        hf_token = hf_token.strip() or None
    if index is None:
        index = _cached_model_index(repo_id)
    if index is None and str(repo_id or "").strip().lower() not in _BASE_SAFETY_CHECKER:
        index = _remote_model_index(repo_id, hf_token)
    if index is None and str(repo_id or "").strip().lower() not in _BASE_SAFETY_CHECKER:
        enabled = default_enabled
    else:
        enabled = checkpoint_enables_safety_checker(repo_id, index)
    if not enabled:
        return None
    key = _probe_key(hf_token)
    now = time.monotonic()
    with _probe_lock:
        if _probe_ok.get(key, 0.0) > now:
            return None
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        get_hf_file_metadata(
            hf_hub_url(GUARDRAIL_REPO, GUARDRAIL_PROBE_FILES[0]),
            token = hf_token,
            timeout = _PROBE_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 -- only an access refusal stops the load
        status = getattr(getattr(exc, "response", None), "status_code", None)
        gated = any(cls.__name__ == "GatedRepoError" for cls in type(exc).__mro__)
        if gated or status in (401, 403):
            logger.warning("video.cosmos3: guardrail access refused before the download: %s", exc)
            return f"'{repo_id}' needs NVIDIA's gated Cosmos guardrail, and this Hugging Face token has no access: {_GUARDRAIL_STEPS}"
        logger.info(
            "video.cosmos3: guardrail access check skipped (%s: %s)", type(exc).__name__, exc
        )
        return None
    with _probe_lock:
        _probe_ok[key] = now + _PROBE_OK_TTL_S
    return None


def guardrail_access_refusal(repo_id: str, exc: BaseException) -> Optional[str]:
    """The load error for a gated-repo refusal of the guardrail's own weights, else None."""
    from .diffusion import _HUB_REPO_RE, _gated_in_chain

    gated = _gated_in_chain(exc)
    found = _HUB_REPO_RE.search(str(gated)) if gated is not None else None
    if found is None or found.group(1).rstrip(".").lower() != GUARDRAIL_REPO.lower():
        return None
    return f"'{repo_id}' needs NVIDIA's gated Cosmos guardrail: {_GUARDRAIL_STEPS}"


_UNSET = object()
_load_hub_token: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "cosmos3_guardrail_token", default = _UNSET
)
_wrap_lock = threading.Lock()


def _token_bound(download: Any) -> Any:
    @functools.wraps(download)
    def snapshot_download(*args: Any, **kwargs: Any) -> Any:
        token = _load_hub_token.get()
        if token is not _UNSET:
            kwargs.setdefault("token", token)
        return download(*args, **kwargs)

    snapshot_download._unsloth_load_token = True
    return snapshot_download


@contextlib.contextmanager
def guardrail_hub_token(token: Any):
    """cosmos_guardrail downloads without a token argument; a permanent wrapper sends this load's token from a
    context variable, so overlapping loads never share one. None keeps the ambient login."""
    if isinstance(token, str):
        token = token.strip() or None
    module = None
    if token is not None:
        try:
            # Written out (GUARDRAIL_PACKAGE twice) so the risky-loader lint sees a fixed module.
            module = importlib.import_module("cosmos_guardrail.cosmos_guardrail")
            with _wrap_lock:
                if not getattr(module.snapshot_download, "_unsloth_load_token", False):
                    module.snapshot_download = _token_bound(module.snapshot_download)
        except (ImportError, AttributeError):
            module = None
    if module is None:
        yield
        return
    reset = _load_hub_token.set(token)
    try:
        yield
    finally:
        _load_hub_token.reset(reset)


def pipeline_kwargs(repo_id: str, index: Optional[dict] = None) -> dict[str, Any]:
    """``from_pretrained`` kwargs: no sound tokenizer; guardrail off only on the explicit opt-out."""
    kwargs: dict[str, Any] = dict.fromkeys(NONE_COMPONENTS)
    if safety_checker_opted_out():
        kwargs["enable_safety_checker"] = False
        if checkpoint_enables_safety_checker(repo_id, index):
            logger.warning(
                "video.cosmos3: %s=0, so '%s' loads without the Cosmos guardrail its checkpoint runs by default",
                SAFETY_CHECKER_ENV,
                repo_id,
            )
    return kwargs


def install_untiled_encode(pipe: Any, logger: Any = None) -> bool:
    """Run ``pipe.vae.encode`` untiled, tiled only on out-of-memory: every step attends to the encoded i2v clip,
    so tiling's small shift grows across it (Nano i2v vs diffusers: 24.7 dB tiled)."""
    vae = getattr(pipe, "vae", None)
    encode = getattr(vae, "encode", None)
    if vae is None or not callable(encode):
        return False
    if getattr(encode, "_unsloth_untiled_encode", False):
        return True

    def untiled_encode(*args: Any, **kwargs: Any) -> Any:
        tiling = bool(getattr(vae, "use_tiling", False))
        if not tiling:
            return encode(*args, **kwargs)
        from .diffusion_batched import is_oom_error

        vae.use_tiling = False
        try:
            return encode(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_oom_error(exc):
                raise
        finally:
            vae.use_tiling = tiling
        # Retried outside the handler so the failed attempt's tensors (held by the traceback) are gone first.
        import torch

        torch.cuda.empty_cache()
        if logger is not None:
            logger.warning("video.cosmos3: untiled VAE encode ran out of memory; encoding tiled")
        return encode(*args, **kwargs)

    untiled_encode._unsloth_untiled_encode = True
    vae.encode = untiled_encode
    if logger is not None:
        logger.info(
            "video.cosmos3: VAE encodes the conditioning clip untiled (tiled on out-of-memory)"
        )
    return True


# Wan@16x seam audit floor (test_vae_tile_geometry): >= 4-latent overlaps, >= 8-latent edge tiles.
_MIN_OVERLAP_LATENTS = 4
_MIN_EDGE_LATENTS = 8


def seam_safe_tile(length: int, tile: int, stride: int) -> tuple[int, int]:
    """(tile, stride) latents over ``length`` whose last tile keeps >= 8 latents: stock when it does, else the smallest
    tile (up to 8 past stock) with the largest stride that does; stock if none."""
    if length <= tile:
        return tile, stride
    for t in range(tile, min(length - 1, tile + 8) + 1):
        for s in range(min(stride + t - tile, t - _MIN_OVERLAP_LATENTS), t // 2 - 1, -1):
            if length - (length - 1) // s * s >= _MIN_EDGE_LATENTS:
                return t, s
    return tile, stride


@contextlib.contextmanager
def seam_safe_tile_strides(pipe: Any, height: int, width: int):
    """Tiles for this canvas while tiling runs: stock Wan@16x tiles leave a 4-latent edge tile at 480 / 832 px."""
    vae = getattr(pipe, "vae", None)
    keys = (
        ("tile_sample_min_height", "tile_sample_stride_height"),
        ("tile_sample_min_width", "tile_sample_stride_width"),
    )
    try:
        ratio = int(vae.spatial_compression_ratio)
        old = tuple((int(getattr(vae, t)), int(getattr(vae, s))) for t, s in keys)
    except (AttributeError, TypeError, ValueError):
        yield
        return
    for (tile_key, stride_key), side, (tile, stride) in zip(keys, (height, width), old):
        t, s = seam_safe_tile(int(side) // ratio, tile // ratio, stride // ratio)
        setattr(vae, tile_key, t * ratio)
        setattr(vae, stride_key, s * ratio)
    try:
        yield
    finally:
        for (tile_key, stride_key), (tile, stride) in zip(keys, old):
            setattr(vae, tile_key, tile)
            setattr(vae, stride_key, stride)


# Same caps as a MiniMax-H3 reference video and a keyframe image: 15 s of source, 4096 px a side.
CONTINUATION_MAX_SECONDS = 15.0
CONTINUATION_MAX_SIDE = 4096


def _upright(image: Any, frame: Any) -> Any:
    # PyAV's rotation is counterclockwise degrees, the same sense as PIL's rotate (matches ffmpeg autorotate).
    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
    return image.rotate(rotation, expand = True) if rotation else image


def decode_continuation_clip(blob: bytes, *, keep: str, fps: int, count: int) -> list:
    """Frames land on that clock by whole-frame drop and duplicate, picked by presentation time so a variable-frame-rate
    clip keeps even spacing; a clip already at ``fps`` passes through unchanged."""
    import collections
    import math

    import av

    from .video_minimax_h3 import _open_reference_media

    picked: Any = [] if keep == "first" else collections.deque(maxlen = count)
    try:
        container = _open_reference_media(av, blob)
    except av.error.FFmpegError as exc:
        raise ValueError("That clip could not be read as a video.") from exc
    with container:
        if not container.streams.video:
            raise ValueError("That clip carries no video track.")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.guessed_rate or fps)
        if source_fps <= 0:
            source_fps = float(fps)
        max_source_frames = math.floor(CONTINUATION_MAX_SECONDS * source_fps + 1e-6)
        next_target = 0
        decoded = 0
        held: Any = None
        origin: Optional[float] = None

        def _take(frame: Any, until: float) -> None:
            nonlocal next_target
            while next_target / fps < until - 1e-6:
                if keep != "first" or len(picked) < count:
                    picked.append(frame)
                next_target += 1

        try:
            for index, frame in enumerate(container.decode(video = 0)):
                decoded = index + 1
                stamp = frame.time if frame.time is not None else index / source_fps
                if origin is None:
                    origin = stamp
                # Count and timestamps both: a VFR clip's average rate can understate a long gap.
                if (
                    decoded > max_source_frames
                    or stamp - origin + 1 / source_fps > CONTINUATION_MAX_SECONDS + 1e-6
                ):
                    raise ValueError(
                        f"The clip to continue runs longer than {CONTINUATION_MAX_SECONDS:g} seconds. Trim it first."
                    )
                if max(frame.width, frame.height) > CONTINUATION_MAX_SIDE:
                    raise ValueError(
                        f"The clip to continue is {frame.width}x{frame.height}; the limit is "
                        f"{CONTINUATION_MAX_SIDE} px a side."
                    )
                # "first" still decodes to the end, so both ends are held to the same limits.
                if held is not None:
                    _take(held, max(stamp - origin, 0.0))
                held = frame
            if held is not None:
                # The last frame lasts one nominal frame, as the clip's duration counts it.
                _take(held, max(stamp - origin, 0.0) + 1 / source_fps)
        except av.error.FFmpegError as exc:
            raise ValueError("That clip could not be decoded.") from exc
        frames = [
            _upright(frame.to_image().convert("RGB"), frame) for frame in list(picked)[:count]
        ]
    if len(frames) < count:
        raise ValueError(
            f"This clip has {len(frames)} frames at {fps} fps, and continuing it needs at least {count}. "
            "Use a longer clip."
        )
    return frames


def fit_continuation_frames(frames: list, width: int, height: int) -> list:
    """The pipeline resizes onto the canvas itself, so a clip already at the canvas aspect passes through untouched."""
    if not frames:
        return frames
    src_w, src_h = frames[0].size
    crop_w = min(src_w, round(src_h * width / height))
    crop_h = min(src_h, round(src_w * height / width))
    if (crop_w, crop_h) == (src_w, src_h):
        return frames
    left, top = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    box = (left, top, left + crop_w, top + crop_h)
    return [frame.crop(box) for frame in frames]
