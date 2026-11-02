# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upscaler catalog holds only redistributable weights pinned by SHA-256; others are user-supplied
in workspace upscale_models; a request never carries a path"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

LATENT = "latent"
LANCZOS = "lanczos"
LOCAL_PREFIX = "local:"
# A1111 "Hires upscaler" strings (modules/shared.py latent_upscale_modes, modules/upscaler.py UpscalerLanczos).
BUILTIN_LABELS = {LATENT: "Latent (nearest-exact)", LANCZOS: "Lanczos"}

# ComfyUI ImageUpscaleWithModel tiling: 512 px tiles, 32 px overlap.
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 32
MIN_TILE = 64
# ComfyUI ImageUpscaleWithModel halves the tile on an OOM and gives up below 128 px.
OOM_MIN_TILE = 128

DOWNLOAD_TIMEOUT_S = 600
_SOCKET_TIMEOUT_S = 60


@dataclass(frozen = True)
class CatalogUpscaler:
    id: str
    label: str
    filename: str
    url: str
    size: int
    sha256: str
    scale: int
    license: str


CATALOG: tuple[CatalogUpscaler, ...] = (
    CatalogUpscaler(
        id = "realesrgan_x4plus",
        label = "R-ESRGAN 4x+",
        filename = "RealESRGAN_x4plus.pth",
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size = 67040989,
        sha256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        scale = 4,
        license = "BSD-3-Clause",
    ),
    CatalogUpscaler(
        id = "realesrgan_x4plus_anime_6b",
        label = "R-ESRGAN 4x+ Anime6B",
        filename = "RealESRGAN_x4plus_anime_6B.pth",
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        size = 17938799,
        sha256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        scale = 4,
        license = "BSD-3-Clause",
    ),
)
_CATALOG_BY_ID = {entry.id: entry for entry in CATALOG}
# Loadable without pickle by spandrel and resolvable by sd.cpp --hires-upscalers-dir.
LOCAL_SUFFIXES = (".pth", ".pt", ".safetensors")
_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.+()\-]{0,127}$")


@functools.lru_cache(maxsize = 1)
def spandrel_available() -> bool:
    """Cached: status polls call it, and a failed import is not cached by Python."""
    try:
        import spandrel  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means no model upscalers
        return False
    return True


def models_dir() -> Path:
    from utils.paths.storage_roots import account_path
    return account_path("upscale_models")


def _local_files() -> dict[str, Path]:
    """Drop-in models by stem. Catalog file names are skipped (they show under their catalog id)."""
    try:
        root = models_dir()
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except (OSError, ValueError):
        return {}
    catalog_names = {entry.filename.lower() for entry in CATALOG}
    found: dict[str, Path] = {}
    for path in entries:
        if path.name.lower() in catalog_names or path.name.endswith(".part"):
            continue
        if path.suffix.lower() not in LOCAL_SUFFIXES or not path.is_file():
            continue
        if _STEM_RE.match(path.stem) and path.stem not in found:
            found[path.stem] = path
    return found


def model_label(upscaler_id: Optional[str]) -> Optional[str]:
    if upscaler_id is None:
        return None
    if upscaler_id in BUILTIN_LABELS:
        return BUILTIN_LABELS[upscaler_id]
    entry = _CATALOG_BY_ID.get(upscaler_id)
    if entry is not None:
        return entry.label
    if upscaler_id.startswith(LOCAL_PREFIX):
        return upscaler_id[len(LOCAL_PREFIX) :]
    return upscaler_id


def is_model_upscaler(upscaler_id: Optional[str]) -> bool:
    return bool(upscaler_id) and upscaler_id not in BUILTIN_LABELS


def model_upscalers() -> list[dict[str, Any]]:
    if not spandrel_available():
        return []
    rows = [{"id": e.id, "label": e.label, "scale": e.scale} for e in CATALOG]
    for stem in _local_files():
        # The scale of a drop-in file is unknown until it loads; 4 is the common case and only labels the row.
        rows.append({"id": LOCAL_PREFIX + stem, "label": stem, "scale": 4})
    return rows


def native_model_upscalers() -> list[dict[str, Any]]:
    rows = [{"id": e.id, "label": e.label, "scale": e.scale} for e in CATALOG]
    for stem, path in _local_files().items():
        rows.append({"id": LOCAL_PREFIX + stem, "label": stem, "scale": 4})
    return rows


def check_upscaler_id(upscaler_id: str) -> None:
    """Refuse an id that is not a built-in, a catalog entry or a well-formed drop-in name (ValueError -> 400)."""
    if upscaler_id in BUILTIN_LABELS or upscaler_id in _CATALOG_BY_ID:
        return
    if upscaler_id.startswith(LOCAL_PREFIX):
        stem = upscaler_id[len(LOCAL_PREFIX) :]
        if _STEM_RE.match(stem) and ".." not in stem:
            return
    raise ValueError(
        f"Unknown upscaler '{upscaler_id}'. Pick one the Images page lists, or drop a model file into the "
        "upscale_models folder."
    )


def _offline() -> bool:
    """Every offline switch the backend honours: HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and a force_hf_offline window."""
    from utils.utils import hf_env_offline
    return hf_env_offline()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_download_lock = threading.Lock()
# (path, size, mtime) -> verified, so a resident file is hashed once per process.
_verified: set[tuple[str, int, int]] = set()


def _verified_file(path: Path, entry: CatalogUpscaler) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size != entry.size:
        return False
    key = (str(path), int(st.st_size), int(st.st_mtime_ns))
    if key in _verified:
        return True
    if _sha256(path) != entry.sha256:
        return False
    _verified.add(key)
    return True


def _raise_cancelled() -> None:
    """Stop raises the canonical message, which the generate route maps to its 409 cancellation."""
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    raise RuntimeError(DIFFUSION_CANCELLED_MSG)


def _download(
    entry: CatalogUpscaler,
    dest: Path,
    cancel: Any = None,
) -> None:
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(entry.url, headers = {"User-Agent": "Unsloth-Studio"})
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_S
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            urllib.request.urlopen(request, timeout = _SOCKET_TIMEOUT_S) as response,
            part.open("wb") as handle,
        ):
            # read1: at most one socket read per call, so Stop and the deadline are rechecked between reads even
            # when the server trickles bytes (read(n) waits until n bytes arrive).
            read = getattr(response, "read1", None) or response.read
            while True:
                if cancel is not None and cancel.is_set():
                    _raise_cancelled()
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Timed out downloading the {entry.label} upscaler.")
                chunk = read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > entry.size:
                    raise RuntimeError(
                        f"The {entry.label} upscaler download is larger than expected."
                    )
                digest.update(chunk)
                handle.write(chunk)
        if total != entry.size or digest.hexdigest() != entry.sha256:
            raise RuntimeError(
                f"The {entry.label} upscaler download failed its integrity check (size or SHA-256)."
            )
        os.replace(part, dest)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Could not download the {entry.label} upscaler: {exc}") from exc
    finally:
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass


def resolve_model_path(upscaler_id: str, *, cancel: Any = None) -> Path:
    check_upscaler_id(upscaler_id)
    if upscaler_id in BUILTIN_LABELS:
        raise ValueError(f"'{upscaler_id}' is not a model upscaler.")
    entry = _CATALOG_BY_ID.get(upscaler_id)
    if entry is None:
        stem = upscaler_id[len(LOCAL_PREFIX) :]
        path = _local_files().get(stem)
        if path is None:
            raise ValueError(f"Upscaler model '{stem}' is not in the upscale_models folder.")
        return path
    root = models_dir()
    dest = root / entry.filename
    with _download_lock:
        if _verified_file(dest, entry):
            return dest
        if _offline():
            raise ValueError(
                f"The {entry.label} upscaler is not downloaded yet and Hugging Face offline mode is on."
            )
        root.mkdir(parents = True, exist_ok = True)
        logger.info("diffusion.upscaler: downloading %s (%d bytes)", entry.filename, entry.size)
        _download(entry, dest, cancel)
        _verified_file(dest, entry)
    return dest


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering ``[0, length)`` with windows of ``tile`` that overlap by at least ``overlap``; the last
    window ends exactly at ``length``. One window when ``length <= tile``."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    count = int(math.ceil((length - tile) / stride)) + 1
    starts = [min(i * stride, length - tile) for i in range(count)]
    out: list[int] = []
    for s in starts:
        if not out or s != out[-1]:
            out.append(s)
    return out


def tile_boxes(width: int, height: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    tw, th = min(tile, width), min(tile, height)
    return [
        (x, y, x + tw, y + th)
        for y in tile_starts(height, th, overlap)
        for x in tile_starts(width, tw, overlap)
    ]


def _feather(length: int, ramp: int, at_start: bool, at_end: bool, torch: Any, device: Any) -> Any:
    w = torch.ones(length, device = device, dtype = torch.float32)
    ramp = min(ramp, length // 2)
    if ramp > 0:
        r = torch.arange(1, ramp + 1, device = device, dtype = torch.float32) / (ramp + 1)
        if not at_start:
            w[:ramp] = r
        if not at_end:
            w[-ramp:] = r.flip(0)
    return w


def tiled_apply(
    image: Any,
    fn: Callable[[Any], Any],
    scale: int,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    cancel: Any = None,
    output_device: Any = None,
) -> Any:
    """Run ``fn`` (BCHW -> BC(sH)(sW)) over overlapping tiles, blending seams linearly in float32.

    The full-scale result accumulates on ``output_device`` (default: the input's device); ComfyUI's
    ImageUpscaleWithModel keeps it on the intermediate (CPU) device the same way, so only one tile is on the GPU."""
    import torch

    _, c, h, w = image.shape
    out_dev = image.device if output_device is None else torch.device(output_device)
    boxes = tile_boxes(w, h, tile, overlap)
    if len(boxes) == 1:
        result = fn(image).float().to(out_dev)
        if cancel is not None and cancel.is_set():
            _raise_cancelled()
        return result
    out = None
    weight = torch.zeros((1, 1, h * scale, w * scale), device = out_dev, dtype = torch.float32)
    ramp = overlap * scale
    for x0, y0, x1, y1 in boxes:
        if cancel is not None and cancel.is_set():
            _raise_cancelled()
        piece = fn(image[:, :, y0:y1, x0:x1]).float().to(out_dev)
        if out is None:
            out = torch.zeros(
                (image.shape[0], piece.shape[1], h * scale, w * scale),
                device = out_dev,
                dtype = torch.float32,
            )
        wy = _feather(piece.shape[2], ramp, y0 == 0, y1 == h, torch, out_dev)
        wx = _feather(piece.shape[3], ramp, x0 == 0, x1 == w, torch, out_dev)
        mask = (wy[:, None] * wx[None, :])[None, None]
        ys, xs = y0 * scale, x0 * scale
        out[:, :, ys : ys + piece.shape[2], xs : xs + piece.shape[3]] += piece * mask
        weight[:, :, ys : ys + piece.shape[2], xs : xs + piece.shape[3]] += mask
    # Stop may land while the last tile runs: never hand back a result for an accepted cancel.
    if cancel is not None and cancel.is_set():
        _raise_cancelled()
    return out / weight.clamp_min(1e-8)


_model_lock = threading.Lock()
_resident: dict[tuple, Any] = {}


def release_models() -> int:
    with _model_lock:
        count = len(_resident)
        _resident.clear()
    if count:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return count


def _pick_dtype(descriptor: Any, device: Any) -> Any:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return torch.float32
    if getattr(descriptor, "supports_half", False):
        return torch.float16
    if getattr(descriptor, "supports_bfloat16", False):
        return torch.bfloat16
    return torch.float32


def _load(path: Path, device: Any) -> tuple[Any, Any]:
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader(device = "cpu").load_from_file(str(path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"'{path.name}' is not a single-image upscaler model.")
    if int(getattr(descriptor, "input_channels", 3)) != 3:
        raise ValueError(
            f"'{path.name}' takes {descriptor.input_channels} channels; only RGB upscalers are supported."
        )
    dtype = _pick_dtype(descriptor, device)
    descriptor.model.to(device = device, dtype = dtype)
    descriptor.model.eval()
    descriptor.model.requires_grad_(False)
    return descriptor, dtype


@dataclass
class UpscaleOutcome:
    image: Any
    model_scale: int
    tiles: int
    dtype: str
    seconds: float
    released: bool


def upscale_with_model(
    image: Any,
    upscaler_id: str,
    target: tuple[int, int],
    *,
    device: Any,
    keep_resident: bool = False,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    cancel: Any = None,
) -> UpscaleOutcome:
    """Model upscale, then resize to ``target`` on the device (ComfyUI ImageUpscaleWithModel -> ImageScale)."""
    import numpy as np
    import torch
    from PIL import Image

    if not spandrel_available():
        raise ValueError(
            "Model upscalers need the spandrel package, which is not installed. Pick Lanczos instead."
        )
    t0 = time.perf_counter()
    path = resolve_model_path(upscaler_id, cancel = cancel)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    st = path.stat()
    # The file's size and mtime too: a drop-in overwritten in place loads again (A1111 reloads on every upscale).
    key = (str(path), int(st.st_size), int(st.st_mtime_ns), str(dev), "auto")
    with _model_lock:
        cached = _resident.get(key)
        if cached is None:
            _resident.clear()
            cached = _load(path, dev)
            if keep_resident:
                _resident[key] = cached
        descriptor, dtype = cached
        rgb = image.convert("RGB")
        arr = torch.from_numpy(np.asarray(rgb, dtype = np.uint8).copy())
        x = arr.permute(2, 0, 1).unsqueeze(0).to(device = dev, dtype = torch.float32) / 255.0
        scale = int(descriptor.scale)

        def _run(piece: Any) -> Any:
            # Not descriptor.model: the descriptor pads, unwraps tuple outputs (MMRealSR) and clamps.
            return descriptor(piece.to(dtype)).float()

        boxes = tile_boxes(x.shape[3], x.shape[2], tile, overlap)
        try:
            with torch.inference_mode():
                # A 4x model on a 2048 px source is an 8192 px float32 result (~1 GiB with its weights): keep it in
                # host memory, so the GPU holds the model and one tile, as in ComfyUI.
                while True:
                    try:
                        y = tiled_apply(
                            x,
                            _run,
                            scale,
                            tile = tile,
                            overlap = overlap,
                            cancel = cancel,
                            output_device = "cpu",
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - only an allocator failure retries
                        from core.inference.diffusion_vae_oom import is_oom

                        if not is_oom(exc) or tile // 2 < OOM_MIN_TILE:
                            raise
                        tile //= 2
                        logger.info(
                            "diffusion.upscaler: out of memory, retrying with %d px tiles", tile
                        )
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        boxes = tile_boxes(x.shape[3], x.shape[2], tile, overlap)
        finally:
            if not keep_resident:
                del descriptor, cached
                _resident.pop(key, None)
        tw, th = int(target[0]), int(target[1])
        if (y.shape[3], y.shape[2]) != (tw, th):
            # Resized where it accumulated (host), as ComfyUI's ImageScale after ImageUpscaleWithModel.
            y = torch.nn.functional.interpolate(y, size = (th, tw), mode = "bicubic", antialias = True)
        out = (y.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray(out, mode = "RGB")
    del x, y
    released = not keep_resident
    if released and dev.type == "cuda":
        torch.cuda.empty_cache()
    return UpscaleOutcome(
        image = result,
        model_scale = scale,
        tiles = len(boxes),
        dtype = str(dtype).replace("torch.", ""),
        seconds = time.perf_counter() - t0,
        released = released,
    )


def iter_catalog() -> Iterator[CatalogUpscaler]:
    return iter(CATALOG)
