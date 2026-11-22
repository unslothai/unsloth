"""Reference-image preprocessing for FLUX.2 edits on the GPU, byte-identical to the CPU path.

The FLUX.2 pipelines prepare each condition image on the CPU: a Pillow Lanczos resize to about 1 MP when the image is
larger (`Flux2ImageProcessor._resize_to_target_area`), then `preprocess(..., resize_mode="crop")` (a centre crop to a
multiple of 16, numpy `/ 255`, `2 x - 1`). For a 1-4 MP reference that is 28-35 ms per image on one core.

`install(pipe)` swaps those two calls on the pipeline's own image processor instance:
- `_resize_to_target_area` returns a `DeferredResize` (the source image plus the size Pillow would produce) instead of
  resizing; any attribute other than `size` / `width` / `height` materialises the Pillow resize, so other callers see
  a normal image.
- `preprocess` for one image with `resize_mode="crop"` uploads the uint8 pixels once, runs the Lanczos resize and the
  crop in Triton (Pillow's fixed-point coefficients and rounding, `diffusion_lanczos_triton`), and maps codes through
  a 256-entry table computed with diffusers' own ops. The result is `torch.equal` to the CPU tensor, same strides.
Everything else (lists, other resize modes, crops coordinates, a failed probe, any GPU error) runs the stock method.

The first GPU call per device runs a probe against Pillow on a fixed image; a mismatch keeps that device on the CPU
path for the process. NVIDIA CUDA only (the kernel uses CUDA integer semantics and is not validated on ROCm).
`UNSLOTH_DIFFUSION_GPU_REF_PREPROCESS=0` turns it off.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Optional

ENV = "UNSLOTH_DIFFUSION_GPU_REF_PREPROCESS"
_MARK = "_unsloth_gpu_ref_preprocess"
# Engagement counters: GPU-prepared images, stock fallbacks after the GPU path was tried, probe results per device.
STATS = {"gpu": 0, "fallback": 0}
_PROBED: dict = {}
_TABLES: dict = {}
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _cuda_ok() -> bool:
    try:
        import torch
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return False
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001 - no torch / triton: stock path
        return False
    return True


class DeferredResize:
    """What the patched `_resize_to_target_area` returns: the source image plus the size Pillow would produce.

    The pipeline reads only `.size` before handing it to `preprocess`. Anything else resizes on the CPU exactly as the
    stock method would and forwards to that image."""

    def __init__(self, image: Any, width: int, height: int):
        self.source = image
        self.size = (int(width), int(height))
        self.width, self.height = self.size
        self._resized = None

    def materialize(self):
        if self._resized is None:
            from PIL import Image
            self._resized = self.source.resize(self.size, Image.Resampling.LANCZOS)
        return self._resized

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("source", "size", "width", "height", "_resized"):
            raise AttributeError(name)
        return getattr(self.materialize(), name)


def _target_size(image: Any, target_area: int) -> tuple:
    # Flux2ImageProcessor._resize_to_target_area's arithmetic.
    width, height = image.size
    scale = math.sqrt(target_area / (width * height))
    return int(width * scale), int(height * scale)


def _table(device):
    """uint8 code -> diffusers' normalised float32: np.float32 array / 255.0, then torch 2.0 * x - 1.0."""
    import numpy as np
    import torch

    key = str(device)
    t = _TABLES.get(key)
    if t is None:
        a = np.arange(256, dtype = np.uint8).astype(np.float32) / 255.0
        t = (2.0 * torch.from_numpy(a) - 1.0).to(device)
        _TABLES[key] = t
    return t


def _gpu_preprocess(
    proc: Any, image: Any, height: Optional[int], width: Optional[int], device
) -> Any:
    """`Flux2ImageProcessor.preprocess(image, height, width, resize_mode="crop")` for one RGB image (or a
    DeferredResize of one) on `device`: float32 [1, 3, H, W] with the CPU path's strides."""
    import numpy as np
    import torch

    src = image.source if isinstance(image, DeferredResize) else image
    if getattr(src, "mode", "RGB") != "RGB":
        src = src.convert("RGB")
    w1, h1 = image.size
    height, width = proc.get_default_height_width(_SizeOnly(w1, h1), height, width)
    if width > w1 or height > h1:
        raise ValueError("crop larger than the image")
    # Flux2ImageProcessor._resize_and_crop is a pure centre crop (no second resize).
    left, top = (w1 - width) // 2, (h1 - height) // 2
    host = torch.from_numpy(np.asarray(src, dtype = np.uint8))
    device = torch.device(device)
    px = (
        host.pin_memory().to(device, non_blocking = True)
        if device.type == "cuda"
        else host.to(device)
    )
    if isinstance(image, DeferredResize):
        from .diffusion_lanczos_triton import lanczos_resize
        out = lanczos_resize(px, h1, w1, crop = (top, left, height, width))
    else:
        out = px[top : top + height, left : left + width]
    f = _table(device)[out.long()]  # [H, W, 3] float32
    return f.unsqueeze(0).permute(0, 3, 1, 2)


def _is_pil(image: Any) -> bool:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return False
    return isinstance(image, Image.Image)


class _SizeOnly:
    def __init__(self, w: int, h: int):
        self.size = (w, h)
        self.width, self.height = w, h


def _probe(proc: Any, stock_area, stock_pre, device) -> bool:
    """One resize + crop against the stock CPU path on a fixed noise image; True when torch.equal with equal strides."""
    import numpy as np
    import torch
    from PIL import Image

    rng = np.random.default_rng(1234)
    img = Image.fromarray(rng.integers(0, 256, size = (601, 1203, 3), dtype = np.uint8), "RGB")
    target = 512 * 256
    ref_img = stock_area(img, target)
    w, h = ref_img.size
    w, h = w // 16 * 16, h // 16 * 16
    ref = stock_pre(ref_img, height = h, width = w, resize_mode = "crop")
    got = _gpu_preprocess(proc, DeferredResize(img, *_target_size(img, target)), h, w, device)
    return bool(torch.equal(got.cpu(), ref) and got.stride() == ref.stride())


def _config_ok(proc: Any) -> bool:
    """The stock PIL branch this path reproduces: resize (crop), RGB, normalise to [-1, 1], nothing else."""
    c = getattr(proc, "config", None)
    try:
        return bool(
            c.do_resize
            and c.do_normalize
            and c.do_convert_rgb
            and not c.do_binarize
            and not c.do_convert_grayscale
        )
    except AttributeError:
        return False


def _device_of(pipe: Any):
    dev = getattr(pipe, "_execution_device", None)
    return dev if dev is not None and getattr(dev, "type", "") == "cuda" else None


def install(pipe: Any, logger: Any = None) -> bool:
    """Patch `pipe.image_processor` (a Flux2ImageProcessor) in place. True when installed."""
    if not enabled() or not _cuda_ok():
        return False
    proc = getattr(pipe, "image_processor", None)
    if proc is None or type(proc).__name__ != "Flux2ImageProcessor" or getattr(proc, _MARK, False):
        return False
    if not callable(getattr(proc, "_resize_to_target_area", None)) or not callable(
        getattr(proc, "preprocess", None)
    ):
        return False
    stock_area = proc._resize_to_target_area
    stock_pre = proc.preprocess

    def _resize_to_target_area(image, target_area = 1024 * 1024):
        if not enabled() or _device_of(pipe) is None or not _is_pil(image):
            return stock_area(image, target_area)
        return DeferredResize(image, *_target_size(image, target_area))

    def preprocess(
        image,
        height = None,
        width = None,
        resize_mode = "default",
        crops_coords = None,
        *args,
        **kwargs,
    ):
        device = _device_of(pipe)
        single = isinstance(image, DeferredResize) or _is_pil(image)
        if (
            enabled()
            and device is not None
            and single
            and resize_mode == "crop"
            and crops_coords is None
            and height is not None
            and width is not None
            and not args
            and not kwargs
            and _config_ok(proc)
        ):
            key = str(device)
            ok = _PROBED.get(key)
            if ok is None:
                with _LOCK:
                    ok = _PROBED.get(key)
                    if ok is None:
                        try:
                            ok = _probe(proc, stock_area, stock_pre, device)
                        except Exception as exc:  # noqa: BLE001 - probe failure: CPU path for this device
                            ok = False
                            if logger is not None:
                                logger.warning(
                                    "diffusion.ref_preprocess: probe failed on %s: %s", key, exc
                                )
                        _PROBED[key] = ok
                        if logger is not None:
                            logger.info(
                                "diffusion.ref_preprocess: GPU reference preprocessing %s on %s",
                                "on" if ok else "off (probe mismatch)",
                                key,
                            )
            if ok:
                try:
                    out = _gpu_preprocess(proc, image, height, width, device)
                    STATS["gpu"] += 1
                    return out
                except Exception as exc:  # noqa: BLE001 - any GPU error: stock path for this image
                    STATS["fallback"] += 1
                    if logger is not None:
                        logger.warning("diffusion.ref_preprocess: CPU fallback: %s", exc)
        if isinstance(image, DeferredResize):
            image = image.materialize()
        return stock_pre(image, height, width, resize_mode, crops_coords, *args, **kwargs)

    proc._resize_to_target_area = _resize_to_target_area
    proc.preprocess = preprocess
    setattr(proc, _MARK, True)
    if logger is not None:
        logger.info("diffusion.ref_preprocess: installed on %s", type(pipe).__name__)
    return True
