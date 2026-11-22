# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU FLUX.2 edit-reference preprocessing (Lanczos resize + crop + normalise), byte-identical to the CPU path.

`install(pipe)` patches the pipe's image processor; a per-device probe against Pillow gates it. NVIDIA CUDA only
(kernel not validated on ROCm). `UNSLOTH_DIFFUSION_GPU_REF_PREPROCESS=0` turns it off.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Optional

ENV = "UNSLOTH_DIFFUSION_GPU_REF_PREPROCESS"
_MARK = "_unsloth_gpu_ref_preprocess"
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
    """Source image plus Pillow's target size; any attribute but `.size` materialises the CPU resize."""

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
    f = _table(device)[out.long()]
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
