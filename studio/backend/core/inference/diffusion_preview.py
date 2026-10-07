# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live latent previews for the image and video denoise loops.

Each snapshot projects the latent to a small RGB picture with a fitted per-family linear map
(``diffusion_preview_factors``), copies it into a pinned slot with ``non_blocking=True`` plus a CUDA
event, and a worker thread JPEG-encodes it once ``Event.query()`` says it landed. The denoise thread
never syncs or reads host memory the GPU is writing, every CUDA call sits inside the CUDA-graph capture
hold-off, and the latent is only read, so the final image is identical with previews on or off.

Snapshots are planned by STEP, not host time: under CUDA graphs the host enqueues every step long
before the GPU runs it. Kill switch: ``UNSLOTH_DIFFUSION_PREVIEW=0``.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import io
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PREVIEW_ENV = "UNSLOTH_DIFFUSION_PREVIEW"
MAX_SIDE = 256
MIN_INTERVAL_S = 0.25
MAX_SNAPSHOTS = 24
JPEG_QUALITY = 80


@dataclass(frozen = True)
class LatentRGB:
    """layout: "tokens" (B, N, D) | "bchw" | "bcthw"; down: output px per grid cell; patch: RGB px per
    cell side (2 for 2x2 packed tokens); weight: D rows of 3*patch*patch outputs ordered (dy, dx, rgb)."""

    layout: str
    down: int
    patch: int
    weight: tuple
    bias: tuple

    @property
    def channels(self) -> int:
        return len(self.weight)


def env_enabled() -> bool:
    return os.environ.get(PREVIEW_ENV, "").strip().lower() not in ("0", "false", "off", "no")


def preview_wanted(requested: Optional[bool]) -> bool:
    if not env_enabled():
        return False
    return True if requested is None else bool(requested)


@functools.lru_cache(maxsize = 64)
def factors_for(family: Optional[str]) -> Optional[LatentRGB]:
    """Cached so a family keeps one object, and its device copy is uploaded once per process."""
    if not family:
        return None
    from .diffusion_preview_factors import FACTORS, FAMILY_FACTORS

    key = FAMILY_FACTORS.get(str(family).strip().lower())
    entry = FACTORS.get(key) if key else None
    if not entry:
        return None
    return LatentRGB(
        layout = entry["layout"],
        down = int(entry["down"]),
        patch = int(entry["patch"]),
        weight = tuple(tuple(float(v) for v in row) for row in entry["weight"]),
        bias = tuple(float(v) for v in entry["bias"]),
    )


def latent_grid(latents: Any, spec: LatentRGB, height: int, width: int) -> Any:
    """First image / frame as a (gh, gw, D) view, or None. Packed tokens take their grid from the render
    size, which must tile the sequence exactly."""
    if spec.layout == "tokens":
        gh, gw = int(height) // spec.down, int(width) // spec.down
        if latents.ndim != 3 or latents.shape[-1] != spec.channels or gh <= 0 or gw <= 0:
            return None
        n = int(latents.shape[1])
        if n < gh * gw or n % (gh * gw):
            return None
        return latents[0, : gh * gw].reshape(gh, gw, spec.channels)
    if spec.layout == "bchw":
        if latents.ndim != 4 or latents.shape[1] != spec.channels:
            return None
        return latents[0].permute(1, 2, 0)
    if spec.layout == "bcthw":
        if latents.ndim != 5 or latents.shape[1] != spec.channels:
            return None
        return latents[0, :, 0].permute(1, 2, 0)
    return None


def project(grid: Any, weight: Any, bias: Any, patch: int) -> Any:
    gh, gw, d = grid.shape
    rgb = grid.reshape(gh * gw, d).float() @ weight + bias
    if patch == 1:
        return rgb.reshape(gh, gw, 3)
    rgb = rgb.reshape(gh, gw, patch, patch, 3).permute(0, 2, 1, 3, 4)
    return rgb.reshape(gh * patch, gw * patch, 3)


def smooth_patch_grid(rgb: Any) -> Any:
    """[1, 2, 1]^2 / 16 blur: zero response at Nyquist, so it removes the period-2 grid that per-pixel
    row mismatches of a 2x2 packed-token map print, and barely softens the rest."""
    import torch
    import torch.nn.functional as F

    chw = rgb.permute(2, 0, 1).unsqueeze(1)
    # Built on device: a host tensor would be a pageable, synchronising copy.
    kernel = torch.full((1, 1, 3, 3), 1.0 / 16.0, device = rgb.device, dtype = rgb.dtype)
    kernel[..., 1, :] *= 2.0
    kernel[..., :, 1] *= 2.0
    out = F.conv2d(F.pad(chw, (1, 1, 1, 1), mode = "replicate"), kernel)
    return out.squeeze(1).permute(1, 2, 0)


def to_uint8(rgb: Any, max_side: int = MAX_SIDE) -> Any:
    import torch
    import torch.nn.functional as F

    h, w, _ = rgb.shape
    k = max(1, math.ceil(max(h, w) / max(1, int(max_side))))
    if k > 1:
        chw = rgb.permute(2, 0, 1).unsqueeze(0)
        chw = F.avg_pool2d(chw, kernel_size = k, stride = k)
        rgb = chw.squeeze(0).permute(1, 2, 0)
    return (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous()


_DEVICE_MAPS: dict = {}
_DEVICE_MAPS_LOCK = threading.Lock()


def _device_map(spec: LatentRGB, device: Any) -> tuple:
    """Uploaded from pinned memory: a pageable host-to-device copy would synchronise."""
    import torch

    key = (id(spec.weight), str(device))
    with _DEVICE_MAPS_LOCK:
        hit = _DEVICE_MAPS.get(key)
        if hit is not None and hit[0] is spec.weight:
            return hit[1], hit[2]
    w = torch.tensor(spec.weight, dtype = torch.float32).pin_memory().to(device, non_blocking = True)
    b = torch.tensor(spec.bias, dtype = torch.float32).pin_memory().to(device, non_blocking = True)
    with _DEVICE_MAPS_LOCK:
        _DEVICE_MAPS[key] = (spec.weight, w, b)
    return w, b


def x0_estimate(prev: Any, cur: Any, pair: Optional[tuple]) -> Any:
    """Euler x0 of the previous step: ``prev * (1 + c) - cur * c``, ``c = s_prev / (s_cur - s_prev)``.
    Weights sum to 1, so it commutes with the affine map. ``pair`` may be 0-dim device tensors (never read back)."""
    if prev is None or pair is None:
        return cur
    s_prev, s_cur = pair
    delta = s_cur - s_prev
    if isinstance(delta, float):
        if abs(delta) <= 1e-6:
            return cur
        c = s_prev / delta
        return prev * (1.0 + c) - cur * c
    import torch

    ok = delta.abs() > 1e-6
    c = torch.where(
        ok, s_prev / torch.where(ok, delta, torch.ones_like(delta)), torch.zeros_like(delta)
    )
    return prev * (1.0 + c) - cur * c


def _hold_off():
    try:
        from .diffusion_cuda_graph import hold_off_capture
        return hold_off_capture()
    except Exception:  # noqa: BLE001 - no graph layer means no capture to collide with
        return contextlib.nullcontext(True)


def _sigma_pair(scheduler: Any) -> Optional[tuple]:
    """(s_prev, s_cur) for the step that just ran; a device sigmas tensor is indexed, never read."""
    sigmas = getattr(scheduler, "sigmas", None)
    idx = getattr(scheduler, "_step_index", None)
    if sigmas is None or not isinstance(idx, int) or idx < 1:
        return None
    try:
        if idx >= int(sigmas.shape[0]):
            return None
        if sigmas.device.type == "cpu":
            return float(sigmas[idx - 1]), float(sigmas[idx])
        return sigmas[idx - 1].float(), sigmas[idx].float()
    except Exception:  # noqa: BLE001 - an odd scheduler just gets the raw-latent preview
        return None


class LatentPreviewer:
    """``on_step`` runs on the denoise thread after each scheduler step; it never raises or blocks."""

    def __init__(
        self,
        spec: LatentRGB,
        *,
        height: int,
        width: int,
        device: Any,
        publish: Callable[[str, int], None],
        total_steps: int = MAX_SNAPSHOTS,
        max_side: int = MAX_SIDE,
        min_interval_s: float = MIN_INTERVAL_S,
        max_snapshots: int = MAX_SNAPSHOTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import torch

        self.spec = spec
        self.height, self.width = int(height), int(width)
        self.device = torch.device(device)
        self.max_side = int(max_side)
        self.min_interval_s = float(min_interval_s)
        self._clock = clock
        self._publish = publish
        self._weight, self._bias = _device_map(spec, self.device)
        # Pre-sized with margin for pipelines that round up: no allocation in the loop.
        side_h = (self.height // spec.down + 2) * spec.patch
        side_w = (self.width // spec.down + 2) * spec.patch
        cap = min(side_h, self.max_side) * min(side_w, self.max_side) * 3
        total = max(1, int(total_steps or 1))
        self.stride = max(1, math.ceil(total / max(1, int(max_snapshots))))
        n_slots = total // self.stride + 2
        pinned = torch.empty(n_slots * cap, dtype = torch.uint8, pin_memory = True)
        self._slots = [
            {
                "host": pinned[i * cap : (i + 1) * cap],
                "event": torch.cuda.Event(),
                "state": "free",
                "shape": None,
                "step": 0,
            }
            for i in range(n_slots)
        ]
        self._slot_lock = threading.Lock()
        self._prev: Any = None
        self._prev_index: Optional[int] = None
        self._steps_seen = 0
        self._last_publish = -1e30
        self._seq = 0
        self.emitted = 0
        self.published = 0
        self.skipped_capture = 0
        self.failed = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker = threading.Thread(target = self._run, name = "diffusion-preview", daemon = True)
        self._worker.start()

    @classmethod
    def create(
        cls,
        *,
        family: Optional[str],
        requested: Optional[bool],
        height: Optional[int],
        width: Optional[int],
        device: Any,
        publish: Callable[[str, int], None],
        **kwargs: Any,
    ) -> Optional["LatentPreviewer"]:
        if not preview_wanted(requested):
            return None
        spec = factors_for(family)
        if spec is None or not height or not width:
            return None
        try:
            import torch

            dev = torch.device(device)
            if dev.type != "cuda" or not torch.cuda.is_available():
                # No event to poll off CUDA, so a preview would block.
                return None
            with _hold_off() as clear:
                if not clear:
                    return None
                return cls(spec, height = height, width = width, device = dev, publish = publish, **kwargs)
        except Exception as exc:  # noqa: BLE001 - previews are optional
            logger.debug("diffusion.preview: disabled for this render (%s)", exc)
            return None

    def on_step(
        self,
        latents: Any,
        scheduler: Any = None,
        final: bool = False,
    ) -> None:
        if self.failed or latents is None:
            return
        try:
            self._on_step(latents, scheduler, final)
        except Exception as exc:  # noqa: BLE001 - a preview must never fail a render
            self.failed = True
            logger.debug("diffusion.preview: stopped after an error (%s)", exc)

    def _on_step(self, latents: Any, scheduler: Any, final: bool) -> None:
        import torch
        if getattr(latents, "device", None) is None or latents.device.type != "cuda":
            return
        with _hold_off() as clear:
            if not clear:
                self.skipped_capture += 1
                self._prev = None
                return
            grid = latent_grid(latents, self.spec, self.height, self.width)
            if grid is None:
                self.failed = True
                return
            with torch.no_grad():
                cur = project(grid, self._weight, self._bias, self.spec.patch)
                index = getattr(scheduler, "_step_index", None)
                prev, prev_index = self._prev, self._prev_index
                self._prev, self._prev_index = cur, index
                if not isinstance(index, int) or prev_index != index - 1:
                    prev = None
                if prev is None and not final and _sigma_pair(scheduler) is not None:
                    return
                self._steps_seen += 1
                if not final and (self._steps_seen - 1) % self.stride:
                    return
                slot = self._free_slot()
                if slot is None:
                    return
                pair = _sigma_pair(scheduler) if prev is not None else None
                shown = x0_estimate(prev, cur, pair)
                if self.spec.patch > 1:
                    shown = smooth_patch_grid(shown)
                img = to_uint8(shown, self.max_side)
                n = img.numel()
                if n > slot["host"].numel():
                    return
                slot["host"][:n].copy_(img.view(-1), non_blocking = True)
                slot["event"].record(torch.cuda.current_stream(self.device))
                slot["shape"] = tuple(img.shape)
                slot["step"] = self._seq = self._seq + 1
                slot["state"] = "copying"
                self.emitted += 1

    def _free_slot(self) -> Optional[dict]:
        with self._slot_lock:
            for slot in self._slots:
                if slot["state"] == "free":
                    return slot
        return None

    def _harvest(self) -> list:
        ready = []
        with _hold_off() as clear:
            if not clear:
                return ready
            for slot in self._slots:
                with self._slot_lock:
                    if slot["state"] != "copying":
                        continue
                if slot["event"].query():
                    with self._slot_lock:
                        slot["state"] = "encoding"
                    ready.append(slot)
        return ready

    def finish(self, timeout: float = 1.0) -> None:
        """One last pass for landed copies, then stop the worker; an in-flight copy is dropped."""
        self._prev = None
        self._stop.set()
        self._wake.set()
        if self._worker is not threading.current_thread():
            self._worker.join(timeout = timeout)

    def _run(self, poll_s: float = 0.05) -> None:
        """Publish the newest landed snapshot at most every ``min_interval_s``; drop older ones unencoded."""
        while True:
            stopping = self._stop.is_set()
            try:
                ready = sorted(self._harvest(), key = lambda slot: slot["step"])
                if ready:
                    newest = ready[-1]
                    for slot in ready[:-1]:
                        self._release(slot)
                    if stopping or self._clock() - self._last_publish >= self.min_interval_s:
                        self._encode(newest)
                        self._last_publish = self._clock()
                    else:
                        self._release(newest, back_to = "copying")
            except Exception as exc:  # noqa: BLE001 - a preview must never fail a render
                logger.debug("diffusion.preview: harvest failed (%s)", exc)
            if stopping:
                return
            self._wake.wait(poll_s)
            self._wake.clear()

    def _release(
        self,
        slot: dict,
        back_to: str = "free",
    ) -> None:
        with self._slot_lock:
            slot["state"] = back_to

    def _encode(self, slot: dict) -> None:
        try:
            h, w, _ = slot["shape"]
            raw = slot["host"][: h * w * 3].numpy().tobytes()
            url = encode_jpeg(raw, w, h)
            self._publish(url, int(slot["step"]))
            self.published += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("diffusion.preview: encode failed (%s)", exc)
        finally:
            with self._slot_lock:
                slot["state"] = "free"


def encode_jpeg(
    raw: bytes,
    width: int,
    height: int,
    quality: int = JPEG_QUALITY,
) -> str:
    from PIL import Image

    img = Image.frombytes("RGB", (int(width), int(height)), raw)
    buf = io.BytesIO()
    img.save(buf, format = "JPEG", quality = int(quality))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@contextlib.contextmanager
def scheduler_step_preview(pipe: Any, previewer: Optional[LatentPreviewer]):
    """Feed ``previewer`` from ``scheduler.step`` for pipelines without a step callback (HunyuanVideo-1.5)."""
    scheduler = getattr(pipe, "scheduler", None)
    original = getattr(scheduler, "step", None)
    if previewer is None or scheduler is None or not callable(original):
        yield
        return
    had_own = "step" in getattr(scheduler, "__dict__", {})

    def _step(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        try:
            sample = out[0] if isinstance(out, tuple) else getattr(out, "prev_sample", None)
            previewer.on_step(sample, scheduler)
        except Exception:  # noqa: BLE001
            pass
        return out

    scheduler.step = _step
    try:
        yield
    finally:
        try:
            if had_own:
                scheduler.step = original
            else:
                del scheduler.step
        except Exception:  # noqa: BLE001
            pass
