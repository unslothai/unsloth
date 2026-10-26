# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Convert a decoded clip to uint8 RGB frames on its own device, bit-identical to the np / pil export
(float32 ``(x * 255).round()`` matches numpy); out-of-range clips or non CUDA / CPU devices use the original path."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

_SLICE_BYTES = 64 << 20
# Elements per finiteness slice: the bool temporary stays at 16 MiB however long the clip.
_FINITE_SLICE_ELEMS = 16 << 20
# "0" turns off the in-render NaN / inf refusal (decode output and postprocess input); the encoder's check remains.
NONFINITE_CHECK_ENV = "UNSLOTH_VIDEO_NONFINITE_CHECK"


def device_uint8(frames: Any) -> Optional[Any]:
    """(F, C, H, W) float frames in [0, 1] -> CPU uint8 (F, H, W, C); None when the fast path does not apply."""
    import torch

    device = frames.device
    if device.type not in ("cuda", "cpu") or frames.numel() == 0:
        return None
    count, channels, height, width = frames.shape
    shape = (count, height, width, channels)
    host = None
    from .host_pinning import host_pinning_allowed

    if device.type == "cuda" and host_pinning_allowed(device):
        try:
            host = torch.empty(shape, dtype = torch.uint8, device = "cpu", pin_memory = True)
        except Exception:  # noqa: BLE001 - pinned allocation refused (host limits); a pageable copy is still exact
            host = None
    pinned = host is not None
    if host is None:
        host = torch.empty(shape, dtype = torch.uint8, device = "cpu")
    step = max(1, _SLICE_BYTES // max(1, channels * height * width * 4))
    in_range = None
    try:
        for start in range(0, count, step):
            piece = frames[start : start + step].to(torch.float32)
            # Range-checked per slice: a whole-clip reduction over the permuted view copies the clip first.
            lo, hi = torch.aminmax(piece)
            fits = (lo >= 0) & (hi <= 1)
            in_range = fits if in_range is None else in_range & fits
            piece = (piece * 255).round_().to(torch.uint8).permute(0, 2, 3, 1)
            host[start : start + step].copy_(piece, non_blocking = pinned)
        if pinned:
            torch.cuda.current_stream(device).synchronize()
    except torch.cuda.OutOfMemoryError:
        # np / pil copy straight to host, so fall back rather than fail a render.
        return None
    return host if bool(in_range) else None


class NonFiniteFramesError(RuntimeError):
    """The decoded clip holds NaN / inf: every encoder would refuse it (PyAV: "Expected numpy array with dtype uint8")."""


NON_FINITE_FRAMES_MSG = "The decoded video holds NaN or infinite values, so it was not saved. See the server log in Settings > Logs."


def require_finite_frames(frames: Any) -> None:
    """Raise ``NonFiniteFramesError`` when float ``frames`` (np array or tensor) hold NaN / inf; uint8 / PIL pass."""
    import numpy as np

    if isinstance(frames, np.ndarray):
        if np.issubdtype(frames.dtype, np.floating) and not bool(np.isfinite(frames).all()):
            raise NonFiniteFramesError(NON_FINITE_FRAMES_MSG)
        return
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    tensor = getattr(torch, "Tensor", None)
    if tensor is not None and isinstance(frames, tensor) and frames.is_floating_point():
        if not bool(torch.isfinite(frames).all()):
            raise NonFiniteFramesError(NON_FINITE_FRAMES_MSG)


def nonfinite_check_enabled() -> bool:
    return os.environ.get(NONFINITE_CHECK_ENV, "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _tensor_all_finite(frames: Any) -> bool:
    """Device-side finiteness of a float tensor, reduced in slices (no host copy, bounded temporary); one sync."""
    import torch

    if frames.numel() == 0:
        return True
    if frames.ndim == 0:
        return bool(torch.isfinite(frames))
    dim = 2 if frames.ndim == 5 else 0
    count = frames.shape[dim]
    step = max(1, _FINITE_SLICE_ELEMS // max(1, frames.numel() // max(1, count)))
    ok = None
    for start in range(0, count, step):
        piece = torch.isfinite(frames.narrow(dim, start, min(step, count - start))).all()
        ok = piece if ok is None else ok & piece
    return bool(ok)


def refuse_non_finite_tensor(frames: Any) -> None:
    """Raise ``NonFiniteFramesError`` when ``frames`` is a float tensor holding NaN / inf; anything else passes."""
    import torch

    tensor = getattr(
        torch, "Tensor", None
    )  # a stubbed torch (tests) has no Tensor: nothing to check
    if (
        tensor is not None
        and isinstance(frames, tensor)
        and frames.is_floating_point()
        and not _tensor_all_finite(frames)
    ):
        raise NonFiniteFramesError(NON_FINITE_FRAMES_MSG)


def _decoded_frames(output: Any) -> Any:
    """The pixel tensor of a ``vae.decode`` result: a tensor, ``(tensor, ...)`` or a ``DecoderOutput``."""
    if isinstance(output, (tuple, list)):
        return output[0] if output else None
    sample = getattr(output, "sample", None)
    return output if sample is None else sample


FINITE_DECODE_LAYER = "video_finite_decode"


@contextlib.contextmanager
def finite_video_decode(pipe: Any, attrs: "tuple[str, ...]" = ("vae",)) -> Iterator[None]:
    """While active, a ``vae.decode`` whose pixels hold NaN / inf raises ``NonFiniteFramesError``: checked on the
    device before any pipeline clamp (MiniMax-H3 clamps to [0, 1] first) or PIL / uint8 conversion hides it. The
    output is returned untouched. A decode layer (decode_layers) over the load-time wrappers, so the fp16 decode's
    fp32 rerun has run; it carries no below-scope marker, so the OOM scope stays intact (re-asserted after the wrap)."""
    if not nonfinite_check_enabled():
        yield
        return
    from . import decode_layers
    from .diffusion_vae_oom import ensure_outermost

    def make(inner: Any, _layer: Any) -> Any:
        def decode(*args: Any, **kwargs: Any) -> Any:
            output = inner(*args, **kwargs)
            refuse_non_finite_tensor(_decoded_frames(output))
            return output

        return decode

    layers: list = []
    for name in attrs:
        owner = getattr(pipe, name, None)
        if owner is None or not callable(getattr(owner, "decode", None)):
            continue
        try:
            layers.append(decode_layers.wrap(owner, "decode", FINITE_DECODE_LAYER, make))
        except Exception:  # noqa: BLE001 - a decoder that refuses the layer keeps the postprocess / encoder checks
            continue
        ensure_outermost(owner)
    try:
        yield
    finally:
        for layer in reversed(layers):
            try:
                decode_layers.unwrap(layer)
                ensure_outermost(layer.owner)
            except Exception:  # noqa: BLE001
                pass


@contextlib.contextmanager
def uint8_video_frames(pipe: Any) -> Iterator[None]:
    """While active, ``pipe.video_processor``'s np / pil ``postprocess_video`` returns CPU uint8 (B, F, H, W, C), and
    refuses a float clip holding NaN / inf (any output type) with ``NonFiniteFramesError``."""
    proc = getattr(pipe, "video_processor", None)
    original = getattr(proc, "postprocess_video", None)
    postprocess = getattr(proc, "postprocess", None)
    if not callable(original) or not callable(postprocess):
        yield
        return
    import torch

    had_own = "postprocess_video" in vars(proc)
    check = nonfinite_check_enabled()

    def postprocess_video(
        video: Any,
        output_type: str = "np",
        **kwargs: Any,
    ) -> Any:
        if check:
            # Before the denormalize clamp (+inf -> 255) and the PIL cast (NaN -> 0) can hide a non-finite decode.
            refuse_non_finite_tensor(video)
        if (
            output_type not in ("np", "pil")
            or not isinstance(video, torch.Tensor)
            or video.ndim != 5
        ):
            return original(video, output_type, **kwargs)
        clips = []
        for batch in range(video.shape[0]):
            # "pt" stops right after the denormalize the np / pil paths apply, on the same view they use.
            clip = device_uint8(postprocess(video[batch].permute(1, 0, 2, 3), "pt", **kwargs))
            if clip is None:
                return original(video, output_type, **kwargs)
            clips.append(clip)
        return clips[0].unsqueeze(0) if len(clips) == 1 else torch.stack(clips)

    try:
        proc.postprocess_video = postprocess_video
    except Exception:  # noqa: BLE001 - a processor that refuses instance attributes keeps its own path
        yield
        return
    try:
        yield
    finally:
        if had_own:
            proc.postprocess_video = original
        else:
            try:
                del proc.postprocess_video
            except AttributeError:
                pass
