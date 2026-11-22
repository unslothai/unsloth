# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Adapted from sglang python/sglang/kernels/ops/mm/process/lanczos.py (commit 9667988e), Copyright the SGLang project
# authors, Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Change: the output crop height and width
# are runtime arguments instead of tl.constexpr, so one compile serves every reference image size.
"""GPU RGB Lanczos resize with optional crop and normalization."""

import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl


def _sinc(value: float) -> float:
    if value == 0.0:
        return 1.0
    angle = value * math.pi
    return math.sin(angle) / angle


@lru_cache(maxsize = 128)
def _coefficients(source: int, target: int, device: torch.device):
    """Pillow's RGB resampling uses normalized signed 22-bit coefficients."""
    scale = source / target
    filter_scale = max(scale, 1.0)
    support = 3.0 * filter_scale
    inv_filter_scale = 1.0 / filter_scale
    taps = math.ceil(support) * 2 + 1
    coefficients = np.zeros((target, taps), dtype = np.int32)
    bounds = np.zeros((target, 2), dtype = np.int32)
    for out in range(target):
        center = (out + 0.5) * scale
        begin = max(int(center - support + 0.5), 0)
        end = min(int(center + support + 0.5), source)
        weights = []
        for pos in range(begin, end):
            distance = (pos - center + 0.5) * inv_filter_scale
            weights.append(
                _sinc(distance) * _sinc(distance / 3.0) if -3.0 <= distance < 3.0 else 0.0
            )
        # Sequential sum: Python 3.12+ sum() uses compensated summation.
        total = 0.0
        for weight in weights:
            total += weight
        coefficients[out, : len(weights)] = [
            int(w / total * (1 << 22) + (0.5 if w >= 0.0 else -0.5)) for w in weights
        ]
        bounds[out] = (begin, len(weights))
    return (
        torch.from_numpy(coefficients).to(device),
        torch.from_numpy(bounds).to(device),
        taps,
    )


@lru_cache(maxsize = 8)
def _normalization(
    mean: tuple[float, float, float], std: tuple[float, float, float], device: torch.device
):
    # Precomputed float32 values avoid FMA / approximate division in the kernel.
    values = np.arange(256, dtype = np.float32)[None, :] / 255.0
    values = (values - np.asarray(mean, dtype = np.float32)[:, None]) / np.asarray(
        std, dtype = np.float32
    )[:, None]
    return torch.from_numpy(values).to(device)


@triton.jit(
    do_not_specialize = [
        "source_height",
        "crop_left",
        "stride_batch",
        "stride_y",
        "stride_x",
        "stride_channel",
        "taps",
        "crop_width",
    ]
)
def _horizontal_lanczos(
    src,
    dst,
    coefficients,
    bounds,
    source_height,
    crop_width,
    crop_left,
    stride_batch,
    stride_y,
    stride_x,
    stride_channel,
    taps,
    block: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < source_height * crop_width * 3
    channel = index % 3
    x = index // 3 % crop_width + crop_left
    y = index // (3 * crop_width)
    first = tl.load(bounds + x * 2, valid, 0)
    count = tl.load(bounds + x * 2 + 1, valid, 0)
    total = tl.full((block,), 1 << 21, tl.int32)
    for tap in range(taps):
        weight = tl.load(coefficients + x * taps + tap, valid, 0)
        value = tl.load(
            src
            + batch * stride_batch
            + y * stride_y
            + (first + tap) * stride_x
            + channel * stride_channel,
            valid & (tap < count),
            0,
        ).to(tl.int32)
        total += value * weight
    pixel = tl.minimum(tl.maximum(total >> 22, 0), 255).to(tl.uint8)
    tl.store(dst + batch * source_height * crop_width * 3 + index, pixel, valid)


@triton.jit(do_not_specialize = ["crop_height", "crop_width", "crop_top", "stride_batch", "taps"])
def _vertical_lanczos(
    src,
    dst,
    coefficients,
    bounds,
    normalization,
    crop_height,
    crop_width,
    crop_top,
    stride_batch,
    taps,
    normalize: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < crop_height * crop_width * 3
    channel = index % 3
    y = index // (3 * crop_width) + crop_top
    x = index // 3 % crop_width
    first = tl.load(bounds + y * 2, valid, 0)
    count = tl.load(bounds + y * 2 + 1, valid, 0)
    total = tl.full((block,), 1 << 21, tl.int32)
    for tap in range(taps):
        weight = tl.load(coefficients + y * taps + tap, valid, 0)
        value = tl.load(
            src + batch * stride_batch + ((first + tap) * crop_width + x) * 3 + channel,
            valid & (tap < count),
            0,
        ).to(tl.int32)
        total += value * weight
    pixel = tl.minimum(tl.maximum(total >> 22, 0), 255)
    if normalize:
        value = tl.load(normalization + channel * 256 + pixel)
    else:
        value = pixel.to(tl.uint8)
    tl.store(dst + batch * crop_height * crop_width * 3 + index, value, valid)


@triton.jit
def _normalize_rgb(src, dst, normalization, count, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    pixel = tl.load(src + index, index < count, 0).to(tl.int32)
    value = tl.load(normalization + index % 3 * 256 + pixel)
    tl.store(dst + index, value, index < count)


def lanczos_resize(
    image: torch.Tensor,
    height: int,
    width: int,
    *,
    crop: Optional[tuple[int, int, int, int]] = None,
    mean: Optional[tuple[float, float, float]] = None,
    std: Optional[tuple[float, float, float]] = None,
    vertical_first: bool = False,
) -> torch.Tensor:
    """Resize CUDA uint8 HWC/NHWC RGB to ``height x width`` matching Pillow's Lanczos rounding.

    ``crop=(top, left, h, w)`` crops the resized image; ``mean`` + ``std`` fuse normalisation (float32 out).
    ``vertical_first`` runs the vertical pass first, as Pillow >= 13 does when the height shrinks by more than twice
    the width.
    """
    from .rocm_bf16 import is_rocm_torch

    if not image.is_cuda or is_rocm_torch(torch):
        raise ValueError("Lanczos RGB preprocessing requires a CUDA tensor")
    if image.dtype != torch.uint8 or image.ndim not in (3, 4) or image.shape[-1] != 3:
        raise ValueError("Lanczos RGB input must be uint8 with shape [H, W, 3] or [N, H, W, 3]")
    if height <= 0 or width <= 0 or any(size <= 0 for size in image.shape):
        raise ValueError("Lanczos RGB input and output dimensions must be positive")
    if crop is None:
        top, left, output_h, output_w = 0, 0, height, width
    else:
        if len(crop) != 4:
            raise ValueError("crop must be (top, left, height, width)")
        top, left, output_h, output_w = crop
        if (
            top < 0
            or left < 0
            or output_h <= 0
            or output_w <= 0
            or top + output_h > height
            or left + output_w > width
        ):
            raise ValueError("crop must be a positive rectangle within the resized image")
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")
    normalize = mean is not None
    if normalize:
        mean, std = tuple(mean), tuple(std)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain three RGB values")
        if any(not math.isfinite(v) for v in mean + std) or any(v <= 0 for v in std):
            raise ValueError("mean/std must be finite and std must be positive")

    # Launch on the operand's device: Triton uses the current device and stream.
    with torch.cuda.device(image.device):
        return _launch(
            image,
            height,
            width,
            top,
            left,
            output_h,
            output_w,
            crop,
            normalize,
            mean,
            std,
            vertical_first,
        )


def _launch(
    image, height, width, top, left, output_h, output_w, crop, normalize, mean, std, vertical_first
):
    source_h, source_w = image.shape[-3:-1]
    if (source_h, source_w) == (height, width) and crop is None:
        image = image.contiguous()
        if not normalize:
            return image.clone()
        output = torch.empty_like(image, dtype = torch.float32)
        _normalize_rgb[(triton.cdiv(image.numel(), 256),)](
            image, output, _normalization(mean, std, image.device), image.numel(), 256
        )
        return output
    batch_size = image.shape[0] if image.ndim == 4 else 1
    coeff_x, bounds_x, taps_x = _coefficients(source_w, width, image.device)
    coeff_y, bounds_y, taps_y = _coefficients(source_h, height, image.device)
    if vertical_first and source_h != height:
        # The vertical pass reads whole contiguous rows; it keeps only the cropped rows, all source columns.
        image = image.contiguous()
        rows = torch.empty(
            (batch_size, output_h, source_w, 3), dtype = torch.uint8, device = image.device
        )
        _vertical_lanczos[(triton.cdiv(output_h * source_w * 3, 256), batch_size)](
            image,
            rows,
            coeff_y,
            bounds_y,
            None,
            output_h,
            source_w,
            top,
            image.stride(0) if image.ndim == 4 else 0,
            taps_y,
            False,
            256,
        )
        pixels = torch.empty(
            (*image.shape[:-3], output_h, output_w, 3), dtype = torch.uint8, device = image.device
        )
        _horizontal_lanczos[(triton.cdiv(output_h * output_w * 3, 256), batch_size)](
            rows,
            pixels,
            coeff_x,
            bounds_x,
            output_h,
            output_w,
            left,
            rows.stride(0),
            *rows.stride()[-3:],
            taps_x,
            256,
        )
        if not normalize:
            return pixels
        output = torch.empty_like(pixels, dtype = torch.float32)
        _normalize_rgb[(triton.cdiv(pixels.numel(), 256),)](
            pixels, output, _normalization(mean, std, image.device), pixels.numel(), 256
        )
        return output
    stride_batch = image.stride(0) if image.ndim == 4 else 0
    intermediate = torch.empty(
        (batch_size, source_h, output_w, 3), dtype = torch.uint8, device = image.device
    )
    output = torch.empty(
        (*image.shape[:-3], output_h, output_w, 3),
        dtype = torch.float32 if normalize else torch.uint8,
        device = image.device,
    )
    _horizontal_lanczos[(triton.cdiv(source_h * output_w * 3, 256), batch_size)](
        image,
        intermediate,
        coeff_x,
        bounds_x,
        source_h,
        output_w,
        left,
        stride_batch,
        *image.stride()[-3:],
        taps_x,
        256,
    )
    _vertical_lanczos[(triton.cdiv(output_h * output_w * 3, 256), batch_size)](
        intermediate,
        output,
        coeff_y,
        bounds_y,
        _normalization(mean, std, image.device) if normalize else None,
        output_h,
        output_w,
        top,
        intermediate.stride(0),
        taps_y,
        normalize,
        256,
    )
    return output
