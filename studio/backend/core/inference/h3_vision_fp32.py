# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3's Qwen3-VL vision tower computed in float32, as the reference implementation runs it.

The reference feeds float32 pixels and upcasts the bf16 weights per op; transformers rounds the pixels and computes in
bf16. ``h3_vision_float32(text_encoder)`` scopes the reference numerics to the tower for one call: float32 pixels kept
(``offer_float32_pixels`` swaps a caller's rounded copy back), weights upcast per op with no resident copy, no TF32
for matmuls, the reference's bf16 position-embedding sum where transformers sums in float32, and outputs returned in
the tower's dtype so the language model is untouched. ``UNSLOTH_H3_VISION_FP32=0`` restores the bf16 tower.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Any, Iterator, Optional

H3_VISION_FP32_ENV = "UNSLOTH_H3_VISION_FP32"
_WRAPPED = ("get_image_features", "get_video_features")
_MARK = "_unsloth_h3_vision_fp32"

# Process totals: tower calls in float32, per-op weight upcasts, reference position sums, float32 pixel sources used.
STATS = {"calls": 0, "upcasts": 0, "position_sums": 0, "sources_used": 0}

_SOURCES: contextvars.ContextVar = contextvars.ContextVar(
    "unsloth_h3_vision_fp32_sources", default = None
)
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "unsloth_h3_vision_fp32_active", default = False
)


def vision_fp32_enabled() -> bool:
    raw = (os.environ.get(H3_VISION_FP32_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def vision_pixel_dtype(default: Any) -> Any:
    """The dtype pixels should reach the conditioner in: float32 with the switch on, else ``default``."""
    if not vision_fp32_enabled():
        return default
    import torch

    return torch.float32


@contextlib.contextmanager
def float32_pixel_sources() -> Iterator[None]:
    """Collect the float32 pixel tensors offered during the block, for ``h3_vision_float32`` to use."""
    token = _SOURCES.set([])
    try:
        yield
    finally:
        _SOURCES.reset(token)


def offer_float32_pixels(*tensors: Any) -> None:
    """Record float32 pixel tensors a later step may round before the vision tower. A no-op outside
    ``float32_pixel_sources()`` or with the switch off."""
    sources = _SOURCES.get()
    if sources is None or not vision_fp32_enabled():
        return
    import torch

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
            sources.append(tensor)


def _float32_source(pixels: Any) -> Any:
    """``pixels`` as float32: its offered source when ``pixels`` is exactly that source rounded, else an upcast."""
    import torch

    if pixels.dtype == torch.float32:
        return pixels
    for source in _SOURCES.get() or ():
        if source.shape != pixels.shape:
            continue
        candidate = source.to(pixels.device)
        if torch.equal(candidate.to(pixels.dtype), pixels):
            STATS["sources_used"] += 1
            return candidate
    return pixels.to(torch.float32)


def _vision_tower(model: Any) -> Any:
    visual = getattr(model, "visual", None)
    return visual if visual is not None else getattr(getattr(model, "model", None), "visual", None)


def _tower_dtype(visual: Any) -> Any:
    import torch
    for param in visual.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _qwen3vl_model(text_encoder: Any) -> Optional[Any]:
    """The module whose ``get_image_features`` runs the tower (``Qwen3VLModel``), or None."""
    for candidate in (getattr(text_encoder, "model", None), text_encoder):
        if candidate is None:
            continue
        if (
            callable(getattr(candidate, "get_image_features", None))
            and getattr(candidate, "visual", None) is not None
        ):
            return candidate
    return None


def reference_position_embeddings(table: Any, grid_thw: Any, side: int, merge: int) -> Any:
    """The reference's interpolated position embeddings, in token order: bilinear (align corners) over the
    ``side x side`` learned grid with weights rounded to the table's dtype and the four taps summed in that dtype."""
    import torch
    import torch.nn.functional as F

    out = []
    for t, h, w in (tuple(int(v) for v in row) for row in grid_thw.tolist()):
        h_idx = torch.linspace(0, side - 1, h, device = table.device)
        w_idx = torch.linspace(0, side - 1, w, device = table.device)
        h_lo, w_lo = h_idx.int(), w_idx.int()
        h_hi, w_hi = (h_lo + 1).clip(max = side - 1), (w_lo + 1).clip(max = side - 1)
        dh, dw = h_idx - h_lo, w_idx - w_lo
        base_lo, base_hi = h_lo * side, h_hi * side
        indices = torch.stack(
            [
                (base_lo[:, None] + w_lo[None]).flatten(),
                (base_lo[:, None] + w_hi[None]).flatten(),
                (base_hi[:, None] + w_lo[None]).flatten(),
                (base_hi[:, None] + w_hi[None]).flatten(),
            ]
        ).long()
        weights = torch.stack(
            [
                ((1 - dh)[:, None] * (1 - dw)[None]).flatten(),
                ((1 - dh)[:, None] * dw[None]).flatten(),
                (dh[:, None] * (1 - dw)[None]).flatten(),
                (dh[:, None] * dw[None]).flatten(),
            ]
        ).to(table.dtype)
        rows = F.embedding(indices, table) * weights[:, :, None]
        pos = rows[0] + rows[1] + rows[2] + rows[3]
        pos = pos.repeat(t, 1).view(t, h // merge, merge, w // merge, merge, -1)
        out.append(pos.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
    return torch.cat(out)


def _float32_mode_class() -> type:
    import torch
    import torch.nn.functional as F
    from torch.overrides import TorchFunctionMode

    low = (torch.bfloat16, torch.float16)
    upcast_ops = {F.linear, F.layer_norm, torch.conv3d, torch.conv2d}
    cast_ops = {torch.Tensor.to, torch.Tensor.type}
    embedding_ops = {F.embedding, torch.embedding}

    def _up(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.dtype in low:
            return value.to(torch.float32)
        return value

    class VisionFloat32Mode(TorchFunctionMode):
        """Float32 inputs make float32 ops; pixel storage keeps float32 through the tower's own downcasts."""

        def __init__(
            self,
            pixels: Any,
            visual: Any = None,
            grid_thw: Any = None,
        ):
            super().__init__()
            self._pixel_storage = {self._storage(p) for p in pixels if p is not None}
            self.upcasts = 0
            self.position_sums = 0
            self._grid = grid_thw
            self._table = None
            table = getattr(getattr(visual, "pos_embed", None), "weight", None)
            self._table_shape = tuple(table.shape) if isinstance(table, torch.Tensor) else None
            self._side = int(getattr(visual, "num_grid_per_side", 0) or 0)
            self._merge = int(getattr(visual, "spatial_merge_size", 0) or 0)
            bilinear = getattr(visual, "interpolation_mode", "bilinear") == "bilinear"
            corners = bool(getattr(visual, "interpolation_align_corners", True))
            self._pos_ready = bool(
                bilinear
                and corners
                and grid_thw is not None
                and self._side
                and self._merge
                and self._table_shape
            )

        def _position_sum(self, args: tuple, kwargs: dict) -> Any:
            """The reference's tap sum for transformers' ``(table[idx] * w).sum(1)``, or None to keep theirs."""
            if not self._pos_ready or self._table is None or kwargs:
                return None
            if len(args) != 2 or args[1] != 1 or not isinstance(args[0], torch.Tensor):
                return None
            taps = args[0]
            if taps.dim() != 3 or taps.shape[1] != 4 or taps.shape[2] != self._table_shape[1]:
                return None
            ours = reference_position_embeddings(
                self._table, self._grid.to(self._table.device), self._side, self._merge
            )
            theirs = taps.sum(1)
            if ours.shape != theirs.shape:
                return None
            ours = ours.to(device = theirs.device, dtype = theirs.dtype)
            # the same interpolation up to the reference's rounding, else a layout this code does not know
            if not bool(
                torch.allclose(
                    ours, theirs, rtol = 2e-2, atol = 2e-2 * float(theirs.detach().abs().max())
                )
            ):
                return None
            self._table = None
            self.position_sums += 1
            return ours

        @staticmethod
        def _storage(tensor: Any) -> int:
            return tensor.untyped_storage().data_ptr()

        def _keeps_pixels(self, args: tuple, kwargs: dict) -> bool:
            tensor = args[0] if args else None
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                return False
            if self._storage(tensor) not in self._pixel_storage:
                return False
            targets = [a for a in args[1:] if isinstance(a, torch.dtype)]
            if isinstance(kwargs.get("dtype"), torch.dtype):
                targets.append(kwargs["dtype"])
            return any(t in low for t in targets)

        def __torch_function__(
            self,
            func: Any,
            types: Any,
            args: tuple = (),
            kwargs: Optional[dict] = None,
        ) -> Any:
            kwargs = kwargs or {}
            if (
                func in upcast_ops
                and args
                and isinstance(args[0], torch.Tensor)
                and args[0].dtype == torch.float32
            ):
                if any(
                    isinstance(a, torch.Tensor) and a.dtype in low
                    for a in (*args[1:], *kwargs.values())
                ):
                    self.upcasts += 1
                    args = (args[0],) + tuple(_up(a) for a in args[1:])
                    kwargs = {k: _up(v) for k, v in kwargs.items()}
            elif func in embedding_ops and self._pos_ready and len(args) >= 2:
                weight = args[1] if func is F.embedding else args[0]
                if isinstance(weight, torch.Tensor) and tuple(weight.shape) == self._table_shape:
                    self._table = weight
            elif func is torch.Tensor.sum:
                replaced = self._position_sum(args, kwargs)
                if replaced is not None:
                    return replaced
            elif func in cast_ops and self._keeps_pixels(args, kwargs):
                args = (args[0],) + tuple(
                    torch.float32 if isinstance(a, torch.dtype) else a for a in args[1:]
                )
                if "dtype" in kwargs:
                    kwargs = {**kwargs, "dtype": torch.float32}
            return func(*args, **kwargs)

    return VisionFloat32Mode


_MODE_CLASS: Optional[type] = None


def _mode(
    pixels: Any,
    visual: Any = None,
    grid_thw: Any = None,
) -> Any:
    global _MODE_CLASS
    if _MODE_CLASS is None:
        _MODE_CLASS = _float32_mode_class()
    return _MODE_CLASS(pixels, visual, grid_thw)


def _cast_floats(value: Any, dtype: Any) -> Any:
    """Every float32 tensor in ``value`` (tensor, tuple, list, dict or ModelOutput) cast to ``dtype``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.dtype == torch.float32 and dtype != torch.float32 else value
    if isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = _cast_floats(value[key], dtype)
        return value
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_cast_floats(v, dtype) for v in value)
    if isinstance(value, list):
        return [_cast_floats(v, dtype) for v in value]
    return value


@contextlib.contextmanager
def _no_matmul_tf32(device: Any) -> Iterator[None]:
    import torch

    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if (
        getattr(device, "type", None) != "cuda"
        or matmul is None
        or not hasattr(matmul, "allow_tf32")
    ):
        yield
        return
    before = bool(matmul.allow_tf32)
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_tf32 = before


def _float32_features(model: Any, original: Any) -> Any:
    visual = _vision_tower(model)

    def features(pixel_values: Any, *args: Any, **kwargs: Any) -> Any:
        if _ACTIVE.get() or pixel_values is None or not pixel_values.is_floating_point():
            return original(pixel_values, *args, **kwargs)
        out_dtype = _tower_dtype(visual)
        pixels = _float32_source(pixel_values)
        grid = (
            args[0]
            if args
            else next(
                (kwargs[k] for k in ("image_grid_thw", "video_grid_thw") if k in kwargs), None
            )
        )
        token = _ACTIVE.set(True)
        try:
            with _no_matmul_tf32(pixels.device), _mode([pixels], visual, grid) as mode:
                out = original(pixels, *args, **kwargs)
            features.upcasts += mode.upcasts
            features.position_sums += mode.position_sums
            STATS["calls"] += 1
            STATS["upcasts"] += mode.upcasts
            STATS["position_sums"] += mode.position_sums
        finally:
            _ACTIVE.reset(token)
        return _cast_floats(out, out_dtype)

    features.upcasts = 0
    features.position_sums = 0
    setattr(features, _MARK, True)
    return features


@contextlib.contextmanager
def h3_vision_float32(text_encoder: Any) -> Iterator[Optional[Any]]:
    """Run the Qwen3-VL vision tower of ``text_encoder`` in float32 for the block. Yields the wrapped model
    (None when the switch is off or ``text_encoder`` has no Qwen3-VL tower); nothing stays patched afterwards."""
    model = _qwen3vl_model(text_encoder) if vision_fp32_enabled() else None
    if model is None or getattr(model.__dict__.get("get_image_features"), _MARK, False):
        yield None
        return
    installed = []
    for name in _WRAPPED:
        original = getattr(model, name, None)
        if callable(original) and name not in model.__dict__:
            setattr(model, name, _float32_features(model, original))
            installed.append(name)
    try:
        yield model
    finally:
        for name in installed:
            model.__dict__.pop(name, None)
