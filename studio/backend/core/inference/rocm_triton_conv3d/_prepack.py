# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/conv/_prepack.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: Conv3D packs only; the pack cache holds a weak reference to the source weight.
# The vendored portions remain under the MIT License below:
#
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import weakref
from collections import OrderedDict

import torch

from ._launch import (
    _launch_ncdhw_to_cblocked,
)
from ._utils import (
    BLOCK_K,
    _winograd_transform_storage_dtype,
)

# Studio: a bounded LRU per packed format, like upstream, but holding only a weak reference to the source weight, so a
# pack is dropped with its module instead of pinning the weight (and itself) until evicted.
_PACK_CACHE_MAXSIZE = 256


class _LRUPackCache:
    def __init__(self, maxsize: int = _PACK_CACHE_MAXSIZE):
        self._d: OrderedDict[tuple, tuple] = OrderedDict()
        self._by_src: dict[int, tuple] = {}
        self._max = max(1, maxsize)

    def get(self, key):
        entry = self._d.get(key)
        if entry is None:
            return None
        if entry[0]() is None:
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return entry

    def put(self, key, src, item):
        # One live pack per source tensor: a weight moved (offload) or edited in place replaces its old pack.
        old = self._by_src.get(id(src))
        if old is not None and old != key:
            self._d.pop(old, None)
        self._by_src[id(src)] = key
        self._d[key] = (weakref.ref(src), item)
        self._d.move_to_end(key)
        weakref.finalize(src, self._drop, id(src), key)
        while len(self._d) > self._max:
            self._d.popitem(last = False)

    def _drop(self, src_id, key):
        self._d.pop(key, None)
        if self._by_src.get(src_id) == key:
            self._by_src.pop(src_id, None)

    def clear(self):
        self._d.clear()
        self._by_src.clear()

    def __len__(self):
        return len(self._d)

    def nbytes(self) -> int:
        total = 0
        for _ref, item in self._d.values():
            t = item[0] if isinstance(item, tuple) else item
            total += t.numel() * t.element_size()
        return total


_PACK_CACHE_3D_GENERAL = _LRUPackCache()
_PACK_CACHE_3D_3X3X3 = _LRUPackCache()
_PACK_CACHE_3D_WINOGRAD_HW = _LRUPackCache()


def clear_conv3d_weight_pack_caches() -> None:
    _PACK_CACHE_3D_GENERAL.clear()
    _PACK_CACHE_3D_3X3X3.clear()
    _PACK_CACHE_3D_WINOGRAD_HW.clear()


def conv3d_weight_pack_cache_bytes() -> int:
    return sum(
        c.nbytes()
        for c in (_PACK_CACHE_3D_GENERAL, _PACK_CACHE_3D_3X3X3, _PACK_CACHE_3D_WINOGRAD_HW)
    )


_WINOGRAD_F4X3_FILTER_MATRIX = (
    (1.0 / 4, 0.0, 0.0),
    (-1.0 / 6, -1.0 / 6, -1.0 / 6),
    (-1.0 / 6, 1.0 / 6, -1.0 / 6),
    (1.0 / 24, 1.0 / 12, 1.0 / 6),
    (1.0 / 24, -1.0 / 12, 1.0 / 6),
    (0.0, 0.0, 1.0),
)


def _transform_winograd_filter_f4x3(weight: torch.Tensor) -> torch.Tensor:
    """Apply ``G @ weight @ G.T`` to every trailing 3x3 filter plane."""
    matrix = torch.tensor(
        _WINOGRAD_F4X3_FILTER_MATRIX,
        dtype = torch.float32,
        device = weight.device,
    )
    return torch.einsum("ij,...jl,lm->...im", matrix, weight.float(), matrix.t())


def _pack_cache_key(w: torch.Tensor, block: int) -> tuple:
    """Identity for a packed weight, including aliases and in-place updates."""
    return (
        w.data_ptr(),
        w.device.type,
        w.device.index,
        tuple(w.shape),
        tuple(w.stride()),
        w.dtype,
        w._version,
        block,
    )


def _round_up_to_block(size: int, block: int) -> int:
    return ((size + block - 1) // block) * block


def _pad_last_dim(tensor: torch.Tensor, padded_size: int) -> torch.Tensor:
    """Zero-pad the last dimension and return contiguous storage."""
    if tensor.shape[-1] == padded_size:
        return tensor.contiguous()
    padding_shape = (*tensor.shape[:-1], padded_size - tensor.shape[-1])
    return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim = -1).contiguous()


def _prepack_to_kmajor(weight: torch.Tensor, reduction_size: int, block: int):
    output_channels = weight.shape[0]
    padded_size = _round_up_to_block(reduction_size, block)
    packed = weight.reshape(output_channels, reduction_size)
    return _pad_last_dim(packed, padded_size), padded_size


def _prepack_fixed_kernel(weight: torch.Tensor, taps: int, block: int):
    output_channels, input_channels = weight.shape[:2]
    padded_channels = _round_up_to_block(input_channels, block)
    packed = weight.reshape(output_channels, input_channels, taps)
    packed = packed.permute(0, 2, 1).contiguous()
    return _pad_last_dim(packed, padded_channels), padded_channels


def prepack_oidhw_to_kmajor(w_oidhw: torch.Tensor, block_k: int = BLOCK_K):
    """Pack ``[K_out,C,T,R,S]`` into zero-padded ``[K_out,K_pad]``."""
    _, C, T, R, S = w_oidhw.shape
    K_red = C * T * R * S
    return _prepack_to_kmajor(w_oidhw, K_red, block_k)


def prepack_oidhw_to_3x3x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    """Pack OIDHW weights as ``[K_out,27,C_pad]`` in depth-major tap order."""
    _, _, T, R, S = w_oidhw.shape
    if (T, R, S) != (3, 3, 3):
        raise ValueError(f"3x3x3 prepack requires a 3x3x3 weight, got {(T, R, S)}")
    return _prepack_fixed_kernel(w_oidhw, 27, block_c)


def prepack_ncdhw_to_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    """Materialize NCDHW as contiguous ``[N,C_blocks,D,H,W,Cb]``."""
    if not x.is_contiguous():
        x = x.contiguous()
    N, C, D, H, W = x.shape
    C_blocks = (C + block_c - 1) // block_c
    C_pad = _round_up_to_block(C, block_c)
    packed = torch.empty((N, C_blocks, D, H, W, block_c), device = x.device, dtype = x.dtype)
    _launch_ncdhw_to_cblocked(x, packed, N, C, D, H, W, C_pad, block_c)
    return packed, C_pad


def prepack_winograd_hw_filter_f4x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    """Apply F(4,3)'s 2-D filter transform to each of three depth taps.

    The result is ``[3,36,K_out,C_pad]``. BF16 weights use FP16 transformed
    storage, reducing the extra rounding introduced by the Winograd domain.
    """
    K_out, C, T, R, S = w_oidhw.shape
    if (T, R, S) != (3, 3, 3):
        raise ValueError(f"Winograd prepack requires 3x3x3, got {(T, R, S)}")
    C_pad = _round_up_to_block(C, block_c)
    transformed = _transform_winograd_filter_f4x3(w_oidhw)
    transformed = transformed.reshape(K_out, C, 3, 36).permute(2, 3, 0, 1).contiguous()
    transformed = _pad_last_dim(transformed, C_pad)
    return transformed.to(_winograd_transform_storage_dtype(w_oidhw.dtype)), C_pad


def _get_or_make_pack(cache, weight: torch.Tensor, block: int, packer):
    if weight.is_inference():
        return packer(weight, block)

    key = _pack_cache_key(weight, block)
    cached = cache.get(key)
    if cached is not None:
        return cached[1]
    packed = packer(weight, block)
    cache.put(key, weight, packed)
    return packed


def get_or_make_weight_pack_3d(w_oidhw: torch.Tensor, block_k: int = BLOCK_K):
    return _get_or_make_pack(_PACK_CACHE_3D_GENERAL, w_oidhw, block_k, prepack_oidhw_to_kmajor)


def get_or_make_weight_pack_3x3x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    return _get_or_make_pack(_PACK_CACHE_3D_3X3X3, w_oidhw, block_c, prepack_oidhw_to_3x3x3)


def get_or_make_winograd_hw_filter_f4x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    return _get_or_make_pack(
        _PACK_CACHE_3D_WINOGRAD_HW,
        w_oidhw,
        block_c,
        prepack_winograd_hw_filter_f4x3,
    )
