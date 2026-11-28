# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes.functional for Unsloth's bitsandbytes stand-in: QuantState and the 4-bit / 8-bit
blockwise (de)quantization subset transformers, PEFT and Unsloth call, on Unsloth's Triton kernels
(torch elsewhere). Serialized states use the bitsandbytes layout, so checkpoints load either way."""

import os
from typing import Any, Optional

import torch

from . import _quantize as _Q

from .cextension import lib  # noqa: F401  (stub handles: callers probe for a native library)
from .utils import pack_dict_to_tensor, unpack_tensor_to_dict

__all__ = [
    "QuantState",
    "create_dynamic_map",
    "dequantize_4bit",
    "dequantize_blockwise",
    "dequantize_fp4",
    "dequantize_nf4",
    "gemv_4bit",
    "get_4bit_type",
    "get_ptr",
    "quantize_4bit",
    "quantize_blockwise",
    "quantize_fp4",
    "quantize_nf4",
]

name2qmap = {}
create_dynamic_map = _Q.create_dynamic_map
_BLOCKSIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
# UNSLOTH_BNB_TRITON=0 (Unsloth's NF4 kernel switch) also keeps this stand-in on torch dequantize.
_TRITON_DEQUANT = os.environ.get("UNSLOTH_BNB_TRITON", "1") != "0"


def get_ptr(A: Optional[torch.Tensor]):
    raise RuntimeError(
        "Unsloth: this is Unsloth's bitsandbytes stand-in, which has no native library to pass "
        "pointers to. Install the bitsandbytes package for this code path."
    )


def get_4bit_type(
    typename,
    device = None,
    blocksize = 64,
):
    if device is None:
        device = "cuda"
    if typename == "nf4":
        data = torch.tensor(_Q.NF4_CODE, dtype = torch.float32)
    elif typename == "fp4":
        data = torch.tensor(
            [
                0,
                0.0625,
                8.0,
                12.0,
                4.0,
                6.0,
                2.0,
                3.0,
                -0,
                -0.0625,
                -8.0,
                -12.0,
                -4.0,
                -6.0,
                -2.0,
                -3.0,
            ],
            dtype = torch.float32,
        )
        data.div_(data.abs().max())
    else:
        raise NotImplementedError(f"Unsloth: 4bit type {typename} is not implemented.")
    return data.to(device)


class QuantState:
    """Container for a 4-bit or 8-bit blockwise quantization state, bitsandbytes compatible."""

    valid_quant_types = ("fp4", "nf4")
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = [
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state",
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape = None,
        code = None,
        blocksize = None,
        quant_type = None,
        dtype = None,
        offset = None,
        state2 = None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __getattr__(self, name):
        # FSDP state_dict traversal resolves the packed key "quant_state.bitsandbytes__nf4" by getattr.
        if name.startswith("bitsandbytes__"):
            qs_dict = self.as_dict(packed = True)
            packed_key = "quant_state." + name
            if packed_key in qs_dict:
                return qs_dict[packed_key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, idx):
        # Legacy list layout: [absmax, shape, dtype, blocksize, [offset, state2] | None, quant_type].
        if self.nested:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                [self.offset, self.state2],
                self.quant_type,
            ]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: dict, device) -> "QuantState":
        qs_key = [
            k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)
        ]
        if "quant_type" not in qs_dict:
            if not qs_key:
                raise ValueError("Expected packed or unpacked quant_state items, found neither")
            elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
                raise ValueError(
                    f"There should be exactly one `quant_state` item with ending from "
                    f"{cls.valid_qs_type_keys}.\nDetected {qs_key}."
                )
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))
        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)
        if "nested_absmax" in qs_dict:
            offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            state2 = cls(
                absmax = qs_dict["nested_absmax"].to(device),
                blocksize = qs_dict["nested_blocksize"],
                code = qs_dict["nested_quant_map"].to(device),
                dtype = getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None
        return cls(
            quant_type = qs_dict["quant_type"],
            absmax = qs_dict["absmax"].to(device),
            blocksize = qs_dict["blocksize"],
            code = qs_dict["quant_map"].to(device),
            dtype = getattr(torch, qs_dict["dtype"]),
            shape = torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset = offset,
            state2 = state2,
        )

    def as_dict(self, packed: bool = False) -> dict:
        qs_dict = {
            "quant_type": self.quant_type,
            "absmax": self.absmax,
            "blocksize": self.blocksize,
            "quant_map": self.code,
            "dtype": str(self.dtype).strip("torch."),
            "shape": tuple(self.shape) if self.shape is not None else None,
        }
        if self.nested:
            qs_dict.update(
                {
                    "nested_absmax": self.state2.absmax,
                    "nested_blocksize": self.state2.blocksize,
                    # Un-shared, or safetensors drops it as a shared tensor.
                    "nested_quant_map": self.state2.code.clone(),
                    "nested_dtype": str(self.state2.dtype).strip("torch."),
                    "nested_offset": self.offset.item(),
                }
            )
        if not packed or self.quant_type is None:
            return qs_dict
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        qs_packed_dict["quant_state.bitsandbytes__" + self.quant_type] = pack_dict_to_tensor(
            non_tensor_dict
        )
        return qs_packed_dict

    def to(self, device):
        self.code = self.code.to(device)
        self.absmax = self.absmax.to(device)
        if self.nested:
            self.offset = self.offset.to(device)
            self.state2.absmax = self.state2.absmax.to(device)
            self.state2.code = self.state2.code.to(device)

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False
        return (
            torch.allclose(self.absmax, other.absmax, atol = 1e-6)
            and self.shape == other.shape
            and torch.allclose(self.code, other.code, atol = 1e-6)
            and self.dtype == other.dtype
            and self.blocksize == other.blocksize
            and self.quant_type == other.quant_type
            and (
                self.offset == other.offset
                if self.offset is not None and other.offset is not None
                else self.offset is other.offset
            )
            and (
                self.state2 == other.state2
                if self.state2 is not None and other.state2 is not None
                else self.state2 is other.state2
            )
        )

    __hash__ = object.__hash__


def _dynamic_code(device):
    code = name2qmap.get(("dynamic", str(device)))
    if code is None:
        code = name2qmap[("dynamic", str(device))] = create_dynamic_map().to(device)
    return code


def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize = 4096,
    nested = False,
):
    if blocksize <= 0:
        raise ValueError(f"blocksize must be positive, got {blocksize}")
    if A.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}"
        )
    if code is None:
        code = _dynamic_code(A.device)
    _out, _absmax = _Q.quantize_8bit_blockwise(A, code.to(A.device), blocksize)
    if nested:
        offset = _absmax.mean()
        _absmax -= offset
        qabsmax, state2 = quantize_blockwise(_absmax, blocksize = blocksize, nested = False)
        quant_state = QuantState(
            absmax = qabsmax,
            code = code.to(A.device, copy = True),
            blocksize = blocksize,
            dtype = A.dtype,
            offset = offset,
            state2 = state2,
        )
    else:
        quant_state = QuantState(
            absmax = _absmax, code = code.to(A.device, copy = True), blocksize = blocksize, dtype = A.dtype
        )
    if out is not None:
        out.copy_(_out)
        _out = out
    if absmax is not None:
        quant_state.absmax = absmax.copy_(quant_state.absmax)
    return _out, quant_state


def dequantize_blockwise(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested = False,
) -> torch.Tensor:
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        code = _dynamic_code(A.device)
    if quant_state is None:
        quant_state = QuantState(absmax = absmax, code = code, blocksize = blocksize, dtype = torch.float32)
    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
    result = _Q.dequantize_8bit_blockwise(
        A, absmax, quant_state.code, quant_state.blocksize, quant_state.dtype
    )
    if out is not None:
        return out.copy_(result)
    return result


def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize = None,
    compress_statistics = False,
    quant_type = "fp4",
    quant_storage = torch.uint8,
):
    if blocksize is None:
        blocksize = 64
    if blocksize not in _BLOCKSIZES:
        raise ValueError(f"invalid blocksize {blocksize}")
    if quant_type not in ("nf4", "fp4"):
        raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type!r}")
    if A.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}"
        )
    input_shape = A.shape
    packed, _absmax = _Q.quantize_4bit_blockwise(A, blocksize, quant_type)
    itemsize = torch.empty((), dtype = quant_storage).element_size()
    n_bytes = packed.numel()
    rows = -(n_bytes // -itemsize)
    if rows * itemsize != n_bytes:
        # Wider storage than the bytes fill: zero pad the tail rather than drop packed codes.
        packed = torch.cat([packed, packed.new_zeros(rows * itemsize - n_bytes)])
    _out = packed.view(quant_storage).view(rows, 1)
    code = get_4bit_type(quant_type, device = A.device)
    if compress_statistics:
        offset = _absmax.mean()
        qabsmax, state2 = quantize_blockwise(_absmax - offset, blocksize = 256)
        del _absmax
        state = QuantState(
            absmax = qabsmax,
            shape = input_shape,
            dtype = A.dtype,
            blocksize = blocksize,
            code = code,
            quant_type = quant_type,
            offset = offset,
            state2 = state2,
        )
    else:
        state = QuantState(
            absmax = _absmax,
            shape = input_shape,
            dtype = A.dtype,
            blocksize = blocksize,
            code = code,
            quant_type = quant_type,
        )
    if out is not None:
        _out = out.copy_(_out)
    if absmax is not None:
        state.absmax = absmax.copy_(state.absmax)
    return _out, state


def quantize_fp4(
    A,
    absmax = None,
    out = None,
    blocksize = None,
    compress_statistics = False,
    quant_storage = torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)


def quantize_nf4(
    A,
    absmax = None,
    out = None,
    blocksize = None,
    compress_statistics = False,
    quant_storage = torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)


def _pow2(x):
    return x > 0 and x & (x - 1) == 0


def _triton_dequant(A, absmax, quant_state, out):
    """Unsloth's Triton NF4 kernel (bit-exact with bitsandbytes), or None where it does not apply."""
    if (
        not A.is_cuda
        or quant_state.quant_type != "nf4"
        or quant_state.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not _pow2(quant_state.blocksize)
        or (quant_state.nested and not _pow2(quant_state.state2.blocksize))
        or not _TRITON_DEQUANT
        or not _Q.triton_quantize_available(A.device)
    ):
        return None
    try:
        from unsloth.kernels.nf4 import dequantize_nf4
    except Exception:
        return None
    if quant_state.nested:
        state2 = quant_state.state2
        code2, absmax2, offset, blocksize2 = (
            state2.code,
            state2.absmax,
            quant_state.offset,
            state2.blocksize,
        )
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset, dtype = torch.float32, device = A.device)
    else:
        code2 = absmax2 = offset = None
        blocksize2 = 0
    code = quant_state.code
    if code is None:
        code = get_4bit_type("nf4", device = A.device)
    return dequantize_nf4(
        A.reshape(-1).view(torch.uint8),
        absmax,
        code2,
        absmax2,
        offset,
        code,
        quant_state.blocksize,
        blocksize2,
        quant_state.shape,
        quant_state.dtype,
        out = out,
    )


def dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type = "fp4",
) -> torch.Tensor:
    if blocksize is None:
        blocksize = 64
    if quant_state is None:
        if absmax is None or out is None:
            raise ValueError(
                "dequantize_4bit requires both absmax and out when quant_state is not provided"
            )
        quant_state = QuantState(
            absmax = absmax,
            shape = out.shape,
            dtype = out.dtype,
            blocksize = blocksize,
            quant_type = quant_type,
        )
    if quant_state.blocksize not in _BLOCKSIZES:
        raise ValueError(f"invalid blocksize {quant_state.blocksize}")
    if quant_state.quant_type not in ("nf4", "fp4"):
        raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_state.quant_type!r}")
    if quant_state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {quant_state.dtype}"
        )
    result = _dequantize_4bit_impl(A, quant_state, out)
    # Callers passing the packed weight transposed ([1, n // 2]) get the output transposed back.
    if A.shape[0] == 1:
        return result.t()
    return result


def _dequantize_4bit_impl(
    A,
    quant_state,
    out = None,
    use_triton = True,
):
    """Dequantize to quant_state.shape (no transposed-input handling)."""
    result = None
    try:
        if use_triton:
            result = _triton_dequant(A, quant_state.absmax, quant_state, out)
    except Exception as error:
        if not _Q.triton_failed(error):
            raise
    if result is None:
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax += quant_state.offset
            if absmax.dtype != torch.float32:
                absmax = absmax.float()
        result = _Q.dequantize_4bit_blockwise(
            A,
            absmax,
            quant_state.blocksize,
            quant_state.quant_type,
            quant_state.shape,
            quant_state.dtype,
            code = quant_state.code,
        )
        if out is not None:
            result = out.copy_(result)
    return result


def dequantize_fp4(
    A,
    quant_state = None,
    absmax = None,
    out = None,
    blocksize = None,
):
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_nf4(
    A,
    quant_state = None,
    absmax = None,
    out = None,
    blocksize = None,
):
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


def gemv_4bit(
    A,
    B,
    out = None,
    transposed_A = False,
    transposed_B = False,
    state = None,
):
    """A @ dequantize(B).T, the shape bitsandbytes' gemv_4bit returns."""
    if state is None:
        raise ValueError(
            "state cannot be None. gemv_4bit() requires the state from quantize_4bit()"
        )
    W = _dequantize_4bit_impl(B.reshape(-1, 1), state).to(A.dtype)
    result = torch.nn.functional.linear(A, W)
    if out is not None:
        return out.copy_(result)
    return result
