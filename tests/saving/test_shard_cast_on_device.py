# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A full fine-tune saved as 16-bit casts each shard tensor to the host as it is written. A blocking
`.to(device = "cpu", dtype = ...)` copies the source dtype over the bus and converts on the host, so
narrowing (an fp32 master saved as fp16 / bf16) is cast on the device first. Widening, host tensors,
a device OOM and NaN-holding tensors keep the host cast, so the bytes written never change.
Sliced with `ast`: importing `unsloth.save` needs a GPU."""

import ast
from pathlib import Path

import pytest
import torch

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
_SOURCE = SAVE_PY.read_text(encoding = "utf-8")


def _cast_to_host():
    namespace = {"torch": torch}
    for node in ast.parse(_SOURCE).body:
        if isinstance(node, ast.ClassDef) and node.name == "_CastShardsOnWrite":
            exec(compile(ast.get_source_segment(_SOURCE, node), str(SAVE_PY), "exec"), namespace)
    return namespace["_CastShardsOnWrite"]._cast_to_host


class _OnDevice:
    """Stands in for an accelerator tensor; records which route the cast took."""

    def __init__(
        self,
        tensor,
        oom = False,
    ):
        self.tensor = tensor
        self.device = torch.device("cuda", 0)
        self.oom = oom
        self.calls = []

    def element_size(self):
        return self.tensor.element_size()

    def to(self, **kwargs):
        self.calls.append(kwargs)
        if "device" not in kwargs and self.oom:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return self.tensor.to(dtype = kwargs["dtype"])


def _values():
    torch.manual_seed(0)
    return torch.cat(
        [
            torch.randn(256),
            torch.logspace(-30, -4, 64, base = 2.0),
            # Overflow and infinities of one sign: +inf and -inf together would sum to NaN and fall back.
            torch.tensor([65504.0, 65520.0, 1e30, float("inf"), 0.0, -0.0]),
        ]
    )


def _same_bytes(a, b):
    return a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrowing_casts_on_the_device_before_the_copy(dtype):
    cast = _cast_to_host()
    v = _OnDevice(_values())
    out = cast(v, dtype)
    assert v.calls == [{"dtype": dtype}]
    assert out.device.type == "cpu"
    assert _same_bytes(out, _values().to(dtype = dtype))


def test_widening_stays_a_host_cast():
    cast = _cast_to_host()
    v = _OnDevice(_values().to(torch.bfloat16))
    out = cast(v, torch.float32)
    assert v.calls == [{"device": "cpu", "dtype": torch.float32}]
    assert out.dtype == torch.float32


def test_host_tensors_are_cast_as_before():
    cast = _cast_to_host()
    v = _values()
    out = cast(v, torch.bfloat16)
    assert out.device.type == "cpu"
    assert _same_bytes(out, v.to(dtype = torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_device_oom_falls_back_to_the_host_cast(dtype):
    cast = _cast_to_host()
    v = _OnDevice(_values(), oom = True)
    out = cast(v, dtype)
    assert v.calls == [{"dtype": dtype}, {"device": "cpu", "dtype": dtype}]
    assert _same_bytes(out, _values().to(dtype = dtype))


def test_a_tensor_holding_nan_keeps_the_host_cast():
    # Device and host casts write different NaN payloads; only the host cast reproduces the old bytes.
    cast = _cast_to_host()
    values = _values()
    values[3] = float("nan")
    v = _OnDevice(values)
    cast(v, torch.float16)
    assert v.calls == [{"dtype": torch.float16}, {"device": "cpu", "dtype": torch.float16}]


def test_opposite_infinities_fall_back_and_still_match():
    cast = _cast_to_host()
    values = torch.tensor([1.0, 1e30, -1e30, float("inf"), float("-inf")])
    v = _OnDevice(values)
    out = cast(v, torch.float16)
    assert v.calls == [{"dtype": torch.float16}, {"device": "cpu", "dtype": torch.float16}]
    assert _same_bytes(out, values.to(dtype = torch.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_bytes_match_the_host_cast(dtype):
    cast = _cast_to_host()
    torch.manual_seed(0)
    patterns = torch.randint(-(2**31), 2**31 - 1, (1 << 20,), dtype = torch.int64).to(torch.int32)
    with_nan = torch.cat([_values(), patterns.view(torch.float32)]).cuda()
    finite = torch.cat([_values(), -_values()[:256]]).cuda()
    for v in (with_nan, finite):
        assert _same_bytes(cast(v, dtype), v.to(device = "cpu", dtype = dtype))
