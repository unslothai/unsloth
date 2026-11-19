# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The NF4 dequant cache must not serve a dense weight decoded with scales that were since written in place:
absmax, and with nested quantization (compress_statistics) the second-level absmax and the offset. Runs on CPU
with real bitsandbytes NF4 states and a faked memory reading."""

import pytest
import torch
from unsloth.kernels import dequant_cache as dc
from unsloth.kernels import utils as kernel_utils

GiB = 2**30
bnb_functional = pytest.importorskip("bitsandbytes.functional")


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    print("dequant_cache imported from", dc.__file__)
    monkeypatch.setattr(dc, "_MODE", "auto")
    monkeypatch.setattr(dc, "_DISABLED", False)
    monkeypatch.setattr(dc, "_auto_allowed", lambda: True)
    monkeypatch.setattr(dc, "_ACTIVE", 1, raising = False)
    monkeypatch.setattr(dc, "_foreign_bytes", lambda index: 0, raising = False)
    for name in (
        "_TRANSIENT",
        "_SEEN",
        "_CALLS",
        "_ROWS_SEEN",
        "_ROWS_NEW",
        "_FOREIGN",
        "_CHECKED",
    ):
        getattr(dc, name).clear()
    dc.clear_dequant_cache()
    dc.STATS.clear()
    stats = {"allocated_bytes.all.current": 3 * GiB, "allocated_bytes.all.peak": 4 * GiB}
    monkeypatch.setattr(dc, "_usable", lambda index: (10 * GiB, 16 * GiB, stats))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index = None: (10 * GiB, 16 * GiB))
    monkeypatch.setattr(dc, "_fraction", lambda index: 1.0, raising = False)
    # CPU tensors: let the eager entry point reach the cache, decoding with bitsandbytes' reference.
    monkeypatch.setattr(dc, "_eligible", lambda W, dtype: True)
    monkeypatch.setattr(
        kernel_utils,
        "fast_dequantize",
        lambda W, quant_state, use_global_buffer = False, dtype = None: bnb_functional.dequantize_4bit(
            W, quant_state
        ),
    )
    dc._BUDGET[None] = GiB
    yield
    dc.clear_dequant_cache()


def _state(nested):
    torch.manual_seed(0)
    dense = torch.randn(64, 128, dtype = torch.bfloat16)
    return bnb_functional.quantize_4bit(dense, quant_type = "nf4", compress_statistics = nested)


def _scale(quant_state, which):
    if which == "absmax":
        if quant_state.nested:
            quant_state.absmax.add_(3)  # uint8 codes into the second-level table
        else:
            quant_state.absmax.mul_(2)
    elif which == "absmax2":
        quant_state.state2.absmax.mul_(2)
    else:
        quant_state.offset.add_(1)


@pytest.mark.parametrize(
    "nested, which",
    [(False, "absmax"), (True, "absmax"), (True, "absmax2"), (True, "offset")],
)
def test_in_place_scale_write_is_not_served_stale(nested, which):
    W, quant_state = _state(nested)
    first = dc.dequantize(W, quant_state)
    assert dc.cached_count() == 1 and dc.STATS["admits"] == 1
    assert dc.dequantize(W, quant_state) is first and dc.STATS["hits"] == 1
    _scale(quant_state, which)
    fresh = bnb_functional.dequantize_4bit(W, quant_state)
    assert not torch.equal(fresh, first)  # the write changed the decoded weight
    out = dc.dequantize(W, quant_state)
    assert torch.equal(out, fresh), (out.float() - fresh.float()).abs().max().item()
    assert dc.STATS["stale"] == 1
    assert dc.dequantize(W, quant_state) is out  # re-admitted, hits again


def test_unchanged_scales_still_hit():
    W, quant_state = _state(True)
    first = dc.dequantize(W, quant_state)
    for _ in range(3):
        assert dc.dequantize(W, quant_state) is first
    assert dc.STATS["hits"] == 3 and dc.STATS.get("stale", 0) == 0
