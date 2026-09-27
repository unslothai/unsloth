# SPDX-License-Identifier: AGPL-3.0-only
"""GLM-5.3 (glm_moe_dsa) ships block-FP8 `kv_a_proj_with_mqa` as (576, 6144) with a (5, 48) scale grid: the
last 128-row block is ragged. transformers' Fp8Dequantize derives the block from rows // scale_rows and raises,
so a 16-bit (dequantizing) load of the checkpoint fails. Unsloth pads to the configured block and slices back."""

from types import SimpleNamespace

import pytest
import torch

fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
Fp8Dequantize = getattr(fp8, "Fp8Dequantize", None)
if Fp8Dequantize is None:
    pytest.skip("transformers has no Fp8Dequantize (FP8 dequantize loads)", allow_module_level = True)

import unsloth.import_fixes as import_fixes  # noqa: E402


def _reference(q, s, bm, bn):
    sf = s.float().repeat_interleave(bm, 0)[: q.shape[0]].repeat_interleave(bn, 1)[:, : q.shape[1]]
    return (q.float() * sf).to(torch.bfloat16)


def _op():
    patch = getattr(import_fixes, "_pad_fp8_dequantize_ragged_blocks", None)
    if patch is not None:
        patch(Fp8Dequantize)
    return Fp8Dequantize(
        SimpleNamespace(quantization_config = SimpleNamespace(weight_block_size = [128, 128]))
    )


def _dequant(q, s):
    op = _op()
    if hasattr(op, "_dequantize_one"):
        return op._dequantize_one(q, s, output_dtype = torch.bfloat16)
    # transformers 5.5: the whole conversion op, block from the config.
    return op.convert({"weight$": [q], "weight_scale_inv": [s]}, full_layer_name = "w")["w"]


def test_ragged_rows_dequantize_like_block_reference():
    torch.manual_seed(0)
    q = (torch.randn(576, 384) * 50).to(torch.float8_e4m3fn)
    s = torch.rand(5, 3) + 0.1
    out = _dequant(q, s)
    assert out.shape == (576, 384)
    assert torch.equal(out.to(torch.bfloat16), _reference(q, s, 128, 128))


def test_divisible_shapes_unchanged():
    torch.manual_seed(1)
    q = (torch.randn(256, 256) * 50).to(torch.float8_e4m3fn)
    s = torch.rand(2, 2) + 0.1
    out = _dequant(q, s)
    assert torch.equal(out.to(torch.bfloat16), _reference(q, s, 128, 128))


def test_inconsistent_grid_still_raises():
    q = (torch.randn(576, 384)).to(torch.float8_e4m3fn)
    s = torch.rand(7, 3)  # ceil(576 / 128) = 5 != 7
    with pytest.raises(ValueError):
        _dequant(q, s)


def test_ragged_rows_already_cast_weight():
    # transformers can hand the FP8 weight over already cast to the load dtype.
    torch.manual_seed(2)
    q = (torch.randn(576, 384) * 50).to(torch.float8_e4m3fn)
    s = torch.rand(5, 3) + 0.1
    out = _dequant(q.to(torch.bfloat16), s)
    assert torch.equal(out.to(torch.bfloat16), _reference(q, s, 128, 128))


def test_ragged_columns_and_stacked_experts():
    # 200 columns over 2 scale columns divides exactly, so transformers would use a 100-wide block.
    torch.manual_seed(3)
    q = (torch.randn(2, 256, 200) * 50).to(torch.float8_e4m3fn)
    s = torch.rand(2, 2, 2) + 0.1
    out = _dequant(q, s)
    assert out.shape == (2, 256, 200)
    for e in range(2):
        assert torch.equal(out[e].to(torch.bfloat16), _reference(q[e], s[e], 128, 128))
