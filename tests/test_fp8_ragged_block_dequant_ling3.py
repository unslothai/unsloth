# SPDX-License-Identifier: AGPL-3.0-only
"""Ling-3.0-flash-VL-fp8: load_in_16bit on a block-fp8 checkpoint whose weight edge is not a whole number of blocks.

Block quantizers tile a ragged edge with one partial block, so MLA's kv_a_proj_with_mqa
(576 x hidden, DeepSeek-V3 / Ling-3.0 layout) ships ceil(576 / 128) = 5 scale rows.
transformers' Fp8Dequantize derived the block as rows // scale_rows and raised
"Weight shape (576, 2560) not divisible by scale grid (5, 20)" (inclusionAI/Ling-3.0-flash-VL-fp8).
import_fixes.py is loaded by file spec so unsloth/__init__ (torch, zoo, GPU init) is not run.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
if not hasattr(fp8, "Fp8Dequantize") or not hasattr(fp8.Fp8Dequantize, "_dequantize_one"):
    pytest.skip(
        "transformers without the Fp8Dequantize conversion op (4.x)", allow_module_level = True
    )
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch without float8", allow_module_level = True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_fp8_ragged_ling3_under_test",
        REPO_ROOT / "unsloth" / "import_fixes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Cfg:
    weight_block_size = [128, 128]


class _Quantizer:
    quantization_config = _Cfg()


def _fresh_op_cls():
    # A subclass, so patching it never leaks into the real class other tests use.
    return type("Fp8DequantizeUnderTest", (fp8.Fp8Dequantize,), {})


def _reference(
    w8,
    s,
    block = 128,
):
    rows, cols = w8.shape
    full = s.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    return (w8.float() * full).to(torch.bfloat16)


def _case(
    rows,
    cols,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(rows, cols, generator = g) * 2).to(torch.float8_e4m3fn)
    s = torch.rand(-(-rows // 128), -(-cols // 128), generator = g) + 0.5
    return w, s


def test_ceil_tiled_rows_dequantize_like_the_block_quantizer():
    ifx = _load_import_fixes()
    cls = _fresh_op_cls()
    op = cls(_Quantizer())
    w, s = _case(576, 256)
    with pytest.raises(ValueError, match = "not divisible by scale grid"):
        op._dequantize_one(w, s, output_dtype = torch.bfloat16)  # before the fix
    ifx._pad_fp8_dequantize_ragged_blocks(cls)
    out = op._dequantize_one(w, s, output_dtype = torch.bfloat16)
    assert out.shape == (576, 256) and out.dtype == torch.bfloat16
    assert torch.equal(out, _reference(w, s))


def test_fp8_values_already_cast_to_the_model_dtype():
    """The loader passes the fp8 weight cast to bf16 (exact), which is what reached the op on Ling-3.0-flash-VL-fp8."""
    ifx = _load_import_fixes()
    cls = _fresh_op_cls()
    op = cls(_Quantizer())
    w, s = _case(576, 2560, seed = 2)
    w16 = w.to(torch.bfloat16)
    with pytest.raises(ValueError, match = "not divisible by scale grid"):
        op._dequantize_one(w16, s, output_dtype = torch.bfloat16)
    ifx._pad_fp8_dequantize_ragged_blocks(cls)
    assert torch.equal(op._dequantize_one(w16, s, output_dtype = torch.bfloat16), _reference(w, s))


def test_divisible_shapes_are_bit_identical_to_the_original():
    ifx = _load_import_fixes()
    cls = _fresh_op_cls()
    op = cls(_Quantizer())
    w, s = _case(512, 384, seed = 1)
    before = op._dequantize_one(w, s, output_dtype = torch.bfloat16)
    ifx._pad_fp8_dequantize_ragged_blocks(cls)
    after = op._dequantize_one(w, s, output_dtype = torch.bfloat16)
    assert torch.equal(before, after)
    ifx._pad_fp8_dequantize_ragged_blocks(cls)  # idempotent
    assert getattr(cls._dequantize_one, "_unsloth_ragged_blocks", False)


def test_a_grid_that_is_not_a_ceil_tiling_still_raises():
    ifx = _load_import_fixes()
    cls = _fresh_op_cls()
    ifx._pad_fp8_dequantize_ragged_blocks(cls)
    op = cls(_Quantizer())
    w, _ = _case(576, 256)
    bad = torch.ones(7, 2)  # no block edge gives ceil(576 / b) == 7
    with pytest.raises(ValueError, match = "not divisible by scale grid"):
        op._dequantize_one(w, bad, output_dtype = torch.bfloat16)
