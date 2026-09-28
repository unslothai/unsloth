# SPDX-License-Identifier: AGPL-3.0-only
"""Block-FP8 weights whose last block is partial (GLM-5.3 / Ling-3.0 kv_a_proj_with_mqa: 576 rows, 5 scale rows) dequantize on 16-bit loads."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
if not hasattr(fp8, "Fp8Dequantize"):
    pytest.skip("transformers 4.x has no Fp8Dequantize, nothing to patch", allow_module_level = True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_fixes():
    # By file spec, so unsloth/__init__ (zoo, GPU init) does not run.
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_fp8_ragged_under_test", REPO_ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Quantizer:
    quantization_config = type("Cfg", (), {"weight_block_size": [128, 128]})()


def _op_cls(patched):
    # A subclass, so patching never leaks into the real class other tests use.
    cls = type("Fp8DequantizeUnderTest", (fp8.Fp8Dequantize,), {})
    if patched:
        _import_fixes()._pad_fp8_dequantize_ragged_blocks(cls)
    return cls


def _dequant(
    q,
    s,
    patched = True,
):
    op = _op_cls(patched)(_Quantizer())
    if hasattr(op, "_dequantize_one"):
        return op._dequantize_one(q, s, output_dtype = torch.bfloat16)
    # transformers 5.5: the whole conversion op, block from the config.
    return op.convert({"weight$": [q], "weight_scale_inv": [s]}, full_layer_name = "w")["w"]


def _reference(q, s):
    sf = s.float().repeat_interleave(128, -2)[..., : q.shape[-2], :]
    sf = sf.repeat_interleave(128, -1)[..., : q.shape[-1]]
    return (q.float() * sf).to(torch.bfloat16)


def _case(*shape, seed = 0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(*shape, generator = g) * 50).to(torch.float8_e4m3fn)
    grid = (*shape[:-2], -(-shape[-2] // 128), -(-shape[-1] // 128))
    return q, torch.rand(*grid, generator = g) + 0.1


@pytest.mark.parametrize("cols", [384, 2560], ids = ["glm5_3", "ling3"])
def test_ragged_rows_match_block_reference(cols):
    q, s = _case(576, cols)
    with pytest.raises(ValueError):
        _dequant(q, s, patched = False)
    out = _dequant(q, s)
    assert out.shape == (576, cols)
    assert torch.equal(out.to(torch.bfloat16), _reference(q, s))


def test_ragged_rows_already_cast_weight():
    # The loader can hand the FP8 weight over already cast to the load dtype.
    q, s = _case(576, 2560, seed = 2)
    assert torch.equal(_dequant(q.to(torch.bfloat16), s).to(torch.bfloat16), _reference(q, s))


def test_ragged_columns_and_stacked_experts():
    # 200 columns over 2 scale columns divides exactly, so transformers 5.17 would use a 100-wide block.
    q, s = _case(2, 256, 200, seed = 3)
    out = _dequant(q, s)
    assert out.shape == (2, 256, 200)
    assert torch.equal(out.to(torch.bfloat16), _reference(q, s))


def test_divisible_shapes_unchanged_and_idempotent():
    q, s = _case(512, 384, seed = 1)
    assert torch.equal(_dequant(q, s), _dequant(q, s, patched = False))
    cls = _op_cls(patched = True)
    attr = "_dequantize_one" if hasattr(cls, "_dequantize_one") else "convert"
    wrapped = getattr(cls, attr)
    _import_fixes()._pad_fp8_dequantize_ragged_blocks(cls)
    assert getattr(cls, attr) is wrapped and wrapped._unsloth_ragged_blocks


def test_grid_that_is_not_a_ceil_tiling_still_raises():
    q, _ = _case(576, 256)
    with pytest.raises(ValueError):
        _dequant(q, torch.ones(7, 2))  # no block edge gives ceil(576 / b) == 7
