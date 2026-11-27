# SPDX-License-Identifier: AGPL-3.0-only
"""Falcon-H1 / Nemotron-H fp16: float32 norm output into a float16 nn.Linear outside autocast."""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from unsloth.models._custom_dtype import cast_mixed_linear_inputs  # noqa: E402

VISION = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.q_proj = torch.nn.Linear(8, 4, bias = False)
        self.in_proj = torch.nn.Linear(8, 4, bias = False)

    def forward(self, x):
        h = self.norm(x)
        return self.q_proj(h).float() + self.in_proj(h)


def _falcon_like():
    torch.manual_seed(0)
    model = _Block().float()
    model.q_proj.to(torch.float16)
    return model


def test_plain_forward_raises_without_the_hook():
    model = _falcon_like()
    with pytest.raises(RuntimeError, match = "dtype"):
        model(torch.randn(2, 3, 8))


def test_hook_casts_only_the_moved_projection():
    model = _falcon_like()
    assert cast_mixed_linear_inputs(model, torch.float32) == 1
    assert getattr(model.q_proj, "_unsloth_casts_input_dtype", False)
    assert not getattr(model.in_proj, "_unsloth_casts_input_dtype", False)
    x = torch.randn(2, 3, 8)
    out = model(x)
    h = model.norm(x)
    ref = (h.to(torch.float16) @ model.q_proj.weight.t()).float() + model.in_proj(h)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)


def test_hook_is_idempotent():
    model = _falcon_like()
    assert cast_mixed_linear_inputs(model, torch.float32) == 1
    assert cast_mixed_linear_inputs(model, torch.float32) == 0
    assert len(model.q_proj._forward_pre_hooks) == 1


def test_matching_dtypes_and_integer_inputs_are_untouched():
    model = _falcon_like()
    cast_mixed_linear_inputs(model, torch.float32)
    hook = next(iter(model.q_proj._forward_pre_hooks.values()))
    x16 = torch.randn(2, 8, dtype = torch.float16)
    assert hook(model.q_proj, (x16,)) is None
    assert hook(model.q_proj, (torch.ones(2, 8, dtype = torch.long),)) is None
    assert hook(model.q_proj, ()) is None


def test_autocast_is_left_to_do_the_cast():
    model = _falcon_like()
    cast_mixed_linear_inputs(model, torch.float32)
    hook = next(iter(model.q_proj._forward_pre_hooks.values()))
    x32 = torch.randn(2, 8)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        assert hook(model.q_proj, (x32,)) is None
    (cast,) = hook(model.q_proj, (x32,))
    assert cast.dtype == torch.float16


def test_no_hook_when_nothing_moved_off_the_load_dtype():
    model = _Block().to(torch.float16)
    assert cast_mixed_linear_inputs(model, torch.float16) == 0


def test_float8_weight_linears_get_no_hook():
    class _Fp8Linear(torch.nn.Linear):
        pass

    model = _falcon_like()
    model.fp8 = _Fp8Linear(8, 4, bias = False)
    model.fp8.weight = torch.nn.Parameter(
        model.fp8.weight.data.to(torch.float8_e4m3fn), requires_grad = False
    )
    assert cast_mixed_linear_inputs(model, torch.float32) == 1  # only the float16 q_proj
    assert not getattr(model.fp8, "_unsloth_casts_input_dtype", False)
    assert not model.fp8._forward_pre_hooks
    assert model.q_proj._unsloth_casts_input_dtype


def test_vision_loader_hooks_after_the_custom_dtype_code():
    tree = ast.parse(VISION.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if test != "custom_datatype is not None":
            continue
        body = ast.unparse(node)
        if "exec(custom_datatype)" not in body:
            continue
        assert "cast_mixed_linear_inputs(model, torch_dtype)" in body
        assert body.index("exec(custom_datatype)") < body.index("cast_mixed_linear_inputs(")
        return
    pytest.fail("custom_datatype exec block not found in vision.py")
