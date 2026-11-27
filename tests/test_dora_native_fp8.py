# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""LoraConfig(use_dora = True) on a native fp8 checkpoint (transformers FineGrainedFP8 / ModelOpt FP8Linear:
float8 weight + weight_scale_inv, FbgemmFP8Linear: weight_scale). PEFT's DoRA read the raw float8 weight, so
get_peft_model failed (or the magnitude ignored the scales). Unsloth now hands DoRA the dequantized weight."""

import ast
import os

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
if getattr(torch, "float8_e4m3fn", None) is None:
    pytest.skip("torch without float8_e4m3fn", allow_module_level = True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FP8Linear(torch.nn.Linear):
    """The parameter layout transformers' FP8Linear keeps: float8 weight, block `weight_scale_inv`, block_size."""

    def __init__(
        self,
        in_features,
        out_features,
        block,
        scale_name = "weight_scale_inv",
        per_row = False,
    ):
        super().__init__(in_features, out_features, bias = False)
        torch.manual_seed(in_features * 7 + out_features)
        dense = torch.randn(out_features, in_features)
        if per_row:
            scale = dense.abs().amax(1, keepdim = True) / 448.0
            q = (dense / scale).to(torch.float8_e4m3fn)
        else:
            bm, bn = block
            p, q_ = -(-out_features // bm), -(-in_features // bn)
            scale = torch.rand(p, q_) * 0.02 + 0.01
            full = scale.repeat_interleave(bm, 0)[:out_features].repeat_interleave(bn, 1)[
                :, :in_features
            ]
            q = (dense / full).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.block_size = tuple(block)
        self.weight = torch.nn.Parameter(q, requires_grad = False)
        setattr(self, scale_name, torch.nn.Parameter(scale, requires_grad = False))
        self._scale_name = scale_name

    def dense(self):
        scale = getattr(self, self._scale_name).double()
        w = self.weight.double()
        if getattr(self, "block_size", None) is None:  # per-row
            return w * scale
        bm, bn = self.block_size
        full = scale.repeat_interleave(bm, 0)[: w.shape[0]].repeat_interleave(bn, 1)[
            :, : w.shape[1]
        ]
        return w * full

    def forward(self, x):
        return x @ self.dense().to(x.dtype).t()


class _Model(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.q_proj = _FP8Linear(48, 32, (16, 16), **kw)
        self.v_proj = _FP8Linear(40, 24, (16, 16), **kw)  # 40 does not tile into 16
        self.k_proj = _FP8Linear(16, 32, (16, 16), **kw)  # one block column: (2, 1) block scales

    def forward(self, x):
        return (
            self.q_proj(x).sum() + self.v_proj(x[..., :40]).sum() + self.k_proj(x[..., :16]).sum()
        )


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kw", [{}, {"scale_name": "weight_scale", "per_row": True}])
def test_dora_on_native_fp8_uses_the_dequantized_weight(kw, device):
    from peft import LoraConfig, get_peft_model
    from unsloth.models import loader_utils

    model = _Model(**kw)
    refs = {name: getattr(model, name).dense() for name in ("q_proj", "v_proj", "k_proj")}
    model.to(device)
    # What FastLanguageModel / FastModel.get_peft_model now run first (missing before: the raw-fp8 crash).
    patch = getattr(loader_utils, "_patch_peft_for_native_fp8", None)
    if patch is not None:
        assert patch(model)
    peft_model = get_peft_model(
        model, LoraConfig(r = 4, target_modules = ["q_proj", "v_proj", "k_proj"], use_dora = True)
    )
    for name, ref in refs.items():
        layer = getattr(peft_model.base_model.model, name)
        magnitude = layer.lora_magnitude_vector["default"].weight
        # lora_B starts at zero, so DoRA's magnitude is the row norm of the dense base weight.
        torch.testing.assert_close(magnitude.double().cpu(), ref.norm(dim = 1), rtol = 1e-2, atol = 1e-3)
        dense = loader_utils._routed_dense_weight(layer.get_base_layer())
        torch.testing.assert_close(dense.double().cpu(), ref, rtol = 1e-2, atol = 1e-3)
    out = peft_model(torch.randn(2, 48, device = device))
    assert torch.isfinite(out)
    out.backward()
    lora_a = peft_model.base_model.model.q_proj.lora_A["default"].weight
    assert lora_a.grad is not None and torch.isfinite(lora_a.grad).all()


def test_no_fp8_linear_leaves_peft_alone():
    from unsloth.models import loader_utils
    assert not loader_utils._patch_peft_for_native_fp8(torch.nn.Sequential(torch.nn.Linear(4, 4)))
    assert loader_utils._native_fp8_dense_weight(torch.nn.Linear(4, 4)) is None


@pytest.mark.parametrize("path", ["unsloth/models/llama.py", "unsloth/models/vision.py"])
def test_get_peft_model_patches_before_creating_the_adapters(path):
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    found = False
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "get_peft_model"):
            continue
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("_patch_peft_for_native_fp8", "_get_peft_model")
        ]
        patch = [c.lineno for c in calls if c.func.id == "_patch_peft_for_native_fp8"]
        create = [c.lineno for c in calls if c.func.id == "_get_peft_model"]
        if create:
            found = True
            assert patch and min(patch) < min(create), path
    assert found, path
