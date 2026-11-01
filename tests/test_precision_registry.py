# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth.precision: recipe table, refusals before model mutation, structural high-precision layers,
engagement, and the existing float8_training / qat_scheme paths wired through it (CPU only)."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import warnings

import pytest
import torch

from unsloth import precision as P
from unsloth.models import float8_training as f8t

needs_torchao = pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason = "require torchao")


def _tiny_qwen3(layers = 2, tie = False, dtype = torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256, hidden_size = 128, intermediate_size = 256, num_hidden_layers = layers,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32, tie_word_embeddings = tie,
    )
    return Qwen3ForCausalLM(config).to(dtype)


def _tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size = 256, hidden_size = 128, intermediate_size = 256, moe_intermediate_size = 64,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        num_experts = 8, num_experts_per_tok = 2, decoder_sparse_step = 1, mlp_only_layers = [],
    )
    return Qwen3MoeForCausalLM(config)


class _Expert(torch.nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.up = torch.nn.Linear(h, i, bias = False)
        self.down = torch.nn.Linear(i, h, bias = False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class _RemoteMoE(torch.nn.Module):
    """Remote-code style MoE block: the router is called `wg`, which no name rule matches."""

    def __init__(self, h = 2048, e = 64):
        super().__init__()
        self.wg = torch.nn.Linear(h, e, bias = False)
        self.experts = torch.nn.ModuleList([_Expert(h, 32) for _ in range(e)])
        self.shared_experts = _Expert(h, 32)


class _RemoteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_RemoteMoE() for _ in range(2)])
        self.lm_head = torch.nn.Linear(2048, 64, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


# ---- the table ----------------------------------------------------------------------------------------

def test_float8_names_and_gates_unchanged():
    # The literals main had before the registry; byte-identical behaviour for the shipped recipe.
    assert f8t.FLOAT8_TRAINING_RECIPES == ("tensorwise",)
    assert f8t._MEASURED_FASTER == {"tensorwise": ((12, 0), (12, 1))}


def test_every_recipe_is_well_formed():
    gpu_classes = {"sm75", "sm80", "sm89", "sm90", "sm100", "sm120", "rocm"}
    for name, r in P.RECIPES.items():
        assert r.kind in ("dtype", "float8_training", "qat", "frozen_base"), name
        assert r.status in ("shipped", "planned", "refused"), name
        assert set(r.gates) <= gpu_classes, name
        for g in r.gates.values():
            assert g.status in ("named", "explicit", "refused", "unprobed"), name
            assert g.status != "refused" or g.reason, name
        assert r.status != "refused" or r.refusal, name
        assert set(r.input_dtypes) <= {"bf16", "fp16", "fp32"}, name
        assert r.grad_scaler in ("none", "dynamic", "per_dtype"), name
        if r.alias_of:
            assert r.alias_of in P.RECIPES, name
    for short, full in P.FLOAT8_TRAINING_NAMES.items():
        assert P.RECIPES[full].kind == "float8_training", short


def test_paper_recipes_encoded():
    mx = P.get_recipe("mxfp8")
    for gemm in (mx.fwd, mx.dgrad, mx.wgrad):  # arXiv 2506.08027: e4m3 everywhere, 1x32, e8m0, round up
        for op in (gemm.a, gemm.b):
            assert (op.dtype, op.granularity, op.scale_dtype, op.rounding) == ("e4m3", "block1x32", "e8m0", "rceil")
    nv = P.get_recipe("nvfp4")  # arXiv 2509.25149
    assert nv.fwd.b.granularity == "block16x16" and nv.dgrad.b.granularity == "block16x16"
    assert nv.fwd.a.rounding == "rn" and nv.dgrad.a.rounding == "sr" and nv.wgrad.a.rounding == "sr"
    assert nv.wgrad.a.rht and nv.wgrad.b.rht and not nv.fwd.a.rht and not nv.dgrad.a.rht
    assert all(op.scale_dtype == "e4m3" for g in (nv.fwd, nv.dgrad, nv.wgrad) for op in (g.a, g.b))
    assert nv.hp_layers.last_frac == pytest.approx(0.15)
    tw = P.get_recipe("fp8_tensorwise")
    assert tw.fwd.a.dtype == "e4m3" and tw.dgrad.a.dtype == "e5m2"
    assert P.get_recipe("fp8_rowwise_gw_hp").wgrad.low_precision is False


def test_qat_schemes_match_prepare_model_for_qat():
    # Every string scheme _prepare_model_for_qat branches on has a registry row, and no row is stale.
    from unsloth.models import _utils

    tree = ast.parse(inspect.getsource(_utils._prepare_model_for_qat))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "qat_scheme":
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    names.add(c.value)
    assert names, "no scheme comparisons found"
    rows = {n[len("qat:"):] for n in P.recipes_of_kind("qat")}
    assert names == rows


# ---- refusals -----------------------------------------------------------------------------------------

def test_resolve_shipped_values_unchanged():
    assert f8t.resolve_float8_training(None) is None
    assert f8t.resolve_float8_training(False) is None
    assert f8t.resolve_float8_training(True) == "tensorwise"
    assert f8t.resolve_float8_training("TensorWise") == "tensorwise"
    assert f8t.resolve_float8_training(True, dtype = torch.float16) == "tensorwise"
    with pytest.raises(ValueError, match = "slower than bfloat16"):
        f8t.resolve_float8_training("rowwise")
    with pytest.raises(ValueError, match = "is not supported. Use True or one of"):
        f8t.resolve_float8_training("int3")


def test_refuse_mxfp6_training():
    with pytest.raises(P.PrecisionRefused, match = "MXFP6 has no training kernel"):
        f8t.resolve_float8_training("mxfp6")


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4", "blockwise"])
def test_refuse_fp16_for_formats_whose_quantizer_rejects_it(name):
    with pytest.raises(P.PrecisionRefused, match = "float16|fp16"):
        f8t.resolve_float8_training(name, dtype = torch.float16)


def test_refuse_blockwise_off_sm90(monkeypatch):
    monkeypatch.setattr(f8t._precision, "current_gpu_class", lambda device = None: "sm120")
    with pytest.raises(P.PrecisionRefused, match = "SM90 only"):
        f8t.resolve_float8_training("blockwise")


@pytest.mark.parametrize("name", ["mxfp8", "nvfp4"])
def test_planned_formats_refuse_with_their_gates(name):
    with pytest.raises(P.PrecisionRefused, match = "not available in Unsloth yet"):
        f8t.resolve_float8_training(name, dtype = torch.bfloat16)


def test_gw_hp_refused_by_name_with_the_config_hint():
    with pytest.raises(ValueError, match = "from_recipe_name\\('rowwise_with_gw_hp'\\)"):
        f8t.resolve_float8_training("rowwise_with_gw_hp")


def test_refusal_happens_before_any_load():
    # The loader resolves float8_training before it looks the (nonexistent) repo up or builds anything.
    from unsloth.models import loader

    with pytest.raises(P.PrecisionRefused, match = "MXFP6"):
        loader.FastModel.from_pretrained(model_name = "unsloth/does-not-exist", full_finetuning = True, float8_training = "mxfp6")
    with pytest.raises(P.PrecisionRefused, match = "float16"):
        loader.FastModel.from_pretrained(
            model_name = "unsloth/does-not-exist", full_finetuning = True, float8_training = "mxfp8", dtype = torch.float16
        )


def test_check_recipe_gate_table():
    with pytest.raises(P.PrecisionRefused, match = "fp8 tensor cores"):
        P.check_recipe("fp8_tensorwise", gpu = "sm75")
    with pytest.raises(P.PrecisionRefused, match = "NaN"):
        P.check_recipe("fp8_tensorwise", gpu = "sm89")
    assert P.check_recipe("fp8_tensorwise", gpu = "sm120").status == "named"
    assert P.check_recipe("fp8_rowwise", gpu = "sm89").status == "explicit"
    with pytest.raises(P.PrecisionRefused, match = "block-scaled"):
        P.check_recipe("mxfp8", gpu = "sm89", allow_planned = True)
    with pytest.raises(P.PrecisionRefused, match = "no bfloat16"):
        P.check_recipe("bf16", gpu = "sm75")
    assert P.check_recipe("fp16", gpu = "sm75").status == "named"
    with pytest.raises(P.PrecisionRefused, match = "training modes"):
        P.check_recipe("fp8_tensorwise", mode = "lora")


def test_gate_table_agrees_with_float8_hardware_check():
    # float8_training._check_hardware refuses tensorwise below sm90 and everything below 8.9.
    for gpu in ("sm75", "sm80", "sm89"):
        assert P.get_recipe("fp8_tensorwise").gate(gpu).status == "refused"
    for gpu in ("sm75", "sm80"):
        assert P.get_recipe("fp8_rowwise").gate(gpu).status == "refused"


def test_gpu_class_and_scaler_kind():
    assert P.gpu_class((7, 5)) == "sm75" and P.gpu_class((8, 6)) == "sm80" and P.gpu_class((8, 9)) == "sm89"
    assert P.gpu_class((9, 0)) == "sm90" and P.gpu_class((10, 0)) == "sm100" and P.gpu_class((12, 0)) == "sm120"
    assert P.gpu_class(None) is None and P.gpu_class((9, 0), hip = True) == "rocm"
    assert P.grad_scaler_kind("fp16", torch.float16) == "GradScaler"
    assert P.grad_scaler_kind("fp16", torch.float16, fsdp2 = True) == "ShardedGradScaler"
    assert P.grad_scaler_kind("bf16", torch.bfloat16) == "none"
    assert P.grad_scaler_kind("fp8_tensorwise", torch.float16) == "GradScaler"
    assert P.grad_scaler_kind("fp8_tensorwise", torch.bfloat16) == "none"
    assert P.grad_scaler_kind("mxfp8", "bfloat16") == "none"


# ---- structural high-precision layers ------------------------------------------------------------------

def test_routers_found_by_structure():
    moe = _tiny_qwen3_moe()
    assert set(P.find_routers(moe)) == {"model.layers.0.mlp.gate", "model.layers.1.mlp.gate"}
    remote = _RemoteModel()
    assert set(P.find_routers(remote)) == {"layers.0.wg", "layers.1.wg"}
    assert P.find_routers(_tiny_qwen3()) == {}


def test_hp_layers_policy():
    m = _tiny_qwen3(layers = 8, tie = True)
    keep = P.high_precision_layers(m, P.get_recipe("nvfp4").hp_layers)
    assert "lm_head" in keep and "model.embed_tokens" in keep
    assert {k for k in keep if k.startswith("model.layers.")} == {"model.layers.6", "model.layers.7"}  # ceil(0.15 * 8)
    is_hp = P.hp_filter(m, P.HPLayerPolicy(first_n = 1, last_n = 1))
    assert is_hp(None, "model.layers.0.mlp.up_proj") and is_hp(None, "model.layers.7.self_attn.q_proj")
    assert not is_hp(None, "model.layers.3.mlp.up_proj")
    assert P.decoder_layers(m) == ("model.layers", 8)


def test_num_experts_nested_config():
    assert P.num_experts_of({"text_config": {"num_local_experts": 16}}) == 16
    assert P.num_experts_of({"hidden_size": 8}) is None


@needs_torchao
def test_float8_filter_skips_structural_router():
    # Failing on main: the name rules (gate / router / score / classifier) convert a router called `wg`.
    remote = _RemoteModel()
    fn = f8t.float8_training_filter(remote, "tensorwise", size_filter = False)
    assert not fn(remote.layers[0].wg, "layers.0.wg")
    assert fn(remote.layers[0].experts[0].up, "layers.0.experts.0.up")
    assert not fn(remote.lm_head, "lm_head")


@needs_torchao
@pytest.mark.parametrize("build", [_tiny_qwen3, _tiny_qwen3_moe])
def test_float8_filter_unchanged_on_standard_models(build, monkeypatch):
    m = build()
    for size_filter in (True, False):
        new = f8t.float8_training_filter(m, "tensorwise", size_filter = size_filter)
        new_set = {n for n, mod in m.named_modules() if new(mod, n)}
        with monkeypatch.context() as mp:
            mp.setattr(f8t, "_find_routers", lambda model: {})
            old = f8t.float8_training_filter(m, "tensorwise", size_filter = size_filter)
            old_set = {n for n, mod in m.named_modules() if old(mod, n)}
        assert new_set == old_set


# ---- engagement ------------------------------------------------------------------------------------------

def _emulated_config(recipe = "tensorwise"):
    from torchao.float8 import Float8LinearConfig

    config = Float8LinearConfig.from_recipe_name(recipe)
    object.__setattr__(config, "emulate", True)
    return config


@needs_torchao
def test_engagement_recorded_and_strict(monkeypatch):
    m = _tiny_qwen3()
    converted = f8t.convert_model_to_float8_training(m, _emulated_config(), size_filter = False)
    assert converted and P.engaged(m)["float8_training:Float8LinearConfig(tensorwise)"]["modules"] == len(converted)

    tiny = _tiny_qwen3()  # 128 / 256 wide: the size filter keeps every layer out
    monkeypatch.delenv("UNSLOTH_STRICT_ENGAGE", raising = False)
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        assert f8t.convert_model_to_float8_training(tiny, _emulated_config()) == []
    assert any("no eligible Linear" in str(x.message) for x in w)
    assert P.engaged(tiny)["float8_training:Float8LinearConfig(tensorwise)"]["modules"] == 0
    monkeypatch.setenv("UNSLOTH_STRICT_ENGAGE", "1")
    with pytest.raises(P.EngagementError, match = "engaged on 0 modules"):
        f8t.convert_model_to_float8_training(_tiny_qwen3(), _emulated_config())


@needs_torchao
def test_qat_engagement_recorded():
    from unsloth.models._utils import _prepare_model_for_qat

    m = _prepare_model_for_qat(_tiny_qwen3(dtype = torch.bfloat16), "int4")
    rec = P.engaged(m)["qat:int4"]
    assert rec["modules"] == sum(1 for x in m.modules() if "FakeQuantized" in type(x).__name__) > 0


def test_require_and_counter(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STRICT_ENGAGE", raising = False)
    assert P.require("x", 3) and not P.require("x", 0)
    monkeypatch.setenv("UNSLOTH_STRICT_ENGAGE", "1")
    with pytest.raises(P.EngagementError):
        P.require("x", 0)
    with P.count_lowp_gemms() as c:
        torch.ones(4, 4) @ torch.ones(4, 4)
    assert all(v == 0 for v in c.values())
    a = torch.ones(16, 16).to(torch.float8_e4m3fn)
    b = torch.ones(16, 16).to(torch.float8_e4m3fn).t()
    one = torch.ones((), dtype = torch.float32)
    try:
        with P.count_lowp_gemms() as c:
            torch._scaled_mm(a, b, scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
    except (RuntimeError, NotImplementedError):
        pytest.skip("no CPU _scaled_mm in this torch")
    assert c["_scaled_mm"] == 1
