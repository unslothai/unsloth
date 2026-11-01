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

needs_torchao = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)


def _tiny_qwen3(
    layers = 2,
    tie = False,
    dtype = torch.float32,
):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 32,
        tie_word_embeddings = tie,
    )
    return Qwen3ForCausalLM(config).to(dtype)


def _tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        moe_intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 32,
        num_experts = 8,
        num_experts_per_tok = 2,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
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

    def __init__(
        self,
        h = 2048,
        e = 64,
    ):
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
    for gemm in (
        mx.fwd,
        mx.dgrad,
        mx.wgrad,
    ):  # arXiv 2506.08027: e4m3 everywhere, 1x32, e8m0, round up
        for op in (gemm.a, gemm.b):
            assert (op.dtype, op.granularity, op.scale_dtype, op.rounding) == (
                "e4m3",
                "block1x32",
                "e8m0",
                "rceil",
            )
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
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "qat_scheme"
        ):
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    names.add(c.value)
    assert names, "no scheme comparisons found"
    rows = {n[len("qat:") :] for n in P.recipes_of_kind("qat")}
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
        loader.FastModel.from_pretrained(
            model_name = "unsloth/does-not-exist", full_finetuning = True, float8_training = "mxfp6"
        )
    with pytest.raises(P.PrecisionRefused, match = "float16"):
        loader.FastModel.from_pretrained(
            model_name = "unsloth/does-not-exist",
            full_finetuning = True,
            float8_training = "mxfp8",
            dtype = torch.float16,
        )


def test_gate_evidence_names_no_local_paths():
    # Evidence is printed in refusals; a scratch path (outputs/, plans/) means nothing to a user.
    for name, r in P.RECIPES.items():
        for gpu, gate in r.gates.items():
            assert not any(p in gate.evidence for p in ("outputs/", "plans/", ".md")), (
                name,
                gpu,
                gate.evidence,
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
    assert (
        P.gpu_class((7, 5)) == "sm75"
        and P.gpu_class((8, 6)) == "sm80"
        and P.gpu_class((8, 9)) == "sm89"
    )
    assert (
        P.gpu_class((9, 0)) == "sm90"
        and P.gpu_class((10, 0)) == "sm100"
        and P.gpu_class((12, 0)) == "sm120"
    )
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


def _tiny_moe_expert_width_collision():
    # Qwen3-Next-80B-A3B has num_experts == moe_intermediate_size == shared_expert_intermediate_size == 512:
    # an expert / shared expert / dense MLP gate_proj is as wide as the router but is not one.
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 16,
        moe_intermediate_size = 16,
        shared_expert_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_experts = 16,
        num_experts_per_tok = 2,
        decoder_sparse_step = 1,
        mlp_only_layers = [0],
    )
    return Qwen2MoeForCausalLM(config)


def test_routers_ignore_mlps_as_wide_as_the_expert_count():
    m = _tiny_moe_expert_width_collision()
    assert set(P.find_routers(m)) == {"model.layers.1.mlp.gate"}


def test_hp_layers_policy():
    m = _tiny_qwen3(layers = 8, tie = True)
    keep = P.high_precision_layers(m, P.get_recipe("nvfp4").hp_layers)
    assert "lm_head" in keep and "model.embed_tokens" in keep
    assert {k for k in keep if k.startswith("model.layers.")} == {
        "model.layers.6",
        "model.layers.7",
    }  # ceil(0.15 * 8)
    is_hp = P.hp_filter(m, P.HPLayerPolicy(first_n = 1, last_n = 1))
    assert is_hp(None, "model.layers.0.mlp.up_proj") and is_hp(
        None, "model.layers.7.self_attn.q_proj"
    )
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
@pytest.mark.parametrize("build", [_tiny_qwen3, _tiny_qwen3_moe, _tiny_moe_expert_width_collision])
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
    assert converted and P.engaged(m)["float8_training:Float8LinearConfig(tensorwise)"][
        "modules"
    ] == len(converted)

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


def test_counter_runs_fullgraph_compiled_calls():
    # Failing before: a plain TorchDispatchMode makes Dynamo refuse fullgraph frames ("non-infra torch dispatch
    # mode present"), so the engagement check crashed around compiled models.
    a = torch.ones(16, 16).to(torch.float8_e4m3fn)
    b = torch.ones(16, 16).to(torch.float8_e4m3fn).t()
    one = torch.ones((), dtype = torch.float32)

    def f(a, b):
        return torch._scaled_mm(a, b, scale_a = one, scale_b = one, out_dtype = torch.bfloat16)

    try:
        expected = f(a, b)
    except (RuntimeError, NotImplementedError):
        pytest.skip("no CPU _scaled_mm in this torch")
    torch._dynamo.reset()
    compiled = torch.compile(f, backend = "aot_eager", fullgraph = True)
    compiled(a, b)
    with P.count_lowp_gemms() as c:
        out = compiled(a, b)
    assert torch.equal(out, expected)
    assert c["_scaled_mm"] == 1


def test_decoder_layers_is_the_block_stack_not_an_experts_list():
    # Failing before: the longest ModuleList won, so a model with more experts than layers returned
    # model.layers.0.mlp.experts and first_n / last_n kept experts in high precision instead of blocks.
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    config = Qwen2MoeConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        shared_expert_intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_experts = 12,
        num_experts_per_tok = 2,
    )
    with torch.device("meta"):
        m = Qwen2MoeForCausalLM(config)
    assert P.decoder_layers(m) == ("model.layers", 4)
    is_hp = P.hp_filter(m, P.HPLayerPolicy(first_n = 1, last_n = 1))
    assert is_hp(None, "model.layers.0.self_attn.q_proj")
    assert is_hp(None, "model.layers.3.mlp.experts.5.up_proj")
    assert not is_hp(None, "model.layers.1.self_attn.q_proj")
    assert not is_hp(None, "model.layers.1.mlp.experts.0.up_proj")


def test_decoder_layers_is_the_language_decoder_not_a_longer_encoder():
    # Failing before: the longest outer list won anywhere, so a vision tower or speech encoder deeper than
    # the language decoder got the first / last block protection.
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        WhisperConfig,
        WhisperForConditionalGeneration,
    )

    text = LlamaConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    vision = CLIPVisionConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        image_size = 32,
        patch_size = 16,
    )
    with torch.device("meta"):
        llava = LlavaForConditionalGeneration(LlavaConfig(vision_config = vision, text_config = text))
    name, n = P.decoder_layers(llava)
    assert n == 2 and "vision" not in name and name.endswith("layers"), (name, n)
    is_hp = P.hp_filter(llava, P.HPLayerPolicy(first_n = 1, last_n = 1))
    assert is_hp(None, f"{name}.0.self_attn.q_proj") and is_hp(None, f"{name}.1.mlp.down_proj")

    whisper = WhisperConfig(
        vocab_size = 256,
        d_model = 32,
        encoder_layers = 4,
        decoder_layers = 2,
        encoder_attention_heads = 4,
        decoder_attention_heads = 4,
        encoder_ffn_dim = 64,
        decoder_ffn_dim = 64,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        decoder_start_token_id = 1,
    )
    with torch.device("meta"):
        w = WhisperForConditionalGeneration(whisper)
    assert P.decoder_layers(w) == ("model.decoder.layers", 2)


def test_decoder_layers_accepts_a_hybrid_stack_of_mixed_block_classes():
    # Failing before: a stack had to be one block class, so Jamba (Mamba + attention blocks) returned
    # (None, 0) and first_n / last_n protected nothing.
    from transformers import JambaConfig, JambaForCausalLM

    config = JambaConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_experts = 1,
        attn_layer_period = 2,
        attn_layer_offset = 1,
        mamba_d_state = 8,
        mamba_d_conv = 4,
        mamba_expand = 2,
        use_mamba_kernels = False,
    )
    with torch.device("meta"):
        m = JambaForCausalLM(config)
    assert len({type(b) for b in m.model.layers}) == 2
    assert P.decoder_layers(m) == ("model.layers", 4)
    is_hp = P.hp_filter(m, P.HPLayerPolicy(first_n = 1, last_n = 1))
    assert is_hp(None, "model.layers.0.mamba.in_proj") and is_hp(
        None, "model.layers.3.self_attn.q_proj"
    )
    assert not is_hp(None, "model.layers.1.self_attn.q_proj")


def test_counter_counts_opaque_custom_ops_once_at_their_boundary(monkeypatch):
    # Failing before: only torch.ops.aten targets were resolved, so a low-precision GEMM inside an opaque
    # custom op (unsloth::nvfp4_qat_linear, fbgemm::f8f8bf16_rowwise) counted zero.
    from unsloth.precision import engage

    a = torch.ones(16, 16).to(torch.float8_e4m3fn)
    b = torch.ones(16, 16).to(torch.float8_e4m3fn).t()
    one = torch.ones((), dtype = torch.float32)
    try:
        torch._scaled_mm(a, b, scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
    except (RuntimeError, NotImplementedError):
        pytest.skip("no CPU _scaled_mm in this torch")

    name = "unsloth_precision_test::opaque_fp8_mm"
    if not hasattr(torch.ops.unsloth_precision_test, "opaque_fp8_mm"):

        @torch.library.custom_op(name, mutates_args = ())
        def _opaque(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            one = torch.ones((), dtype = torch.float32)
            return torch._scaled_mm(a, b, scale_a = one, scale_b = one, out_dtype = torch.bfloat16)

    monkeypatch.setattr(engage, "LOWP_GEMM_CUSTOM_OPS", engage.LOWP_GEMM_CUSTOM_OPS + (name,))
    with P.count_lowp_gemms() as c:
        torch.ops.unsloth_precision_test.opaque_fp8_mm(a, b)
    assert c[name] == 1 and c["_scaled_mm"] == 0  # the boundary, not the hidden inner call
    assert "fbgemm::f8f8bf16_rowwise" in c  # absent namespace: listed, never resolved


@needs_torchao
def test_counter_counts_the_real_nvfp4_qat_linear_op():
    from unsloth.models._utils import _nvfp4_qat_linear_op

    try:
        op = _nvfp4_qat_linear_op()
    except Exception as e:
        pytest.skip(f"torchao NVFP4 internals unavailable: {e!r}")
    x = torch.empty(16, 64, device = "meta", dtype = torch.bfloat16)
    w = torch.empty(32, 64, device = "meta", dtype = torch.bfloat16)
    with P.count_lowp_gemms() as c:
        op(x, w, None, False, False, False, False, None)
    assert c["unsloth::nvfp4_qat_linear"] == 1


def test_counter_hears_eager_fp8_triton_launchers_and_not_the_fp16_gemv():
    # Failing before: eager fp8 block launchers run Triton without dispatching their op (counted 0), and
    # fp8_block_gemv, which upcasts the weight before its dot, was counted as a low-precision GEMM.
    from unsloth.precision import engage

    assert "unsloth::fp8_block_gemv" not in engage.LOWP_GEMM_CUSTOM_OPS
    with P.count_lowp_gemms() as c:
        engage.release_lowp_gemm(engage.note_lowp_gemm("unsloth::fp8_block_small_m_linear"))
    assert c["unsloth::fp8_block_small_m_linear"] == 1
    engage.note_lowp_gemm("unsloth::fp8_block_small_m_linear")  # no counter open: a no-op
    assert c["unsloth::fp8_block_small_m_linear"] == 1 and not engage._ACTIVE

    fp8 = pytest.importorskip("unsloth.kernels.fp8")
    names = fp8._EAGER_COUNTED_LOWP_GEMMS
    assert "fp8_block_gemv" not in names
    for name in ("fp8_block_matmul_triton", "fp8_block_small_m_linear", "fp8_block_matmul_torchao"):
        assert name in names and f"unsloth::{name}" in engage.LOWP_GEMM_CUSTOM_OPS


def test_mxfp8_qat_registry_admits_only_bfloat16_like_its_backend():
    # Failing before: the qat:mxfp8 row allowed fp32 while _prepare_model_for_qat quantizes bf16 layers only.
    assert P.check_recipe("qat:mxfp8", gpu = "sm100", input_dtype = torch.bfloat16, mode = "full")
    with pytest.raises(ValueError):
        P.check_recipe("qat:mxfp8", gpu = "sm100", input_dtype = torch.float32, mode = "full")


def test_nested_counters_close_by_identity_and_delegated_gemms_count_once():
    # Failing before: list.remove matched the outer (equal, all-zero) dict when the inner scope closed, so
    # the outer cleanup raised; and a launcher that delegates to another counted launcher counted twice.
    from unsloth.precision import engage

    with P.count_lowp_gemms() as outer:
        with P.count_lowp_gemms() as inner:
            pass
        assert any(a is outer for a in engage._ACTIVE) and not any(a is inner for a in engage._ACTIVE)
        entered = engage.note_lowp_gemm("unsloth::fp8_block_matmul_torchao")
        try:  # the delegated Triton launch inside the torchao one
            engage.release_lowp_gemm(engage.note_lowp_gemm("unsloth::fp8_block_matmul_triton"))
        finally:
            engage.release_lowp_gemm(entered)
    assert not engage._ACTIVE
    assert outer["unsloth::fp8_block_matmul_torchao"] == 1
    assert outer.get("unsloth::fp8_block_matmul_triton", 0) == 0
    assert inner.get("unsloth::fp8_block_matmul_torchao", 0) == 0

