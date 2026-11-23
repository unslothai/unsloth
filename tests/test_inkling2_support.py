# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 (transformers >= 5.24): the usual LoRA names land on the Linears inside the wrapped attention
projections, PEFT accepts them, and the released checkpoints' "modelopt" marker skips the generic ModelOpt rewrite."""

import importlib.util
import pathlib
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _inkling2():
    # By path: the helpers are plain Python, so the test needs no `import unsloth`.
    spec = importlib.util.spec_from_file_location(
        "_unsloth_inkling2", _ROOT / "unsloth" / "models" / "inkling2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(loop = True):
    from transformers import AutoModelForImageTextToText
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 32,
        num_hidden_layers = 12,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        swa_num_attention_heads = 2,
        swa_num_key_value_heads = 1,
        swa_head_dim = 16,
        d_rel = 4,
        rel_extent = 8,
        sliding_window_size = 4,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 32,
    )
    vision = dict(
        model_type = "inkling2_vision",
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        projection_hidden_dim = 32,
        text_hidden_size = 32,
    )
    if loop:
        text.update(
            loop_layer_index_start = 6, loop_layer_index_end = 12, loop_num_passes = 2, loop_gate_rank = 4
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(config)


@pytest.mark.parametrize("loop", [True, False])
def test_default_targets_are_attention_linears(loop):
    ink = _inkling2()
    model = _tiny(loop)
    targets = ink.inkling2_lora_target_modules(model, None)
    modules = dict(model.named_modules())
    assert targets and all(isinstance(modules[t], torch.nn.Linear) for t in targets)
    leaves = {re.sub(r".*layers\.\d+\.", "", t) for t in targets}
    assert leaves == {
        "self_attn.q_proj.linear",
        "self_attn.k_proj.linear",
        "self_attn.v_proj.linear",
        "self_attn.o_proj.linear",
        "self_attn.o_proj.gate_proj",
    }
    # Every language layer, nothing from the vision tower, the message gates or the small bias / sink / conv heads.
    assert len(targets) == 5 * 12
    assert not any(
        "vision_tower" in t
        or "message_gate" in t
        or "sink_proj" in t
        or "weight_proj" in t
        or "relative_position_bias" in t
        for t in targets
    )


def test_aliases_scoping_and_passthrough():
    ink = _inkling2()
    model = _tiny()
    only_q = ink.inkling2_lora_target_modules(model, ["q_proj"])
    assert only_q and all(t.endswith("self_attn.q_proj.linear") for t in only_q)
    # Experts are 3D parameters: the MLP aliases add no module targets (get_peft_model trains them by parameter).
    assert ink.inkling2_lora_target_modules(model, ["gate_proj", "up_proj", "down_proj"]) == []
    assert ink.inkling2_lora_target_modules(model, ["q_proj"], attention = False) == []
    # An unknown name stays a PEFT suffix; a regex string is the caller's.
    assert ink.inkling2_lora_target_modules(model, ["sink_proj"]) == ["sink_proj"]
    assert ink.inkling2_lora_target_modules(model, r".*q_proj\.linear") == r".*q_proj\.linear"
    with_vision = ink.inkling2_lora_target_modules(model, None, vision = True)
    vision = [t for t in with_vision if "vision_tower" in t]
    assert {t.rsplit(".", 1)[-1] for t in vision} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    }
    modules = dict(model.named_modules())
    assert all(isinstance(modules[t], torch.nn.Linear) for t in with_vision)


def test_peft_accepts_mapped_targets_and_lora_moves_logits():
    peft = pytest.importorskip("peft")
    ink = _inkling2()
    model = _tiny()
    ink.patch_inkling2_model(model)
    targets = model.unsloth_lora_target_modules(None)
    cfg = peft.LoraConfig(
        r = 2,
        lora_alpha = 4,
        target_modules = targets,
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    model = peft.get_peft_model(model, cfg)
    ids = torch.randint(0, 250, (1, 10))
    with torch.no_grad():
        base = model(input_ids = ids, use_cache = False).logits
        for n, p in model.named_parameters():
            if "lora_B" in n and "q_proj" in n:
                p.normal_(0, 0.5)
        moved = model(input_ids = ids, use_cache = False).logits
    assert torch.isfinite(moved).all() and not torch.allclose(base, moved)


def test_modelopt_marker_is_inkling_owned():
    ink = _inkling2()
    quant = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    cfg = SimpleNamespace(
        model_type = "inkling2_loop",
        quantization_config = dict(quant),
        text_config = SimpleNamespace(model_type = "inkling2_loop_text"),
    )
    assert ink.inkling2_owns_modelopt_quantization(cfg)
    ink.drop_inkling2_quantization_config(cfg, verbose = False)
    assert getattr(cfg, "quantization_config", None) is None
    # A real ModelOpt checkpoint of another architecture still goes through the generic rewrite.
    other = SimpleNamespace(model_type = "llama", quantization_config = dict(quant))
    assert not ink.inkling2_owns_modelopt_quantization(other)
    # An Inkling 2 config without the marker, or with a bitsandbytes one, is left alone.
    bnb = SimpleNamespace(
        model_type = "inkling2", quantization_config = {"quant_method": "bitsandbytes"}
    )
    assert not ink.inkling2_owns_modelopt_quantization(bnb)
    assert not ink.inkling2_owns_modelopt_quantization(SimpleNamespace(model_type = "inkling2"))


def test_check_and_disable_keeps_4bit_for_inkling2():
    loader_utils = pytest.importorskip("unsloth.models.loader_utils")
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    config = Inkling2LoopConfig()
    config.quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    out = loader_utils.check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, verbose = False
    )
    assert out == (True, False, None)
    assert getattr(config, "quantization_config", None) is None


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4(w):
    """The released ModelOpt layout: E2M1 pairs (low nibble = even element), e4m3 scale per 16, amax / denominator."""
    E, R, C = w.shape
    w = w.float()
    amax = w.abs().amax(dim = (1, 2))
    denom = 6.0 * 448.0
    factor = (amax / denom).clamp(min = 1e-12)
    blocks = w.view(E, R, C // 16, 16)
    scale = (
        (blocks.abs().amax(-1) / 6.0 / factor[:, None, None])
        .clamp(max = 448.0)
        .to(torch.float8_e4m3fn)
    )
    x = blocks / (scale.float() * factor[:, None, None]).clamp(min = 1e-30)[..., None]
    idx = (x.abs().clamp(max = 6.0)[..., None] - torch.tensor(_E2M1)).abs().argmin(-1)
    code = (idx + 8 * ((x < 0) & (idx > 0)).long()).view(E, R, C)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, scale, amax, torch.tensor([denom])


def _mxfp8(w):
    E, R, C = w.shape
    blocks = w.float().view(E, R, C // 32, 32)
    exp = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp(min = 1e-30) / 448.0)).clamp(-127, 127)
    return (blocks / torch.exp2(exp)[..., None]).to(torch.float8_e4m3fn).view(E, R, C), (
        exp + 127
    ).to(torch.uint8)


@pytest.mark.parametrize("loop", [True, False])
def test_released_nvfp4_mxfp8_experts_decode(tmp_path, loop):
    """Mini (loop) decodes upstream; Small (inkling2) only once its text mapping gets the loop's expert transforms."""
    import json
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForImageTextToText

    ink = _inkling2()
    assert ink.register_inkling2_expert_dequantization()
    model = _tiny(loop)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "experts" in n:
                p.normal_(0, 0.05)
    model.save_pretrained(tmp_path / "bf16")
    sd = load_file(tmp_path / "bf16" / "model.safetensors")
    out = {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$", k)
        if m is None:
            out[k] = v
        elif int(m.group(1)) == 11:
            out[k], out[k + "_scale"] = _mxfp8(v)
        else:
            out[k], out[k + "_scale"], out[k + "_amax"], out[k + "_outer_scale_denominator"] = (
                _nvfp4(v)
            )
    (tmp_path / "q").mkdir()
    save_file(out, tmp_path / "q" / "model.safetensors", metadata = {"format": "pt"})
    config = json.loads((tmp_path / "bf16" / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (tmp_path / "q" / "config.json").write_text(json.dumps(config))
    ref = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "bf16", dtype = torch.float32
    ).state_dict()
    got = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "q", dtype = torch.float32
    ).state_dict()
    assert set(ref) == set(got)
    for k, v in ref.items():
        if "experts" in k:
            # NVFP4 is ~10% relative error, MXFP8 ~3%; a layout mistake is O(1) or worse.
            assert float((got[k] - v).norm() / v.norm()) < (0.05 if ".11." in k else 0.15), k
        else:
            assert torch.equal(got[k], v), k


def test_4bit_keeps_mini_message_gates_out_of_bitsandbytes():
    # MessageGate.forward casts its input to `down_proj.weight.dtype`, uint8 once bitsandbytes packs it, and its two
    # gate projections are fp32-strict: none of the three may become Linear4bit. Everything else still quantizes.
    from transformers.quantizers.quantizers_utils import should_convert_module
    from unsloth.models.vision import _architecture_skip_modules
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    for model_types in (["inkling2_loop"], ["inkling2_loop", "inkling2_loop_text"]):
        assert len(_architecture_skip_modules(model_types)) == 12
    assert _architecture_skip_modules(["inkling2", "inkling2_text"]) == []

    skip = list(SKIP_QUANTIZATION_MODULES) + _architecture_skip_modules(["inkling2_loop"])
    model = _tiny(loop = True)
    kept, quantized = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "language_model" in name:
            short = re.sub(r"^.*layers\.\d+\.", "", name)
            (quantized if should_convert_module(name, skip) else kept).add(short)
    gates = {
        f"{g}_message_gate.{p}"
        for g in ("q", "k", "v", "mlp")
        for p in ("down_proj", "forget_gate_proj", "message_gate_proj")
    }
    assert gates <= kept
    assert {
        "self_attn.q_proj.linear",
        "self_attn.o_proj.linear",
        "self_attn.o_proj.gate_proj",
    } <= quantized
    assert not (kept - gates)


def test_fastmodel_keeps_inkling2_eager_unless_told_otherwise():
    # Compiled, Inkling 2 recompiled per sequence length (A100 Mini QLoRA: 85-840 s per step vs 4.4 s eager). The
    # branch must cover both model types, sit in the model-type chain, and leave an explicit user choice alone.
    import unsloth.models.loader as loader_module

    source = pathlib.Path(loader_module.__file__).read_text()
    branch = re.search(r'elif "inkling2" in model_types_all:(.*?)\n        else:\n', source, re.S)
    assert branch
    body = branch.group(1)
    assert re.search(
        r'if os\.environ\.get\("UNSLOTH_COMPILE_DISABLE", "0"\) == "0":\s*\n\s*'
        r'os\.environ\["UNSLOTH_COMPILE_DISABLE"\] = "partial"',
        body,
    )


@pytest.mark.parametrize(
    "flags",
    [
        dict(finetune_vision_layers = False),
        dict(finetune_vision_layers = False, finetune_mlp_modules = False),
        dict(finetune_attention_modules = False),
    ],
)
def test_all_linear_trains_every_family_through_the_remap(monkeypatch, flags):
    # "all-linear" turns every family flag on; the Inkling remap must see those flags, not the caller's, or it
    # returns a narrower list and get_peft_model's own all-linear handling never runs.
    import unsloth  # noqa: F401
    import unsloth.models.vision as vision
    from unsloth.models.inkling2 import patch_inkling2_model

    monkeypatch.setattr(vision, "clean_gpu_cache", lambda: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    model = _tiny(loop = True)
    assert patch_inkling2_model(model)
    model.max_seq_length = 64
    out = vision.FastBaseModel.get_peft_model(
        model, r = 4, target_modules = "all-linear", use_gradient_checkpointing = False, **flags
    )
    wrapped = {n.rsplit(".lora_A", 1)[0] for n, _ in out.named_parameters() if ".lora_A." in n}
    assert any("vision_tower" in n for n in wrapped), sorted(wrapped)[:5]
    assert any(n.endswith("self_attn.q_proj.linear") and "language_model" in n for n in wrapped)
    params = list(out.peft_config["default"].target_parameters or [])
    assert any(p.endswith("experts.gate_up_proj") for p in params), params


@pytest.mark.parametrize("loop", [True, False])
def test_fast_inference_switches_inkling2_to_unsloth_inference(loop):
    # No vLLM release registers Inkling 2 (0.34.0 has Inkling v1 only and pins transformers < 5.23), so
    # fast_inference = True must fall back before a vLLM engine is built, for Mini and for Small.
    from unsloth.models.loader import _vllm_unsupported_model_types
    from unsloth_zoo.hf_utils import get_transformers_model_type

    model_types = get_transformers_model_type(_tiny(loop).config)
    assert _vllm_unsupported_model_types(model_types)
    assert _vllm_unsupported_model_types(["llama"]) == []


@pytest.mark.parametrize("loop", [True, False])
def test_inkling2_export_methods(loop):
    # merged_16bit and GGUF go through save_inkling2_merged_16bit; the generic merges (merged_4bit) re-read the
    # checkpoint under the wrong names and must still refuse. The adapter save is untouched.
    inkling2 = _inkling2()
    model = _tiny(loop)
    for method in ("merged_4bit", "forced_merged_4bit", "mxfp4"):
        with pytest.raises(NotImplementedError, match = "Inkling 2"):
            inkling2.raise_if_merging_inkling2(model, method)
    for method in ("lora", "merged_16bit", "gguf"):
        inkling2.raise_if_merging_inkling2(model, method)
    inkling2.raise_if_merging_inkling2(
        SimpleNamespace(config = SimpleNamespace(model_type = "llama")), "merged_4bit"
    )
    # Each export entry point runs it right after the Mistral-format guard.
    source = (_ROOT / "unsloth" / "save.py").read_text(encoding = "utf-8")
    sites = re.findall(
        r"raise_if_merging_mistral_format_view\((\w+), ([^)]+)\)[^\n]*\n\s*raise_if_merging_inkling2\(\1, \2\)",
        source,
    )
    assert len(sites) == 4, sites
    # Both merged_16bit entry points route Inkling 2 to its own writer before the generic merge.
    assert source.count("save_inkling2_merged_16bit(") == 2


@pytest.mark.parametrize("loop", [True, False])
def test_merged_16bit_matches_the_adapter(loop, tmp_path):
    # Colab A100, real Mini 4-bit + LoRA r 8: the generic merge wrote the released checkpoint back bit for bit
    # (0 of 1244 tensors changed). The Inkling 2 writer must place every LoRA weight, attention and experts, keep the
    # checkpoint's own names, and reload in transformers to the same logits as base + adapter.
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForImageTextToText

    inkling2 = _inkling2()
    _tiny(loop).to(torch.bfloat16).save_pretrained(tmp_path / "base")
    base = AutoModelForImageTextToText.from_pretrained(tmp_path / "base", dtype = torch.float32)
    targets = inkling2.inkling2_lora_target_modules(base, None)
    config = peft.LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = targets,
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    model = peft.get_peft_model(base, config)
    torch.manual_seed(1)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.normal_(0, 0.1)
    ids = torch.randint(0, 250, (1, 12))
    with torch.no_grad():
        reference = model(input_ids = ids).logits
        with model.disable_adapter():
            original = model(input_ids = ids).logits
    inkling2.save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    merged = AutoModelForImageTextToText.from_pretrained(tmp_path / "merged", dtype = torch.float32)
    with torch.no_grad():
        logits = merged(input_ids = ids).logits
    effect = (reference - original).abs().max()
    assert effect > 0.1
    assert (logits - reference).abs().max() < 0.05 * effect
    import json

    names = json.loads((tmp_path / "merged" / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert any(".attn.wq_du." in k for k in names) and not any("language_model" in k for k in names)
    assert "quantization_config" not in (tmp_path / "merged" / "config.json").read_text()


@pytest.mark.parametrize(
    "targets",
    [["o_proj"], ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]],
)
def test_vision_only_training_maps_the_vision_tower(monkeypatch, targets):
    # finetune_language_layers = False must still run the remap for the vision tower (its leaves are out_proj /
    # fc1 / fc2), and must not add language attention or the routed experts.
    import unsloth  # noqa: F401
    import unsloth.models.vision as vision
    from unsloth.models.inkling2 import patch_inkling2_model

    monkeypatch.setattr(vision, "clean_gpu_cache", lambda: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    model = _tiny(loop = True)
    assert patch_inkling2_model(model)
    model.max_seq_length = 64
    out = vision.FastBaseModel.get_peft_model(
        model,
        r = 4,
        target_modules = targets,
        finetune_language_layers = False,
        finetune_vision_layers = True,
        use_gradient_checkpointing = False,
    )
    wrapped = {n.rsplit(".lora_A", 1)[0] for n, _ in out.named_parameters() if ".lora_A." in n}
    assert wrapped and all("vision_tower" in n for n in wrapped), sorted(wrapped)[:5]
    leaves = {n.rsplit(".", 1)[-1] for n in wrapped}
    assert "out_proj" in leaves
    if len(targets) > 1:
        assert {"fc1", "fc2"} <= leaves
    assert not list(out.peft_config["default"].target_parameters or [])


def _export_setup(
    tmp_path,
    loop = True,
    **lora,
):
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForImageTextToText

    _tiny(loop).to(torch.bfloat16).save_pretrained(tmp_path / "base")
    base = AutoModelForImageTextToText.from_pretrained(tmp_path / "base", dtype = torch.float32)
    lora.setdefault("target_modules", _inkling2().inkling2_lora_target_modules(base, None))
    model = peft.get_peft_model(base, peft.LoraConfig(r = 4, lora_alpha = 8, **lora))
    return model


def _checkpoint(path):
    from safetensors.torch import load_file

    out = {}
    for f in sorted(path.glob("*.safetensors")):
        out.update(load_file(f))
    return out


@pytest.mark.parametrize("init", ["pissa", "olora"])
def test_residual_init_export_before_training_is_the_checkpoint(tmp_path, init):
    # PiSSA / OLoRA subtract the initial adapter from the live base. The writer rereads the original checkpoint, so
    # it must add the delta relative to it (zero before training), not the adapter's full contribution.
    import unsloth  # noqa: F401
    from unsloth.models._utils import snapshot_residual_lora_init

    model = _export_setup(tmp_path, init_lora_weights = init)
    snapshot_residual_lora_init(model, init)
    _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    original, merged = _checkpoint(tmp_path / "base"), _checkpoint(tmp_path / "merged")
    assert original.keys() == merged.keys()
    worst = max((merged[k].float() - original[k].float()).abs().max().item() for k in original)
    assert worst == 0.0, worst
    # The live factors are restored after the export.
    assert all(
        m.lora_A["default"].weight.shape[0] == 4
        for m in model.modules()
        if getattr(m, "_unsloth_initial_lora", None) and "default" in getattr(m, "lora_A", {})
    )


def test_embedding_lora_is_merged(tmp_path):
    import copy

    from transformers import AutoModelForImageTextToText

    model = _export_setup(tmp_path, target_modules = ["embed_tokens"])
    torch.manual_seed(1)
    for name, param in model.named_parameters():
        if "lora_embedding" in name:
            param.data.normal_(0, 0.1)
    _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    reference = copy.deepcopy(model).merge_and_unload().get_input_embeddings().weight.detach()
    merged = AutoModelForImageTextToText.from_pretrained(tmp_path / "merged", dtype = torch.float32)
    original = model.get_base_model().get_input_embeddings().base_layer.weight.detach()
    got = merged.get_input_embeddings().weight.detach()
    assert (reference - original).abs().max() > 0.01
    assert torch.allclose(got, reference.to(torch.bfloat16).float(), atol = 1e-2, rtol = 0)
    assert (got - reference).abs().max() < 0.05 * (reference - original).abs().max()


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_trained_biases_are_written(tmp_path, bias):
    from transformers import AutoModelForImageTextToText

    model = _export_setup(
        tmp_path,
        target_modules = r".*vision_tower\.layers\.\d+\.self_attn\.q_proj",
        bias = bias,
    )
    torch.manual_seed(2)
    trained = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias") and param.requires_grad:
            param.data.normal_(0, 1.0)
            trained[name] = param.detach().clone()
    assert trained and (bias == "all") == (len(trained) > 1)
    _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    merged = dict(
        AutoModelForImageTextToText.from_pretrained(
            tmp_path / "merged", dtype = torch.float32
        ).named_parameters()
    )
    for name, value in trained.items():
        key = re.sub(r"^base_model\.model\.", "", name).replace(".base_layer", "")
        assert torch.equal(merged[key], value.to(torch.bfloat16).float()), key


def test_unsupported_lora_exports_still_refuse(tmp_path):
    model = _export_setup(tmp_path, use_dora = True)
    with pytest.raises(NotImplementedError, match = "DoRA"):
        _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    model = _export_setup(tmp_path / "b", modules_to_save = ["lm_head"])
    with pytest.raises(NotImplementedError, match = "modules_to_save"):
        _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged_b"))


def test_trainable_token_rows_are_merged(tmp_path):
    # LoRA `trainable_token_indices` keeps the trained rows in PEFT's TrainableTokensLayer, not in the embedding the
    # writer rereads; the export must carry them (merge_and_unload replaces those rows) instead of the originals.
    import copy

    from transformers import AutoModelForImageTextToText

    rows = [3, 7, 11, 200]
    model = _export_setup(tmp_path, trainable_token_indices = rows)
    torch.manual_seed(3)
    for name, param in model.named_parameters():
        if "trainable_tokens_delta" in name or "lora_" in name:
            param.data.normal_(0, 0.1)
    original = (
        model.get_base_model()
        .get_input_embeddings()
        .token_adapter.base_layer.weight.detach()
        .clone()
    )
    reference = copy.deepcopy(model).merge_and_unload().get_input_embeddings().weight.detach()
    _inkling2().save_inkling2_merged_16bit(model, None, str(tmp_path / "merged"))
    got = (
        AutoModelForImageTextToText.from_pretrained(tmp_path / "merged", dtype = torch.float32)
        .get_input_embeddings()
        .weight.detach()
    )
    assert (reference[rows] - original[rows]).abs().max() > 0.01
    assert torch.equal(got, reference.to(torch.bfloat16).float())
    others = [i for i in range(got.shape[0]) if i not in rows]
    assert torch.equal(got[others], original[others])


def test_float32_export_writes_float32_merge(tmp_path):
    # save_pretrained_merged(dtype = torch.float32) documents float32 safetensors with the merged delta unrounded;
    # the Inkling 2 writer must honour it (default stays bf16) and refuse dtypes it does not write.
    import copy

    from transformers import AutoModelForImageTextToText

    model = _export_setup(
        tmp_path, target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    )
    torch.manual_seed(4)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.normal_(0, 0.1)
    reference = dict(copy.deepcopy(model).merge_and_unload().named_parameters())
    ink = _inkling2()
    ink.save_inkling2_merged_16bit(model, None, str(tmp_path / "fp32"), dtype = torch.float32)
    ink.save_inkling2_merged_16bit(model, None, str(tmp_path / "default"))
    fp32, default = _checkpoint(tmp_path / "fp32"), _checkpoint(tmp_path / "default")
    assert fp32.keys() == default.keys()
    assert all(v.dtype == torch.float32 for v in fp32.values() if v.is_floating_point())
    assert all(v.dtype == torch.bfloat16 for v in default.values() if v.is_floating_point())
    merged = dict(
        AutoModelForImageTextToText.from_pretrained(
            tmp_path / "fp32", dtype = torch.float32
        ).named_parameters()
    )
    assert merged.keys() == reference.keys()
    worst = max((merged[k] - reference[k].detach()).abs().max().item() for k in reference)
    rounding = max(
        (reference[k].detach().to(torch.bfloat16).float() - reference[k].detach())
        .abs()
        .max()
        .item()
        for k in reference
    )
    # Rounding the merge to bf16 (the old behaviour) would cost `rounding`; float32 matches the fp32 merge.
    assert rounding > 1e-4 and worst < 1e-6, (worst, rounding)
    with pytest.raises(NotImplementedError, match = "dtype"):
        ink.save_inkling2_merged_16bit(model, None, str(tmp_path / "f16"), dtype = torch.float16)


def test_save_pretrained_merged_forwards_dtype_to_inkling2():
    source = (_ROOT / "unsloth" / "save.py").read_text()
    call = source[source.index("        save_inkling2_merged_16bit(\n") :]
    call = call[: call.index(")")]
    assert "dtype = dtype" in call
