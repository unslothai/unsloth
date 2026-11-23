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


def test_merged_export_replaces_an_earlier_save(tmp_path):
    # transformers loads model.safetensors ahead of an index, so weights from an earlier unsharded save
    # left in the directory would silently shadow the new merged shards on reload.
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText

    model = _export_setup(tmp_path, target_parameters = ["mlp.experts.gate_up_proj"])
    torch.manual_seed(5)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.normal_(0, 0.1)
    ink = _inkling2()
    ink.save_inkling2_merged_16bit(model, None, str(tmp_path / "clean"))
    fresh = _checkpoint(tmp_path / "clean")

    stale = tmp_path / "stale"
    stale.mkdir()
    old = {k: torch.zeros_like(v) for k, v in fresh.items()}
    save_file(old, str(stale / "model.safetensors"), metadata = {"format": "pt"})
    save_file(old, str(stale / "model-00007-of-00009.safetensors"), metadata = {"format": "pt"})
    torch.save(old, stale / "pytorch_model.bin")
    (stale / "pytorch_model.bin.index.json").write_text('{"metadata": {}, "weight_map": {}}')
    (stale / "notes.txt").write_text("keep")
    ink.save_inkling2_merged_16bit(model, None, str(stale))

    assert not (stale / "model.safetensors").exists()
    assert not (stale / "model-00007-of-00009.safetensors").exists()
    assert not (stale / "pytorch_model.bin").exists()
    assert not (stale / "pytorch_model.bin.index.json").exists()
    assert (stale / "notes.txt").read_text() == "keep"
    reloaded = dict(AutoModelForImageTextToText.from_pretrained(stale).named_parameters())
    expected = dict(
        AutoModelForImageTextToText.from_pretrained(tmp_path / "clean").named_parameters()
    )
    assert reloaded.keys() == expected.keys()
    assert any(v.abs().max() > 0 for v in expected.values())
    assert all(torch.equal(reloaded[k], expected[k]) for k in expected)


def test_merged_export_removes_an_earlier_lora_save(tmp_path):
    # transformers auto-loads adapter_config.json found next to the weights, so a LoRA save left in
    # the directory would be applied a second time on top of the merged export.
    from transformers import AutoModelForImageTextToText

    model = _export_setup(tmp_path, target_parameters = ["mlp.experts.gate_up_proj"])
    torch.manual_seed(6)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.normal_(0, 0.1)
    ink = _inkling2()
    ink.save_inkling2_merged_16bit(model, None, str(tmp_path / "clean"))

    stale = tmp_path / "stale"
    model.save_pretrained(str(stale))
    assert (stale / "adapter_config.json").exists()
    assert (stale / "adapter_model.safetensors").exists()
    torch.save({}, stale / "adapter_model.bin")
    ink.save_inkling2_merged_16bit(model, None, str(stale))

    assert not (stale / "adapter_config.json").exists()
    assert not (stale / "adapter_model.safetensors").exists()
    assert not (stale / "adapter_model.bin").exists()
    loaded = AutoModelForImageTextToText.from_pretrained(stale)
    assert not getattr(loaded, "_hf_peft_config_loaded", False)
    assert not any("lora_" in n for n, _ in loaded.named_parameters())
    reloaded = dict(loaded.named_parameters())
    expected = dict(
        AutoModelForImageTextToText.from_pretrained(tmp_path / "clean").named_parameters()
    )
    assert reloaded.keys() == expected.keys()
    assert all(torch.equal(reloaded[k], expected[k]) for k in expected)


def _saved_bytes_and_grads(model, ids):
    model.zero_grad(set_to_none = True)
    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(out.loss), saved[0], grads


def test_loop_message_input_is_checkpointed_under_gradient_checkpointing(monkeypatch):
    """The loop message residual runs between the checkpointed layers: without its own checkpoint everything its
    fp32 attention saves stays on the GPU for the whole step (1.4 GB on Inkling-2-Mini at 1024 tokens)."""
    m = _inkling2()
    model = _tiny(loop = True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    ids = torch.randint(0, 250, (2, 16))
    loss0, saved0, grads0 = _saved_bytes_and_grads(model, ids)
    assert m.checkpoint_inkling2_message_inputs(model) == 1
    assert m.checkpoint_inkling2_message_inputs(model) == 0  # idempotent
    loss1, saved1, grads1 = _saved_bytes_and_grads(model, ids)
    assert loss1 == loss0
    assert grads1.keys() == grads0.keys()
    for n in grads0:
        assert torch.equal(grads0[n], grads1[n]), n
    # 2 passes x 6 loop layers -> 6 message calls; each saved its fp32 norm, queries, keys and values before.
    assert saved1 < saved0
    # The opt-out restores the stock residual exactly.
    monkeypatch.setenv("UNSLOTH_INKLING2_MESSAGE_CHECKPOINT", "0")
    loss2, saved2, grads2 = _saved_bytes_and_grads(model, ids)
    assert loss2 == loss0 and saved2 == saved0
    for n in grads0:
        assert torch.equal(grads0[n], grads2[n]), n
    monkeypatch.delenv("UNSLOTH_INKLING2_MESSAGE_CHECKPOINT")
    # Without gradient checkpointing (or in eval) the wrapper is a pass-through.
    model.gradient_checkpointing_disable()
    _, saved_off, grads_off = _saved_bytes_and_grads(model, ids)
    for n in grads0:
        torch.testing.assert_close(grads_off[n], grads0[n])
    assert saved_off > saved0


def test_message_input_checkpoint_skips_inkling2_small():
    m = _inkling2()
    assert m.checkpoint_inkling2_message_inputs(_tiny(loop = False)) == 0


def _message_grad(model, ids):
    # autograd.grad (not .backward) on the hidden states the loop message residual reads.
    out = model.model.language_model(input_ids = ids, use_cache = False).last_hidden_state
    params = [p for n, p in model.named_parameters() if "message_input.query" in n]
    return torch.autograd.grad(out.square().sum(), params)


def test_message_checkpoint_follows_the_models_reentrant_setting():
    # transformers 5.24 checkpoints the layers non-reentrant by default, where autograd.grad works; a reentrant message
    # checkpoint inside that forward raised "torch.utils.checkpoint is incompatible with .grad()".
    m = _inkling2()
    model = _tiny(loop = True)
    model.train()
    ids = torch.randint(0, 250, (2, 16), generator = torch.Generator().manual_seed(0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    want = _message_grad(model, ids)
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return original(*args, **kwargs)

    import unittest.mock

    # The wrapper binds checkpoint when it is installed; the layers keep the one transformers bound at enable.
    with unittest.mock.patch("torch.utils.checkpoint.checkpoint", spy):
        assert m.checkpoint_inkling2_message_inputs(model) == 1
    got = _message_grad(model, ids)
    assert (
        calls.count(False) >= 6 and True not in calls
    ), calls  # the 6 message calls took the checkpoint
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert m._checkpoint_use_reentrant(model.model.language_model) is False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    assert m._checkpoint_use_reentrant(model.model.language_model) is True


def test_message_checkpoint_keeps_accelerate_offload_hook():
    # accelerate CPU offload keeps the residual's query on meta until the hook's pre_forward loads it: the checkpoint
    # path computed on meta. An offloaded module keeps its hooked forward.
    accelerate = pytest.importorskip("accelerate")
    m = _inkling2()
    model = _tiny(loop = True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    ids = torch.randint(0, 250, (2, 16), generator = torch.Generator().manual_seed(0))
    out = model(input_ids = ids, labels = ids, use_cache = False)
    message_input = model.model.language_model.message_input
    accelerate.cpu_offload(message_input, execution_device = "cpu")
    assert message_input.query.device.type == "meta"
    assert m.checkpoint_inkling2_message_inputs(model) == 1
    got = model(input_ids = ids, labels = ids, use_cache = False)
    assert got.logits.device.type == "cpu"
    assert torch.equal(got.logits, out.logits)
    got.loss.backward()
    assert m._weights_offloaded(message_input)
    # Not offloaded: the checkpoint path engages.
    assert not m._weights_offloaded(_tiny(loop = True).model.language_model.message_input)


def _residual_grads(model, monkeypatch, enabled):
    monkeypatch.setenv("UNSLOTH_INKLING2_MESSAGE_CHECKPOINT", "1" if enabled else "0")
    owner = model.model.language_model
    hidden = owner.config.hidden_size
    gen = torch.Generator().manual_seed(1)
    states = torch.randn(1, 2, 3, hidden, generator = gen, requires_grad = True)
    partial = torch.randn(1, 2, hidden, generator = gen, requires_grad = True)
    query = owner.message_input.query
    out = owner.message_input(states, partial, 1)
    grads = torch.autograd.grad(out.square().sum(), (states, partial, query))
    return out.detach(), grads


def test_message_checkpoint_calls_the_residuals_norm_module(monkeypatch):
    # transformers calls `self.norm(stacked_states)`: a forward hook on that norm (or a replaced norm) must apply on the
    # checkpointed path too, on the original forward and on the recomputation. An inline RMSNorm skipped it.
    m = _inkling2()
    model = _tiny(loop = True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    owner = model.model.language_model
    with torch.no_grad():
        owner.message_input.query.normal_(generator = torch.Generator().manual_seed(2))
    calls = []

    def scale(_module, _args, output):
        calls.append(1)
        return output * 0.5

    owner.message_input.norm.register_forward_hook(scale)
    assert m.checkpoint_inkling2_message_inputs(model) == 1
    want, want_grads = _residual_grads(model, monkeypatch, enabled = False)
    assert len(calls) == 1
    calls.clear()
    recomputed = []
    math = m._message_input_math
    monkeypatch.setattr(m, "_message_input_math", lambda *a: recomputed.append(1) or math(*a))
    got, got_grads = _residual_grads(model, monkeypatch, enabled = True)
    assert recomputed == [1, 1]  # the checkpoint path ran (forward + recompute)
    assert len(calls) == 2  # the norm hook fired on both
    assert torch.equal(got, want)
    for a, b in zip(got_grads, want_grads):
        assert torch.equal(a, b)


def test_message_checkpoint_inlines_only_the_stock_norm(monkeypatch):
    # The stock unweighted norm stays plain tensor ops in the recompute (a module call there gave run-to-run different
    # gradients on a model split over two GPUs); a hooked or replaced norm is called as a module.
    m = _inkling2()
    model = _tiny(loop = True)
    norm = model.model.language_model.message_input.norm
    assert m._message_norm(norm) == float(norm.eps)
    handle = norm.register_forward_hook(lambda *_: None)
    assert m._message_norm(norm) is norm
    handle.remove()
    monkeypatch.setattr(norm, "forward", lambda x: x)
    assert m._message_norm(norm) is norm
    monkeypatch.delattr(norm, "forward")
    assert m._message_norm(norm) == float(norm.eps)
    # The inlined eps and the module call give the same values bit for bit.
    x = torch.randn(2, 3, 4, 8, generator = torch.Generator().manual_seed(0))
    q = torch.randn(1, 3, 1, 8, generator = torch.Generator().manual_seed(1))
    args = (x[..., :3, :], x[..., 3, :], 0)
    assert torch.equal(
        m._message_input_math(q, float(norm.eps), 8**-0.5, *args),
        m._message_input_math(q, norm, 8**-0.5, *args),
    )


def test_message_checkpoint_follows_a_deepcopy_and_pickles(monkeypatch):
    # The installed forward captured the original residual: a deepcopied model read the original's query.
    import copy
    import pickle
    import sys

    m = _inkling2()
    monkeypatch.setitem(sys.modules, m.__name__, m)
    model = _tiny(loop = True)
    model.train()
    assert m.checkpoint_inkling2_message_inputs(model) == 1
    # Pickled before enabling gradient checkpointing: its input-grad hook is a local function that never pickles.
    pickled = pickle.loads(pickle.dumps(model))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    pickled.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    for clone in (copy.deepcopy(model), pickled):
        owner = clone.model.language_model
        assert owner.message_input.forward._module() is owner.message_input
        assert owner.message_input.forward._owner() is owner
        with torch.no_grad():
            owner.message_input.query.normal_(generator = torch.Generator().manual_seed(3))
        want, want_grads = _residual_grads(clone, monkeypatch, enabled = False)
        got, got_grads = _residual_grads(clone, monkeypatch, enabled = True)
        assert torch.equal(got, want)
        for a, b in zip(got_grads, want_grads):
            assert torch.equal(a, b)
        original, _ = _residual_grads(model, monkeypatch, enabled = True)
        assert not torch.equal(got, original)


@pytest.mark.parametrize("public", [True, False])
def test_private_push_refuses_an_existing_public_repo(tmp_path, monkeypatch, public):
    # create_repo(private = True) leaves an existing public repo public, so the merged Inkling 2 upload must refuse
    # before merging or uploading anything, as the staged Hub export does.
    import types
    import unittest.mock

    import huggingface_hub

    model = _export_setup(tmp_path, target_parameters = ["mlp.experts.gate_up_proj"])
    api = unittest.mock.MagicMock()
    api.create_repo.return_value = types.SimpleNamespace(repo_id = "someone/inkling-merged")
    api.repo_info.return_value = types.SimpleNamespace(private = not public)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)
    save = lambda: _inkling2().save_inkling2_merged_16bit(
        model, None, "inkling-merged", push_to_hub = True, token = "hf_x", private = True
    )
    if public:
        with pytest.raises(RuntimeError, match = "already exists and is public"):
            save()
        api.upload_folder.assert_not_called()
    else:
        save()
        api.upload_folder.assert_called_once()
        assert api.upload_folder.call_args.kwargs["repo_id"] == "someone/inkling-merged"
