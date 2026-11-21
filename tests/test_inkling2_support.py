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
