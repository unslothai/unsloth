# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ModelOpt NVFP4 checkpoints (nvidia/Qwen3-30B-A3B-NVFP4 layout) load through the compressed-tensors NVFP4A16
rewrite: routed experts packed as Nvfp4ExpertParam (multiply mode, exact), 2-D Linears as weight_packed with an
fp32 global scale 1 / weight_scale_2, only the activation / kv scales dropped, and a baseline that still fails."""

import json
import os

import pytest
import torch

import unsloth  # noqa: F401

transformers = pytest.importorskip("transformers")
pytest.importorskip("compressed_tensors")
pe = pytest.importorskip("unsloth_zoo.packed_experts")
if not hasattr(pe, "Nvfp4ExpertParam"):
    pytest.skip("needs unsloth_zoo with packed expert stacks", allow_module_level = True)
TF5 = int(transformers.__version__.split(".")[0]) >= 5
fused = pytest.mark.skipif(not TF5, reason = "fused expert stacks are transformers 5")

from unsloth.models import modelopt_nvfp4 as mo
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading

E, H, I, L = 4, 64, 32, 2
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LUT = torch.cat([E2M1, -E2M1])


def _config():
    return transformers.Qwen3MoeConfig(
        vocab_size = 96,
        hidden_size = H,
        intermediate_size = 128,
        moe_intermediate_size = I,
        num_experts = E,
        num_experts_per_tok = 2,
        num_hidden_layers = L,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
        tie_word_embeddings = False,
    )


def _codes(x):
    a = x.abs().clamp(max = 6.0)
    mids = (E2M1[1:] + E2M1[:-1]) / 2
    idx = torch.bucketize(a, mids)
    tie = (idx < 7) & (a == mids[idx.clamp(max = 6)]) & (idx % 2 == 1)
    idx = idx + tie.to(idx.dtype)
    return (idx | ((x < 0).to(idx.dtype) << 3)).to(torch.uint8)


def _quantize(w, ws2):
    """ModelOpt NVFP4: value = e2m1 * weight_scale * weight_scale_2."""
    o, i = w.shape
    wb = w.float().view(o, i // 16, 16)
    bs = (wb.abs().amax(-1) / 6.0 / ws2).clamp(max = 448.0).to(torch.float8_e4m3fn)
    eff = bs.float() * ws2
    eff = torch.where(eff == 0, torch.ones_like(eff), eff)
    c = _codes((wb / eff.unsqueeze(-1)).view(o, i))
    return {"weight": (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous(), "weight_scale": bs}


def dequant(weight, weight_scale, weight_scale_2):
    lo, hi = weight & 0xF, weight >> 4
    v = LUT[torch.stack([lo, hi], -1).flatten(-2).long()]
    o, n = v.shape
    s = weight_scale.float() * weight_scale_2.float().reshape(())
    return (v.view(o, n // 16, 16) * s.unsqueeze(-1)).view(o, n)


def _ws2(ws):
    return (torch.stack([w.float().abs().amax() for w in ws]).amax() / (6 * 448)).to(torch.float32)


QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"],
        }
    },
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.34.1"},
    "quant_method": "modelopt",
}


def _write_checkpoint(path, seed = 0):
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    model = transformers.Qwen3MoeForCausalLM(_config()).to(torch.bfloat16)
    sd = model.state_dict()
    state, ref = {}, {}
    ignore = [f"model.layers.{l}.mlp.gate" for l in range(L)] + ["lm_head"]

    def put(prefix, ws_group):
        ws2 = _ws2([w for _, w in ws_group])
        for name, w in ws_group:
            q = _quantize(w, ws2)
            state[f"{prefix}.{name}.weight"] = q["weight"]
            state[f"{prefix}.{name}.weight_scale"] = q["weight_scale"]
            state[f"{prefix}.{name}.weight_scale_2"] = ws2.clone()
            state[f"{prefix}.{name}.input_scale"] = torch.tensor(0.01)
            ref[f"{prefix}.{name}"] = dequant(q["weight"], q["weight_scale"], ws2)

    for key, value in sd.items():
        if ".mlp.experts." in key or ".self_attn." in key and key.endswith("proj.weight"):
            continue
        state[key] = value.contiguous()
    for l in range(L):
        p = f"model.layers.{l}"
        attn = model.model.layers[l].self_attn
        put(
            f"{p}.self_attn",
            [(n, getattr(attn, n).weight.data) for n in ("q_proj", "k_proj", "v_proj")],
        )
        put(f"{p}.self_attn", [("o_proj", attn.o_proj.weight.data)])
        state[f"{p}.self_attn.k_proj.k_scale"] = torch.tensor(1.0)
        state[f"{p}.self_attn.v_proj.v_scale"] = torch.tensor(1.0)
        ex = model.model.layers[l].mlp.experts
        for e in range(E):
            if TF5:
                gate, up, down = (
                    ex.gate_up_proj.data[e, :I],
                    ex.gate_up_proj.data[e, I:],
                    ex.down_proj.data[e],
                )
            else:
                gate, up, down = (
                    getattr(ex[e], n).weight.data for n in ("gate_proj", "up_proj", "down_proj")
                )
            put(f"{p}.mlp.experts.{e}", [("gate_proj", gate), ("up_proj", up)])
            put(f"{p}.mlp.experts.{e}", [("down_proj", down)])
    os.makedirs(path, exist_ok = True)
    save_file(state, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    cfg = model.config.to_dict()
    cfg["quantization_config"] = dict(QUANT_CONFIG, ignore = ignore)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(path, "hf_quant_config.json"), "w") as f:
        json.dump(
            {
                "producer": QUANT_CONFIG["producer"],
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": ignore,
                },
            },
            f,
        )
    return ref


@pytest.fixture(scope = "module")
def ckpt(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("modelopt_nvfp4"))
    return path, _write_checkpoint(path)


def _load(path, arm):
    config = transformers.AutoConfig.from_pretrained(path)
    if arm == "rewrite":
        assert mo.arm_modelopt_nvfp4_loading(config) is not None
    elif arm == "unsloth":
        four, eight, method = check_and_disable_bitsandbytes_loading(
            config, load_in_4bit = True, model_name = path
        )
        assert (four, eight, method) == (False, False, "compressed-tensors")
        assert config.quantization_config["format"] == "nvfp4-pack-quantized"
    return transformers.AutoModelForCausalLM.from_pretrained(
        path, config = config, dtype = torch.bfloat16, output_loading_info = True
    )


def _check(model, info, ref):
    assert not info["missing_keys"], info["missing_keys"]
    assert not info["unexpected_keys"], info["unexpected_keys"]
    assert not info.get("mismatched_keys"), info["mismatched_keys"]
    worst, bf16_diff, total = 0.0, 0, 0
    for l in range(L):
        layer = model.model.layers[l]
        ex = layer.mlp.experts
        linears = {
            f"self_attn.{n}": getattr(layer.self_attn, n)
            for n in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        if not TF5:
            # transformers 4 keeps per-expert Linears: they take the same compressed-tensors route.
            for e in range(E):
                for n in ("gate_proj", "up_proj", "down_proj"):
                    linears[f"mlp.experts.{e}.{n}"] = getattr(ex[e], n)
        for stack in ("gate_up_proj", "down_proj") if TF5 else ():
            p = ex._parameters[stack]
            assert isinstance(p, pe.Nvfp4ExpertParam) and p.packed_global_mode == "multiply"
            got = p.dequantize(torch.float32)
            for e in range(E):
                names = ("gate_proj", "up_proj") if stack == "gate_up_proj" else ("down_proj",)
                want = torch.cat([ref[f"model.layers.{l}.mlp.experts.{e}.{n}"] for n in names])
                assert torch.equal(got[e], want), (l, stack, e)
        for n, lin in linears.items():
            assert "weight" not in lin._parameters
            assert lin.weight_packed.dtype == torch.uint8
            gs = lin.weight_global_scale
            assert gs.dtype == torch.float32 and gs.shape == (1,)
            want = ref[f"model.layers.{l}.{n}"]
            from unsloth.kernels.nvfp4 import nvfp4_dequantize

            # compressed-tensors decode: e2m1 * (scale / (1 / ws2)), a few fp32 ulps from e2m1 * scale * ws2.
            dec = nvfp4_dequantize(lin.weight_packed, lin.weight_scale, gs, torch.float32)
            rel = ((dec - want).abs() / want.abs().clamp_min(1e-30)).max().item()
            worst = max(worst, rel)
            bf16_diff += int((dec.to(torch.bfloat16) != want.to(torch.bfloat16)).sum())
            total += want.numel()
            assert rel <= 4.8e-7, (l, n, rel)
        gate = layer.mlp.gate
        assert gate.weight.dtype == torch.bfloat16 and not hasattr(gate, "weight_packed")
    assert type(model.lm_head) is torch.nn.Linear and model.lm_head.weight.dtype == torch.bfloat16
    print(f"2-D Linears: max rel diff {worst:.3g}, bf16 values differing {bf16_diff}/{total}")
    return worst


@pytest.mark.parametrize("arm", ["rewrite", "unsloth"])
def test_modelopt_nvfp4_loads_packed(ckpt, arm):
    path, ref = ckpt
    model, info = _load(path, arm)
    _check(model, info, ref)
    assert getattr(model.config, mo.MODELOPT_NVFP4_ATTR, None) is None
    out = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(out).all()


def test_modelopt_nvfp4_reload_of_reused_config(ckpt):
    path, ref = ckpt
    config = transformers.AutoConfig.from_pretrained(path)
    mo.arm_modelopt_nvfp4_loading(config)
    for _ in range(2):
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            path, config = config, dtype = torch.bfloat16, output_loading_info = True
        )
        _check(model, info, ref)


@fused
def test_modelopt_nvfp4_16bit_experts_kill_switch(ckpt, monkeypatch):
    path, ref = ckpt
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS", "0")
    model, info = _load(path, "rewrite")
    assert not info["missing_keys"] and not info["unexpected_keys"]
    p = model.model.layers[0].mlp.experts.gate_up_proj
    assert not isinstance(p, pe.Nvfp4ExpertParam) and p.dtype == torch.bfloat16
    want = torch.cat(
        [ref["model.layers.0.mlp.experts.0.gate_proj"], ref["model.layers.0.mlp.experts.0.up_proj"]]
    )
    assert torch.equal(p[0], want.to(torch.bfloat16))


def test_plan_shape(ckpt):
    path, _ = ckpt
    config = transformers.AutoConfig.from_pretrained(path)
    plan = mo.modelopt_nvfp4_plan(config)
    assert plan["format"] == "nvfp4-pack-quantized" and plan["kv_cache_scheme"] is None
    group = plan["config_groups"]["group_0"]
    assert group["input_activations"] is None and group["weights"]["strategy"] == "tensor_group"
    assert "lm_head" in plan["ignore"] and "model.layers.0.mlp.gate" in plan["ignore"]
    glob_plan = mo.modelopt_nvfp4_plan(
        type(
            "C", (), {"quantization_config": dict(QUANT_CONFIG, ignore = ["*mlp.gate", "lm_head"])}
        )()
    )
    assert any(e.startswith("re:") for e in glob_plan["ignore"])
    fp8 = type("C", (), {"quantization_config": dict(QUANT_CONFIG, quant_algo = "FP8")})()
    assert mo.modelopt_nvfp4_plan(fp8) is None


def test_baseline_without_hook_fails(ckpt):
    path, _ = ckpt
    with pytest.raises(Exception, match = "size mismatch|mismatch"):
        transformers.AutoModelForCausalLM.from_pretrained(path, dtype = torch.bfloat16)


def test_merged_save_is_refused(ckpt):
    path, _ = ckpt
    import unsloth_zoo.saving_utils as zoo_saving

    mo.refuse_modelopt_nvfp4_merged_save()
    with pytest.raises(RuntimeError, match = "ModelOpt NVFP4"):
        zoo_saving.determine_base_model_source(path, None, "merged_16bit")


@fused
def test_save_pretrained_writes_a_reloadable_checkpoint(ckpt, tmp_path):
    path, _ = ckpt
    model, _ = _load(path, "rewrite")
    model.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        quant = json.load(f)["quantization_config"]
    assert (
        quant["quant_method"] == "compressed-tensors" and quant["format"] == "nvfp4-pack-quantized"
    )
    reloaded, info = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype = torch.bfloat16, output_loading_info = True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    a, b = model.model.layers[0].self_attn.q_proj, reloaded.model.layers[0].self_attn.q_proj
    for name in ("weight_packed", "weight_scale", "weight_global_scale"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    x = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(model(x).logits, reloaded(x).logits)


def test_rewritten_config_reaches_the_fast_language_model_load(ckpt):
    # FastLlamaModel hands model_config to from_pretrained only when modelopt_rewritten says so; without it
    # transformers rereads the raw ModelOpt config and the NVFP4 checkpoint fails to load.
    from unsloth.models.modelopt_fp8 import modelopt_rewritten

    config = transformers.AutoConfig.from_pretrained(ckpt[0])
    assert not modelopt_rewritten(config)
    check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, model_name = ckpt[0])
    assert modelopt_rewritten(config)
