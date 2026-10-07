# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Tiny compressed-tensors NVFP4 + FP8 mixed-precision checkpoints, laid out like unsloth/Qwen3.8-27B-NVFP4.

MLPs are NVFP4 (weight_packed U8, weight_scale F8_E4M3 per 16 columns, weight_global_scale and input_global_scale F32[1]),
attention and the last layer's MLP are FP8 per channel (weight F8_E4M3, weight_scale BF16 [out, 1]). Built without
llmcompressor: scales are computed here and the tensors come from compressed-tensors' own compressors.

    python tests/_nvfp4_fixtures.py OUT_DIR [--arch qwen3|qwen3_5]
"""

import argparse
import copy
import json
import os
import re
import sys

import torch

QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
QWEN3_5 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"

_FP8_ACT = {
    "actorder": None,
    "block_structure": None,
    "dynamic": True,
    "group_size": None,
    "num_bits": 8,
    "observer": None,
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "token",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
_FP8_W = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": None,
    "num_bits": 8,
    "observer": "memoryless_minmax",
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "channel",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
_FP4 = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": 16,
    "num_bits": 4,
    "observer_kwargs": {},
    "scale_dtype": "torch.float8_e4m3fn",
    "strategy": "tensor_group",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}


def _quant_config(
    num_layers,
    lm_head,
    attn_targets,
    ignore,
    lm_head_kind = "fp8",
):
    last = str(num_layers - 1)
    fp8_targets = list(attn_targets) + [r"re:.*layers\.(" + last + r")\.mlp\.(gate|up|down)_proj$"]
    fp4_targets = [r"re:.*mlp\.(gate|up|down)_proj$"]
    if lm_head:
        (fp8_targets if lm_head_kind == "fp8" else fp4_targets).append("re:.*lm_head")
    fp4_act = dict(_FP4, dynamic = "local", observer = "static_minmax")
    return {
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "input_activations": _FP8_ACT,
                "output_activations": None,
                "targets": fp8_targets,
                "weights": _FP8_W,
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "input_activations": fp4_act,
                "output_activations": None,
                "targets": fp4_targets,
                "weights": dict(_FP4, actorder = "static"),
            },
        },
        "format": "mixed-precision",
        "global_compression_ratio": None,
        "ignore": list(ignore),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.17.0",
    }


def _matches(name, targets):
    return any(re.match(t[3:], name) if t.startswith("re:") else name == t for t in targets)


def nvfp4_tensors(w):
    """weight_packed, weight_scale (fp8), weight_global_scale for w (out, in), via compressed-tensors' compressor."""
    from compressed_tensors.compressors import NVFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    w = w.float()
    out_f, in_f = w.shape
    assert in_f % 16 == 0, in_f
    global_scale = (448.0 * 6.0 / w.abs().amax().clamp(min = 1e-12)).reshape(1).float()
    group_amax = w.reshape(out_f, in_f // 16, 16).abs().amax(-1)
    # Rounded to fp8 values but kept float: the compressor divides by the global scale, then stores fp8.
    scale = (global_scale * (group_amax / 6.0)).clamp(max = 448.0).to(torch.float8_e4m3fn).float()
    scheme = QuantizationScheme(
        targets = ["Linear"],
        weights = QuantizationArgs(
            **{k: v for k, v in _FP4.items() if v is not None and k not in ("observer_kwargs",)}
        ),
    )
    out = NVFP4PackedCompressor.compress(
        {"weight": w, "weight_scale": scale, "weight_global_scale": global_scale}, scheme
    )
    return out["weight_packed"], out["weight_scale"], global_scale


def fp8_channel_tensors(w, scale_dtype = torch.bfloat16):
    w = w.float()
    scale = (w.abs().amax(1, keepdim = True) / 448.0).clamp(min = 1e-12)
    q = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale.to(scale_dtype)


def dequant_nvfp4_reference(
    packed,
    scale,
    global_scale,
    dtype = torch.bfloat16,
):
    from compressed_tensors.compressors import NVFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    scheme = QuantizationScheme(
        targets = ["Linear"],
        weights = QuantizationArgs(
            **{k: v for k, v in _FP4.items() if v is not None and k not in ("observer_kwargs",)}
        ),
    )
    out = NVFP4PackedCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": global_scale},
        scheme,
    )
    return out["weight"].to(dtype)


def build(
    out_dir,
    arch = "qwen3",
    seed = 3407,
    lm_head_kind = "fp8",
):
    """Write a tiny mixed NVFP4/FP8 checkpoint to out_dir; returns {name: kind} for every quantized Linear."""
    from safetensors.torch import save_file
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    torch.manual_seed(seed)
    if arch == "qwen3":
        repo = QWEN3
        config = AutoConfig.from_pretrained(repo)
        config.update(
            dict(
                hidden_size = 64,
                intermediate_size = 128,
                head_dim = 16,
                num_attention_heads = 4,
                num_key_value_heads = 2,
                num_hidden_layers = 2,
            )
        )
        config.torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_config(config, dtype = torch.bfloat16)
        attn = [r"re:.*self_attn\.(q|k|v|o)_proj$"]
        ignore = ["model.layers.0.self_attn.o_proj"]
        lm_head = True
    elif arch == "qwen3_5":
        repo = QWEN3_5
        config = AutoConfig.from_pretrained(repo)
        config.torch_dtype = torch.bfloat16
        model = AutoModelForImageTextToText.from_config(config, dtype = torch.bfloat16)
        attn = [
            r"re:.*self_attn\.(q|k|v|o)_proj$",
            r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
        ]
        ignore = [
            n
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and (".visual." in f".{n}" or n.endswith(("in_proj_a", "in_proj_b")))
        ]
        lm_head = False
    else:
        raise ValueError(arch)

    num_layers = config.get_text_config().num_hidden_layers
    qcfg = _quant_config(num_layers, lm_head, attn, ignore, lm_head_kind)
    fp8_targets = qcfg["config_groups"]["group_0"]["targets"]
    fp4_targets = qcfg["config_groups"]["group_1"]["targets"]

    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    kinds = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name in ignore:
            continue
        w = state.pop(name + ".weight", None)
        if w is None:
            continue
        # The FP8 override wins over the broad MLP NVFP4 target, as in the real checkpoint.
        if _matches(name, fp8_targets):
            q, s = fp8_channel_tensors(w)
            state[name + ".weight"] = q
            state[name + ".weight_scale"] = s
            kinds[name] = "fp8"
        elif _matches(name, fp4_targets):
            packed, scale, gscale = nvfp4_tensors(w)
            state[name + ".weight_packed"] = packed
            state[name + ".weight_scale"] = scale
            state[name + ".weight_global_scale"] = gscale
            state[name + ".input_global_scale"] = torch.tensor(
                [448.0 * 6.0 / 4.0], dtype = torch.float32
            )
            kinds[name] = "nvfp4"
        else:
            state[name + ".weight"] = w
    if getattr(config, "tie_word_embeddings", False) or getattr(
        config.get_text_config(), "tie_word_embeddings", False
    ):
        state = {k: v for k, v in state.items() if not k.endswith("lm_head.weight")}

    os.makedirs(out_dir, exist_ok = True)
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata = {"format": "pt"})
    cfg = copy.deepcopy(config.to_dict())
    cfg["quantization_config"] = qcfg
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(out_dir, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(cfg, f, indent = 2)
    AutoTokenizer.from_pretrained(repo).save_pretrained(out_dir)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(repo).save_pretrained(out_dir)
    except Exception:
        pass
    with open(os.path.join(out_dir, "nvfp4_fixture_kinds.json"), "w", encoding = "utf-8") as f:
        json.dump(kinds, f, indent = 1)
    return kinds


def run_lora_case(path, arch, api, out_dir):
    """Load with Fast{LanguageModel,Model}, LoRA, 5 steps, save, reload; returns a JSON-able report. Run in a fresh process."""
    dtype = getattr(torch, os.environ.get("NVFP4_CASE_DTYPE", "bfloat16"))
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    import unsloth
    import unsloth.kernels.nvfp4 as nv
    from peft import PeftModel

    with open(os.path.join(path, "nvfp4_fixture_kinds.json"), encoding = "utf-8") as f:
        kinds = json.load(f)
    name = next(n for n, k in kinds.items() if k == "nvfp4")
    seen = []
    real = nv.nvfp4_linear

    def recording(X, *args, **kwargs):
        seen.append(str(X.dtype))
        return real(X, *args, **kwargs)

    nv.nvfp4_linear = recording
    Fast = getattr(unsloth, api)
    model, _ = Fast.from_pretrained(path, max_seq_length = 64, load_in_4bit = False, dtype = dtype)
    model = Fast.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    base = next(
        m
        for n, m in model.named_modules()
        if n.endswith((name, name + ".base_layer")) and hasattr(m, "weight_packed")
    )
    packed = base.weight_packed.clone()
    lora_b = lambda: next(p for n, p in model.named_parameters() if "lora_B" in n).detach().clone()
    b0 = lora_b()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-3)
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (2, 32), device = "cuda")
    losses = []
    for _ in range(5):
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(loss.item())
    model.eval()

    # Compiled vs eager bf16 LoRA rounding differ by ~0.3% until recompile, so compare settled forwards.
    def settled(m):
        with torch.no_grad():
            for _ in range(3):
                m(input_ids = ids)
            return m(input_ids = ids).logits.float()

    want = settled(model)
    model.save_pretrained(os.path.join(out_dir, "lora"))
    fresh, _ = Fast.from_pretrained(path, max_seq_length = 64, load_in_4bit = False, dtype = dtype)
    fresh = PeftModel.from_pretrained(fresh, os.path.join(out_dir, "lora"), torch_device = "cuda")
    fresh.eval()
    got = settled(fresh)
    adapters = lambda m: {
        k.replace(".default", ""): v for k, v in m.state_dict().items() if "lora_" in k
    }
    trained, reloaded = adapters(model), adapters(fresh)
    return {
        "losses": losses,
        "input_dtypes": sorted(set(seen)),
        "packed_unchanged": torch.equal(base.weight_packed, packed),
        "packed_dtype": str(base.weight_packed.dtype),
        "lora_changed": not torch.equal(lora_b(), b0),
        "saved": sorted(os.listdir(os.path.join(out_dir, "lora"))),
        "reload_max_abs": (got - want).abs().max().item(),
        "adapters_equal": trained.keys() == reloaded.keys()
        and all(torch.equal(trained[k], reloaded[k]) for k in trained),
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--lora-case":
        report = run_lora_case(*sys.argv[2:6])
        print("LORA_CASE " + json.dumps(report))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--arch", default = "qwen3", choices = ["qwen3", "qwen3_5"])
    a = ap.parse_args()
    k = build(a.out_dir, a.arch)
    print({v: sum(1 for x in k.values() if x == v) for v in set(k.values())})
