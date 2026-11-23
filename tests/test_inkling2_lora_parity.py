# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 Small (``inkling2``) and Mini (``inkling2_loop``) on tiny in-test configs: the released NVFP4 / MXFP8
expert layout decodes bit for bit, Unsloth's default LoRA targets (attention Linears + expert parameters) train,
one LoRA step equals plain transformers + PEFT computed in a process that never imports unsloth, and the saved
adapter reloads with PEFT to the same logits and the same next step."""

import copy
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers.models.inkling2.modeling_inkling2")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
_ATTN_LEAVES = {
    "self_attn.q_proj.linear",
    "self_attn.k_proj.linear",
    "self_attn.v_proj.linear",
    "self_attn.o_proj.linear",
    "self_attn.o_proj.gate_proj",
}
_LAYERS = 12
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _inkling2():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_inkling2_parity", _ROOT / "unsloth" / "models" / "inkling2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(loop):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    # Real layer structure (every 6th layer global, residual blocks of 6, Mini's loop over 6..12), tiny widths;
    # the window is shorter than the batch so the local layers really window.
    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 32,
        num_hidden_layers = _LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        swa_num_attention_heads = 4,
        swa_num_key_value_heads = 2,
        swa_head_dim = 8,
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
        return Inkling2LoopConfig(text_config = text, vision_config = vision)
    return Inkling2Config(text_config = text, vision_config = vision)


def _tiny(loop):
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(_config(loop), dtype = torch.float32)
    # Init leaves residual queries / correction biases at zero: give every path a non-trivial value.
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("query", "static_weight", "proj")) or "experts" in n:
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
            elif n.endswith("gate.weight"):
                p.copy_(torch.randn(p.shape, generator = g) * 0.5)
    return model.eval()


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 250, (2, 12), generator = g)
    am = torch.ones_like(ids)
    am[1, 9:] = 0
    return ids, am, ids.masked_fill(am == 0, -100)


def _canon(name):
    return name.replace("base_model.model.", "")


def _init(train):
    g = torch.Generator().manual_seed(1234)
    return {
        n: torch.randn(train[n].shape, generator = g) * (0.02 if "lora_B" in n else 0.2)
        for n in sorted(train)
    }


def _step(
    model,
    train,
    ids,
    am,
    labels,
    lr = 0.05,
):
    """One plain SGD step on the stock shifted fp32 CE; returns (loss, grads)."""
    for p in train.values():
        p.grad = None
    logits = model(input_ids = ids, attention_mask = am, use_cache = False).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index = -100,
    )
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in train.items()}
    with torch.no_grad():
        for p in train.values():
            p -= lr * p.grad
    return float(loss), grads


def _unsloth_peft(monkeypatch, model):
    import unsloth  # noqa: F401
    import unsloth.models.vision as vision
    from unsloth.models.inkling2 import patch_inkling2_model

    monkeypatch.setattr(vision, "clean_gpu_cache", lambda: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    assert patch_inkling2_model(model)
    model.max_seq_length = 64
    out = vision.FastBaseModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 8,
        lora_dropout = 0.0,
        random_state = 0,
        finetune_vision_layers = False,
        use_gradient_checkpointing = False,
    )
    out.train()
    return out, {_canon(n): p for n, p in out.named_parameters() if p.requires_grad}


def _assert_default_trainables(train):
    """Exactly the 5 attention Linears and the 2 expert parameters of every language layer, lora_A + lora_B."""
    per_layer = {}
    for n in train:
        m = re.search(r"language_model\.layers\.(\d+)\.(.*)\.lora_[AB]\.", n)
        assert m and "vision_tower" not in n, n
        per_layer.setdefault(int(m.group(1)), set()).add(m.group(2))
    assert sorted(per_layer) == list(range(_LAYERS))
    for leaves in per_layer.values():
        # PEFT stacks one ParamWrapper per expert parameter: experts (down_proj) wraps experts.base_layer (gate_up).
        assert leaves == _ATTN_LEAVES | {"mlp.experts", "mlp.experts.base_layer"}, leaves
    assert len(train) == _LAYERS * (len(_ATTN_LEAVES) + 2) * 2


_REFERENCE = textwrap.dedent(
    """
    import json, re, sys, torch
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    work = sys.argv[1]
    model = AutoModelForImageTextToText.from_pretrained(work + "/base", dtype=torch.float32).eval()
    leaves = json.load(open(work + "/leaves.json"))
    names = [n for n, _ in model.named_modules()
             if re.search(r"language_model\\.layers\\.\\d+\\.(" + "|".join(map(re.escape, leaves)) + r")$", n)]
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=names,
                     target_parameters=%r, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    assert not any(m.startswith("unsloth") for m in sys.modules)
    train = {n.replace("base_model.model.", ""): p for n, p in model.named_parameters() if p.requires_grad}
    init = torch.load(work + "/init.pt")
    assert set(init) == set(train), sorted(set(init) ^ set(train))[:5]
    with torch.no_grad():
        for n, v in init.items():
            train[n].copy_(v)
    b = torch.load(work + "/batch.pt")
    logits = model(input_ids=b["ids"], attention_mask=b["am"], use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             b["labels"][:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    torch.save({"loss": float(loss), "grads": {n: p.grad for n, p in train.items()}}, work + "/ref.pt")
    """
    % (_EXPERTS,)
)


def _run_reference(work):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", _REFERENCE, str(work)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return torch.load(work / "ref.pt")


def _nvfp4(w):
    E, R, C = w.shape
    w = w.float()
    amax = w.abs().amax(dim = (1, 2))
    factor = (amax / (6.0 * 448.0)).clamp(min = 1e-12)
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
    return packed, scale, amax, torch.tensor([6.0 * 448.0])


def _mxfp8(w):
    E, R, C = w.shape
    blocks = w.float().view(E, R, C // 32, 32)
    exp = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp(min = 1e-30) / 448.0)).clamp(-127, 127)
    vals = (blocks / torch.exp2(exp)[..., None]).to(torch.float8_e4m3fn).view(E, R, C)
    return vals, (exp + 127).to(torch.uint8)


def _decode(sd, key):
    """Independent decoder of the released layout, written from the format description."""
    w = sd[key]
    if key + "_amax" not in sd:
        E, R, C = w.shape
        scale = sd[key + "_scale"].float()
        return (w.float().view(E, R, C // 32, 32) * torch.exp2(scale - 127)[..., None]).view(
            E, R, C
        )
    code = torch.stack(((w & 0xF).long(), (w >> 4).long()), -1).reshape(w.shape[0], w.shape[1], -1)
    val = torch.tensor(_E2M1)[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    E, R, C = val.shape
    val = val.view(E, R, C // 16, 16) * sd[key + "_scale"].float()[..., None]
    factor = sd[key + "_amax"].float() / sd[key + "_outer_scale_denominator"].float()
    return (val * factor[:, None, None, None]).view(E, R, C)


@pytest.mark.parametrize("loop", [True, False], ids = ["inkling2_loop", "inkling2"])
def test_released_layout_decodes_bit_exact(tmp_path, loop):
    """Every expert stack of a released-layout checkpoint (NVFP4 layers, MXFP8 last layers, as Mini 0-19 / 20-23
    and Small 0-39 / 40-47) loads exactly as an independent decode of the packed bytes; every other tensor is
    unchanged. Small needs Unsloth's conversion (transformers 5.24 maps its packed experts as dense tensors)."""
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForImageTextToText

    assert _inkling2().register_inkling2_expert_dequantization()
    _tiny(loop).save_pretrained(tmp_path / "bf16")
    sd = load_file(tmp_path / "bf16" / "model.safetensors")
    packed, plain = {}, {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$", k)
        if m is None:
            packed[k] = plain[k] = v
            continue
        if int(m.group(1)) >= _LAYERS - 2:
            packed[k], packed[k + "_scale"] = _mxfp8(v)
        else:
            (
                packed[k],
                packed[k + "_scale"],
                packed[k + "_amax"],
                packed[k + "_outer_scale_denominator"],
            ) = _nvfp4(v)
        plain[k] = _decode(packed, k)
    config = json.loads((tmp_path / "bf16" / "config.json").read_text())
    for name, tensors, quant in (("q", packed, True), ("ref", plain, False)):
        (tmp_path / name).mkdir()
        save_file(tensors, tmp_path / name / "model.safetensors", metadata = {"format": "pt"})
        cfg = dict(config)
        if quant:
            cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        (tmp_path / name / "config.json").write_text(json.dumps(cfg))
    ref = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "ref", dtype = torch.float32
    ).state_dict()
    got = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "q", dtype = torch.float32
    ).state_dict()
    assert set(ref) == set(got)
    experts = [k for k in ref if re.search(r"mlp\.experts\.(gate_up_proj|down_proj)$", k)]
    assert len(experts) == 2 * _LAYERS
    for k in ref:
        assert torch.equal(got[k], ref[k]), k


@pytest.mark.parametrize("loop", [True, False], ids = ["inkling2_loop", "inkling2"])
def test_one_lora_step_matches_peft(tmp_path, monkeypatch, loop):
    """Unsloth's get_peft_model with the default targets (attention Linears + routed experts by parameter): same
    trainable tensors as plain PEFT on the exact names, same loss and same gradients after one step (fp32)."""
    base = _tiny(loop)
    base.save_pretrained(tmp_path / "base")
    model, train = _unsloth_peft(monkeypatch, copy.deepcopy(base))
    _assert_default_trainables(train)
    assert list(model.peft_config["default"].target_parameters) == _EXPERTS
    init = _init(train)
    with torch.no_grad():
        for n, v in init.items():
            train[n].copy_(v)
    ids, am, labels = _batch()
    torch.save(init, tmp_path / "init.pt")
    torch.save({"ids": ids, "am": am, "labels": labels}, tmp_path / "batch.pt")
    (tmp_path / "leaves.json").write_text(json.dumps(sorted(_ATTN_LEAVES)))
    loss, grads = _step(model, train, ids, am, labels)
    ref = _run_reference(tmp_path)
    assert set(ref["grads"]) == set(grads)
    assert abs(ref["loss"] - loss) < 1e-5, (ref["loss"], loss)
    for n, g in grads.items():
        r = ref["grads"][n].double()
        assert float(r.norm()) > 0, n
        assert float((g.double() - r).norm() / r.norm()) < 1e-4, n


@pytest.mark.parametrize("loop", [True, False], ids = ["inkling2_loop", "inkling2"])
def test_adapter_save_reload_round_trip(tmp_path, monkeypatch, loop):
    """After a step, save_pretrained writes every LoRA tensor (attention and both expert ParamWrappers) with its
    trained value; PeftModel.from_pretrained on a fresh base gives the same logits and the same next step."""
    from peft import PeftModel
    from safetensors.torch import load_file

    base = _tiny(loop)
    model, train = _unsloth_peft(monkeypatch, copy.deepcopy(base))
    with torch.no_grad():
        for n, v in _init(train).items():
            train[n].copy_(v)
    ids, am, labels = _batch()
    _step(model, train, ids, am, labels)
    with torch.no_grad():
        before = model(input_ids = ids, attention_mask = am, use_cache = False).logits
    model.save_pretrained(tmp_path / "adapter")
    saved = load_file(tmp_path / "adapter" / "adapter_model.safetensors")
    assert len(saved) == len(train)
    expert_keys = [k for k in saved if ".mlp.experts." in k]
    assert len(expert_keys) == _LAYERS * 2 * 2
    assert any(k.endswith("mlp.experts.base_layer.lora_A.weight") for k in expert_keys)
    memory = {re.sub(r"\.default(?=\.|$)", "", n): p for n, p in train.items()}
    for k, v in saved.items():
        assert torch.equal(memory[_canon(k)].detach(), v), k
    cfg = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert list(cfg["target_parameters"]) == _EXPERTS
    next_loss, next_grads = _step(model, train, ids, am, labels)

    # The model lives on CPU; without torch_device PEFT reads the (spoofed in CI) CUDA probe.
    reloaded = PeftModel.from_pretrained(
        copy.deepcopy(base), tmp_path / "adapter", is_trainable = True, torch_device = "cpu"
    )
    reloaded.train()
    train2 = {_canon(n): p for n, p in reloaded.named_parameters() if p.requires_grad}
    assert set(train2) == set(train)
    with torch.no_grad():
        after = reloaded(input_ids = ids, attention_mask = am, use_cache = False).logits
    assert torch.allclose(after, before, rtol = 0, atol = 1e-5), float((after - before).abs().max())
    loss2, grads2 = _step(reloaded, train2, ids, am, labels)
    assert abs(loss2 - next_loss) < 1e-5
    for n, g in next_grads.items():
        assert float((grads2[n] - g).norm() / g.norm()) < 1e-4, n
