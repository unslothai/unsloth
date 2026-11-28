# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""Export round trip of a packed GGUF MoE model with LoRA on attention, shared expert and the routed expert stacks.

merged_16bit and save_pretrained_gguf both merge onto the dequantized base. Every merged tensor, the 3-D expert
stacks included, must be the fp32 merge of transformers' own dense load of the same GGUF plus the same adapter
(base rounded to 16 bits once, delta added, rounded once), so an expert stack merged transposed, per expert in the
wrong order, or without the LoRA scale is caught here. CPU, a synthetic Qwen3.6 (qwen35moe) GGUF written by the test.
"""

import os

import numpy as np
import pytest
import torch

from unsloth.kernels import gguf_reference as ref
from unsloth.models import gguf_packed as gp

_C = dict(
    hidden = 256,
    heads = 2,
    kv = 1,
    head_dim = 128,
    k_heads = 1,
    v_heads = 2,
    layers = 4,
    experts = 4,
    used = 2,
    expert_ffn = 256,
    shared_ffn = 256,
    vocab = 512,
)
_F16_FIELDS = {"Q8_0": (0,), "Q4_K": (0, 2), "Q6_K": (208,)}
# Block scales that keep every weight O(0.01 to 0.1) (the K-quants multiply theirs by 6 / 8-bit sub-scales), so
# activations stay O(1) and routing is stable under fp32 rounding.
_SCALES = {"Q8_0": 1e-3, "Q4_K": 1e-5, "Q6_K": 1e-5}
_EXPERT_STACKS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def _blocks(name, rows, cols, seed):
    t = ref.GGML_TYPE_IDS[name]
    elems, size = ref.GGML_BLOCK[t]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size = (rows * cols // elems, size), dtype = np.uint8)
    half = np.array([_SCALES[name]], dtype = np.float16).view(np.uint8)
    for offset in _F16_FIELDS[name]:
        raw[:, offset : offset + 2] = half
    return raw.reshape(rows, ref.row_bytes(t, cols))


def _write_qwen35moe_gguf(path):
    gguf = pytest.importorskip("gguf")
    c = _C
    value_dim, key_dim = c["v_heads"] * c["head_dim"], c["k_heads"] * c["head_dim"]
    arch = "qwen35moe"
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in {
        "context_length": 256,
        "embedding_length": c["hidden"],
        "block_count": c["layers"],
        "attention.head_count": c["heads"],
        "attention.head_count_kv": c["kv"],
        "attention.key_length": c["head_dim"],
        "attention.value_length": c["head_dim"],
        "rope.dimension_count": 32,
        "full_attention_interval": 4,
        "ssm.conv_kernel": 4,
        "ssm.state_size": c["head_dim"],
        "ssm.group_count": c["k_heads"],
        "ssm.time_step_rank": c["v_heads"],
        "ssm.inner_size": value_dim,
        "expert_count": c["experts"],
        "expert_used_count": c["used"],
        "expert_feed_forward_length": c["expert_ffn"],
        "expert_shared_feed_forward_length": c["shared_ffn"],
    }.items():
        writer.add_uint32(f"{arch}.{key}", value)
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32(f"{arch}.rope.freq_base", 10000.0)
    writer.add_array(f"{arch}.rope.dimension_sections", [6, 5, 5, 0])
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(c["vocab"])])
    rng = np.random.default_rng(0)
    seed = iter(range(10_000))

    def quantized(
        name,
        rows,
        cols,
        type_name,
        experts = None,
    ):
        data = _blocks(type_name, rows * (experts or 1), cols, next(seed))
        if experts:
            data = data.reshape(experts, rows, -1)
        writer.add_tensor(name, data, raw_dtype = getattr(gguf.GGMLQuantizationType, type_name))

    def floats(
        name,
        *shape,
        offset = 0.0,
        std = 0.1,
    ):
        writer.add_tensor(name, (offset + std * rng.standard_normal(shape)).astype(np.float32))

    H, E = c["hidden"], c["experts"]
    quantized("token_embd.weight", c["vocab"], H, "Q8_0")
    quantized("output.weight", c["vocab"], H, "Q6_K")
    floats("output_norm.weight", H, offset = 1.0)
    for layer in range(c["layers"]):
        p = f"blk.{layer}."
        floats(p + "attn_norm.weight", H, offset = 1.0)
        floats(p + "post_attention_norm.weight", H, offset = 1.0)
        # Routed experts (gate / up share a type: transformers concatenates them before dequantizing) and the shared expert.
        floats(p + "ffn_gate_inp.weight", E, H, std = 1.0)
        quantized(p + "ffn_gate_exps.weight", c["expert_ffn"], H, "Q4_K", experts = E)
        quantized(p + "ffn_up_exps.weight", c["expert_ffn"], H, "Q4_K", experts = E)
        quantized(p + "ffn_down_exps.weight", H, c["expert_ffn"], "Q6_K", experts = E)
        floats(p + "ffn_gate_inp_shexp.weight", H)
        quantized(p + "ffn_gate_shexp.weight", c["shared_ffn"], H, "Q4_K")
        quantized(p + "ffn_up_shexp.weight", c["shared_ffn"], H, "Q4_K")
        quantized(p + "ffn_down_shexp.weight", H, c["shared_ffn"], "Q6_K")
        if (layer + 1) % 4 == 0:
            quantized(p + "attn_q.weight", 2 * c["heads"] * c["head_dim"], H, "Q4_K")
            quantized(p + "attn_k.weight", c["kv"] * c["head_dim"], H, "Q8_0")
            quantized(p + "attn_v.weight", c["kv"] * c["head_dim"], H, "Q6_K")
            quantized(p + "attn_output.weight", H, c["heads"] * c["head_dim"], "Q4_K")
            floats(p + "attn_q_norm.weight", c["head_dim"], offset = 1.0)
            floats(p + "attn_k_norm.weight", c["head_dim"], offset = 1.0)
            continue
        quantized(p + "attn_qkv.weight", 2 * key_dim + value_dim, H, "Q8_0")
        quantized(p + "attn_gate.weight", value_dim, H, "Q8_0")
        quantized(p + "ssm_beta.weight", c["v_heads"], H, "Q8_0")
        quantized(p + "ssm_alpha.weight", c["v_heads"], H, "Q8_0")
        quantized(p + "ssm_out.weight", H, value_dim, "Q4_K")
        writer.add_tensor(
            p + "ssm_a", -np.exp(0.1 * rng.standard_normal(c["v_heads"])).astype(np.float32)
        )
        floats(p + "ssm_dt.bias", c["v_heads"])
        floats(p + "ssm_conv1d.weight", 2 * key_dim + value_dim, 4)
        floats(p + "ssm_norm.weight", c["head_dim"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _lora_config(peft):
    return peft.LoraConfig(
        r = 4,
        lora_alpha = 16,  # scale 4: a merge that drops it is far outside the tolerance below
        target_modules = ["q_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "down_proj"],
        target_parameters = list(_EXPERT_STACKS),
    )


def _randomize_lora(model):
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator = generator) * 0.05)


def _read_checkpoint(folder):
    from safetensors.torch import load_file

    tensors = {}
    for name in os.listdir(folder):
        if name.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(folder, name)))
    return tensors


def _assert_is_the_oracle_merge(
    got,
    base,
    merged,
    dtype = torch.bfloat16,
):
    """`got` (a saved checkpoint) against the fp32 oracle: base rounded once, fp32 delta added, rounded once."""
    assert not [k for k in got if "weight_packed" in k or "lora_" in k]
    assert set(got) == set(merged), sorted(set(got) ^ set(merged))[:10]
    changed = []
    for key, want32 in merged.items():
        value = got[key]
        assert value.dtype == dtype, (key, value.dtype)
        delta = want32 - base[key]
        want = (base[key].to(dtype).float() + delta).to(dtype)
        if not torch.any(delta):
            assert torch.equal(value, base[key].to(dtype)), key
            continue
        changed.append(key)
        assert value.shape == want.shape, (key, value.shape, want.shape)
        # Within one 16-bit ulp of the oracle; the LoRA delta is many ulps, so a wrong expert merge fails here.
        spacing = torch.ldexp(
            torch.ones_like(want, dtype = torch.float32),
            torch.frexp(want.float().abs().clamp_min(1e-30))[1] - 8,
        )
        ulps = (value.float() - want.float()).abs() / spacing
        assert float(ulps.max()) <= 1.0, (key, float(ulps.max()))
        assert float((delta.abs() / spacing).max()) > 32, key
    return changed


@pytest.fixture(scope = "module")
def moe_case(tmp_path_factory):
    if gp.gguf_packed_supported() is not None:
        pytest.skip(f"packed GGUF loading needs transformers >= {gp.GGUF_PACKED_MIN_TRANSFORMERS}")
    pytest.importorskip("transformers.models.qwen3_5_moe")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    folder = tmp_path_factory.mktemp("moe")
    _write_qwen35moe_gguf(folder / "tiny-moe.gguf")
    dense = AutoModelForCausalLM.from_pretrained(
        folder, gguf_file = "tiny-moe.gguf", dtype = torch.float32, attn_implementation = "eager"
    )
    with gp.gguf_packed_load(str(folder / "tiny-moe.gguf")):
        packed = AutoModelForCausalLM.from_pretrained(
            folder,
            gguf_file = "tiny-moe.gguf",
            dtype = torch.bfloat16,  # what users train in, and what the merge writes
            attn_implementation = "eager",
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    census = packed._unsloth_gguf_packed_census
    # Older unsloth_zoo dequantizes the experts at load; the stacks are only exercised packed with a current one.
    if census["packed_experts"] == 0:
        pytest.skip("this unsloth_zoo has no packed expert stacks")
    assert census["packed_experts"] == 2 * _C["layers"], census
    base = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    lora = peft.get_peft_model(packed, _lora_config(peft))
    _randomize_lora(lora)
    # The oracle: the same adapter on transformers' dense load, merged in fp32.
    oracle = peft.get_peft_model(dense, _lora_config(peft))
    state = peft.get_peft_model_state_dict(lora)
    assert any(".experts." in k for k in state), sorted(state)[:5]
    peft.set_peft_model_state_dict(oracle, state)
    ids = torch.randint(0, _C["vocab"], (2, 11), generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        oracle_logits = oracle(input_ids = ids).logits
        with oracle.disable_adapter():
            base_logits = oracle(input_ids = ids).logits
    merged = {k: v.detach().clone() for k, v in oracle.merge_and_unload().state_dict().items()}
    # (The packed forward itself is covered by test_gguf_packed_lora.py; it is slow on CPU in bf16.)
    packed_state = {k: v.detach().clone() for k, v in lora.state_dict().items()}
    return dict(
        folder = folder,
        lora = lora,
        packed_state = packed_state,
        base = base,
        merged = merged,
        ids = ids,
        oracle_logits = oracle_logits,
        base_logits = base_logits,
    )


def _assert_packed_model_untouched(case):
    now = case["lora"].state_dict()
    assert now.keys() == case["packed_state"].keys()
    for key, value in case["packed_state"].items():
        assert now[key].dtype == value.dtype and torch.equal(now[key], value), key
    assert gp.is_gguf_packed_model(case["lora"])


def test_the_oracle_merge_is_the_lora_forward(moe_case):
    # The adapter changes the model, and the fp32 merge of the dense load reproduces the unmerged dense forward,
    # so the oracle the saves are held to is the trained function.
    effect = (moe_case["oracle_logits"] - moe_case["base_logits"]).abs().max()
    assert effect > 0.1, float(effect)
    from transformers import AutoModelForCausalLM

    config = moe_case["lora"].get_base_model().config
    reloaded = AutoModelForCausalLM.from_config(
        config, dtype = torch.float32, attn_implementation = "eager"
    )
    reloaded.load_state_dict(moe_case["merged"], strict = True)
    with torch.no_grad():
        got = reloaded.eval()(input_ids = moe_case["ids"]).logits
    scale = float(moe_case["oracle_logits"].abs().max())
    torch.testing.assert_close(
        got, moe_case["oracle_logits"], rtol = 1e-4, atol = 1e-5 * max(scale, 1.0)
    )


@pytest.fixture
def cpu_merge(monkeypatch):
    # The suite's CUDA spoof says DEVICE_TYPE == "cuda"; the merge's T4 probe would then initialise CUDA.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "NVIDIA A100-SPOOFED")


def test_merged_16bit_save_of_packed_moe_lora(moe_case, tmp_path, cpu_merge):
    from unsloth.save import unsloth_generic_save

    out = tmp_path / "merged"
    unsloth_generic_save(moe_case["lora"], None, str(out), save_method = "merged_16bit")
    changed = _assert_is_the_oracle_merge(
        _read_checkpoint(out), moe_case["base"], moe_case["merged"]
    )
    for layer in range(_C["layers"]):
        for stack in _EXPERT_STACKS:
            assert f"model.layers.{layer}.{stack}" in changed, (layer, stack)
    # The temporary dense base is created beside the save directory.
    assert not list(tmp_path.glob(".unsloth_gguf_dense_base_*"))
    # The save leaves the live model as it was: packed base, same adapter.
    _assert_packed_model_untouched(moe_case)


def test_gguf_export_of_packed_moe_lora_hands_the_converter_the_merge(
    moe_case, tmp_path, monkeypatch, cpu_merge
):
    import unsloth.save as save_mod

    seen = {}

    def _save_to_gguf(**kwargs):
        folder = kwargs.get("model_directory") or kwargs.get("save_directory")
        seen.update(_read_checkpoint(folder))
        output = tmp_path / "out_gguf" / "model.Q8_0.gguf"
        output.parent.mkdir(exist_ok = True)
        output.write_bytes(b"GGUF")
        return [str(output)], True, False

    monkeypatch.setattr(save_mod, "_is_vlm", lambda _model: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _model: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _t: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_kw: None)
    monkeypatch.setattr(save_mod, "_preflight_gguf_disk", lambda **kw: (kw["save_directory"], True))
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_kw: None)
    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)

    class _Tokenizer:
        bos_token_id = 1
        chat_template = None

        def __call__(self, text, **kwargs):
            return {"input_ids": [[self.bos_token_id, 7]]}

        def save_pretrained(
            self,
            save_directory = None,
            **kwargs,
        ):
            pass

    save_mod.unsloth_save_pretrained_gguf(
        moe_case["lora"], str(tmp_path / "out"), tokenizer = _Tokenizer(), quantization_method = "q8_0"
    )
    assert seen
    changed = _assert_is_the_oracle_merge(seen, moe_case["base"], moe_case["merged"])
    assert sum(k.endswith(_EXPERT_STACKS) for k in changed) == 2 * _C["layers"]
    assert not list(tmp_path.glob(".unsloth_gguf_dense_base_*"))
    _assert_packed_model_untouched(moe_case)
