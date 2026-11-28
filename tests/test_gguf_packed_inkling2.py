# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`gguf_packed = True` on Inkling 2 GGUFs (`inkling2` = Small, `inkling2_loop` = Mini).

The fixtures are written the way llama.cpp's conversion/inkling2.py writes a checkpoint (ggml leaf names, dynamic
short-conv kernels as `flip(0).T`, static ones squeezed, the relative-bias projection transposed, the padded
unembedding, the loop message queries per layer), with llama.cpp's optional per-expert `<name>.scale` tensors, and
split across shards like the Small repos. Every load must give back the transformers model exactly."""

import os
import struct

import numpy as np
import pytest
import torch

from unsloth.kernels import gguf_reference as ref
from unsloth.models import gguf_packed as gp

_LAYER_TYPES = ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"]
_HIDDEN, _EXPERTS, _MOE = 256, 4, 128


def _skip_unless_supported():
    if gp.gguf_packed_supported() is not None:
        pytest.skip(f"packed GGUF loading needs transformers >= {gp.GGUF_PACKED_MIN_TRANSFORMERS}")
    pytest.importorskip("transformers.models.inkling2_loop")


def _tiny_model(loop):
    from transformers import AutoModelForCausalLM

    common = dict(
        hidden_size = _HIDDEN,
        num_hidden_layers = len(_LAYER_TYPES),
        vocab_size = 96,
        unpadded_vocab_size = 90,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 64,
        swa_num_attention_heads = 4,
        swa_num_key_value_heads = 2,
        swa_head_dim = 64,
        sliding_window_size = 8,
        d_rel = 4,
        rel_extent = 16,
        n_layers_per_attn_res_block = 2,
        n_routed_experts = _EXPERTS,
        num_experts_per_tok = 2,
        moe_intermediate_size = _MOE,
        eos_token_id = 89,
        pad_token_id = 88,
        layer_types = _LAYER_TYPES,
    )
    if loop:
        from transformers.models.inkling2_loop.configuration_inkling2_loop import (
            Inkling2LoopTextConfig,
        )
        config = Inkling2LoopTextConfig(
            **common,
            loop_layer_index_start = 2,
            loop_layer_index_end = 4,
            loop_num_passes = 2,
            loop_gate_rank = 8,
        )
    else:
        from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
        config = Inkling2TextConfig(**common)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(
                torch.randn_like(param) * 0.05 + (1.0 if name.endswith("norm.weight") else 0.0)
            )
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn_like(buffer) * 0.1)
    return model.eval()


def _gguf_tensors(model):
    """{gguf name: float32 array in torch order}, the layout llama.cpp's Inkling 2 converter writes."""
    config = model.config
    sd = {k: v.detach().float() for k, v in model.state_dict().items()}
    out = {}

    def put(name, tensor):
        out[name] = tensor.contiguous().numpy()

    put("token_embd.weight", sd.pop("model.embed_tokens.weight"))
    put("token_embd_norm.weight", sd.pop("model.embed_tokens.embed_norm.weight"))
    put("output_norm.weight", sd.pop("model.norm.weight"))
    put("output_res_query.weight", sd.pop("model.final_input.query"))
    head = sd.pop("lm_head.weight")
    padding = head.new_zeros(config.vocab_size - head.shape[0], head.shape[1])
    put("output.weight", torch.cat([head, padding]))
    leaves = {
        "self_attn.q_proj.linear.weight": "attn_q.weight",
        "self_attn.k_proj.linear.weight": "attn_k.weight",
        "self_attn.v_proj.linear.weight": "attn_v.weight",
        "self_attn.o_proj.linear.weight": "attn_output.weight",
        "self_attn.o_proj.gate_proj.weight": "attn_gate.weight",
        "self_attn.sink_proj.weight": "attn_sink_proj.weight",
        "self_attn.relative_position_bias.linear.weight": "attn_rel.weight",
        "self_attn.q_proj.norm.scale": "attn_q_clip.weight",
        "self_attn.relative_position_bias.norm.scale": "attn_rel_clip.weight",
        "self_attn.k_proj.sconv.weight_proj.weight": "attn_k_conv_proj.weight",
        "self_attn.v_proj.sconv.weight_proj.weight": "attn_v_conv_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "q_input_layernorm.weight": "attn_q_inp_norm.weight",
        "k_input_layernorm.weight": "attn_k_inp_norm.weight",
        "v_input_layernorm.weight": "attn_v_inp_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "attention_input.query": "attn_res_query.weight",
        "mlp_input.query": "ffn_res_query.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
        "mlp.experts.gate_up_proj": "ffn_gate_up_exps.weight",
        "mlp.experts.down_proj": "ffn_down_exps.weight",
    }
    for gate, gguf_gate in (("q", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")):
        for hf, gguf in (
            ("down_proj", "down"),
            ("forget_gate_proj", "forget"),
            ("message_gate_proj", "gain"),
        ):
            leaves[f"{gate}_message_gate.{hf}.weight"] = f"loop_{gguf_gate}_ctrl_{gguf}.weight"
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        for hf, gguf in leaves.items():
            if prefix + hf in sd:
                put(f"blk.{layer}.{gguf}", sd.pop(prefix + hf))
        put(
            f"blk.{layer}.attn_rel_bias.weight",
            sd.pop(prefix + "self_attn.relative_position_bias.proj").t(),
        )
        for proj in ("k", "v"):
            static = sd.pop(prefix + f"self_attn.{proj}_proj.sconv.static_weight", None)
            if static is not None:
                # llama.cpp: data_torch.flip(0).T for the dynamic convolution's static kernel
                put(f"blk.{layer}.attn_{proj}_conv.weight", static.flip(0).t())
            else:
                put(
                    f"blk.{layer}.attn_{proj}_conv.weight",
                    sd.pop(prefix + f"self_attn.{proj}_proj.sconv.conv1d.weight").squeeze(1),
                )
        put(
            f"blk.{layer}.attn_post_conv.weight",
            sd.pop(prefix + "post_attention_convolution.conv1d.weight").squeeze(1),
        )
        put(
            f"blk.{layer}.ffn_post_conv.weight",
            sd.pop(prefix + "post_mlp_convolution.conv1d.weight").squeeze(1),
        )
    if "model.message_input.query" in sd:
        queries = sd.pop("model.message_input.query")
        for i in range(queries.shape[0]):
            put(f"blk.{config.loop_layer_index_start + i}.loop_msg_query.weight", queries[i])
    assert not sd, sorted(sd)
    return out


def _metadata(config, loop):
    arch = "inkling2_loop" if loop else "inkling2"
    sliding = [t == "hybrid_sliding" for t in config.layer_types]
    kv = {
        "general.architecture": arch,
        f"{arch}.block_count": config.num_hidden_layers,
        f"{arch}.context_length": config.max_position_embeddings,
        f"{arch}.embedding_length": config.hidden_size,
        f"{arch}.attention.head_count": [config.num_attention_heads] * config.num_hidden_layers,
        f"{arch}.attention.head_count_kv": [config.num_key_value_heads] * config.num_hidden_layers,
        f"{arch}.attention.layer_norm_rms_epsilon": np.float32(config.rms_norm_eps),
        f"{arch}.expert_count": config.n_routed_experts,
        f"{arch}.expert_used_count": config.num_experts_per_tok,
        f"{arch}.expert_gating_func": 2,
        f"{arch}.attention.key_length": config.head_dim,
        f"{arch}.attention.key_length_swa": config.swa_head_dim,
        f"{arch}.vocab_size": config.vocab_size,
        f"{arch}.vocab_size_unpadded": config.unpadded_vocab_size,
        f"{arch}.expert_feed_forward_length": config.moe_intermediate_size,
        f"{arch}.attention.sliding_window": config.sliding_window_size,
        f"{arch}.attention.sliding_window_pattern": sliding,
        f"{arch}.shortconv.l_cache": config.conv_kernel_size,
        f"{arch}.expert_weights_scale": np.float32(config.route_scale),
        f"{arch}.activation.situ_beta": np.float32(config.mlp_gate_softcap),
        f"{arch}.activation.situ_linear_beta": np.float32(config.mlp_up_softcap),
        f"{arch}.attn_res.block_size": config.n_layers_per_attn_res_block,
        f"{arch}.final_logit_softcapping": np.float32(config.final_logit_softcapping),
        f"{arch}.attention.relative_length": config.d_rel,
        f"{arch}.attention.relative_extent": config.rel_extent,
        f"{arch}.attention.qk_clip": np.float32(config.qk_l2_clip_threshold),
        f"{arch}.attention.relative_clip": np.float32(config.rel_l2_clip_threshold),
        f"{arch}.attention.log_scale_floor": config.log_scaling_n_floor,
        f"{arch}.attention.log_scale_alpha": np.float32(config.log_scaling_alpha),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": [f"<t{i}>" for i in range(config.unpadded_vocab_size)],
        "tokenizer.ggml.eos_token_id": config.eos_token_id,
        "tokenizer.ggml.padding_token_id": config.pad_token_id,
    }
    if loop:
        kv |= {
            f"{arch}.loop.start": config.loop_layer_index_start,
            f"{arch}.loop.end": config.loop_layer_index_end,
            f"{arch}.loop.count": config.loop_num_passes,
            f"{arch}.loop.gate_rank": config.loop_gate_rank,
            f"{arch}.loop.varying": False,
        }
    return kv


# A minimal GGUF v3 writer: the gguf package on PyPI does not know every ggml type Inkling 2 ships (NVFP4, Q2_0).
def _kv_bytes(value):
    if isinstance(value, str):
        data = value.encode()
        return struct.pack("<I", 8), struct.pack("<Q", len(data)) + data
    if isinstance(value, (bool, np.bool_)):
        return struct.pack("<I", 7), struct.pack("<?", bool(value))
    if isinstance(value, (np.floating, float)):
        return struct.pack("<I", 6), struct.pack("<f", float(value))
    if isinstance(value, (int, np.integer)):
        return struct.pack("<I", 4), struct.pack("<I", int(value))
    assert isinstance(value, list) and value
    sub_type, _ = _kv_bytes(value[0])
    body = b"".join(_kv_bytes(v)[1] for v in value)
    return struct.pack("<I", 9), sub_type + struct.pack("<Q", len(value)) + body


def _write_gguf(
    path,
    kv,
    tensors,
    shard = None,
):
    """tensors: {name: (ggml type, torch-order shape, raw bytes)}; shard = (index, count) adds split.* keys."""
    kv = dict(kv)
    if shard is not None:
        kv |= {"split.no": np.uint16(shard[0]), "split.count": np.uint16(shard[1])}
    blob = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, value in kv.items():
        if isinstance(value, np.uint16):
            vtype, data = struct.pack("<I", 2), struct.pack("<H", int(value))
        else:
            vtype, data = _kv_bytes(value)
        blob += struct.pack("<Q", len(key)) + key.encode() + vtype + data
    offset, infos, datas = 0, b"", []
    for name, (ggml_type, shape, raw) in tensors.items():
        dims = list(reversed(shape))
        infos += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack(
            "<IQ", ggml_type, offset
        )
        datas.append(raw)
        offset += len(raw)
        pad = (-offset) % 32
        datas.append(b"\0" * pad)
        offset += pad
    blob += infos
    blob += b"\0" * ((-len(blob)) % 32)
    with open(path, "wb") as f:
        f.write(blob + b"".join(datas))


def _f32(array):
    return (0, tuple(array.shape), np.ascontiguousarray(array, dtype = np.float32).tobytes())


def _random_blocks(name, rows, cols, seed):
    """Random blocks of a ggml type with small finite block scales."""
    t = ref.GGML_TYPE_IDS[name]
    elems, size = ref.GGML_BLOCK[t]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size = (rows * cols // elems, size), dtype = np.uint8)
    if name == "NVFP4":
        raw[:, :4] = rng.integers(
            0x18, 0x28, size = (raw.shape[0], 4), dtype = np.uint8
        )  # UE4M3 ~ 2^-4 .. 2^-2
    else:
        raw[:, 0:2] = np.array([0.002], dtype = np.float16).view(np.uint8)
    return raw.reshape(rows, -1)


def _quantized_file(
    model,
    loop,
    path,
    split = False,
):
    """The model's file with the linears as Q8_0 and the expert stacks as NVFP4 / Q2_0 blocks (llama.cpp's
    Inkling 2 experts), a per-expert `.scale` on some stacks. Returns (scales, expected dense stacks)."""
    tensors = {name: _f32(value) for name, value in _gguf_tensors(model).items()}
    expected, scales = {}, {}
    plan = {0: ("NVFP4", "Q2_0"), 1: ("Q2_0", "NVFP4"), 2: ("NVFP4", "NVFP4"), 3: ("Q8_0", "Q2_0")}
    for layer, (gate_up_type, down_type) in plan.items():
        for leaf, type_name, rows, cols in (
            ("ffn_gate_up_exps", gate_up_type, 2 * _MOE, _HIDDEN),
            ("ffn_down_exps", down_type, _HIDDEN, _MOE),
        ):
            name = f"blk.{layer}.{leaf}.weight"
            t = ref.GGML_TYPE_IDS[type_name]
            blocks = np.stack(
                [
                    _random_blocks(type_name, rows, cols, 100 * layer + e + len(leaf))
                    for e in range(_EXPERTS)
                ]
            )
            tensors[name] = (t, (_EXPERTS, rows, cols), blocks.tobytes())
            dense = ref.dequantize_gguf(
                torch.from_numpy(blocks.copy()), t, (_EXPERTS, rows, cols), torch.float32
            )
            if layer != 3:
                scale = np.linspace(0.5, 2.0, _EXPERTS, dtype = np.float32) * (1 + layer)
                tensors[name[: -len(".weight")] + ".scale"] = _f32(scale)
                scales[name] = scale
                dense = dense * torch.from_numpy(scale).view(-1, 1, 1)
            expected[name] = dense
    for name in list(tensors):
        if name.endswith(("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_gate.weight")):
            ggml_type, shape, _ = tensors[name]
            tensors[name] = (8, shape, _random_blocks("Q8_0", *shape, len(name)).tobytes())
    kv = _metadata(model.config, loop)
    if not split:
        _write_gguf(path, kv, tensors)
        return [path], scales, expected
    if split == "two":
        # Two shards, every tensor in the second: the one shard that holds tensors is not the first.
        stem = str(path)[: -len(".gguf")]
        paths = [f"{stem}-{i:05d}-of-00002.gguf" for i in (1, 2)]
        for i, (shard_path, shard) in enumerate(zip(paths, ({}, tensors))):
            _write_gguf(shard_path, kv, shard, shard = (i, 2))
        return paths, scales, expected
    # Three shards as the Small repos ship them: the first holds the metadata only, and each expert stack's `.scale`
    # sits in a different shard from the stack itself.
    second = {
        n: v
        for i, (n, v) in enumerate(tensors.items())
        if n.endswith(".scale") or (i % 2 and not n.endswith("_exps.weight"))
    }
    third = {n: v for n, v in tensors.items() if n not in second}
    stem = str(path)[: -len(".gguf")]
    paths = [f"{stem}-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    for i, (shard_path, shard) in enumerate(zip(paths, ({}, second, third))):
        _write_gguf(shard_path, kv, shard, shard = (i, 3))
    return paths, scales, expected


def _load(
    directory,
    filename,
    dtype = torch.float32,
):
    from transformers import AutoModelForCausalLM
    with gp.gguf_packed_load():
        return AutoModelForCausalLM.from_pretrained(
            directory,
            gguf_file = filename,
            dtype = dtype,
            attn_implementation = "eager",
            quantization_config = gp.gguf_packed_quantization_config(),
        )


@pytest.mark.parametrize("loop", [True, False], ids = ["inkling2_loop", "inkling2"])
def test_inkling2_gguf_loads_back_the_model(tmp_path, loop):
    _skip_unless_supported()
    model = _tiny_model(loop)
    tensors = {name: _f32(value) for name, value in _gguf_tensors(model).items()}
    _write_gguf(tmp_path / "tiny.gguf", _metadata(model.config, loop), tensors)
    loaded = _load(tmp_path, "tiny.gguf")
    assert type(loaded).__name__ == type(model).__name__
    assert loaded.config.layer_types == model.config.layer_types
    want, got = model.state_dict(), loaded.state_dict()
    assert want.keys() == got.keys()
    for key in want:
        assert torch.equal(got[key].float(), want[key].float()), key
    ids = torch.randint(0, 90, (1, 12))
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, model(input_ids = ids).logits)


def test_bf16_load_keeps_the_float32_modules(tmp_path):
    # Inkling 2's router, residual queries, short convolutions and message gates are float32 modules
    # (`_keep_in_fp32_modules_strict`); transformers' GGUF loader would round them to the load dtype.
    _skip_unless_supported()
    model = _tiny_model(loop = True)
    tensors = {name: _f32(value) for name, value in _gguf_tensors(model).items()}
    _write_gguf(tmp_path / "tiny.gguf", _metadata(model.config, True), tensors)
    loaded = _load(tmp_path, "tiny.gguf", dtype = torch.bfloat16)
    want = model.state_dict()
    kept = 0
    for key, value in loaded.state_dict().items():
        if value.dtype == torch.float32:
            kept += 1
            assert torch.equal(value, want[key]), key
        else:
            assert value.dtype == torch.bfloat16 and torch.equal(
                value, want[key].to(torch.bfloat16)
            ), key
    assert loaded.model.layers[0].mlp.gate.weight.dtype == torch.float32
    assert loaded.model.message_input.query.dtype == torch.float32
    assert loaded.model.layers[2].q_message_gate.forget_gate_proj.weight.dtype == torch.float32
    assert loaded.model.layers[0].self_attn.q_proj.linear.weight.dtype == torch.bfloat16
    assert kept >= 6 * len(_LAYER_TYPES)


@pytest.mark.parametrize(
    "split", [False, True, "two"], ids = ["single", "split", "split_one_tensor_shard"]
)
def test_inkling2_packed_experts_carry_the_gguf_scales(tmp_path, split):
    _skip_unless_supported()
    if gp.gguf_expert_param_class() is None:
        pytest.skip("this unsloth_zoo cannot keep GGUF expert stacks packed")
    model = _tiny_model(loop = True)
    paths, scales, expected = _quantized_file(model, True, tmp_path / "tiny.gguf", split = split)
    loaded = _load(tmp_path, os.path.basename(str(paths[0])))
    census = loaded._unsloth_gguf_packed_census
    assert census["packed_experts"] == 8 and census["scaled_experts"] == 6
    assert census["types"]["NVFP4"] == 4 and census["types"]["Q2_0"] == 3
    assert census["packed_linear"] == 4 * len(_LAYER_TYPES)
    for name, dense in expected.items():
        layer, leaf = int(name.split(".")[1]), name.split(".")[2]
        param = getattr(
            loaded.model.layers[layer].mlp.experts,
            "gate_up_proj" if "gate_up" in leaf else "down_proj",
        )
        assert param.packed_format == "gguf"
        assert (param.packed_scales.numel() > 0) == (name in scales)
        # float32 decode times the float32 scale, rounded once: the value llama.cpp multiplies the product by.
        assert torch.equal(param.dequantize(dtype = torch.float32), dense), name
        assert torch.equal(param.dequantize_experts(1, 3, dtype = torch.float32), dense[1:3]), name
        assert torch.equal(param.dequantize(dtype = torch.bfloat16), dense.to(torch.bfloat16)), name
    # Unscaled weights are untouched: the F32 tensors equal the model's.
    want = model.state_dict()
    for key, value in loaded.state_dict().items():
        if (
            value.dtype != torch.uint8
            and getattr(value, "packed_format", None) is None
            and "experts" not in key
        ):
            assert torch.equal(value.float(), want[key].float()), key


def test_packed_grouped_mm_applies_the_expert_scales(monkeypatch):
    cls = gp.gguf_expert_param_class()
    if cls is None:
        pytest.skip("this unsloth_zoo cannot keep GGUF expert stacks packed")
    t = ref.GGML_TYPE_IDS["NVFP4"]
    blocks = torch.from_numpy(np.stack([_random_blocks("NVFP4", 32, 128, e) for e in range(3)]))
    scale = torch.tensor([0.5, 1.0, 3.0])
    param = cls(
        blocks, ggml_type = t, packed_cols = 128, packed_dtype = torch.float32, packed_scales = scale
    )
    plain = ref.dequantize_gguf(blocks, t, (3, 32, 128), torch.float32)

    def grouped(x, packed, ggml_type, shape, counts, transpose_b):
        out, start = [], 0
        for e, n in enumerate(counts.tolist()):
            w = plain[e]
            out.append(x[start : start + n] @ (w.t() if transpose_b else w))
            start += n
        return torch.cat(out)

    ops = gp.gguf_ops()
    monkeypatch.setattr(gp, "gguf_ops", lambda: type(ops)(**{**vars(ops), "grouped_mm": grouped}))
    counts = torch.tensor([2, 0, 3])
    x = torch.randn(5, 128)
    rows = torch.tensor([0.5, 0.5, 3.0, 3.0, 3.0])
    torch.testing.assert_close(
        param.packed_grouped_mm(x, counts, True),
        grouped(x, None, t, None, counts, True) * rows[:, None],
    )
    g = torch.randn(5, 32)
    torch.testing.assert_close(
        param.packed_grouped_mm(g, counts, False),
        grouped(g, None, t, None, counts, False) * rows[:, None],
    )
    # The dense decode agrees with the product: x @ (s W)^T.
    dense = param.dequantize(dtype = torch.float32)
    torch.testing.assert_close(dense, plain * scale.view(-1, 1, 1))


def test_split_gguf_needs_every_shard(tmp_path):
    (tmp_path / "m-00001-of-00003.gguf").write_bytes(b"")
    (tmp_path / "m-00002-of-00003.gguf").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match = "00003-of-00003"):
        gp.gguf_split_paths(tmp_path / "m-00001-of-00003.gguf")
    assert gp.gguf_split_paths(tmp_path / "single.gguf") == [str(tmp_path / "single.gguf")]


def test_config_reader_registration_is_scoped():
    _skip_unless_supported()
    import transformers.integrations.gguf.gguf_config_mapping as config_module

    before = dict(config_module.GGUF_CONFIG_ARCHS)
    with gp._extra_gguf_config_archs():
        assert {"inkling2", "inkling2_loop"} <= set(config_module.GGUF_CONFIG_ARCHS)
    assert dict(config_module.GGUF_CONFIG_ARCHS) == before
