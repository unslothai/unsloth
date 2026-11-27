# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 sharded loading of per-expert checkpoints transformers 5 stacks on load (CPU, no process group)."""

import glob
import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from transformers5_loading import require_core_model_loading  # noqa: E402

require_core_model_loading()


def _tp():
    _load_native()
    import importlib
    return importlib.import_module("unsloth.models.tp_native")


def _qwen3_moe(directory, experts = 12):
    import transformers

    config = transformers.Qwen3MoeConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
        moe_intermediate_size = 8,
        num_experts = experts,
        num_experts_per_tok = 2,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    model.save_pretrained(directory, safe_serialization = True)
    return model


def _meta(directory):
    import transformers
    from accelerate import init_empty_weights

    config = transformers.AutoConfig.from_pretrained(directory)
    with init_empty_weights():
        return transformers.AutoModelForCausalLM.from_config(config)


def _index(directory):
    files = glob.glob(os.path.join(directory, "*.safetensors"))
    from safetensors import safe_open

    out = {}
    for path in files:
        with safe_open(path, framework = "pt") as f:
            out.update(dict.fromkeys(f.keys(), path))
    return out


def test_gate_up_rows_are_each_experts_gate_then_up():
    tp = _tp()
    with tempfile.TemporaryDirectory() as d:
        full = _qwen3_moe(d)
        index = _index(d)
        aliases = tp._checkpoint_aliases(_meta(d), index, stacked_experts = True)
        gate_up = aliases["model.layers.0.mlp.experts.gate_up_proj"]
        assert gate_up[10] == (
            "model.layers.0.mlp.experts.10.gate_proj.weight",
            "model.layers.0.mlp.experts.10.up_proj.weight",
        )
        assert aliases["model.layers.0.mlp.experts.down_proj"][11] == (
            "model.layers.0.mlp.experts.11.down_proj.weight",
        )
        # Tensor / pipeline parallel (no stacked_experts) keep the refusal.
        assert (
            tp._checkpoint_aliases(_meta(d), index)["model.layers.0.mlp.experts.gate_up_proj"]
            is None
        )
        reader = tp._SliceReader(index)
        want = full.model.layers[0].mlp.experts.gate_up_proj.detach()
        assert reader.shape(gate_up) == tuple(want.shape)
        assert torch.equal(reader.rows_to(gate_up, 5, 12, torch.float32, "cpu"), want[5:12])
        assert reader.slice_bytes == want[5:12].numel() * 4 and reader.full_bytes == 0


def test_expert_index_is_read_after_the_renamings():
    """A checkpoint naming experts otherwise (here `moe.experts.N`), renamed before the stacking converter."""
    tp = _tp()
    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory() as d:
        full = _qwen3_moe(d)
        for path in glob.glob(os.path.join(d, "*.safetensors")):
            tensors = {
                k.replace(".mlp.experts.", ".moe.experts."): v for k, v in load_file(path).items()
            }
            save_file(tensors, path, metadata = {"format": "pt"})
        index = _index(d)
        aliases = tp._checkpoint_aliases(
            _meta(d), index, {r"\.moe\.experts\.": ".mlp.experts."}, stacked_experts = True
        )
        gate_up = aliases["model.layers.0.mlp.experts.gate_up_proj"]
        assert gate_up[10][1] == "model.layers.0.moe.experts.10.up_proj.weight"
        want = full.model.layers[0].mlp.experts.gate_up_proj.detach()
        assert torch.equal(
            tp._SliceReader(index).rows_to(gate_up, 0, 12, torch.float32, "cpu"), want
        )


def test_missing_expert_keeps_the_refusal():
    tp = _tp()
    with tempfile.TemporaryDirectory() as d:
        _qwen3_moe(d)
        index = {k: v for k, v in _index(d).items() if ".experts.3.up_proj." not in k}
        aliases = tp._checkpoint_aliases(_meta(d), index, stacked_experts = True)
        assert aliases["model.layers.0.mlp.experts.gate_up_proj"] is None
        assert isinstance(aliases["model.layers.0.mlp.experts.down_proj"], tuple)


def test_packed_per_expert_checkpoint_refuses_as_unsupported():
    """Packed 4-bit bytes per expert raise _TensorByTensorUnsupported (the pre-quantized fallback) before any read."""
    tp = _tp()
    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory() as d:
        _qwen3_moe(d)
        for path in glob.glob(os.path.join(d, "*.safetensors")):
            tensors = load_file(path)
            for key in list(tensors):
                if ".experts." in key and key.endswith("down_proj.weight"):
                    tensors[key] = torch.zeros((tensors[key].numel() // 2, 1), dtype = torch.uint8)
                    tensors[key + ".absmax"] = torch.ones(2)
            save_file(tensors, path, metadata = {"format": "pt"})
        with open(os.path.join(d, "config.json"), encoding = "utf-8") as f:
            assert "quantization_config" not in json.load(f)
        model = _meta(d)
        reads = []

        def read_local(old, reader, key):
            reads.append(key)
            return None

        with pytest.raises(tp._TensorByTensorUnsupported, match = "stack to"):
            tp._fill_from_checkpoint(
                model, d, {}, torch.device("cpu"), read_local, stacked_experts = True
            )
        assert reads == []


FAMILIES = {
    "Qwen3MoeConfig": dict(head_dim = 8, moe_intermediate_size = 8, num_experts = 12),
    "MixtralConfig": dict(head_dim = 8, num_local_experts = 12),
    "Glm4MoeConfig": dict(
        head_dim = 8,
        moe_intermediate_size = 8,
        n_routed_experts = 12,
        n_shared_experts = 1,
        first_k_dense_replace = 0,
        n_group = 1,
        topk_group = 1,
    ),
    "DeepseekV3Config": dict(
        moe_intermediate_size = 8,
        n_routed_experts = 12,
        n_shared_experts = 1,
        first_k_dense_replace = 0,
        n_group = 1,
        topk_group = 1,
        kv_lora_rank = 8,
        q_lora_rank = 8,
        qk_rope_head_dim = 4,
        qk_nope_head_dim = 4,
        v_head_dim = 8,
    ),
    "Qwen3_5MoeTextConfig": dict(
        head_dim = 8,
        moe_intermediate_size = 8,
        shared_expert_intermediate_size = 8,
        num_experts = 12,
        linear_num_value_heads = 2,
        linear_num_key_heads = 2,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
    ),
    "NemotronHConfig": dict(
        layers_block_type = ["moe", "full_attention"],
        head_dim = 8,
        n_routed_experts = 12,
        moe_intermediate_size = 8,
        moe_shared_expert_intermediate_size = 8,
        moe_latent_size = 8,
        use_mamba_kernels = False,
    ),
    "GptOssConfig": dict(head_dim = 8, num_local_experts = 12),
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_stacked_expert_rows_match_the_full_load(name):
    """Rows read from each expert stack's checkpoint tensors equal the full load."""
    import transformers

    if not hasattr(transformers, name):
        pytest.skip(f"transformers {transformers.__version__} has no {name}")
    tp = _tp()
    config = getattr(transformers, name)(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        num_experts_per_tok = 2,
        **FAMILIES[name],
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization = True)
        full = transformers.AutoModelForCausalLM.from_pretrained(d, torch_dtype = torch.float32)
        index = _index(d)
        meta = _meta(d)
        aliases = tp._checkpoint_aliases(meta, index, stacked_experts = True)
        reader = tp._SliceReader(index)
        stacks = [
            (n, p)
            for n, p in full.named_parameters()
            if n.rsplit(".", 2)[-2] == "experts" and p.ndim == 3
        ]
        assert stacks
        for n, p in stacks:
            key = tp._checkpoint_key([n], meta, index) or aliases.get(n)
            assert key is not None, (n, sorted(index)[:4])
            assert reader.shape(key) == tuple(p.shape), n
            got = reader.rows_to(key, 3, 9, torch.float32, "cpu")
            assert torch.equal(got, p.detach()[3:9]), n


def test_expert_index_keeps_the_converter_regex():
    tp = _tp()
    assert (
        tp._expert_number(r"\.experts.*.w1.weight", "model.layers.0.mlp.experts.10.w1.weight") == 10
    )
    assert (
        tp._expert_number(
            "mlp.experts.*.gate_proj.weight", "model.layers.1.mlp.experts.3.gate_proj.weight"
        )
        == 3
    )
    assert (
        tp._expert_number(
            "mlp.experts.*.up_proj.weight", "model.layers.1.mlp.experts.3.up_proj.weight_scale"
        )
        is None
    )
    assert (
        tp._expert_number(
            "mlp.experts.*.up_proj.weight", "model.layers.1.mlp.shared_experts.up_proj.weight"
        )
        is None
    )


def test_multi_lora_sizes_experts_from_the_whole_stack():
    """inject_multi_lora_experts between sharded loading and shard_model sizes A / B from the whole stack."""
    import importlib
    import types

    pytest.importorskip("triton")
    _tp()
    kernels = types.ModuleType("unsloth.kernels")
    kernels.__path__ = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unsloth", "kernels")
    ]
    sys.modules.setdefault("unsloth.kernels", kernels)
    ml = importlib.import_module("unsloth.kernels.multi_lora")
    local = torch.nn.Parameter(torch.zeros(6, 4, 8))
    assert ml._expert_stack_shape(local) == (6, 4, 8)
    setattr(
        local, "_unsloth_fsdp2_preload", {"shape": (12, 4, 8), "rows": (6, 12), "packed": False}
    )
    assert ml._expert_stack_shape(local) == (12, 4, 8)
    local._original_shape = torch.Size((12, 4, 8))
    assert ml._expert_stack_shape(local) == (12, 4, 8)
