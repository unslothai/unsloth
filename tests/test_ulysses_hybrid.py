# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Gated DeltaNet hybrids (Qwen3.5, Qwen3-Next), on CPU gloo ranks."""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3NextConfig"):
    pytest.skip("transformers without Qwen3-Next", allow_module_level = True)
KINDS = ["qwen3_5", "qwen3_next"] if hasattr(transformers, "Qwen3_5TextConfig") else ["qwen3_next"]


def _config(kind, **overrides):
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        layer_types = ["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
        attention_dropout = 0.0,
        tie_word_embeddings = False,
    )
    common.update(overrides)
    if kind == "qwen3_next":
        common.update(
            num_experts = 2,
            num_experts_per_tok = 1,
            moe_intermediate_size = 16,
            shared_expert_intermediate_size = 16,
            # Dense MLPs: the MoE path may pick CUDA kernels in a forked CPU worker.
            mlp_only_layers = list(range(common["num_hidden_layers"])),
        )
        return transformers.Qwen3NextConfig(**common), transformers.Qwen3NextForCausalLM
    return transformers.Qwen3_5TextConfig(**common), transformers.Qwen3_5ForCausalLM


def _model(kind, **overrides):
    torch.manual_seed(0)
    config, cls = _config(kind, **overrides)
    model = cls(config).train()
    model.config._attn_implementation = "sdpa"
    return model


def _force_torch_kernels(model):
    # The parity check is about the exchange, not the kernels: fla / Triton would touch CUDA in a
    # forked CPU worker whenever a GPU is visible. Only called inside the worker process.
    import inspect

    from unsloth.utils.packing import _hub_closure
    for module in model.modules():
        if not type(module).__name__.endswith("GatedDeltaNet"):
            continue
        globs = inspect.getmodule(type(module)).__dict__
        for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
            torch_function = _hub_closure(globs.get(name)).get("torch_function")
            if torch_function is not None:
                globs[name] = torch_function
        if "chunk_gated_delta_rule" in vars(module):
            # transformers 4.57 binds the kernels on the instance.
            module.chunk_gated_delta_rule = globs["torch_chunk_gated_delta_rule"]
            module.causal_conv1d_fn = None


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
        )
        / count
    )


def _parity_worker(rank, world, kind, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind)
    _force_torch_kernels(model)
    ids = torch.randint(0, 64, (2, 13), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(model(input_ids = ids, use_cache = False).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none = True)

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        _loss(out.logits, inputs["shift_labels"], count).backward()
    for name, param in model.named_parameters():
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-6, msg = name)


def _spawn(fn, world, *args):
    # Fork: a spawned child would re-import unsloth without the conftest CUDA spoof.
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", KINDS)
def test_ulysses_hybrid_gradients_match_one_rank(kind):
    _spawn(_parity_worker, 2, kind, False)


def test_shard_local_gated_delta_is_wrong():
    # Negative control: the stock forward on a sequence shard restarts the scan at the shard seam.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, KINDS[0], True)


def test_hybrid_gates():
    model = _model(KINDS[0])
    assert "ulysses" in cp._unsupported_attention(model.config, "ring")
    assert cp._unsupported_attention(model.config, "ulysses") is None
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    with pytest.raises(NotImplementedError, match = "context_parallel_backend = 'ulysses'"):
        cp._check_stateful_mixers(model, "ring", 2)
    cp._check_stateful_mixers(model, "ulysses", 2)
    with pytest.raises(ValueError, match = "num_k_heads"):
        cp._check_stateful_mixers(model, "ulysses", 4)

    class FakeMamba2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = torch.nn.Parameter(torch.zeros(2))
            self.dt_bias = torch.nn.Parameter(torch.zeros(2))
            self.conv1d = torch.nn.Conv1d(2, 2, 4, groups = 2)

    model.model.layers[0].mixer = FakeMamba2()
    with pytest.raises(NotImplementedError, match = "Gated DeltaNet linear attention only"):
        cp._check_stateful_mixers(model, "ulysses", 2)


def test_eager_attention_is_refused():
    model = _model(KINDS[0])
    attn = model.model.layers[1].self_attn
    with pytest.raises(NotImplementedError, match = "attn_implementation"):
        uh.install_hf_attention_wrapper(attn, "eager", cp.get_cp_manager)


def test_hybrid_packing_metadata_is_off_under_context_parallelism():
    from unsloth.utils import packing

    kwargs = {
        "input_ids": torch.ones(1, 8, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
        "packed_seq_lengths": torch.tensor([3, 5], dtype = torch.int32),
    }
    assert packing._hybrid_varlen_metadata(kwargs) is not None
    manager = _manager(2, 0)
    with manager.apply({"input_ids": torch.ones(1, 8, dtype = torch.long)}):
        # The shard sees 4 tokens: local boundaries would split documents at the wrong places.
        assert packing._hybrid_varlen_metadata(kwargs) is None


def test_ulysses_refuses_position_only_packing():
    inputs = {
        "input_ids": torch.ones(1, 6, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
    }
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        _manager(2, 0)._prepare_inputs(inputs)
    inputs["packed_seq_lengths"] = torch.tensor([3, 3], dtype = torch.int32)
    _manager(2, 0)._prepare_inputs(inputs)
