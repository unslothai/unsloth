# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for MiniMax (M1 Lightning hybrid, M2 full attention), on CPU gloo ranks."""

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
if not hasattr(transformers, "MiniMaxConfig"):
    pytest.skip("transformers without MiniMax", allow_module_level = True)
KINDS = ["m1", "m2"] if hasattr(transformers, "MiniMaxM2Config") else ["m1"]


def _config(kind):
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 32,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_local_experts = 2,
        num_experts_per_tok = 1,
        attention_dropout = 0.0,
        tie_word_embeddings = False,
    )
    if kind == "m1":
        common.update(
            layer_types = ["linear_attention", "full_attention", "linear_attention"],
            # Small blocks so 13 tokens cross several Lightning blocks and the shard seam.
            block_size = 4,
        )
        return transformers.MiniMaxConfig(**common), transformers.MiniMaxForCausalLM
    common.update(
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        }
    )
    return transformers.MiniMaxM2Config(**common), transformers.MiniMaxM2ForCausalLM


def _model(kind):
    torch.manual_seed(0)
    config, cls = _config(kind)
    model = cls(config).train()
    model.config._attn_implementation = "sdpa"
    for module in model.modules():
        # Grouped expert kernels may touch CUDA in a forked CPU worker.
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    return model


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
def test_ulysses_minimax_gradients_match_one_rank(kind):
    _spawn(_parity_worker, 2, kind, False)


def test_shard_local_lightning_is_wrong():
    # Negative control: the stock Lightning forward on a shard restarts its decayed state at the seam.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, "m1", True)


def test_minimax_gates():
    model = _model("m1")
    lightning = model.model.layers[0].self_attn
    assert uh.is_ulysses_lightning(lightning) and uh.is_ulysses_mixer(lightning)
    assert cp._unsupported_attention(model.config, "ulysses") is None
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    cp._check_stateful_mixers(model, "ulysses", 2)
    with pytest.raises(ValueError, match = "num_attention_heads"):
        cp._check_stateful_mixers(model, "ulysses", 8)


@pytest.mark.skipif("m2" not in KINDS, reason = "transformers without MiniMax-M2")
def test_minimax_m2_runs_ulysses_without_a_stateful_mixer():
    model = _model("m2")
    assert not cp._stateful_mixers(model)
    assert cp._supports_context_parallel(model, "ulysses")
    model.config.output_router_logits = True
    assert not cp._supports_context_parallel(model, "ulysses")


def test_lightning_refuses_packing_and_cache():
    lightning = _model("m1").model.layers[0].self_attn
    h = torch.zeros(1, 4, 32)
    with pytest.raises(NotImplementedError, match = "packing"):
        uh.ulysses_lightning_forward(lightning, h, packed_seq_lengths = torch.tensor([4]), group = None)
    with pytest.raises(RuntimeError, match = "cache"):
        uh.ulysses_lightning_forward(lightning, h, past_key_values = object(), group = None)
