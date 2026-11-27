# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses CP keeps a compiled decoder layer dynamic in the shard length: two gloo CPU ranks run three batches
of different lengths through a layer compiled like the block compiler (_mark_seq_dynamic) and the
layer must compile once. The caller padding lookup keyed by (bsz, total) specialised the length."""

import importlib
import os

import pytest
import torch
import torch.distributed as dist

import unsloth  # noqa: F401
from unsloth import context_parallel as cp  # noqa: F401
from unsloth.models._block_compile import _mark_seq_dynamic
from unsloth.utils import attention_dispatch as ad

pytest.importorskip("transformers")
generic = importlib.import_module("tests.test_cp_ulysses_generic_gloo")

LENGTHS = (16, 12, 20)


def _model():
    from transformers import AutoModelForCausalLM, Qwen3Config

    config = Qwen3Config(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        use_cache = False,
        pad_token_id = 0,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation = "sdpa")
    return model.float().train()


def _worker(rank, world, out_dir):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model()
    manager = generic._manager(world, rank)
    manager.attach_attention_hooks(model)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(1)
        return gm.forward

    layer = model.model.layers[0]
    compiled = torch.compile(layer.forward, backend = backend)

    def forward(hidden_states, *args, **kwargs):
        _mark_seq_dynamic(hidden_states.shape[1], (hidden_states, args, kwargs))
        return compiled(hidden_states, *args, **kwargs)

    layer.forward = forward
    torch._dynamo.reset()
    for length in LENGTHS:
        ids = torch.randint(1, 96, (1, length), generator = torch.Generator().manual_seed(length))
        inputs = {"input_ids": ids, "labels": ids.clone()}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            assert inputs["input_ids"].shape[1] == length // world
            out = model(
                input_ids = inputs["input_ids"],
                position_ids = inputs["position_ids"],
                use_cache = False,
            )
            out.logits.float().sum().backward()
    with open(os.path.join(out_dir, f"graphs_{rank}.txt"), "w") as f:
        f.write(str(len(graphs)))


def test_ulysses_shard_length_stays_dynamic(tmp_path):
    generic._spawn(_worker, 2, str(tmp_path))
    counts = [int((tmp_path / f"graphs_{r}.txt").read_text()) for r in range(2)]
    assert counts == [1, 1], f"compiled layer graphs per rank over lengths {LENGTHS}: {counts}"
