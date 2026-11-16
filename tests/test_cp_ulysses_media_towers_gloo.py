# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses CP on a Gemma 4 checkpoint that ships vision and audio towers (Gemma-4-E2B / E4B): the towers'
attention layers are also named self_attn, but CP batches are text only, so they must not decide the attention
route. Two gloo CPU ranks vs one process on a tiny random config (sliding window crossing the shard seam, a
global layer with its own head dim, a KV-shared layer): the loss and every language model gradient."""

import importlib
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad

transformers = pytest.importorskip("transformers")
generic = importlib.import_module("tests.test_cp_ulysses_generic_gloo")

SEQ = 16


def _model():
    try:
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    except ImportError as e:
        pytest.skip(f"Gemma 4: {e}")
    text = dict(
        vocab_size = 96,
        vocab_size_per_layer_input = 96,
        hidden_size = 32,
        hidden_size_per_layer_input = 8,
        intermediate_size = 64,
        num_hidden_layers = 5,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        global_head_dim = 32,
        sliding_window = 5,
        layer_types = [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers = 2,
        use_cache = False,
        pad_token_id = 0,
    )
    vision = dict(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    audio = dict(hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 2)
    config = Gemma4Config(text_config = text, vision_config = vision, audio_config = audio)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(config)
    model.config.text_config._attn_implementation = "sdpa"
    return model.float().train()


def test_towers_do_not_decide_the_route():
    model = _model()
    assert cp._generic_hf_attention(model)
    cp._check_cp_model(model, 2, "ulysses")


def _worker(rank, world, out_dir):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model()
    ids = torch.randint(1, 96, (2, SEQ), generator = torch.Generator().manual_seed(1))
    labels = ids.clone()
    labels[labels % 5 == 0] = -100
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    generic._loss(model(input_ids = ids, use_cache = False).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)

    manager = generic._manager(world, rank)
    manager.attach_attention_hooks(model)
    inputs = {"input_ids": ids.clone(), "labels": labels.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        generic._loss(out.logits, inputs["shift_labels"], count).backward()
    assert inputs["input_ids"].shape[1] == SEQ // world
    checked = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or "language_model" not in name:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        dist.all_reduce(grad)
        want = reference.get(name, torch.zeros_like(param))
        # Scale-relative atol: tied / per-layer embedding rows sum many tokens' float32 contributions.
        atol = 1e-5 * max(1.0, float(want.abs().max()))
        # Keep torch's mismatch report (count, greatest abs / rel difference): a bare name hides how far off it is.
        torch.testing.assert_close(
            grad, want, rtol = 1e-4, atol = atol, msg = lambda report, name = name: f"{name}: {report}"
        )
        checked += 1
    assert checked > 10
    if rank == 0:
        with open(os.path.join(out_dir, "ok"), "w") as f:
            f.write(str(checked))


def test_gemma4_with_towers_matches_one_process(tmp_path):
    _model()  # skip before forking when Gemma 4 is unavailable
    generic._spawn(_worker, 2, str(tmp_path))
    assert (tmp_path / "ok").exists()
