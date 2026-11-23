# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism on Inkling 2 (inkling2_loop = Mini, inkling2 = Small), two gloo CPU ranks against
one process in float32: the loss and every LoRA (or, full finetuning, every parameter) gradient.

The tiny models keep what a sequence shard could break: a sliding window (5) and a full-layer relative bias extent
(7) longer than the gap to the shard seam, per-token sinks, log-scaled queries whose scale depends on the global
position (log_scaling_n_floor 4), short convolutions of kernel 4 on the keys, values and the residual stream (the
full layers' dynamic taps too), the loop layers run twice, and the MoE. Lengths 22 (11 per shard, not a multiple of
the kernel) and 21 (CP pads to 22), and right padding inside a shard. The control drops the convolution halo and
must miss the reference."""

import dataclasses
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
from unsloth.utils import ulysses_inkling2 as ui

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
except ImportError as e:  # transformers < 5.24
    pytest.skip(f"no Inkling 2 in this transformers: {e}", allow_module_level = True)

WINDOW = 5
VOCAB = 96


def _tiny(loop):
    from transformers import AutoModelForImageTextToText

    text = dict(
        vocab_size = VOCAB,
        unpadded_vocab_size = VOCAB,
        hidden_size = 32,
        num_hidden_layers = 4,
        layer_types = ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        n_layers_per_attn_res_block = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        swa_num_attention_heads = 4,
        swa_num_key_value_heads = 2,
        swa_head_dim = 8,
        d_rel = 4,
        rel_extent = 7,
        sliding_window_size = WINDOW,
        conv_kernel_size = 4,
        log_scaling_n_floor = 4,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 16,
        attention_dropout = 0.0,
        use_cache = False,
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
            loop_layer_index_start = 2, loop_layer_index_end = 4, loop_num_passes = 2, loop_gate_rank = 4
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(
        config, dtype = torch.float32, experts_implementation = "eager"
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_floating_point():
                noise = torch.randn(p.shape, generator = generator)
                p.copy_(
                    noise * (0.5 if p.ndim == 1 else 0.2)
                    + (1.0 if p.ndim == 1 and "norm" in name else 0.0)
                )
    return model.float().train()


def _lora(model):
    from peft import LoraConfig, get_peft_model

    targets = r".*language_model.*\.(linear|gate_proj|sink_proj|weight_proj)$"
    model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 4, lora_dropout = 0.0, target_modules = targets)
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (
                "lora_B" in name
            ):  # zero-initialised: give every adapter a gradient path through B too
                p.copy_(torch.randn(p.shape, generator = generator) * 0.1)
    return model


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.packed_seq_lengths = None
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
        )
        / count
    )


def _batch(case):
    seq = 21 if case == "odd" else 22
    ids = torch.randint(3, VOCAB, (2, seq), generator = torch.Generator().manual_seed(seq))
    labels = ids.clone()
    labels[labels % 7 == 0] = -100
    mask = None
    if case == "padded":
        mask = torch.ones_like(ids)
        mask[1, -6:] = 0  # right padding that starts inside rank 1's shard
        labels[mask == 0] = -100
    return ids, labels, mask


def _rel(a, b):
    diff = float((a.double() - b.double()).norm())
    return 0.0 if diff == 0 else diff / max(float(b.double().norm()), 1e-30)


def _parity_worker(
    rank,
    world,
    loop,
    case,
    lora,
    flex,
    control,
    out_dir,
    gc = False,
):
    model = _tiny(loop)
    if gc:
        # Recompute in backward repeats the exchanges and the halo on every rank in the same order.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    if flex:
        from unsloth.models import inkling2_flex as fx

        from torch.nn.attention import flex_attention as flex_mod

        fx._ALLOW_CPU = True
        flex_mod._validate_device = (
            lambda *a, **k: None
        )  # flex's reference math runs backward on CPU too
        assert fx.enable_inkling2_flex(model, verbose = False), fx._FAILED
    if lora:
        model = _lora(model)
    assert cp._inkling2_model(model) and cp._supports_context_parallel(model, "ulysses")
    cp._check_cp_model(model, world, "ulysses")
    with pytest.raises(NotImplementedError):
        cp._check_cp_model(model, world, "ring")

    ids, labels, mask = _batch(case)
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    ref_loss = _loss(
        model(input_ids = ids, attention_mask = mask, use_cache = False).logits, shift, count
    )
    ref_loss.backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)

    calls = {"attention": 0, "conv": 0, "lengths": set()}
    attention, conv, halo = (
        ui.ulysses_inkling2_attention_forward,
        ui.ulysses_inkling2_conv_forward,
        ui.previous_rank_halo,
    )

    def counted_attention(module, hidden_states, *args, **kwargs):
        calls["attention"] += 1
        calls["lengths"].add(int(hidden_states.shape[1]))
        return attention(module, hidden_states, *args, **kwargs)

    def counted_conv(*args, **kwargs):
        calls["conv"] += 1
        return conv(*args, **kwargs)

    ui.ulysses_inkling2_attention_forward = counted_attention
    ui.ulysses_inkling2_conv_forward = counted_conv
    if control == "no_halo":
        ui.previous_rank_halo = lambda tail, group: torch.zeros_like(tail)
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    inputs = {"input_ids": ids.clone(), "labels": labels.clone()}
    if mask is not None:
        inputs["attention_mask"] = mask.clone()
    manager._prepare_inputs(inputs)
    try:
        with manager.apply(inputs):
            out = model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs.get("attention_mask"),
                position_ids = inputs["position_ids"],
                use_cache = False,
            )
            loss = _loss(out.logits, inputs["shift_labels"], count)
            loss.backward()
    finally:
        ui.ulysses_inkling2_attention_forward = attention
        ui.ulysses_inkling2_conv_forward = conv
        ui.previous_rank_halo = halo
    loss = loss.detach()
    dist.all_reduce(loss)

    text = model.get_base_model() if lora else model
    layers = text.config.text_config.num_hidden_layers
    if loop:
        cfg = text.config.text_config
        layers += (cfg.loop_layer_index_end - cfg.loop_layer_index_start) * (
            cfg.loop_num_passes - 1
        )
    # Per layer call: one attention, four convolutions (k, v, after attention, after the MoE); recomputed once
    # more under gradient checkpointing.
    layers *= 2 if gc else 1
    assert calls["attention"] == layers, calls
    assert calls["conv"] == 4 * layers, calls
    assert calls["lengths"] == {11}, calls
    assert inputs["input_ids"].shape[1] == 11

    errors, unused = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        dist.all_reduce(grad)
        want = reference.get(name)
        if want is None:
            # No gradient in one process (the vision tower of a text batch): none under CP either.
            assert not grad.any(), name
            unused.append(name)
            continue
        errors.append((_rel(grad, want), name))
    errors.sort()
    loss_err = abs(float(loss) - float(ref_loss)) / abs(float(ref_loss))
    worst = errors[-1]
    if rank == 0 and out_dir:
        tag = f"{'loop' if loop else 'small'}_{case}_{'lora' if lora else 'full'}{'_flex' if flex else ''}"
        tag += "_gc" if gc else ""
        with open(os.path.join(out_dir, f"inkling2_{tag}_{control or 'ok'}.txt"), "w") as f:
            f.write(
                f"loss cp {float(loss):.8f} ref {float(ref_loss):.8f} rel {loss_err:.2e}; "
                f"{len(errors)} grads, worst rel {worst[0]:.2e} ({worst[1]}), median "
                f"{errors[len(errors) // 2][0]:.2e}; attention calls {calls['attention']}, conv calls "
                f"{calls['conv']}\n"
            )
    if control == "no_halo":
        assert worst[0] > 1e-3 or loss_err > 1e-4, (loss_err, worst)
        return
    assert all("vision" in n for n in unused), unused[:5]
    assert loss_err < 1e-5, (float(loss), float(ref_loss))
    assert worst[0] < 1e-4, errors[-3:]
    if lora:
        assert all("lora_" in n for _, n in errors) and len(errors) > 20


def _spawn(fn, world, *args):
    spawn = torch.cuda.is_available() or os.environ.get("UNSLOTH_TEST_GLOO_START") == "spawn"
    if not spawn and "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    hidden = {"CUDA_VISIBLE_DEVICES": "", "UNSLOTH_DISABLE_CPU_FALLBACK": None} if spawn else {}
    saved = {k: os.environ.get(k) for k in hidden}
    for k, v in hidden.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        with tempfile.TemporaryDirectory() as d:
            mp.start_processes(
                _entry,
                args = (world, os.path.join(d, "init"), fn, args),
                nprocs = world,
                join = True,
                start_method = "spawn" if spawn else "fork",
            )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


_OUT = os.environ.get("UNSLOTH_TEST_PARITY_OUT", "")


@pytest.mark.parametrize(
    "loop, case, lora, flex, gc",
    [
        (True, "even", True, False, False),
        (True, "odd", False, False, False),
        (True, "padded", True, False, False),
        (True, "odd", True, False, True),
        (False, "even", False, False, False),
        (False, "padded", True, False, False),
        (True, "even", True, True, False),
    ],
    ids = [
        "mini-22-lora",
        "mini-21-full",
        "mini-padded-lora",
        "mini-21-lora-gc",
        "small-22-full",
        "small-padded-lora",
        "mini-flex",
    ],
)
def test_inkling2_ulysses_matches_one_process(loop, case, lora, flex, gc):
    _spawn(_parity_worker, 2, loop, case, lora, flex, None, _OUT, gc)


def test_inkling2_ulysses_without_halo_misses_one_process():
    _spawn(_parity_worker, 2, True, "even", False, False, "no_halo", _OUT)


def test_inkling2_cp_capabilities():
    """Ulysses on Inkling 2 has a cell; ring stays refused; packed rows refuse (no cell, and the packing blocker)."""
    from unsloth import parallel_capabilities as pc

    q = pc.Query(
        trainer = "sft",
        family = "inkling2_loop",
        hybrid = True,
        multimodal = True,
        windowed = True,
        mode = "lora",
        cp_backend = "ulysses",
        cp = 2,
        attention = "eager",
        memory = "unsloth_offload",
        moe = True,
        torch = (2, 10),
        sm = 7,
    )
    pc.check(q)
    assert pc.cell_status(q)[0] == "supported"
    pc.check(dataclasses.replace(q, family = "inkling2", mode = "full", memory = "gc", attention = "sdpa"))
    with pytest.raises(NotImplementedError, match = "ring"):
        pc.check(dataclasses.replace(q, cp_backend = "ring", sm = 8))
    with pytest.raises(NotImplementedError, match = "composition"):
        pc.check(dataclasses.replace(q, packing = True))
    model = _tiny(True)
    assert cp.ulysses_packing_blocker(model) == "Inkling2LoopSlidingAttention"
    assert cp.ulysses_packing_blocker("x", ("inkling2_text",)) == "Inkling2SlidingAttention"
    assert cp._unsupported_attention(model.config, "ulysses") is None
    assert "hybrid" in cp._unsupported_attention(model.config, "ring")
