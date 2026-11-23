# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 under pipeline parallelism (models/pp_native.py + the Inkling2PipelineAdapter in models/inkling2.py),
2-rank gloo CPU world vs one process, fp32, LoRA on the attention Linears plus the routed experts (PEFT
target_parameters): losses and every LoRA gradient at every step match, for gpipe and 1f1b and for stage boundaries
before, at and after Inkling-2-Mini's looped layers (the block attention residual stack crosses the boundary with 1,
2 or 3 entries); a boundary inside the loop is refused before anything is cut."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_parallel_inkling2_gloo import EXPERTS, _load  # noqa: E402

STEPS = 3
LR = 0.05


def build(
    loop,
    layers = 12,
    loop_range = (6, 12),
    reentrant = None,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 32,
        num_hidden_layers = layers,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        swa_num_attention_heads = 2,
        swa_num_key_value_heads = 1,
        swa_head_dim = 16,
        d_rel = 4,
        rel_extent = 8,
        sliding_window_size = 4,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 32,
        layer_types = ["hybrid" if i % 6 == 5 else "hybrid_sliding" for i in range(layers)],
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
            loop_layer_index_start = loop_range[0],
            loop_layer_index_end = loop_range[1],
            loop_num_passes = 3,
            loop_gate_rank = 4,
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).float()
    with torch.no_grad():  # init leaves residual queries / gates at zero: give every path signal
        g = torch.Generator().manual_seed(0)
        for name, p in model.named_parameters():
            if (
                name.endswith(("query", "static_weight", "proj"))
                or "e_score_correction_bias" in name
            ):
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    model.config._attn_implementation = "eager"
    targets = [
        n
        for n, m in model.named_modules()
        if ".language_model.layers." in n
        and n.endswith(
            ("q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "o_proj.gate_proj")
        )
    ]
    lora = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = 0.0,
        target_modules = targets,
        target_parameters = EXPERTS,
        init_lora_weights = False,
    )
    torch.manual_seed(1)
    model = get_peft_model(model, lora)
    if reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": reentrant}
        )
        if reentrant:
            model.enable_input_require_grads()
        assert _load("inkling2").checkpoint_inkling2_message_inputs(model) == int(loop)
    model.train()
    return model


def batch(rows = 4, seq = 12):
    ids = torch.randint(1, 250, (rows, seq), generator = torch.Generator().manual_seed(3))
    labels = ids.clone()
    labels[
        0, :3
    ] = (
        -100
    )  # unequal token counts per micro-batch: the loss must be normalised over the whole batch
    return ids, labels


def _grads(model):
    return {
        n.replace(".default", ""): p.grad.detach().clone().double()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def reference(
    model,
    ids,
    labels,
    n_microbatches = 2,
):
    """One process, gradient accumulation over the pipeline's micro-batches, each loss summed over its tokens and
    divided by the token count of the whole batch (what PipelineParallel.step computes)."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = LR)
    items = (labels[:, 1:] != -100).sum()
    losses, grads = [], []
    for _ in range(STEPS):
        total = 0.0
        for mb_ids, mb_labels in zip(ids.chunk(n_microbatches), labels.chunk(n_microbatches)):
            out = model(
                input_ids = mb_ids, labels = mb_labels, use_cache = False, num_items_in_batch = items
            )
            out.loss.backward()
            total += float(out.loss.detach())
        grads.append(_grads(model))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(total)
    return losses, grads


def full_batch(model, ids, labels):
    """One forward over the whole batch: the micro-batch split changes only fp32 reduction order (the noise floor)."""
    out = model(input_ids = ids, labels = labels, use_cache = False)
    out.loss.backward()
    return float(out.loss.detach()), _grads(model)


def pipeline(model, pp, ids, labels):
    opt = torch.optim.SGD(pp.parameters(), lr = LR)
    losses, grads = [], []
    for _ in range(STEPS):
        loss = pp.step(ids, labels)
        grads.append(_grads(model))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return losses, grads


def _case(
    rank,
    world,
    loop,
    layers,
    loop_range,
    layouts,
    reentrant = None,
):
    from torch.distributed.device_mesh import init_device_mesh

    ppn = _load("pp_native")
    ink = _load("inkling2")
    ids, labels = batch()
    ref = reference(build(loop, layers, loop_range, reentrant), ids, labels)
    full = full_batch(build(loop, layers, loop_range, reentrant), ids, labels)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {"ref": ref, "full": full, "cells": {}}
    stage_calls = [0]
    layers_fn = ink.inkling2_stage_layers

    def counted(*args, **kwargs):
        stage_calls[0] += 1
        return layers_fn(*args, **kwargs)

    ink.inkling2_stage_layers = counted
    for sizes, schedule in layouts:
        model = build(loop, layers, loop_range, reentrant)
        key = f"{'-'.join(map(str, sizes)) if sizes else 'auto'}_{schedule}"
        try:
            pp = ppn.apply_pipeline_parallel(
                model,
                mesh,
                n_microbatches = 2,
                schedule = schedule,
                config = ppn.PPConfig(text_only = True, layers_per_stage = sizes),
            )
        except ValueError as exc:
            out["cells"][key] = {"refused": str(exc)}
            continue
        stage_calls[0] = 0
        losses, grads = pipeline(model, pp, ids, labels)
        frozen = sorted(
            n
            for n, p in model.named_parameters()
            if ("final_input" in n or "message_input" in n) and p.requires_grad
        )
        out["cells"][key] = {
            "losses": losses,
            "grads": grads,
            "bounds": pp.bounds,
            "stage_calls": stage_calls[0],
            "carry": [pp.ctx.layout(b) for b, _ in pp.bounds[1:]],
            "trainable_extra": frozen,
        }
    return out


def _worker(rank, world, port, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        res = _case(rank, world, **kwargs)
    except Exception as exc:
        import traceback
        res = {
            "error": f"{type(exc).__name__}: {exc}"[:1500] + "\n" + traceback.format_exc()[-3000:]
        }
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _spawn(world = 2, **kwargs):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, kwargs)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        for p in procs:
            if p.is_alive():
                p.kill()
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _check(results, key):
    ref_losses, ref_grads = results[0]["ref"]
    cells = [r["cells"][key] for r in results]
    assert all("refused" not in c for c in cells), cells[0].get("refused")
    worst = (0.0, "")
    for step, want in enumerate(ref_grads):
        got = {}
        for cell in cells:
            for name, g in cell["grads"][step].items():
                assert name not in got, f"{name} has a gradient on two stages"
                got[name] = g
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in n for n in got)
        for name, w in want.items():
            scale = float(w.abs().max())
            assert scale > 0, name
            worst = max(worst, (float((got[name] - w).abs().max()) / scale, name))
    assert worst[0] <= 1e-5, (key, worst)
    for cell in cells:
        assert cell["losses"] == pytest.approx(ref_losses, rel = 1e-5), (
            key,
            cell["losses"],
            ref_losses,
        )
        assert cell["stage_calls"] == STEPS * 2, (
            key,
            cell["stage_calls"],
        )  # the adapter ran every micro-batch
        assert not cell["trainable_extra"]
    assert len(set(ref_losses)) == len(ref_losses)  # the steps train
    # The reference itself is the model: one forward over the whole batch gives the same loss and gradients up to
    # fp32 reduction order.
    full_loss, full_grads = results[0]["full"]
    assert full_loss == pytest.approx(ref_losses[0], rel = 1e-5)
    noise = max(
        float((full_grads[n] - w).abs().max()) / float(w.abs().max())
        for n, w in ref_grads[0].items()
    )
    assert noise <= 1e-4, noise
    return worst, cells[0]["bounds"], cells[0]["carry"]


MINI_LAYOUTS = [
    (None, "gpipe"),  # balanced: the stage boundary at the loop start, block stack of 1
    (None, "1f1b"),
    ([3, 9], "1f1b"),  # before the loop, mid block
    ([8, 4], "gpipe"),  # inside the loop: refused
]
MINI18_LAYOUTS = [
    ([12, 6], "gpipe"),  # at the loop end: block stack of 2 (layers 0, 6)
    ([14, 4], "1f1b"),  # after the loop: block stack of 3 (layers 0, 6, 12)
]
SMALL_LAYOUTS = [([6, 6], "gpipe"), ([7, 5], "1f1b")]


def test_pp2_inkling2_mini_matches_one_process():
    results = _spawn(loop = True, layers = 12, loop_range = (6, 12), layouts = MINI_LAYOUTS)
    assert all(r is not None and "error" not in r for r in results), [
        r and r.get("error") for r in results
    ]
    worst, bounds, carry = _check(results, "auto_gpipe")
    assert bounds == [(0, 6), (6, 12)] and carry == [[((32,), None), ((1, 32), None)]], (
        bounds,
        carry,
    )
    print("mini auto gpipe", worst)
    print("mini auto 1f1b", _check(results, "auto_1f1b"))
    worst, bounds, carry = _check(results, "3-9_1f1b")
    assert bounds == [(0, 3), (3, 12)] and carry[0][1] == ((1, 32), None)
    refused = results[0]["cells"]["8-4_gpipe"]["refused"]
    assert "safe cut points" in refused and "[8]" in refused, refused


def test_pp2_inkling2_mini_cut_after_the_loop_matches_one_process():
    results = _spawn(loop = True, layers = 18, loop_range = (6, 12), layouts = MINI18_LAYOUTS)
    assert all(r is not None and "error" not in r for r in results), [
        r and r.get("error") for r in results
    ]
    worst, bounds, carry = _check(results, "12-6_gpipe")
    assert carry[0][1] == ((2, 32), None), carry
    print("mini18 12-6 gpipe", worst)
    worst, bounds, carry = _check(results, "14-4_1f1b")
    assert carry[0][1] == ((3, 32), None), carry
    print("mini18 14-4 1f1b", worst)


def test_pp2_inkling2_small_matches_one_process():
    results = _spawn(loop = False, layers = 12, loop_range = None, layouts = SMALL_LAYOUTS)
    assert all(r is not None and "error" not in r for r in results), [
        r and r.get("error") for r in results
    ]
    for key in ("6-6_gpipe", "7-5_1f1b"):
        print("small", key, _check(results, key))


def test_pp2_inkling2_mini_gradient_checkpointing_matches_one_process():
    # Non-reentrant checkpointing (transformers 5.x default) with Unsloth's checkpointed loop message residual.
    results = _spawn(
        loop = True, layers = 12, loop_range = (6, 12), layouts = [(None, "1f1b")], reentrant = False
    )
    assert all(r is not None and "error" not in r for r in results), [
        r and r.get("error") for r in results
    ]
    print("mini gc 1f1b", _check(results, "auto_1f1b"))


def test_adapter_cut_points_and_carry():
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    ink = _load("inkling2")
    real = Inkling2LoopTextConfig(
        num_hidden_layers = 24,
        n_layers_per_attn_res_block = 6,
        loop_layer_index_start = 6,
        loop_layer_index_end = 18,
        loop_num_passes = 3,
        hidden_size = 2304,
    )
    holder = torch.nn.Module()
    holder.config = real
    holder.message_input = torch.nn.Module()
    adapter = ink.Inkling2PipelineAdapter(holder)
    assert adapter.cut_points() == [1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23]
    assert [adapter.blocks_before(c) for c in (1, 6, 7, 18, 19, 23)] == [1, 1, 2, 3, 4, 4]
    assert adapter.carry(18) == [((2304,), None), ((3, 2304), None)]
    assert adapter.owner_layer("message_input") == 6 and adapter.owner_layer("final_input") is None
