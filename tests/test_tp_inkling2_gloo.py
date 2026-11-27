# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 tensor parallel (gloo CPU, 2 ranks, fp32) vs one process: Inkling-2-Mini (inkling2_loop, layers
6-11 looped three times with message gates) and Inkling-2-Small (inkling2), LoRA on the attention wrapper
Linears (q / k / v / o .linear, o_proj.gate_proj) and on the routed experts (PEFT target_parameters).

The attention shards by heads through the wrapper projections: query / key / value / output gate / sink /
relative position bias rows, the per-head clip scales, the key / value short convolution channels (static and
dynamic), o_proj.linear row parallel; the relative bias profiles stay replicated. Routed experts shard by index
(UNSLOTH_TP_MOE_SHARD, default) or stay replicated. Also: bitsandbytes 4-bit attention Linears, full finetuning,
a fault injection (key / value inputs without their tp gradient reduce must break parity), the whole state dict
gathered back, and generate on the gathered replica."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from unsloth_models_loader import load_models_module  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
ATTENTION = ("q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "o_proj.gate_proj")


_load = load_models_module  # tests/_shared/unsloth_models_loader.py


def _bnb():
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def _quantize_attention(model):
    """Every Linear inside the language attention -> a bitsandbytes NF4 Linear4bit (same bytes in every process)."""
    import bitsandbytes as bnb

    for name, module in list(model.named_modules()):
        if ".language_model.layers." not in f".{name}." or not name.endswith("self_attn"):
            continue
        for child_name, child in list(module.named_modules()):
            if type(child) is not torch.nn.Linear:
                continue
            parent = (
                module.get_submodule(child_name.rsplit(".", 1)[0]) if "." in child_name else module
            )
            q = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias = False,
                compute_dtype = torch.float32,
                quant_type = "nf4",
            )
            packed, state = bnb.functional.quantize_4bit(
                child.weight.data, blocksize = 64, quant_type = "nf4"
            )
            q.weight = bnb.nn.Params4bit(
                packed,
                requires_grad = False,
                quant_state = state,
                blocksize = 64,
                quant_type = "nf4",
                module = q,
                bnb_quantized = True,
            )
            setattr(parent, child_name.rsplit(".", 1)[-1], q)
    return model


def build(
    loop,
    lora = True,
    four_bit = False,
    dtype = torch.float32,
    text_overrides = None,
):
    from transformers import AutoModelForImageTextToText
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    # 4 query / 2 key-value heads: tp 2 keeps 2 / 1 per rank (GQA grouping kept). 4-bit: head_dim 32, so the
    # row-parallel o_proj.linear reads 2 x 32 = 64 features per rank (one bitsandbytes block).
    head_dim = 32 if four_bit else 16
    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 64,
        num_hidden_layers = 12,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = head_dim,
        swa_num_attention_heads = 4,
        swa_num_key_value_heads = 2,
        swa_head_dim = head_dim,
        d_rel = 4,
        rel_extent = 8,
        sliding_window_size = 4,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 32,
    )
    text.update(text_overrides or {})
    vision = dict(
        model_type = "inkling2_vision",
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        projection_hidden_dim = 32,
        text_hidden_size = 64,
    )
    if loop:
        text.update(
            loop_layer_index_start = 6, loop_layer_index_end = 12, loop_num_passes = 3, loop_gate_rank = 4
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).to(dtype)
    with (
        torch.no_grad()
    ):  # init leaves residual queries / gates / per-head scales flat: give every path signal
        g = torch.Generator().manual_seed(0)
        for name, p in model.named_parameters():
            if name.endswith(("query", "static_weight", "proj", "conv1d.weight")) or (
                "e_score_correction_bias" in name
            ):
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
            elif name.endswith(".scale"):
                p.add_(
                    torch.randn(p.shape, generator = g) * 0.3
                )  # per head: a wrong head split shows
            elif name.endswith(("sink_proj.weight", "relative_position_bias.linear.weight")):
                p.normal_(0, 0.3, generator = g)
    model.config._attn_implementation = "eager"
    if four_bit:
        _quantize_attention(model)
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    targets = [
        n
        for n, m in model.named_modules()
        if ".language_model.layers." in n and n.endswith(ATTENTION)
    ]
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = 0.0,
        target_modules = targets,
        target_parameters = EXPERTS,
        init_lora_weights = False,
    )
    torch.manual_seed(1)
    return get_peft_model(model, config).to(dtype).train()


def batch(rows = 2, seq = 12):
    ids = torch.randint(1, 250, (rows, seq), generator = torch.Generator().manual_seed(3))
    labels = ids.clone()
    labels[0, :3] = -100
    return ids, labels


def _train(
    model,
    ids,
    labels,
    full = None,
    steps = 2,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.05)
    losses, grads, after = [], None, None
    for step in range(steps):
        loss = model(input_ids = ids, labels = labels, use_cache = False).loss
        loss.backward()
        if step == 0:
            grads = {
                n: (full(p.grad, p) if full else p.grad).detach().clone().double()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
        if step == 0:  # the update itself: every shard / replica moved as on one GPU
            after = {
                n: (full(p, p) if full else p).detach().clone().double()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
        losses.append(float(loss.detach()))
    return {"losses": losses, "grads": grads, "after": after}


def _case(
    rank,
    world,
    loop,
    lora = True,
    four_bit = False,
    moe_shard = True,
    fault = False,
    generate = False,
    dtype = torch.float32,
):
    from torch.distributed.device_mesh import init_device_mesh

    os.environ["UNSLOTH_TP_MOE_SHARD"] = "1" if moe_shard else "0"
    tpn = _load("tp_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    ids, labels = batch()
    ref_model = build(loop, lora, four_bit, dtype)
    ref = _train(ref_model, ids, labels)
    model = build(loop, lora, four_bit, dtype)
    whole = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    tpn.apply_tensor_parallel(model, mesh)
    plan = model.__dict__.get("_unsloth_tp_plan") or tpn._unwrap(model).__dict__["_unsloth_tp_plan"]
    attn = [r.module for r in plan.regions]
    local = {
        "heads": sorted({m.num_heads for m in attn}),
        "kv_heads": sorted({m.num_key_value_heads for m in attn}),
        "bias_heads": sorted({m.relative_position_bias.num_heads for m in attn}),
        "sharded": sorted(
            {
                n.split("self_attn.", 1)[1]
                for n, m in model.named_modules()
                if "self_attn." in n and m.__dict__.get("_unsloth_tp_style") is not None
            }
        ),
    }
    state = tpn.tp_full_state_dict(model)
    state_equal = all(
        torch.equal(state[n].float(), w.float())
        for n, w in whole.items()
        if n in state and not tpn._is_params4bit(w)
    )
    original = tpn._copy_extra_inputs
    if fault:
        tpn._copy_extra_inputs = lambda kwargs, names, group: kwargs
    try:
        got = _train(model, ids, labels, full = tpn._full)
    finally:
        tpn._copy_extra_inputs = original
    out = {
        "ref": ref,
        "tp": got,
        "regions": len(plan.regions),
        "moe_sharded": plan.sharded_moe,
        "moe_blocks": plan.replicated_moe,
        "local": local,
        "state_equal": state_equal,
    }
    if generate:
        kw = dict(input_ids = ids[:1, :5], max_new_tokens = 4, do_sample = False, pad_token_id = 0)
        with torch.no_grad():
            out["generated"] = (ref_model.generate(**kw), model.generate(**kw))
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
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-1500:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


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
        deadline = time.monotonic() + 600
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _worst(
    ref,
    got,
    norm = True,
):
    """Worst relative gradient error over the parameters: ||diff|| / ||ref|| per tensor (norm), or
    max |diff| / max |ref| per tensor."""
    assert set(got) == set(ref), sorted(set(got) ^ set(ref))[:6]
    worst = 0.0
    for name, want in ref.items():
        scale = float(want.norm() if norm else want.abs().max())
        diff = got[name] - want
        if scale == 0:  # a parameter with no gradient on one GPU must get none here either
            assert float(diff.abs().max()) == 0, name
            continue
        worst = max(worst, float(diff.norm() if norm else diff.abs().max()) / scale)
    return worst


def _check(ref, got, tol):
    # Step-1 loss bit-for-bit up to fp32 reduction order, step-0 gradients and the parameters after one SGD
    # step within tol. The step-2 loss only loosely: an fp32 rounding difference after the update can flip a
    # top-k expert choice on these tiny random models (lr 1e-3 makes them equal again).
    assert got["losses"][0] == pytest.approx(ref["losses"][0], rel = 1e-6, abs = 1e-6)
    assert got["losses"][1] == pytest.approx(ref["losses"][1], rel = 1e-3)
    assert _worst(ref["grads"], got["grads"]) <= tol
    assert _worst(ref["after"], got["after"]) <= tol * 1e-1


_SHARDED = sorted(
    [
        "q_proj.linear",
        "k_proj.linear",
        "v_proj.linear",
        "o_proj.linear",
        "o_proj.gate_proj",
        "sink_proj",
        "relative_position_bias.linear",
    ]
)


@pytest.mark.parametrize("moe_shard", [True, False], ids = ["experts_sharded", "experts_replicated"])
@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
def test_tp_lora_matches_one_process(loop, moe_shard):
    results = _spawn(loop = loop, moe_shard = moe_shard)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["regions"] == 12, r["regions"]
        assert r["local"]["heads"] == [2] and r["local"]["kv_heads"] == [1], r["local"]
        assert r["local"]["bias_heads"] == [2], r["local"]
        # The dynamic short convolutions' weight_proj only exists in the full attention layers.
        assert set(_SHARDED) <= set(r["local"]["sharded"]), r["local"]["sharded"]
        assert any("sconv.weight_proj" in n for n in r["local"]["sharded"]), r["local"]["sharded"]
        assert r["moe_sharded"] == (12 if moe_shard else 0), r["moe_sharded"]
        assert r["state_equal"]
        _check(r["ref"], r["tp"], 1e-5)
        assert any(".experts." in n for n in r["ref"]["grads"])
        assert any(".q_proj.linear." in n for n in r["ref"]["grads"])


def test_tp_full_finetune_matches_one_process():
    results = _spawn(loop = True, lora = False)
    assert all(r is not None and "error" not in r for r in results), results
    r = results[0]
    assert r["moe_sharded"] == 12
    _check(r["ref"], r["tp"], 1e-5)
    grads = r["ref"]["grads"]
    # Per-head scales, short convolution channels, sinks and the replicated bias profiles all train.
    for leaf in (
        "q_proj.norm.scale",
        "sconv.static_weight",
        "sconv.conv1d.weight",
        "relative_position_bias.proj",
    ):
        assert any(n.endswith(leaf) for n in grads), leaf


@pytest.mark.skipif(not _bnb(), reason = "bitsandbytes not installed")
def test_tp_four_bit_attention_matches_one_process():
    results = _spawn(loop = True, four_bit = True)
    assert all(r is not None and "error" not in r for r in results), results
    r = results[0]
    assert r["regions"] == 12
    _check(r["ref"], r["tp"], 1e-4)


def test_tp_fault_injection_breaks_parity():
    # Without the gradient all-reduce on key_hidden_states / value_hidden_states, every rank's earlier layers see
    # only its own heads' share of the key / value input gradient.
    results = _spawn(loop = True, fault = True)
    assert all(r is not None and "error" not in r for r in results), results
    r = results[0]
    assert _worst(r["ref"]["grads"], r["tp"]["grads"]) > 1e-2


def test_tp_generate_on_the_gathered_replica():
    results = _spawn(loop = True, generate = True)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        want, got = r["generated"]
        assert torch.equal(want, got), (want, got)
