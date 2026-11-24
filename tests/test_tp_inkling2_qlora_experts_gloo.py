# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 QLoRA under tensor parallel (gloo CPU, 2 ranks, fp32) with bitsandbytes 4-bit routed experts.

- The 4-bit expert stacks (zoo moe_utils_bnb4bit layout: packed [n, 1] bytes, expert-major, logical shape in
  _original_shape) shard by expert index like 16-bit ones: each rank keeps the bytes of its E / tp experts, the
  partial expert outputs are summed over tp. Losses, LoRA grads and the post-step parameters match one process,
  and generate on the gathered replica (the whole stacks regathered) matches one process.
- A 4-bit quantize-on-load under sharded_loading keeps only this rank's shards of the 4-bit attention
  projections and expert stacks as transformers sets them: bitwise the shards apply_tensor_parallel cuts from a
  full load, about half the 4-bit bytes per rank, and the same training losses.

The experts forward here dequantizes the 4-bit stacks and runs transformers' own forward (zoo's bnb 4-bit
backend does the same on a GPU); a set_param_for_module wrapper quantizes the 3-D stacks on load as zoo's
Bnb4bitQuantize patch does."""

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

pytest.importorskip("bitsandbytes")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_tp_inkling2_gloo as ink  # noqa: E402

STACKS = ("gate_up_proj", "down_proj")
# Expert rows of whole 64-wide bitsandbytes blocks (hidden 64, expert intermediate 64), as in real checkpoints;
# head_dim 32: the row-parallel o_proj.linear reads one 64-wide block per rank.
TEXT = {"moe_intermediate_size": 64, "head_dim": 32, "swa_head_dim": 32}


def _q4(value, module):
    """A stacked expert tensor as zoo's Bnb4bitQuantize patch leaves it (nf4, double quantized)."""
    import bitsandbytes as bnb

    data, state = bnb.functional.quantize_4bit(
        value.detach().contiguous(), blocksize = 64, compress_statistics = True, quant_type = "nf4"
    )
    new = torch.Tensor._make_subclass(bnb.nn.Params4bit, data, False)
    new.quant_state = state
    new.blocksize = 64
    new.compress_statistics = True
    new.quant_type = "nf4"
    new.quant_storage = torch.uint8
    new.bnb_quantized = True
    new.module = module
    new._original_shape = value.shape
    if hasattr(value, "_is_hf_initialized"):
        new._is_hf_initialized = value._is_hf_initialized
    return new


def _experts(model):
    return [m for m in model.modules() if type(m).__name__.endswith("Experts")]


def _patch_experts_forward(model):
    """Dequantize 4-bit stacks for the call, as zoo's forward_moe_backend_bnb4bit (W4A16), then the original."""
    import bitsandbytes as bnb

    cls = type(_experts(model)[0])
    if cls.__dict__.get("_test_4bit"):
        return
    prior = cls.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        saved = {}
        for name in STACKS:
            p = self._parameters[name]
            if type(p).__name__ == "Params4bit":
                saved[name] = p
                dense = bnb.functional.dequantize_4bit(p.data, p.quant_state)
                self._parameters[name] = dense.reshape(p._original_shape)
        try:
            return prior(self, hidden_states, top_k_index, top_k_weights)
        finally:
            self._parameters.update(saved)

    cls.forward = forward
    cls._test_4bit = True


def _lora(model):
    from peft import LoraConfig, get_peft_model

    targets = [
        n
        for n, m in model.named_modules()
        if ".language_model.layers." in n and n.endswith(ink.ATTENTION)
    ]
    config = LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, target_modules = targets, init_lora_weights = False
    )
    torch.manual_seed(1)
    return get_peft_model(model, config).train()


def _build(loop):
    model = ink.build(loop, lora = False, four_bit = True, text_overrides = TEXT)
    for experts in _experts(model):
        for name in STACKS:
            experts._parameters[name] = _q4(experts._parameters[name], experts)
    _patch_experts_forward(model)
    return _lora(model)


def _dense(p):
    import bitsandbytes as bnb
    return bnb.functional.dequantize_4bit(p.data, p.quant_state).reshape(p.quant_state.shape)


def _case_sharded(rank, world, tmp, loop):
    from torch.distributed.device_mesh import init_device_mesh

    tpn = ink._load("tp_native")
    ink._load("tp_replica")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    ids, labels = ink.batch()
    ref_model = _build(loop)
    ref = ink._train(ref_model, ids, labels)
    model = _build(loop)
    full = [{n: _dense(e._parameters[n]) for n in STACKS} for e in _experts(model)]
    tpn.apply_tensor_parallel(model, mesh)
    plan = tpn._unwrap(model).__dict__["_unsloth_tp_plan"]
    per = None
    sliced_equal = True
    for experts, want in zip(_experts(model), full):
        for name in STACKS:
            p = experts._parameters[name]
            per = p.quant_state.shape[0]
            lo = rank * per
            sliced_equal &= type(p).__name__ == "Params4bit" and torch.equal(
                _dense(p), want[name][lo : lo + per]
            )
    got = ink._train(model, ids, labels, full = tpn._full)
    kw = dict(input_ids = ids[:1, :5], max_new_tokens = 4, do_sample = False, pad_token_id = 0)
    with torch.no_grad():
        generated = (ref_model.generate(**kw), model.generate(**kw))
    return {
        "ref": ref,
        "tp": got,
        "moe_sharded": plan.sharded_moe,
        "per_rank": per,
        "sliced_equal": sliced_equal,
        "generated": generated,
    }


def _quantize_stacks_on_load():
    """Wrap set_param_for_module: a stacked 3-D expert tensor is quantized right after it is set (zoo's patch)."""
    from transformers import core_model_loading as loading

    original = loading.set_param_for_module

    def set_param(model, target_name, *args, **kwargs):
        original(model, target_name, *args, **kwargs)
        path, _, name = target_name.rpartition(".")
        if name in STACKS:
            module = model.get_submodule(path)
            p = module._parameters[name]
            if p.dim() == 3 and type(p).__name__ != "Params4bit":
                module._parameters[name] = _q4(p, module)

    loading.set_param_for_module = set_param
    return original


def _load_4bit(ckpt):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.float32,
        ),
        dtype = torch.float32,
        attn_implementation = "eager",
        device_map = {"": "cpu"},
    )
    _patch_experts_forward(model)
    return model


def _held_4bit(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if type(p).__name__ == "Params4bit"
    )


def _snapshot(model):
    """Every 4-bit parameter: (packed bytes, float32 absmax, QuantState shape)."""
    import bitsandbytes as bnb

    out = {}
    for name, p in model.named_parameters():
        if type(p).__name__ != "Params4bit":
            continue
        state = p.quant_state
        absmax = state.absmax
        if state.nested:
            absmax = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
        out[name] = (
            p.data.reshape(-1).view(torch.uint8).clone(),
            absmax.float().clone(),
            tuple(state.shape),
        )
    return out


def _case_on_load(rank, world, tmp, loop):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    tpn = ink._load("tp_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = ink.build(loop, lora = False, text_overrides = TEXT)
        source.generation_config.pad_token_id = None
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    _quantize_stacks_on_load()
    ids, labels = ink.batch()

    full = _load_4bit(ckpt)
    full_bytes = _held_4bit(full)
    full = _lora(full)
    tpn.apply_tensor_parallel(full, mesh)
    want = _snapshot(full)
    ref = ink._train(full, ids, labels, full = tpn._full)

    with tpn.sharded_loading(mesh) as state:
        model = _load_4bit(ckpt)
    stats = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    held = _held_4bit(model)
    refused = None
    try:
        model(input_ids = ids[:1, :3])
    except RuntimeError as exc:
        refused = str(exc)
    model = _lora(model)
    tpn.apply_tensor_parallel(model, mesh)
    have = _snapshot(model)
    diffs = sorted(
        n
        for n in set(want) | set(have)
        if n not in want
        or n not in have
        or want[n][2] != have[n][2]
        or not torch.equal(want[n][0], have[n][0])
        or not torch.equal(want[n][1], have[n][1])
    )
    plan = tpn._unwrap(model).__dict__["_unsloth_tp_plan"]
    got = ink._train(model, ids, labels, full = tpn._full)
    return {
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_bytes, held),
        "refused": refused,
        "diffs": diffs,
        "compared": len(want),
        "moe_sharded": plan.sharded_moe,
        "ref": ref,
        "tp": got,
    }


def _worker(rank, world, port, tmp, case, kwargs):
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
        res = globals()[case](rank, world, tmp, **kwargs)
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-6000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, case, kwargs))
            for r in range(world)
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


@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
def test_tp_four_bit_experts_shard_by_index(loop):
    results = _spawn("_case_sharded", loop = loop)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        # 12 MoE blocks, 2 of the 4 experts per rank, each rank's stacks bitwise its slice of the full ones.
        assert r["moe_sharded"] == 12, r["moe_sharded"]
        assert r["per_rank"] == 2
        assert r["sliced_equal"]
        ink._check(r["ref"], r["tp"], 1e-4)
        want, got = r["generated"]
        assert torch.equal(want, got), (want, got)


def test_tp_quantize_on_load_keeps_rank_shards():
    results = _spawn("_case_on_load", loop = True)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["models"] == 1
        stats = r["stats"]
        assert stats.get("quantized_on_load"), stats
        # 12 layers x (q / k / v / o / gate) column / row projections, 12 blocks x 2 expert stacks.
        assert stats["sharded_expert_stacks"] == 24, stats
        assert stats["sharded_weights"] >= 60, stats
        full_bytes, held = r["bytes"]
        assert held < 0.6 * full_bytes, r["bytes"]
        assert r["refused"] is not None and "apply_tensor_parallel" in r["refused"]
        assert r["compared"] > 0 and not r["diffs"], r["diffs"][:5]
        assert r["moe_sharded"] == 12
        assert r["tp"]["losses"] == r["ref"]["losses"]
