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


def _q4(
    value,
    module,
    blocksize = 64,
):
    """A stacked expert tensor as zoo's Bnb4bitQuantize patch leaves it (nf4, double quantized)."""
    import bitsandbytes as bnb

    data, state = bnb.functional.quantize_4bit(
        value.detach().contiguous(), blocksize = blocksize, compress_statistics = True, quant_type = "nf4"
    )
    new = torch.Tensor._make_subclass(bnb.nn.Params4bit, data, False)
    new.quant_state = state
    new.blocksize = blocksize
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


# Per-stack blocksize of the on-load quantization, and a hook run after every tensor the load sets.
_ON_LOAD = {"blocksize": {}, "hook": None}


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
                blocksize = _ON_LOAD["blocksize"].get(name, 64)
                module._parameters[name] = _q4(p, module, blocksize)
        if _ON_LOAD["hook"] is not None:
            _ON_LOAD["hook"](model, target_name)

    loading.set_param_for_module = set_param
    return original


def _load_4bit(
    ckpt,
    skip_modules = None,
    **kwargs,
):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.float32,
            llm_int8_skip_modules = skip_modules,
        ),
        dtype = torch.float32,
        attn_implementation = "eager",
        device_map = {"": "cpu"},
        **kwargs,
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


def _setup(rank, tmp):
    """(tp_native, tp mesh, the saved Mini fake, a batch) with the on-load stack quantization installed."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    tpn = ink._load("tp_native")
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names = ("tp",))["tp"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = ink.build(True, lora = False, text_overrides = TEXT)
        source.generation_config.pad_token_id = None
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    _quantize_stacks_on_load()
    return tpn, mesh, ckpt, ink.batch()


def _apply_and_run(
    tpn,
    model,
    mesh,
    ids,
    labels,
    forward = True,
):
    """(MoE blocks sharded, apply error, one forward's loss)."""
    try:
        tpn.apply_tensor_parallel(model, mesh)
        moe = tpn._unwrap(model).__dict__["_unsloth_tp_plan"].sharded_moe
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:400], None
    if not forward:
        return moe, None, None
    with torch.no_grad():
        loss = float(model(input_ids = ids, labels = labels, use_cache = False).loss)
    return moe, None, loss


def _case_ineligible(rank, world, tmp):
    """Blocks apply keeps replicated must stay whole through the sharded load (both stacks, or neither)."""
    from transformers.models.inkling2_loop import modeling_inkling2_loop as mod

    tpn, mesh, ckpt, (ids, labels) = _setup(rank, tmp)
    out = {}

    def run(label, forward = True):
        with tpn.sharded_loading(mesh):
            model = _load_4bit(ckpt)
        stacks = model.__dict__["_unsloth_tp_load_stats"]["sharded_expert_stacks"]
        out[label] = (stacks,) + _apply_and_run(tpn, _lora(model), mesh, ids, labels, forward)

    # down_proj in 128-wide blocks: its 64-wide rows are not whole blocks, gate_up_proj's are.
    _ON_LOAD["blocksize"] = {"down_proj": 128}
    try:
        run("down_proj_blocks")
    finally:
        _ON_LOAD["blocksize"] = {}
    # A structural refusal of apply (swapped expert LoRA factors).
    mod.Inkling2LoopExperts._unsloth_expert_lora_factors_swapped = True
    try:
        run("swapped")
    finally:
        del mod.Inkling2LoopExperts._unsloth_expert_lora_factors_swapped

    # The experts module declares 8 experts; the stacks hold 4.
    def declare(model, target_name):
        if target_name.endswith(STACKS):
            model.get_submodule(target_name.rpartition(".")[0]).num_experts = 8

    _ON_LOAD["hook"] = declare
    try:
        run("num_experts", forward = False)
    finally:
        _ON_LOAD["hook"] = None
    return out


def _case_refusals(
    rank,
    world,
    tmp,
    only = None,
):
    """A refused apply changes nothing (the model stays as loaded and a later apply succeeds). `only`: run just
    that refusal."""
    tpn, mesh, ckpt, (ids, labels) = _setup(rank, tmp)
    with tpn.sharded_loading(mesh):
        model = _load_4bit(ckpt)
    model = _lora(model)
    tag = tpn._EXPERT_PRELOAD
    out = {"tp4": tpn.expert_shard_problem(tpn.derive_tp_plan(model, 2).moe_blocks[0], 4)}
    layout = lambda: {
        n: (id(p), type(p).__name__, tuple(p.shape)) for n, p in model.named_parameters()
    }  # noqa: E731
    before = layout()

    def refused(label):
        if only not in (None, label):
            return
        try:
            tpn.apply_tensor_parallel(model, mesh)
            message = None
        except ValueError as exc:
            message = str(exc)
        out[label] = (message, layout() == before, tpn.is_tensor_parallel(model))

    stacks = [p for e in _experts(model) for p in e._parameters.values() if getattr(p, tag, None)]
    tags = [getattr(p, tag) for p in stacks]
    for p, t in zip(stacks, tags):  # preloaded over the ranks in the other order
        setattr(p, tag, t[:2] + (tuple(reversed(t[2])),))
    refused("ranks")
    for p, t in zip(stacks, tags):
        setattr(p, tag, t)
    os.environ["UNSLOTH_TP_MOE_SHARD"] = "0"
    try:
        refused("moe_off")
    finally:
        del os.environ["UNSLOTH_TP_MOE_SHARD"]
    embed = model.get_input_embeddings().weight
    setattr(embed, tag, tags[0])  # an expert-preloaded tensor outside every MoE block of the plan
    refused("stray")
    delattr(embed, tag)
    out["retry"] = _apply_and_run(tpn, model, mesh, ids, labels)

    # Expert stacks preloaded, every projection 16-bit: a weight-dependent LoRA init is still refused.
    with tpn.sharded_loading(mesh):
        model = _load_4bit(ckpt, skip_modules = [".*"])
    stats = model.__dict__["_unsloth_tp_load_stats"]
    model = _lora(model)
    for config in model.peft_config.values():
        config.init_lora_weights = "pissa"
    out["pissa"] = (stats["sharded_weights"], stats["sharded_expert_stacks"]) + _apply_and_run(
        tpn, model, mesh, ids, labels, forward = False
    )
    return out


def _case_nested(rank, world, tmp):
    """The sharded load touches only its own model: not a from_pretrained nested in it, and stacked
    sharded_loading contexts shard once."""
    from transformers import AutoModelForImageTextToText

    tpn, mesh, ckpt, (ids, labels) = _setup(rank, tmp)
    out = {}
    nested = {}

    def load_inside(model, target_name):
        if not nested:
            nested["busy"] = True
            nested["model"] = AutoModelForImageTextToText.from_pretrained(
                ckpt, dtype = torch.float32, device_map = {"": "cpu"}
            )

    _ON_LOAD["hook"] = load_inside
    try:
        with tpn.sharded_loading(mesh):
            model = _load_4bit(ckpt)
    finally:
        _ON_LOAD["hook"] = None
    inner = nested["model"]
    out["inner"] = (
        sorted(
            {tuple(e._parameters[n].quant_state.shape)[0] for e in _experts(inner) for n in STACKS}
        ),
        sum("_unsloth_tp_load_guard" in m.__dict__ for m in inner.modules()),
    )
    stats = model.__dict__["_unsloth_tp_load_stats"]
    out["outer"] = (stats["sharded_weights"], stats["sharded_expert_stacks"])
    try:
        with tpn.sharded_loading(mesh):
            with tpn.sharded_loading(mesh):
                model = _load_4bit(ckpt)
        stats = model.__dict__["_unsloth_tp_load_stats"]
        out["stacked"] = (
            stats["sharded_weights"],
            stats["sharded_expert_stacks"],
        ) + _apply_and_run(tpn, _lora(model), mesh, ids, labels)
    except Exception as exc:
        out["stacked"] = f"{type(exc).__name__}: {exc}"[:400]
    try:
        with tpn.sharded_loading(mesh):
            _load_4bit(ckpt, tp_plan = "auto")
        out["tp_plan"] = None
    except Exception as exc:
        out["tp_plan"] = f"{type(exc).__name__}: {exc}"[:400]
    return out


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(  # a file rendezvous: no port to lose to another process
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 300),
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
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, tmp, case, kwargs)) for r in range(world)
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


def test_on_load_keeps_unshardable_experts_whole():
    results = _spawn("_case_ineligible")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        for label in ("down_proj_blocks", "swapped", "num_experts"):
            stacks, moe, error, loss = r[label]
            # Neither stack of any block cut at load; apply keeps the experts replicated.
            assert error is None and stacks == 0 and moe == 0, (label, r[label])
        for label in ("down_proj_blocks", "swapped"):
            assert r[label][3] == r[label][3], (label, r[label])  # a finite forward


def test_refused_apply_leaves_the_model_retryable():
    results = _spawn("_case_refusals")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["tp4"] is not None  # a tp 2 preload is not shardable over tp 4
        for label, match in (
            ("ranks", "(experts, tp, ranks)"),
            ("moe_off", "UNSLOTH_TP_MOE_SHARD=0"),
            ("stray", "not in this tensor parallel plan"),
        ):
            message, unchanged, applied = r[label]
            assert message is not None and match in message, (label, message)
            assert unchanged and not applied, (label, r[label])
        moe, error, loss = r["retry"]
        assert error is None and moe == 12 and loss == loss, r["retry"]
        weights, stacks, moe, error, _ = r["pissa"]
        assert weights == 0 and stacks == 24, r["pissa"]
        assert error is not None and "init_lora_weights" in error, r["pissa"]


def test_sharded_load_touches_only_its_own_model():
    results = _spawn("_case_nested")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["inner"] == ([4], 0), r["inner"]  # the nested load: whole stacks, no guards
        assert r["outer"][1] == 24, r["outer"]
        assert isinstance(r["stacked"], tuple), r["stacked"]
        weights, stacks, moe, error, loss = r["stacked"]
        assert (weights, stacks) == r["outer"] and error is None and moe == 12 and loss == loss, r[
            "stacked"
        ]
        assert r["tp_plan"] is not None and "tp_plan" in r["tp_plan"], r["tp_plan"]


def test_export_staging_charges_no_merge_for_expert_stacks():
    from types import SimpleNamespace

    tpn = ink._load("tp_native")
    rep = ink._load("tp_replica")
    stack = torch.zeros(4096, 1, dtype = torch.uint8)
    stack._unsloth_tp_layout = (tpn.EXPERT_STACK, (8,), 0, 2)
    stack.quant_state = SimpleNamespace(absmax = torch.zeros(128))
    items = [(stack, None)]
    # Gathered (3 x the whole stack), never dequantized and merged like a Linear4bit projection.
    assert rep._merge_workspace(items) == rep._workspace(items) == 3 * rep._bytes(stack)
