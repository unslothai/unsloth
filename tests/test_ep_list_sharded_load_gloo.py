# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""expert_parallel_loading of per-expert ModuleList experts (the transformers 4.x MoE layout): each rank reads only
its experts from the checkpoint, other ranks' experts stay meta until apply_expert_parallel drops them. gloo CPU,
2 ranks, fp32: held bytes, a refused forward before apply, then LoRA + one step bitwise equal to the same step after
a full load (and close to one process)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

FAMILIES = ("qwen3_moe", "qwen2_moe", "mixtral", "deepseek_v3")


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


def _load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype = torch.float32, attn_implementation = "eager", device_map = {"": "cpu"}
    )


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    model = get_peft_model(model, config).train()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                value = (
                    torch.randn(p.shape, generator = g) * 0.02
                )  # drawn for meta ones too: same stream
                if not p.is_meta:
                    p.copy_(value)
    return model


def _held(model):
    dense = experts = 0
    for k, p in model.named_parameters():
        if p.is_meta or "lora_" in k:
            continue
        if ".experts." in k:
            experts += p.numel() * p.element_size()
        else:
            dense += p.numel() * p.element_size()
    return dense, experts


def _step(epn, model, rank, world, mesh, vocab):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    rows = []
    for r in range(world):
        ids = torch.randint(
            0, vocab, (2, (7, 5)[r]), generator = torch.Generator().manual_seed(20 + r)
        )
        rows.append((ids, ids.clone()))
    n = sum(int(lab[:, 1:].numel()) for _, lab in rows)

    def loss_of(m, ids, labels):
        logits = m(input_ids = ids, use_cache = False).logits
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), reduction = "sum"
        )

    if mesh is None:
        for ids, lab in rows:
            (loss_of(model, ids, lab) / n).backward()
        value = sum(float(loss_of(model, i, lab).detach()) for i, lab in rows) / n
        return value, {
            k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None
        }
    epn.apply_expert_parallel(model, mesh)
    ddp = DistributedDataParallel(model)
    loss = loss_of(ddp, *rows[rank]) * world / n
    loss.backward()
    epn.ep_sync_expert_grads(model)
    value = loss.detach() / world
    dist.all_reduce(value)
    lists = epn._lists()
    mine = {
        k: p.grad.detach().clone()
        for k, p in model.named_parameters()
        if p.grad is not None and getattr(p, epn._EP, (None,))[0] == lists.LIST
    }
    parts = [None] * world
    dist.all_gather_object(parts, mine)
    grads = {
        k: p.grad.detach().clone()
        for k, p in model.named_parameters()
        if p.grad is not None and k not in mine
    }
    for part in parts:
        grads.update(part)
    return float(value), grads


def _case(
    rank,
    world,
    tmp,
    family,
    parallel = False,
):
    import torch.distributed as dist
    from test_ep_expert_lists_gloo import _spec
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    lists = epn._lists()
    spec = _spec(family)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = build(spec["repo"], overrides = spec.get("overrides"), lora = False)
        source.generation_config.pad_token_id = None
        source.config.__dict__.pop(
            "quantization_config", None
        )  # deepseek_v3's tiny config names FP8 (GPU only)
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    out = {"ep_native": epn.__file__}
    full = _load(ckpt)
    blocks = tpn.moe_blocks_with_experts(full)
    if not blocks or lists.list_shard_problem(blocks[0], world) is not None:
        return {"skip": "no per-expert ModuleList experts here"}
    vocab = tpn._text_config(full).vocab_size
    if parallel:  # transformers 4.x loads shards on a thread pool
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(ckpt)
    os.environ.pop("HF_ENABLE_PARALLEL_LOADING", None)
    out["models"] = len(state["models"])
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    out["held"], out["full"] = _held(model), _held(full)
    remote = getattr(lists, "is_remote", lambda p: False)
    out["remote"] = sum(1 for p in model.parameters() if remote(p))
    whole = dict(full.named_parameters())
    out["local_equal"] = all(
        torch.equal(p, whole[k]) for k, p in model.named_parameters() if not p.is_meta
    )
    try:
        model(input_ids = torch.zeros(1, 3, dtype = torch.long))
        out["unapplied_forward"] = "ran"
    except RuntimeError as exc:
        out["unapplied_forward"] = str(exc)
    ref = _lora(_load(ckpt))
    model, full = _lora(model), _lora(full)
    out["ref"] = _step(epn, ref, rank, world, None, vocab)
    out["full_step"] = _step(epn, full, rank, world, mesh, vocab)
    out["sharded"] = _step(epn, model, rank, world, mesh, vocab)
    out["left_remote"] = sum(1 for p in model.parameters() if remote(p) or p.is_meta)
    out["calls"] = lists.list_blocks(model)[0].__dict__[lists._SHARD].calls
    return out


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 300),
    )
    try:
        res = globals()[case](rank, world, tmp, **kwargs)
    except BaseException as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
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


@pytest.mark.parametrize("family, parallel", [(f, False) for f in FAMILIES] + [("qwen3_moe", True)])
def test_ep_list_load_reads_only_local_experts_and_matches_full_load(family, parallel):
    results = _spawn("_case", family = family, parallel = parallel)
    assert all(r is not None and "error" not in r for r in results), results
    if any("skip" in r for r in results):
        pytest.skip(results[0]["skip"])
    for r in results:
        dense, experts = r["held"]
        full_dense, full_experts = r["full"]
        # Every dense weight, half of the routed experts (the shared experts are dense here: not a ModuleList).
        assert dense == full_dense and experts * 2 == full_experts, (r["held"], r["full"])
        # Other ranks' expert tensors were never read: dropped before the load, none freed after it.
        assert r["models"] == 1 and r["stats"]["remote_expert_tensors"] > 0, r["stats"]
        assert r["stats"]["remote_loaded_then_freed"] == 0, r["stats"]
        assert r["remote"] > 0 and r["local_equal"], r
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        assert r["left_remote"] == 0 and r["calls"] >= 1, r
        assert r["sharded"][0] == r["full_step"][0], (r["sharded"][0], r["full_step"][0])
        got, want = r["sharded"][1], r["full_step"][1]
        assert set(got) == set(want) and any(".experts." in k and "lora_" in k for k in got)
        for k in want:
            assert torch.equal(got[k], want[k]), k
        assert r["sharded"][0] == pytest.approx(r["ref"][0], rel = 1e-6)
        for k, w in r["ref"][1].items():
            torch.testing.assert_close(
                got[k], w, rtol = 1e-5, atol = 1e-6, msg = lambda m, k = k: f"{k}: {m}"
            )


def test_ep_list_load_device_map_listing_remote_experts_collapses_to_one_device():
    """accelerate's Trainer refuses a model whose hf_device_map has more than one entry (device_map="auto"); a map
    that only adds other ranks' experts on meta is one device."""
    from unsloth.models import tp_native

    model = torch.nn.Module()
    model.hf_device_map = {"": 0, "layers.0.mlp.experts.3": "meta"}
    tp_native._drop_remote_device_map(model, 2, 0)
    assert model.hf_device_map == {"": 0}
    split = torch.nn.Module()
    split.hf_device_map = {"layers.0": 0, "layers.1": 1}
    tp_native._drop_remote_device_map(split, 2, 0)
    assert split.hf_device_map == {"layers.0": 0, "layers.1": 1}


def test_ep_list_load_helpers_scope_devices_threads_and_quant_state():
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from unsloth.models import ep_expert_lists as lists, tp_native

    # CPU / disk offload keeps the full load: accelerate's hooks place tensors directly.
    assert (
        tp_native._one_device({"": 0})
        and tp_native._one_device({"": "cpu"})
        and tp_native._one_device(None)
    )
    assert not tp_native._one_device({"a": "cpu", "b": "disk"}) and not tp_native._one_device(
        "auto"
    )
    # Pool threads run in the submitting context: a nested ordinary load keeps its own owner token.
    var = contextvars.ContextVar("owner", default = None)
    pool = type("_ContextPool", (ThreadPoolExecutor,), {"submit": tp_native._submit_in_context})
    token = var.set("load")
    try:
        with pool(max_workers = 1) as executor:
            assert executor.submit(var.get).result() == "load"
    finally:
        var.reset(token)
    # A Linear4bit's own quant_state reference follows the stand-in, so remote buffers are freed.
    bnb = pytest.importorskip("bitsandbytes")
    linear = bnb.nn.Linear4bit(64, 32, compute_dtype = torch.float16)
    state = object()
    linear.weight.quant_state = state
    linear.quant_state = state
    lists.replace_remote(linear, "weight", (4, 2, 1, (0, 1)))
    assert linear.weight.is_meta and lists.is_remote(linear.weight)
    assert linear.quant_state is linear.weight.quant_state and linear.quant_state is not state
    # Stand-ins loaded for one ep layout refuse another before apply_expert_parallel changes anything.
    holder = torch.nn.Module()
    holder.linear = linear
    lists.check_remote_layout(holder, 2, 1, [0, 1])
    with pytest.raises(ValueError, match = "same ep mesh"):
        lists.check_remote_layout(holder, 2, 0, [0, 1])
    with pytest.raises(ValueError, match = "same ep mesh"):
        lists.check_remote_layout(holder, 4, 1, [0, 1, 2, 3])
