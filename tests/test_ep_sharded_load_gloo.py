# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""expert_parallel_loading of a 16-bit MoE checkpoint (gloo CPU, 2 ranks, fp32, transformers 5)."""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

pytest.importorskip("transformers.core_model_loading")

REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
STACKS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
_SEQ = (7, 5)


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


def _load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype = torch.float32,
        attn_implementation = "eager",
        experts_implementation = "eager",
        device_map = {"": "cpu"},
    )


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = "all-linear",
        target_parameters = list(STACKS),
        init_lora_weights = False,
    )
    return get_peft_model(model, config).train()


def _step(epn, model, rank, world, mesh, vocab):
    from torch.nn.parallel import DistributedDataParallel

    import torch.distributed as dist

    rows = []
    for r in range(world):
        ids = torch.randint(0, vocab, (2, _SEQ[r]), generator = torch.Generator().manual_seed(20 + r))
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
    ddp = DistributedDataParallel(
        model
    )  # alive through backward: its reducer averages the dense grads
    loss = loss_of(ddp, *rows[rank]) * world / n
    loss.backward()
    epn.ep_sync_expert_grads(model)
    value = loss.detach() / world
    dist.all_reduce(value)
    group = epn._mesh(model).get_group()
    grads = {
        k: (epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad)
        .detach()
        .clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }
    return float(value), grads


def _case_sharded_16bit(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = build(REPO, lora = False)
        source.generation_config.pad_token_id = None
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    out = {"ep_native": epn.__file__}

    full = _load(ckpt)
    vocab = tpn._text_config(full).vocab_size
    whole = {k: p.detach().clone() for k, p in full.named_parameters() if k.endswith(STACKS)}
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(ckpt)
    out["models"] = len(state["models"])
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    sliced, per = True, None
    for k, p in model.named_parameters():
        if k.endswith(STACKS):
            per = p.shape[0]
            sliced &= torch.equal(p, whole[k][rank * per : (rank + 1) * per])
    out["per_rank"], out["experts"], out["sliced"] = (
        per,
        tpn.moe_blocks_with_experts(full)[0].experts.num_experts,
        sliced,
    )
    out["held"] = sum(p.numel() for k, p in model.named_parameters() if k.endswith(STACKS))
    out["whole"] = sum(p.numel() for p in whole.values())
    try:
        model(input_ids = torch.zeros(1, 3, dtype = torch.long))
        out["unapplied_forward"] = "ran"
    except RuntimeError as exc:
        out["unapplied_forward"] = str(exc)

    ref, model, full = _lora(_load(ckpt)), _lora(model), _lora(full)
    lora = lambda m: {k: p.detach().clone() for k, p in m.named_parameters() if "lora_" in k}  # noqa: E731
    want_lora = lora(full)
    got_lora = lora(model)
    out["lora_equal"] = set(want_lora) == set(got_lora) and all(
        torch.equal(want_lora[k], got_lora[k]) for k in want_lora
    )
    out["ref"] = _step(epn, ref, rank, world, None, vocab)
    out["full"] = _step(epn, full, rank, world, mesh, vocab)
    out["sharded"] = _step(epn, model, rank, world, mesh, vocab)
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
    except BaseException as exc:  # a rank must report, not vanish
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


def test_ep_sharded_16bit_load_keeps_only_local_experts_and_matches_full_load():
    results = _spawn("_case_sharded_16bit")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["models"] == 1, r
        assert r["per_rank"] * 2 == r["experts"] and r["sliced"], r
        assert r["held"] * 2 == r["whole"], (r["held"], r["whole"])
        assert (
            r["stats"]["sharded_expert_stacks"] > 0 and r["stats"]["quantized_on_load"] is False
        ), r["stats"]
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        # PEFT sized and seeded the expert LoRA from the whole stack, as on one GPU.
        assert r["lora_equal"]
        assert r["sharded"][0] == r["full"][0], (r["sharded"][0], r["full"][0])
        got, want = r["sharded"][1], r["full"][1]
        assert set(got) == set(want) and any(".experts." in k for k in got)
        for k in want:
            assert torch.equal(got[k], want[k]), k
        assert r["sharded"][0] == pytest.approx(r["ref"][0], rel = 1e-6)
        for k, w in r["ref"][1].items():
            torch.testing.assert_close(
                got[k], w, rtol = 1e-5, atol = 1e-6, msg = lambda m, k = k: f"{k}: {m}"
            )
