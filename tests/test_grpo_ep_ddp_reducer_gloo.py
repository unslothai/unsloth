# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""GRPO's manual DDP reducer (`_unsloth_average_gradients`, unsloth/models/rl.py) under expert parallel.

zoo's GRPO loss runs the unwrapped model, so DDP never reduces and Unsloth averages the gradients itself. Expert
parallel keeps every expert shard out of DDP (`_ddp_params_and_buffers_to_ignore`): each rank holds different
experts of the same shape and ep_native syncs them (/ world hook, replica sum). The reducer must skip them too,
else rank 0's experts are averaged with rank 1's. Expert LoRA plus LoRA on every Linear (so DDP wraps), gloo CPU,
fp32, each rank its own rows; gradients vs one process on all rows."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
REPO = "hf-internal-testing/tiny-random-OlmoeForCausalLM"


def _average_gradients():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_average_gradients"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), str(SOURCE), "exec"), namespace)
    return namespace["_unsloth_average_gradients"]


def _case(rank, world, ep):
    from test_ep_moe_families_gloo import _build, _modules, _rows, _sum_loss
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    if ep == world:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    else:
        mesh = init_device_mesh("cpu", (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
    ref = _build(build, REPO)
    vocab = tpn._text_config(ref).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}

    model = _build(build, REPO)
    epn.apply_expert_parallel(model, mesh)
    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    assert ddp.parameters_to_ignore, "expert parallel left no parameter out of DDP"
    # GRPO: the loss runs the unwrapped module (DDP's reducer never fires), then the manual average.
    (_sum_loss(model, *rows[rank]) * world / n).backward()
    epn.ep_sync_expert_grads(model)
    _average_gradients()(ddp)
    got = {
        k: (epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad)
        .detach()
        .clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }
    return {"got": got, "want": want}


def _worker(rank, world, tmp, ep):
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
        res = _case(rank, world, ep)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(world, ep):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, tmp, ep)) for r in range(world)]
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


def _fused_expert_layout():
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


@pytest.mark.skipif(
    not _fused_expert_layout(),
    reason = "expert parallel shards the fused expert stacks of transformers 5.x",
)
@pytest.mark.parametrize("world, ep", [(2, 2), (4, 2)], ids = ["ep2", "ep2x2replicas"])
def test_grpo_manual_reducer_leaves_expert_shards_to_expert_parallel(world, ep):
    results = _spawn(world, ep)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        got, want = r["got"], r["want"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        experts = [k for k in want if ".experts." in k and "lora_" in k]
        assert experts and len(experts) < len(want)  # expert and non-expert LoRA both train
        for k in want:
            torch.testing.assert_close(
                got[k], want[k], rtol = 1e-5, atol = 1e-6, msg = lambda m, k = k: f"{k}: {m}"
            )


def test_reducer_skips_ddp_ignored_parameters(monkeypatch):
    # Single process stand-in: whatever DDP ignores keeps this rank's gradient.
    import torch.distributed as dist
    from types import SimpleNamespace

    monkeypatch.setattr(dist, "get_world_size", lambda group = None: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda t, op = None, group = None: t.mul_(3))
    module = torch.nn.Module()
    module.dense = torch.nn.Parameter(torch.zeros(3))
    module.expert = torch.nn.Parameter(torch.zeros(3))
    module.dense.grad = torch.ones(3)
    module.expert.grad = torch.full((3,), 5.0)
    ddp = SimpleNamespace(process_group = None, module = module, parameters_to_ignore = {"expert"})
    _average_gradients()(ddp)
    assert torch.equal(module.dense.grad, torch.full((3,), 1.5))
    assert torch.equal(module.expert.grad, torch.full((3,), 5.0))
