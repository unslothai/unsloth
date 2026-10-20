# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 on a sub-mesh (gloo world of 4, two disjoint 2-rank meshes [0, 1] and [2, 3], each
sharding its own model replica on its own data):

* a trainable parameter left replicated (ignored_params) gets its gradient averaged over its own
  mesh's two ranks, not over all four (the other pair's data would leak in); on a 2 x 2 HSDP mesh
  over all four ranks, over all four;
* the per-input media agreement counts only the mesh's ranks: a mixed batch in one pair raises
  there, while the other pair, whose ranks agree, trains."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _lora  # noqa: E402
from test_fsdp2_native_per_input_gloo import _MoshiLike  # noqa: E402

WORLD = 4


def _submesh(rank, whole = False):
    from torch.distributed.device_mesh import DeviceMesh
    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names = ("pair", "shard"))
    return mesh if whole else mesh["shard"]


def _case_grads(
    native,
    rank,
    hsdp = False,
):
    import torch.distributed as dist

    mesh = _submesh(rank, whole = hsdp)
    model = _lora(1)
    reference = _lora(1)
    name, first = next((n, p) for n, p in model.named_parameters() if "lora_A" in n)
    cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first}, mesh = mesh)
    model = native.shard_model(model, cfg)
    ids = torch.randint(1, 100, (1, 8), generator = torch.Generator().manual_seed(rank))
    model(input_ids = ids, labels = ids).loss.backward()
    reference(input_ids = ids, labels = ids).loss.backward()
    local = dict(reference.named_parameters())[name].grad.detach().clone()
    every = [torch.zeros_like(local) for _ in range(WORLD)]
    dist.all_gather(every, local)  # the test's own view of every rank's unreduced gradient
    pair = [0, 1] if rank < 2 else [2, 3]
    want_pair = sum(every[r] for r in pair) / 2
    want_world = sum(every) / WORLD
    got = dict(model.named_parameters())[name].grad
    return {
        "mesh_ranks": mesh.mesh.tolist(),
        "pair_diff": (got - want_pair).abs().max().item(),
        "world_diff": (got - want_world).abs().max().item(),
    }


def _case_media(rank):
    from torch.distributed.fsdp import fully_shard

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = _submesh(rank)
    model = _MoshiLike()
    fully_shard(model.audio_encoder.proj, mesh = mesh)
    fully_shard(model, mesh = mesh)
    vision.install_vision_dummy(model)
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    audio = lambda: torch.randn(1, 6, 3)  # noqa: E731
    # Pair [0, 1]: both ranks user audio only. Pair [2, 3]: rank 2 has both inputs, rank 3 one.
    kwargs = {"user_input_values": audio()}
    if rank == 2:
        kwargs["moshi_input_values"] = audio()
    try:
        model(ids, **kwargs).backward()
        return {"step": "ok"}
    except Exception as exc:
        return {"step": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _worker(rank, world, port, out, case):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        if case == "media":
            res = _case_media(rank)
        else:
            res = _case_grads(native, rank, hsdp = case == "hsdp")
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(case, timeout = 180):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (WORLD, _free_port(), out, case),
        nprocs = WORLD,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"{case}: the ranks did not finish within {timeout} s (hung collective)"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == list(range(WORLD)), results
    for rank, r in results.items():
        print("ROUND11_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_replicated_grads_average_over_the_model_mesh_only():
    pytest.importorskip("peft")
    for rank, r in _run("grads").items():
        assert r["mesh_ranks"] == ([0, 1] if rank < 2 else [2, 3]), (rank, r)
        # The two pairs' data differ, so the world average is measurably another value.
        assert r["world_diff"] > 1e-6 and r["pair_diff"] < r["world_diff"] / 100, (rank, r)


def test_media_agreement_counts_only_the_model_mesh():
    results = _run("media")
    assert results[0]["step"] == "ok" and results[1]["step"] == "ok", results
    for rank in (2, 3):
        r = results[rank]
        assert r["step"].startswith("RuntimeError") and "on 1 of 2 ranks" in r["step"], (rank, r)


def test_replicated_grads_average_over_a_whole_hsdp_mesh():
    pytest.importorskip("peft")
    for rank, r in _run("hsdp").items():
        assert r["mesh_ranks"] == [[0, 1], [2, 3]], (rank, r)
        assert r["pair_diff"] > 1e-6 and r["world_diff"] < r["pair_diff"] / 100, (rank, r)
