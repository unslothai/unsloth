# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 on a sub-mesh (gloo world of 4, two disjoint 2-rank meshes [0, 1] and [2, 3], each
sharding its own model replica on its own data):

* a trainable parameter left replicated (ignored_params) gets its gradient averaged over its own
  mesh's two ranks, not over all four (the other pair's data would leak in); on a 2 x 2 HSDP mesh
  over all four ranks, over all four;
* the per-input media agreement counts only the mesh's ranks: a mixed batch in one pair raises
  there, while the other pair, whose ranks agree, trains;
* the one-time layout agreement in shard_model runs over the mesh too: pairs whose GPUs differ pick
  their own fit layout, and one pair shards while the other never calls shard_model;
* collective and Trainer saves write from the mesh's own first rank (pair [2, 3] has no rank 0),
  also with every parameter left unsharded by a forward, both pairs saving at once."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import (  # noqa: E402
    _free_port,
    _model,
    _stub_unsloth_save,
    _trainer,
)
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


def _case_fit(native, rank):
    """Rank 3's GPU is too small for the fit layout, the others' are not: only pair [2, 3] falls back
    to the sharded layout."""
    perf = sys.modules["unsloth.models.fsdp2_perf"]
    perf._device_bytes = lambda device = None: 0 if rank == 3 else 1 << 40
    agreed = {}
    original = native._agree_over_ranks

    def record(*args, **kwargs):
        agreed["flags"] = original(*args, **kwargs)
        return agreed["flags"]

    native._agree_over_ranks = record
    mesh = _submesh(rank)
    native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32, mesh = mesh))
    return {"fits": agreed["flags"][0]}


def _case_subgroup(native, rank):
    """Only pair [0, 1] shards a model; ranks 2 and 3 never call shard_model."""
    import torch.distributed as dist

    mesh = _submesh(rank)
    res = {"sharded": False}
    if rank < 2:
        model = native.shard_model(
            _lora(1), native.FSDP2Config(param_dtype = torch.float32, mesh = mesh)
        )
        res["sharded"] = native.is_native_fsdp2(model)
    dist.barrier()
    return res


def _case_save(native, rank):
    from torch.distributed.fsdp import fully_shard

    save = sys.modules["unsloth.models.fsdp2_save"]
    mesh = _submesh(rank)
    torch.manual_seed(rank // 2)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    for block in model:
        fully_shard(block, mesh = mesh)
    fully_shard(model, mesh = mesh)
    res = {}
    with save.fsdp2_gathered(model) as writer:
        res["writer"] = writer
        res["gathered_shape"] = tuple(model[0].weight.shape) if writer else None
    res["full_keys"] = sorted(native.native_full_state_dict(model))
    for unit in save.fsdp2_units(model):
        unit.set_reshard_after_forward(False)
    with torch.no_grad():
        model(torch.randn(2, 8))
    DTensor = save._dtensor_class()
    res["unsharded_plain"] = not any(isinstance(p, DTensor) for p in model.parameters())
    res["unsharded_writer_rank"] = save.mesh_writer_rank(model)
    with save.fsdp2_gathered(model) as writer:
        res["unsharded_writer"] = writer
    for unit in save.fsdp2_units(model):
        unit.set_reshard_after_forward(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    path = os.path.join(os.environ["UNSLOTH_TEST_SAVE_DIR"], f"optimizer_pair{rank // 2}.pt")
    native.native_save_optimizer(model, optimizer, path)
    state = torch.load(path, weights_only = False)
    res["optimizer_entries"] = len(state.get("state", {}))
    # Full (gathered) moments: 8 * 8 + 8 + 4 * 8 + 4 elements, not one rank's shards.
    res["exp_avg_numel"] = sum(v["exp_avg"].numel() for v in state["state"].values())
    import torch.distributed as dist

    dist.barrier()  # rank 0 hosts the store the other pair still waits on
    return res


def _case_trainer_save(native, rank):
    """Each pair saves its LoRA model, a full model and the optimizer + scheduler through the
    Trainer hooks at the same time. Pair [2, 3]'s should_save is False on both ranks."""
    import torch.distributed as dist

    _stub_unsloth_save()
    mesh = _submesh(rank)
    root = os.path.join(os.environ["UNSLOTH_TEST_SAVE_DIR"], f"pair{rank // 2}")
    cfg = native.FSDP2Config(param_dtype = torch.float32, mesh = mesh)
    res = {}
    lora = native.shard_model(_lora(1), cfg)
    trainer = _trainer(lora, root)
    res["should_save"] = bool(trainer.args.should_save)
    trainer.save_model(os.path.join(root, "lora"))
    res["lora_written"] = os.path.isfile(os.path.join(root, "lora", "adapter_model.safetensors"))
    trainer.create_optimizer_and_scheduler(num_training_steps = 2)
    os.makedirs(os.path.join(root, "ckpt"), exist_ok = True)  # Trainer._save_checkpoint does this
    trainer._save_optimizer_and_scheduler(os.path.join(root, "ckpt"))
    res["optimizer_written"] = os.path.isfile(os.path.join(root, "ckpt", "optimizer.pt"))
    full = native.shard_model(_model(False, 1, root), cfg)
    _trainer(full, root).save_model(os.path.join(root, "full"))
    res["full_written"] = os.path.isfile(os.path.join(root, "full", "model.safetensors"))
    dist.barrier()  # rank 0 hosts the store the other pair still waits on
    # The scheduler is written after the optimizer wait, so only after the barrier on every rank.
    res["scheduler_written"] = os.path.isfile(os.path.join(root, "ckpt", "scheduler.pt"))
    return res


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

    # Ranks 2 and 3 can wait at a world barrier while the other pair shards a model, which takes
    # more than 30 s on a loaded CI runner; _run's deadline still catches a real hang.
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 150)
    )
    try:
        native = _load_native()
        if case == "media":
            res = _case_media(rank)
        elif case == "fit":
            importlib.import_module("unsloth.models.fsdp2_perf")
            res = _case_fit(native, rank)
        elif case == "subgroup":
            res = _case_subgroup(native, rank)
        elif case == "save":
            os.environ["UNSLOTH_FSDP2_SAVE_TIMEOUT"] = "60"
            importlib.import_module("unsloth.models.fsdp2_save")
            res = _case_save(native, rank)
        elif case == "trainer_save":
            os.environ["UNSLOTH_FSDP2_SAVE_TIMEOUT"] = "30"
            importlib.import_module("unsloth.models.fsdp2_save")
            res = _case_trainer_save(native, rank)
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


def test_layout_agreement_is_per_mesh():
    pytest.importorskip("peft")
    results = _run("fit")
    assert [results[r]["fits"] for r in range(WORLD)] == [True, True, False, False], results


def test_a_sub_mesh_shards_without_the_other_ranks():
    pytest.importorskip("peft")
    results = _run("subgroup")
    assert [results[r]["sharded"] for r in range(WORLD)] == [True, True, False, False], results


def test_sub_mesh_saves_write_from_the_mesh_first_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_TEST_SAVE_DIR", str(tmp_path))
    results = _run("save")
    assert [results[r]["writer"] for r in range(WORLD)] == [True, False, True, False], results
    assert results[0]["gathered_shape"] == results[2]["gathered_shape"] == (8, 8), results
    for rank in (0, 2):
        assert results[rank]["full_keys"] == ["0.bias", "0.weight", "1.bias", "1.weight"], results
    for rank in (1, 3):
        assert results[rank]["full_keys"] == [], results
    for rank in range(WORLD):
        assert results[rank]["unsharded_plain"], results
        assert results[rank]["unsharded_writer_rank"] == (0 if rank < 2 else 2), results
    assert [results[r]["unsharded_writer"] for r in range(WORLD)] == [True, False, True, False]
    for rank in range(WORLD):
        assert results[rank]["optimizer_entries"] == 4, results
        assert results[rank]["exp_avg_numel"] == 108, results


def test_sub_mesh_trainer_saves_write_from_the_mesh_first_rank(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    monkeypatch.setenv("UNSLOTH_TEST_SAVE_DIR", str(tmp_path))
    results = _run("trainer_save", timeout = 240)
    assert [results[r]["should_save"] for r in range(WORLD)] == [True, False, False, False]
    for rank in range(WORLD):
        # Every rank returns after its pair's writer wrote (the store wait), so all see the files.
        for key in ("lora_written", "optimizer_written", "scheduler_written", "full_written"):
            assert results[rank][key], (rank, key, results)
