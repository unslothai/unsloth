# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel checkpoint integrity on gloo CPU ranks with tiny OLMoE."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_shared"))

OLMOE = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
EXPERT_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
_TIMEOUT_S = 20  # gloo collective timeout


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    assert os.path.dirname(epn.__file__) == os.path.join(os.path.dirname(HERE), "unsloth", "models")
    return build, tpn, epn


def _ep_model(
    build,
    epn,
    mesh,
    seed = 0,
):
    model = build(OLMOE, seed = seed, **EXPERT_LORA).train()
    g = torch.Generator().manual_seed(1234 + seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    return epn.apply_expert_parallel(model, mesh)


def _step(model, optimizer, rank):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    ids = torch.randint(0, 64, (2, 6), generator = torch.Generator().manual_seed(rank))
    ddp = DistributedDataParallel(model)  # kept referenced through backward: its reducer averages
    loss = ddp(input_ids = ids, labels = ids, use_cache = False).loss
    loss.backward()
    _modules()[2].ep_sync_expert_grads(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    dist.barrier()
    return ddp


def _flat_mesh(world):
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]


def _readable(path):
    try:
        return torch.load(path, weights_only = False)
    except Exception as exc:
        return f"UNREADABLE ({type(exc).__name__})"


def _case_rank_failure(rank, world, tmp):
    """Overwrite a committed rank state while rank 1 is capped by RLIMIT_FSIZE (simulated full disk)."""
    import resource
    import signal

    build, tpn, epn = _modules()
    model = _ep_model(build, epn, _flat_mesh(world))
    directory = os.path.join(tmp, "ckpt")
    epn._save_rank_state(model, None, directory)
    path = epn._rank_path(directory)
    committed = torch.load(path, weights_only = False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    ddp = _step(model, optimizer, rank)
    if rank == 1:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        limit = os.path.getsize(path) + 512
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.RLIM_INFINITY))
    start = time.monotonic()
    try:
        epn._save_rank_state(model, optimizer, directory)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    if rank == 1:
        resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    after = _readable(path)
    del ddp
    return {
        "error": None,
        "save_error": error,
        "elapsed": elapsed,
        "readable": isinstance(after, dict),
        "old_kept": isinstance(after, dict)
        and after["optimizer"] is None
        and all(torch.equal(after["weights"][k], v) for k, v in committed["weights"].items()),
        "new_written": isinstance(after, dict) and after["optimizer"] is not None,
        "leftovers": sorted(f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f),
    }


def _case_reordered_mesh(rank, world, tmp):
    """Saved with ep groups [0, 1] / [2, 3], resumed with [0, 2] / [1, 3]: same sizes and shapes, other experts."""
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    build, tpn, epn = _modules()
    names = ("ep_rep", "ep")
    saved_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = names)
    other_mesh = DeviceMesh(
        "cpu", torch.arange(4).reshape(2, 2).T.contiguous(), mesh_dim_names = names
    )
    directory = os.path.join(tmp, "ckpt")
    epn._save_rank_state(_ep_model(build, epn, saved_mesh), None, directory)

    def load(mesh, where):
        try:
            epn._load_rank_weights(_ep_model(build, epn, mesh, seed = 1), where)
            return None
        except ValueError as exc:
            return str(exc)

    legacy = os.path.join(tmp, "legacy")
    path = epn._rank_path(legacy)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    payload = torch.load(epn._rank_path(directory), weights_only = False)
    recorded = {k: payload["layout"].pop(k, None) for k in ("ep_rank", "ep_members")}
    torch.save(payload, path)
    return {
        "error": None,
        "recorded": recorded,
        "same": load(saved_mesh, directory),
        "reordered": load(other_mesh, directory),
        "legacy_same": load(saved_mesh, legacy),
    }


def _case_round_trip(rank, world, tmp):
    build, tpn, epn = _modules()
    mesh = _flat_mesh(world)
    model = _ep_model(build, epn, mesh)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    ddp = _step(model, optimizer, rank)
    directory = os.path.join(tmp, "ckpt")
    epn._save_rank_state(model, optimizer, directory)
    epn._save_rank_state(model, optimizer, directory)
    fresh = _ep_model(build, epn, mesh)
    fresh_opt = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr = 1e-3)
    epn._load_rank_weights(fresh, directory)
    epn._load_rank_optimizer(fresh, fresh_opt, directory)
    want = dict(model.named_parameters())
    weights = all(torch.equal(p, want[n]) for n, p in fresh.named_parameters())
    a, b = optimizer.state_dict(), fresh_opt.state_dict()
    optim = a["param_groups"] == b["param_groups"] and all(
        torch.equal(a["state"][i][k], b["state"][i][k]) for i in a["state"] for k in a["state"][i]
    )
    layout = torch.load(epn._rank_path(directory), weights_only = False)["layout"]
    del ddp
    return {
        "error": None,
        "weights": weights,
        "optim": optim and len(a["state"]) > 0,
        "layout": {k: layout.get(k) for k in ("ep_rank", "ep_members")},
        "leftovers": sorted(
            f for f in os.listdir(os.path.dirname(epn._rank_path(directory))) if ".tmp" in f
        ),
    }


def _case_each_node(rank, world, tmp):
    """save_on_each_node, writers 0 and 2 over one ep group of 4: only writers get the gathered LoRA."""
    import types

    import torch.distributed as dist
    from safetensors.torch import load_file

    build, tpn, epn = _modules()
    model = _ep_model(build, epn, _flat_mesh(world))
    received = []
    gather, all_gather = dist.gather, dist.all_gather

    def counting_gather(
        tensor,
        gather_list = None,
        *args,
        **kwargs,
    ):
        received.append(sum(t.numel() for t in gather_list or []))
        return gather(tensor, gather_list, *args, **kwargs)

    def counting_all_gather(tensor_list, tensor, *args, **kwargs):
        received.append(sum(t.numel() for t in tensor_list))
        return all_gather(tensor_list, tensor, *args, **kwargs)

    out = os.path.join(tmp, f"node{rank // 2}")
    args = types.SimpleNamespace(
        save_on_each_node = True,
        should_save = rank % 2 == 0,
        output_dir = out,
        load_best_model_at_end = False,
        save_only_model = False,
    )
    accelerator = types.SimpleNamespace(
        backward = lambda *a, **k: None, prepare_model = lambda *a, **k: None, sync_gradients = True
    )
    trainer = types.SimpleNamespace(
        model = model,
        model_init = None,
        accelerator = accelerator,
        args = args,
        processing_class = None,
        save_model = None,
        _save_optimizer_and_scheduler = None,
        _load_optimizer_and_scheduler = None,
        _load_from_checkpoint = None,
    )
    assert epn.ep_trainer_setup(trainer)
    dist.gather, dist.all_gather = counting_gather, counting_all_gather
    try:
        trainer.save_model(out)
    finally:
        dist.gather, dist.all_gather = gather, all_gather
    one = os.path.join(tmp, "one")
    model.save_pretrained(one)
    want = load_file(os.path.join(one, "adapter_model.safetensors"))
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    same = sorted(got) == sorted(want) and all(torch.equal(got[k], want[k]) for k in want)
    return {"error": None, "received": sum(received), "same": same}


def _plain_model(build, seed = 0):
    """The one-process twin of _ep_model (same weights, nothing sharded)."""
    model = build(OLMOE, seed = seed, **EXPERT_LORA).train()
    g = torch.Generator().manual_seed(1234 + seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    return model


def _utils_save_lora_adapter():
    """models/_utils.py save_lora_adapter (model.save_lora) and the helpers it calls, without importing
    _utils (which pulls in the whole package)."""
    import ast

    path = os.path.join(os.path.dirname(HERE), "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    wanted = {
        "_unsloth_model_parallel_marker",
        "_is_unsloth_tensor_parallel",
        "_is_unsloth_expert_parallel",
        "save_lora_adapter",
    }
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert "save_lora_adapter" in {n.name for n in body}
    scope = {"__name__": "unsloth.models._utils", "__package__": "unsloth.models", "torch": torch}
    exec(compile(ast.Module(body = body, type_ignores = []), path, "exec"), scope)
    return scope["save_lora_adapter"]


def _case_save_lora(rank, world, tmp):
    """model.save_lora (the GRPO notebook idiom) on every rank: one 1-GPU adapter, written by rank 0 only."""
    import json

    import torch.distributed as dist
    from safetensors.torch import load_file

    build, tpn, epn = _modules()
    save_lora = _utils_save_lora_adapter()
    model = _ep_model(build, epn, _flat_mesh(world))
    writes = []
    level = model
    prior = level.save_pretrained

    def counted(*args, **kwargs):
        writes.append(kwargs.get("state_dict") is not None)
        return prior(*args, **kwargs)

    level.save_pretrained = counted
    directory = os.path.join(tmp, "lora")
    try:
        save_lora(model, directory)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    dist.barrier()
    out = {"error": None, "save_error": error, "writes": len(writes)}
    if rank == 0:
        ref = os.path.join(tmp, "one_process")
        save_lora(_plain_model(build), ref)
        want = load_file(os.path.join(ref, "adapter_model.safetensors"))
        try:
            got = load_file(os.path.join(directory, "adapter_model.safetensors"))
        except Exception as exc:
            got = {"UNREADABLE": f"{type(exc).__name__}"}
        out["missing"] = sorted(set(want) - set(got))
        out["differ"] = sorted(
            k
            for k in want
            if k in got and not (got[k].shape == want[k].shape and torch.equal(got[k], want[k]))
        )
        out["experts"] = sorted(k for k in want if "experts" in k)
        out["config_same"] = json.load(open(os.path.join(ref, "adapter_config.json"))) == json.load(
            open(os.path.join(directory, "adapter_config.json"))
        )
    return out


def _trainer_ns(model, rank, out):
    """What ep_trainer_setup and the stock Trainer optimizer save / load read, around real optimizer and
    scheduler objects (the stock methods are transformers' own, bound to this namespace)."""
    import types

    import transformers

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
    args = types.SimpleNamespace(
        save_on_each_node = False,
        should_save = rank == 0,
        output_dir = out,
        load_best_model_at_end = False,
        save_only_model = False,
        world_size = 2,
        process_index = rank,
        device = torch.device("cpu"),
    )
    accelerator = types.SimpleNamespace(
        backward = lambda *a, **k: None, prepare_model = lambda *a, **k: None, sync_gradients = True
    )
    trainer = types.SimpleNamespace(
        model = model,
        model_init = None,
        accelerator = accelerator,
        args = args,
        processing_class = None,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        is_deepspeed_enabled = False,
        is_fsdp_enabled = False,
        is_fsdp_xla_v1_enabled = False,
        is_fsdp_xla_v2_enabled = False,
        save_model = None,
        _load_from_checkpoint = None,
    )
    stock = transformers.Trainer
    trainer._save_optimizer_and_scheduler = types.MethodType(
        stock._save_optimizer_and_scheduler, trainer
    )
    trainer._load_optimizer_and_scheduler = types.MethodType(
        stock._load_optimizer_and_scheduler, trainer
    )
    assert _modules()[2].ep_trainer_setup(trainer)
    return trainer


def _case_optimizer_save_failure(rank, world, tmp):
    """A Trainer checkpoint over a committed one while rank 0 (the stock writer) runs out of disk."""
    import resource
    import signal

    build, tpn, epn = _modules()
    model = _ep_model(build, epn, _flat_mesh(world))
    directory = os.path.join(tmp, "ckpt")
    os.makedirs(directory, exist_ok = True)
    trainer = _trainer_ns(model, rank, directory)
    trainer._save_optimizer_and_scheduler(directory)  # committed: no optimizer state yet
    path = epn._rank_path(directory)
    files = sorted(os.listdir(directory))
    committed = {f: _readable(os.path.join(directory, f)) for f in files if f.endswith(".pt")}
    committed_rank = torch.load(path, weights_only = False)
    ddp = _step(model, trainer.optimizer, rank)
    trainer.lr_scheduler.step()
    if rank == 0:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        sizes = [os.path.getsize(os.path.join(directory, f)) for f in files if f.endswith(".pt")]
        limit = max(sizes + [os.path.getsize(path)]) + 512
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.RLIM_INFINITY))
    start = time.monotonic()
    try:
        trainer._save_optimizer_and_scheduler(directory)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    if rank == 0:
        resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    after_rank = _readable(path)
    kept = {}
    for f, before in committed.items():
        now = _readable(os.path.join(directory, f))
        kept[f] = isinstance(now, dict) and isinstance(before, dict) and repr(now) == repr(before)
    del ddp
    return {
        "error": None,
        "save_error": error,
        "elapsed": elapsed,
        "rank_state_kept": isinstance(after_rank, dict)
        and after_rank["optimizer"] == committed_rank["optimizer"]
        and all(
            torch.equal(after_rank["weights"][k], v) for k, v in committed_rank["weights"].items()
        ),
        "stock_files_kept": kept,
        "leftovers": sorted(
            f for d in (directory, os.path.dirname(path)) for f in os.listdir(d) if ".tmp" in f
        ),
    }


def _case_optimizer_round_trip(rank, world, tmp):
    """Trainer checkpoint save then resume: optimizer and scheduler state back bitwise, from the rank state."""
    import torch.distributed as dist

    build, tpn, epn = _modules()
    mesh = _flat_mesh(world)
    model = _ep_model(build, epn, mesh)
    directory = os.path.join(tmp, "ckpt")
    os.makedirs(directory, exist_ok = True)
    trainer = _trainer_ns(model, rank, directory)
    ddp = _step(model, trainer.optimizer, rank)
    trainer.lr_scheduler.step()
    trainer._save_optimizer_and_scheduler(directory)
    dist.barrier()
    fresh = _trainer_ns(_ep_model(build, epn, mesh), rank, directory)
    fresh._load_optimizer_and_scheduler(directory)
    a, b = trainer.optimizer.state_dict(), fresh.optimizer.state_dict()
    optim = a["param_groups"] == b["param_groups"] and all(
        torch.equal(a["state"][i][k], b["state"][i][k]) for i in a["state"] for k in a["state"][i]
    )
    del ddp
    return {
        "error": None,
        "optim": optim and len(a["state"]) > 0,
        "scheduler": fresh.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()
        and trainer.lr_scheduler.last_epoch == 1,
        "files": sorted(os.listdir(directory)),
        # The 1-GPU optimizer.pt (keyed by parameter name, every expert), not rank 0's own state.
        "portable_names": sorted(
            n
            for g in torch.load(os.path.join(directory, "optimizer.pt"), weights_only = True)[
                "param_groups"
            ]
            for n in g["params"]
        ),
        # Every rank's trainable names: per-expert modules (transformers 4.x) are held by one rank each.
        "trainable_names": sorted({n for names in _all_names(model) for n in names}),
    }


def _all_names(model):
    import torch.distributed as dist

    mine = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, mine)
    return every


def _worker(rank, world, tmp, case):
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
        timeout = datetime.timedelta(seconds = _TIMEOUT_S),
    )
    try:
        res = globals()[case](rank, world, tmp)
    except BaseException as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-3000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    # Stay alive until every rank reported; no destroy_process_group: a rank whose peer raised must not wait on it.
    deadline = time.monotonic() + 3 * _TIMEOUT_S
    while time.monotonic() < deadline and not all(
        os.path.exists(os.path.join(tmp, f"result_{r}.pt")) for r in range(world)
    ):
        time.sleep(0.2)
    os._exit(0)


def _spawn(case, world = 2):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, tmp, case)) for r in range(world)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 300
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


_CHILD = r"""
import os, resource, signal, sys
sys.path[:0] = sys.argv[1:3]
import torch, torch.distributed as dist
from unsloth_models_loader import load_models_module
epn = load_models_module("ep_native")
assert os.path.dirname(epn.__file__) == sys.argv[3], epn.__file__
epn._layout = lambda model: {"world": 1, "ep": 1, "replicas": 1, "lora_b": {}}  # needs no ep mesh
dist.init_process_group("gloo", store = dist.FileStore(sys.argv[4] + ".store", 1), rank = 0, world_size = 1)
n, limit = int(sys.argv[5]), int(sys.argv[6])
torch.manual_seed(0)
model = torch.nn.Linear(n, n)
optimizer = torch.optim.AdamW(model.parameters())
model(torch.randn(2, n)).sum().backward()
optimizer.step()
if limit:
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # EFBIG instead of a kill: the write fails like a full disk
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
try:
    epn._save_rank_state(model, optimizer, sys.argv[4])
    print("SAVED")
except Exception as exc:
    print("FAILED", type(exc).__name__)
"""


def test_failed_overwrite_keeps_the_committed_rank_state():
    models = os.path.join(os.path.dirname(HERE), "unsloth", "models")
    with tempfile.TemporaryDirectory() as tmp:
        directory = os.path.join(tmp, "ckpt")

        def child(n, limit):
            out = subprocess.run(
                [sys.executable, "-c", _CHILD, HERE, os.path.join(HERE, "_shared"), models]
                + [directory, str(n), str(limit)],
                capture_output = True,
                text = True,
                timeout = 120,
            )
            assert out.returncode == 0, out.stderr[-2000:]
            return out.stdout.strip().splitlines()[-1]

        assert child(16, 0) == "SAVED"
        path = os.path.join(directory, "unsloth_ep", "rank00000.pt")
        committed = torch.load(path, weights_only = False)
        assert child(64, os.path.getsize(path) + 512).startswith("FAILED")
        after = _readable(path)
        assert isinstance(after, dict), after
        assert after["weights"]["weight"].shape == (16, 16)
        assert torch.equal(after["weights"]["weight"], committed["weights"]["weight"])
        assert not [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]


def test_write_failure_on_one_rank_raises_on_every_rank():
    results = _spawn("_case_rank_failure")
    assert all(r is not None and r["error"] is None for r in results), results
    rank0, rank1 = results
    assert rank1["save_error"] and "RuntimeError" in rank1["save_error"], rank1
    # Without the agreement rank 0 would wait out the whole gloo timeout.
    assert rank0["elapsed"] < _TIMEOUT_S / 2, rank0
    assert rank0["save_error"] and "another rank failed" in rank0["save_error"], rank0
    assert rank1["readable"] and rank1["old_kept"], rank1
    assert rank0["readable"] and rank0["new_written"], rank0
    assert not rank0["leftovers"] and not rank1["leftovers"], results


def test_resume_on_another_ep_membership_is_refused():
    results = _spawn("_case_reordered_mesh", world = 4)
    assert all(r is not None and r["error"] is None for r in results), results
    for rank, r in enumerate(results):
        assert r["recorded"] == {
            "ep_rank": rank % 2,
            "ep_members": [rank // 2 * 2, rank // 2 * 2 + 1],
        }, r
        assert r["same"] is None and r["legacy_same"] is None, r
        assert r["reordered"] is not None and "same rank layout" in r["reordered"], r


def test_save_resume_round_trip_is_bitwise():
    results = _spawn("_case_round_trip")
    assert all(r is not None and r["error"] is None for r in results), results
    for rank, r in enumerate(results):
        assert r["weights"] and r["optim"], r
        assert r["layout"] == {"ep_rank": rank, "ep_members": [0, 1]}, r
        assert not r["leftovers"], r


def test_save_on_each_node_gathers_only_onto_the_writers():
    results = _spawn("_case_each_node", world = 4)
    assert all(r is not None and r["error"] is None for r in results), results
    for rank, r in enumerate(results):
        assert r["same"], r
        if rank % 2:
            assert r["received"] == 0, r  # not a writer: never holds the gathered experts
        else:
            assert r["received"] > 0, r


def test_save_lora_writes_the_one_process_adapter_from_rank_0():
    results = _spawn("_case_save_lora")
    assert all(r is not None and r["error"] is None for r in results), results
    rank0, rank1 = results
    assert rank0["save_error"] is None and rank1["save_error"] is None, results
    assert rank0["experts"], rank0  # the adapter trains the expert stacks
    assert not rank0["missing"] and not rank0["differ"], rank0
    assert rank0["config_same"], rank0
    assert rank0["writes"] == 1 and rank1["writes"] == 0, results  # only rank 0 wrote


def test_optimizer_save_failure_raises_on_every_rank_and_keeps_the_checkpoint():
    results = _spawn("_case_optimizer_save_failure")
    assert all(r is not None and r["error"] is None for r in results), results
    for r in results:
        # Without the agreement rank 1 would wait out the whole gloo timeout for rank 0.
        assert r["save_error"] and r["elapsed"] < _TIMEOUT_S / 2, results
        assert all(r["stock_files_kept"].values()), r  # optimizer.pt / scheduler.pt not truncated
        assert not r["leftovers"], r
    assert results[0]["rank_state_kept"], results[0]  # the failed overwrite left the committed file
    assert "another rank failed" in results[1]["save_error"], results


def test_optimizer_checkpoint_round_trip_from_the_rank_state():
    results = _spawn("_case_optimizer_round_trip")
    assert all(r is not None and r["error"] is None for r in results), results
    for r in results:
        assert r["optim"] and r["scheduler"], r
        # rank 0's state lives in its rank file; optimizer.pt is the 1-GPU form (portable_resume.py)
        assert r["portable_names"] == r["trainable_names"], r
