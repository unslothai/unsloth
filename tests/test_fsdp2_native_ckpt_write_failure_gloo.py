# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 checkpoint writes that fail on the writer rank (gloo CPU world of 2, RLIMIT_FSIZE as
a full disk, or a raising write):

* an optimizer.pt overwrite keeps the committed file whole, and every rank raises within seconds
  instead of the other rank running on into a collective the writer never joins;
* the Trainer hook keeps the other format's optimizer_dcp when the new optimizer.pt fails, so the
  checkpoint still resumes;
* a failed scheduler write, or a failed write inside fsdp2_gathered, raises on every rank too;
* a sharded (DCP) optimizer save over a committed optimizer_dcp whose write fails on one rank keeps
  the committed directory and resumes it bitwise, with one renaming rank or one per "node";
* a save that stops between its two renames resumes from the new, complete copy, and the next save
  clears the leftovers.

UNSLOTH_TEST_BACKEND=nccl runs the same cases on two GPUs."""

import multiprocessing
import os

_BACKEND = os.environ.get("UNSLOTH_TEST_BACKEND", "gloo")
if _BACKEND == "gloo" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_native():
    """fsdp2_native.py without importing the unsloth package (which needs a GPU), as
    test_fsdp2_native_units does; not imported from there, since that module hides the GPUs from
    spawned ranks and this file also runs on NCCL."""
    import importlib.util
    import types

    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    for name, path in (
        ("unsloth.fsdp2_launch", os.path.join(REPO, "unsloth", "fsdp2_launch.py")),
        ("unsloth.models.fsdp2_native", os.path.join(REPO, "unsloth", "models", "fsdp2_native.py")),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["unsloth.models.fsdp2_native"]


REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _model(seed):
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None
    config.tie_word_embeddings = False
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator = g) * 0.01)
    return model


def _batch(step, rank):
    g = torch.Generator().manual_seed(100 * step + rank)
    ids = torch.randint(3, 1000, (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone(), "use_cache": False}


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason = "RLIMIT_FSIZE (the full disk) is POSIX only"
)

_PG_TIMEOUT_S = 20  # a rank stranded in a collective fails after this, not after 30 minutes
_FAST_S = 10  # every rank must raise well inside the collective timeout


def _new(native, seed):
    model = _model(seed)
    if _BACKEND == "nccl":
        model = model.cuda()
    model = native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
    return model, torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)


def _train(model, opt, step, rank):
    batch = _batch(step, rank)
    if _BACKEND == "nccl":
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model(**batch).loss.backward()
    opt.step()
    opt.zero_grad(set_to_none = True)


def _cap_file_size(limit):
    import resource
    import signal

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # EFBIG instead of a kill
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.RLIM_INFINITY))


def _uncap():
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def _attempt(fn):
    start = time.monotonic()
    try:
        fn()
        error = None
    except BaseException as exc:  # DCP's CheckpointException is a BaseException
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return error, time.monotonic() - start


def _post_collective(dist):
    """What the training loop does next on a rank whose save returned."""
    device = "cuda" if _BACKEND == "nccl" else "cpu"
    start = time.monotonic()
    try:
        flag = torch.ones(1, device = device)
        dist.all_reduce(flag)
        if _BACKEND == "nccl":
            torch.cuda.synchronize()
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__} after {time.monotonic() - start:.1f}s: {str(exc)[:160]}"


def _writer_stays_alive(dist, rank):
    """The failed writer's process does not exit at once (a caller catching the save error, its own
    teardown): it stays up until the other rank is done, so a stranded rank shows its wait instead of
    a closed connection."""
    store = dist.distributed_c10d._get_default_store()
    if rank == 0:
        store.wait(["unsloth_test_peer_done"], datetime.timedelta(seconds = _PG_TIMEOUT_S + 60))
    else:
        store.set("unsloth_test_peer_done", "1")


def _same_state(a, b):
    if not (isinstance(a, dict) and isinstance(b, dict)) or sorted(a["state"]) != sorted(
        b["state"]
    ):
        return False
    for k, v in a["state"].items():
        for name, t in v.items():
            if isinstance(t, torch.Tensor) and not torch.equal(t, b["state"][k][name]):
                return False
    return True


def _readable(path):
    try:
        return torch.load(path, map_location = "cpu", weights_only = True)
    except Exception as exc:
        return f"UNREADABLE ({type(exc).__name__})"


def _leftovers(directory):
    return sorted(f for f in os.listdir(directory) if ".tmp" in f)


def _case_optimizer_overwrite(native, dist, rank, tmp):
    """Overwrite a committed optimizer.pt while the writer's disk is full."""
    model, opt = _new(native, 1)
    _train(model, opt, 1, rank)
    path = os.path.join(tmp, "optimizer.pt")
    native.native_save_optimizer(model, opt, path, should_save = rank == 0)
    committed = torch.load(path, map_location = "cpu", weights_only = True)
    size = os.path.getsize(path)
    _train(model, opt, 2, rank)
    if rank == 0:
        _cap_file_size(size // 2)
    error, elapsed = _attempt(
        lambda: native.native_save_optimizer(model, opt, path, should_save = rank == 0)
    )
    if rank == 0:
        _uncap()
    res = {"save_error": error, "elapsed": elapsed, "committed_bytes": size}
    res["post_collective"] = _post_collective(dist) if error is None else None
    after = _readable(path)
    res["readable"] = isinstance(after, dict)
    res["bytes_after"] = os.path.getsize(path)
    res["old_kept"] = _same_state(after, committed)
    res["leftovers"] = _leftovers(tmp)
    return res


def _case_trainer_cross_format(native, dist, rank, tmp):
    """A checkpoint directory holding a committed optimizer_dcp is saved again in the full format
    through the Trainer hook while the writer's disk is full: the committed state must survive."""
    import types

    from transformers import Trainer

    native._install_trainer_hooks()
    model, opt = _new(native, 1)
    _train(model, opt, 1, rank)
    ckpt = os.path.join(tmp, "checkpoint-1")
    os.makedirs(ckpt, exist_ok = True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    trainer = types.SimpleNamespace(
        model = model,
        optimizer = opt,
        lr_scheduler = sched,
        args = types.SimpleNamespace(should_save = rank == 0),
        is_fsdp_enabled = False,
    )
    os.environ["UNSLOTH_FSDP2_DCP_OPTIM"] = "1"
    Trainer._save_optimizer_and_scheduler(trainer, ckpt)
    os.environ.pop("UNSLOTH_FSDP2_DCP_OPTIM", None)
    dist.barrier()
    dcp_dir = os.path.join(ckpt, "optimizer_dcp")
    committed = sorted(os.listdir(dcp_dir))
    _train(model, opt, 2, rank)
    if rank == 0:
        _cap_file_size(4096)  # below the full optimizer state, above scheduler.pt
    error, elapsed = _attempt(lambda: Trainer._save_optimizer_and_scheduler(trainer, ckpt))
    if rank == 0:
        _uncap()
    res = {"save_error": error, "elapsed": elapsed}
    res["dcp_kept"] = os.path.isdir(dcp_dir) and sorted(os.listdir(dcp_dir)) == committed
    res["optimizer_pt"] = (
        _readable(os.path.join(ckpt, "optimizer.pt"))
        if os.path.exists(os.path.join(ckpt, "optimizer.pt"))
        else None
    )
    res["optimizer_pt"] = (
        None
        if res["optimizer_pt"] is None
        else ("readable" if isinstance(res["optimizer_pt"], dict) else res["optimizer_pt"])
    )
    res["leftovers"] = _leftovers(ckpt)
    if error is not None and res["dcp_kept"]:
        # Both ranks raised, so both are here: resume the optimizer from the kept checkpoint.
        fresh, fresh_opt = _new(native, 1)
        load_error, _ = _attempt(lambda: native.native_load_optimizer(fresh, fresh_opt, dcp_dir))
        res["dcp_resume_error"] = load_error
    return res


def _case_scheduler_fail(native, dist, rank, tmp):
    """The optimizer write succeeds, the writer's scheduler write raises."""
    import types

    from transformers import Trainer

    native._install_trainer_hooks()
    model, opt = _new(native, 1)
    _train(model, opt, 1, rank)
    ckpt = os.path.join(tmp, "checkpoint-1")
    os.makedirs(ckpt, exist_ok = True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    if rank == 0:

        def broken():
            raise OSError(28, "No space left on device (injected)")

        sched.state_dict = broken
    trainer = types.SimpleNamespace(
        model = model,
        optimizer = opt,
        lr_scheduler = sched,
        args = types.SimpleNamespace(should_save = rank == 0),
        is_fsdp_enabled = False,
    )
    error, elapsed = _attempt(lambda: Trainer._save_optimizer_and_scheduler(trainer, ckpt))
    res = {"save_error": error, "elapsed": elapsed}
    res["post_collective"] = _post_collective(dist) if error is None else None
    return res


def _case_gathered_body_fail(native, dist, rank, tmp):
    """The writer's write inside fsdp2_gathered raises."""
    import importlib

    fsdp2_save = importlib.import_module("unsloth.models.fsdp2_save")
    model, _ = _new(native, 1)

    def save():
        with fsdp2_save.fsdp2_gathered(model) as writer:
            if writer:
                raise OSError(28, "No space left on device (injected)")

    error, elapsed = _attempt(save)
    res = {"save_error": error, "elapsed": elapsed}
    res["post_collective"] = _post_collective(dist) if error is None else None
    return res


def _local_state(native, model, opt):
    """This rank's optimizer shards (FQN keyed, plain local tensors), cloned."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    from torch.distributed.tensor import DTensor

    state = get_optimizer_state_dict(model, opt, options = StateDictOptions(full_state_dict = False))
    out = {}
    for fqn, entry in state["state"].items():
        for name, t in entry.items():
            if isinstance(t, DTensor):
                t = t.to_local()
            if isinstance(t, torch.Tensor):
                out[(fqn, name)] = t.detach().cpu().clone()
    return out


def _bitwise(a, b):
    return sorted(a) == sorted(b) and all(
        a[k].dtype == b[k].dtype and torch.equal(a[k].view(-1), b[k].view(-1)) for k in a
    )


def _tree_digest(directory):
    import hashlib

    if not os.path.isdir(directory):
        return None
    out = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def _dcp_leftovers(tmp):
    return sorted(f for f in os.listdir(tmp) if f != "optimizer_dcp")


def _resume_bitwise(native, rank, directory, expected):
    fresh, fresh_opt = _new(native, 1)
    _train(fresh, fresh_opt, 7, rank)  # some other state, so a skipped load cannot pass
    error, _ = _attempt(lambda: native.native_load_optimizer(fresh, fresh_opt, directory))
    if error is not None:
        return error
    return _bitwise(_local_state(native, fresh, fresh_opt), expected)


def _case_dcp_overwrite(
    native,
    dist,
    rank,
    tmp,
    per_node = False,
):
    """A committed optimizer_dcp is saved over while rank 1's disk is full."""
    if per_node:
        native._node_id = lambda: f"node{rank}"  # every rank renames, in turn, on one filesystem
    model, opt = _new(native, 1)
    _train(model, opt, 1, rank)
    directory = os.path.join(tmp, "optimizer_dcp")
    native.native_save_optimizer_sharded(model, opt, directory)
    dist.barrier()
    committed = _local_state(native, model, opt)
    digest = _tree_digest(directory)
    shard = os.path.join(directory, f"__{rank}_0.distcp")
    size = os.path.getsize(shard)
    _train(model, opt, 2, rank)
    if rank == 1:
        _cap_file_size(size // 2)
    error, elapsed = _attempt(lambda: native.native_save_optimizer_sharded(model, opt, directory))
    if rank == 1:
        _uncap()
    res = {"save_error": error, "elapsed": elapsed}
    res["post_collective"] = _post_collective(dist) if error is None else None
    dist.barrier()
    res["dcp_kept"] = _tree_digest(directory) == digest
    res["leftovers"] = _dcp_leftovers(tmp)
    res["resume_bitwise"] = _resume_bitwise(native, rank, directory, committed)
    return res


def _case_dcp_overwrite_per_node(native, dist, rank, tmp):
    return _case_dcp_overwrite(native, dist, rank, tmp, per_node = True)


def _case_dcp_crash_between_renames(native, dist, rank, tmp):
    """The save stops after `optimizer_dcp` -> `.old`, before the temp copy is renamed in."""
    model, opt = _new(native, 1)
    _train(model, opt, 1, rank)
    directory = os.path.join(tmp, "optimizer_dcp")
    native.native_save_optimizer_sharded(model, opt, directory)
    _train(model, opt, 2, rank)
    second = _local_state(native, model, opt)
    real_rename = os.rename
    calls = []

    def rename(src, dst, *args, **kwargs):
        if os.fspath(dst) == directory or os.fspath(src) == directory:
            calls.append((src, dst))  # the swap's renames (DCP also renames its .metadata)
            if os.fspath(dst) == directory:
                raise OSError(5, "crash between the renames (injected)")
        return real_rename(src, dst, *args, **kwargs)

    os.rename = rename
    try:
        error, elapsed = _attempt(
            lambda: native.native_save_optimizer_sharded(model, opt, directory)
        )
    finally:
        os.rename = real_rename
    dist.barrier()
    res = {"save_error": error, "elapsed": elapsed, "renames": len(calls)}
    res["layout"] = sorted(os.listdir(tmp))
    res["resume_second"] = _resume_bitwise(native, rank, directory, second)
    dist.barrier()
    # The next save swaps in cleanly and clears what the stopped one left.
    _train(model, opt, 3, rank)
    third = _local_state(native, model, opt)
    error3, _ = _attempt(lambda: native.native_save_optimizer_sharded(model, opt, directory))
    dist.barrier()
    res["third_error"] = error3
    res["layout_after"] = sorted(os.listdir(tmp))
    res["resume_third"] = _resume_bitwise(native, rank, directory, third)
    return res


_CASES = {
    "optimizer_overwrite": _case_optimizer_overwrite,
    "trainer_cross_format": _case_trainer_cross_format,
    "scheduler_fail": _case_scheduler_fail,
    "gathered_body_fail": _case_gathered_body_fail,
    "dcp_overwrite": _case_dcp_overwrite,
    "dcp_overwrite_per_node": _case_dcp_overwrite_per_node,
    "dcp_crash_between_renames": _case_dcp_crash_between_renames,
}


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_SAVE_TIMEOUT = "120",  # a missed wake-up fails the test, not after 7 days
    )
    import torch.distributed as dist

    if _BACKEND == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        _BACKEND,
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = _PG_TIMEOUT_S),
    )
    try:
        native = _load_native()
        assert os.path.dirname(native.__file__) == os.path.join(REPO, "unsloth", "models")
        res = _CASES[case](native, dist, rank, tmp)
        _writer_stays_alive(dist, rank)
        out.put((rank, res))
    except BaseException as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp,
    world = 2,
    timeout = 300,
):
    import queue

    import torch.multiprocessing as mp

    from tests._rendezvous_port import free_port

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, out, case, tmp)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, r = out.get(timeout = timeout)
            results[rank] = r
    except queue.Empty:
        pass
    finally:
        for p in procs:
            p.join(timeout = 30)
            if p.is_alive():
                p.kill()
    assert sorted(results) == list(range(world)), f"{case}: ranks did not report: {results}"
    for rank, r in sorted(results.items()):
        print("CKPT_WRITE_FAILURE", _BACKEND, case, rank, r)
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


def _assert_all_raised(results, case):
    writer, other = results[0], results[1]
    assert writer["save_error"] is not None, (case, results)
    # The other rank raises with the writer, within seconds, naming the writer's error.
    assert other["save_error"] is not None, (case, results)
    assert "failed on rank 0" in other["save_error"], (case, results)
    assert other["elapsed"] < _FAST_S, (case, results)
    assert other.get("post_collective") is None, (case, results)


def test_optimizer_overwrite_failure_keeps_file_and_raises_everywhere(tmp_path):
    pytest.importorskip("transformers")
    results = _run("optimizer_overwrite", str(tmp_path))
    _assert_all_raised(results, "optimizer_overwrite")
    for r in results.values():
        assert r["readable"] and r["old_kept"], r
        assert r["bytes_after"] == r["committed_bytes"], r
        assert r["leftovers"] == [], r


def test_trainer_full_save_failure_keeps_committed_dcp(tmp_path):
    pytest.importorskip("transformers")
    results = _run("trainer_cross_format", str(tmp_path))
    _assert_all_raised(results, "trainer_cross_format")
    for r in results.values():
        assert r["dcp_kept"], r
        assert r["dcp_resume_error"] is None, r
        assert r["optimizer_pt"] is None, r  # no truncated optimizer.pt next to it
        assert r["leftovers"] == [], r


def test_scheduler_write_failure_raises_everywhere(tmp_path):
    pytest.importorskip("transformers")
    _assert_all_raised(_run("scheduler_fail", str(tmp_path)), "scheduler_fail")


def test_gathered_write_failure_raises_everywhere(tmp_path):
    pytest.importorskip("transformers")
    _assert_all_raised(_run("gathered_body_fail", str(tmp_path)), "gathered_body_fail")


@pytest.mark.parametrize("case", ["dcp_overwrite", "dcp_overwrite_per_node"])
def test_sharded_save_failure_keeps_committed_dcp(tmp_path, case):
    pytest.importorskip("transformers")
    results = _run(case, str(tmp_path))
    for r in results.values():
        assert r["save_error"] is not None, (case, results)  # no rank runs on
        assert r["elapsed"] < _FAST_S, (case, results)
        assert r["dcp_kept"], (case, results)  # the committed directory, byte for byte
        assert r["leftovers"] == [], (case, results)
        assert r["resume_bitwise"] is True, (case, results)


def test_sharded_save_stopped_between_renames_resumes_new_copy(tmp_path):
    pytest.importorskip("transformers")
    results = _run("dcp_crash_between_renames", str(tmp_path))
    writer = results[0]
    assert writer["renames"] == 2, results
    assert writer["layout"] == ["optimizer_dcp.old", "optimizer_dcp.tmp-2"], results
    for r in results.values():
        assert r["save_error"] is not None, results
        assert r["resume_second"] is True, results  # the newer, complete copy
        assert r["third_error"] is None, results
        assert r["layout_after"] == ["optimizer_dcp"], results
        assert r["resume_third"] is True, results


def test_resolve_dcp_dir_prefers_the_complete_copy(tmp_path):
    native = _load_native()
    d = str(tmp_path / "optimizer_dcp")

    def make(path, complete = True):
        os.makedirs(path, exist_ok = True)
        if complete:
            open(os.path.join(path, ".metadata"), "wb").close()

    assert native._resolve_dcp_dir(d) is None
    make(d + ".old")
    assert native._resolve_dcp_dir(d) == d + ".old"  # stopped after the first rename, no temp copy
    make(d + ".tmp-3", complete = False)
    assert native._resolve_dcp_dir(d) == d + ".old"  # a temp copy without metadata is never read
    make(d + ".tmp-2")
    make(d + ".tmp-4")
    assert native._resolve_dcp_dir(d) == d + ".tmp-4"  # stopped between the renames: the new copy
    make(d, complete = False)
    assert native._resolve_dcp_dir(d) is None  # an incomplete directory in place: not guessed past
    make(d)
    assert native._resolve_dcp_dir(d) == d  # a stopped save before its swap: the committed copy
