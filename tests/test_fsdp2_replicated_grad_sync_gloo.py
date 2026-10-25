# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The replicated-gradient all_reduce (`fsdp2_save._sync_replicated_grads`), on a gloo CPU world of 2:

* gradients are bitwise equal to the previous implementation (fresh zeros buffer per sync, flags
  always read back) over several steps, accumulated micro-steps, fp16 / fp32 / fp64 parameters and
  rank-dependent None gradients;
* the flat buffer is reused per (device, dtype) and rebuilt when the parameter set or a size
  changes, and no gradient aliases it (the next sync overwrites it);
* no device to host read (Tensor.tolist / item) nor host scalar item assignment when every local
  gradient is present, exactly one read when some local gradient is None;
* end to end through reduce_unsharded_trainable_grads on a fully_shard model, for every
  UNSLOTH_FSDP2_GA_SYNC mode (skip / defer / each / auto), replicated and sharded gradients are
  bitwise equal to the previous implementation."""

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
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402


def _reference_sync(
    params,
    dist,
    mesh = None,
):
    """The implementation before the buffer reuse / host-sync removal, verbatim."""
    from unsloth.models.fsdp2_save import mesh_all_reduce

    groups = {}
    for p in params:
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        groups.setdefault((str(p.device), dtype), []).append(p)
    for (device, dtype), group in groups.items():
        sizes = [p.numel() for p in group]
        total = sum(sizes)
        flat = torch.zeros(total + len(group), dtype = dtype, device = group[0].device)
        offset = 0
        for i, (p, n) in enumerate(zip(group, sizes)):
            if p.grad is not None:
                flat[offset : offset + n].copy_(p.grad.detach().reshape(-1))
                flat[total + i] = 1
            offset += n
        world = mesh_all_reduce(flat, mesh, dist)
        flat[:total].div_(world)
        used = flat[total:].tolist()
        offset = 0
        for i, (p, n) in enumerate(zip(group, sizes)):
            if used[i] > 0:
                value = flat[offset : offset + n].view(p.shape)
                if p.grad is None:
                    p.grad = value.to(p.dtype).clone()
                else:
                    p.grad.copy_(value)
            offset += n


def _save():
    _load_native()
    return importlib.import_module("unsloth.models.fsdp2_save")


class _Reads:
    """Counts Tensor.tolist / Tensor.item calls and item assignment of a Python scalar (the device to
    host reads, and host scalar copies, a CUDA run would sync on)."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._saved = (torch.Tensor.tolist, torch.Tensor.item, torch.Tensor.__setitem__)
        reads, (tolist0, item0, setitem0) = self, self._saved

        def tolist(t, *a, **k):
            reads.n += 1
            return tolist0(t, *a, **k)

        def item(t, *a, **k):
            reads.n += 1
            return item0(t, *a, **k)

        def setitem(t, key, value):
            if not isinstance(value, torch.Tensor):
                reads.n += 1
            return setitem0(t, key, value)

        torch.Tensor.tolist, torch.Tensor.item, torch.Tensor.__setitem__ = tolist, item, setitem
        return self

    def __exit__(self, *exc):
        torch.Tensor.tolist, torch.Tensor.item, torch.Tensor.__setitem__ = self._saved


def _make_params(spec):
    return [torch.nn.Parameter(torch.zeros(shape, dtype = dtype)) for shape, dtype in spec]


def _grads(params):
    return [None if p.grad is None else p.grad.clone() for p in params]


def _same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and (x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y)):
            return False
    return True


def _case_unit(rank):
    import torch.distributed as dist

    save = _save()
    spec = [
        ((3, 4), torch.float32),
        ((5,), torch.float16),
        ((2, 2), torch.float64),
        ((7,), torch.float32),
        ((4,), torch.bfloat16),
        ((6,), torch.float16),
    ]
    old, new = _make_params(spec), _make_params(spec)
    buffers = {}
    gen = torch.Generator().manual_seed(100 + rank)
    res = {"mismatch": [], "reads_all_present": [], "reads_some_none": [], "alias": 0}
    # step pattern: which parameters get a local gradient this micro-step on this rank; rank 1
    # leaves some None (a skipped branch), parameter 3 is never used on any rank in steps 0-1.
    patterns = [
        [0, 1, 2, 4, 5] if rank == 0 else [0, 1, 2],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 2] if rank == 1 else [0, 1, 2, 3, 4, 5],
    ]
    for step, used in enumerate(patterns):
        if step in (2, 4):  # optimizer step boundary: zero_grad(set_to_none = True)
            for p in old + new:
                p.grad = None
        for i in used:
            g = (torch.randn(spec[i][0], generator = gen) * 1000).to(spec[i][1])
            for p in (old[i], new[i]):
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.add_(g)  # accumulated micro-step
        _reference_sync(old, dist)
        everyone_all = all(p.grad is not None for p in new)
        with _Reads() as reads:
            save._sync_replicated_grads(new, dist, None, buffers)
        (res["reads_all_present"] if everyone_all else res["reads_some_none"]).append(reads.n)
        if not _same(_grads(old), _grads(new)):
            res["mismatch"].append(step)
        for p in new:
            if p.grad is None:
                continue
            for _, flat in buffers.values():
                lo, hi = flat.data_ptr(), flat.data_ptr() + flat.numel() * flat.element_size()
                if lo <= p.grad.data_ptr() < hi:
                    res["alias"] += 1
    res["n_buffers"] = len(buffers)
    ids_before = {k: id(v[1]) for k, v in buffers.items()}

    # Buffer reused on an unchanged set, rebuilt when the set grows or a size changes.
    for p in old + new:
        p.grad = torch.ones_like(p)
    save._sync_replicated_grads(new, dist, None, buffers)
    res["reused"] = {str(k): id(v[1]) == ids_before[k] for k, v in buffers.items()}
    extra = _make_params([((9,), torch.float32)])
    old2, new2 = old + _make_params([((9,), torch.float32)]), new + extra
    for p in old2 + new2:
        p.grad = (
            (torch.arange(p.numel(), dtype = torch.float32) * (rank + 1)).view(p.shape).to(p.dtype)
        )
    _reference_sync(old2, dist)
    save._sync_replicated_grads(new2, dist, None, buffers)
    res["grown_same"] = _same(_grads(old2), _grads(new2))
    res["grown_rebuilt"] = (
        id(buffers[("cpu", torch.float32)][1]) != ids_before[("cpu", torch.float32)]
    )
    res["fp64_reused"] = (
        id(buffers[("cpu", torch.float64)][1]) == ids_before[("cpu", torch.float64)]
    )
    # A size change at the same position (same parameter count).
    old3, new3 = _make_params([((4,), torch.float64)]), _make_params([((4,), torch.float64)])
    for p in old3 + new3:
        p.grad = torch.full((4,), float(rank + 2), dtype = torch.float64)
    before64 = id(buffers[("cpu", torch.float64)][1])
    _reference_sync(old3, dist)
    save._sync_replicated_grads(new3, dist, None, buffers)
    res["resized_same"] = _same(_grads(old3), _grads(new3))
    res["resized_rebuilt"] = id(buffers[("cpu", torch.float64)][1]) != before64
    # No buffers dict (direct callers): still correct.
    old4, new4 = _make_params(spec), _make_params(spec)
    for p in old4 + new4:
        p.grad = torch.full(p.shape, float(rank + 1)).to(p.dtype)
    if rank == 0:
        old4[1].grad = new4[1].grad = None
    _reference_sync(old4, dist)
    save._sync_replicated_grads(new4, dist)
    res["nobuf_same"] = _same(_grads(old4), _grads(new4))
    return res


class _Net(torch.nn.Module):
    """Sharded fp32 body plus replicated (ignored) parameters of mixed dtype, one used only on some
    micro-steps of some ranks."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.body = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
        self.norm = torch.nn.Parameter(torch.randn(8))
        self.norm16 = torch.nn.Parameter(torch.randn(8).half())
        self.branch = torch.nn.Parameter(torch.randn(8))

    def forward(self, x, use_branch):
        h = self.body(x) * self.norm + self.norm16.float()
        if use_branch:
            h = h * self.branch
        return (h * h).mean()


def _build(save):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    model = _Net()
    mesh = init_device_mesh("cpu", (2,))
    for layer in model.body:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh, ignored_params = {model.norm, model.norm16, model.branch})
    names = save.reduce_unsharded_trainable_grads(model)
    return model, names


GA = 4


def _train(save, native, rank, mode, impl):
    """3 optimizer steps x GA micro-batches, non-final ones under fsdp2_native._no_grad_sync.
    impl: "old" = the previous code (reference sync on every backward), "nodefer" = the new sync on
    every backward, "new" = as shipped (the sync follows FSDP2's gradient sync flag), "oracle" = no
    sync in backward, the reference sync once after the last micro-batch."""
    import torch.distributed as dist

    os.environ["UNSLOTH_FSDP2_GA_SYNC"] = mode
    saved = (save._sync_replicated_grads, save._fsdp_grad_sync_deferred, save.mesh_all_reduce)
    calls = []

    def counting_all_reduce(*args, **kw):
        calls[-1] += 1
        return saved[2](*args, **kw)

    save.mesh_all_reduce = counting_all_reduce
    if impl == "old":
        save._sync_replicated_grads = lambda params, d, mesh = None, buffers = None: _reference_sync(
            params, d, mesh
        )
    if impl in ("old", "nodefer"):
        save._fsdp_grad_sync_deferred = lambda *a: False
    if impl == "oracle":
        save._sync_replicated_grads = lambda *a, **k: None
    try:
        model, names = _build(save)
        rep = [p for n, p in model.named_parameters() if n in set(names)]
        opt = torch.optim.AdamW(model.parameters(), lr = 1e-2)
        gen = torch.Generator().manual_seed(7 + rank)
        out = []
        for step in range(3):
            for micro in range(GA):
                x = torch.randn(4, 8, generator = gen)
                # step 0: `branch` unused everywhere; later only some micro-batches of some ranks.
                use_branch = (micro + rank + step) % 3 == 0 and step != 0
                calls.append(0)
                if micro == GA - 1:
                    model(x, use_branch).backward()
                else:
                    with native._no_grad_sync(model):
                        model(x, use_branch).backward()
            if impl == "oracle":
                _reference_sync(rep, dist)  # counted on the last micro-batch
            grads = {}
            for n, p in model.named_parameters():
                g = p.grad
                if g is not None and hasattr(g, "full_tensor"):
                    g = g.full_tensor()
                grads[n] = None if g is None else g.detach().clone()
            out.append(grads)
            opt.step()
            opt.zero_grad(set_to_none = True)
        return names, out, calls
    finally:
        save._sync_replicated_grads, save._fsdp_grad_sync_deferred, save.mesh_all_reduce = saved


def _diff(a_steps, b_steps):
    bad, worst = [], 0.0
    for step, (a, b) in enumerate(zip(a_steps, b_steps)):
        for n in a:
            x, y = a[n], b[n]
            if (x is None) != (y is None):
                bad.append((step, n, "none"))
            elif x is not None and not torch.equal(x, y):
                bad.append((step, n))
                worst = max(worst, (x.double() - y.double()).abs().max().item())
    return bad, worst


def _case_modes(rank):
    native = _load_native()
    save = _save()
    res = {}
    for mode in ("skip", "defer", "each", "auto"):
        runs = {
            impl: _train(save, native, rank, mode, impl)
            for impl in ("old", "nodefer", "new", "oracle")
        }
        names = runs["old"][0]
        r = {"names": sorted(names), "branch_none_step0": runs["old"][1][0]["branch"] is None}
        r["calls"] = {impl: v[2] for impl, v in runs.items()}
        r["nodefer_vs_old"] = _diff(runs["old"][1], runs["nodefer"][1])[0]
        r["new_vs_old"], r["new_vs_old_max"] = _diff(runs["old"][1], runs["new"][1])
        r["new_vs_oracle"] = _diff(runs["oracle"][1], runs["new"][1])[0]
        res[mode] = r
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

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        res = _case_unit(rank) if case == "unit" else _case_modes(rank)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(case, timeout = 240):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out, case), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + timeout
    while not procs.join(timeout = 5):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail(f"{case}: the ranks did not finish within {timeout} s (hung collective)")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("REPL_SYNC_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_replicated_sync_bitwise_buffer_reuse_and_no_host_read():
    for rank, r in _run("unit").items():
        assert r["mismatch"] == [], (rank, r)
        assert r["alias"] == 0, (rank, r)
        # Steps 2 and 3 (and 4 on rank 0) have every local gradient; the rest leave some None.
        assert r["reads_all_present"] == [0] * (3 if rank == 0 else 2), (rank, r)
        assert r["reads_some_none"] == [1] * (2 if rank == 0 else 3), (rank, r)
        assert r["n_buffers"] == 2 and all(r["reused"].values()), (rank, r)
        assert r["grown_same"] and r["grown_rebuilt"] and r["fp64_reused"], (rank, r)
        assert r["resized_same"] and r["resized_rebuilt"], (rank, r)
        assert r["nobuf_same"], (rank, r)


def test_replicated_sync_follows_the_ga_sync_flag():
    for rank, r in _run("modes", timeout = 400).items():
        for mode, m in r.items():
            assert m["names"] == ["branch", "norm", "norm16"], (rank, mode, m)
            assert m["branch_none_step0"], (rank, mode, m)
            # The new sync on every backward is bitwise the previous code (buffer / host-read change).
            assert m["nodefer_vs_old"] == [], (rank, mode, m)
            # Replicated all_reduces per micro-batch (GA 4, 3 steps): the previous code syncs every
            # backward; shipped code only on the synced one, unless the mode syncs every micro-batch.
            assert m["calls"]["old"] == [1] * (3 * GA), (rank, mode, m)
            deferred = mode != "each"  # auto without a fit layout = skip
            want = ([0] * (GA - 1) + [1]) * 3 if deferred else [1] * (3 * GA)
            assert m["calls"]["new"] == want, (rank, mode, m)
            assert m["calls"]["oracle"] == ([0] * (GA - 1) + [1]) * 3, (rank, mode, m)
            if deferred:
                # Bitwise the accumulated local sum averaged once over the ranks.
                assert m["new_vs_oracle"] == [], (rank, mode, m)
                assert m["new_vs_old_max"] < 1e-2, (rank, mode, m)
            else:
                assert m["new_vs_old"] == [], (rank, mode, m)
