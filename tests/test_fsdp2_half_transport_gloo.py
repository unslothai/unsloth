# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in 16-bit gradient transport for native FSDP2's float32 reduce-scatter, on two and three gloo ranks.

With 16-bit compute and reduce_dtype float32, the installed comm moves 16-bit gradients as 16-bit and adds them
in float32. The sharded float32 gradients must equal FSDP2's own reduce-scatter on every rank: bitwise at two
ranks (one float32 add), within float32 summation order at three. Anything that would make ranks pick different
collectives (accumulated gradients on one rank only, mixed switches or piece sizes) must agree instead, and a
reduce no rank can carry exactly takes the original collective on every rank.
"""

from __future__ import annotations

import datetime
import multiprocessing
import os
from types import SimpleNamespace

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
fsdp_pkg = pytest.importorskip("torch.distributed.fsdp")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

requires_custom_reduce_scatter = pytest.mark.skipif(
    not hasattr(getattr(fsdp_pkg, "FSDPModule", None), "set_custom_reduce_scatter"),
    reason = "torch without FSDP2 custom reduce-scatter comms: nothing is installed",
)


def _reduce_module():
    # Loaded by path: the module needs only torch, and importing the unsloth package wants an accelerator.
    import importlib.util
    import sys

    name = "_unsloth_fsdp2_reduce_under_test"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "fsdp2_reduce.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


class _Block(torch.nn.Sequential):
    def forward(
        self, x
    ):  # residual: a block whose inner path is cut still passes gradients upstream
        return x + super().forward(x)


def _model(features = 32, hidden = 64):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        *[
            _Block(
                torch.nn.Linear(features, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, features),
            )
            for _ in range(3)
        ]
    )


def _per_rank(value, rank):
    return value[rank] if isinstance(value, (list, tuple)) else value


def _grads(rank, world, dtype, transport, w):
    from torch.distributed import fsdp
    from torch.distributed.device_mesh import init_device_mesh

    install_half_transport_reduce = _reduce_module().install_half_transport_reduce

    os.environ["UNSLOTH_FSDP2_REDUCE_TRANSPORT"] = (
        _per_rank(w["transport"], rank) if transport else "off"
    )
    os.environ["UNSLOTH_FSDP2_REDUCE_TRANSPORT_MIB"] = _per_rank(w["piece_mib"], rank)
    features = w["features"]
    model = _model(features, w["hidden"])
    policy = fsdp.MixedPrecisionPolicy(param_dtype = dtype, reduce_dtype = torch.float32)
    mesh = init_device_mesh("cpu", (world,))  # the gloo world is CPU, also on a host with CUDA
    for block in model:
        fsdp.fully_shard(block, mesh = mesh, mp_policy = policy)
    fsdp.fully_shard(model, mesh = mesh, mp_policy = policy)
    comms = install_half_transport_reduce([*model, model])
    if w["reinstall"]:
        again = install_half_transport_reduce([*model, model])
        assert len(again) == len(comms) and all(
            a is b for a, b in zip(again, comms)
        ), "re-install stacked"
    if w["setter"]:
        for block in model:
            block.set_allocate_memory_from_process_group_for_comm(False)
    if w["unused"]:
        model.set_reduce_scatter_unused_params(True)
    state = {"i": 0}
    if w["conditional"] and rank == 1:
        # Rank 1 sends no gradient into block 1's parameters on the first micro-batch: only rank 0 accumulates
        # there. The residual still runs every block's backward, so the stock collectives match.
        model[1][2].register_forward_hook(
            lambda _m, _i, out: out.detach() if state["i"] == 0 else out
        )
    g = torch.Generator().manual_seed(100 + rank)
    pattern = w["sync_pattern"]
    for i in range(len(pattern)):
        state["i"] = i
        if i and pattern[i - 1]:
            model.zero_grad(set_to_none = False)
        model.set_requires_gradient_sync(pattern[i])
        x = torch.randn(8, features, generator = g).to(dtype)
        (model(x).float().square().mean() * 3.0).backward()
    grads = [p.grad.to_local().detach().clone() for p in model.parameters()]
    calls = sum(c.calls for c in comms)
    fallbacks = sum(c.fallbacks for c in comms)
    return grads, len(comms), calls, fallbacks


def _worker(rank, world, rendezvous, dtype_name, out, w):
    dist.init_process_group(
        "gloo",
        init_method = rendezvous,
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),
    )
    try:
        dtype = getattr(torch, dtype_name)
        ref, n_ref, _, _ = _grads(rank, world, dtype, False, w)
        new, n_new, calls, fallbacks = _grads(rank, world, dtype, True, w)
        assert {t.dtype for t in new} == {torch.float32}
        mismatch = int(not all(torch.equal(a, b) for a, b in zip(ref, new)))
        max_rel = max(
            float((a - b).abs().max() / a.abs().max().clamp_min(1e-30)) for a, b in zip(ref, new)
        )
        metrics = torch.tensor([mismatch, max_rel], dtype = torch.float64)
        dist.all_reduce(metrics, op = dist.ReduceOp.MAX)  # every rank's shard, not rank 0's alone
        counts = torch.tensor([n_ref, n_new, calls, fallbacks], dtype = torch.int64)
        gathered = [torch.zeros_like(counts) for _ in range(world)]
        dist.all_gather(gathered, counts)
        if rank == 0:
            rows = [t.tolist() for t in gathered]
            with open(out, "w") as f:
                f.write(f"{rows}|{metrics[0].item() == 0}|{metrics[1].item()}")
    finally:
        dist.destroy_process_group()


def _run(
    tmp_path,
    world,
    dtype_name,
    micro = 1,
    sync_each = True,
    piece_mib = "32",
    transport = "on",
    sync_pattern = None,
    features = 32,
    hidden = 64,
    conditional = False,
    reinstall = False,
    setter = False,
    unused = False,
):
    """Returns (n_ref, n_new, calls, fallbacks) from rank 0, every rank's counts, bitwise, max_rel."""
    import torch.multiprocessing as mp

    if sync_pattern is None:
        sync_pattern = tuple(sync_each or i == micro - 1 for i in range(micro))
    w = dict(
        transport = transport,
        piece_mib = piece_mib,
        sync_pattern = tuple(sync_pattern),
        features = features,
        hidden = hidden,
        conditional = conditional,
        reinstall = reinstall,
        setter = setter,
        unused = unused,
    )
    out = tmp_path / "result.txt"
    rendezvous = (tmp_path / "store").resolve().as_uri()
    mp.spawn(_worker, args = (world, rendezvous, dtype_name, str(out), w), nprocs = world, join = True)
    rows, bitwise, max_rel = out.read_text().split("|")
    rows = eval(rows)  # noqa: S307 - written by this test's own ranks
    assert all(
        r == rows[0] for r in rows
    ), rows  # every rank installed and carried the same reduces
    return (*rows[0], bitwise == "True", float(max_rel))


@requires_custom_reduce_scatter
@pytest.mark.parametrize("piece_mib", ["32", "0.0005"])  # one piece per unit / 131-element pieces
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_two_ranks_bitwise_equal_to_float32_reduce_scatter(tmp_path, dtype_name, piece_mib):
    n_ref, n_new, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, dtype_name, micro = 2, piece_mib = piece_mib
    )
    assert (
        n_ref == 0 and n_new == 3
    )  # three blocks (the root holds no parameters, so no param group)
    assert (calls, fallbacks) == (2 * 3, 0)  # every block reduces each micro-batch
    assert bitwise, max_rel


@requires_custom_reduce_scatter
@pytest.mark.parametrize("world", [2, 3])
def test_uneven_shards_and_pieces(tmp_path, world):
    _, _, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, world, "bfloat16", micro = 2, piece_mib = "0.0005", features = 33, hidden = 65
    )
    assert (calls, fallbacks) == (6, 0)
    assert bitwise if world == 2 else max_rel < 1e-6, max_rel


@requires_custom_reduce_scatter
def test_three_ranks_float32_summation_order_only(tmp_path):
    _, _, calls, fallbacks, _, max_rel = _run(tmp_path, 3, "bfloat16", micro = 2, piece_mib = "0.0005")
    assert (calls, fallbacks) == (6, 0)
    assert max_rel < 1e-6


@requires_custom_reduce_scatter
def test_float32_accumulated_micro_batches_take_the_original_collective(tmp_path):
    _, _, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", micro = 3, sync_each = False
    )
    assert (calls, fallbacks) == (0, 3)  # one reduce per block, of a float32 sum over micro-batches
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_conditional_no_sync_accumulation_agrees_on_the_stock_collective(tmp_path):
    # Only rank 0 accumulated blocks 0 and 1: every rank must still run the same collective.
    _, _, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", micro = 2, sync_each = False, conditional = True
    )
    assert (calls, fallbacks) == (0, 3)
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_transport_reengages_after_accumulation(tmp_path):
    _, _, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", sync_pattern = (False, True, True)
    )
    assert (calls, fallbacks) == (3, 3)
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_mixed_switch_across_ranks_agrees_off(tmp_path):
    n_ref, n_new, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", micro = 2, transport = ["on", "off"]
    )
    assert (n_new, calls, fallbacks) == (0, 0, 0)
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_mixed_piece_size_across_ranks_agrees(tmp_path):
    _, n_new, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", micro = 2, piece_mib = ["0.0005", "32"]
    )
    assert (n_new, calls, fallbacks) == (3, 6, 0)
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_transport_installation_is_idempotent(tmp_path):
    _, n_new, calls, fallbacks, bitwise, max_rel = _run(tmp_path, 2, "bfloat16", reinstall = True)
    assert (n_new, calls, fallbacks) == (3, 3, 0)
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_public_allocator_setter_replaces_the_comm(tmp_path):
    _, n_new, calls, fallbacks, bitwise, max_rel = _run(tmp_path, 2, "bfloat16", setter = True)
    assert (n_new, calls, fallbacks) == (3, 0, 0)  # the setter's comm reduces; ours is detached
    assert bitwise, max_rel


@requires_custom_reduce_scatter
@pytest.mark.skipif(
    not hasattr(getattr(fsdp_pkg, "FSDPModule", None), "set_reduce_scatter_unused_params"),
    reason = "torch without reduce_scatter_unused_params",
)
def test_unused_param_padding_keeps_the_stock_collective(tmp_path):
    _, _, calls, fallbacks, bitwise, max_rel = _run(
        tmp_path, 2, "bfloat16", micro = 2, sync_each = False, conditional = True, unused = True
    )
    assert calls == 0 and fallbacks == 3
    assert bitwise, max_rel


@requires_custom_reduce_scatter
def test_float32_compute_carries_nothing(tmp_path):
    n_ref, n_new, calls, fallbacks, bitwise, _ = _run(tmp_path, 2, "float32")
    assert (n_ref, n_new, calls, fallbacks) == (
        0,
        3,
        0,
        3,
    )  # float32 gradients: no 16-bit gradients to move
    assert bitwise


def _avg_worker(rank, world, rendezvous, out):
    from torch.distributed import fsdp
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group(
        "gloo",
        init_method = rendezvous,
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),
    )
    try:
        os.environ["UNSLOTH_FSDP2_REDUCE_TRANSPORT"] = "on"
        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        mesh = init_device_mesh("cpu", (world,))
        policy = fsdp.MixedPrecisionPolicy(param_dtype = torch.bfloat16, reduce_dtype = torch.float32)
        fsdp.fully_shard(model, mesh = mesh, mp_policy = policy)
        (comm,) = _reduce_module().install_half_transport_reduce([model])
        model(torch.randn(2, 8).bfloat16()).float().sum().backward()  # lazy init: dtypes resolved
        big = float(
            torch.tensor(3.0e38).bfloat16()
        )  # a finite bf16 value; two of them overflow float32
        inp = torch.full((world * 4,), big, dtype = torch.float32)
        output = torch.empty(4, dtype = torch.float32)
        before, comm.fresh = comm.calls, True
        comm(output, inp, group = dist.group.WORLD, op = dist.ReduceOp.AVG)
        if rank == 0:
            with open(out, "w") as f:
                f.write(
                    f"{comm.calls - before}|{bool(torch.isfinite(output).all())}|{output[0].item()}|{big}"
                )
    finally:
        dist.destroy_process_group()


@requires_custom_reduce_scatter
def test_avg_of_finite_bf16_sums_does_not_overflow(tmp_path):
    import torch.multiprocessing as mp

    out = tmp_path / "avg.txt"
    rendezvous = (tmp_path / "store").resolve().as_uri()
    mp.spawn(_avg_worker, args = (2, rendezvous, str(out)), nprocs = 2, join = True)
    calls, finite, value, big = out.read_text().split("|")
    assert int(calls) == 1
    assert finite == "True" and float(value) == float(
        big
    )  # (a + a) / 2 == a, as NCCL's scaled average


def _param(
    dtype,
    requires_grad = True,
    **extra,
):
    return SimpleNamespace(
        sharded_param = torch.nn.Parameter(torch.empty(1, dtype = dtype), requires_grad = requires_grad)
        if dtype.is_floating_point
        else torch.empty(1, dtype = dtype),
        param_dtype = torch.bfloat16 if requires_grad else None,
        orig_dtype = dtype,
        reduce_dtype = torch.float32,
        **extra,
    )


def test_frozen_parameters_do_not_block_trainable_half_gradients():
    group = SimpleNamespace(fsdp_params = [_param(torch.uint8, False), _param(torch.float32)])
    assert _reduce_module()._group_grad_dtype(group) == torch.bfloat16
    group = SimpleNamespace(fsdp_params = [_param(torch.float32, False), _param(torch.float32)])
    assert _reduce_module()._group_grad_dtype(group) == torch.bfloat16


def test_gradients_reduced_outside_dp_keep_the_stock_collective():
    for extra in (
        {"may_reduce_grad_outside_dp": True},
        {"is_dtensor": True},
        {"_has_sharded_grad_dtype_override": True},
    ):
        group = SimpleNamespace(fsdp_params = [_param(torch.float32, **extra)])
        assert _reduce_module()._group_grad_dtype(group) is None, extra


def test_wide_source_gradient_is_not_carried():
    accepts = _reduce_module()._all_have_dtype
    wide = torch.tensor([1.001], dtype = torch.float32)
    assert not torch.equal(wide, wide.bfloat16().float())
    assert not accepts([wide], torch.bfloat16)
    assert not accepts([wide.bfloat16(), wide], torch.bfloat16)
    assert accepts([wide.bfloat16()], torch.bfloat16)
    assert not accepts([], torch.bfloat16)


def test_switch_is_opt_in(monkeypatch):
    enabled = _reduce_module().half_transport_enabled
    monkeypatch.delenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT", raising = False)
    assert not enabled()
    for value in ("off", "0", "auto_typo", ""):
        monkeypatch.setenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT", value)
        assert not enabled(), value
    for value in ("on", "1", "true", "ON"):
        monkeypatch.setenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT", value)
        assert enabled(), value


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-inf", "1e308", "0", "-5"])
def test_invalid_piece_budget_uses_default(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT_MIB", value)
    assert _reduce_module()._piece_bytes() == 32 * 2**20
