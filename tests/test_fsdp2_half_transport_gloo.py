# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""16-bit gradient transport for native FSDP2's float32 reduce-scatter, on two and three gloo ranks.

With 16-bit compute and reduce_dtype float32, the installed comm moves each fresh 16-bit gradient as 16-bit
and adds it in float32. The sharded float32 gradients must equal FSDP2's own reduce-scatter: bitwise at two
ranks (one float32 add), within float32 summation order at three. A micro-batch that skipped its reduce
leaves a float32-accumulated gradient, which must take the original collective (fallback count > 0).
"""

from __future__ import annotations

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)


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


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        *[
            torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32))
            for _ in range(3)
        ]
    )


def _grads(rank, world, dtype, transport, micro, sync_each):
    from torch.distributed import fsdp
    from torch.distributed.device_mesh import init_device_mesh

    install_half_transport_reduce = _reduce_module().install_half_transport_reduce

    os.environ["UNSLOTH_FSDP2_REDUCE_TRANSPORT"] = "auto" if transport else "off"
    model = _model()
    policy = fsdp.MixedPrecisionPolicy(param_dtype = dtype, reduce_dtype = torch.float32)
    mesh = init_device_mesh("cpu", (world,))  # the gloo world is CPU, also on a host with CUDA
    for block in model:
        fsdp.fully_shard(block, mesh = mesh, mp_policy = policy)
    fsdp.fully_shard(model, mesh = mesh, mp_policy = policy)
    comms = install_half_transport_reduce([*model, model])
    g = torch.Generator().manual_seed(100 + rank)
    for i in range(micro):
        if not sync_each:
            model.set_requires_gradient_sync(i == micro - 1)
        x = torch.randn(8, 32, generator = g).to(dtype)
        (model(x).float().square().mean() * 3.0).backward()
    grads = [p.grad.to_local().detach().clone() for p in model.parameters()]
    calls = sum(c.calls for c in comms)
    fallbacks = sum(c.fallbacks for c in comms)
    return grads, len(comms), calls, fallbacks


def _worker(rank, world, port, dtype_name, micro, sync_each, out, piece_mib):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    os.environ["UNSLOTH_FSDP2_REDUCE_TRANSPORT_MIB"] = piece_mib
    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        dtype = getattr(torch, dtype_name)
        ref, n_ref, calls_ref, _ = _grads(rank, world, dtype, False, micro, sync_each)
        new, n_new, calls, fallbacks = _grads(rank, world, dtype, True, micro, sync_each)
        bitwise = all(torch.equal(a, b) for a, b in zip(ref, new))
        max_rel = max(
            float((a - b).abs().max() / a.abs().max().clamp_min(1e-30)) for a, b in zip(ref, new)
        )
        dtypes = {str(t.dtype) for t in new}
        if rank == 0:
            with open(out, "w") as f:
                f.write(
                    f"{n_ref}|{n_new}|{calls_ref}|{calls}|{fallbacks}|{bitwise}|{max_rel}|{dtypes}"
                )
    finally:
        dist.destroy_process_group()


def _run(
    tmp_path,
    world,
    dtype_name,
    micro = 1,
    sync_each = True,
    piece_mib = "32",
):
    import torch.multiprocessing as mp

    out = str(tmp_path / f"r_{world}_{dtype_name}_{micro}_{sync_each}_{piece_mib}.txt")
    mp.spawn(
        _worker,
        args = (world, _free_port(), dtype_name, micro, sync_each, out, piece_mib),
        nprocs = world,
        join = True,
    )
    n_ref, n_new, calls_ref, calls, fallbacks, bitwise, max_rel, dtypes = (
        open(out).read().split("|")
    )
    return (
        int(n_ref),
        int(n_new),
        int(calls),
        int(fallbacks),
        bitwise == "True",
        float(max_rel),
        dtypes,
    )


@pytest.mark.parametrize("piece_mib", ["32", "0.0005"])  # one piece per unit / 131-element pieces
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_two_ranks_bitwise_equal_to_float32_reduce_scatter(tmp_path, dtype_name, piece_mib):
    n_ref, n_new, calls, fallbacks, bitwise, max_rel, dtypes = _run(
        tmp_path, 2, dtype_name, micro = 2, piece_mib = piece_mib
    )
    assert n_new == 3  # three blocks (the root holds no parameters, so no param group)
    assert (
        calls == 2 * 3 and fallbacks == 0
    )  # every block reduces each micro-batch; root holds no params
    assert bitwise, max_rel
    assert dtypes == "{'torch.float32'}"


def test_three_ranks_float32_summation_order_only(tmp_path):
    _, _, calls, fallbacks, _, max_rel, _ = _run(
        tmp_path, 3, "bfloat16", micro = 2, piece_mib = "0.0005"
    )
    assert calls == 6 and fallbacks == 0
    assert max_rel < 1e-6


def test_float32_accumulated_micro_batches_take_the_original_collective(tmp_path):
    _, _, calls, fallbacks, bitwise, max_rel, _ = _run(
        tmp_path, 2, "bfloat16", micro = 3, sync_each = False
    )
    assert (
        calls == 0 and fallbacks == 3
    )  # one reduce per block, of a float32 sum over micro-batches
    assert bitwise, max_rel


def test_off_switch_and_float32_compute_install_nothing(tmp_path, monkeypatch):
    half_transport_enabled = _reduce_module().half_transport_enabled

    monkeypatch.setenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT", "off")
    assert not half_transport_enabled()
    monkeypatch.setenv("UNSLOTH_FSDP2_REDUCE_TRANSPORT", "auto")
    assert half_transport_enabled()
    n_ref, n_new, calls, fallbacks, bitwise, _, _ = _run(tmp_path, 2, "float32")
    assert calls == 0 and fallbacks == 3 and bitwise  # float32 compute: no 16-bit gradients to move
