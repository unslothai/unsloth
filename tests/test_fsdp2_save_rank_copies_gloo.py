# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full DCP loads of sharded NF4 weights (`_restore_quant_state`), two gloo ranks on CPU.

- broadcast_from_rank0: only rank 0 copies its QuantState entries to the host; a peer's own dict is a
  placeholder (empty, real tensors or meta tensors) that it neither copies nor reads.
- A packed weight saved without its QuantState (an older checkpoint) is compared with the live bytes
  only on the rank holding the loaded data, and the verdict is agreed across ranks, so a rank whose
  bytes differ cannot raise while the others go on loading.
The module is loaded from its file (importing unsloth needs a GPU).
"""

import datetime
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]

fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)

WORLD = 2


def _load(name):
    path = _ROOT / "unsloth" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sharded_nf4(m):
    import bitsandbytes as bnb
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    def lin(seed):
        torch.manual_seed(seed)
        layer = bnb.nn.Linear4bit(
            128,
            96,
            bias = False,
            compute_dtype = torch.bfloat16,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
        )
        layer.weight = bnb.nn.Params4bit(
            torch.randn(96, 128) * 0.05,
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            module = layer,
        )
        return layer.to("cpu")  # quantizes the weight

    model = torch.nn.Sequential(lin(0), lin(1))
    _load("fsdp2_qlora").prepare_fsdp2_qlora(model)
    mesh = init_device_mesh("cpu", (WORLD,))
    for layer in model:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    m.install_dcp_bnb_state_dict()
    return model


def _saved(model):
    import torch.distributed.checkpoint.state_dict as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    full = StateDictOptions(full_state_dict = True)  # every rank gets the full dict
    return {k: v.clone() for k, v in dcp.get_model_state_dict(model, options = full).items()}


def _counting_cpu(fn):
    """Runs fn, returning (its result or the error it raised, the number of Tensor.cpu() calls made
    by fsdp2_save itself; on CPU ranks `.cpu()` is a no-op, so the calls are recorded)."""
    original = torch.Tensor.cpu
    calls = []

    def recording(self, *args, **kwargs):
        if sys._getframe(1).f_code.co_filename.endswith("fsdp2_save.py"):
            calls.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    torch.Tensor.cpu = recording
    try:
        result = fn()
    except Exception as error:
        result = f"{type(error).__name__}: {error}"
    finally:
        torch.Tensor.cpu = original
    return result, len(calls)


def _case(rank, case, m):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    model = _sharded_nf4(m)
    saved = _saved(model)
    states = [layer.weight.quant_state for layer in model]
    before = [s.absmax.clone() for s in states]
    for s in states:
        s.absmax.mul_(2)  # a requantized / differently initialized base
    legacy = {k: v for k, v in saved.items() if not k.startswith("0.weight.")}
    broadcast = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True)

    def restore(state_dict, options = broadcast):
        def run():
            m._restore_quant_state(model, state_dict, options)
            return "loaded"

        return run

    def meta(state_dict):
        return {k: torch.empty_like(v, device = "meta") for k, v in state_dict.items()}

    out = {}
    if case == "meta_peer":
        # A peer's placeholder of meta tensors (a model built on the meta device elsewhere).
        out["full"], _ = _counting_cpu(restore(saved if rank == 0 else meta(saved)))
        out["restored"] = all(torch.equal(s.absmax, b) for s, b in zip(states, before))
    elif case == "meta_peer_legacy":
        out["legacy"], _ = _counting_cpu(restore(legacy if rank == 0 else meta(legacy)))
    elif case == "peer_copies":
        # A peer passing real tensors: they are replaced by rank 0's, so it must not copy them.
        out["full"], out["full_cpu"] = _counting_cpu(restore(saved))
        out["restored"] = all(torch.equal(s.absmax, b) for s, b in zip(states, before))
        out["legacy"], out["legacy_cpu"] = _counting_cpu(restore(legacy))
        out["layer1_entries"] = sum(k.startswith("1.weight.") for k in saved)
    elif case == "divergent":
        # No broadcast: every rank loads its own dict. Rank 1's layer 0 bytes are not the live ones.
        mine = dict(legacy)
        if rank == 1:
            mine["0.weight"] = saved["1.weight"].clone()
        loose = StateDictOptions(full_state_dict = True)
        out["legacy"], _ = _counting_cpu(restore(mine, loose))
        out["legacy"] = "ValueError" if "`0.weight`" in out["legacy"] else out["legacy"]
    return out


def _worker(rank, port, case, queue):
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(WORLD),
        )
        import torch.distributed as dist

        # A rank left waiting on a collective its peer never joined fails instead of hanging.
        dist.init_process_group(
            "gloo", rank = rank, world_size = WORLD, timeout = datetime.timedelta(seconds = 30)
        )
        result = _case(rank, case, _load("fsdp2_save"))
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        queue.put((rank, {"error": f"{type(error).__name__}: {error}"}))


def _run(case):
    import torch.multiprocessing as mp
    from tests._rendezvous_port import free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target = _worker, args = (r, port, case, queue)) for r in range(WORLD)]
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # inherited by the spawned children before torch loads
    try:
        for p in procs:
            p.start()
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    results = {}
    try:
        for _ in procs:
            rank, result = queue.get(timeout = 240)
            results[rank] = result
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.terminate()
    return results


@pytest.fixture(autouse = True)
def _needs_bitsandbytes():
    pytest.importorskip("bitsandbytes")


def test_broadcast_load_with_a_meta_placeholder_on_the_peer():
    """The peer copied its own placeholder to the host before the broadcast replaced it, which
    raises on meta tensors (and left rank 0 waiting in the broadcast)."""
    results = _run("meta_peer")
    for rank in range(WORLD):
        assert results[rank].get("full") == "loaded", results
        assert results[rank].get("restored") is True, results


def test_broadcast_legacy_load_does_not_read_the_peer_placeholder():
    """A packed weight without its QuantState: the peer compared its own (meta) placeholder bytes
    with the live ones on the host, which raises on meta tensors."""
    results = _run("meta_peer_legacy")
    for rank in range(WORLD):
        assert results[rank].get("legacy") == "loaded", results


def test_broadcast_load_copies_to_the_host_only_on_rank0():
    results = _run("peer_copies")
    for rank in range(WORLD):
        assert results[rank].get("full") == "loaded", results
        assert results[rank].get("restored") is True, results
        assert results[rank].get("legacy") == "loaded", results
    # The peer's dict is replaced by rank 0's: no host copy of it, nor of the live bytes.
    assert results[1]["full_cpu"] == 0, results
    assert results[1]["legacy_cpu"] == 0, results
    # Rank 0 copies only what it sends (layer 1's entries); the byte check stays on device.
    assert results[0]["legacy_cpu"] == results[0]["layer1_entries"], results


def test_legacy_byte_check_verdict_is_agreed_across_ranks():
    """Without broadcast_from_rank0 each rank decided alone: rank 1 (other bytes) raised while rank 0
    went on loading into the next collective, where it would wait for rank 1 forever."""
    results = _run("divergent")
    assert results[0].get("legacy") == "ValueError", results
    assert results[1].get("legacy") == "ValueError", results
