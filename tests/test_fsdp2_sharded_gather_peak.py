# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 sharded loading gathers a replicated table (the frozen lm_head kept whole, Qwen3.8-27B on a
T4) from this rank's chunk. The chunk must leave the GPU before the full table is allocated: holding both
put the sharding peak at 13.9 GiB of a T4's 14.56."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")

MIB = 1 << 20


@pytest.fixture(scope = "module")
def mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _rendezvous_port import free_port

    own = not dist.is_initialized()
    if own:
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(free_port()))
        dist.init_process_group("cuda:gloo,cpu:gloo", rank = 0, world_size = 1)
    yield init_device_mesh("cuda", (1,))
    if own:
        dist.destroy_process_group()


def _pending(
    rows,
    cols,
    shared = False,
):
    """A model whose head weight is a sharded-load chunk (world 1: the whole table) behind its meta
    placeholder, as begin_shard leaves it."""
    import importlib

    _load_native()
    load = importlib.import_module("unsloth.models.fsdp2_load")
    torch.manual_seed(0)
    chunk = torch.randn(rows, cols, device = "cuda").to(torch.float16)
    expect = chunk.clone().cpu()
    model = nn.Module()
    model.head = nn.Linear(cols, rows, bias = False, device = "meta", dtype = torch.float16)
    meta = model.head.weight
    meta.requires_grad_(False)
    meta.__dict__["_unsloth_fsdp2_pending"] = True
    info = {
        "shape": (rows, cols),
        "world": 1,
        "rank": 0,
        "ranks": [0],
        "rows": (0, rows),
        "packed": False,
    }
    pending = {(id(model.head), "weight"): (chunk, info)}
    if shared:
        model.twin = nn.Linear(cols, rows, bias = False, device = "meta", dtype = torch.float16)
        model.twin._parameters["weight"] = meta
        pending[(id(model.twin), "weight")] = (chunk, info)
    del chunk
    return load, model, pending, expect


def test_chunk_leaves_the_gpu_before_the_full_table(mesh, monkeypatch):
    load, model, pending, expect = _pending(4096, 32768)  # 256 MiB float16
    monkeypatch.setattr(load, "_GATHER_SLICE_BYTES", 8 * MIB)
    replicated = {model.head.weight}
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    done = load.gather_replicated(model, pending, replicated, mesh, device = torch.device("cuda"))
    torch.cuda.synchronize()
    over = torch.cuda.max_memory_allocated() - start
    assert done == 1 and not pending
    weight = model.head.weight
    assert weight.device.type == "cuda" and weight in replicated
    assert torch.equal(weight.cpu(), expect)  # bytes unchanged
    # Base held chunk + full (256 MiB over the start); now only the gather slices.
    assert over <= 64 * MIB, f"{over / MIB:.0f} MiB over the start"
    assert torch.cuda.memory_allocated() - start <= 1 * MIB


def test_shared_chunk_is_gathered_once_and_exact(mesh, monkeypatch):
    load, model, pending, expect = _pending(512, 4096, shared = True)
    monkeypatch.setattr(load, "_GATHER_SLICE_BYTES", 1 * MIB)
    replicated = {model.head.weight}
    done = load.gather_replicated(model, pending, replicated, mesh, device = torch.device("cuda"))
    assert done == 1 and not pending
    assert model.head.weight is model.twin.weight
    assert torch.equal(model.head.weight.cpu(), expect)


def test_gather_to_host_frees_the_gpu_chunk_first(mesh, monkeypatch):
    load, model, pending, expect = _pending(4096, 32768)
    monkeypatch.setattr(load, "_GATHER_SLICE_BYTES", 8 * MIB)
    replicated = {model.head.weight}
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    done = load.gather_replicated(model, pending, replicated, mesh, device = torch.device("cpu"))
    assert done == 1
    assert model.head.weight.device.type == "cpu"
    assert torch.equal(model.head.weight, expect)
    assert start - torch.cuda.memory_allocated() >= 255 * MIB  # the GPU chunk is gone
