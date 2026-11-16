# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Decode KV buffers on CUDA virtual memory: sleep() hands the pages back (free device memory grows), wake() maps
new zeroed pages at the same addresses, and a CUDA graph captured on the buffer replays correctly afterwards."""

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)
from unsloth.models import _decode_vmm as V

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not V.available(torch.device("cuda", torch.cuda.current_device())),
        reason = "CUDA virtual memory management",
    ),
]

MB = 1 << 20


def test_sleep_returns_memory_and_wake_keeps_addresses():
    device = torch.device("cuda", torch.cuda.current_device())
    t = V.zeros((256, 1024, 1024), torch.bfloat16, device)  # 512 MB
    assert t is not None and torch.count_nonzero(t).item() == 0
    ptr = t.data_ptr()
    t.fill_(3)
    torch.cuda.synchronize()
    awake = torch.cuda.mem_get_info()[0]
    assert V.sleep(t) >= 512 * MB
    asleep = torch.cuda.mem_get_info()[0]
    # Other processes may allocate meanwhile: most of the 512 MB must come back.
    assert asleep - awake >= 400 * MB, (awake, asleep)
    assert V.sleep(t) == 0  # idempotent
    assert V.wake(t) is True and V.wake(t) is False
    assert t.data_ptr() == ptr and torch.count_nonzero(t).item() == 0


def test_graph_replays_on_woken_pages():
    device = torch.device("cuda", torch.cuda.current_device())
    cache = V.zeros((64, 4096), torch.float32, device)
    x = torch.randn(4096, device = device)
    slot = torch.zeros((), dtype = torch.long, device = device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache.index_copy_(0, slot.view(1), (x * 2).view(1, -1))  # warmup
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.index_copy_(0, slot.view(1), (x * 2).view(1, -1))
        out = cache.sum(0)
    for cycle in range(3):
        V.sleep(cache)
        V.wake(cache)
        slot.fill_(cycle)
        x.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cache[cycle], x * 2) and torch.equal(out, cache.sum(0)), cycle
