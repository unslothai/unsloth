# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 generate with partial block residency.

When the whole gather does not fit, "auto" used to stream every decoder block (one blocking all-gather
per block per decoded token). It now keeps the longest prefix of blocks that fits on every rank gathered
for the whole call and streams only the rest. Checks: the plan picks the largest prefix every rank can
hold (and the old all-or-nothing answer with UNSLOTH_FSDP2_GENERATE_RESIDENT=0); on a gloo world of 2
greedy tokens equal one process while only the streamed blocks are gathered per step."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

F = _load_native()

GIB = 1 << 30
MIB = 1 << 20


class _Unit(torch.nn.Module):
    pass


class _Net(torch.nn.Module):
    def __init__(self, blocks = 6):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Unit() for _ in range(blocks)])


def _plan(
    monkeypatch,
    free,
    block = 100 * MIB,
    root = 300 * MIB,
    blocks = 6,
    other_free = None,
):
    """_generate_gather_plan on a fake CUDA world: units = root + `blocks` blocks of `block` bytes.
    other_free: a second rank's free memory; the agreement ANDs its flags with this rank's."""
    import torch.distributed as dist

    net = _Net(blocks)
    units = [net] + list(net.layers)
    sizes = {id(net): root, **{id(m): block for m in net.layers}}
    group = types.SimpleNamespace(
        mesh_info = types.SimpleNamespace(mesh = None), device = torch.device("cuda", 0)
    )
    monkeypatch.setattr(F, "_fsdp_module_class", lambda: torch.nn.Module)
    monkeypatch.setattr(F, "is_native_fsdp2", lambda model: True)
    monkeypatch.setattr(F, "_generate_gather_mode", lambda: "auto")
    monkeypatch.setattr(
        F, "_stream_block_lists", lambda model, us: [(net.layers, dict(enumerate(net.layers)))]
    )
    monkeypatch.setattr(F, "_unit_param_groups", lambda unit: [group])
    monkeypatch.setattr(F, "_unit_gathered_bytes", lambda unit: sizes[id(unit)])
    monkeypatch.setattr(F, "_synchronized_empty_cache", lambda: None)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    model = types.SimpleNamespace(modules = lambda: iter(units))

    def run(free_bytes, agree):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (free_bytes, free_bytes))
        monkeypatch.setattr(F, "_agree_over_ranks", agree)
        return F._generate_gather_plan(model)

    if other_free is None:
        return run(free, lambda d, dev, values, mesh: values)
    other = {}

    def capture(d, dev, values, mesh):
        other["flags"] = list(values)
        return values

    run(other_free, capture)
    return run(free, lambda d, dev, values, mesh: [v and o for v, o in zip(values, other["flags"])])


def test_plan_whole_gather_when_it_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # root 300 + 6 x 100 + one 300 buffer + 1 GiB scratch
    assert _plan(monkeypatch, free = 4 * GIB) == (False, 0)


def test_plan_keeps_the_prefix_that_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # k resident needs 300 + 100 k + 200 (two streamed buffers) MiB + 1 GiB; 1024 + 850 fits k = 3.
    stream, resident = _plan(monkeypatch, free = GIB + 850 * MIB)
    assert stream is True and resident == 3
    # Nothing beyond the root and two buffers fits: every block streams (the old answer).
    assert _plan(monkeypatch, free = GIB + 520 * MIB) == (True, 0)


def test_plan_resident_zero_restores_all_or_nothing(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "0")
    assert _plan(monkeypatch, free = GIB + 850 * MIB) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "2")
    assert _plan(monkeypatch, free = GIB + 850 * MIB) == (True, 2)  # capped by the override
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "-1")
    with pytest.raises(ValueError):
        _plan(monkeypatch, free = GIB + 850 * MIB)


def test_plan_every_rank_picks_the_same_prefix(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # This rank could hold 5 blocks, the other only 2: both must keep 2.
    assert _plan(monkeypatch, free = GIB + 1050 * MIB, other_free = GIB + 720 * MIB) == (True, 2)


def test_block_mode_resident_override(monkeypatch):
    monkeypatch.setattr(F, "_fsdp_module_class", lambda: torch.nn.Module)
    monkeypatch.setattr(F, "is_native_fsdp2", lambda model: True)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_GATHER", "block")
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    assert F._generate_gather_plan(_Net()) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "4")
    assert F._generate_gather_plan(_Net()) == (True, 4)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_GATHER", "all")
    assert F._generate_gather_plan(_Net()) == (False, 0)


def test_generate_keeps_resident_prefix_gathered(monkeypatch):
    """gloo world of 2, UNSLOTH_FSDP2_GENERATE_GATHER=block + 3 of 6 blocks resident: greedy tokens equal
    one process (HF generate and a decode loop that reads weights directly), the 3 resident blocks stay
    gathered (peak at least root + 3 blocks, which pure streaming never reaches) and only the other 3
    stream (peak below root + 3 + 2 blocks)."""
    pytest.importorskip("transformers")
    from test_fsdp2_native_generate_streams_blocks_gloo import _reference, _run

    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "3")
    want = _reference()
    for rank, r in _run("block").items():
        for name in ("hf", "loop"):
            got = r[name]
            assert "error" not in got, (rank, name, got)
            assert got["tokens"] == want[name], (rank, name, got, want[name])
            assert got["sharded_after"] and got["swap_cleared"], (rank, name, got)
            assert got["peak"] >= got["root"] + 3 * got["block"], (rank, name, got)
            assert got["peak"] <= got["root"] + 5 * got["block"], (rank, name, got)
            assert got["peak"] < got["total"], (rank, name, got)
