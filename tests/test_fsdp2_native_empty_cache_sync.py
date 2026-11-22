# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2: every `torch.cuda.empty_cache()` is preceded by a device synchronize.

Under expandable segments each stream owns its own segment, and releasing a page waits only on that
segment's stream. FSDP2 allocates gathered weights and reduced gradients on side streams that the
default stream reads, so an empty_cache that follows `zero_grad()` / `reshard()` with no host sync
(`logging_steps > 1`) could unmap memory a queued optimizer / decode kernel still reads. The race
itself needs a GPU; this checks the call order at both sites on CPU with the CUDA calls recorded."""

import ast
import inspect
import types

import pytest
import torch

import unsloth.models.fsdp2_native as F


class _Unit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def reshard(self):
        self.calls.append("reshard")

    def unshard(self):
        self.calls.append("unshard")


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _Unit()
        self.b = _Unit()


@pytest.fixture
def cuda_log(monkeypatch):
    log = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: log.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: log.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *a, **k: (log.append("mem_get_info"), (1 << 50, 1 << 50))[1],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(F, "_fsdp_module_class", lambda: _Unit)
    monkeypatch.setattr(F, "is_native_fsdp2", lambda model: True)
    return log


def _assert_synced(log):
    assert "empty_cache" in log, f"site not reached: {log}"
    for i, call in enumerate(log):
        if call == "empty_cache":
            assert (
                i > 0 and log[i - 1] == "synchronize"
            ), f"empty_cache without a synchronize first: {log}"


def test_generate_gather_probe_synchronizes_before_empty_cache(cuda_log, monkeypatch):
    import torch.distributed as dist

    net = _Net()
    group = types.SimpleNamespace(
        mesh_info = types.SimpleNamespace(mesh = None), device = torch.device("cuda", 0)
    )
    monkeypatch.setattr(F, "_generate_gather_mode", lambda: "auto")
    monkeypatch.setattr(F, "_stream_block_lists", lambda model, units: [(None, {0: units[0]})])
    monkeypatch.setattr(F, "_unit_param_groups", lambda unit: [group])
    monkeypatch.setattr(F, "_unit_gathered_bytes", lambda unit: 1 << 20)
    monkeypatch.setattr(F, "_agree_over_ranks", lambda dist_, device, values, mesh: values)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    assert F._generate_streams_blocks(net) is False  # everything fits: whole gather
    _assert_synced(cuda_log)
    assert cuda_log.index("mem_get_info") > cuda_log.index("empty_cache")


def test_unsharded_exit_synchronizes_before_empty_cache(cuda_log, monkeypatch):
    net = _Net()

    class _Layers:
        pass

    layers = _Layers()
    monkeypatch.setattr(F, "_native_root", lambda model: model)
    monkeypatch.setattr(F, "_rebind_swapped_modules", lambda root, units: None)
    monkeypatch.setattr(F, "_lazy_init_root", lambda root, units: None)
    monkeypatch.setattr(F, "_stream_block_lists", lambda model, units: [(layers, {0: net.b})])
    monkeypatch.setattr(F, "_GenerateBlockStream", lambda blocks: object())
    monkeypatch.setattr(F, "_reshard_after_forward_on", lambda unit: None)
    monkeypatch.setattr(F, "_reshard_after_forward_off", lambda unit: None)
    monkeypatch.setattr(F, "_reshard_after_forward_restore", lambda unit, saved: None)
    with F.native_fsdp2_unsharded(net, stream_blocks = True):
        assert cuda_log == []
    _assert_synced(cuda_log)
    assert net.a.calls[-1] == "reshard" and net.b.calls[-1] == "reshard"


def test_no_bare_empty_cache_in_fsdp2_native():
    """Any new empty_cache in this module must go through the synchronized helper."""
    tree = ast.parse(inspect.getsource(F))
    bare = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "_synchronized_empty_cache":
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "empty_cache"
            ):
                bare.append(f"{fn.name}:{node.lineno}")
    assert not bare, f"empty_cache without the synchronized helper: {bare}"
