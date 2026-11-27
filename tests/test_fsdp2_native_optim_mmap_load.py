# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 resume reads the full optimizer.pt memory-mapped: every rank loads the whole file and keeps only
its own shards, so an eager torch.load gives each rank an anonymous copy of the full optimizer state. A legacy
(non-zip) file and a file object still load eagerly. fsdp2_native.py is loaded on its own (no GPU)."""

import io
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_native_resume_old_torch import _load  # noqa: E402


class _Stop(Exception):
    pass


def _state():
    g = torch.Generator().manual_seed(0)
    return {
        "state": {
            "w": {
                "step": torch.tensor(3.0),
                "exp_avg": torch.randn(7, 5, generator = g),
                "exp_avg_sq": torch.rand(7, 5, generator = g),
            }
        },
        "param_groups": [{"lr": 1e-3, "params": ["w"]}],
    }


def _same(a, b):
    sa, sb = a["state"]["w"], b["state"]["w"]
    return a["param_groups"] == b["param_groups"] and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_native_load_optimizer_maps_optimizer_pt(tmp_path, monkeypatch):
    m = _load("unsloth.models.fsdp2_native")
    path = tmp_path / "optimizer.pt"
    torch.save(_state(), path)
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(torch, "load", spy)
    with pytest.raises(_Stop):
        m.native_load_optimizer(None, None, str(path))
    assert calls and calls[0].get("mmap") is True
    assert calls[0].get("map_location") == "cpu" and calls[0].get("weights_only") is True


def test_mapped_load_matches_eager(tmp_path):
    m = _load("unsloth.models.fsdp2_native")
    path = tmp_path / "optimizer.pt"
    torch.save(_state(), path)
    assert _same(m._load_full_optimizer_state(str(path)), torch.load(path, weights_only = True))
    assert _same(m._load_full_optimizer_state(path), _state())  # os.PathLike


def test_legacy_serialization_falls_back_to_eager(tmp_path, monkeypatch):
    m = _load("unsloth.models.fsdp2_native")
    path = tmp_path / "optimizer.pt"
    torch.save(_state(), path, _use_new_zipfile_serialization = False)
    calls = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("mmap"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert _same(m._load_full_optimizer_state(str(path)), _state())
    assert calls == [True, None]


def test_file_object_loads_eagerly(monkeypatch):
    m = _load("unsloth.models.fsdp2_native")
    buf = io.BytesIO()
    torch.save(_state(), buf)
    buf.seek(0)
    calls = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("mmap"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert _same(m._load_full_optimizer_state(buf), _state())
    assert calls == [None]


def test_other_load_errors_propagate(tmp_path):
    m = _load("unsloth.models.fsdp2_native")
    path = tmp_path / "optimizer.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception) as e:
        m._load_full_optimizer_state(str(path))
    assert "mmap can only be used" not in str(e.value)
