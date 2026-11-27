# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A diffusion resume bundle's .pt files are never torch.load-ed on torch < 2.6 (CVE-2025-32434)."""

from __future__ import annotations

import pytest
import torch

from core.training import diffusion_checkpoint as dc


def _bundle(tmp_path):
    torch.save({"state": {}, "param_groups": []}, tmp_path / "optimizer.pt")
    return dc.LoadedCheckpoint(path = tmp_path, manifest = {"files": {"optimizer": "optimizer.pt"}})


def _record_loads(monkeypatch):
    calls = []
    real = torch.load

    def load(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("f"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    return calls


@pytest.mark.parametrize("version", ["2.5.1+cu124", "2.4.0"])
def test_old_torch_refuses_the_pickle_before_loading(tmp_path, monkeypatch, version):
    loaded = _bundle(tmp_path)
    calls = _record_loads(monkeypatch)
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(dc.ResumeError, match = "CVE-2025-32434"):
        loaded.torch_state("optimizer")
    assert calls == []


@pytest.mark.parametrize("version", ["2.6.0", "2.11.0+cu130", "2.13.0.dev20261101"])
def test_current_torch_still_resumes(tmp_path, monkeypatch, version):
    loaded = _bundle(tmp_path)
    calls = _record_loads(monkeypatch)
    monkeypatch.setattr(torch, "__version__", version)
    assert loaded.torch_state("optimizer") == {"state": {}, "param_groups": []}
    assert len(calls) == 1
