# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted probe verdicts are keyed by the installed GPU driver; an unreadable driver persists nothing. CPU-only fakes."""

from __future__ import annotations

import builtins
import ctypes
import io
import types

import pytest
import torch

from core.inference import diffusion_probe_cache as pc

_REAL_OPEN = builtins.open
_PROC = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {}  Release Build  (builder)  Mon Dec  8 2025\n"


@pytest.fixture
def card(monkeypatch, tmp_path):
    """A visible CUDA card with a UUID, torch without ``_cuda_getDriverVersion``, and a controllable driver."""
    props = types.SimpleNamespace(name = "Fake GPU", uuid = "GPU-1234", major = 8, minor = 9)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_CACHE", raising = False)
    monkeypatch.setattr(pc, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    state = {"build": "580.95.05"}

    def fake_open(path, *a, **kw):
        if str(path) == "/proc/driver/nvidia/version":
            if state["build"] is None:
                raise FileNotFoundError(path)
            return io.StringIO(_PROC.format(state["build"]))
        if str(path) == "/proc/self/maps":
            return io.StringIO("")
        return _REAL_OPEN(path, *a, **kw)

    def fake_cdll(name, *a, **kw):
        raise OSError(f"no {name}")

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    def new_process():
        monkeypatch.setattr(pc, "_DRIVER", [], raising = False)

    new_process()
    state["new_process"] = new_process
    return state


def test_a_driver_update_changes_the_fingerprint(card):
    before = pc.fingerprint("cuda:0")
    card["build"] = "580.105.08"
    card["new_process"]()
    after = pc.fingerprint("cuda:0")
    assert before is not None and after is not None
    assert pc._key(before) != pc._key(after)


def test_a_verdict_stored_under_one_driver_is_not_adopted_under_another(card):
    ns = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert ns.put("cuda:0", "verdict", {"ok": True})
    assert pc.store("cuda:0", {"int8": True})
    card["build"] = "580.105.08"
    card["new_process"]()
    fresh = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert fresh.get("cuda:0", "verdict") is None
    assert pc.load("cuda:0") is None


def test_an_unreadable_driver_persists_and_admits_nothing(card):
    card["build"] = None
    assert pc.fingerprint("cuda:0") is None
    ns = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert ns.put("cuda:0", "verdict", {"ok": True}) is False
    assert ns.get("cuda:0", "verdict") is None
    assert pc.store("cuda:0", {"int8": True}) is False


def test_the_same_driver_still_reuses_the_verdict(card):
    ns = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert ns.put("cuda:0", "verdict", {"ok": True})
    card["new_process"]()
    fresh = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert fresh.get("cuda:0", "verdict") == {"ok": True}


def test_rocm_keys_on_the_hip_driver_and_refuses_an_unread_one(card, monkeypatch):
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(torch.version, "hip", "7.1.0")
    monkeypatch.setattr(A, "_ROCM_DRIVER_VERSION", ["driver:hipdriver70100,amdgpu6.14"])
    first = pc.fingerprint("cuda:0")
    assert first is not None and first["driver"] == "driver:hipdriver70100,amdgpu6.14"
    card["new_process"]()
    monkeypatch.setattr(A, "_ROCM_DRIVER_VERSION", ["driver:?"])
    assert pc.fingerprint("cuda:0") is None
