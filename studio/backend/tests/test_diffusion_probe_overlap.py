# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cold image load's quant scheme probe (a spawned child, ~4.5 s on an L4) starts before the planner's diffusers
import instead of after it, and hands the load the same table the serial probe would have produced."""

from __future__ import annotations

import threading
import time
import types

import pytest

pytest.importorskip("torch")

_TABLE = {"int8": True, "fp8": False}


def _fam(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


@pytest.fixture
def probe_env(monkeypatch):
    import torch

    from core.inference import diffusion as D
    from core.inference import diffusion_transformer_quant as tq

    ev = {
        "child": [],
        "import": [],
        "snapshot": [],
        "table": dict(_TABLE),
        "keeps": None,
        "persisted": None,
    }

    def child(card):
        ev["child"].append(("start", time.monotonic(), threading.current_thread().name, card))
        time.sleep(0.3)
        ev["child"].append(("end", time.monotonic()))
        return dict(ev["table"])

    def keeps_bf16(*_a, **_k):
        ev["import"].append(("start", time.monotonic()))
        time.sleep(0.3)  # stands in for the diffusers import this call pays on a cold process
        ev["import"].append(("end", time.monotonic()))
        return ev["keeps"]

    def snapshot(_target):
        ev["snapshot"].append(any(e[0] == "end" for e in ev["child"]) or not ev["child"])
        raise RuntimeError("stop after the snapshot")  # swallowed by the planner

    def planned(_fam, target, _mode, **_k):
        ok = tq._scheme_supported("int8", str(target.device))
        ev["verdict"] = ok
        return None

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_OVERLAP", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tq, "_smoke_cache_device_key", lambda device, ordinal = None: "cuda:0")
    monkeypatch.setattr(tq, "_persisted_probe_table", lambda card: ev["persisted"])
    monkeypatch.setattr(tq, "_child_probe_table", child)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a, **k: pytest.fail("in-process probe reached"))
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {})
    if hasattr(tq, "_PREWARMED_TABLES"):
        monkeypatch.setattr(tq, "_PREWARMED_TABLES", {})
    monkeypatch.setattr(D, "dense_transformer_supported", lambda *a, **k: True)
    monkeypatch.setattr(D, "_pipeline_quant_uncompilable_reason", lambda *a, **k: None)
    monkeypatch.setattr(D, "_auto_keeps_bf16_reason", keeps_bf16)
    monkeypatch.setattr(D, "snapshot_device_memory", snapshot)
    monkeypatch.setattr(D, "_planned_quant_scheme", planned)
    backend = D.DiffusionBackend.__new__(D.DiffusionBackend)
    backend._target_for_ordinal = lambda *a, **k: types.SimpleNamespace(device = "cuda", ordinal = 0)
    fam = _fam("flux.2-klein")

    def plan():
        return backend._pipeline_planned_denoiser_scheme(
            fam,
            base = fam.base_repo,
            kind = "pipeline",
            transformer_quant = None,
            speed_mode = None,
            repo_id = fam.base_repo,
        )

    ev["plan"] = plan
    ev["tq"] = tq
    return ev


def test_the_probe_child_runs_during_the_planner_import(probe_env):
    ev = probe_env
    assert ev["plan"]() is None
    starts = [e for e in ev["child"] if e[0] == "start"]
    assert len(starts) == 1, "exactly one probe child per cold load"
    import_end = next(e[1] for e in ev["import"] if e[0] == "end")
    # Base starts the child only once the import has returned (inside _planned_quant_scheme).
    assert starts[0][1] < import_end, "the probe child must overlap the planner's import"
    assert ev["verdict"] is True
    tq = ev["tq"]
    assert tq._SMOKE_CACHE == {("int8", "cuda:0"): True, ("fp8", "cuda:0"): False}
    assert tq._PREWARMED_TABLES == {}


def test_the_kill_switch_keeps_the_serial_probe(probe_env, monkeypatch):
    ev = probe_env
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_OVERLAP", "0")
    ev["plan"]()
    starts = [e for e in ev["child"] if e[0] == "start"]
    assert len(starts) == 1
    assert starts[0][1] >= next(e[1] for e in ev["import"] if e[0] == "end")
    assert ev["verdict"] is True


def test_a_persisted_table_starts_no_child(probe_env):
    ev = probe_env
    ev["persisted"] = dict(_TABLE)
    ev["plan"]()
    assert ev["child"] == []
    assert ev["verdict"] is True


def test_an_out_of_memory_verdict_is_honoured_without_a_second_child(probe_env):
    # None (the child's OOM) is not persisted, so the load must take the early table, not spawn again.
    ev = probe_env
    ev["table"] = {"int8": None, "fp8": False}
    ev["plan"]()
    assert len([e for e in ev["child"] if e[0] == "start"]) == 1
    assert ev["verdict"] is False  # unproven_ok defaults to False, as with the serial probe
    assert ("int8", "cuda:0") not in ev["tq"]._SMOKE_CACHE


def test_free_vram_is_read_only_after_the_child_exits(probe_env):
    ev = probe_env
    ev["keeps"] = "small DiT keeps bf16"
    ev["plan"]()
    assert ev["snapshot"] == [True], "snapshot_device_memory ran while the probe child was alive"
    assert ev["tq"]._PREWARMED_TABLES == {}


def test_a_held_probe_lock_skips_the_early_child_and_join_returns(probe_env):
    # Another card's probe holds the lock: the early thread must not queue behind it (its join would then outlast
    # one child's deadline), so it gives up at once and the join does not wait on the lock holder.
    tq = probe_env["tq"]
    assert tq._CHILD_PROBE_LOCK.acquire(timeout = 1.0)
    try:
        started = time.monotonic()
        handle = tq.start_probe_table_prewarm("cuda")
        assert handle is not None
        tq.join_probe_table_prewarm(handle)
        assert time.monotonic() - started < 5.0
        assert not handle[0].is_alive()
    finally:
        tq._CHILD_PROBE_LOCK.release()
    assert probe_env["child"] == []
