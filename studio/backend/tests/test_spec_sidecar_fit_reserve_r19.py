# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nineteenth-round cells: a terminal Metal retirement is a full unload.

After the second Metal out-of-memory strike nothing respawns the child, so the backend is
unloaded like any other model (attempt log, pid record, stats, codec, temp files, plan), and
the routes drop its kept slot once no request holds it and release the CHAT claim, as a
crashed worker's reap does. A first strike still respawns with the same settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_metal_poisoned_respawn as _respawn  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference import model_slots  # noqa: E402

harness = _respawn.harness  # the respawn fixture, reused as-is

_OFF = _respawn._Intent(speculative_type = "off")


class _Fh:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def write(self, *_a):
        return 0

    def flush(self):
        pass


def _two_strikes(backend, first, monkeypatch, calls):
    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    assert backend._respawn_if_dead() is True
    fh = backend._llama_log_fh = _Fh()
    monkeypatch.setattr(backend, "_clear_server_pid", lambda: calls.append("pid"))
    monkeypatch.setattr(backend, "_unload_audio_codec", lambda: calls.append("codec"))
    backend._planned_vram_mib = {0: 9000}
    backend._retire_device_lost_server(backend._process, metal_poison = {"draft": False})
    return fh


def test_the_terminal_strike_unloads_the_backend(harness, monkeypatch):
    backend, first, _loads = harness(_OFF, "none")
    calls = []
    fh = _two_strikes(backend, first, monkeypatch, calls)
    assert backend._process is None and backend._model_identifier is None
    assert backend.is_active is False and backend.is_loaded is False
    assert fh.closed and "pid" in calls and "codec" in calls
    assert not backend._planned_vram_mib
    assert backend._released_after_oom is True
    assert backend._respawn_if_dead() is False


def test_a_first_strike_still_respawns_with_the_same_settings(harness, monkeypatch):
    backend, first, loads = harness(_OFF, "none")
    seen = []
    monkeypatch.setattr(_lc, "_TERMINAL_RETIREMENT_HOOKS", [seen.append], raising = False)
    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    assert backend._model_identifier is not None
    assert getattr(backend, "_released_after_oom", False) is False
    assert backend._respawn_if_dead() is True
    assert loads == [_OFF] and seen == []


def test_the_terminal_strike_runs_the_cleanup_hooks_once(harness, monkeypatch):
    backend, first, _loads = harness(_OFF, "none")
    seen = []
    monkeypatch.setattr(_lc, "_TERMINAL_RETIREMENT_HOOKS", [seen.append], raising = False)
    _two_strikes(backend, first, monkeypatch, [])
    assert seen == [backend]


class _Orchestrator:
    active_model_name = None
    loading_models = ()

    def _cleanup(self):
        pass


def _released_slot(refs):
    llama = _lc.LlamaCppBackend()
    llama._released_after_oom = True
    return model_slots.ExtraSlot(llama = llama, orchestrator = _Orchestrator(), account = "a", refs = refs)


def test_the_routes_release_the_slot_and_the_chat_claim(monkeypatch):
    import routes.inference as inf

    assert inf._release_after_terminal_retirement in _lc._TERMINAL_RETIREMENT_HOOKS
    held, idle = _released_slot(refs = 1), _released_slot(refs = 0)
    monkeypatch.setattr(model_slots, "slots", [held, idle])
    cleared = []
    monkeypatch.setattr(inf, "release_chat_after_kept_models", lambda: True)
    monkeypatch.setattr(inf.account_access, "clear_resident", cleared.append)
    inf._release_after_terminal_retirement(idle.llama)
    # A request still holds the other: reaped on a later routed request.
    assert model_slots.slots == [held] and cleared == ["chat"]
    held.refs = 0
    assert model_slots.reap_released_after_oom() == [held]
    assert model_slots.slots == []


def test_runtime_teardown_includes_a_released_slot(monkeypatch):
    slot = _released_slot(refs = 0)
    monkeypatch.setattr(model_slots, "slots", [slot])
    monkeypatch.setattr(model_slots, "stuck", [])
    assert model_slots.unload_llama_slots() == 1
    assert model_slots.slots == []
