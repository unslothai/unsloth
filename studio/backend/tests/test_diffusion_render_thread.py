# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextvars
import threading

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_render_thread as rt  # noqa: E402


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(rt, "enabled", lambda: True)
    devices = {"current": 0, "set": []}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: devices["current"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: devices["set"].append(d))
    return devices


def test_every_call_lands_on_one_thread_whatever_the_caller(armed):
    idents = []
    for _ in range(3):
        t = threading.Thread(target = lambda: idents.append(rt.run("t1", threading.get_ident)))
        t.start()
        t.join()
    idents.append(rt.run("t1", threading.get_ident))
    assert len(set(idents)) == 1
    assert idents[0] != threading.get_ident()


def test_caller_state_is_carried(armed):
    var = contextvars.ContextVar("account", default = None)
    var.set("alice")
    armed["current"] = 3
    with torch.inference_mode():
        seen = rt.run("t2", lambda: (var.get(), torch.is_inference_mode_enabled()))
    assert seen == ("alice", True)
    assert armed["set"][-1] == 3
    assert rt.run("t2", torch.is_inference_mode_enabled) is False


def test_exceptions_reach_the_caller(armed):
    class Boom(RuntimeError):
        pass

    def fail():
        raise Boom("x")

    with pytest.raises(Boom):
        rt.run("t3", fail)
    assert rt.run("t3", lambda: 5) == 5


def test_nested_call_runs_inline(armed):
    outer = rt.run("t4", lambda: (threading.get_ident(), rt.run("t4", threading.get_ident)))
    assert outer[0] == outer[1]


def test_disabled_runs_inline(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_RENDER_THREAD", "0")
    assert rt.enabled() is False
    assert rt.run("t5", threading.get_ident) == threading.get_ident()


def test_one_thread_per_card(armed):
    armed["current"] = 0
    a = rt.run("t6", threading.get_ident)
    armed["current"] = 1
    b = rt.run("t6", threading.get_ident)
    armed["current"] = 0
    assert rt.run("t6", threading.get_ident) == a
    assert a != b
