# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend


def _child():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


@pytest.fixture
def loaded():
    children = []
    backend = LlamaCppBackend()
    intent = GgufLoadIntent(model_identifier = "unsloth/test-GGUF")
    loads = []
    reloaded = threading.Event()

    def spawn():
        child = _child()
        children.append(child)
        backend._process = child
        backend._healthy = True
        backend._start_mtp_crash_watchdog()
        return child

    def fake_load_model(load_intent):
        loads.append(load_intent)
        spawn()
        reloaded.set()
        return True

    backend._model_identifier = intent.model_identifier
    backend._last_load_intent = intent
    backend.load_model = fake_load_model
    yield backend, spawn, intent, loads, reloaded
    backend._stop_mtp_crash_watchdog()
    for child in children:
        child.kill()
        child.wait()


def test_idle_server_death_reloads_without_a_request(loaded):
    backend, spawn, intent, loads, reloaded = loaded
    first = spawn()
    first.kill()
    first.wait()
    assert reloaded.wait(timeout = 10)
    assert loads == [intent]
    assert backend._process is not first
    assert backend._process.poll() is None
    assert backend.is_loaded


def test_recovered_server_dying_again_is_not_reloaded_in_a_loop(loaded):
    backend, spawn, intent, loads, reloaded = loaded
    spawn().kill()
    assert reloaded.wait(timeout = 10)
    reloaded.clear()
    backend._process.kill()
    assert not reloaded.wait(timeout = 3)
    assert loads == [intent]


def test_unload_does_not_bring_the_server_back(loaded):
    backend, spawn, intent, loads, reloaded = loaded
    spawn()
    backend.unload_model()
    assert not reloaded.wait(timeout = 3)
    assert loads == []
