# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama-server whose Metal backend is poisoned is retired instead of failing forever.

After a command buffer fails (kIOGPUCommandBufferCallbackErrorOutOfMemory) ggml-metal
rejects every later graph until the backend is recreated, so the process stays up and
answers each request with "Compute error.". Seen with DeepSeek-V4.1 UD-IQ1_S plus the
DSpark sidecar on a Mac: llama.cpp's --fit cannot size that drafter, so its memory was
never reserved. Recovery is bounded: one reload without the drafter for Auto, then stop.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import LlamaCppBackend

# The reporter's lines, in the order llama-server logged them.
METAL_OOM = "0.30.864.334 E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
METAL_LATCHED = (
    "0.30.882.907 E ggml_metal_graph_compute: backend is in error state from a previous "
    "command buffer failure - recreate the backend to recover"
)
DRAFT_FAILED = "0.22.891.174 E process: llama_process(ctx_dft) failed rc=-3 (n_tokens=14, offset=0)"


@dataclass(frozen = True)
class _Intent:
    speculative_type: Optional[str] = None
    extra_args: tuple = ()
    gpu_ids: Optional[tuple] = None
    verified_gguf: Optional[str] = None


def _child(lines = ()):
    code = "import sys, time\n"
    for line in lines:
        code += f"print({line!r}, flush = True)\n"
    code += "time.sleep(120)\n"
    return subprocess.Popen([sys.executable, "-c", code], stdout = subprocess.PIPE, text = True)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_DEVICE_LOST_LOG_GRACE_S", 0.0)
    children = []

    def make(intent: _Intent, speculative_type: str = "draft-dspark", lines = ()):
        backend = LlamaCppBackend()
        first = _child(lines)
        children.append(first)
        backend._process = first
        backend._healthy = True
        backend._model_identifier = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
        backend._last_load_intent = intent
        backend._speculative_type = speculative_type
        loads = []

        def fake_load_model(new_intent):
            loads.append(new_intent)
            replacement = _child()
            children.append(replacement)
            backend._process = replacement
            backend._healthy = True
            backend._last_load_intent = new_intent
            backend._speculative_type = (
                "none" if new_intent.speculative_type == "off" else speculative_type
            )
            return True

        backend.load_model = fake_load_model
        backend._server_socket_is_open = lambda: False
        return backend, first, loads

    yield make
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()


def _wait(predicate, timeout = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize(
    "lines, retired",
    [
        ([METAL_OOM], True),
        ([METAL_LATCHED], True),
        (["0.01.000.000 I slot launch_slot_: id  0 | prompt: what is kIOGPUCommandBufferCallbackErrorOutOfMemory?"], False),
        (["what does backend is in error state from a previous command buffer failure mean?"], False),
    ],
    ids = ["oom_line", "latched_line", "info_line", "unprefixed_request_dump"],
)
def test_a_poisoned_metal_backend_in_the_log_retires_the_server(harness, lines, retired):
    backend, first, _loads = harness(_Intent(speculative_type = "off"), "none", lines)
    threading.Thread(target = backend._drain_stdout, daemon = True).start()
    try:
        first.wait(timeout = 5)
    except subprocess.TimeoutExpired:
        pass
    assert (first.poll() is not None) is retired


def test_auto_dspark_draft_oom_reloads_once_without_the_drafter_then_stops(harness):
    backend, first, loads = harness(_Intent(), "draft-dspark", [METAL_OOM, METAL_LATCHED, DRAFT_FAILED])
    threading.Thread(target = backend._drain_stdout, daemon = True).start()

    assert _wait(lambda: len(loads) == 1)
    assert first.poll() is not None
    assert loads[0].speculative_type == "off"
    assert _wait(lambda: not backend._mtp_runtime_fallback_in_progress)

    # The drafter-free server runs out of Metal memory too: the second strike is terminal.
    second = backend._process
    backend._retire_device_lost_server(second, metal_poison = {"draft": False})
    assert second.poll() is not None
    assert backend._respawn_if_dead() is False
    assert len(loads) == 1


def test_explicit_dspark_is_not_silently_dropped(harness):
    intent = _Intent(speculative_type = "dspark")
    backend, first, loads = harness(intent, "draft-dspark")

    backend._retire_device_lost_server(first, metal_poison = {"draft": True})
    time.sleep(0.3)

    # No drafter-free reload; the next request replays the user's own settings once.
    assert loads == [] and first.poll() is not None
    assert backend._respawn_if_dead() is True
    assert loads == [intent]


def test_a_launch_without_a_sidecar_respawns_once_then_stops(harness):
    intent = _Intent(speculative_type = "off")
    backend, first, loads = harness(intent, "none")

    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    assert backend._respawn_if_dead() is True and loads == [intent]

    backend._retire_device_lost_server(backend._process, metal_poison = {"draft": False})
    assert backend._respawn_if_dead() is False and loads == [intent]


def test_unload_clears_the_strikes(harness):
    intent = _Intent(speculative_type = "off")
    backend, first, _loads = harness(intent, "none")
    backend._metal_oom_strikes = (intent, 2)
    backend._metal_oom_terminal_key = intent

    backend.unload_model()

    assert backend._metal_oom_strikes is None and backend._metal_oom_terminal_key is None
