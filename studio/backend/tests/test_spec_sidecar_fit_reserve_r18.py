# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eighteenth-round cells: a retired child takes its engine-stats poller with it.

_kill_process stops the poller; a retirement (Metal out of memory, a lost device) killed the
child without it, so after a terminal Metal strike it polled the dead child's loopback port
until the next load, which may never come. The replacement's load starts its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_metal_poisoned_respawn as _respawn  # noqa: E402
import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402

harness = _respawn.harness  # the respawn fixture, reused as-is


class _Poller:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


@pytest.mark.parametrize("metal", [True, False], ids = ["metal", "device-lost"])
def test_a_retired_child_stops_its_stats_poller(harness, metal):
    backend, first, _loads = harness(_respawn._Intent(speculative_type = "off"), "none")
    poller = backend._stats_logger = _Poller()
    if metal:
        backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    else:
        backend._retire_device_lost_server(first)
    assert first.poll() is not None
    assert poller.stopped and backend._stats_logger is None


def test_a_terminal_metal_retirement_leaves_no_poller(harness):
    backend, first, _loads = harness(_respawn._Intent(speculative_type = "off"), "none")
    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    assert backend._respawn_if_dead() is True
    poller = backend._stats_logger = _Poller()
    backend._retire_device_lost_server(backend._process, metal_poison = {"draft": False})
    # Nothing respawns it, so it is unloaded like any other model (r41).
    assert backend._process is None and backend._model_identifier is None
    assert poller.stopped and backend._stats_logger is None


def test_the_replacement_load_starts_its_own_poller(tmp_path, monkeypatch):
    """The respawn replays load_model, which starts a poller for the new child."""
    import core.inference.llama_stats as _stats

    started = []
    monkeypatch.setattr(
        _stats,
        "maybe_start_stats_logger",
        lambda base_url, logger, headers = None: started.append(base_url) or _Poller(),
    )
    backend, gguf = _r2._cuda_backend(tmp_path)
    caps = dict(_r2._CAPS, supports_metrics = True)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    _placement._launch(backend, gguf, n_ctx = 4096)
    first = backend._stats_logger
    # What a retirement leaves: the poller stopped and cleared.
    first.stop()
    backend._stats_logger = None
    _placement._launch(backend, gguf, n_ctx = 4096, force_reload = True)
    assert len(started) == 2 and isinstance(backend._stats_logger, _Poller)
    assert not backend._stats_logger.stopped
