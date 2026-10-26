# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seventeenth-round cells: a server Studio retired on purpose is not an MTP crash.

After a first Metal out-of-memory strike or a lost Vulkan device, the next request finds the
port closed. The MTP crash fallback must leave that to _respawn_if_dead, which replays the
load as it was, both at its entry and in a reload queued before the retirement; and the
retirement marker is published before the kill a racing request can observe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_metal_poisoned_respawn as _respawn  # noqa: E402

harness = _respawn.harness  # the respawn fixture, reused as-is

_MTP = _respawn._Intent(speculative_type = "mtp")


def _mtp_server(harness):
    backend, first, loads = harness(_MTP, "draft-mtp")
    backend._mtp_runtime_fallback_active = True
    return backend, first, loads


def _retire(backend, first, metal):
    if metal:
        backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    else:
        backend._retire_device_lost_server(first)


@pytest.mark.parametrize("metal", [True, False], ids = ["metal-strike-1", "vulkan-device-lost"])
def test_a_retired_mtp_server_is_respawned_not_reloaded_without_mtp(harness, metal):
    backend, first, loads = _mtp_server(harness)
    _retire(backend, first, metal)
    assert first.poll() is not None
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is False
    assert backend._respawn_if_dead() is True
    assert [load.speculative_type for load in loads] == ["mtp"]


def test_a_real_mtp_crash_still_reloads_without_mtp(harness):
    """Control: a child that died on its own is the crash the fallback is for."""
    backend, first, loads = _mtp_server(harness)
    first.kill()
    first.wait()
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is True
    assert _respawn._wait(lambda: bool(loads))
    assert loads[-1].speculative_type == "off"


def test_a_reload_queued_before_the_retirement_stands_down(harness):
    """The request saw the port close while the child was still exiting, so the reload was
    queued; the retirement lands during its exit wait."""
    backend, first, loads = _mtp_server(harness)
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is True
    backend._retire_device_lost_server(first)
    assert _respawn._wait(lambda: not backend._mtp_runtime_fallback_in_progress)
    assert loads == []


def test_the_retirement_is_published_before_the_kill(harness):
    backend, first, _loads = _mtp_server(harness)
    seen = []
    kill = first.kill

    def recording_kill():
        seen.append(getattr(backend, "_retired_process", None) is first)
        kill()

    first.kill = recording_kill
    backend._retire_device_lost_server(first)
    assert seen == [True]
