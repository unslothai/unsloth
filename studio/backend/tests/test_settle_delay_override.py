# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_SETTLE_DELAY_S shortens only the wait, never the retry count; production default is 1s."""

import time

import pytest

from core.inference import diffusion_memory as dm


def test_the_production_default_is_still_a_full_second(monkeypatch):
    """Unset env var returns the caller's delay untouched, so production keeps its one-second default."""
    monkeypatch.delenv("UNSLOTH_SETTLE_DELAY_S", raising = False)
    assert dm._settle_delay(1.0) == 1.0
    assert dm._settle_delay(0.25) == 0.25


def test_the_override_replaces_the_callers_delay(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "0")
    assert dm._settle_delay(1.0) == 0.0
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "0.05")
    assert dm._settle_delay(1.0) == pytest.approx(0.05)


@pytest.mark.parametrize("bad", ["", "fast", "1,0", "None"])
def test_an_unparseable_override_leaves_production_behaviour_alone(monkeypatch, bad):
    """A typo in the env must not be read as "do not wait"."""
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", bad)
    assert dm._settle_delay(1.0) == 1.0


def test_a_negative_override_is_clamped_rather_than_passed_to_sleep(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "-5")
    assert dm._settle_delay(1.0) == 0.0


def test_the_override_shortens_the_wait_without_dropping_a_read(monkeypatch):
    """The override must shorten the sleep and keep every read; skipping the loop would hide the retry."""
    reads, slept = [], []

    def snapshot(target):
        reads.append(1)
        return dm.DeviceMemory("cuda", "cuda:0", "vram", 1024, 100_000)

    monkeypatch.setattr(dm, "snapshot_device_memory", snapshot)
    # Record delays instead of timing: the first cuda sync + empty_cache costs ~0.6s on a live card.
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "0")

    target = type("T", (), {"device": "cuda", "backend": "cuda"})()
    dm.settled_snapshot_device_memory(target, attempts = 4, delay_s = 1.0)

    assert (
        len(reads) == 4
    ), f"the override changed the number of reads, not just their spacing: {len(reads)}"
    assert slept == [
        0.0,
        0.0,
        0.0,
    ], f"the override did not reach time.sleep; the loop asked for {slept}"


def test_the_backend_conftest_pins_the_override_for_the_whole_suite():
    """Set by conftest at import, so it holds for subprocess-spawning tests too."""
    import os
    assert os.environ.get("UNSLOTH_SETTLE_DELAY_S") == "0", (
        "the backend conftest no longer pins UNSLOTH_SETTLE_DELAY_S; the diffusion and "
        "video suites go back to paying a real second per retried VRAM read"
    )
