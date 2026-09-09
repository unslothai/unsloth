# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``UNSLOTH_SETTLE_DELAY_S`` shortens the settle wait for tests, and only for tests.

``settled_snapshot_device_memory`` spaces its retried ``mem_get_info`` reads a second apart
so a transient tenant on a live card has time to clear before the next read. Under test the
snapshots are stubs whose answers do not change with time, so the wait buys nothing --
``test_diffusion_backend.py`` spent 142s of a 328s suite sitting in it, most of that in
tests parked at exactly 4.00s. The tests that call the function directly already pass
``delay_s = 0``; the expensive ones reach it through ``_plan_memory``, which has no way to
forward the argument. Hence an env override, defaulted to 0 in the backend conftest.

Two things have to stay true and neither is loud when it stops being true:

  * The PRODUCTION default is still a full second. A change that quietly made the fast path
    the default would turn a transient undercount into a silent fallback to offloaded GGUF
    on a card that could have gone resident, and nothing would fail.
  * The override changes only the WAIT, never the retry count or the ``max`` over the reads.
    That is what makes zeroing it safe: a test asserting "retries once on a transient
    undercount" still exercises the retry.
"""

import time

import pytest

from core.inference import diffusion_memory as dm


def test_the_production_default_is_still_a_full_second(monkeypatch):
    """No env var set means the caller's delay is returned untouched.

    The conftest pins the override for the suite, so this has to unset it to see what a
    production process sees.
    """
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
    """The retry still runs the same number of times; only the spacing collapses.

    This is the assertion that makes the speed-up safe to take. If the override were ever
    implemented by skipping the loop instead of shortening the sleep, every test that
    exercises "a transient undercount is retried past" would still pass -- because the
    first read already carries the stubbed answer -- and the real behaviour would be gone.
    """
    reads, slept = [], []

    def snapshot(target):
        reads.append(1)
        return dm.DeviceMemory("cuda", "cuda:0", "vram", 1024, 100_000)

    monkeypatch.setattr(dm, "snapshot_device_memory", snapshot)
    # Record the requested delays rather than timing the call. The loop's first act on cuda
    # is a real torch.cuda.synchronize() + empty_cache(), which costs ~0.6s on a live card
    # and has nothing to do with the spacing under test; asserting on wall-clock here would
    # be a bound on the driver, not on this change.
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
