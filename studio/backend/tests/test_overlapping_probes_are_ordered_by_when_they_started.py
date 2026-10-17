# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two `/slots` probes that overlap a measurement mark are ordered by when they started."""

from __future__ import annotations

from core.inference.llama_preemption import PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65076")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


def test_the_newer_probe_is_kept_after_the_older_one_returns_first():
    controller = _controller()
    controller.register("chat", lease = _Lease(), tokens = 4000)
    older = controller.residency_epoch()
    controller.note_measured("chat")  # the mark between the two probes
    newer = controller.residency_epoch()
    assert newer > older
    assert controller.note_resident(3000, started_at_seq = older) is not None
    assert (
        controller.note_resident(9000, started_at_seq = newer) is not None
    ), "the probe that started after the mark was dropped as stale"
    assert controller._resident == 9000


def test_an_older_probe_returning_last_is_still_dropped():
    controller = _controller()
    older = controller.residency_epoch()
    controller.note_measured("missing")  # nothing to mark, the epoch is unchanged
    controller.register("chat", lease = _Lease(), tokens = 4000)
    controller.note_measured("chat")
    newer = controller.residency_epoch()
    assert controller.note_resident(9000, started_at_seq = newer) is not None
    assert controller.note_resident(3000, started_at_seq = older) is None
    assert controller._resident == 9000


def test_a_reading_with_no_start_outranks_every_probe_already_out():
    controller = _controller()
    out = controller.residency_epoch()
    controller.note_resident(7000)
    assert controller.note_resident(1000, started_at_seq = out) is None
