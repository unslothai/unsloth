# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A first token report stamps the holder; only a reading started after it folds the holder in.

Promoted on the token itself, a `/slots` probe already in flight landed afterwards and the
holder's prompt was folded into a reading from before its prefill: the cache was undercounted by
that prompt, and a later admission could overfill it.
"""

from __future__ import annotations

from core.inference.llama_preemption import PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65074")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


def test_a_probe_in_flight_does_not_fold_the_new_prompt_in():
    controller = _controller()
    controller.register("old", lease = _Lease(), tokens = 3000)
    controller.note_measured("old")
    controller.note_resident(3000, started_at_seq = controller.residency_epoch())
    controller.register("new", lease = _Lease(), tokens = 5000, prompt_tokens = 5000)
    in_flight = controller.residency_epoch()
    controller.observe("new", 1)
    controller.note_resident(3000 + 4000, started_at_seq = in_flight)
    assert controller.participant("new").measured is False
    assert (
        controller.snapshot().committed >= 7000 + 5001
    ), "the new prompt was folded into a reading taken before its prefill"


def test_a_probe_started_after_the_token_folds_it_in():
    controller = _controller()
    controller.register("new", lease = _Lease(), tokens = 5000, prompt_tokens = 5000)
    controller.observe("new", 1)
    controller.note_resident(5001, started_at_seq = controller.residency_epoch())
    assert controller.participant("new").measured is True
    assert controller.snapshot().committed == 5001


def test_later_tokens_do_not_restamp_it():
    controller = _controller()
    controller.register("new", lease = _Lease(), tokens = 5000, prompt_tokens = 5000)
    controller.observe("new", 1)
    stamped = controller.participant("new").measured_at_seq
    controller.observe("new", 40)
    assert controller.participant("new").measured_at_seq == stamped
