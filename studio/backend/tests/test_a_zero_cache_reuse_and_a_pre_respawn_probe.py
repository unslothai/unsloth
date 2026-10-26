# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two small parses. ``--cache-reuse +0``, ``00`` and ``-0`` are zero to llama-server and to the
env check, so they do not contradict exact concurrency. And a ``/slots`` probe sent before a
respawn is refused even when no sample has been recorded yet: the old server's reading would
otherwise become the new one's count and inflate the commitment."""

import pytest

from core.inference import llama_exact
from core.inference.llama_preemption import PreemptionController


@pytest.mark.parametrize("value", ["0", "+0", "00", "-0"])
def test_a_zero_cache_reuse_is_not_a_contradiction(value):
    assert llama_exact.contradicting_args(["--cache-reuse", value]) == []
    assert llama_exact.contradicting_args([f"--cache-reuse={value}"]) == []


@pytest.mark.parametrize("value", ["1", "256", "abc"])
def test_a_nonzero_cache_reuse_still_is(value):
    assert llama_exact.contradicting_args(["--cache-reuse", value]) == ["--cache-reuse"]


def test_a_probe_from_before_a_respawn_is_refused_before_any_sample():
    controller = PreemptionController("http://127.0.0.1:65093")
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    stale = controller.residency_epoch()  # sent to the old server, still in flight
    controller.forget_residency()  # the respawn
    assert controller.note_resident(8000, started_at_seq = stale) is None
    assert controller._resident is None, "the old server's reading became the new one's count"
    fresh = controller.residency_epoch()
    assert controller.note_resident(3000, started_at_seq = fresh) is not None
    assert controller._resident == 3000
