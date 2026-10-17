# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/train/status reports eta and elapsed time, and a stopped run never reads "Training completed"."""

from __future__ import annotations

import asyncio

from .test_training_status_terminal import _DONE, _run_route_helpers_inline, _running, rt

__all__ = ["_run_route_helpers_inline"]


def _status():
    return asyncio.run(rt.get_training_status(current_subject = "t"))


def test_status_details_carry_elapsed_and_eta(monkeypatch):
    b = _running(monkeypatch)
    b._progress.elapsed_seconds = 30.0
    b._progress.eta_seconds = 90.5
    details = _status().details
    assert (details["elapsed_seconds"], details["eta_seconds"]) == (30.0, 90.5)


def test_status_details_eta_is_none_before_the_first_log(monkeypatch):
    _running(monkeypatch)
    details = _status().details
    assert (details["elapsed_seconds"], details["eta_seconds"]) == (None, None)


def test_a_cancel_during_the_final_save_reads_cancelled(monkeypatch):
    # The worker checked the stop before saving, so it still sends "Training completed!".
    b = _running(monkeypatch)
    b._should_stop = b._cancel_requested = True
    b._handle_event(dict(_DONE))
    st = _status()
    assert st.phase == "stopped"
    assert st.message == "Training cancelled."
    assert b._progress.is_completed is False
    assert b._terminal_finalize_payload["status"] == "stopped"


def test_a_stop_during_the_drafter_phase_reads_stopped_and_keeps_the_save(monkeypatch):
    b = _running(monkeypatch)
    b._should_stop = True
    b._handle_event(dict(_DONE))
    st = _status()
    assert st.phase == "stopped"
    assert st.message == "Training stopped. Model saved to /tmp/out"
    assert st.details["output_dir"] == "/tmp/out"


def test_a_cancelled_message_with_a_period_counts_as_stopped(monkeypatch):
    b = _running(monkeypatch)
    b._handle_event({**_DONE, "status_message": "Training cancelled."})
    assert b._progress.is_completed is False
    assert b._terminal_finalize_payload["status"] == "stopped"


def test_a_run_with_no_stop_stays_completed(monkeypatch):
    b = _running(monkeypatch)
    b._handle_event(dict(_DONE))
    st = _status()
    assert st.phase == "completed"
    assert st.message.startswith("Training completed!")
    assert b._progress.is_completed is True
