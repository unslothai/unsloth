# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Deep Research stream parked by KV preemption must not be failed as a silent model.

The plain /v1/chat/completions stream research reads is armed for preemption like any chat, so
a pause arrives as `: preempt-paused`, renewed by `: preempt-keepalive` and ended by
`: preempt-resumed`. The first-output and idle budgets are suspended for the pause, bounded by
the same park cap the routes use, and a stream that never pauses is timed exactly as before.
"""

from __future__ import annotations

import asyncio
import time

from types import SimpleNamespace

import pytest

from core import research_runs

_CONTENT = 'data: {"choices":[{"delta":{"content":"%s"}}]}'
_STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'


class _Stream:
    status_code = 200

    def __init__(self, script):
        self._script = script

    def raise_for_status(self):
        return self

    async def aclose(self):
        return None

    async def aiter_lines(self):
        for item in self._script:
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
            else:
                yield item


def _paused(seconds: float, interval: float = 0.05) -> list:
    script = [": preempt-paused"]
    for _ in range(int(seconds / interval)):
        script += [interval, ": preempt-keepalive"]
    return script


@pytest.fixture
def short_budgets(monkeypatch):
    monkeypatch.setattr(research_runs, "_MODEL_FIRST_OUTPUT_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(research_runs, "_MODEL_OUTPUT_IDLE_TIMEOUT_SECONDS", 0.2)


def _run(monkeypatch, script):
    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def build_request(self, method, url, **kwargs):
            return {"method": method, "url": url, **kwargs}

        async def send(
            self,
            request,
            *,
            stream = False,
        ):
            return _Stream(script)

    async def _check_active(run_id):
        return None

    monkeypatch.setattr(research_runs.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(
        research_runs.auth_storage, "create_api_key", lambda **kwargs: ("token", {"id": 1})
    )
    monkeypatch.setattr(research_runs.auth_storage, "revoke_internal_api_key", lambda key_id: None)
    supervisor = research_runs.ResearchSupervisor(
        SimpleNamespace(state = SimpleNamespace(server_port = 1))
    )
    supervisor._check_active = _check_active
    run = {
        "id": "run-1",
        "ownerSubject": "user-1",
        "config": {"budgets": {"modelTimeoutSeconds": 30.0}},
    }
    return asyncio.run(
        supervisor._stream_completion(run, [{"role": "user"}], report_progress = False)
    )


def test_a_pause_before_the_first_token_does_not_spend_the_first_output_budget(
    monkeypatch, short_budgets
):
    script = [*_paused(1.0), ": preempt-resumed", _CONTENT % "report", _STOP, "data: [DONE]"]
    assert _run(monkeypatch, script) == ("report", "", "stop", None)


def test_a_pause_mid_report_does_not_spend_the_idle_budget(monkeypatch, short_budgets):
    script = [
        _CONTENT % "first ",
        *_paused(1.0),
        ": preempt-resumed",
        ": preempt-recomputed",
        _CONTENT % "second",
        _STOP,
        "data: [DONE]",
    ]
    assert _run(monkeypatch, script) == ("first second", "", "stop", None)


def test_real_output_ends_a_pause_without_a_resume_notice(monkeypatch, short_budgets):
    script = [*_paused(0.5), _CONTENT % "report", 1.0, _STOP, "data: [DONE]"]
    with pytest.raises(research_runs.ModelOutputIdleTimeout):
        _run(monkeypatch, script)


def test_the_budget_runs_again_after_the_resume(monkeypatch, short_budgets):
    script = [*_paused(0.5), ": preempt-resumed", 30.0, "data: [DONE]"]
    started = time.monotonic()
    with pytest.raises(research_runs.ModelFirstOutputTimeout):
        _run(monkeypatch, script)
    assert time.monotonic() - started < 10.0


def test_a_pause_past_the_park_cap_still_times_out(monkeypatch, short_budgets):
    monkeypatch.setattr(research_runs, "_PREEMPT_PARK_STALL_CAP_SECONDS", 1.0)
    started = time.monotonic()
    with pytest.raises(research_runs.ModelFirstOutputTimeout):
        _run(monkeypatch, _paused(60.0))
    elapsed = time.monotonic() - started
    # Past the 0.2 s budget the pause suspended, and well short of the keepalives running out.
    assert 0.8 <= elapsed < 15.0


def test_the_park_cap_matches_the_routes(monkeypatch):
    from routes import inference
    assert research_runs._PREEMPT_PARK_STALL_CAP_SECONDS == inference._RAW_PARK_STALL_CAP_S


def test_without_preemption_a_silent_stream_times_out_as_before(monkeypatch, short_budgets):
    started = time.monotonic()
    with pytest.raises(research_runs.ModelFirstOutputTimeout):
        _run(monkeypatch, [30.0, "data: [DONE]"])
    assert time.monotonic() - started < 10.0


def test_without_preemption_an_idle_stream_times_out_as_before(monkeypatch, short_budgets):
    started = time.monotonic()
    with pytest.raises(research_runs.ModelOutputIdleTimeout):
        _run(monkeypatch, [_CONTENT % "report", 30.0, "data: [DONE]"])
    assert time.monotonic() - started < 10.0


def test_a_keepalive_without_a_pause_does_not_renew_the_budget(monkeypatch, short_budgets):
    started = time.monotonic()
    with pytest.raises(research_runs.ModelFirstOutputTimeout):
        _run(monkeypatch, [0.05, ": preempt-keepalive"] * 600)
    assert time.monotonic() - started < 10.0
