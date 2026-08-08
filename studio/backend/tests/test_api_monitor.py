# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import itertools
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.api_monitor import ApiMonitor, _trim
import routes.inference as inference_route


def _get_monitor(monkeypatch, *, enabled: bool):
    """Call GET /monitor against a monitor in a known state.

    Swaps the whole singleton rather than poking ``_enabled`` on it: the route
    reads ``snapshot()``, which does not consult the flag, so a shared monitor
    carrying rows from an earlier test would leak into the assertions.
    """
    monkeypatch.setattr(inference_route, "api_monitor", ApiMonitor(enabled = enabled))
    app = FastAPI()
    app.include_router(inference_route.studio_router)
    # Dict literal, not `overrides[key] = ...`: verify_import_hoist.py does not
    # see Load names inside an assignment target and reports the import unused.
    app.dependency_overrides = {get_current_subject: lambda: "test-user"}
    return TestClient(app).get("/monitor")


def test_api_monitor_tracks_reply_usage_and_context():
    monitor = ApiMonitor(max_entries = 3)

    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
        context_length = 100,
    )
    monitor.append_reply(entry_id, "hi")
    monitor.append_reply(entry_id, " there")
    monitor.set_usage(
        entry_id,
        prompt_tokens = 4,
        completion_tokens = 6,
    )
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["status"] == "completed"
    assert entry["reply"] == "hi there"
    assert entry["total_tokens"] == 10
    assert entry["context_usage"] == 0.1
    assert entry["duration_ms"] is not None


def test_api_monitor_summary_omits_full_prompt_and_reply():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "p" * 500,
    )
    monitor.set_reply(entry_id, "r" * 500)

    [summary] = monitor.snapshot(include_details = False)
    assert "prompt" not in summary
    assert "reply" not in summary
    assert summary["prompt_preview"].endswith("...")
    assert summary["reply_preview"].endswith("...")
    assert summary["prompt_truncated"] is True
    assert summary["reply_truncated"] is True

    detail = monitor.get(entry_id)
    assert detail is not None
    assert detail["prompt"] == "p" * 500
    assert detail["reply"] == "r" * 500


def test_api_monitor_filters_entries_by_subject():
    monitor = ApiMonitor(max_entries = 3)
    alice = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "alice prompt",
        subject = "alice",
    )
    bob = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "bob prompt",
        subject = "bob",
    )
    monitor.finish(bob)

    alice_entries = monitor.snapshot(subject = "alice")
    assert [entry["id"] for entry in alice_entries] == [alice]
    assert monitor.get(bob, subject = "alice") is None
    assert monitor.get(bob, subject = "bob")["id"] == bob
    assert monitor.active_count(subject = "alice") == 1
    assert monitor.active_count(subject = "bob") == 0


def test_api_monitor_keeps_bounded_recent_history():
    monitor = ApiMonitor(max_entries = 2)

    first = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "first",
    )
    second = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "second",
    )
    third = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "third",
    )
    monitor.finish(first)
    monitor.finish(second)
    monitor.finish(third)

    entries = monitor.snapshot()
    ids = [entry["id"] for entry in entries]
    assert ids[0] == third
    assert [entry["prompt"] for entry in entries] == ["third", "second"]
    assert first not in ids
    assert monitor.active_count() == 0


def test_api_monitor_keeps_running_entries_beyond_history_limit():
    monitor = ApiMonitor(max_entries = 1)

    running = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "running",
    )
    for prompt in ("done-1", "done-2", "done-3"):
        entry_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "m",
            prompt = prompt,
        )
        monitor.finish(entry_id)

    entries = monitor.snapshot()
    ids = [entry["id"] for entry in entries]
    assert running in ids
    assert monitor.active_count() == 1

    monitor.finish(running)
    [entry] = monitor.snapshot()
    assert entry["id"] == running
    assert entry["status"] == "completed"
    assert monitor.active_count() == 0


def test_api_monitor_finish_is_idempotent():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.finish(entry_id)
    first = monitor.snapshot()[0]
    monitor.finish(entry_id)
    second = monitor.snapshot()[0]
    assert first["finished_at"] == second["finished_at"]
    assert first["duration_ms"] == second["duration_ms"]


def test_api_monitor_preserves_authoritative_total_tokens():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_usage(
        entry_id,
        prompt_tokens = 10,
        completion_tokens = 20,
        total_tokens = 33,
    )
    # A later partial chunk omitting `total_tokens` must not clobber 33.
    monitor.set_usage(entry_id, prompt_tokens = 11)
    assert monitor.snapshot()[0]["total_tokens"] == 33


def test_api_monitor_recomputes_derived_total_tokens():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_usage(entry_id, prompt_tokens = 10)
    assert monitor.snapshot()[0]["total_tokens"] == 10

    monitor.set_usage(entry_id, completion_tokens = 20)
    entry = monitor.snapshot()[0]
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 20
    assert entry["total_tokens"] == 30


def test_api_monitor_duration_non_negative_under_clock_step(monkeypatch):
    import core.inference.api_monitor as m

    fake_now = [1000.0]
    monkeypatch.setattr(m.time, "time", lambda: fake_now[0])
    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/x",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    fake_now[0] = 500.0
    monitor.finish(entry_id)
    assert monitor.snapshot()[0]["duration_ms"] >= 0


def test_api_monitor_trim_guards_tiny_limit():
    assert _trim("abcdefgh", 2) == ".."
    assert _trim("abcdefgh", 0) == ""
    assert _trim("abcdefgh", 3) == "..."
    assert _trim("abcdefgh", 4) == "a..."
    assert _trim("abcdefgh", 100) == "abcdefgh"


def test_api_monitor_append_reply_caps_without_regrowing():
    import core.inference.api_monitor as m

    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "go",
    )
    monitor.append_reply(entry_id, "x" * (m._MAX_REPLY_CHARS + 500))
    capped = monitor.snapshot()[0]["reply"]
    assert len(capped) == m._MAX_REPLY_CHARS and capped.endswith("...")

    # Chunks past the cap must not change or grow the stored preview.
    monitor.append_reply(entry_id, "y" * 1000)
    assert monitor.snapshot()[0]["reply"] == capped


def test_api_monitor_append_reply_exact_cap_then_more_marks_truncated():
    import core.inference.api_monitor as m

    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "go",
    )
    # A reply landing exactly on the cap has no "..." marker yet.
    monitor.append_reply(entry_id, "x" * m._MAX_REPLY_CHARS)
    assert not monitor.snapshot()[0]["reply"].endswith("...")
    # One more chunk must record the truncation, not silently freeze.
    monitor.append_reply(entry_id, "y")
    reply = monitor.snapshot()[0]["reply"]
    assert len(reply) == m._MAX_REPLY_CHARS and reply.endswith("...")


def test_clear_keeps_the_callers_own_request_that_is_still_running():
    """Clear log drops history, and a request in flight is not history yet. Dropping it
    loses the request outright: the active count falls to zero and the finish that follows
    has no entry left to land on, so a completed call never appears at all."""
    monitor = ApiMonitor(max_entries = 4)
    done = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "finished",
        subject = "alice",
    )
    monitor.finish(done)
    live = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "in flight",
        subject = "alice",
    )

    monitor.clear(subject = "alice")

    assert [entry["id"] for entry in monitor.snapshot(subject = "alice")] == [live]
    assert monitor.active_count(subject = "alice") == 1
    # And the row is still there to be completed.
    monitor.finish(live)
    assert monitor.get(live, subject = "alice")["status"] == "completed"


def test_api_monitor_clear_is_scoped_to_one_subject():
    # Every other read is subject-scoped; an unscoped clear would erase another's history.
    monitor = ApiMonitor(max_entries = 4)
    alice = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "alice prompt",
        subject = "alice",
    )
    bob = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "bob prompt",
        subject = "bob",
    )
    # Finished first: a running row is a request in flight, not history, and clear keeps
    # it. This test is about the subject scoping, so it clears history and nothing else.
    monitor.finish(alice)

    monitor.clear(subject = "alice")
    assert monitor.snapshot(subject = "alice") == []
    assert [entry["id"] for entry in monitor.snapshot(subject = "bob")] == [bob]
    assert monitor.active_count(subject = "bob") == 1
    assert monitor.get(alice, subject = "alice") is None

    # Passing no subject is the explicit "everything" path.
    monitor.clear()
    assert monitor.snapshot(subject = "bob") == []


def test_api_monitor_records_whether_the_caller_used_an_api_key():
    # Studio's chat hits these endpoints on a JWT, and the panel auto-opens off this flag.
    monitor = ApiMonitor(max_entries = 4)
    ui = monitor.start(
        endpoint = "/api/inference/chat",
        method = "POST",
        model = "m",
        prompt = "hi",
        subject = "u",
    )
    api = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
        subject = "u",
        via_api_key = True,
    )
    by_id = {entry["id"]: entry for entry in monitor.snapshot(subject = "u")}
    assert by_id[ui]["via_api_key"] is False
    assert by_id[api]["via_api_key"] is True


def test_api_monitor_disabled_is_noop():
    monitor = ApiMonitor(max_entries = 3, enabled = False)

    request_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
        context_length = 100,
    )
    load_id = monitor.record_lifecycle(
        event = "load",
        model = "local-model",
        running = True,
    )
    unload_id = monitor.record_lifecycle(
        event = "unload",
        model = "local-model",
    )
    assert request_id == load_id == unload_id == ""

    # Every mutator must be a safe no-op on the falsy id.
    monitor.append_reply(request_id, "hi")
    monitor.set_reply(request_id, "hi")
    monitor.set_usage(request_id, prompt_tokens = 4, completion_tokens = 6)
    monitor.relabel(load_id, "renamed-model")
    monitor.set_progress(load_id, 50)
    monitor.finish(load_id)
    monitor.fail_open(load_id, "boom")
    monitor.fail(request_id, "boom")
    monitor.discard(unload_id)

    assert monitor.snapshot() == []
    assert monitor.active_count() == 0
    assert monitor.get(request_id) is None


def test_api_monitor_disable_env_var_truthy(monkeypatch):
    import core.inference.api_monitor as m
    for value in ("1", "true", "yes", "on", "TRUE", "On", " yes "):
        monkeypatch.setenv(m._DISABLE_ENV, value)
        assert m._api_monitor_disabled() is True, value


def test_api_monitor_disable_env_var_falsy(monkeypatch):
    import core.inference.api_monitor as m
    for value in ("", "0", "false", "no", "off", "disabled"):
        monkeypatch.setenv(m._DISABLE_ENV, value)
        assert m._api_monitor_disabled() is False, value


def test_api_monitor_disable_env_var_unset(monkeypatch):
    import core.inference.api_monitor as m
    monkeypatch.delenv(m._DISABLE_ENV, raising = False)
    assert m._api_monitor_disabled() is False


# ── model lifecycle rows (load / unload) ────────────────────────────


def test_lifecycle_load_row_opens_running_then_closes():
    monitor = ApiMonitor(max_entries = 5)
    event_id = monitor.record_lifecycle(event = "load", model = "org/A-GGUF", running = True)
    row = monitor.snapshot()[0]
    assert row["kind"] == "lifecycle" and row["event"] == "load"
    assert row["status"] == "running" and row["duration_ms"] is None
    # A load in progress is not an in-flight API request.
    assert monitor.active_count() == 0

    monitor.relabel(event_id, "org/A-GGUF:Q4_K_M")
    monitor.finish(event_id)
    row = monitor.snapshot()[0]
    assert row["status"] == "completed"
    assert row["model"] == "org/A-GGUF:Q4_K_M"
    assert row["duration_ms"] is not None


def test_lifecycle_unload_row_is_terminal_on_arrival():
    monitor = ApiMonitor(max_entries = 5)
    monitor.record_lifecycle(event = "unload", model = "org/A-GGUF", reason = "idle")
    row = monitor.snapshot()[0]
    assert row["status"] == "completed"
    assert (row["event"], row["reason"]) == ("unload", "idle")
    assert monitor.active_count() == 0


def test_lifecycle_rows_are_visible_to_every_subject():
    # A load is server-wide, so it must not vanish for other API keys like a request does.
    monitor = ApiMonitor(max_entries = 5)
    monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
        subject = "alice",
    )
    event_id = monitor.record_lifecycle(event = "unload", model = "org/A-GGUF")

    bob = monitor.snapshot(subject = "bob")
    assert [r["kind"] for r in bob] == ["lifecycle"]
    assert monitor.get(event_id, subject = "bob") is not None
    assert len(monitor.snapshot(subject = "alice")) == 2


def test_request_rows_stay_private_to_their_subject():
    monitor = ApiMonitor(max_entries = 5)
    rid = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
        subject = "alice",
    )
    assert monitor.snapshot(subject = "bob") == []
    assert monitor.get(rid, subject = "bob") is None


def test_discard_drops_a_row_that_never_happened():
    # A load that found the model already resident must leave no trace.
    monitor = ApiMonitor(max_entries = 5)
    event_id = monitor.record_lifecycle(event = "load", model = "org/A-GGUF", running = True)
    monitor.discard(event_id)
    assert monitor.snapshot() == []
    monitor.discard(event_id)  # idempotent


def test_fail_open_never_touches_a_finished_row():
    # Called from a finally, so it must not stamp an error onto a load that succeeded.
    monitor = ApiMonitor(max_entries = 5)
    event_id = monitor.record_lifecycle(event = "load", model = "org/A-GGUF", running = True)
    monitor.finish(event_id)
    monitor.fail_open(event_id, "Load did not complete")
    row = monitor.snapshot()[0]
    assert row["status"] == "completed" and row["error"] is None

    still_open = monitor.record_lifecycle(event = "load", model = "org/B-GGUF", running = True)
    monitor.fail_open(still_open, "Load did not complete")
    assert monitor.snapshot()[0]["status"] == "error"


def test_lifecycle_rows_share_the_retention_budget():
    monitor = ApiMonitor(max_entries = 2)
    for i in range(4):
        monitor.record_lifecycle(event = "unload", model = f"org/M{i}")
    models = [r["model"] for r in monitor.snapshot()]
    assert models == ["org/M3", "org/M2"]


def test_request_rows_report_kind_request():
    monitor = ApiMonitor(max_entries = 2)
    monitor.start(endpoint = "/v1/chat/completions", method = "POST", model = "m", prompt = "hi")
    assert monitor.snapshot()[0]["kind"] == "request"


def test_clear_hides_shared_lifecycle_rows_for_that_caller_only():
    """A lifecycle row is shared, so it is visible to every caller but owned by
    none. A subject-scoped clear dropped only that subject's own rows, so the
    shared ones survived and the reload straight after "Clear log" brought them
    back: the button visibly did nothing to them. Dropping them outright is not
    an option either, since that erases another caller's history.
    """
    monitor = ApiMonitor(max_entries = 10)
    mine = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "org/A",
        prompt = "user: hi",
        subject = "alice",
    )
    monitor.finish(mine)
    shared = monitor.record_lifecycle(event = "unload", model = "org/A")

    assert {e["id"] for e in monitor.snapshot(subject = "alice")} == {mine, shared}
    assert {e["id"] for e in monitor.snapshot(subject = "bob")} == {shared}

    monitor.clear(subject = "alice")

    assert monitor.snapshot(subject = "alice") == []
    # Hidden for alice, not deleted, so bob's view is untouched.
    assert {e["id"] for e in monitor.snapshot(subject = "bob")} == {shared}
    assert monitor.get(shared, subject = "alice") is None
    assert monitor.get(shared, subject = "bob") is not None


def test_clear_leaves_a_running_shared_row_visible():
    """A load still in progress is live state, not history, so clearing the log
    must not hide the row that shows it."""
    monitor = ApiMonitor(max_entries = 10)
    running = monitor.record_lifecycle(event = "load", model = "org/A", running = True)
    monitor.clear(subject = "alice")
    assert {e["id"] for e in monitor.snapshot(subject = "alice")} == {running}


def test_hidden_shared_ids_do_not_outlive_their_entries():
    """The hidden set names rows that exist, so it stays bounded by the ring
    buffer instead of growing for the life of the process."""
    monitor = ApiMonitor(max_entries = 2)
    monitor.record_lifecycle(event = "unload", model = "org/A")
    monitor.clear(subject = "alice")
    assert monitor._hidden_shared.get("alice")
    for i in range(5):
        monitor.record_lifecycle(event = "unload", model = f"org/M{i}")
    assert not monitor._hidden_shared.get("alice")


def test_an_api_triggered_lifecycle_row_carries_the_attribution():
    """The overlay opens on API-key traffic only. An auto-switch or auto-download
    that is refused never reaches api_monitor.start, so the lifecycle row is the
    whole trace of that request; without the attribution the monitor stayed shut
    on exactly the failures it exists to surface."""
    monitor = ApiMonitor(max_entries = 5)

    api_load = monitor.record_lifecycle(
        event = "load", model = "org/Repo-GGUF", running = True, via_api_key = True
    )
    monitor.record_lifecycle(event = "unload", model = "org/Repo-GGUF", reason = "idle")

    rows = {e["id"]: e for e in monitor.snapshot()}
    assert rows[api_load]["via_api_key"] is True
    # A background unload is not API traffic and must not pop the overlay.
    idle = [e for e in rows.values() if e["event"] == "unload"]
    assert idle and all(e["via_api_key"] is False for e in idle)

    # The failure path keeps it: failing the row must not drop the attribution.
    monitor.fail(api_load, error = "auto-switch refused")
    after = {e["id"]: e for e in monitor.snapshot()}
    assert after[api_load]["via_api_key"] is True
    assert after[api_load]["status"] == "error"


def test_an_api_lifecycle_row_pops_the_overlay_only_for_its_own_caller():
    """A lifecycle row is shared so it appears in every monitor list, and it also
    carries via_api_key, which is what the floating panel auto-opens on. Reported
    to everyone, the panel springs open in a browser that had nothing to do with
    the traffic. The row stays visible to all; only the attribution is scoped."""
    monitor = ApiMonitor(max_entries = 5)

    row = monitor.record_lifecycle(
        event = "load",
        model = "org/Repo-GGUF",
        running = True,
        via_api_key = True,
        subject = "alice",
    )

    mine = {e["id"]: e for e in monitor.snapshot(subject = "alice")}
    theirs = {e["id"]: e for e in monitor.snapshot(subject = "bob")}
    # Shared visibility is deliberate and must survive: bob still sees the load.
    assert row in mine and row in theirs
    assert mine[row]["via_api_key"] is True
    assert theirs[row]["via_api_key"] is False

    # The details read is scoped the same way, so the panel cannot re-derive it.
    assert monitor.get(row, subject = "alice")["via_api_key"] is True
    assert monitor.get(row, subject = "bob")["via_api_key"] is False
    # An unscoped read (internal callers) still sees the row's own flag.
    assert monitor.get(row)["via_api_key"] is True


def test_clearing_hides_a_shared_row_this_caller_owns_rather_than_deleting_it():
    """An API-key load now owns its shared row. A subject-scoped clear drops that
    subject's rows, so without this the owner's Clear would delete a row every
    other caller can still see and wipe it out of their history too."""
    monitor = ApiMonitor(max_entries = 10)
    row = monitor.record_lifecycle(
        event = "unload", model = "org/Repo-GGUF", via_api_key = True, subject = "alice"
    )

    monitor.clear(subject = "alice")

    assert monitor.snapshot(subject = "alice") == []
    assert {e["id"] for e in monitor.snapshot(subject = "bob")} == {row}
    assert monitor.get(row, subject = "bob") is not None


# ── stream framing must not depend on the monitor ───────────────────


def test_sse_done_detection_accepts_both_spacings():
    done = inference_route._is_openai_sse_done
    assert done("data: [DONE]") is True
    assert done("data:[DONE]") is True
    assert done('data: {"choices": []}') is False
    assert done("event: ping") is False
    assert done("") is False


def test_sse_done_detection_is_independent_of_the_monitor():
    """The external-provider proxy sets ``sent_done`` from the line itself.

    It used to read the monitor helper's return, which is None for every line
    once recording is off -- so the proxy appended a second [DONE] after the
    provider's own, changing client-visible framing based on a logging flag.
    """
    line = "data: [DONE]"
    assert inference_route._monitor_openai_sse_line(None, line) is None
    assert inference_route._is_openai_sse_done(line) is True


# ── /monitor route: the disabled state has to reach the UI ──────────


def test_monitor_route_reports_enabled(monkeypatch):
    response = _get_monitor(monkeypatch, enabled = True)

    assert response.status_code == 200
    assert response.json()["logging_enabled"] is True


def test_monitor_route_reports_disabled(monkeypatch):
    response = _get_monitor(monkeypatch, enabled = False)

    # An empty list on its own is indistinguishable from "no traffic yet", so the
    # console needs the flag to explain itself instead of claiming idleness.
    assert response.status_code == 200
    payload = response.json()
    assert payload["logging_enabled"] is False
    assert payload["entries"] == []


def test_monitor_route_disabled_still_hides_recorded_rows(monkeypatch):
    """A disabled monitor records nothing, so the route reports an empty list
    even after traffic that would otherwise have shown up."""
    monkeypatch.setattr(inference_route, "api_monitor", ApiMonitor(enabled = False))
    inference_route.api_monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
        subject = "test-user",
    )
    app = FastAPI()
    app.include_router(inference_route.studio_router)
    app.dependency_overrides = {get_current_subject: lambda: "test-user"}
    payload = TestClient(app).get("/monitor").json()

    assert payload["logging_enabled"] is False
    assert payload["entries"] == []


def test_set_perf_records_stats_and_snapshot_reports_them():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    # set_reply never stamps TTFT; this is the case the prompt_ms fallback exists for.
    monitor.set_reply(entry_id, "hi")
    monitor.set_perf(entry_id, tok_per_sec = 42.5, prompt_ms = 123.4, stop_reason = "length")
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["tok_per_sec"] == 42.5
    assert entry["ttft_ms"] == 123
    assert entry["stop_reason"] == "length"


def test_measured_ttft_wins_over_engine_prefill():
    # Queue wait precedes llama-server, so prefill-only prompt_ms under-reports TTFT.
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    entry = next(e for e in monitor._entries if e.id == entry_id)
    entry.started_monotonic -= 2.0
    monitor.append_reply(entry_id, "hi")
    monitor.set_perf(entry_id, tok_per_sec = 42.5, prompt_ms = 120.0)
    monitor.finish(entry_id)

    [snapshot] = monitor.snapshot()
    assert snapshot["ttft_ms"] >= 2000


def test_set_perf_rejects_non_finite_values():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.set_perf(entry_id, tok_per_sec = float("nan"), prompt_ms = float("inf"))
    monitor.set_perf(entry_id, tok_per_sec = "bogus", prompt_ms = None)
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["tok_per_sec"] is None
    assert entry["ttft_ms"] is None


def test_full_response_reply_does_not_stamp_ttft():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.set_reply(entry_id, "full response")
    monitor.append_reply(entry_id, " tail", stamp_first_token = False)
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["ttft_ms"] is None
    assert entry["tok_per_sec"] is None


def test_set_usage_rejects_malformed_token_counts():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.append_reply(entry_id, "hi")
    monitor.set_usage(
        entry_id,
        prompt_tokens = -3,
        completion_tokens = "bogus",
        total_tokens = 12,
    )
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["prompt_tokens"] is None
    assert entry["completion_tokens"] is None
    assert entry["total_tokens"] == 12
    assert entry["tok_per_sec"] is None


def test_mark_first_token_stamps_ttft_for_reasoning_only_streams():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.mark_first_token(entry_id)
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["ttft_ms"] is not None

    entry_id2 = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.mark_first_token(entry_id2)
    first = monitor._find_locked(entry_id2).first_token_monotonic
    monitor.append_reply(entry_id2, "visible")
    assert monitor._find_locked(entry_id2).first_token_monotonic == first


def test_queue_state_counts_direct_overflow_as_queued(monkeypatch):
    # Direct calls hold no lease, so overflow past capacity must show as queued,
    # not get clamped out of the readout.
    from types import SimpleNamespace

    import routes.inference as inf

    monkeypatch.setattr(
        inf,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_diffusion = False,
            base_url = "http://llama.test",
            effective_parallel_slots = 1,
        ),
    )
    monkeypatch.setattr(inf, "peek_llama_admission_snapshot", lambda _base: None)
    monkeypatch.setattr(inf, "_direct_llama_inflight", 2)

    state = inf._monitor_queue_state()
    assert state == {"capacity": 1, "active": 1, "queued": 1, "free": 0}


def test_non_streaming_responses_reports_its_finish_reason(monkeypatch):
    # Stop reason must be read off the choice, or the row shows a blank.
    import routes.inference as inf

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/responses",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
        "timings": {"predicted_per_second": 12.5, "prompt_ms": 80.0},
    }
    choices = body.get("choices", [])
    # Mirrors the call the route makes at the end of _responses_non_streaming.
    inf._monitor_usage(
        entry_id,
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        None,
        timings = body.get("timings"),
        stop_reason = (choices[0].get("finish_reason") if choices else None),
    )
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["stop_reason"] == "length"
    assert entry["tok_per_sec"] == 12.5
    assert entry["ttft_ms"] == 80


def test_parked_tool_resume_counts_as_queued():
    # Without resume tickets counted, the readout shows a full server with nothing queued.
    from core.inference.llama_admission import LlamaAdmissionQueue

    queue = LlamaAdmissionQueue("http://llama.test")
    queue._unpark_tickets.append(1)
    assert queue.snapshot().queued == 1


def test_free_never_reports_a_slot_admission_would_refuse():
    """`free` is what a new arrival could take, so it must track _can_admit_locked:
    a resume ticket holds a slot back, so counting it in `queued` without dropping it
    from `free` prints free slots next to a queued request.
    """
    from core.inference.llama_admission import LlamaAdmissionQueue
    for capacity, held, tickets in itertools.product(range(1, 5), range(0, 5), range(0, 3)):
        if held > capacity:
            continue
        queue = LlamaAdmissionQueue("http://llama.test")
        queue._resize_pool_locked(capacity)
        if any(queue._take_slot_locked(0) is None for _ in range(held)):
            continue
        for _ in range(tickets):
            queue._unpark_seq += 1
            queue._unpark_tickets.append(queue._unpark_seq)

        snapshot = queue.snapshot()
        admittable = queue._can_admit_locked(len(queue._unpark_tickets))
        assert (snapshot.free > 0) == admittable, (
            f"capacity={capacity} held={held} tickets={tickets}: "
            f"free={snapshot.free} but _can_admit_locked={admittable}"
        )


def test_queue_panel_never_shows_a_free_slot_next_to_a_resume(monkeypatch):
    """The panel derives free from the snapshot, so it inherits that invariant."""
    from types import SimpleNamespace

    import routes.inference as inf
    from core.inference.llama_admission import LlamaAdmissionQueue

    queue = LlamaAdmissionQueue("http://llama.test")
    queue._unpark_tickets.append(1)
    monkeypatch.setattr(
        inf,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_diffusion = False,
            base_url = "http://llama.test",
            effective_parallel_slots = 1,
        ),
    )
    monkeypatch.setattr(inf, "peek_llama_admission_snapshot", lambda _base: queue.snapshot())
    monkeypatch.setattr(inf, "_direct_llama_inflight", 0)

    assert inf._monitor_queue_state() == {"capacity": 1, "active": 0, "queued": 1, "free": 0}


def test_set_perf_survives_an_out_of_range_engine_number():
    """float() on a huge upstream int raises OverflowError, which is not ValueError,
    and these helpers run inside streaming generators where a raise truncates the
    user's response.
    """
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    huge = int("9" * 400)
    monitor.set_perf(entry_id, tok_per_sec = huge, prompt_ms = huge, stop_reason = "stop")
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["tok_per_sec"] is None
    assert entry["ttft_ms"] is None
    assert entry["stop_reason"] == "stop"


@pytest.mark.parametrize(
    "chunk",
    [
        {"choices": [{"delta": {"content": "x"}}], "timings": {"predicted_per_second": 10**400}},
        {"choices": [{"delta": {"content": "x"}}], "timings": {"prompt_ms": 10**400}},
        {"choices": [{"delta": {"content": "x"}}], "usage": {"total_tokens": 10**400}},
        {"choices": [{"delta": {"content": "x"}}], "usage": "not-a-dict"},
        {"choices": [{"delta": {"content": "x"}}], "timings": "not-a-dict"},
        {"choices": "not-a-list"},
        {"choices": [{"delta": "not-a-dict"}]},
        {},
    ],
)
def test_monitor_chunk_never_raises_on_a_malformed_upstream_chunk(monkeypatch, chunk):
    """A raise here escapes into the SSE generator and truncates the response."""
    import routes.inference as inf

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    inf._monitor_openai_chunk(entry_id, chunk, 4096, streaming = True)
    # snapshot() divides by the recorded counts, so it has to survive them too.
    assert monitor.snapshot()


def test_direct_llama_counter_is_started_last_before_its_guarding_try():
    """Anything between started() and the try leaks a permanent +1 if it raises, and
    this counter has no reset hook, so one leak pins the slot panel at busy until the
    process restarts.
    """
    import ast
    import inspect
    import textwrap

    for func_name in ("openai_completions", "openai_embeddings", "_direct_llama_request"):
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(inference_route, func_name))))
        started = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_direct_llama_request_started"
        ]
        assert started, f"{func_name}: no _direct_llama_request_started() call found"

        for node in ast.walk(tree):
            block = getattr(node, "body", None)
            if not isinstance(block, list):
                continue
            for index, stmt in enumerate(block):
                if not any(stmt is call for call in started):
                    continue
                assert index + 1 < len(block) and isinstance(block[index + 1], ast.Try), (
                    f"{func_name}: _direct_llama_request_started() is not immediately "
                    f"followed by the try whose finally decrements it:\n"
                    + "\n".join(ast.unparse(s) for s in block[index : index + 3])
                )


def _llama_slot_readout(
    monkeypatch,
    *,
    is_audio = False,
    slots = 4,
):
    """Point the slot readout at a loaded llama-server with ``slots`` free slots."""
    from types import SimpleNamespace

    import routes.inference as inf

    backend = SimpleNamespace(
        is_loaded = True,
        is_diffusion = False,
        is_vision = True,
        base_url = "http://llama.test",
        effective_parallel_slots = slots,
        context_length = 4096,
        model_identifier = "some/tts-GGUF",
        _is_audio = is_audio,
        _audio_type = "codec",
        _auth_headers = None,
    )
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf, "peek_llama_admission_snapshot", lambda _base: None)
    monkeypatch.setattr(inf, "_direct_llama_inflight", 0)
    return backend


def test_queue_state_counts_rag_vision_captioning(monkeypatch):
    """RAG captioning/OCR reaches llama-server with no lease (see the
    LlamaAdmissionQueue docstring), so without the direct count the panel reported an
    idle server for the whole ingestion.
    """
    import routes.inference as inf
    from core.rag import captioner

    _llama_slot_readout(monkeypatch)
    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "a caption"}}]}

    def _fake_post(*_a, **_k):
        seen["state"] = inf._monitor_queue_state()
        return _Resp()

    monkeypatch.setattr("httpx.post", _fake_post)

    assert captioner._caption_one("http://llama.test", "local", b"png", 5.0) == "a caption"
    assert seen["state"] == {"capacity": 4, "active": 1, "queued": 0, "free": 3}
    assert inf._monitor_queue_state() == {"capacity": 4, "active": 0, "queued": 0, "free": 4}


@pytest.mark.parametrize("fails", [False, True])
def test_rag_vision_call_balances_the_direct_counter(monkeypatch, fails):
    """Both outcomes must return the count to zero: a failed vision call is swallowed
    as non-fatal, so a leak here would pin the panel at busy for the whole session.
    """
    import routes.inference as inf
    from core.rag import captioner

    _llama_slot_readout(monkeypatch)

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise RuntimeError("llama-server said no")

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def _fake_post(*_a, **_k):
        if fails:
            raise RuntimeError("connection refused")
        return _Resp()

    monkeypatch.setattr("httpx.post", _fake_post)

    assert captioner._ocr_one("http://llama.test", "local", b"png", 5.0) is None
    assert inf._direct_llama_inflight == 0


def _run_gguf_tts(monkeypatch, backend, generate):
    """Drive POST /audio/generate onto the GGUF (llama-server) branch."""
    import asyncio

    import routes.inference as inf
    from models.inference import ChatCompletionRequest

    backend.generate_audio_response = generate
    monkeypatch.setattr(inf, "_llama_public_model_id", lambda *_a, **_k: "some/tts-GGUF")
    monkeypatch.setattr(inf, "_fill_recommended_sampling_openai", lambda *_a, **_k: None)

    async def _noop_switch(*_a, **_k):
        return None

    monkeypatch.setattr(inf, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = "some/tts-GGUF", messages = [{"role": "user", "content": "hi"}]
    )
    return asyncio.run(inf.generate_audio(payload, request = None, current_subject = "t"))


def test_queue_state_counts_gguf_tts(monkeypatch):
    """GGUF TTS holds a llama-server slot for the whole request without a lease."""
    import routes.inference as inf

    backend = _llama_slot_readout(monkeypatch, is_audio = True)
    seen = {}

    def _generate(**_kwargs):
        seen["state"] = inf._monitor_queue_state()
        return (b"RIFFfake", 24000)

    _run_gguf_tts(monkeypatch, backend, _generate)
    assert seen["state"] == {"capacity": 4, "active": 1, "queued": 0, "free": 3}
    assert inf._monitor_queue_state() == {"capacity": 4, "active": 0, "queued": 0, "free": 4}


@pytest.mark.parametrize("outcome", ["completed", "raised", "disconnected"])
def test_gguf_tts_balances_the_direct_counter(monkeypatch, outcome):
    """Every exit from the TTS branch gives the slot back; the counter has no reset hook."""
    from fastapi import HTTPException

    import routes.inference as inf

    backend = _llama_slot_readout(monkeypatch, is_audio = True)

    def _generate(**kwargs):
        if outcome == "raised":
            raise RuntimeError("llama-server died")
        if outcome == "disconnected":
            kwargs["cancel_event"].set()
            raise RuntimeError("stream closed")
        return (b"RIFFfake", 24000)

    if outcome == "completed":
        _run_gguf_tts(monkeypatch, backend, _generate)
    else:
        with pytest.raises(HTTPException) as excinfo:
            _run_gguf_tts(monkeypatch, backend, _generate)
        assert excinfo.value.status_code == (499 if outcome == "disconnected" else 500)

    assert inf._direct_llama_inflight == 0


def test_top_level_provider_tool_event_stamps_first_token(monkeypatch):
    # _toolEvent rides the chunk itself, beside choices, with an empty delta.
    import routes.inference as inf

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "provider/model",
        prompt = "hi",
    )
    inf._monitor_openai_chunk(
        entry_id,
        {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "_toolEvent": {"type": "web_search"},
        },
        streaming = True,
    )
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["ttft_ms"] is not None


def test_disagreeing_choice_finish_reasons_report_no_stop_reason(monkeypatch):
    # The row aggregates every choice, so one choice's reason is only the request's
    # when they all agree.
    import routes.inference as inf

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf, "api_monitor", monitor)

    def row_for(reasons):
        entry_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        inf._monitor_openai_chunk(
            entry_id,
            {
                "choices": [
                    {"index": i, "message": {"content": "x"}, "finish_reason": r}
                    for i, r in enumerate(reasons)
                ]
            },
        )
        monitor.finish(entry_id)
        return next(r for r in monitor.snapshot() if r["id"] == entry_id)["stop_reason"]

    assert row_for(["stop"]) == "stop"
    assert row_for(["stop", "stop"]) == "stop"
    assert row_for(["stop", "length"]) is None


def test_streamed_choice_finish_reasons_are_compared_across_chunks(monkeypatch):
    # llama-server streams an n > 1 request as one single-choice chunk per sample, so
    # agreement can only be judged across the whole stream, not inside one chunk.
    import routes.inference as inf

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf, "api_monitor", monitor)

    def row_for(reasons):
        entry_id = monitor.start(
            endpoint = "/v1/completions",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        for i, reason in enumerate(reasons):
            inf._monitor_openai_sse_line(
                entry_id,
                "data: "
                + json.dumps({"choices": [{"index": i, "text": "x", "finish_reason": reason}]}),
            )
        monitor.finish(entry_id)
        return next(r for r in monitor.snapshot() if r["id"] == entry_id)["stop_reason"]

    assert row_for(["stop"]) == "stop"
    assert row_for(["stop", "stop"]) == "stop"
    # Used to report "length": each chunk agreed with itself and the last one won.
    assert row_for(["stop", "length"]) is None


def test_streamed_stop_reason_is_withheld_until_the_request_finishes(monkeypatch):
    # An n > 1 stream finishes its choices in separate chunks, so publishing the first one
    # would state a request-level verdict while the rest are still running, then retract it.
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )

    def row():
        return next(r for r in monitor.snapshot() if r["id"] == entry_id)

    monitor.note_stop_reason(entry_id, "stop")
    assert row()["finished_at"] is None
    assert row()["stop_reason"] is None
    monitor.note_stop_reason(entry_id, "stop")
    assert row()["stop_reason"] is None
    monitor.finish(entry_id)
    assert row()["stop_reason"] == "stop"


@pytest.mark.parametrize("writer", ["note_stop_reason", "set_perf"])
@pytest.mark.parametrize(
    "status, expected",
    [("completed", "stop"), ("cancelled", None), ("failed", None), ("error", None)],
)
def test_stop_reason_is_kept_only_by_completed_requests(writer, status, expected):
    # A cancelled n > 1 stream stopped its remaining choices rather than hearing from
    # them, and several local streams record "stop" through set_perf on the way out of a
    # cancelled loop, before the cancellation is stamped. Neither describes how it ended.
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    if writer == "set_perf":
        monitor.set_perf(entry_id, stop_reason = "stop")
    else:
        monitor.note_stop_reason(entry_id, "stop")
    if status == "error":
        monitor.fail(entry_id, "boom")
    else:
        monitor.finish(entry_id, status)
    row = next(r for r in monitor.snapshot() if r["id"] == entry_id)
    assert row["status"] == status
    assert row["stop_reason"] == expected


def test_non_streaming_stop_reason_survives_the_finish(monkeypatch):
    # Nothing accumulates on that path, so resolving at finish must not clear what
    # set_perf already recorded.
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_perf(entry_id, stop_reason = "stop")
    monitor.finish(entry_id)
    assert next(r for r in monitor.snapshot() if r["id"] == entry_id)["stop_reason"] == "stop"


@pytest.mark.parametrize(
    "queue, expected",
    [
        ({"capacity": 4, "active": 1, "queued": 0, "free": 3}, "generating"),
        ({"capacity": 4, "active": 0, "queued": 2, "free": 0}, "generating"),
        ({"capacity": 4, "active": 0, "queued": 0, "free": 4}, "ready"),
        (None, "ready"),
    ],
)
def test_monitor_status_counts_slots_no_row_can_see(monkeypatch, queue, expected):
    # A direct llama call (RAG caption/OCR) opens no row, logging may be off, and another
    # subject's work is not counted here, so rows alone would report Ready beside a busy
    # slot readout the same response carries.
    import routes.inference as inf

    monkeypatch.setattr(inf, "api_monitor", ApiMonitor(max_entries = 3))
    monkeypatch.setattr(inf, "_monitor_active_model", lambda: "org/M-GGUF")
    monkeypatch.setattr(inf, "_monitor_context_length", lambda: 4096)
    monkeypatch.setattr(inf, "_monitor_queue_state", lambda: queue)

    app = FastAPI()
    app.include_router(inf.studio_router)
    app.dependency_overrides = {get_current_subject: lambda: "alice"}
    body = TestClient(app).get("/monitor").json()

    assert body["active_requests"] == 0
    assert body["status"] == expected


def test_monitor_status_is_idle_without_a_model(monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr(inf, "api_monitor", ApiMonitor(max_entries = 3))
    monkeypatch.setattr(inf, "_monitor_active_model", lambda: None)
    monkeypatch.setattr(inf, "_monitor_context_length", lambda: None)
    monkeypatch.setattr(inf, "_monitor_queue_state", lambda: None)

    app = FastAPI()
    app.include_router(inf.studio_router)
    app.dependency_overrides = {get_current_subject: lambda: "alice"}
    assert TestClient(app).get("/monitor").json()["status"] == "idle"


def test_tool_card_starts_ttft_but_not_the_token_rate_clock(monkeypatch):
    # A tool card is client output, so it starts TTFT. It is not decoded output, so the
    # tool run (or a human confirming one) after it must not count as decoding time --
    # dividing by that wait reports a rate near zero.
    import core.inference.api_monitor as m

    clock = [100.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )

    monitor.mark_first_token(entry_id, decoded = False)
    clock[0] = 160.0  # a minute of tool run / human confirmation
    monitor.append_reply(entry_id, "first real token")
    clock[0] = 162.0  # two seconds of decoding
    monitor.set_usage(entry_id, completion_tokens = 21)
    monitor.finish(entry_id)

    row = next(r for r in monitor.snapshot() if r["id"] == entry_id)
    # TTFT still measures from the card the user actually saw.
    assert row["ttft_ms"] == 0
    # 20 gaps over 2s, not over 62s.
    assert row["tok_per_sec"] == 10.0


def test_a_decoded_first_token_starts_both_clocks(monkeypatch):
    import core.inference.api_monitor as m

    clock = [100.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )

    # Reasoning tokens are decoded output, so they start the rate clock as before.
    monitor.mark_first_token(entry_id)
    clock[0] = 102.0
    monitor.set_usage(entry_id, completion_tokens = 21)
    monitor.finish(entry_id)

    assert next(r for r in monitor.snapshot() if r["id"] == entry_id)["tok_per_sec"] == 10.0


def test_a_stop_reason_written_after_finish_escapes_the_clearing():
    # Why every route records the reason before finish(): the settle runs once, at the
    # terminal transition, so a later write would put a natural stop reason back onto a
    # cancelled row.
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_perf(entry_id, stop_reason = "stop")
    monitor.finish(entry_id, "cancelled")
    assert next(r for r in monitor.snapshot() if r["id"] == entry_id)["stop_reason"] is None

    monitor.set_perf(entry_id, stop_reason = "stop")
    assert next(r for r in monitor.snapshot() if r["id"] == entry_id)["stop_reason"] == "stop"


@pytest.mark.parametrize(
    "line, dropped",
    [
        ('data: {"choices":[],"usage":{"completion_tokens":9}}', True),
        # A content chunk carrying inline usage still has to reach the client.
        ('data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":9}}', False),
        ('data: {"choices":[{"delta":{"content":"x"}}]}', False),
        # Content that quotes the key gets past the cheap prefilter, so only the parse
        # can tell it apart from a real usage chunk.
        ('data: {"choices":[{"delta":{"content":"\\"usage\\":1"}}]}', False),
        ("data: [DONE]", False),
        ("data: not json", False),
        (": keepalive comment", False),
    ],
)
def test_usage_only_sse_is_recognized_for_relay_filtering(line, dropped):
    # Providers are asked for stream usage regardless of what the caller wanted, so the
    # proxy has to drop the standalone chunk on the way out: a client that did not opt in
    # would index choices[0] on it. Same rule _cmpl_stream_event_out applies locally.
    import routes.inference as inf
    assert inf._is_openai_usage_only_sse(line) is dropped


@pytest.mark.parametrize("include_usage, expected", [(True, True), (False, False)])
def test_wants_stream_usage_reads_the_callers_opt_in(include_usage, expected):
    import routes.inference as inf
    from types import SimpleNamespace

    payload = SimpleNamespace(stream_options = {"include_usage": include_usage})
    assert inf._wants_stream_usage(payload) is expected
    assert inf._wants_stream_usage(SimpleNamespace(stream_options = None)) is False


def test_direct_llama_work_is_busy_without_the_admission_snapshot(monkeypatch):
    # With UNSLOTH_LLAMA_ADMISSION_CONTROL=off the queue readout is None, so a caption or
    # OCR call (which opens no row) would leave the row saying Ready while the server works.
    import routes.inference as inf

    monkeypatch.setattr(inf, "api_monitor", ApiMonitor(max_entries = 3))
    monkeypatch.setattr(inf, "_monitor_active_model", lambda: "org/M-GGUF")
    monkeypatch.setattr(inf, "_monitor_context_length", lambda: 4096)
    monkeypatch.setattr(inf, "_monitor_queue_state", lambda: None)
    monkeypatch.setattr(inf, "_direct_llama_inflight", 1)

    app = FastAPI()
    app.include_router(inf.studio_router)
    app.dependency_overrides = {get_current_subject: lambda: "alice"}
    body = TestClient(app).get("/monitor").json()

    assert body["active_requests"] == 0
    assert body["queue"] is None
    assert body["status"] == "generating"

    # Back to ready once it finishes.
    monkeypatch.setattr(inf, "_direct_llama_inflight", 0)
    assert TestClient(app).get("/monitor").json()["status"] == "ready"


def test_direct_busy_reads_the_live_counter(monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr(inf, "_direct_llama_inflight", 0)
    assert inf._direct_llama_is_busy() is False
    monkeypatch.setattr(inf, "_direct_llama_inflight", 2)
    assert inf._direct_llama_is_busy() is True


def test_the_decode_span_comes_only_from_engine_timings(monkeypatch):
    """The tile rates on decode_ms, so it must never carry a guess. Regression guard
    for a model generating at 50 tok/s reading as 5 behind a busy slot: duration_ms
    carries that wait and decode_ms must not."""
    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    inference_route._monitor_usage(
        entry_id,
        {"prompt_tokens": 11, "completion_tokens": 50},
        4096,
        timings = {"prompt_ms": 9000.0, "predicted_ms": 1000.0, "predicted_per_second": 50.0},
    )
    monitor.finish(entry_id)

    [row] = monitor.snapshot()
    assert row["decode_ms"] == 1000
    assert row["completion_tokens"] / (row["decode_ms"] / 1000) == 50.0


def test_a_timings_only_final_chunk_still_sets_the_decode_span(monkeypatch):
    """llama-server can end a stream with timings and no usage."""
    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    inference_route._monitor_usage(entry_id, None, None, timings = {"predicted_ms": 1000})
    monitor.finish(entry_id)

    assert monitor.snapshot()[0]["decode_ms"] == 1000


def test_a_streamed_reply_alone_reports_no_decode_span():
    """Timing the stream cannot say how many tokens rode in the first chunk, and never
    sees reasoning tokens. Report nothing rather than a rate that inflates."""
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    monitor.append_reply(entry_id, "hi")
    monitor.append_reply(entry_id, " there")
    monitor.finish(entry_id)

    row = monitor.snapshot()[0]
    assert row["decode_ms"] is None
    assert row["duration_ms"] is not None


@pytest.mark.parametrize(
    # "1000" is absent on purpose: _finite_float_or_none coerces a numeric string, the
    # same as it already does for tok_per_sec and prompt_ms.
    "predicted_ms",
    [float("inf"), float("nan"), -1, "abc", None, 1e308, 10**400, {}, []],
)
def test_a_bad_predicted_ms_is_dropped_rather_than_raising(monkeypatch, predicted_ms):
    """json.loads accepts a bare Infinity, and this runs inside streaming generators
    where a raise truncates the user's response."""
    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
    )
    inference_route._monitor_usage(
        entry_id,
        {"completion_tokens": 50},
        None,
        timings = {"predicted_ms": predicted_ms},
    )
    monitor.finish(entry_id)

    assert monitor.snapshot()[0]["decode_ms"] is None
