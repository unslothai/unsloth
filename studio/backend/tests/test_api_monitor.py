# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    monitor.append_reply(entry_id, "hi")
    monitor.set_perf(entry_id, tok_per_sec = 42.5, prompt_ms = 123.4, stop_reason = "length")
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["tok_per_sec"] == 42.5
    assert entry["ttft_ms"] == 123
    assert entry["stop_reason"] == "length"


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
