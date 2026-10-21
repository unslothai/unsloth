# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core import research_runs as worker
from storage import research_runs_db as research_db
from storage import studio_db
from .test_research_synthesis_recovery import _claimed_run, research_home


REPORT = "## Findings\n\n" + "The evidence explains the result. " * 80
RAW = f"Private preamble.\n{worker._REPORT_BOUNDARY_MARKER}\n{REPORT}"


def _interrupt(monkeypatch, text, error, channel, final_reason):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(text), 300):
                chunk = {"choices": [{"delta": {channel: text[offset : offset + 300]}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            if final_reason is not None:
                done = {"choices": [{"delta": {}, "finish_reason": final_reason}]}
                yield f"data: {json.dumps(done)}\n\n".encode()
            raise error

    def respond(request):
        # Only the report call is interrupted; the steps before it have to land, and their
        # unparseable bodies make each one take its single seed action.
        if b"untrusted_synthesis_audit_json" in request.content:
            return httpx.Response(200, stream = Stream())
        return httpx.Response(200, content = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')

    original = httpx.AsyncClient
    monkeypatch.setattr(
        worker.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport = httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(
        worker.auth_storage, "create_api_key", lambda **kwargs: ("fixture", {"id": 1})
    )
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", lambda key_id: None)

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = _claimed_run(supervisor)
    asyncio.run(supervisor._process(run))
    return research_db.get_run(run["id"])


@pytest.mark.parametrize(
    ("error", "channel", "final_reason", "lead"),
    [
        (httpx.ReadError("connection reset"), "content", None, "Incomplete report."),
        (worker.ModelWallClockTimeout("timed out"), "content", None, "Incomplete report."),
        # Some local models write the report only as reasoning, which synthesis accepts.
        (httpx.ReadError("connection reset"), "reasoning_content", None, "Incomplete report."),
        # Cut after the model said it was done: the report is whole, only delivery was not.
        (httpx.ReadError("connection reset"), "content", "stop", "Report complete."),
    ],
)
def test_interrupted_stream_retains_last_bytes_in_failed_report_and_chat(
    research_home, monkeypatch, error, channel, final_reason, lead
):
    run = _interrupt(monkeypatch, RAW, error, channel, final_reason)
    assert run["status"] == "failed"
    assert run["error"] == worker._safe_error(error)
    notice = f"> **{lead}** Research failed: `{run['error']}`"
    assert run["report"] == f"{notice}\n\n{REPORT.strip()}"
    message = studio_db.get_chat_message(run["threadId"], run["assistantMessageId"])
    text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
    assert text == run["report"]
    assert "Private preamble" not in text
    terminal = research_db.list_events(run["id"], 0)[-1]
    assert terminal["type"] == "run.failed"
    assert terminal["data"]["report"] == run["report"]


@pytest.mark.parametrize(
    "text", ["", REPORT, f"```\n{RAW}\n```", f"{worker._REPORT_BOUNDARY_MARKER}\n"]
)
def test_unidentified_report_is_not_exposed(research_home, monkeypatch, text):
    run = _interrupt(monkeypatch, text, httpx.ReadError("connection reset"), "content", None)
    assert run["status"] == "failed"
    assert run["report"] is None


def test_a_returned_report_is_kept_before_the_checks_that_could_lose_it(research_home, monkeypatch):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    claimed = _claimed_run(supervisor)

    async def gone(run_id):
        raise RuntimeError("database is gone")

    async def streamed(run, messages, **kwargs):
        if kwargs.get("phase") != "synthesis":
            return "not json", "", "stop", None
        monkeypatch.setattr(supervisor, "_check_active", gone)
        return RAW, "", "stop", None

    monkeypatch.setattr(supervisor, "_stream_completion", streamed)
    asyncio.run(supervisor._process(claimed))

    # Work remains between a returned report and the row storing it; none of it can lose the report.
    assert research_db.get_run(claimed["id"])["report"] == (
        f"> **Report complete.** Research failed: `database is gone`\n\n{REPORT.strip()}"
    )


def test_interrupted_report_validates_citations(research_home, monkeypatch):
    raw = RAW + "\n[Backed](https://backed.test) [invented](https://invented.test)"
    raw += "\n\n## Sources\n\nUntrusted source list"
    monkeypatch.setattr(
        worker,
        "execute_tool",
        lambda *args, **kwargs: "Title: Backed\nURL: https://backed.test\nSnippet: evidence",
    )
    run = _interrupt(monkeypatch, raw, httpx.ReadError("connection reset"), "content", None)
    assert "https://backed.test" in run["report"]
    assert "invented.test" not in run["report"]
    assert "Untrusted source list" not in run["report"]


@pytest.mark.parametrize("error", [worker.RunCancelled(), worker.LeaseLost()])
def test_control_flow_interruptions_do_not_publish_partial_report(
    research_home, monkeypatch, error
):
    run = _interrupt(monkeypatch, RAW, error, "content", None)
    assert run["report"] is None


def test_failed_finish_does_not_promote_raw_progress(research_home):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = _claimed_run(supervisor)
    research_db.set_report_progress(run["id"], RAW, worker_id = supervisor.worker_id)
    assert research_db.finish(run["id"], supervisor.worker_id, "failed", "failed") == "failed"
    assert research_db.get_run(run["id"])["report"] is None
