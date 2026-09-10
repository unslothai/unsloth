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


def _transport(monkeypatch, text, error):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(text), 300):
                chunk = {"choices": [{"delta": {"content": text[offset : offset + 300]}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            raise error

    original = httpx.AsyncClient
    monkeypatch.setattr(
        worker.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport = httpx.MockTransport(lambda request: httpx.Response(200, stream = Stream())),
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        worker.auth_storage, "create_api_key", lambda **kwargs: ("fixture", {"id": 1})
    )
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", lambda key_id: None)


def _interrupt(monkeypatch, text, error):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = _claimed_run(supervisor)
    _transport(monkeypatch, text, error)

    async def research(run):
        await supervisor._stream_completion(
            run, [{"role": "user", "content": "Report"}], phase = "synthesis"
        )

    monkeypatch.setattr(supervisor, "_research", research)
    asyncio.run(supervisor._process(run))
    return research_db.get_run(run["id"])


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("connection reset"),
        worker.ModelOutputIdleTimeout("Model stopped producing output"),
        worker.ModelWallClockTimeout("Model exceeded its time budget"),
    ],
)
def test_interrupted_stream_retains_last_bytes_in_failed_report_and_chat(
    research_home, monkeypatch, error
):
    run = _interrupt(monkeypatch, RAW, error)
    assert run["status"] == "failed"
    assert run["report"] == REPORT.strip()
    assert run["error"] == worker._safe_error(error)
    message = studio_db.get_chat_message(run["threadId"], run["assistantMessageId"])
    text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
    assert "Incomplete report" in text and REPORT.strip() in text
    assert "Private preamble" not in text
    terminal = research_db.list_events(run["id"], 0)[-1]
    assert terminal["type"] == "run.failed"
    assert terminal["data"]["report"] == REPORT.strip()


@pytest.mark.parametrize(
    "text", ["", REPORT, f"```\n{RAW}\n```", f"{worker._REPORT_BOUNDARY_MARKER}\n"]
)
def test_unidentified_report_is_not_exposed(research_home, monkeypatch, text):
    run = _interrupt(monkeypatch, text, httpx.ReadError("connection reset"))
    assert run["status"] == "failed"
    assert run["report"] is None


def test_interrupted_report_validates_citations(research_home, monkeypatch):
    raw = RAW + "\n[invented](https://invented.test)\n\n## Sources\n\nUntrusted source list"
    run = _interrupt(monkeypatch, raw, httpx.ReadError("connection reset"))
    assert "invented.test" not in run["report"]
    assert "Untrusted source list" not in run["report"]


@pytest.mark.parametrize("error", [worker.RunCancelled(), worker.LeaseLost()])
def test_control_flow_interruptions_do_not_publish_partial_report(
    research_home, monkeypatch, error
):
    run = _interrupt(monkeypatch, RAW, error)
    assert run["report"] is None


def test_failed_finish_does_not_promote_raw_progress(research_home):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = _claimed_run(supervisor)
    research_db.set_report_progress(run["id"], RAW, worker_id = supervisor.worker_id)
    assert research_db.finish(run["id"], supervisor.worker_id, "failed", "failed") == "failed"
    assert research_db.get_run(run["id"])["report"] is None
