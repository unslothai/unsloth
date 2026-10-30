# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.inference.tools import RAG_SOURCES_SENTINEL
from storage import research_runs_db as research_db
from storage import studio_db

STEPS = 12
REPORT = "## Findings\n\nThe manual covers it [Document: manual.pdf, p. 1]."


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "how does the pump work?"}],
            "createdAt": 2,
        }
    )
    return tmp_path


def _web_results(step: int) -> str:
    return "\n\n---\n\n".join(
        f"Title: Result {step}-{rank}\nURL: https://example.test/{step}/{rank}\n"
        f"Snippet: web-{step}-{rank} " + "w" * 400
        for rank in range(5)
    )


def _passage(index: int) -> str:
    return "d" * 800 + f" manual-fact-{index} " + "d" * 1_200


def _knowledge_base_results(step: int) -> str:
    sources = [
        {
            "chunkId": f"chunk-{index}",
            "documentId": "doc-1",
            "filename": "manual.pdf",
            "page": index + 1,
            "text": _passage(index),
        }
        for index in range(step, step + 5)
    ]
    text = "\n\n".join(source["text"] for source in sources)
    return text + RAG_SOURCES_SENTINEL + json.dumps(sources)


def _synthesis_prompt(
    monkeypatch,
    knowledge_base,
    context_length = None,
    query = "pump",
) -> str:
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": {"thread_id": "thread-1"},
            "instructions": "",
            "question": "how does the pump work?",
            "budgets": {
                "maxSteps": STEPS,
                "maxSources": 100,
                "modelTimeoutSeconds": 900,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan(
        "run-1",
        {
            "title": "Pump",
            "steps": [
                {"title": f"{query} {step}", "query": f"{query} {step}"} for step in range(STEPS)
            ],
        },
    )
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    searches = {"web_search": 0, "search_knowledge_base": 0}

    def fake_execute_tool(name, arguments, **kwargs):
        step = searches[name]
        searches[name] += 1
        return _web_results(step) if name == "web_search" else knowledge_base(step)

    prompts = []

    async def fake_stream_completion(run, messages, **kwargs):
        if kwargs.get("phase") == "synthesis":
            prompts.append(messages[1]["content"])
            return REPORT, "", "stop", None
        return "not json", "", "stop", None

    monkeypatch.setattr(worker, "_loaded_context_length", lambda _inf = None: context_length)
    monkeypatch.setattr(worker, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(research_db.claim_next(supervisor.worker_id)))
    assert research_db.get_run("run-1")["status"] == "completed"
    return prompts[-1]


def test_every_step_keeps_its_top_document_passage_in_the_report_prompt(research_home, monkeypatch):
    prompt = _synthesis_prompt(monkeypatch, _knowledge_base_results)

    assert all(f"manual-fact-{step}" in prompt for step in range(STEPS))
    assert all(f"web-{step}-0" in prompt for step in range(STEPS))


def test_web_results_keep_their_room_when_the_documents_match_nothing(research_home, monkeypatch):
    prompt = _synthesis_prompt(
        monkeypatch, lambda _step: "No matching chunks were found in the knowledge base."
    )

    assert all(f"web-{step}-{rank}" in prompt for step in range(STEPS) for rank in range(5))


def test_small_context_keeps_web_results_when_the_documents_match_nothing(
    research_home, monkeypatch
):
    prompt = _synthesis_prompt(
        monkeypatch,
        lambda _step: "No matching chunks were found in the knowledge base.",
        context_length = 8_192,
        query = "centrifugal pump cooling data sheet maintenance interval seal replacement guide",
    )

    assert all(f"web-{step}-0" in prompt for step in range(STEPS))
