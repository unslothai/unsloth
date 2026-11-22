# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


LINKED = "https://example.org/blog/2026/new-release"
PAGE = "The new release ships a quantized 4-bit runtime called Lantern."
SEARCH = "Title: Similar tools\nURL: https://other.test/tools\nSnippet: A list of similar tools."
QUESTION = f"Research {LINKED}. and compare it with similar tools"


@pytest.fixture
def question():
    return QUESTION


@pytest.fixture
def research_home(tmp_path, monkeypatch, question):
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
            "content": [{"type": "text", "text": question}],
            "createdAt": 2,
        }
    )
    return tmp_path


def _run(
    monkeypatch,
    website_policy: dict | None = None,
    page: str = PAGE,
    question: str = QUESTION,
    max_sources: int = 5,
):
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
            "ragScope": None,
            "instructions": "",
            "question": question,
            "websitePolicy": website_policy,
            "budgets": {
                "maxSteps": 2,
                "maxSources": max_sources,
                "modelTimeoutSeconds": 900,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan(
        "run-1",
        {"title": "Plan", "steps": [{"title": "Similar", "query": "similar tools"}]},
    )
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    claimed = research_db.claim_next(supervisor.worker_id)
    calls: list[dict] = []
    synthesis_prompts: list[str] = []

    def fake_execute_tool(name, arguments, **kwargs):
        calls.append(arguments)
        return page if "url" in arguments else SEARCH

    async def fake_stream_completion(run, messages, **kwargs):
        if kwargs.get("phase") == "synthesis":
            synthesis_prompts.append(messages[-1]["content"])
            return "## Findings\n\nLantern is new.", "", "stop", None
        return "not json", "", "stop", None

    monkeypatch.setattr(worker, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(claimed))
    return research_db.get_run("run-1"), calls, synthesis_prompts


def test_the_page_linked_in_the_question_is_read_and_citable(research_home, monkeypatch):
    finished, calls, synthesis_prompts = _run(monkeypatch)

    assert calls[0] == {"url": LINKED}
    assert {"query": "similar tools"} in calls
    assert finished["status"] == "completed"
    assert LINKED in [source["url"] for source in finished["sources"]]
    assert any(PAGE in prompt and LINKED in prompt for prompt in synthesis_prompts)


def test_a_linked_page_the_website_policy_blocks_is_not_read(research_home, monkeypatch):
    finished, calls, _ = _run(monkeypatch, {"blockedDomains": ["example.org"]})

    assert {"url": LINKED} not in calls
    assert LINKED not in [source["url"] for source in finished["sources"]]


def test_a_linked_page_that_fails_to_load_is_not_cited(research_home, monkeypatch):
    finished, calls, _ = _run(monkeypatch, page = "Error: page returned HTTP 404")

    assert calls[0] == {"url": LINKED}
    assert LINKED not in [source["url"] for source in finished["sources"]]


@pytest.mark.parametrize(
    "question",
    [
        f'Research "{LINKED}" and compare it',
        f"Research `{LINKED}` and compare it",
        f"Research '{LINKED}' and compare it",
        f"Research **{LINKED}** and compare it",
        f"Research [{LINKED}] and compare it",
        f"研究{LINKED}，并比较",
        f"Research **[the release]({LINKED})** and compare it",
        f"Research “{LINKED}” and compare it",
        f"Research {LINKED.replace('https', 'HTTPS')} and compare it",
    ],
)
def test_a_quoted_link_is_read_without_its_delimiter(research_home, monkeypatch, question):
    _, calls, _ = _run(monkeypatch, question = question)

    assert calls[0] == {"url": LINKED}


@pytest.mark.parametrize(
    "page",
    [
        "(page returned no readable text)",
        "(non-text content: image/png, 2048 bytes; not readable as text)",
        "(binary content, 2048 bytes; not readable as text)",
        "(PDF contains no extractable text)",
        "(PDF contains no extractable text in the first 40 pages)",
        "(PDF content could not be read as text)",
    ],
)
def test_a_linked_page_without_readable_text_is_not_cited(research_home, monkeypatch, page):
    finished, calls, _ = _run(monkeypatch, page = page)

    assert calls[0] == {"url": LINKED}
    assert LINKED not in [source["url"] for source in finished["sources"]]


@pytest.mark.parametrize("question", [f"Research {LINKED}_ and compare it"])
def test_a_trailing_underscore_in_a_bare_link_is_kept(research_home, monkeypatch, question):
    _, calls, _ = _run(monkeypatch, question = question)

    assert calls[0] == {"url": f"{LINKED}_"}


SECOND = "https://example.net/second"


@pytest.mark.parametrize("question", [f"Compare {LINKED} with {SECOND}"])
def test_linked_pages_past_the_source_cap_are_not_fetched(research_home, monkeypatch, question):
    finished, calls, _ = _run(monkeypatch, question = question, max_sources = 1)

    assert calls[0] == {"url": LINKED}
    assert {"url": SECOND} not in calls
    assert [source["url"] for source in finished["sources"]] == [LINKED]


@pytest.mark.parametrize("question", [f"Research [the search]({LINKED}?q=why?) and compare it"])
def test_a_markdown_link_destination_keeps_its_trailing_punctuation(
    research_home, monkeypatch, question
):
    _, calls, _ = _run(monkeypatch, question = question)

    assert calls[0] == {"url": f"{LINKED}?q=why?"}
