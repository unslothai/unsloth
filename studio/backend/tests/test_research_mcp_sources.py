# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.inference import mcp_client
from storage import mcp_servers_db
from storage import research_runs_db as research_db
from storage import studio_db

PLAN = {"title": "Zephyr", "steps": [{"title": "Launch", "query": "zephyr beta launch"}]}


def _tool(name, *required):
    return {
        "name": name,
        "description": f"{name} tool",
        "inputSchema": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in required},
            "required": list(required),
        },
    }


@pytest.fixture
def notes_server(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "R",
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
            "content": [{"type": "text", "text": "When did Zephyr launch?"}],
            "createdAt": 2,
        }
    )
    mcp_servers_db.create_server(
        id = "notes", display_name = "Project Notes", url = "https://notes.example/mcp"
    )
    mcp_client.cache_tools(
        "notes",
        [
            _tool("search_notes", "query"),
            _tool("search_papers", "query"),
            _tool("delete_note", "note_id"),
            _tool("lookup_pair", "left", "right"),
            _tool("get_secret_token", "query"),
            _tool("schedule_meeting", "query"),
            _tool("complete_task", "query"),
        ],
    )
    yield
    mcp_client.invalidate_tool_cache("notes")


def test_research_lists_only_read_only_single_query_mcp_tools(notes_server):
    from routes.mcp_servers import list_research_search_tools
    tools = asyncio.run(list_research_search_tools(current_subject = "alice"))

    assert [(tool["serverName"], tool["tool"]) for tool in tools] == [
        ("Project Notes", "search_notes"),
        ("Project Notes", "search_papers"),
    ]


def test_credentialless_discovery_never_starts_stdio_servers(monkeypatch):
    from core.inference import tools as tools_mod

    servers = [
        {
            "id": "local",
            "display_name": "Local",
            "url": "python server.py",
            "is_enabled": True,
            "use_oauth": False,
        },
        {
            "id": "remote",
            "display_name": "Remote",
            "url": "https://notes.example/mcp",
            "is_enabled": True,
            "use_oauth": False,
        },
    ]
    calls = []

    async def fake_list_tools(url, **kwargs):
        calls.append(url)
        return [_tool("search_notes", "query")]

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_servers_db, "list_servers", lambda: servers)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(tools_mod, "list_tools_async", fake_list_tools)

    tools = asyncio.run(tools_mod.mcp_search_tools(include_stdio = False))

    assert calls == ["https://notes.example/mcp"]
    assert [tool["serverId"] for tool in tools] == ["remote"]


def test_run_discovery_only_probes_selected_servers(monkeypatch):
    from core.inference import tools as tools_mod

    servers = [
        {
            "id": server_id,
            "display_name": server_id.title(),
            "url": f"https://{server_id}.example/mcp",
            "is_enabled": True,
            "use_oauth": False,
        }
        for server_id in ("selected", "unused")
    ]
    calls = []

    async def fake_list_tools(url, **kwargs):
        calls.append(url)
        return [_tool("search_notes", "query")]

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_servers_db, "list_servers", lambda: servers)
    monkeypatch.setattr(tools_mod, "list_tools_async", fake_list_tools)

    tools = asyncio.run(tools_mod.mcp_search_tools(server_ids = {"selected"}))

    assert calls == ["https://selected.example/mcp"]
    assert [tool["serverId"] for tool in tools] == ["selected"]


def test_research_drops_mcp_servers_the_account_does_not_have(notes_server):
    from routes.research_runs import CreateResearchRun, _sanitize_config
    config = _sanitize_config(
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-1",
            inferenceRequest = {"model": "local-model"},
            mcpSources = [
                {"serverId": "someone-elses", "tool": "search_notes"},
                {"serverId": "notes", "tool": "search_notes"},
            ],
        ),
        {"modelId": "local-model"},
    )
    assert config["mcpSources"] == [{"serverId": "notes", "tool": "search_notes"}]


def test_mcp_evidence_keeps_content_from_every_source():
    from core.research_runs import _mcp_evidence

    sources = [
        {
            "kind": "mcp",
            "filename": f"Source {index}",
            "snippet": f"marker-{index} " + "x" * 4000,
        }
        for index in range(3)
    ]
    evidence = _mcp_evidence(sources)

    assert len(evidence) < 6500
    for index in range(3):
        assert f"marker-{index}" in evidence


def test_mcp_citation_normalizes_untrusted_source_labels():
    from core.research.citations import (
        _document_source_citation,
        _validate_report_document_sources,
    )

    source = {"kind": "mcp", "filename": "Project ]\n[ Notes · search"}
    citation = _document_source_citation(source)

    assert citation == "[MCP: Project Notes · search]"
    assert _validate_report_document_sources(f"Claim {citation}.", [source]) == (
        f"Claim {citation}."
    )
    assert _validate_report_document_sources(
        "Claim [Document: Project Notes · search].", [source]
    ) == ("Claim .")


def test_selected_mcp_tools_search_every_step_and_are_cited(notes_server, monkeypatch):
    from core import research_runs as worker
    from routes.research_runs import CreateResearchRun, _sanitize_config

    config = _sanitize_config(
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-1",
            inferenceRequest = {"model": "local-model"},
            mcpSources = [
                {"serverId": "notes", "tool": "search_notes"},
                {"serverId": "notes", "tool": "search_papers"},
                {"serverId": "notes", "tool": "delete_note"},
            ],
        ),
        {"modelId": "local-model"},
    )
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = config,
        created_at = 10,
    )
    plan = research_db.set_plan("run-1", PLAN)
    research_db.approve("run-1", plan["planRevision"], plan["planHash"])
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))

    decisions = iter(
        (
            json.dumps({"action": "search", "title": "Launch", "query": "zephyr beta launch"}),
            json.dumps({"action": "search", "title": "Languages", "query": "zephyr languages"}),
            json.dumps({"action": "finish", "title": "Enough"}),
        )
    )
    decision_prompts = []
    synthesis_prompts = []
    report = "# Zephyr\n\nThe beta shipped on 14 March 2031 [MCP: Project Notes · search_notes]."

    async def fake_stream_completion(run, messages, **kwargs):
        if "iterative research process" in messages[0]["content"]:
            decision_prompts.append(messages[1]["content"])
            return next(decisions), "", "stop", None
        synthesis_prompts.append(messages[1]["content"])
        research_db.set_report_progress(run["id"], report)
        return report, "", "stop", None

    calls = []

    def fake_tool(name, arguments, *args, **kwargs):
        calls.append((name, arguments))
        if name == "web_search":
            return "No results found."
        if name == "mcp__notes__search_papers":
            return "Error: MCP tool 'search_papers' timed out after 120s"
        return f"Zephyr notes for {arguments['query']}: the beta shipped on 14 March 2031."

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    monkeypatch.setattr(worker, "execute_tool", fake_tool)
    monkeypatch.setattr(worker, "execute_mcp_tool", fake_tool)
    asyncio.run(supervisor._process(research_db.claim_next(supervisor.worker_id)))

    completed = research_db.get_run("run-1")
    assert completed["status"] == "completed"
    assert [step["status"] for step in completed["steps"]] == ["completed", "completed"]
    assert ("mcp__notes__search_notes", {"query": "zephyr beta launch"}) in calls
    assert ("mcp__notes__search_notes", {"query": "zephyr languages"}) in calls
    assert ("mcp__notes__search_papers", {"query": "zephyr languages"}) in calls
    assert not [name for name, _ in calls if name == "mcp__notes__delete_note"]
    assert [(s["kind"], s["filename"]) for s in completed["documentSources"]] == [
        ("mcp", "Project Notes · search_notes"),
        ("mcp", "Project Notes · search_notes"),
    ]
    assert "Citation: [MCP: Project Notes · search_notes]" in synthesis_prompts[-1]
    assert "14 March 2031" in synthesis_prompts[-1]
    assert not any("14 March 2031" in prompt for prompt in decision_prompts)
    assert "[MCP: Project Notes · search_notes]" in completed["report"]


def test_resumed_run_keeps_mcp_sources_citable(notes_server, monkeypatch):
    from core import research_runs as worker

    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "mcpSources": [{"serverId": "notes", "tool": "search_notes"}],
            "budgets": {
                "maxSteps": 1,
                "maxSources": 15,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
        created_at = 10,
    )
    plan = research_db.set_plan("run-1", PLAN)
    research_db.approve("run-1", plan["planRevision"], plan["planHash"])
    research_db.claim_next("old-worker")
    research_db.reset_execution_steps("run-1", "old-worker")
    source = {
        "kind": "mcp",
        "chunkId": "mcp__notes__search_notes:0",
        "documentId": "mcp__notes__search_notes",
        "filename": "Project Notes · search_notes",
        "snippet": "The beta shipped on 14 March 2031.",
    }
    research_db.upsert_document_source("run-1", 0, source, "old-worker")
    research_db.upsert_execution_step(
        "run-1",
        0,
        "Launch",
        "zephyr beta launch",
        "completed",
        {"action": "search", "input": "zephyr beta launch", "evidenceSources": [source]},
        "old-worker",
    )
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE research_runs SET lease_expires_at=0 WHERE id='run-1'")
        conn.commit()
    finally:
        conn.close()
    assert research_db.recover_expired() == 1
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    recovered = research_db.claim_next(supervisor.worker_id)
    synthesis_prompts = []
    report = "# Zephyr\n\nShipped 14 March 2031 [MCP: Project Notes · search_notes]."

    async def fake_stream_completion(run, messages, **kwargs):
        synthesis_prompts.append(messages[1]["content"])
        research_db.set_report_progress(run["id"], report)
        return report, "", "stop", None

    def unexpected_tool(*args, **kwargs):
        raise AssertionError("Recovered evidence should be synthesized without searching again")

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    monkeypatch.setattr(worker, "execute_tool", unexpected_tool)
    monkeypatch.setattr(worker, "execute_mcp_tool", unexpected_tool)
    asyncio.run(supervisor._process(recovered))

    completed = research_db.get_run("run-1")
    assert completed["status"] == "completed"
    assert [s["kind"] for s in completed["documentSources"]] == ["mcp"]
    assert "Citation: [MCP: Project Notes · search_notes]" in synthesis_prompts[-1]
    assert "14 March 2031" in synthesis_prompts[-1]
    assert "[MCP: Project Notes · search_notes]" in completed["report"]


def test_mcp_sources_only_accept_the_mcp_citation():
    from core.research.citations import _validate_report

    source = {"kind": "mcp", "filename": "Project Notes · search_notes"}
    report = "A [MCP: Project Notes · search_notes]. B [Document: Project Notes · search_notes]."

    assert _validate_report(report, [], [source]) == "A [MCP: Project Notes · search_notes]. B ."


def test_every_selected_mcp_tool_reaches_the_notes():
    from core.research_runs import _mcp_evidence

    sources = [
        {"kind": "mcp", "filename": f"Notes · search_{i}", "snippet": f"fact-{i} " + "x" * 3990}
        for i in range(3)
    ]

    evidence = _mcp_evidence(sources)
    assert all(f"fact-{i}" in evidence for i in range(3))
    assert len(evidence) < 6500


def test_research_tools_for_an_api_key_never_start_stdio_servers(notes_server, monkeypatch):
    from core.inference import tools
    from routes.mcp_servers import list_research_search_tools

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    mcp_servers_db.create_server(id = "local", display_name = "Local", url = "/bin/sh -c id")
    probed = []

    async def probe(**kwargs):
        probed.append(kwargs["url"])
        return []

    monkeypatch.setattr(tools, "list_tools_async", probe)
    listed = asyncio.run(list_research_search_tools(current_subject = "alice", via_api_key = True))

    assert probed == []
    assert [tool["tool"] for tool in listed] == ["search_notes", "search_papers"]


def test_research_ignores_mcp_sources_while_tools_are_disabled(notes_server):
    from routes.mcp_servers import list_research_search_tools
    from routes.research_runs import CreateResearchRun, _sanitize_config
    from state.tool_policy import tools_force_disabled

    with tools_force_disabled():
        listed = asyncio.run(list_research_search_tools(current_subject = "alice"))
        config = _sanitize_config(
            CreateResearchRun(
                threadId = "thread-1",
                userMessageId = "user-1",
                inferenceRequest = {"model": "local-model"},
                mcpSources = [{"serverId": "notes", "tool": "search_notes"}],
            ),
            {"modelId": "local-model"},
        )

    assert listed == []
    assert "mcpSources" not in config


def test_research_mcp_calls_cannot_reach_built_in_tools():
    from core.inference.tools import execute_mcp_tool
    assert execute_mcp_tool("terminal", {"command": "id"}).startswith("Error:")
