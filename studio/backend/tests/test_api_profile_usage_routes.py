# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route attribution contract for durable Profile API usage receipts."""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route
import routes.profile_stats as profile_stats_route


def _monitor_start_keywords(function) -> list[set[str]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "api_monitor" and node.func.attr == "start":
            calls.append({keyword.arg for keyword in node.keywords if keyword.arg is not None})
    return calls


@pytest.mark.parametrize(
    "function",
    [
        inference_route.openai_chat_completions,
        inference_route._responses_non_streaming,
        inference_route.openai_responses,
        inference_route.anthropic_messages,
        inference_route.openai_completions,
        inference_route.openai_embeddings,
    ],
)
def test_every_profile_tracked_route_preserves_external_identity(function):
    starts = _monitor_start_keywords(function)
    assert starts, f"{function.__name__} must create a monitor request row"
    for keywords in starts:
        assert "via_api_key" in keywords
        assert "subject" in keywords


def test_production_lifespan_installs_and_removes_the_usage_sink():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding = "utf-8")
    assert "_api_monitor.acquire_terminal_callback(_enqueue_api_usage)" in source
    assert "_api_monitor.release_terminal_callback(_api_usage_callback_lease)" in source
    assert "await asyncio.to_thread(_release_api_usage_writer" in source


def test_profile_endpoint_forwards_each_authenticated_subject(monkeypatch):
    seen: list[str] = []

    def fake_stats(**kwargs):
        seen.append(kwargs["subject"])
        return {"subject": kwargs["subject"]}

    monkeypatch.setattr(profile_stats_route, "compute_profile_stats", fake_stats)
    active = {"subject": "alice"}
    app = FastAPI()
    app.include_router(profile_stats_route.router, prefix = "/api/profile")
    app.dependency_overrides = {
        get_current_subject: lambda: active["subject"],
    }
    client = TestClient(app)

    assert client.get("/api/profile/stats").json() == {"subject": "alice"}
    active["subject"] = "bob"
    assert client.get("/api/profile/stats").json() == {"subject": "bob"}
    assert seen == ["alice", "bob"]
