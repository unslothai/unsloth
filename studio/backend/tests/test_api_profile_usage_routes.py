# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route attribution contract for durable Profile API usage receipts."""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

import routes.inference as inference_route


def _monitor_start_keywords(function) -> list[set[str]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Name)
            and owner.id == "api_monitor"
            and node.func.attr == "start"
        ):
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
    assert "_api_monitor.set_terminal_callback(_record_api_usage)" in source
    assert "_api_monitor.set_terminal_callback(None)" in source
