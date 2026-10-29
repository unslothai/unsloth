"""Each caller profile reads only about the tools it was given, at a bounded cost per message."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from mcp_server import create_studio_mcp, surface, withheld
from studio_mcp import surface as text_surface
from studio_mcp.profile import AGENT, CHAT, CallerProfile, OUTLIVES_CALL

# What the chat model reads on every message, in characters. 52,513 before profiles, 47,499 with
# them; lower it when the surface shrinks, never raise it to fit.
CHAT_BUDGET_CHARS = 48_000


def _texts(profile):
    tools = asyncio.run(create_studio_mcp(profile).list_tools())
    return {t.name: t.to_mcp_tool() for t in tools}


def test_chat_withholds_each_tool_by_its_facts_and_never_names_one():
    absent = set(withheld(CHAT))
    assert absent == {"load_model", "unload_model", "plan_load", "chat", "run_recipe"} | {
        "start_training",
        "export_model",
        "cancel",
    }
    tools = _texts(CHAT)
    assert not absent & set(tools)
    # Called like a tool or named in prose; `cancel_id` and "cancel" as a verb are not the tool.
    names = "|".join(sorted(re.escape(n) for n in absent - {"chat", "cancel"}))
    named = re.compile(rf"\b({names})(\(|\b(?![_ ]?(id|ids)\b))|\b(chat|cancel)\(")
    for name, t in tools.items():
        text = json.dumps([t.description, t.inputSchema, t.outputSchema])
        assert not named.search(text) and "<<" not in text, name
        if "name one in ``model``" in (t.description or ""):
            assert "model" in t.inputSchema.get("properties", {}), name
    size = sum(
        len(json.dumps({"n": n, "d": t.description, "p": t.inputSchema})) for n, t in tools.items()
    )
    assert size <= CHAT_BUDGET_CHARS, size


def test_agent_text_still_points_at_load_model():
    tools = _texts(AGENT)
    assert 'load_model(kind="image")' in tools["generate_image"].description
    for name, t in tools.items():
        assert "<<" not in json.dumps([t.description, t.inputSchema, t.outputSchema]), name


def test_a_profile_withholds_by_the_facts_tools_declare():
    narrow = CallerProfile("narrow", refused = frozenset({OUTLIVES_CALL}))
    assert surface(narrow) == surface(AGENT) - {"start_training", "export_model"}


@pytest.mark.parametrize(
    "text, tools, expected",
    [
        ("Load <<load_model: with it | on the Image page>>.", {"load_model"}, "Load with it."),
        ("Load <<load_model: with it | on the Image page>>.", set(), "Load on the Image page."),
        ("Done.<<cancel: Stop it with cancel(kind='video').>>", set(), "Done."),
    ],
)
def test_render(text, tools, expected):
    assert text_surface.render(text, tools) == expected
