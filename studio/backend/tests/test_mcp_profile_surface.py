"""Each caller profile reads only about the tools it was given, at a bounded cost per message."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from mcp_server import create_studio_mcp, surface, withheld
from studio_mcp import surface as text_surface
from studio_mcp.profile import AGENT, CHAT, CallerProfile, OUTLIVES_CALL


def _listed(profile):
    return asyncio.run(create_studio_mcp(profile).list_tools())


def _mentions(names):
    # Called like a tool, or named as one in prose; `cancel_id` and "cancel" as a verb are not the tool.
    return re.compile(
        r"\b("
        + "|".join(sorted(re.escape(name) for name in names))
        + r")(\(|\b(?![_ ]?(id|ids)\b))"
    )


def _mcp_tool_text(tool) -> str:
    mcp = tool.to_mcp_tool()
    return json.dumps({"d": mcp.description, "p": mcp.inputSchema, "o": mcp.outputSchema})


def test_chat_text_never_names_a_tool_the_chat_was_not_given():
    absent = [name for name in withheld(CHAT) if name not in {"chat", "cancel"}]
    pattern = _mentions(absent)
    calls = re.compile(r"\b(chat|cancel)\(")
    found = {
        tool.name: sorted({m.group(0) for m in pattern.finditer(_mcp_tool_text(tool))})
        + sorted({m.group(0) for m in calls.finditer(_mcp_tool_text(tool))})
        for tool in _listed(CHAT)
    }
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_agent_text_still_points_at_load_model():
    text = {tool.name: _mcp_tool_text(tool) for tool in _listed(AGENT)}
    assert 'load_model(kind=\\"image\\")' in text["generate_image"]
    assert "<<" not in "".join(text.values())


def test_no_markup_reaches_any_caller():
    for profile in (AGENT, CHAT):
        assert "<<" not in "".join(_mcp_tool_text(tool) for tool in _listed(profile))


def test_a_profile_withholds_by_the_facts_tools_declare():
    narrow = CallerProfile("narrow", refused = frozenset({OUTLIVES_CALL}))
    assert set(withheld(narrow)) == {"start_training", "export_model"}
    assert surface(narrow) == surface(AGENT) - {"start_training", "export_model"}


@pytest.mark.parametrize(
    "text, tools, expected",
    [
        (
            "Load one <<load_model: with load_model | on the Image page>> first.",
            {"load_model"},
            "Load one with load_model first.",
        ),
        (
            "Load one <<load_model: with load_model | on the Image page>> first.",
            set(),
            "Load one on the Image page first.",
        ),
        ("Done.<<cancel: Stop it with cancel(kind='video').>>", set(), "Done."),
        ("Plain text stays as is.", set(), "Plain text stays as is."),
    ],
)
def test_render(text, tools, expected):
    assert text_surface.render(text, tools) == expected


# What the chat model reads on every message (name, description, input schema), in characters.
# 52,513 before profiles, 47,499 with them; lower it when the surface shrinks, never raise it to fit.
CHAT_BUDGET_CHARS = 48_000


def test_the_chat_surface_stays_within_its_budget():
    size = sum(
        len(
            json.dumps(
                {"n": t.name, "d": t.to_mcp_tool().description, "p": t.to_mcp_tool().inputSchema}
            )
        )
        for t in _listed(CHAT)
    )
    assert size <= CHAT_BUDGET_CHARS, size


def test_chat_text_only_offers_model_on_tools_that_take_it():
    for tool in _listed(CHAT):
        mcp = tool.to_mcp_tool()
        if "name one in ``model``" in (mcp.description or ""):
            assert "model" in mcp.inputSchema.get("properties", {}), tool.name
