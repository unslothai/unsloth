# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``run_tools_locally`` resolves the one request shape names cannot.

On a provider shipping hosted builtins of the same name, ``enabled_tools:
["web_search"]`` is byte-identical whether an older bundle means the PROVIDER's
search or the current composer means the local Search pill. The backend guessed
hosted, so the newer client silently got hosted search. The flag lets the caller
say; absent, the hosted reading still wins and old clients are unaffected.
"""

from __future__ import annotations

import asyncio

import pytest


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


HOSTED_PROVIDERS = ("openai", "gemini", "kimi", "openrouter")
SELF_HOSTED_PROVIDERS = ("llama_cpp", "vllm", "ollama", "custom")


def _payload(**overrides):
    from models.inference import ChatCompletionRequest

    base = dict(
        messages = [{"role": "user", "content": "what is 2+2?"}],
        provider_id = "saved-1",
        external_model = "gpt-5.4",
        stream = True,
        enable_tools = True,
    )
    base.update(overrides)
    return ChatCompletionRequest(**base)


# ── the ambiguous shape, both readings ───────────────────────────────


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_search_only_without_the_flag_stays_hosted(provider_type):
    """An old cached bundle sends exactly this and means the provider's search."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["web_search"])
    assert payload.run_tools_locally is None, "the old client does not send it"
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_search_only_with_the_flag_runs_here(provider_type):
    """The current composer sends the same names plus the flag, and means us."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["web_search"], run_tools_locally = True)
    assert _selects_only_provider_hosted_tools(payload, provider_type) is False


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_an_explicit_false_is_not_read_as_a_local_request(provider_type):
    """Only True flips it. False and None both keep the hosted reading."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["web_search"], run_tools_locally = False)
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


# ── everything the flag must not change ──────────────────────────────


@pytest.mark.parametrize("provider_type", SELF_HOSTED_PROVIDERS)
def test_self_hosted_providers_are_unaffected(provider_type):
    """They declare no hosted tools, so this was never ambiguous for them."""
    from routes.inference import _selects_only_provider_hosted_tools
    for flag in (None, True, False):
        payload = _payload(enabled_tools = ["web_search"], run_tools_locally = flag)
        assert _selects_only_provider_hosted_tools(payload, provider_type) is False


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_a_local_only_name_never_needed_the_flag(provider_type):
    """python has no hosted counterpart, so it was always unambiguous."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["python"])
    assert _selects_only_provider_hosted_tools(payload, provider_type) is False


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_a_hosted_only_name_with_no_local_stand_in_stays_hosted(provider_type):
    """image_generation has no local implementation, so a local reading would
    drop it: the flag must not blanket-override the hosted vocabulary."""
    from core.inference.tools import ALL_TOOLS
    from routes.inference import _selects_only_provider_hosted_tools

    # The premise: Unsloth has nothing to substitute.
    assert "image_generation" not in {tool["function"]["name"] for tool in ALL_TOOLS}

    payload = _payload(enabled_tools = ["image_generation"])
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_the_flag_cannot_route_a_hosted_only_selection_into_the_loop(provider_type):
    """The flag decides ambiguous names, it does not override the whole rule.

    Honouring it for image_generation would enter the loop, find nothing to
    execute, fall back to the same passthrough, and skip the confirmation
    rejection on the way, since that guard keys on not having taken the loop.
    """
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(
        enabled_tools = ["image_generation"],
        run_tools_locally = True,
        confirm_tool_calls = True,
    )
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_code_execution_alone_stays_hosted_because_its_stand_ins_are_unselected(provider_type):
    """code_execution has a stand-in mapping, but this request selects neither half.

    Unsloth ships no code_execution, so the loop has nothing to execute. Reading
    the flag off the mere existence of the mapping enters the loop, finds an
    empty catalog, falls back to the same passthrough, and skips the
    confirmation rejection on the way.
    """
    from core.inference.providers import LOCAL_STANDINS_FOR_HOSTED_TOOLS
    from core.inference.tools import ALL_TOOLS
    from routes.inference import _select_request_tools, _selects_only_provider_hosted_tools

    # The premise, derived rather than asserted: a mapping exists, nothing it
    # points at was selected, and Unsloth has no code_execution of its own.
    assert LOCAL_STANDINS_FOR_HOSTED_TOOLS["code_execution"] == frozenset({"python", "terminal"})
    assert "code_execution" not in {tool["function"]["name"] for tool in ALL_TOOLS}

    payload = _payload(
        enabled_tools = ["code_execution"],
        run_tools_locally = True,
        confirm_tool_calls = True,
    )
    # The local catalog this request would take into the loop: empty.
    assert _drive(_select_request_tools(payload, tools_on = True, mcp_allowed = False)) == []
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_code_execution_beside_a_selected_stand_in_still_takes_the_loop(provider_type):
    """The other half of the rule: web_search is Unsloth's to run and
    code_execution rides along, so the check above must not sweep this up."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(
        enabled_tools = ["web_search", "code_execution"],
        run_tools_locally = True,
    )
    assert _selects_only_provider_hosted_tools(payload, provider_type) is False


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_a_mixed_selection_with_the_flag_still_takes_the_loop(provider_type):
    """One name Unsloth can run is enough; the hosted one rides along as a flag."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(
        enabled_tools = ["web_search", "image_generation"],
        run_tools_locally = True,
    )
    assert _selects_only_provider_hosted_tools(payload, provider_type) is False


def test_mcp_still_wins_regardless_of_the_flag():
    """MCP is unambiguous on its own and predates the flag."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["web_search"], mcp_enabled = True)
    assert _selects_only_provider_hosted_tools(payload, "openai") is False


def test_the_field_defaults_to_absent_not_false():
    """Same decision as False, but the wire must still distinguish "did not know
    about this" from "asked for hosted"."""
    payload = _payload(enabled_tools = ["web_search"])
    dumped = payload.model_dump(exclude_unset = True)
    assert "run_tools_locally" not in dumped


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_armed_research_is_never_a_purely_hosted_selection(provider_type):
    """deep_research is Unsloth's own and rides past enabled_tools, so it is not in the names.

    Read by name alone, an armed turn with only hosted pills lit looks like a hosted request,
    the turn proxies through, the tool is never offered and arming Deep Research does nothing.
    """
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enabled_tools = ["web_search"], deep_research_armed = True)
    assert _selects_only_provider_hosted_tools(payload, provider_type) is False


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_an_unarmed_hosted_selection_is_unchanged(provider_type):
    from routes.inference import _selects_only_provider_hosted_tools
    payload = _payload(enabled_tools = ["web_search"], deep_research_armed = False)
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


def test_local_gguf_search_only_is_not_a_hosted_request():
    """#9730: a loaded GGUF has no provider_type; web_search is Unsloth's tool."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _select_request_tools, _selects_only_provider_hosted_tools

    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "Search the web for the Linux kernel version."}],
        enable_tools = True,
        enabled_tools = ["web_search"],
        permission_mode = "off",
        tool_choice = {"type": "function", "function": {"name": "web_search"}},
    )
    assert payload.provider_type is None
    assert payload.provider_id is None
    assert _selects_only_provider_hosted_tools(payload, None) is False
    names = [
        t["function"]["name"]
        for t in _drive(_select_request_tools(payload, tools_on = True, mcp_allowed = False))
    ]
    assert names == ["web_search"]
