# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``run_tools_locally`` resolves the one request shape names cannot.

On a provider that ships hosted builtins of the same name, ``enable_tools: true``
with ``enabled_tools: ["web_search"]`` is ambiguous by construction. It is what a
bundle written before Studio ran tools for external providers sent to ask the
PROVIDER to search, and it is also what the current composer sends when the user
lights the Search pill, which the connections dialog says runs on this machine.
The bytes are identical, so the backend guessed hosted to protect the older
client, and the newer one silently got hosted search while being told otherwise.

The flag lets the caller say. Absent, the hosted reading still wins, so nothing
about an old client changes.
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
    """image_generation has no local implementation at all.

    Even with the flag set, reading this as a local request would drop it, so
    the flag must not be a blanket override of the hosted vocabulary.
    """
    from core.inference.tools import ALL_TOOLS
    from routes.inference import _selects_only_provider_hosted_tools

    # The premise: Studio has nothing to substitute, so a "local" reading of this
    # request would silently drop the tool rather than run something else.
    assert "image_generation" not in {tool["function"]["name"] for tool in ALL_TOOLS}

    payload = _payload(enabled_tools = ["image_generation"])
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_the_flag_cannot_route_a_hosted_only_selection_into_the_loop(provider_type):
    """The flag decides ambiguous names, it does not override the whole rule.

    There is no local loop for image_generation to run in, so honouring the flag
    here would enter the studio-tools path, find nothing to execute, fall back to
    the same passthrough, and skip the confirmation rejection on the way, since
    that guard keys on the request not having taken the loop. A caller asking for
    confirmation would get none.
    """
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(
        enabled_tools = ["image_generation"],
        run_tools_locally = True,
        confirm_tool_calls = True,
    )
    assert _selects_only_provider_hosted_tools(payload, provider_type) is True


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_a_mixed_selection_with_the_flag_still_takes_the_loop(provider_type):
    """One name Studio can run is enough; the hosted one rides along as a flag."""
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
    """Absent and False are the same decision here, but the wire must show the
    difference, so a client can be read as "did not know about this" rather than
    "asked for hosted"."""
    payload = _payload(enabled_tools = ["web_search"])
    dumped = payload.model_dump(exclude_unset = True)
    assert "run_tools_locally" not in dumped
