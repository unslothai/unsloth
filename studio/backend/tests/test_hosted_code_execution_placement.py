# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which side runs the code when a turn mixes the Code pill with an Unsloth tool.

``code_execution`` runs in the provider's sandbox; ``python`` / ``terminal`` run
on the machine Unsloth is installed on. They are two trust boundaries, not two
spellings of one feature, so the request says which one it wants by name and
this server forwards accordingly. Unsloth has no implementation of
``code_execution`` (``ALL_TOOLS`` is web_search / python / terminal / render_html
/ search_knowledge_base), so filtering it out as "locally replaced" does not
substitute anything -- it drops the tool while its pill stays lit, and the model
is never offered a sandbox at all.

The one case that IS a substitution is a request naming both, which no Unsloth
build sends: there the local names win and the hosted one is dropped, so a
single pill can never bill the provider and run on this host at the same time.
"""

import pytest

from core.inference.providers import hosted_only_tools
from core.inference.tools import ALL_TOOLS


def test_studio_really_has_no_local_code_execution():
    """The premise. If this ever fails, the filtering rule below is wrong."""
    assert "code_execution" not in {tool["function"]["name"] for tool in ALL_TOOLS}


@pytest.mark.parametrize("provider_type", ["openai", "anthropic", "gemini"])
def test_hosted_code_execution_rides_along_with_a_studio_tool(provider_type):
    """RAG or MCP selects the Unsloth loop; the Code pill must still reach the
    provider's sandbox rather than being dropped on the way."""
    assert hosted_only_tools(provider_type, ["search_knowledge_base", "code_execution"]) == [
        "code_execution"
    ]


def test_the_local_code_tools_still_win_when_a_request_names_both():
    """Belt and braces for a third-party client: never both sides of one tool."""
    assert hosted_only_tools("openai", ["python", "code_execution"]) == []
    assert hosted_only_tools("openai", ["terminal", "code_execution"]) == []


def test_web_search_is_still_never_forwarded():
    """Unsloth's catalog does contain web_search, so that one really is replaced."""
    assert hosted_only_tools("openai", ["web_search", "search_knowledge_base"]) == []
    assert hosted_only_tools("anthropic", ["web_search", "python"]) == []


def test_a_provider_without_a_sandbox_is_not_offered_one():
    assert "code_execution" not in hosted_only_tools(
        "openrouter", ["search_knowledge_base", "code_execution"]
    )
    assert hosted_only_tools("llama_cpp", ["code_execution"]) == []
