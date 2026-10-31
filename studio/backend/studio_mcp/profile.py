# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who an MCP call answers to, and what follows from it: the tools it is offered, whether host
files are reachable, and whether its key outlives the call. Tool modules only declare facts about
their tools (``FACTS``); a profile withholds the tools whose facts it cannot allow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping

# Loads or unloads models, so it can evict the model serving a chat.
SWAPS_MODELS = "swaps_models"
# Only previews what a model swap would do.
PREVIEWS_SWAPS = "previews_swaps"
# Runs a chat turn, so a chat calling it would re-enter itself.
REENTERS_CHAT = "reenters_chat"
# Stops running work by kind, including the chat that calls it.
STOPS_WORK = "stops_work"
# Keeps working with the caller's key after the call returns.
OUTLIVES_CALL = "outlives_call"
# Can send host env credentials and host files to an endpoint its input names.
SENDS_HOST_SECRETS = "sends_host_secrets"

REASONS = {
    SWAPS_MODELS: "it could unload the model serving the chat",
    PREVIEWS_SWAPS: "it previews load_model, which a chat does not have",
    REENTERS_CHAT: "it would re-enter the chat that calls it",
    STOPS_WORK: "it could stop the chat that calls it",
    OUTLIVES_CALL: "it keeps working after the chat's per-call key is revoked",
    SENDS_HOST_SECRETS: "it can send host credentials and files to an endpoint a prompt names",
}


@dataclass(frozen = True)
class CallerProfile:
    name: str
    # Facts a tool may not have for this caller to be offered it.
    refused: frozenset[str] = frozenset()
    # The key ends with the tool call, so nothing may keep it for later work.
    call_scoped: bool = False
    # Host file paths may be read when the request itself comes from this computer.
    host_files: bool = True
    # The MCP prompts (guided workflows) are listed.
    prompts: bool = True

    def withheld(self, facts: Mapping[str, Collection[str]]) -> dict[str, str]:
        """Each withheld tool with the reason, in ``facts`` order."""
        return {
            name: REASONS[fact]
            for name, tool_facts in facts.items()
            for fact in sorted(self.refused & set(tool_facts))[:1]
        }

    def tools(self, facts: Mapping[str, Collection[str]], names: Collection[str]) -> frozenset[str]:
        return frozenset(names) - set(self.withheld(facts))


# An agent on /mcp: every tool the owner left on.
AGENT = CallerProfile("agent")
# A chat turn calling Unsloth Studio's own tools in process.
CHAT = CallerProfile(
    "chat",
    refused = frozenset(REASONS),
    call_scoped = True,
    # A prompt-injected model must not reach host files, even the owner's.
    host_files = False,
    prompts = False,
)
