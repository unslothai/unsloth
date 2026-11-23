# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-caller MCP profiles set tools offered, host-file reach and key lifetime; tools only declare
FACTS, and a profile withholds the tools whose facts it cannot allow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Optional

SWAPS_MODELS = "swaps_models"
# Never withheld (no REASONS entry): a profile bounds it with load_kinds.
LOADS_BY_KIND = "loads_by_kind"
PREVIEWS_SWAPS = "previews_swaps"
REENTERS_CHAT = "reenters_chat"
STOPS_WORK = "stops_work"
OUTLIVES_CALL = "outlives_call"
SENDS_HOST_SECRETS = "sends_host_secrets"

REASONS = {
    SWAPS_MODELS: "it could unload the model serving the chat",
    PREVIEWS_SWAPS: "a chat's load_model refuses any load that would unload its model, so it needs no preview",
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
    prompts: bool = True
    # The model kinds load_model, unload_model and plan_load take; None: every kind.
    load_kinds: Optional[frozenset[str]] = None
    # Loads and downloads only Unsloth Studio catalog models, so a prompt cannot pick any Hub repo.
    catalog_only: bool = False

    def withheld(self, facts: Mapping[str, Collection[str]]) -> dict[str, str]:
        """Each withheld tool with the reason, in ``facts`` order."""
        return {
            name: REASONS[fact]
            for name, tool_facts in facts.items()
            for fact in sorted(self.refused & set(tool_facts))[:1]
        }

    def tools(self, facts: Mapping[str, Collection[str]], names: Collection[str]) -> frozenset[str]:
        return frozenset(names) - set(self.withheld(facts))


AGENT = CallerProfile("agent")
CHAT = CallerProfile(
    "chat",
    refused = frozenset(REASONS),
    call_scoped = True,
    # A prompt-injected model must not reach host files, even the owner's.
    host_files = False,
    prompts = False,
    # Image, video and speech-to-text load beside the chat model; chat and audio models would replace it.
    load_kinds = frozenset({"image", "video", "stt"}),
    catalog_only = True,
)
