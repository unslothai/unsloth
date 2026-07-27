# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect which `unsloth start <agent>` coding-agent CLIs are on PATH.

The web UI only ever shows the user the "claude" flavor of the `unsloth start`
command (see agent-command.ts), leaving anyone using Codex, OpenCode, and the
other supported agents to manually edit the copied command. This module gives
the frontend a way to ask which of those CLIs are actually installed so it can
default to one the user can run immediately.
"""

import shutil

# Keep in sync with the `unsloth start <agent>` subcommands defined in
# unsloth_cli/commands/start.py. Each entry is the exact executable name that
# subcommand launches, so a hit here means `unsloth start <agent>` can find the
# binary on PATH without the user installing anything first.
CODING_AGENTS: tuple[str, ...] = ("claude", "codex", "openclaw", "opencode", "hermes", "pi")


def _is_on_path(agent: str) -> bool:
    # shutil.which is documented to return None on a miss, but PATH lookups can
    # still raise (e.g. a permission error while probing a directory entry);
    # this is an advisory check, so a lookup failure should read as "not
    # installed" instead of breaking the settings endpoint.
    try:
        return shutil.which(agent) is not None
    except OSError:
        return False


def detect_installed_coding_agents() -> list[str]:
    """Return the subset of CODING_AGENTS whose CLI binary is on PATH.

    Order follows CODING_AGENTS, not discovery order, so callers can treat the
    first entry as the preferred default among the installed agents.
    """
    return [agent for agent in CODING_AGENTS if _is_on_path(agent)]
