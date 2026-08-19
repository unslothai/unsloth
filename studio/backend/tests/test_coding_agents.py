# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for coding-agent CLI detection used by the API-keys settings panel."""

from unittest.mock import patch

from utils.coding_agents import CODING_AGENTS, detect_installed_coding_agents


def test_matches_unsloth_start_subcommands():
    # Each entry must be an actual `unsloth start <agent>` subcommand name
    # (unsloth_cli/commands/start.py). Spelled out here rather than imported
    # from that module, which pulls in the CLI's heavier dependencies.
    assert CODING_AGENTS == ("claude", "codex", "openclaw", "opencode", "hermes", "pi")


def test_detects_only_agents_present_on_path():
    installed = {"claude", "opencode"}
    with patch(
        "utils.coding_agents.shutil.which",
        side_effect = lambda name: f"/usr/bin/{name}" if name in installed else None,
    ):
        assert detect_installed_coding_agents() == ["claude", "opencode"]


def test_returns_empty_list_when_nothing_is_installed():
    with patch("utils.coding_agents.shutil.which", return_value = None):
        assert detect_installed_coding_agents() == []


def test_preserves_declared_order_regardless_of_path_lookup_order():
    with patch(
        "utils.coding_agents.shutil.which",
        side_effect = lambda name: name if name in ("pi", "claude", "hermes") else None,
    ):
        assert detect_installed_coding_agents() == ["claude", "hermes", "pi"]


def test_treats_a_path_lookup_error_as_not_installed():
    # An advisory check: shutil.which raising for one entry (e.g. a permission
    # error walking a PATH directory) should not take down the whole endpoint,
    # and should not stop the remaining agents from being checked.
    def flaky_which(name: str):
        if name == "codex":
            raise OSError("permission denied")
        return name if name == "claude" else None

    with patch("utils.coding_agents.shutil.which", side_effect = flaky_which):
        assert detect_installed_coding_agents() == ["claude"]
