# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for coding-agent CLI detection used by the API-keys settings panel."""

import os

from subprocess import CompletedProcess
from unittest.mock import patch

from utils.coding_agents import (
    CODING_AGENTS,
    detect_installed_coding_agents,
    is_deepseek_harness_executable,
)


def test_matches_unsloth_start_subcommands():
    # Each entry must be an actual `unsloth start <agent>` subcommand name
    # (unsloth_cli/commands/start.py). Spelled out here rather than imported
    # from that module, which pulls in the CLI's heavier dependencies.
    assert CODING_AGENTS == ("claude", "codex", "openclaw", "opencode", "hermes", "pi", "dsh")


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


def test_rejects_distributed_shell_as_deepseek_harness():
    with (
        patch(
            "utils.coding_agents.shutil.which",
            side_effect = lambda name, path = None: "/usr/bin/dsh" if name == "dsh" else None,
        ),
        patch("utils.coding_agents.is_deepseek_harness_executable", return_value = False),
    ):
        assert detect_installed_coding_agents() == []


def test_detects_harness_behind_an_unrelated_earlier_path_entry(monkeypatch):
    system_dir = os.path.abspath("system-bin")
    harness_dir = os.path.abspath("user-bin")
    monkeypatch.setenv("PATH", os.pathsep.join((system_dir, harness_dir)))

    def which(name, path = None):
        if name != "dsh":
            return None
        return os.path.join(system_dir if path is None else path, "dsh")

    with (
        patch("utils.coding_agents.shutil.which", side_effect = which),
        patch(
            "utils.coding_agents.is_deepseek_harness_executable",
            side_effect = lambda executable, **_: os.path.dirname(executable) == harness_dir,
        ),
    ):
        assert detect_installed_coding_agents() == ["dsh"]


def test_detects_dsh_that_identifies_as_deepseek_harness():
    with (
        patch(
            "utils.coding_agents.shutil.which",
            side_effect = lambda name: "/usr/local/bin/dsh" if name == "dsh" else None,
        ),
        patch("utils.coding_agents.is_deepseek_harness_executable", return_value = True),
    ):
        assert detect_installed_coding_agents() == ["dsh"]


def test_identifies_an_npm_dsh_launcher_without_executing_it(tmp_path):
    launcher = tmp_path / "dsh"
    launcher.write_text(
        '#!/bin/sh\nexec node /usr/lib/node_modules/@deepseek-ai/dsh/lib/bin.js "$@"\n'
    )
    with patch(
        "utils.coding_agents.subprocess.run",
        side_effect = AssertionError("advisory detection must not execute the launcher"),
    ):
        assert is_deepseek_harness_executable(str(launcher), allow_execution = False)


def test_explicit_launch_can_probe_a_custom_dsh_wrapper(tmp_path):
    launcher = tmp_path / "dsh"
    launcher.write_text("#!/bin/sh\n")
    with patch(
        "utils.coding_agents.subprocess.run",
        return_value = CompletedProcess(
            [str(launcher), "--help"],
            0,
            stdout = "dsh: boot a DeepSeek Harness profile\n",
            stderr = "",
        ),
    ):
        assert is_deepseek_harness_executable(str(launcher))


def test_explicit_probe_replaces_undecodable_help_bytes(tmp_path):
    launcher = tmp_path / "dsh"
    launcher.write_text("#!/bin/sh\n")

    def probe(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return CompletedProcess(
            command,
            0,
            stdout = "\ufffd dsh: boot a DeepSeek Harness profile\n",
            stderr = "",
        )

    with patch("utils.coding_agents.subprocess.run", side_effect = probe):
        assert is_deepseek_harness_executable(str(launcher))
