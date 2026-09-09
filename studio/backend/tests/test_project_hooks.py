# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for trusted project lifecycle hooks."""

import json
import os
from pathlib import Path

import pytest

from core.agent_workspace import hooks
from core.agent_workspace.verification_context import AgentWorkspaceError


def _document(
    *handlers: dict,
    event: str = "PreToolUse",
    matcher: str = "Bash",
) -> bytes:
    return json.dumps(
        {
            "description": "repository policy",
            "hooks": {
                event: [
                    {
                        "matcher": matcher,
                        "hooks": list(handlers),
                    }
                ]
            },
        },
        separators = (",", ":"),
    ).encode()


def _command(command: str, **extra) -> dict:
    return {"type": "command", "command": command, **extra}


@pytest.mark.skipif(os.name == "nt", reason = "Secure hook discovery requires POSIX descriptors")
def test_discovers_exact_file_hash_and_requires_retrust_after_change(tmp_path):
    root = tmp_path / "repository"
    config_path = root / ".codex" / "hooks.json"
    config_path.parent.mkdir(parents = True)
    first = _document(_command("printf first"))
    config_path.write_bytes(first)
    identity = (root.stat().st_dev, root.stat().st_ino)

    discovered = hooks.discover_project_hooks(root, expected_identity = identity)

    assert discovered["sourcePath"] == ".codex/hooks.json"
    assert discovered["contentHash"] == hooks.project_hooks_content_hash(first)
    assert discovered["handlerCount"] == 1
    assert hooks.project_hooks_are_trusted(discovered, discovered["contentHash"])

    trusted_hash = discovered["contentHash"]
    second = _document(_command("printf second"))
    config_path.write_bytes(second)
    changed = hooks.discover_project_hooks(root, expected_identity = identity)

    assert changed["contentHash"] != trusted_hash
    assert not hooks.project_hooks_are_trusted(changed, trusted_hash)


@pytest.mark.skipif(os.name == "nt", reason = "Secure hook discovery requires POSIX descriptors")
def test_discovery_is_empty_when_file_is_absent_and_rejects_symlinks(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    assert hooks.discover_project_hooks(root)["exists"] is False

    outside = tmp_path / "outside.json"
    outside.write_bytes(_document(_command("true")))
    codex = root / ".codex"
    codex.mkdir()
    (codex / "hooks.json").symlink_to(outside)

    with pytest.raises(AgentWorkspaceError, match = "unsafe filesystem boundary"):
        hooks.discover_project_hooks(root)


@pytest.mark.parametrize(
    "raw, message",
    [
        (b'{"hooks":{},"hooks":{}}', "duplicate field"),
        (b'{"hooks":{"MadeUp":[]}}', "unsupported events"),
        (
            _document(_command("true"), matcher = "(a+)+"),
            "unsupported or unsafe pattern",
        ),
        (
            _document(_command("true"), matcher = "^" + "a*" * 20 + "b$"),
            "unsupported or unsafe pattern",
        ),
        (
            _document({"type": "mcp_tool", "server": "scanner", "tool": "scan"}),
            "must be command",
        ),
        (
            _document({"type": "command", "commandWindows": "echo windows"}),
            "command is required",
        ),
        (
            _document(_command("true", timeout = 4), event = "SessionEnd", matcher = "other"),
            "outside the supported range",
        ),
    ],
)
def test_validation_rejects_ambiguous_or_unsupported_schema(raw, message):
    with pytest.raises(AgentWorkspaceError, match = message):
        hooks.validate_project_hooks(raw)


def test_matchers_support_aliases_and_ignore_matcher_for_stop():
    edit_config = hooks.validate_project_hooks(
        _document(_command("true"), matcher = "^Edit$|^Write$")
    )
    stop_config = hooks.validate_project_hooks(
        _document(_command("true"), event = "Stop", matcher = "will-not-match")
    )

    assert (
        len(
            hooks.matching_project_hooks(
                edit_config,
                "PreToolUse",
                {"tool_name": "apply_patch"},
            )
        )
        == 1
    )
    assert (
        hooks.matching_project_hooks(
            edit_config,
            "PreToolUse",
            {"tool_name": "Bash"},
        )
        == []
    )
    assert len(hooks.matching_project_hooks(stop_config, "Stop", {})) == 1


@pytest.mark.parametrize(
    "matcher, tool_name, matches",
    [
        ("Bash", "Bash", True),
        ("Edit|Write", "Write", True),
        ("^Bash$", "Bash", True),
        ("^Bash$", "BashTool", False),
        ("mcp__memory__.*", "mcp__memory__read_graph", True),
        ("^mcp__.*__read$", "mcp__memory__read", True),
        (r"^file\.txt$", "file.txt", True),
        (r"^Literal\|Pipe$", "Literal|Pipe", True),
        ("^.at$", "cat", True),
        ("^.at$", "chat", False),
    ],
)
def test_safe_matchers_support_official_style_patterns(matcher, tool_name, matches):
    config = hooks.validate_project_hooks(_document(_command("true"), matcher = matcher))

    selected = hooks.matching_project_hooks(config, "PreToolUse", {"tool_name": tool_name})

    assert bool(selected) is matches


@pytest.mark.parametrize("matcher", ["", "*"])
def test_empty_and_star_matchers_match_every_value(matcher):
    config = hooks.validate_project_hooks(_document(_command("true"), matcher = matcher))

    assert hooks.matching_project_hooks(config, "PreToolUse", {"tool_name": "anything"})


@pytest.mark.parametrize(
    "matcher",
    [
        "^(Edit|Write)$",
        "[A-Z]+",
        "Bash?",
        "Bash{1,3}",
        "Bash*",
        "Bash||Write",
        "Bash\\",
        "Ba^sh",
        "Ba$sh",
    ],
)
def test_matcher_rejects_groups_classes_quantifiers_and_ambiguous_forms(matcher):
    with pytest.raises(AgentWorkspaceError, match = "unsupported or unsafe pattern"):
        hooks.validate_project_hooks(_document(_command("true"), matcher = matcher))


def test_hook_module_exposes_no_command_runner():
    assert not hasattr(hooks, "run_project_hooks")
    assert not hasattr(hooks, "_run_project_hooks_from_config")
    assert not hasattr(hooks, "_run_command")


@pytest.mark.skipif(os.name != "nt", reason = "Windows discovery refusal")
def test_windows_hook_discovery_refuses_without_reading_commands(tmp_path):
    from core.agent_workspace.hooks import discover_project_hooks
    with pytest.raises(AgentWorkspaceError):
        discover_project_hooks(tmp_path)


@pytest.mark.parametrize(
    "matcher,tool_name",
    [
        ("Bash", "terminal"),
        ("Terminal", "terminal"),
        ("Edit", "edit_file"),
        ("Write", "edit_file"),
        ("Python", "python"),
    ],
)
def test_studio_tool_names_match_reviewed_hook_aliases(matcher, tool_name):
    config = hooks.validate_project_hooks(_document(_command("true"), matcher = matcher))
    assert len(hooks.matching_project_hooks(config, "PreToolUse", {"tool_name": tool_name})) == 1
    assert hooks.matching_project_hooks(config, "PreToolUse", {"tool_name": "web_search"}) == []
