# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os

import pytest

from core import agents_md
from routes import chat_history


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(agents_md, "_owner_home", lambda: tmp_path / "home")
    monkeypatch.setattr(agents_md, "workspace_root", lambda: tmp_path / "home/.unsloth/studio")
    monkeypatch.setattr(agents_md, "is_owner_context", lambda: True)
    monkeypatch.delenv("UNSLOTH_STUDIO_AGENTS_MD", raising = False)
    for name in ("home/.agents", "home/.claude", "home/.unsloth/studio", "proj/sandbox", "host"):
        (tmp_path / name).mkdir(parents = True)
    return tmp_path


def _project(home, **extra):
    return {
        "rootPath": str(home / "proj"),
        "sandboxPath": str(home / "proj/sandbox"),
        "archived": False,
        **extra,
    }


def _write(path, text):
    path.write_text(text, encoding = "utf-8")


def test_nothing_on_disk_reads_as_empty(home):
    assert agents_md.agents_md_text(None) == ""
    assert agents_md.agents_md_text(_project(home)) == ""


def test_global_then_project_then_sandbox_each_under_its_source(home):
    _write(home / "home/.unsloth/studio/AGENTS.md", "Be brief.\n")
    _write(home / "proj/AGENTS.md", "Use pnpm.")
    _write(home / "proj/sandbox/AGENTS.md", "Notes: tests pass.")
    assert agents_md.agents_md_text(_project(home)) == (
        "# Source: ~/.unsloth/studio/AGENTS.md\n\nBe brief.\n\n"
        f"# Source: {(home / 'proj/AGENTS.md').as_posix()}\n\nUse pnpm.\n\n"
        f"# Source: {(home / 'proj/sandbox/AGENTS.md').as_posix()}\n\nNotes: tests pass."
    )
    # A chat outside any project still gets the global file.
    assert agents_md.agents_md_text(None) == "# Source: ~/.unsloth/studio/AGENTS.md\n\nBe brief."


def test_other_tools_global_files_are_not_read(home):
    _write(home / "home/.agents/AGENTS.md", "agents rules")
    _write(home / "home/.claude/CLAUDE.md", "claude rules")
    assert agents_md.agents_md_text(_project(home)) == ""


def test_claude_md_only_stands_in_for_a_missing_agents_md(home):
    _write(home / "proj/CLAUDE.md", "From CLAUDE.md")
    assert agents_md.agents_md_text(_project(home)).endswith("From CLAUDE.md")
    _write(home / "proj/AGENTS.md", "From AGENTS.md")
    text = agents_md.agents_md_text(_project(home))
    assert text.endswith("From AGENTS.md") and "CLAUDE.md" not in text


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "needs symlinks")
def test_project_root_link_reads_but_sandbox_link_does_not(home):
    _write(home / "proj/AGENTS.md", "Use pnpm.")
    # CLAUDE.md -> AGENTS.md at the root is common and stays readable.
    os.symlink(home / "proj/AGENTS.md", home / "proj/CLAUDE.md")
    (home / "proj/AGENTS.md").unlink()
    _write(home / "proj/real.md", "Linked rules")
    os.symlink(home / "proj/real.md", home / "proj/AGENTS.md")
    assert agents_md.agents_md_text(_project(home)).endswith("Linked rules")
    # A link the model plants in its sandbox must not pull a host file into the prompt.
    _write(home / "host/secret.txt", "HOST SECRET")
    os.symlink(home / "host/secret.txt", home / "proj/sandbox/AGENTS.md")
    os.symlink(home / "host/secret.txt", home / "proj/sandbox/CLAUDE.md")
    assert "HOST SECRET" not in agents_md.agents_md_text(_project(home))


def test_edits_are_seen_and_size_is_capped(home):
    target = home / "home/.unsloth/studio/AGENTS.md"
    _write(target, "first")
    assert agents_md.agents_md_text(None).endswith("first")
    _write(target, "second edit")
    assert agents_md.agents_md_text(None).endswith("second edit")
    _write(target, "x" * (agents_md.MAX_AGENTS_MD_BYTES * 2))
    text = agents_md.agents_md_text(None)
    assert text.endswith(agents_md.TRUNCATED_NOTE)
    assert len(text) == agents_md.MAX_AGENTS_MD_BYTES + len(agents_md.TRUNCATED_NOTE)


def test_managed_account_reads_its_workspace_never_the_host(home, monkeypatch):
    _write(home / "proj/AGENTS.md", "Host project rules")
    monkeypatch.setattr(agents_md, "is_owner_context", lambda: False)
    monkeypatch.setattr(agents_md, "workspace_root", lambda: home / "account")
    (home / "account").mkdir()
    assert agents_md.agents_md_text(_project(home)) == ""
    _write(home / "account/AGENTS.md", "Account rules")
    text = agents_md.agents_md_text(_project(home))
    assert text.endswith("Account rules") and "Host" not in text


def test_archived_project_and_kill_switch(home, monkeypatch):
    _write(home / "home/.unsloth/studio/AGENTS.md", "Be brief.")
    _write(home / "proj/AGENTS.md", "Use pnpm.")
    assert "pnpm" not in agents_md.agents_md_text(_project(home, archived = True))
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("UNSLOTH_STUDIO_AGENTS_MD", off)
        assert agents_md.agents_md_text(_project(home)) == ""


def test_route_returns_the_text(home, monkeypatch):
    _write(home / "proj/AGENTS.md", "Use pnpm.")
    project = _project(home)
    monkeypatch.setattr(
        chat_history, "get_chat_project", lambda pid: project if pid == "p1" else None
    )
    assert chat_history.get_agents_md("p1", current_subject = "owner").text.endswith("Use pnpm.")
    assert chat_history.get_agents_md(None, current_subject = "owner").text == ""
    assert chat_history.get_agents_md("missing", current_subject = "owner").text == ""


def test_sandbox_hard_link_and_fifo_are_not_read(home):
    root = home / "p"
    sandbox = root / "sandbox"
    sandbox.mkdir(parents = True)
    project = {"rootPath": str(root), "sandboxPath": str(sandbox), "archived": False}
    secret = home / "secret.txt"
    secret.write_text("host secret", encoding = "utf-8")
    os.link(secret, sandbox / "AGENTS.md")
    assert "host secret" not in agents_md.agents_md_text(project)
    (sandbox / "AGENTS.md").unlink()
    os.mkfifo(sandbox / "AGENTS.md")
    assert agents_md.agents_md_text(project) == ""
