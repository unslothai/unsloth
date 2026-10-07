# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth start` — config merging and launch env, no network."""

from __future__ import annotations

import http.client
import http.server
import io
import json
import os
import re
import shlex
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path
from types import ModuleType, SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "unsloth/gemma-4-26B-A4B-it-GGUF", "context_length": 131072}


def _assert_env_set(output: str, name: str, value: str) -> None:
    needle = f'$env:{name} = "{value}"' if os.name == "nt" else f"export {name}={value}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_unset(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_kept(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle not in output, f"{needle!r} unexpectedly found in:\n{output}"


def _assert_env_cwd(output: str, name: str) -> None:
    needle = f"$env:{name} = (Get-Location).Path" if os.name == "nt" else f'export {name}="$PWD"'
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _launch_command(output: str) -> list:
    last = [ln for ln in output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    for i, part in enumerate(parts):
        name = part.partition("=")[0]
        if "=" not in part or not name.replace("_", "").isalnum():
            return parts[i:]
    return []


def _fake_claude(monkeypatch, version_output: str) -> None:
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_probe_env", lambda **_: {})
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = version_output),
    )


def _path_aware_which(binaries: dict):
    # A which() fake that resolves a name only when its directory is on PATH at call time.
    def _which(name):
        directory = binaries.get(name)
        if directory is None:
            return None
        entries = os.environ.get("PATH", "").split(os.pathsep)
        # os.path.join, not Path(): pathlib raises under a simulated os.name.
        return os.path.join(str(directory), name) if str(directory) in entries else None

    return _which


def _simulate_windows(monkeypatch) -> None:
    # Flipping os.name makes pathlib build the non-native flavour and raise; pin the host class.
    monkeypatch.setattr(start, "Path", type(Path()))
    monkeypatch.setattr(start.os, "name", "nt")


def test_claude_flags_passed_to_supported_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.1.98 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_dynamic_sections_skipped_on_old_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.0.14 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_settings_retained_on_unparseable_version(monkeypatch):
    _fake_claude(monkeypatch, "weird build string\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_detected_when_version_not_first_token(monkeypatch):
    _fake_claude(monkeypatch, "claude version 2.1.98\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_settings_overlay_pins_served_model():
    # A user's availableModels allowlist would reject --model; only a NON-EMPTY override takes effect.
    overlay = json.loads(start._claude_settings_overlay(MODEL["id"]))
    assert overlay["availableModels"] == [MODEL["id"]]


def test_claude_settings_overlay_pins_local_routing_and_auth():
    local_env = start._claude_local_env(BASE, "sk-unsloth-test", MODEL)
    overlay = json.loads(start._claude_settings_overlay(MODEL["id"], local_env))
    for name, value in local_env.items():
        assert overlay["env"][name] == value
    assert overlay["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert overlay["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-test"
    for name in start._CLAUDE_ENV_UNSET:
        assert overlay["env"][name] == ""
    assert overlay["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert overlay["env"]["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"
    assert overlay["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"


def test_claude_settings_files_preserve_concurrent_sessions(tmp_path):
    first_env = start._claude_local_env("http://127.0.0.1:8001", "first-key", MODEL)
    second_env = start._claude_local_env("http://127.0.0.1:8002", "second-key", MODEL)
    first = start._write_claude_settings(tmp_path, MODEL["id"], first_env)
    second = start._write_claude_settings(tmp_path, MODEL["id"], second_env)
    assert first != second
    assert json.loads(first.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "first-key"
    assert json.loads(second.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "second-key"


def test_install_agent_prompts_then_installs(monkeypatch):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: "/usr/local/bin/npm")
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)
    ran = []
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    executable = start._install_agent("codex", "npm install -g @openai/codex")
    assert executable == "/usr/local/bin/codex"
    assert ran == [["/usr/local/bin/npm", "install", "-g", "@openai/codex"]]


def test_install_agent_uses_powershell_on_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    ran = []
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: r"C:\Users\samle\bin\hermes.exe")

    install_hint = "& ([scriptblock]::Create((irm https://x/install.ps1))) -SkipSetup"
    executable = start._install_agent("hermes", install_hint)

    assert executable == r"C:\Users\samle\bin\hermes.exe"
    assert ran == [
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", install_hint]
    ]


def test_install_agent_windows_failure_hints_execution_policy(monkeypatch, capsys):
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        start, "_npm_executable", lambda: r"C:\Users\me\AppData\Roaming\npm\npm.cmd"
    )
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: None)

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" in err


def test_install_command_uses_resolved_npm_cmd_on_windows(monkeypatch):
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start, "_npm_executable", lambda: r"C:\Managed Node\npm.cmd")

    command, env = start._install_command(start._npm_install_hint("@openai/codex"))

    assert command == [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "& 'C:\\Managed Node\\npm.cmd' install -g '@openai/codex'",
    ]
    assert env is not None


def test_install_agent_posix_failure_omits_execution_policy_hint(monkeypatch, capsys):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy" not in err


def test_npm_install_hint_uses_user_prefix_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)

    hint = start._npm_install_hint("@openai/codex")

    assert shlex.split(hint) == [
        "npm",
        "install",
        "-g",
        "--prefix",
        str(tmp_path / ".local"),
        "@openai/codex",
    ]


@pytest.mark.skipif(os.name == "nt", reason = "managed node layout is POSIX")
def test_npm_executable_uses_studio_managed_node(monkeypatch, tmp_path):
    start.ensure_studio_backend_path()
    from utils import node_runtime

    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    node = managed_bin / "node"
    npm = managed_bin / "npm"
    node.touch()
    npm.touch()
    node.chmod(0o755)
    npm.chmod(0o755)
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.setattr(node_runtime, "managed_node_binary", lambda: node)
    monkeypatch.setattr(node_runtime, "resolve_node_executable", lambda: str(node))

    assert start._npm_executable() == str(npm)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_npm_executable_skips_wsl_shim_and_finds_native_npm(monkeypatch):
    native = "/usr/bin/npm"
    shim = "/mnt/c/Program Files/nodejs/npm"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("PATH", "/mnt/c/Program Files/nodejs:/usr/bin")
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)

    found = {"/mnt/c/Program Files/nodejs": shim, "/usr/bin": native}

    def fake_which(name, path = None):
        if os.path.isabs(name):
            return name
        for entry in (path or os.environ["PATH"]).split(os.pathsep):
            if entry in found:
                return found[entry]
        return None

    monkeypatch.setattr(start.shutil, "which", fake_which)

    assert start._npm_executable() == native


@pytest.mark.skipif(os.name == "nt", reason = "POSIX hint form")
def test_npm_install_hint_without_resolvable_home(monkeypatch):
    def no_home():
        raise RuntimeError("no home directory")

    monkeypatch.setattr(start.Path, "home", staticmethod(no_home))

    assert start._npm_install_hint("@openai/codex") == "npm install -g @openai/codex"


def test_managed_node_probe_tolerates_unsupported_path_flavour(monkeypatch):
    def unsupported_backend_path():
        raise RuntimeError("unsupported path flavour")

    monkeypatch.setattr(start, "ensure_studio_backend_path", unsupported_backend_path)

    assert start._managed_node_tools() is None


@pytest.mark.skipif(os.name == "nt", reason = "managed node layout is POSIX")
def test_npm_executable_uses_managed_npm_when_system_node_has_none(monkeypatch, tmp_path):
    start.ensure_studio_backend_path()
    from utils import node_runtime

    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    node = managed_bin / "node"
    npm = managed_bin / "npm"
    node.touch()
    npm.touch()
    node.chmod(0o755)
    npm.chmod(0o755)
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr(node_runtime, "managed_node_binary", lambda: node)
    monkeypatch.setattr(node_runtime, "resolve_node_executable", lambda: str(node))

    assert start._npm_executable() == str(npm)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_npm_executable_prefers_managed_npm_over_windows_npm_in_wsl(monkeypatch, tmp_path):
    start.ensure_studio_backend_path()
    from utils import node_runtime

    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    node = managed_bin / "node"
    npm = managed_bin / "npm"
    node.touch(mode = 0o755)
    npm.touch(mode = 0o755)
    windows_npm = "/mnt/c/Program Files/nodejs/npm"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda name: windows_npm if name in ("npm", windows_npm) else None,
    )
    monkeypatch.setattr(node_runtime, "managed_node_binary", lambda: node)
    monkeypatch.setattr(node_runtime, "resolve_node_executable", lambda: str(node))

    assert start._npm_executable() == str(npm)


@pytest.mark.skipif(os.name == "nt", reason = "managed node layout is POSIX")
def test_augment_path_includes_studio_managed_node(monkeypatch, tmp_path):
    start.ensure_studio_backend_path()
    from utils import node_runtime

    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    node = managed_bin / "node"
    npm = managed_bin / "npm"
    node.touch(mode = 0o755)
    npm.touch(mode = 0o755)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(node_runtime, "managed_node_binary", lambda: node)
    monkeypatch.setattr(node_runtime, "resolve_node_executable", lambda: str(node))
    monkeypatch.setenv("PATH", os.pathsep.join([os.defpath, str(managed_bin)]))

    start._augment_path_with_install_dirs()

    path = os.environ["PATH"].split(os.pathsep)
    assert path[0] == str(managed_bin)
    assert path.count(str(managed_bin)) == 1


def test_install_agent_missing_npm_names_node_requirement(monkeypatch, capsys):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: None)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: pytest.fail("should not run an installer without npm"),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "npm is required" in err
    assert "Unsloth-managed Node" in err
    assert "Install Node.js with npm" in err


def test_install_agent_reports_os_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: "/broken/npm")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("permission denied")),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Could not run the install command: permission denied" in err
    assert "Run it yourself, then re-run" in err


@pytest.mark.skipif(os.name == "nt", reason = "POSIX install command")
def test_install_agent_runs_managed_npm_with_its_node_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    npm = tmp_path / "managed-node" / "bin" / "npm"
    monkeypatch.setattr(start, "_npm_executable", lambda: str(npm))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    hint = start._npm_install_hint("@openai/codex")

    assert start._install_agent("codex", hint) == "/usr/local/bin/codex"
    assert captured["command"] == [
        str(npm),
        "install",
        "-g",
        "--prefix",
        str(tmp_path / ".local"),
        "@openai/codex",
    ]
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(npm.parent)


def test_install_agent_warns_remote_installer_is_unverified_third_party(monkeypatch, capsys):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    hint = "& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -SkipSetup"
    assert start._install_agent("hermes", hint) is None
    err = capsys.readouterr().err
    assert "Security warning" in err
    assert "unverified third-party script" in err
    assert "https://hermes-agent.nousresearch.com/install.ps1" in err
    assert "Unsloth does not pin or verify the downloaded content" in err
    assert "Continue only if you trust this source" in err


def test_install_agent_reports_immutable_remote_installer_pin(monkeypatch, capsys):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    assert start._install_agent("hermes", start._HERMES_POSIX_INSTALL_HINT) is None
    err = capsys.readouterr().err
    assert start._HERMES_INSTALL_COMMIT in err
    assert "immutable upstream commit" in err
    assert "does not independently verify or sandbox it" in err
    assert "does not pin or verify" not in err


def test_install_agent_warns_for_package_installer(monkeypatch, capsys):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    assert start._install_agent("codex", "npm install -g @openai/codex") is None
    err = capsys.readouterr().err
    assert "npm install -g @openai/codex" in err
    assert "with your privileges" in err


def test_hermes_install_hint_is_windows_native_on_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "nt")

    assert start._hermes_install_hint() == (
        f"& ([scriptblock]::Create((irm {start._HERMES_INSTALL_BASE}/install.ps1)))"
        f" -SkipSetup -Commit {start._HERMES_INSTALL_COMMIT}"
    )


def test_hermes_install_hint_is_bash_on_posix(monkeypatch):
    monkeypatch.setattr(start.os, "name", "posix")

    assert start._hermes_install_hint() == (
        f"curl -fsSL {start._HERMES_INSTALL_BASE}/install.sh | bash -s --"
        f" --skip-setup --commit {start._HERMES_INSTALL_COMMIT}"
    )


def test_hermes_install_hints_pin_script_and_checkout_to_full_commit():
    commit = start._HERMES_INSTALL_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    for hint in (start._HERMES_WINDOWS_INSTALL_HINT, start._HERMES_POSIX_INSTALL_HINT):
        assert hint.count(commit) == 2
        assert "/main/" not in hint
        assert "hermes-agent.nousresearch.com" not in hint


def test_refresh_windows_path_noop_off_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "posix")
    before = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", before)
    start._refresh_windows_path()
    assert os.environ.get("PATH", "") == before


def test_refresh_windows_path_merges_registry_hives(monkeypatch):
    hkcu, hklm = object(), object()
    reg = {
        (hkcu, "Environment"): r"C:\existing;C:\Users\me\hermes\bin",
        (
            hklm,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ): r"C:\Windows\System32",
    }

    class _Key:
        def __init__(self, value):
            self._value = value

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_key(root, sub):
        if (root, sub) in reg:
            return _Key(reg[(root, sub)])
        raise OSError("missing hive")

    fake_winreg = SimpleNamespace(
        HKEY_CURRENT_USER = hkcu,
        HKEY_LOCAL_MACHINE = hklm,
        OpenKey = open_key,
        QueryValueEx = lambda key, name: (key._value, 1),
    )
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.os, "pathsep", ";")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setenv("PATH", r"C:\custom;C:\existing")

    start._refresh_windows_path()

    assert os.environ["PATH"].split(";") == [
        r"C:\custom",
        r"C:\existing",
        r"C:\Users\me\hermes\bin",
        r"C:\Windows\System32",
    ]


def test_augment_path_adds_existing_local_bin(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(local_bin) in entries
    assert entries[-1] == str(local_bin)


def test_augment_path_skips_missing_and_duplicate_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"] == str(tmp_path / "existing")

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "existing"), str(local_bin)]))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"].split(os.pathsep).count(str(local_bin)) == 1


def test_augment_path_adds_npm_global_bin_on_windows(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert str(npm_dir) in os.environ["PATH"].split(os.pathsep)


def test_which_with_install_dirs_finds_agent_and_restores_path(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    original = str(tmp_path / "existing")
    monkeypatch.setenv("PATH", original)
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    assert start._which_with_install_dirs("claude") == str(local_bin / "claude")
    assert os.environ["PATH"] == original


def test_claude_flags_probes_old_agent_only_in_install_dir(monkeypatch, tmp_path):
    # The version probe must augment PATH first, so an OLD claude only in ~/.local/bin gets no new flags.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_detects_supported_agent_only_in_install_dir(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.1.98 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_probes_npm_install_dir_on_windows(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": npm_dir}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_codex_catalog_probes_old_codex_only_in_install_dir(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"codex": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "codex-cli 0.109.0")
    assert start._codex_supports_model_catalog() is False


def test_opencode_native_auto_probes_old_opencode_only_in_install_dir(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"opencode": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "1.17.11")
    assert start._opencode_supports_native_auto() is False


def test_opencode_command_prefers_installed_v2(monkeypatch):
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode2" if name == "opencode2" else None,
    )
    assert start._opencode_command() == ("/usr/local/bin/opencode2", True)


def test_opencode_command_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    assert start._opencode_command() == ("opencode", False)


def test_opencode_command_finds_official_v2_install_dir(monkeypatch, tmp_path):
    install_dir = tmp_path / ".opencode" / "bin"
    install_dir.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"opencode2": install_dir}))

    assert start._opencode_command() == (str(install_dir / "opencode2"), True)


def test_augment_path_preserves_defpath_when_path_unset(monkeypatch, tmp_path):
    # PATH unset: which() falls back to os.defpath, so augmentation must keep those dirs.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    start._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    for default_dir in os.defpath.split(os.pathsep):
        if default_dir:
            assert default_dir in entries
    assert str(local_bin) in entries


def test_which_with_install_dirs_keeps_defpath_when_path_unset(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    sysdir = next(part for part in reversed(os.defpath.split(os.pathsep)) if part)
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": Path(sysdir)}))
    assert start._which_with_install_dirs("claude") == os.path.join(sysdir, "claude")
    assert "PATH" not in os.environ


def test_install_agent_declined_returns_none(monkeypatch):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install when declined")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def test_install_agent_non_interactive_returns_none(monkeypatch):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: False))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install without a TTY")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


def test_project_declares_direct_cli_dependencies():
    project = _parse_toml((_REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))
    assert "click>=8.0" in project["project"]["dependencies"]
    assert "huggingface-hub>=0.34.0" in project["project"]["dependencies"]


def test_agent_paths_use_cli_studio_home_without_backend_imports(monkeypatch, tmp_path):
    studio = ModuleType("unsloth_cli.commands.studio")
    studio.STUDIO_HOME = tmp_path
    monkeypatch.setitem(sys.modules, studio.__name__, studio)
    monkeypatch.setattr(
        start,
        "ensure_studio_backend_path",
        lambda: pytest.fail("agent paths should not import backend runtime packages"),
    )

    assert start._key_cache_path() == tmp_path / "auth" / "agent_api_key.json"
    assert start._agents_config_root() == tmp_path / "auth" / "agents"


def test_merge_codex_config_fresh():
    merged = start._merge_codex_config("", BASE)
    parsed = _parse_toml(merged)
    assert parsed["oss_provider"] == "unsloth_api"
    provider = parsed["model_providers"]["unsloth_api"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False


def test_merge_codex_config_raises_the_stream_idle_timeout():
    """Codex's 300s default cancels the stream while llama-server is still reading.

    llama-server emits nothing at all during prompt processing, so the whole wait counts
    as idle. Measured on a 2-core CI host: 16.1 tok/s against Codex's several-thousand
    token preamble is ~460s of silence before the first token exists, and the default
    trips at 300s. The reconnect then lands on a different parallel slot whose KV cache
    shares no prefix, so every retry restarts from zero and the turn never completes --
    a job hung for its full 600s cap with `Reconnecting... 1/5` and one request logged at
    exactly 300056ms.

    Asserted as a floor rather than an equality: raising it further is fine, and the
    number is not the contract. Losing it entirely is the regression.
    """
    provider = _parse_toml(start._merge_codex_config("", BASE))["model_providers"]["unsloth_api"]
    assert "stream_idle_timeout_ms" in provider, (
        "the Codex provider block no longer sets stream_idle_timeout_ms, so Codex falls "
        "back to its 300s default and cancels any first turn whose prompt takes longer "
        "than that to process -- which is an ordinary local CPU host, not a corner case"
    )
    assert provider["stream_idle_timeout_ms"] > 300_000, (
        f"stream_idle_timeout_ms is {provider['stream_idle_timeout_ms']}, at or below "
        f"Codex's own 300000 default, so setting it changes nothing"
    )


def test_merge_codex_config_replaces_stale_block():
    existing = (
        'model = "gpt-5"\n'
        "\n"
        "[model_providers.unsloth_api]\n"
        'base_url = "http://old-host:9999/v1"\n'
        'wire_api = "chat"\n'
        "\n"
        "[model_providers.unsloth_api.http_headers]\n"
        'x-old = "1"\n'
        "\n"
        "[model_providers.ollama]\n"
        'base_url = "http://localhost:11434/v1"\n'
    )
    merged = start._merge_codex_config(existing, BASE)
    parsed = _parse_toml(merged)
    assert parsed["model"] == "gpt-5"
    assert parsed["model_providers"]["unsloth_api"]["base_url"] == f"{BASE}/v1"
    assert parsed["model_providers"]["unsloth_api"]["wire_api"] == "responses"
    assert "http_headers" not in parsed["model_providers"]["unsloth_api"]
    assert parsed["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert start._merge_codex_config(merged, BASE) == merged


def test_merge_codex_config_keeps_user_oss_provider():
    merged = start._merge_codex_config('oss_provider = "ollama"\n', BASE)
    assert _parse_toml(merged)["oss_provider"] == "ollama"


def test_write_codex_config_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    monkeypatch.setattr(start, "_codex_supports_patch_line_endings", lambda: True)
    start.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    assert profile["oss_provider"] == "unsloth_api"
    assert profile["model_provider"] == "unsloth_api"
    assert profile["model"] == MODEL["id"]
    assert profile["model_context_window"] == 131072
    assert profile["features"]["apply_patch_preserve_line_endings"] is True
    assert profile["suppress_unstable_features_warning"] is True

    catalog_path = Path(profile["model_catalog_json"])
    assert catalog_path == Path("model-catalog.json")
    catalog = json.loads((tmp_path / catalog_path).read_text())
    assert catalog["models"][0]["slug"] == MODEL["id"]
    assert catalog["models"][0]["context_window"] == 131072
    assert catalog["models"][0]["max_context_window"] == 131072
    assert catalog["models"][0]["supports_reasoning_summary_parameter"] is False
    assert catalog["models"][0]["supports_parallel_tool_calls"] is False
    assert catalog["models"][0]["apply_patch_tool_type"] == "freeform"

    assert catalog["models"][0]["base_instructions"] == start._CODEX_FALLBACK_PROMPT.read_text(
        encoding = "utf-8"
    )
    assert '{"command"' not in catalog["models"][0]["base_instructions"]
    config = _parse_toml((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["unsloth_api"]["env_key"] == "UNSLOTH_STUDIO_AUTH_TOKEN"


def test_write_codex_config_catalog_without_context_length(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    start.write_codex_config(BASE, {"id": "unsloth/no-window"}, tmp_path)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    catalog = json.loads((tmp_path / profile["model_catalog_json"]).read_text())
    entry = catalog["models"][0]
    assert entry["slug"] == "unsloth/no-window"
    assert "context_window" not in entry
    assert "max_context_window" not in entry


@pytest.mark.parametrize(
    ("version", "expected"),
    [("codex-cli 0.109.0", False), ("codex-cli 0.110.0", True), ("codex-cli 0.144.4", True)],
)
def test_codex_model_catalog_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._codex_supports_model_catalog() is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("codex-cli 0.147.0", False),
        ("codex-cli 0.148.0", True),
        ("codex-cli 0.150.0", True),
        ("codex-cli 0.151.0", True),
        ("codex-cli 1.0.0", True),
    ],
)
def test_codex_patch_line_endings_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._codex_supports_patch_line_endings() is expected


def test_codex_patch_line_endings_assumes_current_when_not_installed(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    assert start._codex_supports_patch_line_endings() is True


def test_write_codex_config_omits_catalog_for_old_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    assert "model_catalog_json" not in profile
    assert not (tmp_path / "model-catalog.json").exists()


def test_write_codex_subagent_bridge_keeps_parent_credentials_out(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    path = start.write_codex_subagent_bridge(
        BASE,
        "private-token",
        local,
        tmp_path,
        yolo = False,
    )
    assert json.loads(path.read_text(encoding = "utf-8")) == {
        "api_key": "private-token",
        "codex_home": str(tmp_path / "child"),
        "bypass_permissions": False,
    }
    assert path.stat().st_mode & 0o077 == 0
    profile = _parse_toml((tmp_path / "child" / "unsloth_api.config.toml").read_text())
    assert profile["model"] == local["id"]
    assert profile["model_provider"] == start._CODEX_PROFILE
    assert profile["model_context_window"] == MODEL["context_length"]
    config = _parse_toml((tmp_path / "child" / "config.toml").read_text())
    assert config["model_providers"][start._CODEX_PROFILE]["base_url"] == f"{BASE}/v1"
    catalog = json.loads((tmp_path / "child" / profile["model_catalog_json"]).read_text())
    assert catalog["models"][0]["slug"] == local["id"]


def test_write_codex_parent_overlay_preserves_user_state_and_instructions(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "config.toml").write_text('model = "cloud-model"\n')
    (source / "auth.json").write_text('{"auth": "cloud"}\n')
    (source / "sessions").mkdir()
    (source / "AGENTS.override.md").write_text("Keep my existing instructions.\n")
    monkeypatch.setenv("CODEX_HOME", str(source))

    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "config.toml").read_text() == 'model = "cloud-model"\n'
    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    assert (overlay / "sessions").is_dir()
    instructions = (overlay / "AGENTS.override.md").read_text()
    assert instructions.startswith("Keep my existing instructions.\n")
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in instructions
    assert not (overlay / "AGENTS.md").exists()
    assert (overlay / "AGENTS.override.md").stat().st_mode & 0o077 == 0
    assert (source / "AGENTS.override.md").read_text() == "Keep my existing instructions.\n"


def test_write_codex_parent_overlay_refreshes_reused_entries(tmp_path, monkeypatch):
    first = tmp_path / "first-codex"
    first.mkdir()
    (first / "auth.json").write_text('{"auth": "old"}\n')
    (first / "old-only.toml").write_text("old\n")
    second = tmp_path / "second-codex"
    second.mkdir()
    (second / "auth.json").write_text('{"auth": "new"}\n')
    overlay_path = tmp_path / "managed" / "parent"

    monkeypatch.setenv("CODEX_HOME", str(first))
    overlay = start.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "old"}\n'
    assert (overlay / "old-only.toml").exists()

    monkeypatch.setenv("CODEX_HOME", str(second))
    overlay = start.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "new"}\n'
    assert not (overlay / "old-only.toml").exists()


def test_write_codex_parent_overlay_does_not_use_itself_as_source(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text('{"auth": "cloud"}\n')
    overlay_path = tmp_path / "managed" / "parent"
    monkeypatch.setenv("CODEX_HOME", str(source))
    overlay = start.write_codex_parent_overlay(overlay_path)

    monkeypatch.setenv("CODEX_HOME", str(overlay))
    overlay = start.write_codex_parent_overlay(overlay_path)

    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    manifest = json.loads((overlay / start._CODEX_PARENT_OVERLAY_MANIFEST).read_text())
    assert manifest["source_home"] == str(source)


def test_write_codex_parent_overlay_refreshes_fallback_copies(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    config = source / "config.toml"
    config.write_text('model = "first"\n')
    sessions = source / "sessions"
    sessions.mkdir()
    (sessions / "existing.jsonl").write_text("existing session\n")
    monkeypatch.setenv("CODEX_HOME", str(source))

    def deny_symlink(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    monkeypatch.setattr(start, "_create_directory_junction", lambda source, target: False)
    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")
    (overlay / "history.jsonl").write_text("session state\n")
    config.write_text('model = "second"\n')

    overlay = start.write_codex_parent_overlay(overlay)

    assert (overlay / "config.toml").read_text() == 'model = "second"\n'
    assert (overlay / "sessions" / "existing.jsonl").read_text() == "existing session\n"
    assert (overlay / "history.jsonl").read_text() == "session state\n"

    config.unlink()
    overlay = start.write_codex_parent_overlay(overlay)
    assert not (overlay / "config.toml").exists()
    assert (overlay / "history.jsonl").read_text() == "session state\n"


def test_create_directory_junction_uses_windows_mklink(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(start.os, "name", "nt")

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    source = tmp_path / "source"
    target = tmp_path / "target"

    assert start._create_directory_junction(source, target) is True
    assert captured["command"] == [
        "cmd.exe",
        "/d",
        "/c",
        "mklink",
        "/J",
        str(target),
        str(source),
    ]
    # Pin the needed values (and no shell) rather than whole-dict equality, which broke on #10192.
    kwargs = captured["kwargs"]
    for key, value in {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
        "check": False,
    }.items():
        assert kwargs[key] == value, f"{key} = {kwargs.get(key)!r}"
    assert not kwargs.get("shell", False), "mklink must not go through a shell"


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_write_codex_parent_overlay_uses_windows_home_for_windows_codex(tmp_path, monkeypatch):
    windows_profile = tmp_path / "windows-profile"
    source = windows_profile / ".codex"
    source.mkdir(parents = True)
    (source / "auth.json").write_text('{"auth": "windows"}\n')
    executable = "/mnt/c/Users/x/AppData/Roaming/npm/codex"
    monkeypatch.delenv("CODEX_HOME", raising = False)
    monkeypatch.delenv("USERPROFILE", raising = False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(start.shutil, "which", lambda _: executable)

    def check_output(command, **kwargs):
        if command[0] == "cmd.exe":
            assert kwargs["cwd"] == str(Path(executable).parent)
            return r"C:\Users\x" + "\n"
        assert command == ["wslpath", "-u", r"C:\Users\x"]
        return str(windows_profile) + "\n"

    monkeypatch.setattr(start.subprocess, "check_output", check_output)

    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "auth.json").read_text() == '{"auth": "windows"}\n'


def test_codex_parent_overlay_can_use_session_home(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text("{}\n")
    monkeypatch.setenv("CODEX_HOME", str(source))
    session_home = tmp_path / "session"

    overlay = start.write_codex_parent_overlay(session_home / "parent")

    assert overlay == session_home / "parent"
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in (overlay / "AGENTS.md").read_text()
    assert overlay.exists()


def test_ephemeral_codex_parent_overlay_is_cleaned_with_session(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source))
    agents_root = tmp_path / "agents"
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)

    with start._session_config("codex-subagent", launch = True) as session_home:
        overlay = start.write_codex_parent_overlay(session_home / "parent")
        assert overlay.exists()
        assert session_home.exists()

    assert not overlay.exists()
    assert not session_home.exists()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_codex_subagent_bridge_uses_wsl_for_windows_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex.exe",
    )
    flags = start._codex_subagent_flags(tmp_path / "subagent.json")
    prefix = f"mcp_servers.{start._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in flags if value.startswith(prefix))
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    assert server["command"] == "wsl.exe"
    assert server["args"] == [
        "-d",
        "Ubuntu",
        "--",
        sys.executable,
        "-c",
        server["args"][5],
        str(tmp_path / "subagent.json"),
    ]
    assert "sys.path.insert" in server["args"][5]
    assert f"from {start._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][5]
    assert server["required"] is True
    assert server["enabled_tools"] == [start._CODEX_SUBAGENT_MCP_TOOL]
    assert server["default_tools_approval_mode"] == "approve"
    assert not any(value.startswith("developer_instructions=") for value in flags)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_agent_config_path_translates_for_windows_agent(monkeypatch, tmp_path):
    windows_path = r"\\wsl.localhost\Ubuntu\tmp\unsloth.toml"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex",
    )
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: windows_path)

    assert start._agent_config_path(tmp_path / "unsloth.toml", ["codex"]) == windows_path


def test_subagent_model_id_preserves_explicit_variant(monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *args, **kwargs: pytest.fail("explicit variant should not need status"),
    )
    assert (
        start._subagent_model_id(BASE, "key", MODEL, MODEL["id"], "UD-Q4_K_XL")
        == MODEL["id"] + ":UD-Q4_K_XL"
    )


def test_subagent_model_id_uses_loaded_variant(monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *args, **kwargs: {"is_gguf": True, "gguf_variant": "Q5_K_M"},
    )
    assert start._subagent_model_id(BASE, "key", MODEL, None, None) == MODEL["id"] + ":Q5_K_M"


def test_subagent_model_id_warns_when_status_unavailable(monkeypatch, capsys):
    def raise_error(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(start, "_http_json", raise_error)
    assert start._subagent_model_id(BASE, "key", MODEL, None, None) == MODEL["id"]
    assert "could not verify the loaded GGUF variant" in capsys.readouterr().err


@pytest.mark.parametrize("agent", ["openclaw", "hermes", "dsh", "vibe"])
@pytest.mark.parametrize("flag", ["--as-subagent", "--as-subagent=true", "--as-subagent=false"])
def test_unsupported_agents_reject_as_subagent(agent, flag):
    result = CliRunner().invoke(start.start_app, [agent, flag])
    assert result.exit_code == 1
    assert f"--as-subagent is not supported for {agent}." in result.output


@pytest.mark.parametrize(
    "agent", ["claude", "codex", "openclaw", "opencode", "hermes", "pi", "dsh", "vibe"]
)
def test_launch_preflights_agent_before_connect(agent, monkeypatch):
    events = []
    if agent == "opencode":
        monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))

    def require(name, hint, launch):
        assert name == agent
        assert hint
        assert launch is True
        events.append("agent")

    def connect(*args, **kwargs):
        events.append("connect")
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(start, "_require_agent_for_launch", require)
    monkeypatch.setattr(start, "_connect", connect)

    result = CliRunner().invoke(start.start_app, [agent])

    assert result.exit_code == 1
    assert events == ["agent", "connect"]


def test_dsh_rejects_an_unrelated_executable_before_connect(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: False)
    monkeypatch.setattr(start, "_install_agent", lambda *_: None)
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail("the wrong dsh must be rejected before connection"),
    )

    result = CliRunner().invoke(start.start_app, ["dsh"])

    assert result.exit_code == 1
    assert "`/usr/bin/dsh` is not DeepSeek Harness" in result.output


def test_dsh_resolver_searches_past_an_unrelated_earlier_path_entry(monkeypatch, tmp_path):
    shadow_dir = tmp_path / "system-bin"
    harness_dir = tmp_path / "user-bin"
    shadow_dir.mkdir()
    harness_dir.mkdir()
    suffix = ".cmd" if os.name == "nt" else ""
    for directory in (shadow_dir, harness_dir):
        executable = directory / f"dsh{suffix}"
        executable.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n")
        if os.name != "nt":
            executable.chmod(0o755)

    monkeypatch.setenv("PATH", os.pathsep.join((str(shadow_dir), str(harness_dir))))
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path / "missing-home")
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)
    monkeypatch.setattr(
        start,
        "is_deepseek_harness_executable",
        lambda executable: Path(executable).parent == harness_dir,
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *_: pytest.fail("an existing later Harness must be used without reinstalling"),
    )

    resolved = start._resolve_or_install_agent(
        "dsh",
        "npm install -g @deepseek-ai/dsh",
        start._which_with_install_dirs,
    )

    assert Path(resolved).parent == harness_dir


def test_declined_opencode_subagent_install_stops_before_connect(monkeypatch):
    installs = []
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda name, hint: installs.append((name, hint)),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *a, **k: pytest.fail("declined install must stop before model connection"),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 1
    assert len(installs) == 1
    assert installs[0][0] == "opencode"


@pytest.mark.parametrize(
    "agent", ["claude", "codex", "openclaw", "opencode", "hermes", "pi", "dsh", "vibe"]
)
def test_noninteractive_missing_agent_stops_before_connect(agent, monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("non-interactive launch must not install"),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail("missing agent must stop before connection"),
    )

    result = CliRunner().invoke(start.start_app, [agent])

    assert result.exit_code == 1
    assert f"`{agent}` not found on PATH" in result.output


@pytest.mark.parametrize("agent", ["claude", "codex", "openclaw", "hermes", "pi", "dsh", "vibe"])
def test_no_launch_skips_agent_resolution(agent, monkeypatch):
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda _: pytest.fail("--no-launch must not resolve an agent"),
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *args: pytest.fail("--no-launch must not install an agent"),
    )

    def stop_at_connect(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(start, "_connect", stop_at_connect)

    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


def test_opencode_no_launch_resolves_generation_without_installing(monkeypatch):
    resolved = []
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: resolved.append(name)
        or ("/home/me/.opencode/bin/opencode2" if name == "opencode2" else None),
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *args: pytest.fail("--no-launch must not install an agent"),
    )
    monkeypatch.setattr(
        start, "_connect", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError)
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert resolved == ["opencode2"]


def test_missing_pi_subagent_extension_fails_before_install_or_connect(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_PI_SUBAGENT_EXTENSION", tmp_path / "missing.ts")
    monkeypatch.setattr(
        start,
        "_require_agent_for_launch",
        lambda *args: pytest.fail("local prerequisites must be checked before installation"),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail(
            "local prerequisites must be checked before connection"
        ),
    )

    result = CliRunner().invoke(start.start_app, ["pi", "--as-subagent"])

    assert result.exit_code == 1
    assert "Missing Pi subagent extension" in result.output


@pytest.fixture()
def fake_studio(tmp_path, monkeypatch):
    calls = []
    state = {"models": [MODEL]}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url, payload))
        if url.endswith("/api/inference/loaded-models"):
            return {"object": "list", "data": state["models"]}
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "model_identifier": state["models"][0]["id"]}
        if url.endswith("/api/auth/api-keys"):
            if method == "GET":
                return {"api_keys": [{"key_prefix": "feedface"}]}
            return {"key": "sk-unsloth-feedfacefeedface"}
        if url.endswith("/api/settings/embedding-model"):
            return {"embedding_model": "unsloth/bge-small-en-v1.5"}
        if url.endswith("/api/inference/load"):
            already_loaded = state["models"][0]["id"] == payload["model_path"]
            state["models"] = [{"id": payload["model_path"], "context_length": 4096}]
            return {
                "status": "already_loaded" if already_loaded else "loaded",
                "model": payload["model_path"],
                "display_name": payload["model_path"],
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(start, "_studio_token", lambda: "jwt-token")
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_require_agent_for_launch", lambda *args: None)
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    # No `claude` on PATH, so _claude_flags never probes the real binary.
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    return calls


def test_connect_claude_no_launch(fake_studio):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    for name in start._CLAUDE_ENV_UNSET:
        _assert_env_unset(result.output, name)
    _assert_env_set(result.output, "ANTHROPIC_BASE_URL", BASE)
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    _assert_env_set(result.output, "CLAUDE_CODE_NO_FLICKER", "1")
    _assert_env_set(result.output, "CLAUDE_CODE_ATTRIBUTION_HEADER", "0")
    _assert_env_set(result.output, "CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off")
    _assert_env_set(result.output, "CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(MODEL["context_length"]))
    _assert_env_set(result.output, "CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(MODEL["context_length"]))
    _assert_env_set(result.output, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "90")
    assert f"claude --model {MODEL['id']} --exclude-dynamic-system-prompt-sections" in result.output
    command = _launch_command(result.output)
    settings_path = Path(command[command.index("--settings") + 1])
    settings = json.loads(settings_path.read_text())
    assert settings["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    for name in start._CLAUDE_ENV_UNSET:
        assert settings["env"][name] == ""
    if os.name != "nt":
        assert settings_path.stat().st_mode & 0o777 == 0o600
    assert "--plugin-dir" not in command
    assert ".claude/settings.json" not in result.output


def test_connect_prints_the_running_models_load_warning(fake_studio, monkeypatch):
    notice = (
        "Not enough disk space to download BF16 (7.5 GB needed, 7.5 GB free), "
        "so Q4_1 (2.4 GB) was loaded instead."
    )
    http_json = start._http_json

    def with_warning(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "model_identifier": MODEL["id"], "memory_warning": notice}
        return http_json(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", with_warning)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", MODEL["id"]])

    assert result.exit_code == 0, result.output
    assert f"Warning: {notice}" in result.stderr
    assert f"Warning: {notice}" not in result.stdout


def test_connect_skips_the_load_warning_of_another_active_model(fake_studio, monkeypatch):
    http_json = start._http_json

    def other_model_warns(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {
                "is_gguf": True,
                "active_model": "unsloth/Other-GGUF",
                "model_identifier": "unsloth/Other-GGUF",
                "memory_warning": "Not enough disk space to download BF16, so Q4_1 was loaded instead.",
            }
        return http_json(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", other_model_warns)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", MODEL["id"]])

    assert result.exit_code == 0, result.output
    assert "Warning: Not enough disk space" not in result.output


def test_connect_claude_session_settings_follow_forwarded_settings(fake_studio):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--settings", forwarded],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    positions = [index for index, arg in enumerate(command) if arg == "--settings"]
    assert len(positions) == 2
    assert command[positions[0] + 1] == forwarded
    assert Path(command[positions[1] + 1]).name.startswith("settings-")


@pytest.mark.parametrize(
    "settings_arg",
    [
        lambda value: ["--settings", value],
        lambda value: [f"--settings={value}"],
    ],
)
def test_connect_claude_session_settings_precede_subcommand(fake_studio, settings_arg):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "mcp", "list", *settings_arg(forwarded)],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    subcommand = command.index("mcp")
    assert command.index("--model") < subcommand
    settings_positions = [
        index
        for index, arg in enumerate(command)
        if arg == "--settings" or arg.startswith("--settings=")
    ]
    assert len(settings_positions) == 2
    assert settings_positions[0] < settings_positions[1] < subcommand


def test_connect_claude_session_settings_precede_forwarded_delimiter(fake_studio):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--", "--settings", forwarded],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    positions = [index for index, arg in enumerate(command) if arg == "--settings"]
    assert len(positions) == 2
    assert positions[0] < command.index("--") < positions[1]
    assert Path(command[positions[0] + 1]).name.startswith("settings-")


def test_connect_claude_as_subagent_preserves_cloud_parent(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"] + ":UD-Q4_K_XL",
            "hello",
        ],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    plugin = tmp_path / "agents" / "claude-subagent" / "unsloth-local-agent"
    assert command == [
        "claude",
        "--plugin-dir",
        str(plugin),
        f"--allowedTools={start._CLAUDE_SUBAGENT_TOOL},{start._CLAUDE_SUBAGENT_PLAN_TOOL}",
        "hello",
    ]
    assert "--model" not in command
    parent_base = "$env:ANTHROPIC_BASE_URL" if os.name == "nt" else "export ANTHROPIC_BASE_URL="
    parent_token = (
        "$env:ANTHROPIC_AUTH_TOKEN" if os.name == "nt" else "export ANTHROPIC_AUTH_TOKEN="
    )
    assert parent_base not in result.output
    assert parent_token not in result.output
    assert "unset ANTHROPIC_API_KEY" not in result.output
    assert "UNSLOTH_CLAUDE_SUBAGENT_API_KEY" not in result.output
    assert "sk-unsloth-feedfacefeedface" not in result.output
    assert json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())["name"] == (
        "unsloth-local-agent"
    )
    mcp = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["unsloth"]
    settings_path = next(plugin.glob("settings-*.json"))
    assert mcp["command"] == sys.executable
    assert mcp["args"] == ["-m", start._CLAUDE_SUBAGENT_MCP_MODULE]
    assert mcp["env"] == {
        "UNSLOTH_CLAUDE_SUBAGENT_BASE_URL": BASE,
        "UNSLOTH_CLAUDE_SUBAGENT_API_KEY": "sk-unsloth-feedfacefeedface",
        "UNSLOTH_CLAUDE_SUBAGENT_MODEL": MODEL["id"] + ":UD-Q4_K_XL",
        "UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS": "0",
        "UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW": "4096",
        start._CLAUDE_SUBAGENT_SETTINGS_ENV: str(settings_path),
    }
    settings = json.loads(settings_path.read_text())
    assert settings["availableModels"] == [MODEL["id"] + ":UD-Q4_K_XL"]
    assert settings["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    for name in start._CLAUDE_ENV_UNSET:
        assert settings["env"][name] == ""
    skill = (plugin / "skills" / "local-agent" / "SKILL.md").read_text()
    assert "spawn an Unsloth agent or local agent" in skill
    assert "In plan mode" in skill
    assert "Ask Claude to spawn an Unsloth or local agent." in result.output


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_claude_subagent_plugin_uses_wsl_for_windows_claude(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("WSLENV", "EXISTING")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Local/Programs/claude.exe",
    )
    server_env = {
        "UNSLOTH_CLAUDE_SUBAGENT_BASE_URL": BASE,
        "UNSLOTH_CLAUDE_SUBAGENT_API_KEY": "secret",
        "UNSLOTH_CLAUDE_SUBAGENT_MODEL": MODEL["id"],
    }
    plugin = start.write_claude_subagent_plugin(tmp_path, server_env)
    mcp = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["unsloth"]
    settings_path = next(plugin.glob("settings-*.json"))
    assert mcp["command"] == "wsl.exe"
    assert mcp["args"] == [
        "-d",
        "Ubuntu",
        "--",
        sys.executable,
        "-m",
        start._CLAUDE_SUBAGENT_MCP_MODULE,
    ]
    assert mcp["env"]["UNSLOTH_CLAUDE_SUBAGENT_API_KEY"] == "secret"
    assert mcp["env"][start._CLAUDE_SUBAGENT_SETTINGS_ENV] == str(settings_path)
    assert mcp["env"]["WSLENV"].split(":") == [
        "EXISTING",
        "UNSLOTH_CLAUDE_SUBAGENT_BASE_URL",
        "UNSLOTH_CLAUDE_SUBAGENT_API_KEY",
        "UNSLOTH_CLAUDE_SUBAGENT_MODEL",
        start._CLAUDE_SUBAGENT_SETTINGS_ENV,
    ]


def test_connect_claude_compact_window_omitted_without_context(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_resolve_model", lambda *a, **k: {"id": "local-model"})
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in result.output
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in result.output
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result.output


def test_launch_native_posix_child_gets_current_pwd(fake_studio, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)

    result = CliRunner().invoke(start.start_app, ["opencode"])

    assert result.exit_code == 0, result.output
    assert captured["command"][0] == "/usr/local/bin/opencode"
    if os.name != "nt":
        assert captured["env"]["PWD"] == os.getcwd()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX exec signal semantics")
def test_launch_leaves_child_able_to_handle_sigint(monkeypatch, tmp_path):
    # SIG_IGN reached the agent too, so hermes could never be interrupted.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import signal, sys\n"
        "sys.exit(17 if signal.getsignal(signal.SIGINT) == signal.SIG_IGN else 0)\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(start, "_augment_path_with_install_dirs", lambda: None)
    before = signal.getsignal(signal.SIGINT)

    code = start._launch([sys.executable, str(probe)], {}, install_hint = "n/a")

    assert code == 0, "child saw SIG_IGN and could never be interrupted"
    assert signal.getsignal(signal.SIGINT) is before


def test_connect_claude_launch_scrubs_conflicting_auth_env(fake_studio, monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
    monkeypatch.setenv("ANTHROPIC_UNIX_SOCKET", "/tmp/remote-claude.sock")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "https://corporate-gateway.azure-api.net/anthropic-stream",
    )
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-foundry-resource")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_ANTHROPIC_AWS", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_MANTLE", "1")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert captured["command"] == ["/usr/local/bin/claude", "--model", MODEL["id"]]
    for name in start._CLAUDE_ENV_UNSET:
        assert name not in captured["env"]
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert captured["env"]["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_windows_shim_from_wsl_bridges_env(fake_studio, monkeypatch, tmp_path):
    captured = {}
    windows_settings = r"C:\\Users\\samle\\AppData\\Local\\unsloth\\settings.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )
    monkeypatch.setattr(start, "_wsl_windows_path", lambda _: windows_settings)
    monkeypatch.setattr(
        start,
        "_claude_flags",
        lambda model_id, settings: ["--settings", settings],
    )

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert captured["command"] == [
        "/mnt/c/Users/samle/AppData/Roaming/npm/claude",
        "--model",
        MODEL["id"],
        "--settings",
        windows_settings,
    ]
    for name in start._CLAUDE_ENV_UNSET:
        assert captured["env"][name] == ""
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["PWD"] == str(tmp_path)

    assert "PWD/p" in captured["env"]["WSLENV"].split(":")
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        *start._CLAUDE_ENV_UNSET,
    ):
        assert name in captured["env"]["WSLENV"].split(":")


def _npm_node_cmd_shim(
    target: str,
    *,
    node_args: str = "",
    environment: tuple[tuple[str, str], ...] = (),
    cmd_shim_version: int = 7,
) -> str:
    environment_lines = "".join(f"@SET {name}={value}\r\n" for name, value in environment)
    legacy_pathext = "  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n" if cmd_shim_version < 9 else ""
    current_pathext = "set PATHEXT=%PATHEXT:;.JS;=;% & " if cmd_shim_version >= 9 else ""
    return (
        "@ECHO off\r\n"
        "GOTO start\r\n"
        ":find_dp0\r\n"
        "SET dp0=%~dp0\r\n"
        "EXIT /b\r\n"
        ":start\r\n"
        "SETLOCAL\r\n"
        "CALL :find_dp0\r\n"
        f"{environment_lines}"
        "\r\n"
        'IF EXIST "%dp0%\\node.exe" (\r\n'
        '  SET "_prog=%dp0%\\node.exe"\r\n'
        ") ELSE (\r\n"
        '  SET "_prog=node"\r\n'
        f"{legacy_pathext}"
        ")\r\n"
        "\r\n"
        f"endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & {current_pathext}"
        f'"%_prog%" {node_args} "%dp0%\\{target}" %*\r\n'
    )


def test_launch_windows_npm_shim_preserves_multiline_argument(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    captured = {}

    def which(name):
        return str(cmd) if name == "fake-agent" else r"C:\Program Files\nodejs\node.exe"

    def run(command, env):
        captured["command"] = command
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.shutil, "which", which)
    monkeypatch.setattr(start.subprocess, "run", run)

    code = start._launch(
        ["fake-agent", 'first line\nsecond "quoted" line'],
        {},
        install_hint = "unused",
    )

    assert code == 0
    assert captured["command"] == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        'first line\nsecond "quoted" line',
    ]


def test_resolved_launch_command_handles_current_npm_node_shim(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(
        _npm_node_cmd_shim(
            r"node_modules\fake-agent\index.js",
            cmd_shim_version = 9,
        ).encode()
    )
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_ignores_node_js_pathext_shadow(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    resolved_names = []

    def which(name):
        resolved_names.append(name)
        return {
            "node": r"C:\shadow\node.js",
            "node.exe": r"C:\Program Files\nodejs\node.exe",
        }.get(name)

    monkeypatch.setattr(start.shutil, "which", which)

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]
    assert resolved_names == ["node.exe"]


def test_resolved_launch_command_accepts_extensionless_node_target(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "cli"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\cli").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_prefers_cmd_sibling_of_extensionless_shim(monkeypatch, tmp_path):
    # which() can resolve the extensionless POSIX shim, which CreateProcess rejects (#9167).
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_keeps_extensionless_shim_without_sibling(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.write_text("#!/bin/sh\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_prefer_cmd_sibling_leaves_an_unreadable_resolution_alone(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.mkdir()
    (tmp_path / "fake-agent.cmd").write_text("@ECHO off\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_prefer_cmd_sibling_is_none_safe_and_posix_noop(monkeypatch, tmp_path):
    assert start._prefer_windows_cmd_sibling(None) is None
    # Pin os.name: on a Windows runner the rescue would fire.
    monkeypatch.setattr(start.os, "name", "posix")
    shim = tmp_path / "fake-agent"
    shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    (tmp_path / "fake-agent.cmd").write_text("@ECHO off\n", encoding = "utf-8")
    assert start._prefer_windows_cmd_sibling(str(shim)) == str(shim)


def test_resolved_launch_command_rescues_uppercase_cmd_sibling(monkeypatch, tmp_path):
    # Compare identity: a case-insensitive volume answers the .cmd probe with the same file.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.CMD"
    cmd.write_text("@ECHO off\ncustom-wrapper %*\n", encoding = "utf-8")

    resolved = start._resolved_launch_command(str(posix_shim), ["--flag"])
    assert resolved[1:] == ["--flag"]
    assert os.path.samefile(resolved[0], str(cmd))


def test_which_with_install_dirs_applies_the_cmd_sibling_preference(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    cmd.write_text("@ECHO off\n", encoding = "utf-8")
    monkeypatch.setattr(start, "_augment_path_with_install_dirs", lambda: None)
    monkeypatch.setattr(start.shutil, "which", lambda name: str(posix_shim))

    assert start._which_with_install_dirs("fake-agent") == str(cmd)


def test_resolved_launch_command_rescues_dotted_bin_name_shim(monkeypatch, tmp_path):
    # cmd-shim appends .cmd to the whole bin name.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake.agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake.agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_keeps_extensionless_pe_binary_over_stale_sibling(
    monkeypatch, tmp_path
):
    # CreateProcess runs a PE regardless of its name; only shebang files are shims.
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.write_bytes(b"MZ\x90\x00")
    stale = tmp_path / "fake-agent.cmd"
    stale.write_text("@ECHO off\nold-wrapper %*\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_resolved_launch_command_falls_through_to_non_npm_cmd_sibling(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    cmd.write_text("@ECHO off\ncustom-wrapper %*\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [str(cmd), "--flag"]


def test_launch_windows_npm_shim_preserves_shebang_args_and_environment(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text(
        "#!/usr/bin/env NODE_OPTIONS=--trace-warnings node --no-warnings\n",
        encoding = "utf-8",
    )
    cmd.write_bytes(
        _npm_node_cmd_shim(
            r"node_modules\fake-agent\index.js",
            node_args = "--no-warnings",
            environment = (("NODE_OPTIONS", "--trace-warnings"),),
        ).encode()
    )
    captured = {}

    def which(name):
        return str(cmd) if name == "fake-agent" else r"C:\Program Files\nodejs\node.exe"

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.shutil, "which", which)
    monkeypatch.setattr(start.subprocess, "run", run)

    code = start._launch(
        ["fake-agent", 'first line\nsecond "quoted" line'],
        {},
        install_hint = "unused",
    )

    assert code == 0
    assert captured["command"] == [
        r"C:\Program Files\nodejs\node.exe",
        "--no-warnings",
        str(target),
        'first line\nsecond "quoted" line',
    ]
    assert captured["env"]["NODE_OPTIONS"] == "--trace-warnings"


def test_resolved_launch_command_uses_native_npm_entrypoint(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "agent.exe"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"")
    cmd.write_bytes(
        (
            "@ECHO off\r\n"
            "GOTO start\r\n"
            ":find_dp0\r\n"
            "SET dp0=%~dp0\r\n"
            "EXIT /b\r\n"
            ":start\r\n"
            "SETLOCAL\r\n"
            "CALL :find_dp0\r\n"
            '"%dp0%\\node_modules\\fake-agent\\agent.exe" %*\r\n'
        ).encode()
    )

    assert start._resolved_launch_command(str(cmd), ["--flag", "two words"]) == [
        str(target),
        "--flag",
        "two words",
    ]


def test_resolved_launch_command_handles_project_local_npm_shim(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    node_modules = tmp_path / "project" / "node_modules"
    cmd = node_modules / ".bin" / "fake-agent.cmd"
    target = node_modules / "fake-agent" / "index.js"
    cmd.parent.mkdir(parents = True)
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"..\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_leaves_custom_npm_like_wrapper_unchanged(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "custom-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    contents = _npm_node_cmd_shim(r"node_modules\fake-agent\index.js")
    cmd.write_bytes(
        contents.replace(
            "CALL :find_dp0\r\n", "CALL :find_dp0\r\nSET AGENT_MODE=custom\r\n"
        ).encode()
    )

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]


def test_resolved_launch_command_leaves_non_npm_batch_file_unchanged(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "custom-agent.cmd"
    cmd.write_bytes(b'@echo off\r\n"%dp0%\\custom.exe" %*\r\n')

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_no_launch_windows_shim_from_wsl_prints_wslenv(
    fake_studio, monkeypatch, tmp_path
):
    windows_settings = r"C:\\Users\\samle\\AppData\\Local\\unsloth\\settings.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )
    monkeypatch.setattr(start, "_wsl_windows_path", lambda _: windows_settings)

    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])

    assert result.exit_code == 0, result.output
    for name in start._CLAUDE_ENV_UNSET:
        assert f"export {name}=" in result.output
    assert "export WSLENV=" in result.output
    # PWD must not be frozen into the recipe: WSLENV PWD/p translates the live PWD.
    assert "export PWD=" not in result.output
    assert "PWD/p" in result.output
    assert "ANTHROPIC_AUTH_TOKEN" in result.output
    assert "CLAUDE_CODE_OAUTH_TOKEN" in result.output
    command = _launch_command(result.output)
    assert command[command.index("--settings") + 1] == windows_settings


def test_connect_codex_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    for name in start._CODEX_ENV_UNSET:
        _assert_env_unset(result.output, name)
    _assert_env_set(result.output, "UNSLOTH_STUDIO_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    assert "codex --oss --profile unsloth_api" in result.output
    home = tmp_path / "agents" / "codex"
    _assert_env_set(result.output, "CODEX_HOME", str(home))
    assert (home / "config.toml").exists()
    assert (home / "unsloth_api.config.toml").exists()


def test_connect_codex_as_subagent_preserves_cloud_parent(fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    source_home = tmp_path / "user-codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model = "cloud-model"\n')
    (source_home / "AGENTS.md").write_text("Keep the user's guidance.\n")
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    result = CliRunner().invoke(
        start.start_app,
        [
            "codex",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"] + ":UD-Q4_K_XL",
        ],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[0] == "codex"
    assert "--oss" not in command
    assert "--profile" not in command
    assert "--model" not in command
    parent_home = tmp_path / "agents" / "codex-subagent" / "parent"
    _assert_env_set(result.output, "CODEX_HOME", str(parent_home))
    for name in start._CODEX_ENV_UNSET:
        _assert_env_kept(result.output, name)
    assert start._CODEX_ENV_KEY not in result.output
    assert "sk-unsloth-feedfacefeedface" not in result.output
    home = tmp_path / "agents" / "codex-subagent"
    bridge_path = home / "subagent.json"
    bridge = json.loads(bridge_path.read_text())
    assert bridge["api_key"] == "sk-unsloth-feedfacefeedface"
    assert bridge["codex_home"] == str(home / "child")
    assert bridge["bypass_permissions"] is False
    profile = _parse_toml((home / "child" / "unsloth_api.config.toml").read_text())
    assert profile["model"] == MODEL["id"] + ":UD-Q4_K_XL"
    prefix = f"mcp_servers.{start._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in command if value.startswith(prefix))
    assert override.startswith(prefix)
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-c", server["args"][1], str(bridge_path)]
    assert "sys.path.insert" in server["args"][1]
    assert f"from {start._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][1]
    assert server["enabled_tools"] == [start._CODEX_SUBAGENT_MCP_TOOL]
    assert not any(value.startswith("developer_instructions=") for value in command)
    parent_instructions = (parent_home / "AGENTS.md").read_text()
    assert parent_instructions.startswith("Keep the user's guidance.\n")
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in parent_instructions
    assert "Ask Codex to spawn an Unsloth or local agent." in result.output


def test_connect_codex_matches_requested_model_case_insensitively(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        [
            "codex",
            "--no-launch",
            "--model",
            "unsloth/gemma-4-26b-a4b-it-gguf",
        ],
    )
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "codex"
    profile = _parse_toml((home / "unsloth_api.config.toml").read_text())
    assert profile["model"] == MODEL["id"]


def test_resolve_model_matches_loaded_canonical_case_after_load(monkeypatch, capsys):
    calls = []
    state = {"loaded": False}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url, payload))
        if url.endswith("/api/inference/loaded-models"):
            return {
                "data": [
                    {
                        "id": "unsloth/gemma-4-E2B-it-GGUF" if state["loaded"] else "other/model",
                        "context_length": 131072,
                    }
                ]
            }
        if url.endswith("/api/inference/load"):
            state["loaded"] = True
            return {"model": "unsloth/gemma-4-E2B-it-GGUF"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model(
        BASE,
        "sk-test",
        "unsloth/gemma-4-e2b-it-gguf",
        start.LoadOptions(gguf_variant = "UD-Q4_K_XL"),
    )

    assert entry["id"] == "unsloth/gemma-4-E2B-it-GGUF"
    assert any(c[1].endswith("/api/inference/load") for c in calls)
    output = capsys.readouterr().out
    assert "please wait" not in output


def test_resolve_model_matches_snapshot_path_by_public_id(monkeypatch):
    """A GGUF loaded by snapshot path is advertised by its basename, not the path."""
    snapshot = "/home/u/.cache/legacy/models--Org--Model/snapshots/abc123"
    state = {"loaded": False}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models"):
            return {"data": [{"id": "abc123"}] if state["loaded"] else []}
        if url.endswith("/api/inference/load"):
            state["loaded"] = True
            return {"model": snapshot, "display_name": snapshot}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model(BASE, "sk-test", snapshot, start.LoadOptions())

    assert entry["id"] == "abc123"


def test_subagent_model_id_warns_when_a_path_load_cannot_pin_the_quant(capsys):
    """A path is advertised as a bare basename, so the quant cannot be recorded."""
    model_id = start._subagent_model_id(BASE, "sk-test", {"id": "abc123"}, None, "UD-Q4_K_XL")

    assert model_id == "abc123"
    assert "cannot pin the UD-Q4_K_XL quant" in capsys.readouterr().err


def test_subagent_model_id_pins_the_quant_for_repo_ids(capsys):
    model_id = start._subagent_model_id(
        BASE, "sk-test", {"id": "unsloth/gemma-4-E4B-it-GGUF"}, None, "UD-Q4_K_XL"
    )

    assert model_id == "unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"
    assert capsys.readouterr().err == ""


def test_public_model_id_leaves_repo_ids_alone():
    """Only a path gets reduced; a repo id must not match some unrelated model.

    Relative and multi-segment paths are covered too: _looks_like_path is defined
    twice in this module (the WSLENV one wins), so this must use its own classifier.
    """
    assert start._public_model_id("unsloth/gemma-4-E4B-it-GGUF") is None
    assert start._public_model_id("org/model") is None
    assert start._public_model_id("/srv/models/Qwen3-Q4_K_M.gguf") == "Qwen3-Q4_K_M"
    assert start._public_model_id("/a/b/snapshots/rev1") == "rev1"
    assert start._public_model_id("./models/foo") == "foo"
    assert start._public_model_id("cache/snapshots/rev") == "rev"
    assert start._public_model_id("a/b/c") == "c"


def test_resolve_model_loads_when_catalog_hit_is_not_loaded(monkeypatch):
    calls = []
    state = {"loaded": False}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url))
        if url.endswith("/api/inference/loaded-models"):
            return {
                "data": [
                    {
                        "id": "unsloth/Gemma-4-GGUF",
                        "loaded": state["loaded"],
                        "context_length": 131072,
                    }
                ]
            }
        if url.endswith("/api/inference/load"):
            state["loaded"] = True
            return {"model": "unsloth/Gemma-4-GGUF"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model(BASE, "sk-test", "unsloth/gemma-4-gguf")

    assert entry["id"] == "unsloth/Gemma-4-GGUF"
    assert any(u.endswith("/api/inference/load") for _, u in calls)


def test_resolve_model_does_not_attach_if_catalog_stays_unloaded(monkeypatch):
    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models"):
            return {
                "data": [
                    {
                        "id": "unsloth/Gemma-4-GGUF",
                        "loaded": False,
                        "context_length": 131072,
                    }
                ]
            }
        if url.endswith("/api/inference/load"):
            return {"status": "loaded", "model": "unsloth/Gemma-4-GGUF"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    with pytest.raises(typer.Exit):
        start._resolve_model(BASE, "sk-test", "unsloth/gemma-4-gguf")


def test_resolve_model_attaches_to_loaded_catalog_hit_without_reload(monkeypatch):
    calls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url))
        if url.endswith("/api/inference/loaded-models"):
            return {
                "data": [{"id": "unsloth/Gemma-4-GGUF", "loaded": True, "context_length": 131072}]
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model(BASE, "sk-test", "unsloth/gemma-4-gguf")

    assert entry["id"] == "unsloth/Gemma-4-GGUF"
    assert not any(u.endswith("/api/inference/load") for _, u in calls)


def test_resolve_model_without_request_rejects_unloaded_catalog(monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: {
            "data": [
                {
                    "id": "unsloth/Gemma-4-GGUF",
                    "loaded": False,
                    "context_length": 131072,
                }
            ]
        },
    )

    with pytest.raises(typer.Exit):
        start._resolve_model(BASE, "sk-test", None)


def test_resolve_model_remote_studio_does_not_casefold_attach(monkeypatch):
    # Remotely the existence probe cannot see server paths, so a case variant must not attach.
    calls = []
    state = {"loaded": False}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url))
        if url.endswith("/api/inference/loaded-models"):
            return {
                "data": [{"id": "unsloth/Gemma-4-GGUF", "loaded": True, "context_length": 131072}]
            }
        if url.endswith("/api/inference/load"):
            state["loaded"] = True
            return {"model": "unsloth/Gemma-4-GGUF"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model("http://10.0.0.5:8888", "sk-test", "unsloth/gemma-4-gguf")

    assert entry["id"] == "unsloth/Gemma-4-GGUF"
    assert any(u.endswith("/api/inference/load") for _, u in calls)


def test_model_id_matching_does_not_casefold_local_paths(tmp_path):
    existing_local = tmp_path / "Org" / "Foo"
    existing_local.mkdir(parents = True)

    assert start._model_id_matches("Org/Foo", "org/foo")
    assert not start._model_id_matches(str(existing_local), str(existing_local).lower())
    assert not start._model_id_matches("./Models/Foo", "./models/foo")
    assert not start._model_id_matches(r".\Models\Foo", r".\models\foo")
    assert not start._is_hub_model_id("models/Llama/Foo.gguf")
    assert not start._model_id_matches("models/Llama/Foo.gguf", "models/llama/foo.gguf")
    assert start._is_hub_model_id("unsloth/Gemma-3-4b-it-GGUF")
    assert start._model_id_matches("unsloth/Gemma-3-4b-it-GGUF", "unsloth/gemma-3-4b-it-gguf")
    assert not start._model_id_matches(
        "unsloth/Gemma-3-4b-it-GGUF", "unsloth/gemma-3-4b-it-gguf", allow_casefold = False
    )
    assert start._model_id_matches("unsloth/Foo", "unsloth/Foo", allow_casefold = False)


def test_connect_codex_launch_uses_ephemeral_home(fake_studio, monkeypatch):
    captured = {}
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")

    def run(command, env):
        captured["home"] = env["CODEX_HOME"]
        captured["config_present"] = (Path(env["CODEX_HOME"]) / "config.toml").exists()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["codex"])
    assert result.exit_code == 0, result.output
    home = Path(captured["home"])
    assert captured["config_present"]
    parent = start._ephemeral_session_parent("codex")
    assert home.name.startswith(start._ephemeral_session_prefix("codex", parent))
    assert not home.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason = "the #6547 CI parser is bash-only; on Windows --no-launch prints PowerShell",
)
def test_no_launch_output_is_parseable(fake_studio):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    skip = ("export ", "unset ", "Unsloth ", "Updated ", "Disabled ", "Warning", "Loading")
    body = [ln for ln in lines if not ln.startswith(skip)]
    assert "codex --oss --profile unsloth_api" in body[-1]
    assert any(ln.startswith("export CODEX_HOME=") for ln in lines)


def test_no_launch_last_line_is_self_contained(fake_studio, tmp_path):
    # People copy just the last line, so it must inline every session env var.
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    assignments = {}
    command = []
    for i, part in enumerate(parts):
        if "=" not in part:
            command = parts[i:]
            break
        name, _, value = part.partition("=")
        assignments[name] = value
    assert command and command[0] == "codex"
    assert assignments["CODEX_HOME"] == str(tmp_path / "agents" / "codex")
    assert assignments["UNSLOTH_STUDIO_AUTH_TOKEN"].startswith("sk-unsloth-")


def test_no_launch_claude_last_line_blanks_conflicting_auth(fake_studio):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    for name in start._CLAUDE_ENV_UNSET:
        assert f"{name}= " in last
    assert "ANTHROPIC_AUTH_TOKEN=" in last


def test_opencode_inline_config_beats_project_config(fake_studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--yolo"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert inline["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }
    assert "sk-unsloth" not in result.output


def test_opencode_inline_config_omits_permission_without_yolo(fake_studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "permission" not in inline


def test_https_loopback_never_auto_serves(fake_studio, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "https://127.0.0.1:8443")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {"called": False}
    monkeypatch.setattr(
        start, "_start_studio_server", lambda *a, **k: started.__setitem__("called", True)
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF"])
    assert result.exit_code == 1
    assert "No running Unsloth server" in result.output
    assert started["called"] is False


def test_connect_alias_still_works(fake_studio):
    from unsloth_cli import app

    result = CliRunner().invoke(app, ["connect", "claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])


def test_connect_key_minted_once_then_cached(fake_studio, tmp_path):
    CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    mints = [c for c in fake_studio if c[0] == "POST" and c[1].endswith("/api/auth/api-keys")]
    assert len(mints) == 1
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface"]


def test_connect_skips_a_minted_key_the_owner_does_not_hold(fake_studio, tmp_path):
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"minted": ["sk-unsloth-a11ce000a11ce000"]}}}))
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")


def test_connect_explicit_key_remembered_for_keyless_runs(fake_studio, tmp_path):
    CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["saved"] == ["sk-unsloth-deadbeefdeadbeef"]


def test_connect_skips_cached_keys_the_server_rejects(fake_studio, tmp_path, monkeypatch):
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(
        json.dumps(
            {"servers": {BASE: {"minted": ["sk-unsloth-stale", "sk-unsloth-feedfacefeedface"]}}}
        )
    )
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models") and token == "sk-unsloth-stale":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    cached = json.loads(cache.read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface", "sk-unsloth-stale"]


def test_connect_saved_key_server_outage_surfaces_not_reminted(fake_studio, tmp_path, monkeypatch):
    # A 5xx/timeout on a saved key is an outage, not a rejected key.
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"saved": ["sk-unsloth-saved"]}}}))
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models") and token == "sk-unsloth-saved":
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code != 0, result.output
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert mints == []


def test_connect_legacy_unscoped_cache_not_replayed(fake_studio, tmp_path):
    (tmp_path / "agent_api_key.json").write_text(json.dumps({"key": "sk-unsloth-oldformat"}))
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface"]
    assert "key" not in cached


def test_connect_model_flag_loads_on_server(fake_studio):
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--model", "unsloth/Qwen3.5-35B-A3B"]
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        ("POST", f"{BASE}/api/inference/load", {"model_path": "unsloth/Qwen3.5-35B-A3B"})
    ]
    assert result.output.index(
        f"Switching the Unsloth server from {MODEL['id']} to unsloth/Qwen3.5-35B-A3B.\n"
    ) < result.output.index("This unloads the current model for every attached session.\n")
    _assert_env_set(result.output, "ANTHROPIC_MODEL", "unsloth/Qwen3.5-35B-A3B")


def _fake_path_resident(
    monkeypatch,
    listed_id,
    load_status,
    *,
    load_error = None,
    after_failure = "kept",
):
    inner = start._http_json
    state = {"after": None, "listed": listed_id}

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models"):
            if state["after"] == "unreachable":
                raise TimeoutError("timed out")
            return {"data": [{"id": state["listed"], "loaded": state["after"] != "gone"}]}
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "active_model": listed_id, "model_identifier": "ref:0123"}
        if url.endswith("/api/inference/load"):
            if load_error is not None:
                state["after"] = after_failure
                raise load_error
            name = os.path.basename(payload["model_path"]).removesuffix(".gguf")
            if load_status == "loaded":
                state["listed"] = name
            return {"status": load_status, "model": payload["model_path"], "display_name": name}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)


@pytest.mark.parametrize(
    "requested, listed_id",
    [
        ("/models/old/foo-Q4_K_M.gguf", "foo-Q4_K_M"),
        (
            "/cache/hub/models--unsloth--Foo-GGUF/snapshots/abc123/Foo-UD-IQ1_S.gguf",
            "unsloth/Foo-GGUF",
        ),
    ],
)
def test_connect_model_path_reattach_announces_no_switch(
    fake_studio, monkeypatch, requested, listed_id
):
    _fake_path_resident(monkeypatch, listed_id, "already_loaded")
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", requested])
    assert result.exit_code == 0, result.output
    assert "Switching" not in result.output
    assert "unload" not in result.output
    assert f"Reusing loaded model: {requested}" in result.output


def test_connect_model_path_same_name_switch_announced_after_load(fake_studio, monkeypatch):
    _fake_path_resident(monkeypatch, "foo-Q4_K_M", "loaded")
    requested = "/models/new/foo-Q4_K_M.gguf"
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", requested])
    assert result.exit_code == 0, result.output
    assert "Switching" not in result.output
    assert f"Loaded {requested} in place of foo-Q4_K_M.\n" in result.output
    assert "This unloaded the previous model for every attached session.\n" in result.output


def test_connect_model_path_other_name_switch_announced_before_load(fake_studio, monkeypatch):
    _fake_path_resident(monkeypatch, "foo-Q4_K_M", "loaded")
    requested = "/models/bar-Q4_K_M.gguf"
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", requested])
    assert result.exit_code == 0, result.output
    assert f"Switching the Unsloth server from foo-Q4_K_M to {requested}.\n" in result.output
    assert "in place of" not in result.output


@pytest.mark.parametrize("after_failure", ["kept", "gone", "unreachable"])
def test_connect_model_path_failed_same_name_load_reports_eviction(
    fake_studio, monkeypatch, after_failure
):
    failure = urllib.error.HTTPError(
        f"{BASE}/api/inference/load", 500, "Internal Server Error", None, None
    )
    _fake_path_resident(
        monkeypatch, "foo-Q4_K_M", None, load_error = failure, after_failure = after_failure
    )
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--model", "/models/new/foo-Q4_K_M.gguf"]
    )
    assert result.exit_code != 0
    evicted = "foo-Q4_K_M was unloaded for every attached session." in result.output
    assert evicted is (after_failure == "gone")
    assert "Nothing was unloaded" not in result.output


def test_connect_model_flag_forwards_load_options(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--no-launch",
            "--model",
            "unsloth/Qwen3-4B-GGUF",
            "--gguf-variant",
            "UD-Q4_K_XL",
            "--context-length",
            "8192",
            "--no-load-in-4bit",
            "--tensor-parallel",
        ],
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        (
            "POST",
            f"{BASE}/api/inference/load",
            {
                "model_path": "unsloth/Qwen3-4B-GGUF",
                "gguf_variant": "UD-Q4_K_XL",
                "max_seq_length": 8192,
                "load_in_4bit": False,
                "tensor_parallel": True,
            },
        )
    ]


def test_connect_model_flag_matches_canonical_id(fake_studio, monkeypatch):
    requested = "Unsloth/Qwen3.5-35B-A3B"
    canonical = "unsloth/Qwen3.5-35B-A3B"
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/load"):
            return {"model": canonical, "display_name": canonical}
        if url.endswith("/api/inference/loaded-models"):
            # Decoy sorts first, so models[0] is the wrong pick on the old code.
            return {"object": "list", "data": [MODEL, {"id": canonical, "context_length": 4096}]}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", requested])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", canonical)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL", ("unsloth/Qwen3-1.7B-GGUF", "UD-Q4_K_XL")),
        ("unsloth/gemma-4-E2B-it-GGUF:Q8_0", ("unsloth/gemma-4-E2B-it-GGUF", "Q8_0")),
        ("unsloth/Qwen3-1.7B-GGUF", ("unsloth/Qwen3-1.7B-GGUF", None)),
        ("/models/local.gguf", ("/models/local.gguf", None)),
        ("./rel.gguf", ("./rel.gguf", None)),
        ("C:\\models\\x.gguf", ("C:\\models\\x.gguf", None)),
        ("repo:with/slash", ("repo:with/slash", None)),
        ("", ("", None)),
    ],
)
def test_split_repo_variant(model, expected):
    assert start._split_repo_variant(model) == expected


@pytest.mark.parametrize(
    "token, expected",
    [
        ("unsloth/gemma-4-E2B-it-GGUF", True),
        ("unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL", True),
        ("some-org/model.name_1", True),
        ("--continue", False),
        ("resume", False),
        ("/models/local.gguf", False),
        ("./rel.gguf", False),
        ("C:\\models\\x.gguf", False),
        ("my models/foo", False),
        ("owner/repo/extra", False),
    ],
)
def test_looks_like_model(token, expected):
    assert start._looks_like_model(token) is expected


def test_consume_positional_model_leading_token():
    model, rest = start._consume_positional_model(None, ["unsloth/Model-GGUF", "--continue"])
    assert model == "unsloth/Model-GGUF"
    assert rest == ["--continue"]


def test_looks_like_model_leaves_existing_local_dir_for_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "repo").mkdir(parents = True)
    assert start._looks_like_model("owner/repo") is False
    model, rest = start._consume_positional_model(None, ["owner/repo"])
    assert model is None and rest == ["owner/repo"]
    assert start._looks_like_model("owner/absent-repo") is True


def test_consume_positional_model_ignores_non_leading_and_explicit_model():
    model, rest = start._consume_positional_model(None, ["--profile", "owner/repo"])
    assert model is None and rest == ["--profile", "owner/repo"]
    model, rest = start._consume_positional_model("explicit/model", ["owner/repo"])
    assert model == "explicit/model" and rest == ["owner/repo"]


def test_start_separator_preserves_model_shaped_agent_argument(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--no-launch", "--", "owner/repo"],
    )

    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-2:] == ["--", "owner/repo"]

    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--no-launch", MODEL["id"], "--", "--continue"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-2:] == ["--", "--continue"]


def test_start_positional_model_routes_to_model_on_auto_serve(fake_studio, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    captured = {}
    fake = SimpleNamespace(pid = 1, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        captured["model"] = model
        captured["load"] = load
        captured["server_options"] = server_options
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))

    result = CliRunner().invoke(start.start_app, ["claude", "unsloth/gemma-4-E2B-it-GGUF"])
    assert result.exit_code == 0, result.output
    assert captured["model"] == "unsloth/gemma-4-E2B-it-GGUF"
    assert captured["load"].gguf_variant is None


def test_start_local_gguf_path_keeps_no_default_variant(fake_studio, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    local = tmp_path / "Qwen3-1.7B-GGUF"
    local.mkdir()
    captured = {}
    fake = SimpleNamespace(pid = 1, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        captured["load"] = load
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))

    result = CliRunner().invoke(start.start_app, ["claude", "--model", str(local)])
    assert result.exit_code == 0, result.output
    assert captured["load"].gguf_variant is None


def test_start_studio_server_forwards_tool_flags_via_command_and_env(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)
    monkeypatch.delenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", raising = False)
    monkeypatch.delenv("UNSLOTH_TOOL_CALL_NUDGE", raising = False)

    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    cmd, env = captured["command"], captured["kwargs"]["env"]
    assert "--disable-tools" in cmd and "--enable-tools" not in cmd
    assert "--reasoning" not in cmd
    assert env["LLAMA_ARG_REASONING"] == "auto"
    assert "--gpu-memory-mode" not in cmd
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "0"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "1"

    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(
            enable_tools = True,
            tool_call_healing = False,
            tool_call_nudging = False,
            reasoning = "auto",
        ),
    )
    cmd, env = captured["command"], captured["kwargs"]["env"]
    assert "--enable-tools" in cmd and "--disable-tools" not in cmd
    assert "--reasoning" not in cmd
    assert env["LLAMA_ARG_REASONING"] == "auto"
    assert env["LLAMA_ARG_REASONING_EFFORT"] == "default"
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "1"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "0"


def test_start_studio_server_respects_inherited_tool_call_env(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["kwargs"] = kwargs
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)
    monkeypatch.setenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", "1")
    monkeypatch.setenv("UNSLOTH_TOOL_CALL_NUDGE", "0")

    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    env = captured["kwargs"]["env"]
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "1"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "0"

    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(tool_call_healing = True, tool_call_nudging = True),
    )
    env = captured["kwargs"]["env"]
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "0"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "1"


def test_start_studio_server_drops_an_inherited_pin_the_agent_overrides(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["env"] = kwargs["env"]
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.7")
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_P", "0.5")
    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(unpinned = frozenset({"temperature"})),
    )
    assert "UNSLOTH_SAMPLING_TEMPERATURE" not in captured["env"]
    assert captured["env"]["UNSLOTH_SAMPLING_TOP_P"] == "0.5"


def test_start_studio_server_forwards_sampling_via_env(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["kwargs"] = kwargs
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)
    for _v in ("TEMPERATURE", "TOP_P", "TOP_K", "MIN_P", "REPETITION_PENALTY", "PRESENCE_PENALTY"):
        monkeypatch.delenv(f"UNSLOTH_SAMPLING_{_v}", raising = False)

    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    env = captured["kwargs"]["env"]
    assert not any(k.startswith("UNSLOTH_SAMPLING_") for k in env)

    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(temperature = 0.3, top_k = 40, min_p = 0.05),
    )
    env = captured["kwargs"]["env"]
    assert env["UNSLOTH_SAMPLING_TEMPERATURE"] == "0.3"
    assert env["UNSLOTH_SAMPLING_TOP_K"] == "40"
    assert env["UNSLOTH_SAMPLING_MIN_P"] == "0.05"
    assert "UNSLOTH_SAMPLING_TOP_P" not in env


def test_require_studio_warns_on_sampling_pin_when_reusing_server(monkeypatch, capsys):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    base, server = start._require_studio(
        "unsloth/M-GGUF",
        start.LoadOptions(),
        serve = True,
        launch = True,
        server_options = start.ServerOptions(temperature = 0.3, top_k = 40),
    )
    assert base == BASE
    assert server is None
    err = capsys.readouterr().err
    assert "already running" in err
    assert "--temperature" in err and "--top-k" in err
    assert "--top-p" not in err


def test_require_studio_no_warning_without_server_flags(monkeypatch, capsys):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    base, server = start._require_studio(
        "unsloth/M-GGUF",
        start.LoadOptions(),
        serve = True,
        server_options = start.ServerOptions(),
    )
    assert base == BASE and server is None
    assert capsys.readouterr().err == ""


def test_require_studio_warns_on_tool_flags_when_reusing_server(monkeypatch, capsys):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    base, server = start._require_studio(
        "unsloth/M-GGUF",
        start.LoadOptions(),
        serve = True,
        server_options = start.ServerOptions(
            enable_tools = True, tool_call_healing = False, tool_call_nudging = None
        ),
    )
    assert base == BASE and server is None
    err = capsys.readouterr().err
    assert "already running" in err
    assert "--enable-tools" in err and "--disable-tool-call-healing" in err
    assert "nudging" not in err


_SESSION_FLAGS = ["--temperature", "0.3", "--top-k", "40", "--reasoning", "off"]
_SESSION_BODY = {"temperature": 0.3, "top_k": 40, "enable_thinking": False}


def _session_request_body(agent, root, output):
    yaml = pytest.importorskip("yaml")
    home = root / "agents" / agent
    model_ref = f"unsloth/{MODEL['id']}"
    if agent == "pi":
        config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
        return config["providers"]["unsloth"]["models"][0].get("samplingParams")
    if agent == "hermes":
        config = yaml.safe_load((home / "config.yaml").read_text())
        return config["providers"]["unsloth"].get("extra_body")
    if agent == "openclaw":
        config = json.loads((home / "openclaw.json").read_text())
        models = config["agents"]["defaults"].get("models", {})
        return models.get(model_ref, {}).get("params", {}).get("extra_body")
    if agent == "opencode":
        provider = json.loads((home / "opencode.json").read_text())["provider"]
        provider = provider[start._OPENCODE_PROVIDER]
        options = provider["models"][MODEL["id"]].get("options")
        assert provider["options"].get("body") == options
        return options
    if agent == "claude":
        settings = Path(shlex.split(re.search(r"--settings (\S+)", output).group(1))[0])
        extra = json.loads(settings.read_text())["env"].get("CLAUDE_CODE_EXTRA_BODY")
        return json.loads(extra) if extra else None
    raise AssertionError(agent)


@pytest.mark.parametrize("agent", ["pi", "hermes", "openclaw", "opencode", "claude"])
def test_session_flags_ride_in_the_agent_config_on_a_running_server(
    agent, fake_studio, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    assert _session_request_body(agent, tmp_path, result.output) == _SESSION_BODY


@pytest.mark.parametrize("agent", ["pi", "hermes", "openclaw", "opencode", "claude"])
def test_session_flags_from_an_earlier_run_do_not_stick(agent, fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for argv in ([agent, "--no-launch", *_SESSION_FLAGS], [agent, "--no-launch"]):
        result = CliRunner().invoke(start.start_app, argv)
        assert result.exit_code == 0, result.output
    assert not _session_request_body(agent, tmp_path, result.output)


def test_openclaw_session_flags_keep_the_users_own_extra_body(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents = True)
    model_ref = f"unsloth/{MODEL['id']}"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "models": {
                            model_ref: {"params": {"extra_body": {"tool_choice": "required"}}}
                        }
                    }
                }
            }
        )
    )
    result = CliRunner().invoke(
        start.start_app, ["openclaw", "--no-launch", "--temperature", "0.3"]
    )
    assert result.exit_code == 0, result.output
    assert _session_request_body("openclaw", tmp_path, result.output) == {
        "tool_choice": "required",
        "temperature": 0.3,
    }
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _session_request_body("openclaw", tmp_path, result.output) == {"tool_choice": "required"}


def test_opencode_session_temperature_needs_the_capability(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    for argv, capability in ((["--temperature", "0.3"], True), (["--top-k", "40"], None)):
        result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", *argv])
        assert result.exit_code == 0, result.output
        provider = json.loads(config_path.read_text())["provider"][start._OPENCODE_PROVIDER]
        assert provider["models"][MODEL["id"]].get("temperature") is capability


def test_pi_subagent_carries_the_session_flags(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["pi", "--as-subagent", "--no-launch", *_SESSION_FLAGS]
    )
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    config = json.loads((tmp_path / "agents" / "pi-subagent" / "subagent.json").read_text())
    assert config["samplingParams"] == _SESSION_BODY


def test_codex_carries_reasoning_and_warns_about_sampling(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert 'model_reasoning_effort = "none"' in profile
    assert "cannot send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


def test_codex_warns_about_reasoning_it_cannot_express(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", "--reasoning", "on"])
    assert result.exit_code == 0, result.output
    assert "cannot send --reasoning on itself" in result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


@pytest.mark.parametrize("mode", ["--persist", "--no-launch"])
@pytest.mark.parametrize("agent, version", [("codex", (0, 144, 0)), ("pi", (0, 83, 0))])
def test_agent_too_old_to_send_the_flags_keeps_the_server_pin(
    agent, version, mode, fake_studio, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda name: f"/bin/{name}")
    monkeypatch.setattr(start, "_codex_executable_version", lambda executable: version)
    monkeypatch.setattr(start, "_launch", lambda *args, **kwargs: None)
    result = CliRunner().invoke(start.start_app, [agent, mode, *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "cannot send --temperature, --top-k itself" in result.output
    assert "cannot send --reasoning off itself" in result.output
    if agent == "pi":
        models = tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json"
        provider = json.loads(models.read_text())["providers"]["unsloth"]
        assert "samplingParams" not in provider["models"][0]
    else:
        profile = (
            tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml"
        ).read_text()
        assert "model_reasoning_effort" not in profile


def test_opencode_v1_reads_the_effort_as_reasoning_effort_option(
    fake_studio, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--reasoning-effort", "low"]
    )
    assert result.exit_code == 0, result.output
    provider = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    provider = provider["provider"][start._OPENCODE_PROVIDER]
    assert provider["models"][MODEL["id"]]["options"] == {"reasoningEffort": "low"}
    assert provider["options"]["body"] == {"reasoning_effort": "low"}


def test_dsh_carries_reasoning_and_warns_about_sampling(fake_studio, tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "cannot send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    patch = tmp_path / "agents" / "dsh" / start._DSH_PATCH_FILE
    provider = yaml.safe_load(patch.read_text())[0]["config"]["providers"][start._DSH_PROVIDER]
    assert provider["compat"]["thinkingFormat"] == "chat-template"
    assert provider["compat"]["chatTemplateKwargs"] == {"enable_thinking": False}
    assert "off" in provider["models"][0]["reasoningEfforts"]


@pytest.mark.parametrize(
    "agent, server_keeps",
    [("pi", {}), ("codex", {"temperature": 0.3, "top_k": 40})],
)
def test_spawned_server_keeps_only_what_the_agent_cannot_send(
    agent, server_keeps, fake_studio, monkeypatch
):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started["options"] = server_options
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    monkeypatch.setattr(start, "_codex_executable_version", lambda executable: (1, 0, 0))
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))
    result = CliRunner().invoke(
        start.start_app, [agent, "--model", "unsloth/Qwen3-1.7B-GGUF", *_SESSION_FLAGS]
    )
    assert result.exit_code == 0, result.output
    options = started["options"]
    sampling = {name: getattr(options, name) for name in ("temperature", "top_k")}
    assert {k: v for k, v in sampling.items() if v is not None} == server_keeps
    assert options.reasoning is None
    assert options.unpinned == {"reasoning", *({"temperature", "top_k"} - set(server_keeps))}
    assert ("applies them to every client" in result.output) == bool(server_keeps)


@pytest.mark.parametrize("reasoning, warns", [("on", True), ("auto", False)])
def test_require_studio_warns_when_the_started_server_pins_reasoning(
    monkeypatch, capsys, reasoning, warns
):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.setattr(
        start, "_start_studio_server", lambda base, model, load, server: (base, None)
    )
    start._require_studio(
        "unsloth/M-GGUF", serve = True, server_options = start.ServerOptions(reasoning = reasoning)
    )
    assert ("every client" in capsys.readouterr().err) is warns


@pytest.mark.parametrize("inherited", ["0.7", None])
def test_require_studio_warns_when_a_flag_drops_an_inherited_pin(monkeypatch, capsys, inherited):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    if inherited:
        monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", inherited)
    else:
        monkeypatch.delenv("UNSLOTH_SAMPLING_TEMPERATURE", raising = False)
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_P", "0.5")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.setattr(
        start, "_start_studio_server", lambda base, model, load, server: (base, None)
    )
    options = start.ServerOptions(temperature = 0.3, carried = frozenset({"temperature"}))
    start._require_studio("unsloth/M-GGUF", serve = True, server_options = options)
    err = capsys.readouterr().err
    assert ("replace the inherited UNSLOTH_SAMPLING_TEMPERATURE" in err) is bool(inherited)
    assert "UNSLOTH_SAMPLING_TOP_P" not in err


@pytest.mark.parametrize(
    "reasoning, carried",
    [
        ("on", frozenset()),
        ("off", frozenset()),
        ("auto", frozenset()),
        ("auto", start._ALL_REQUEST_FIELDS),
    ],
)
def test_require_studio_warns_on_explicit_reasoning_when_reusing_server(
    monkeypatch, capsys, reasoning, carried
):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    base, server = start._require_studio(
        "unsloth/M-GGUF",
        start.LoadOptions(),
        serve = True,
        server_options = start.ServerOptions(reasoning = reasoning, carried = carried),
    )
    assert base == BASE and server is None
    err = capsys.readouterr().err
    assert "already running" in err
    assert f"--reasoning {reasoning}" in err
    assert "unsloth studio stop" in err


def test_start_studio_server_forwards_reasoning_effort(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["kwargs"] = kwargs
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)

    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(reasoning_effort = "medium"),
    )
    env = captured["kwargs"]["env"]
    assert env["LLAMA_ARG_REASONING"] == "auto"
    assert env["LLAMA_ARG_REASONING_EFFORT"] == "medium"


def test_start_studio_server_overrides_inherited_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_REASONING_EFFORT", "high")
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["kwargs"] = kwargs
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)

    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    assert captured["kwargs"]["env"]["LLAMA_ARG_REASONING_EFFORT"] == "default"


def test_require_studio_warns_on_explicit_reasoning_effort_when_reusing_server(monkeypatch, capsys):
    monkeypatch.setattr(start, "find_studio_server", lambda: BASE)
    base, server = start._require_studio(
        "unsloth/M-GGUF",
        start.LoadOptions(),
        serve = True,
        server_options = start.ServerOptions(reasoning = "on", reasoning_effort = "high"),
    )
    assert base == BASE and server is None
    err = capsys.readouterr().err
    assert "--reasoning on" in err and "--reasoning-effort high" in err
    assert "unsloth studio stop" in err


def test_start_claude_sends_sampling_flags_itself(fake_studio, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    captured = {}
    fake = SimpleNamespace(pid = 1, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        captured["server_options"] = server_options
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, env: captured.update(env = env) or SimpleNamespace(returncode = 0),
    )

    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--model",
            "unsloth/gemma-4-E2B-it-GGUF",
            "--temperature",
            "0.3",
            "--top-k",
            "40",
            "--reasoning",
            "on",
        ],
    )
    assert result.exit_code == 0, result.output
    so = captured["server_options"]
    assert so.temperature is None and so.top_k is None and so.reasoning is None
    assert json.loads(captured["env"]["CLAUDE_CODE_EXTRA_BODY"]) == {
        "temperature": 0.3,
        "top_k": 40,
        "enable_thinking": True,
    }


def test_connect_model_bare_id_matches_loaded_without_reload(fake_studio):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", MODEL["id"]])
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == []
    assert f"Reusing loaded model: {MODEL['id']}\n" in result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])


def test_connect_model_variant_suffix_defers_to_server_dedup(fake_studio):
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--model", MODEL["id"] + ":UD-Q4_K_XL"]
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        (
            "POST",
            f"{BASE}/api/inference/load",
            {"model_path": MODEL["id"], "gguf_variant": "UD-Q4_K_XL"},
        )
    ]
    assert f"Reusing loaded model: {MODEL['id']}:UD-Q4_K_XL\n" in result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])


def test_connect_load_knobs_reach_server_even_when_id_loaded(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--model", MODEL["id"], "--gguf-variant", "Q8_0"],
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        ("POST", f"{BASE}/api/inference/load", {"model_path": MODEL["id"], "gguf_variant": "Q8_0"})
    ]


@pytest.mark.parametrize(
    "command_name", ["claude", "codex", "openclaw", "opencode", "hermes", "pi", "dsh", "vibe"]
)
def test_start_agents_expose_gpu_memory_mode_option(command_name):
    import inspect

    command = getattr(start, command_name)
    opt = inspect.signature(command).parameters["gpu_memory_mode"].default
    assert set(getattr(opt, "param_decls", None) or []) == {"--gpu-memory-mode"}
    assert getattr(opt, "default", None) is None
    assert getattr(opt, "rich_help_panel", None) == start._PANEL_MODEL


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("auto", {"model_path": MODEL["id"], "gpu_memory_mode": "auto"}),
        (
            "manual",
            {
                "model_path": MODEL["id"],
                "gpu_memory_mode": "manual",
                "gpu_layers": -1,
            },
        ),
    ],
)
def test_start_gpu_memory_mode_reaches_running_server(fake_studio, mode, expected):
    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--no-launch",
            "--model",
            MODEL["id"],
            "--gpu-memory-mode",
            mode,
        ],
    )
    assert result.exit_code == 0, result.output
    loads = [call for call in fake_studio if call[1].endswith("/api/inference/load")]
    assert loads == [("POST", f"{BASE}/api/inference/load", expected)]


def test_start_rejects_invalid_gpu_memory_mode(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--gpu-memory-mode", "invalid"],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--gpu-memory-mode'" in result.output


def test_connect_model_variant_suffix_loads_split_repo(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--model", "unsloth/Qwen3-4B-GGUF:UD-Q4_K_XL"],
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        (
            "POST",
            f"{BASE}/api/inference/load",
            {"model_path": "unsloth/Qwen3-4B-GGUF", "gguf_variant": "UD-Q4_K_XL"},
        )
    ]


def test_connect_explicit_gguf_variant_wins_over_suffix(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--no-launch",
            "--model",
            "unsloth/Qwen3-4B-GGUF:Q8_0",
            "--gguf-variant",
            "UD-Q4_K_XL",
        ],
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        (
            "POST",
            f"{BASE}/api/inference/load",
            {"model_path": "unsloth/Qwen3-4B-GGUF", "gguf_variant": "UD-Q4_K_XL"},
        )
    ]


def test_connect_no_model_loaded_errors(fake_studio, monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda method, url, token, payload = None, timeout = 30, error = None: (
            {"key": "sk-unsloth-feedfacefeedface"}
            if url.endswith("/api/auth/api-keys")
            else {"object": "list", "data": []}
        ),
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "No model is loaded" in result.output


def test_connect_requested_model_not_loaded_fails(fake_studio, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/load"):
            return {}
        if url.endswith("/api/inference/loaded-models"):
            return {"object": "list", "data": [MODEL]}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--model", "unsloth/Missing-7B"]
    )
    assert result.exit_code == 1
    assert "unsloth/Missing-7B" in result.output


def test_connect_gguf_only_agents_reject_non_gguf_model(fake_studio, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "model_identifier": "unsloth/Qwen3-0.6B"}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 1
    assert "Codex needs a GGUF model" in result.output
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "Claude Code needs a GGUF model" in result.output
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output


def test_connect_nonloopback_keyless_refuses_to_send_credential(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://studio.evil.example:8888")
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 1
    assert "Settings → API" in result.output
    assert "--api-key" in result.output
    assert fake_studio == []


def test_connect_nonloopback_explicit_key_is_allowed(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://studio.example:8888")
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output


def test_connect_nonloopback_replays_saved_key(fake_studio, tmp_path, monkeypatch):
    remote = "http://studio.example:8888"
    monkeypatch.setattr(start, "find_studio_server", lambda: remote)
    (tmp_path / "agent_api_key.json").write_text(
        json.dumps({"servers": {remote: {"saved": ["sk-unsloth-deadbeefdeadbeef"]}}})
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)


def test_connect_studio_server_errors_on_explicit_remote(monkeypatch):
    import typer

    import unsloth_cli._inference as inference

    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://studio.example:8888")
    monkeypatch.setattr(
        inference, "find_studio_server", lambda *a, **k: "http://studio.example:8888"
    )
    with pytest.raises(typer.Exit):
        inference.connect_studio_server("m", hf_token = None, max_seq_length = 4096, load_in_4bit = False)


def test_connect_studio_server_falls_back_locally_on_default_discovery(monkeypatch):
    import unsloth_cli._inference as inference

    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(inference, "find_studio_server", lambda *a, **k: "http://127.0.0.1:8888")
    monkeypatch.setattr(inference, "verify_studio_identity", lambda *a, **k: False)
    assert (
        inference.connect_studio_server("m", hf_token = None, max_seq_length = 4096, load_in_4bit = False)
        is None
    )


def test_connect_unverified_loopback_without_cached_key_refuses_to_mint(
    fake_studio, tmp_path, monkeypatch
):
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "--api-key" in result.output
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)


def test_connect_replays_saved_key_without_identity_check(fake_studio, tmp_path, monkeypatch):
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"saved": ["sk-unsloth-deadbeefdeadbeef"]}}}))
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)


def test_connect_minted_cache_requires_identity_check(fake_studio, tmp_path, monkeypatch):
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"minted": ["sk-unsloth-feedfacefeedface"]}}}))
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "--api-key" in result.output
    assert not any(c[1].endswith("/api/inference/loaded-models") for c in fake_studio)


def test_connect_explicit_key_skips_identity_check(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")


def _serve_identity(proof_for):
    """Start a localhost HTTP server answering /api/auth/identity with
    proof_for(nonce_bytes). Returns (base_url, shutdown)."""
    import base64
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/auth/identity":
                self.send_response(404)
                self.end_headers()
                return
            nonce = base64.urlsafe_b64decode(parse_qs(parsed.query)["nonce"][0])
            host, port = self.server.server_address[0], self.server.server_address[1]
            body = json.dumps({"proof": proof_for(nonce, host, port)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return base, server.shutdown


def test_verify_studio_identity_end_to_end(tmp_path, monkeypatch):
    import unsloth_cli._inference as inference

    inference.ensure_studio_backend_path()
    try:
        from studio.backend.auth import storage
    except Exception as exc:
        pytest.skip(f"studio backend not importable: {exc}")

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)

    good = lambda nonce, host, port: storage.compute_identity_proof(nonce, host, port)
    bad = lambda nonce, host, port: "00" * 32
    base_ok, stop_ok = _serve_identity(good)
    base_bad, stop_bad = _serve_identity(bad)
    try:
        assert inference.verify_studio_identity(base_ok) is True
        assert inference.verify_studio_identity(base_bad) is False
    finally:
        stop_ok()
        stop_bad()


def test_studio_token_is_issued_for_the_owner_not_a_managed_account(tmp_path, monkeypatch):
    import secrets

    import jwt

    import unsloth_cli._inference as inference

    inference.ensure_studio_backend_path()
    from studio.backend.auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    storage.issue_account_setup_code(username = "alice")

    token = inference._studio_token()
    assert token
    assert jwt.decode(token, options = {"verify_signature": False})["sub"] == "unsloth"


def _serve_redirect(target):
    """Start a localhost server that 302-redirects every GET to target+path."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target + self.path)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return base, server.shutdown


def test_verify_studio_identity_rejects_redirect(tmp_path, monkeypatch):
    import unsloth_cli._inference as inference

    inference.ensure_studio_backend_path()
    try:
        from studio.backend.auth import storage
    except Exception as exc:
        pytest.skip(f"studio backend not importable: {exc}")

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)

    real_base, stop_real = _serve_identity(
        lambda nonce, host, port: storage.compute_identity_proof(nonce, host, port)
    )
    squatter_base, stop_squatter = _serve_redirect(real_base)
    try:
        assert inference.verify_studio_identity(real_base) is True
        assert inference.verify_studio_identity(squatter_base) is False
    finally:
        stop_real()
        stop_squatter()


def test_verify_studio_identity_rejects_relayed_proof(tmp_path, monkeypatch):
    import unsloth_cli._inference as inference

    inference.ensure_studio_backend_path()
    try:
        from studio.backend.auth import storage
    except Exception as exc:
        pytest.skip(f"studio backend not importable: {exc}")

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)

    real_base, stop_real = _serve_identity(
        lambda nonce, host, port: storage.compute_identity_proof(nonce, host, port)
    )
    real_port = int(real_base.rsplit(":", 1)[1])
    squatter_base, stop_squatter = _serve_identity(
        lambda nonce, host, port: storage.compute_identity_proof(nonce, host, real_port)
    )
    try:
        assert inference.verify_studio_identity(real_base) is True
        assert inference.verify_studio_identity(squatter_base) is False
    finally:
        stop_real()
        stop_squatter()


@pytest.mark.parametrize(
    "url, loopback",
    [
        ("http://127.0.0.1:8888", True),
        ("http://localhost:8888", True),
        ("http://[::1]:8888", True),
        ("http://127.0.0.5:9001", True),
        ("http://0.0.0.0:8888", False),
        ("http://10.0.0.5:8888", False),
        ("http://studio.evil.example:8888", False),
        ("https://studio.example.com", False),
    ],
)
def test_is_loopback_url(url, loopback):
    assert start.is_loopback_url(url) is loopback


def test_connect_no_studio_errors(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "No running Unsloth server" in result.output


@pytest.fixture(autouse = True)
def _studio_is_this_machine(monkeypatch):
    # The gate reads the local filesystem only for a confirmed identity; tunnel tests override this.
    monkeypatch.setattr(start, "verify_studio_identity", lambda base, **_kw: True)


@pytest.fixture(autouse = True)
def _reset_auto_served():
    # Never leave a fake server in the module slot: the atexit backstop would signal it.
    yield
    start._auto_served_server = None


def test_start_studio_server_builds_command_and_waits(monkeypatch, capsys):
    captured = {}
    monkeypatch.setenv(start._START_API_KEY_MARKER_ENV, "parent")

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            self.pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(start.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start, "_studio_healthy", lambda base, timeout = 3.0: True)
    monkeypatch.setattr(start, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-abc123")
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)

    returned_base, server = start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL",
        start.LoadOptions(
            gguf_variant = "UD-Q4_K_XL",
            max_seq_length = 8192,
            load_in_4bit = True,
            tensor_parallel = True,
            gpu_memory_mode = "manual",
        ),
    )
    cmd = captured["command"]
    assert cmd[1] == "run"
    assert "--disable-tools" in cmd and "--no-cloudflare" in cmd
    assert "--reasoning" not in cmd
    assert captured["kwargs"]["env"]["LLAMA_ARG_REASONING"] == "auto"
    assert cmd[cmd.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL"
    assert cmd[cmd.index("--gguf-variant") + 1] == "UD-Q4_K_XL"
    assert cmd[cmd.index("--context-length") + 1] == "8192"
    assert "--tensor-parallel" in cmd
    assert cmd[cmd.index("--gpu-memory-mode") + 1] == "manual"
    assert "--start-api-key-marker" not in cmd
    assert captured["kwargs"]["env"][start._START_API_KEY_MARKER_ENV] == "1"
    assert start.os.environ[start._START_API_KEY_MARKER_ENV] == "parent"
    assert cmd[cmd.index("-p") + 1] == "8888"
    assert returned_base == "http://127.0.0.1:8888"
    assert start.LoadOptions().load_in_4bit is True and "--no-load-in-4bit" not in cmd
    assert captured["kwargs"].get("start_new_session") is True
    assert server.pid == 4321
    output = capsys.readouterr().out
    assert "Starting Unsloth server\n" in output
    assert "Model: unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL\n" in output
    assert "No Unsloth server at" not in output
    assert "server ready" not in output


def test_start_studio_server_polls_progress_from_early_key(monkeypatch):
    class FakePopen:
        pid = 4321

        def poll(self):
            return None

    tails = iter(
        [
            "UNSLOTH_START_API_KEY: sk-unsloth-early\nLoading model...",
            "UNSLOTH_START_API_KEY: sk-unsloth-early\nModel loaded: owner/model",
        ]
    )
    created = []

    class FakeProgress:
        downloaded_bytes = 0

        def __init__(self, base, key, model, variant):
            created.append((base, key, model, variant, "created"))

        def poll(self):
            created.append("poll")

        def close(self):
            created.append("close")

        def complete(self):
            created.append("complete")

    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: FakePopen())
    monkeypatch.setattr(start, "_studio_healthy", lambda *a, **k: True)
    monkeypatch.setattr(start, "_log_tail", lambda *a, **k: next(tails))
    monkeypatch.setattr(start, "_ModelDownloadProgress", FakeProgress)
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        start.typer,
        "echo",
        lambda message = "", **_kwargs: created.append(("echo", message)),
    )

    returned_base, server = start._start_studio_server(
        BASE,
        "owner/model-GGUF",
        start.LoadOptions(gguf_variant = "Q4_K_M"),
    )

    assert (returned_base, server.pid) == (BASE, 4321)
    assert (BASE, "sk-unsloth-early", "owner/model-GGUF", "Q4_K_M", "created") in created
    assert created.count("poll") == 2
    assert created[-2:] == ["complete", "close"]
    assert not any(isinstance(event, tuple) and "server ready" in event[-1] for event in created)


def test_start_studio_server_follows_the_port_the_child_bound(monkeypatch, tmp_path):
    requests = {"occupant": [], "studio": []}

    def handler(name, status, body):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests[name].append(self.path)
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        return Handler

    occupant = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler("occupant", 404, b"{}"))
    studio = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), handler("studio", 200, b'{"status": "healthy"}')
    )
    for httpd in (occupant, studio):
        threading.Thread(target = httpd.serve_forever, daemon = True).start()
    requested_port, bound_port = occupant.server_address[1], studio.server_address[1]
    fake = SimpleNamespace(pid = 4242, poll = lambda: None)
    commands = []
    progress_bases = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write(
            f"UNSLOTH_START_PORT: {bound_port}\n"
            "UNSLOTH_START_API_KEY: sk-unsloth-early\n"
            "Model loaded: owner/model\n".encode()
        )
        kwargs["stdout"].flush()
        return fake

    class FakeProgress:
        downloaded_bytes = 0

        def __init__(self, base, *_args):
            progress_bases.append(base)

        def poll(self):
            pass

        def complete(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(start.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(start.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start, "_ModelDownloadProgress", FakeProgress)
    monkeypatch.setattr(start, "_SERVER_START_TIMEOUT_S", 5)
    try:
        base, server = start._start_studio_server(
            f"http://127.0.0.1:{requested_port}", "owner/model", start.LoadOptions()
        )
    finally:
        for httpd in (occupant, studio):
            httpd.shutdown()
            httpd.server_close()

    assert (base, server) == (f"http://127.0.0.1:{bound_port}", fake)
    assert commands[0][commands[0].index("-p") + 1] == str(requested_port)
    assert progress_bases == [base]
    assert requests == {"occupant": [], "studio": ["/api/health"]}


def test_start_studio_server_reads_the_port_the_child_reported_once(monkeypatch):
    class FakePopen:
        pid = 4321

        def poll(self):
            return None

    # 401 lines: the 400-line tail starts one line past the port.
    log = (
        "UNSLOTH_START_PORT: 8889\n"
        "UNSLOTH_START_API_KEY: sk-unsloth-early\n"
        + "loading tensors\n" * 398
        + "Model loaded: owner/model"
    )
    healthy = []
    progress_bases = []

    class FakeProgress:
        downloaded_bytes = 0

        def __init__(self, base, *_args):
            progress_bases.append(base)

        def poll(self):
            pass

        def complete(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: FakePopen())
    monkeypatch.setattr(start, "_studio_healthy", lambda base, **_k: healthy.append(base) or True)
    monkeypatch.setattr(start, "_read_log", lambda _path: log)
    monkeypatch.setattr(start, "_log_tail", lambda *a, **k: "\n".join(log.splitlines()[-400:]))
    monkeypatch.setattr(start, "_ModelDownloadProgress", FakeProgress)
    monkeypatch.setattr(start.time, "sleep", lambda _s: None)

    base, _server = start._start_studio_server(BASE, "owner/model", start.LoadOptions())

    tail = "\n".join(log.splitlines()[-400:])
    assert start._START_API_KEY_PREFIX in tail and start._START_PORT_PREFIX not in tail
    assert base == "http://127.0.0.1:8889"
    assert progress_bases == [base]
    assert healthy == [base]


def test_load_model_with_progress_uses_selected_gguf_size(monkeypatch, capsys):
    release = start.threading.Event()
    calls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        calls.append((method, url, payload))
        if url.endswith("/api/inference/load"):
            assert release.wait(timeout = 2)
            return {"model": "owner/model-GGUF"}
        if "/api/hub/gguf-variants?" in url:
            return {
                "default_variant": "Q8_0",
                "variants": [
                    {
                        "quant": "UD-Q4_K_XL",
                        "filename": "model-UD-Q4_K_XL.gguf",
                        "size_bytes": 4 * 1024**3,
                        "download_size_bytes": 4 * 1024**3,
                    }
                ],
            }
        if "/api/hub/gguf-download-progress?" in url:
            release.set()
            return {
                "downloaded_bytes": 2 * 1024**3,
                "expected_bytes": 4 * 1024**3,
                "progress": 0.5,
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_DOWNLOAD_POLL_INTERVAL_S", 0.001)
    result = start._load_model_with_progress(
        BASE,
        "sk-test",
        "owner/model-GGUF",
        start.LoadOptions(gguf_variant = "UD-Q4_K_XL"),
        {"model_path": "owner/model-GGUF", "gguf_variant": "UD-Q4_K_XL"},
    )

    assert result == {"model": "owner/model-GGUF"}
    output = capsys.readouterr().out
    assert "Downloading model" in output
    assert "100%" in output
    progress_url = next(url for method, url, _ in calls if "gguf-download-progress" in url)
    assert "variant=UD-Q4_K_XL" in progress_url
    assert f"expected_bytes={4 * 1024**3}" in progress_url


# /api/inference/load commits its 200 early and pads; a late failure travels as `_deferred_error`.

_DEFERRED_OOM = {
    "_deferred_error": {"status_code": 507, "detail": "CUDA out of memory"},
}


def _padded(body: dict) -> io.BytesIO:
    """A padded response body: keepalive spaces, then the JSON payload."""
    return io.BytesIO(b"   " + json.dumps(body).encode())


def test_http_json_reads_a_padded_success(monkeypatch):
    """Leading pad bytes are legal JSON, so a slow success is unchanged."""
    monkeypatch.setattr(
        start, "urlopen_no_redirect", lambda request, timeout: _padded({"status": "loaded"})
    )
    assert start._http_json("POST", f"{BASE}/api/inference/load", "sk-test") == {"status": "loaded"}


def test_http_json_raises_a_deferred_error_as_an_http_error(monkeypatch):
    monkeypatch.setattr(
        start, "urlopen_no_redirect", lambda request, timeout: _padded(_DEFERRED_OOM)
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        start._http_json("POST", f"{BASE}/api/inference/load", "sk-test")
    assert excinfo.value.code == 507
    assert "CUDA out of memory" in str(excinfo.value)


def test_http_json_deferred_error_fails_like_an_http_failure(monkeypatch, capsys):
    """With `error` set, a late failure exits 1 with the server's detail, as an early 507
    does, via the same _fail path."""
    monkeypatch.setattr(
        start, "urlopen_no_redirect", lambda request, timeout: _padded(_DEFERRED_OOM)
    )
    with pytest.raises(typer.Exit) as excinfo:
        start._http_json(
            "POST",
            f"{BASE}/api/inference/load",
            "sk-test",
            error = "Model load failed",
        )
    assert excinfo.value.exit_code == 1
    assert "Model load failed: CUDA out of memory" in capsys.readouterr().err


def test_load_model_with_progress_fails_on_a_deferred_error(monkeypatch):
    """The real /load caller: a late failure must not be returned as a result."""

    def urlopen(request, timeout):
        if request.full_url.endswith("/api/inference/load"):
            return _padded(_DEFERRED_OOM)
        raise urllib.error.HTTPError(request.full_url, 404, "not found", None, None)

    monkeypatch.setattr(start, "urlopen_no_redirect", urlopen)
    monkeypatch.setattr(start, "_DOWNLOAD_POLL_INTERVAL_S", 0.001)
    with pytest.raises(typer.Exit) as excinfo:
        start._load_model_with_progress(
            BASE,
            "sk-test",
            "owner/model-GGUF",
            start.LoadOptions(),
            {"model_path": "owner/model-GGUF"},
        )
    assert excinfo.value.exit_code == 1


@pytest.mark.parametrize(
    ("body", "what"),
    [
        (b"", "an empty body"),
        (b"   ", "pad bytes only"),
        (b'  {"status": "loa', "a payload cut in half"),
    ],
)
def test_load_model_with_progress_rejects_a_truncated_padded_body(monkeypatch, body, what):
    """A proxy that gives up mid-pad leaves a 200 the load never finished under.

    Measured: one byte at t=90s, silence, killed ~125s later, a 200 with an EMPTY body.
    `_http_json` decodes a blank body as `{}`, so without the check this returned a
    successful-looking result and the agent connected to whatever was still resident.
    """

    def urlopen(request, timeout):
        if request.full_url.endswith("/api/inference/load"):
            return io.BytesIO(body)
        raise urllib.error.HTTPError(request.full_url, 404, "not found", None, None)

    monkeypatch.setattr(start, "urlopen_no_redirect", urlopen)
    monkeypatch.setattr(start, "_DOWNLOAD_POLL_INTERVAL_S", 0.001)
    with pytest.raises(RuntimeError) as excinfo:
        start._load_model_with_progress(
            BASE,
            "sk-test",
            "owner/model-GGUF",
            start.LoadOptions(),
            {"model_path": "owner/model-GGUF"},
        )
    assert "did not report completion" in str(excinfo.value), what
    assert "/api/inference/load" in str(excinfo.value)


def test_download_progress_ignores_fully_cached_bytes(capsys):
    display = start._DownloadProgressDisplay()
    display.update(
        {
            "downloaded_bytes": 4 * 1024**3,
            "completed_bytes": 4 * 1024**3,
            "expected_bytes": 4 * 1024**3,
            "progress": 0.99,
        }
    )
    display.close()

    assert capsys.readouterr().out == ""


def test_resolve_model_warns_on_same_repo_quant_switch(monkeypatch, capsys):
    models = [{"id": "owner/model-GGUF", "loaded": True}]

    def http_json(
        method,
        url,
        key,
        payload = None,
        timeout = 30,
        error = None,
    ):
        assert url.endswith("/api/inference/status"), url
        return {"is_gguf": True, "gguf_variant": "Q4_K_M"}

    monkeypatch.setattr(start, "_loaded_models", lambda base, key: models)
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: {"status": "loaded", "model": "owner/model-GGUF"},
    )

    start._resolve_model(BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "Q8_0"))

    out = capsys.readouterr().out
    assert (
        "Switching the Unsloth server from owner/model-GGUF:Q4_K_M to owner/model-GGUF:Q8_0." in out
    )
    assert "every attached session" in out


def test_resolve_model_same_quant_prints_no_switch_warning(monkeypatch, capsys):
    models = [{"id": "owner/model-GGUF", "loaded": True}]

    monkeypatch.setattr(start, "_loaded_models", lambda base, key: models)
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: {"is_gguf": True, "gguf_variant": "Q8_0"},
    )
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: {
            "status": "already_loaded",
            "model": "owner/model-GGUF",
        },
    )

    start._resolve_model(BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "Q8_0"))

    out = capsys.readouterr().out
    assert "Switching" not in out
    assert "Reusing loaded model: owner/model-GGUF:Q8_0" in out


def test_resolve_model_refused_load_reports_survivor(monkeypatch, capsys):
    models = [{"id": "owner/model-GGUF", "loaded": True}]

    def http_json(
        method,
        url,
        key,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "gguf_variant": "Q4_K_M"}
        assert url.endswith("/api/inference/loaded-models"), url
        return {"data": models}

    def refuse_load(base, key, model, load, payload):
        typer.echo("Model load failed: GGUF variant 'NOPE_Q9' not found", err = True)
        raise typer.Exit(code = 1)

    monkeypatch.setattr(start, "_loaded_models", lambda base, key: models)
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_load_model_with_progress", refuse_load)

    with pytest.raises(typer.Exit):
        start._resolve_model(
            BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "NOPE_Q9")
        )

    captured = capsys.readouterr()
    assert "This unloads the current model" in captured.out
    assert "Nothing was unloaded; owner/model-GGUF is still serving." in captured.err


def _other_account_studio(
    monkeypatch,
    capsys,
    load_status = "loaded",
):
    state = {"models": [], "loads": []}

    def http_json(
        method,
        url,
        key,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models"):
            return {"object": "list", "data": state["models"]}
        assert url.endswith("/api/inference/status"), url
        if state["models"]:
            return {"is_gguf": True, "active_model": "owner/model-GGUF", "gguf_variant": "Q4_K_M"}
        return {"loaded": [], "loading": [], "yours": False}

    def load_model(base, key, model, load, payload):
        state["loads"].append((payload, capsys.readouterr().out))
        state["models"] = [{"id": model, "loaded": True}]
        return {"status": load_status, "model": model}

    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_load_model_with_progress", load_model)
    return state


@pytest.mark.parametrize(
    "load",
    [start.LoadOptions(), start.LoadOptions(max_seq_length = 4096)],
    ids = ["no-settings", "settings-only"],
)
def test_resolve_model_says_another_account_holds_the_resident(monkeypatch, capsys, load):
    state = _other_account_studio(monkeypatch, capsys)

    with pytest.raises(typer.Exit):
        start._resolve_model(BASE, "key", None, load)

    err = capsys.readouterr().err
    assert "belongs to another account" in err
    assert "No model is loaded" not in err and "No chat model" not in err
    assert state["loads"] == []


def test_resolve_model_warns_before_replacing_another_accounts_resident(monkeypatch, capsys):
    state = _other_account_studio(monkeypatch, capsys)

    start._resolve_model(BASE, "key", "owner/model-GGUF")

    [(payload, before_load)] = state["loads"]
    assert payload == {"model_path": "owner/model-GGUF"}
    assert "from another account's model to owner/model-GGUF." in before_load
    assert "unloads it for every attached session" in before_load


def test_resolve_model_exact_match_shares_another_accounts_resident(monkeypatch, capsys):
    state = _other_account_studio(monkeypatch, capsys, load_status = "already_loaded")

    entry = start._resolve_model(
        BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "Q4_K_M")
    )

    assert entry["id"] == "owner/model-GGUF"
    assert state["loads"][0][0]["gguf_variant"] == "Q4_K_M"
    assert "Reusing loaded model: owner/model-GGUF:Q4_K_M" in capsys.readouterr().out


def test_resolve_model_interrupt_skips_survivor_probe(monkeypatch, capsys):
    models = [{"id": "owner/model-GGUF", "loaded": True}]
    probes = []

    def http_json(
        method,
        url,
        key,
        payload = None,
        timeout = 30,
        error = None,
    ):
        return {"is_gguf": True, "gguf_variant": "Q4_K_M"}

    def interrupted_load(base, key, model, load, payload):
        raise KeyboardInterrupt

    monkeypatch.setattr(start, "_loaded_models", lambda base, key: models)
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_load_model_with_progress", interrupted_load)
    monkeypatch.setattr(
        start, "_model_still_loaded", lambda base, key, model_id: probes.append(model_id) or True
    )

    with pytest.raises(KeyboardInterrupt):
        start._resolve_model(
            BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "Q8_0")
        )

    assert probes == []
    assert "Nothing was unloaded" not in capsys.readouterr().err


def test_resolve_model_failed_load_stays_quiet_when_model_gone(monkeypatch, capsys):
    models = [{"id": "owner/model-GGUF", "loaded": True}]

    def http_json(
        method,
        url,
        key,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "gguf_variant": "Q4_K_M"}
        assert url.endswith("/api/inference/loaded-models"), url
        return {"data": []}

    def failing_load(base, key, model, load, payload):
        raise typer.Exit(code = 1)

    monkeypatch.setattr(start, "_loaded_models", lambda base, key: models)
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_load_model_with_progress", failing_load)

    with pytest.raises(typer.Exit):
        start._resolve_model(
            BASE, "key", "owner/model-GGUF", start.LoadOptions(gguf_variant = "Q8_0")
        )

    assert "Nothing was unloaded" not in capsys.readouterr().err


def test_auto_serves_when_no_server_then_keeps_server(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started.update(base = base, model = model, load = load)
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(
        start, "_shutdown_server", lambda server: started.__setitem__("down", server)
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))

    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL"]
    )
    assert result.exit_code == 0, result.output
    assert started["model"] == "unsloth/Qwen3-1.7B-GGUF"
    assert started["load"].gguf_variant == "UD-Q4_K_XL"
    assert started["base"] == BASE
    assert "down" not in started
    assert start._auto_served_server is None
    assert "is still running" in result.output
    assert "unsloth studio stop" in result.output


def test_auto_served_session_uses_the_port_the_server_bound(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    bound = "http://127.0.0.1:8889"
    fake = SimpleNamespace(pid = 999, poll = lambda: None)
    launched = {}

    def fake_start(*_args):
        start._auto_served_server = fake
        return bound, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_launch", lambda command, env, **_kwargs: launched.update(env) or 0)

    result = CliRunner().invoke(start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF"])

    assert result.exit_code == 0, result.output
    assert f"Unsloth ready at {bound} " in result.output
    assert launched["ANTHROPIC_BASE_URL"] == bound
    assert fake_studio and all(
        url.startswith(f"{bound}/") for _method, url, _payload in fake_studio
    )


def test_auto_served_agent_launch_failure_stops_server(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    stopped = []
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(*_args):
        start._auto_served_server = fake
        return _args[0], fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", stopped.append)
    monkeypatch.setattr(
        start,
        "_launch",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("agent launch failed")),
    )

    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF"],
    )

    assert result.exit_code == 1
    assert stopped == [fake]
    assert "is still running" not in result.output


def test_auto_served_server_exit_is_not_reported_as_running(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    fake = SimpleNamespace(pid = 999, poll = lambda: 1)

    def fake_start(*_args):
        start._auto_served_server = fake
        return _args[0], fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_launch", lambda *a, **k: 0)

    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF"],
    )

    assert result.exit_code == 0, result.output
    assert "stopped during the session" in result.output
    assert "is still running" not in result.output


def test_attached_server_prints_stop_hint_after_agent_exits(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, env: SimpleNamespace(returncode = 0),
    )

    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert f"Unsloth ready at {BASE} · model {MODEL['id']}\n" in result.output
    assert f"Unsloth Studio is still running at {BASE}." in result.output
    assert "Stop it with: unsloth studio stop\n" in result.output


def test_no_launch_recipe_does_not_print_stop_hint(fake_studio):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "is still running" not in result.output


def test_nonzero_agent_exit_notes_code_before_stop_hint(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, env: SimpleNamespace(returncode = 3),
    )

    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 3
    assert "The agent exited with code 3." in result.output
    assert f"Unsloth Studio is still running at {BASE}." in result.output


def test_redacted_log_tail_strips_minted_keys(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "booting\nUNSLOTH_START_API_KEY: sk-unsloth-feedfacefeedface\nerror: load failed\n",
        encoding = "utf-8",
    )

    tail = start._redacted_log_tail(log)

    assert "sk-unsloth-feedfacefeedface" not in tail
    assert "sk-unsloth-[redacted]" in tail
    assert "error: load failed" in tail


def test_startup_failure_output_redacts_minted_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(start.tempfile, "gettempdir", lambda: str(tmp_path))
    fake = SimpleNamespace(pid = 4242, poll = lambda: 1)

    def fake_popen(command, **kwargs):
        kwargs["stdout"].write(b"UNSLOTH_START_API_KEY: sk-unsloth-secretsecret\nload failed\n")
        kwargs["stdout"].flush()
        return fake

    monkeypatch.setattr(start.subprocess, "Popen", fake_popen)

    with pytest.raises(start.typer.Exit):
        start._start_studio_server(BASE, "owner/model-GGUF", start.LoadOptions())

    err = capsys.readouterr().err
    assert "stopped before it was ready" in err
    assert "sk-unsloth-secretsecret" not in err
    assert "sk-unsloth-[redacted]" in err


def test_codex_preflight_failure_tears_down_auto_served(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started.update(base = base, model = model)
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(
        start, "_shutdown_server", lambda server: started.__setitem__("down", server)
    )
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "model_identifier": "transformers-model"}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["codex", "--model", "unsloth/Qwen3-1.7B", "--launch"]
    )
    assert result.exit_code != 0, result.output
    assert "GGUF" in result.output
    assert started.get("down") is fake


def test_no_serve_preserves_error(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {"called": False}
    monkeypatch.setattr(
        start, "_start_studio_server", lambda *a, **k: started.__setitem__("called", True)
    )
    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF", "--no-serve"]
    )
    assert result.exit_code == 1
    assert "No running Unsloth server" in result.output
    assert started["called"] is False


def test_no_launch_never_serves(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {"called": False}
    monkeypatch.setattr(
        start, "_start_studio_server", lambda *a, **k: started.__setitem__("called", True)
    )
    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF", "--no-launch"]
    )
    assert result.exit_code == 1
    assert "No running Unsloth server" in result.output
    assert started["called"] is False


def test_no_server_no_model_hints_model_flag(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    result = CliRunner().invoke(start.start_app, ["claude"])
    assert result.exit_code == 1
    assert "--model" in result.output


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://127.0.0.1", "http://127.0.0.1:8888"),
        ("http://127.0.0.1:8888", "http://127.0.0.1:8888"),
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("http://localhost", "http://localhost:8888"),
        ("http://[::1]", "http://[::1]:8888"),
        ("http://[::1]:8888", "http://[::1]:8888"),
        ("http://127.0.0.1:8888/studio", "http://127.0.0.1:8888"),
        ("http://127.0.0.1/studio", "http://127.0.0.1:8888"),
    ],
)
def test_effective_base(base, expected):
    assert start._effective_base(base) == expected


def test_auto_serve_normalizes_portless_url(fake_studio, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started["base"] = base
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))

    result = CliRunner().invoke(start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B-GGUF"])
    assert result.exit_code == 0, result.output
    assert started["base"] == "http://127.0.0.1:8888"


def test_connect_explicit_api_key_skips_mint(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)


def test_write_openclaw_config_fresh(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["models"]["providers"]["unsloth"]
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-unsloth-abc"
    assert provider["api"] == "openai-completions"
    assert provider["models"] == [
        {
            "id": MODEL["id"],
            "name": MODEL["id"],
            "contextWindow": MODEL["context_length"],
            "maxTokens": 32000,
        }
    ]
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["agents"]["defaults"]["workspace"] == str(tmp_path / "workspace")
    assert (tmp_path / "workspace").is_dir()
    assert config["gateway"]["mode"] == "local"
    assert config["gateway"]["auth"]["mode"] == "none"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("window, expected", [(32_768, 8_192), (131_072, 32_000)])
def test_openclaw_output_limit_follows_the_context(tmp_path, window, expected):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", {**MODEL, "context_length": window}, path)
    model = json.loads(path.read_text())["models"]["providers"]["unsloth"]["models"][0]
    assert model["maxTokens"] == expected


@pytest.mark.parametrize("max_tokens, expected", [("40000", 40000), ("200000", 65536)])
def test_connect_openclaw_max_tokens(fake_studio, tmp_path, max_tokens, expected):
    result = CliRunner().invoke(
        start.start_app, ["openclaw", "--no-launch", "--max-tokens", max_tokens]
    )
    assert result.exit_code == 0, result.output
    assert "--max-tokens" not in _launch_command(result.output)
    config = json.loads((tmp_path / "agents" / "openclaw" / "openclaw.json").read_text())
    assert config["models"]["providers"]["unsloth"]["models"][0]["maxTokens"] == expected


def test_write_openclaw_config_clears_per_agent_path_overrides(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {"workspace": "/old/default"},
                    "list": [
                        {
                            "id": "main",
                            "default": True,
                            "workspace": "/old/main-workspace",
                            "agentDir": "/old/main-agent",
                            "model": "keep/me",
                        },
                        {
                            "id": "reviewer",
                            "workspace": "/old/reviewer-workspace",
                            "agentDir": "/old/reviewer-agent",
                        },
                    ],
                }
            }
        )
    )

    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)

    agents = json.loads(path.read_text())["agents"]
    assert agents["defaults"]["workspace"] == str(tmp_path / "workspace")
    assert agents["list"] == [
        {"id": "main", "default": True, "model": "keep/me"},
        {"id": "reviewer"},
    ]


def test_write_openclaw_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "agents": {"defaults": {"temperature": 0.5}},
                "models": {"mode": "replace", "providers": {"openrouter": {"baseUrl": "x"}}},
            }
        )
    )
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["theme"] == "dark"
    assert config["agents"]["defaults"]["temperature"] == 0.5
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["models"]["mode"] == "replace"
    assert config["models"]["providers"]["openrouter"]["baseUrl"] == "x"
    assert config["models"]["providers"]["unsloth"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == before


def test_write_openclaw_config_corrupt_left_alone(tmp_path, capsys):
    path = tmp_path / "openclaw.json"
    path.write_text("{not json")
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == "{not json"
    assert "couldn't parse" in capsys.readouterr().err


def test_connect_openclaw_no_launch(fake_studio, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "openclaw" in result.output
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    _assert_env_set(result.output, "OPENCLAW_CONFIG_PATH", str(config_path))
    _assert_env_set(result.output, "OPENCLAW_STATE_DIR", str(tmp_path / "agents" / "openclaw"))
    for name in start._OPENCLAW_ENV_UNSET:
        _assert_env_unset(result.output, name)
    config = json.loads(config_path.read_text())
    assert config["models"]["providers"]["unsloth"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["agents"]["defaults"]["skipBootstrap"] is True
    assert config["agents"]["defaults"]["workspace"] == "${OPENCLAW_WORKSPACE_DIR}"
    assert config["memory"]["search"] == {
        "provider": "openai-compatible",
        "model": "unsloth/bge-small-en-v1.5",
        "fallback": "none",
        "remote": {"baseUrl": f"{BASE}/v1", "apiKey": "sk-unsloth-feedfacefeedface"},
    }
    _assert_env_cwd(result.output, "OPENCLAW_WORKSPACE_DIR")
    assert _launch_command(result.output) == ["openclaw", "tui", "--local"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_connect_openclaw_wsl_windows_shim_translates_live_workspace(
    fake_studio, tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/openclaw"
    )
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])

    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config = json.loads(config_path.read_text())
    assert config["agents"]["defaults"]["workspace"] == "${OPENCLAW_WORKSPACE_DIR}"
    _assert_env_cwd(result.output, "OPENCLAW_WORKSPACE_DIR")
    assert "OPENCLAW_WORKSPACE_DIR/p" in result.output
    assert "PWD/p" in result.output
    assert not (config_path.parent / "workspace").exists()


def test_connect_openclaw_saved_recipe_workspace_is_not_clobbered(
    fake_studio, tmp_path, monkeypatch
):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    monkeypatch.chdir(project_a)
    recipe_a = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    monkeypatch.chdir(project_b)
    recipe_b = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])

    assert recipe_a.exit_code == 0, recipe_a.output
    assert recipe_b.exit_code == 0, recipe_b.output
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config = json.loads(config_path.read_text())
    assert config["agents"]["defaults"]["workspace"] == "${OPENCLAW_WORKSPACE_DIR}"
    _assert_env_cwd(recipe_a.output, "OPENCLAW_WORKSPACE_DIR")
    _assert_env_cwd(recipe_b.output, "OPENCLAW_WORKSPACE_DIR")
    assert str(project_a) not in recipe_a.output
    assert str(project_b) not in recipe_b.output


def test_connect_openclaw_no_launch_keeps_explicit_subcommand(fake_studio):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", "crestodian"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "crestodian"]


def test_connect_openclaw_no_launch_passes_global_flags_through(fake_studio):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", "--profile", "test"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "--profile", "test"]


def test_connect_openclaw_no_launch_keeps_explicit_tui(fake_studio):
    result = CliRunner().invoke(
        start.start_app, ["openclaw", "--no-launch", "tui", "--message", "hi"]
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "tui", "--message", "hi"]


def test_write_opencode_config_fresh(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": "sk-unsloth-abc"}
    assert provider["models"] == {
        MODEL["id"]: {
            "name": MODEL["id"],
            "limit": {"context": 131072, "input": 131072, "output": 32_000},
        }
    }
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "disabled_providers" not in config
    assert config["compaction"] == {"auto": True, "reserved": 131072 // 10}


@pytest.mark.parametrize(
    "window, max_tokens, expected",
    [
        (16_384, None, 4_096),
        (32_768, None, 8_192),
        (131_072, None, 32_000),
        (143_616, None, 32_000),
        (143_616, 65_536, 65_536),
        (143_616, 200_000, 71_808),
        (32_768, 4_000, 4_000),
    ],
)
def test_opencode_output_limit(window, max_tokens, expected):
    assert start.opencode_output_limit(window, max_tokens) == expected


def test_opencode_max_tokens_sets_limit_and_raises_opencode_ceiling(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "65536"]
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    config = json.loads(config_path.read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit == {"context": 131072, "input": 131072, "output": 65536}
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "65536")


def test_opencode_limit_input_keeps_compaction_off_the_output_limit(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, max_tokens = 65536)
    config = json.loads(path.read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["input"] == limit["context"] == 131072
    assert config["compaction"]["reserved"] == 131072 // 10


@pytest.mark.parametrize(
    "window, expected_reserved, expected_compacts_at",
    [
        (16_384, 4_096, 12_288),
        (32_768, 8_192, 24_576),
        (131_072, 13_107, 117_965),
        (262_144, 26_214, 235_930),
    ],
)
def test_opencode_compaction_reserved(window, expected_reserved, expected_compacts_at):
    reserved = start.opencode_compaction_reserved(window, start.opencode_output_limit(window))
    assert reserved == expected_reserved
    assert window - reserved == expected_compacts_at


def test_opencode_subagent_drops_the_compaction_a_normal_session_wrote(tmp_path):
    path = tmp_path / "opencode.json"
    small = {**MODEL, "context_length": 16_384}
    start.write_opencode_config(BASE, "sk-unsloth-abc", small, path)
    assert json.loads(path.read_text())["compaction"] == {"auto": True, "reserved": 4_096}
    start.write_opencode_config(BASE, "sk-unsloth-abc", small, path, as_subagent = True)
    assert "compaction" not in json.loads(path.read_text())


def test_opencode_max_tokens_without_a_window_warns(capsys):
    assert start._opencode_output_env({"id": "m"}, 65536) == {}
    assert "--max-tokens is ignored" in capsys.readouterr().err


def test_opencode_max_tokens_under_ceiling_leaves_opencode_env_alone(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "16000"]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["output"] == 16000
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in result.output


def test_opencode_max_tokens_raises_a_smaller_inherited_ceiling(fake_studio, monkeypatch):
    monkeypatch.setenv("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "8000")
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "16000"]
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "16000")


def test_opencode_max_tokens_recipe_keeps_a_larger_inherited_ceiling(fake_studio, monkeypatch):
    monkeypatch.setenv("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "100000")
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "65536"]
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "100000")


def test_opencode_max_tokens_past_half_the_window_is_capped(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "120000"]
    )
    assert result.exit_code == 0, result.output
    assert "leaves too little" in result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["output"] == 131072 // 2


def test_write_opencode_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "theme": "tokyonight",
                "disabled_providers": ["ollama", "unsloth"],
                "provider": {"anthropic": {"name": "Anthropic"}},
            }
        )
    )
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["theme"] == "tokyonight"
    assert config["disabled_providers"] == ["ollama", "unsloth"]
    assert config["provider"]["anthropic"]["name"] == "Anthropic"
    assert config["provider"][start._OPENCODE_PROVIDER]["options"]["baseURL"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == before


def test_write_opencode_config_keeps_foreign_disabled_providers(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"disabled_providers": ["openai", "gemini"]}))
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["disabled_providers"] == ["openai", "gemini"]


def test_write_opencode_config_as_subagent_preserves_parent_model(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "model": "anthropic/claude-sonnet-4-5",
                "small_model": "anthropic/claude-haiku-4-5",
                "compaction": {"auto": False},
            }
        )
    )
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    start.write_opencode_config(
        BASE,
        "sk-unsloth-abc",
        local,
        path,
        as_subagent = True,
    )
    config = json.loads(path.read_text())
    assert config["model"] == "anthropic/claude-sonnet-4-5"
    assert config["small_model"] == "anthropic/claude-haiku-4-5"
    assert config["compaction"] == {"auto": False}
    agent = config["agent"]["unsloth"]
    assert agent["mode"] == "subagent"
    assert agent["model"] == f"{start._OPENCODE_PROVIDER}/{local['id']}"
    assert "local agent" in agent["description"].lower()
    assert local["id"] in config["provider"][start._OPENCODE_PROVIDER]["models"]


def test_opencode_subagent_inline_keeps_parent_provider_filters(monkeypatch, tmp_path):
    config_path = tmp_path / "opencode.json"
    inherited = {
        "theme": "tokyonight",
        "enabled_providers": ["anthropic"],
    }
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(
                {
                    "enabled_providers": ["opencode-go"],
                    "disabled_providers": ["ollama", start._OPENCODE_PROVIDER],
                    "subagent_depth": 0,
                }
            ),
            stderr = "",
        )

    monkeypatch.setattr(start.subprocess, "run", run)
    permission = {"edit": "allow"}
    inline = start._opencode_subagent_inline_config(config_path, permission)

    assert captured["command"] == ["/usr/bin/opencode", "debug", "config"]
    assert captured["env"]["OPENCODE_CONFIG"] == str(config_path)
    assert inline == {
        "theme": "tokyonight",
        "enabled_providers": [
            "anthropic",
            "opencode-go",
            start._OPENCODE_PROVIDER,
        ],
        "disabled_providers": ["ollama"],
        "subagent_depth": 1,
        "permission": permission,
    }


def test_opencode_subagent_inline_preserves_positive_depth(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps({"subagent_depth": 3}),
            stderr = "",
        ),
    )

    inline = start._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["subagent_depth"] == 3


def test_opencode_subagent_inline_merges_inherited_filters_without_binary(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps(
            {
                "enabled_providers": ["opencode-go"],
                "disabled_providers": ["ollama", start._OPENCODE_PROVIDER],
            }
        ),
    )
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)

    inline = start._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["enabled_providers"] == ["opencode-go", start._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == ["ollama"]
    assert inline["subagent_depth"] == 1


def test_opencode_v2_subagent_uses_native_depth_without_debug_probe(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"enabled_providers": ["anthropic"], "subagent_depth": 2}),
    )
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("V2 debug config is not a resolved object"),
    )

    inline = start._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "subagent_depth" not in inline
    assert inline["enabled_providers"] == ["anthropic", start._OPENCODE_PROVIDER]
    assert inline["experimental"] == {"subagent_depth": 2}


def test_opencode_v2_subagent_does_not_override_configured_depth(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising = False)

    inline = start._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "experimental" not in inline


def _opencode_inline_config(output: str) -> dict:
    name = "OPENCODE_CONFIG_CONTENT"
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(f"export {name}="):
            return json.loads(shlex.split(line.removeprefix(f"export {name}="))[0])
        prefix = f'$env:{name} = "'
        if line.startswith(prefix) and line.endswith('"'):
            escaped = line[len(prefix) : -1]
            value = escaped.replace("`$", "$").replace('`"', '"').replace("``", "`")
            return json.loads(value)
    raise AssertionError(f"{name} not found in:\n{output}")


def test_opencode_inline_scopes_session_to_studio_provider(fake_studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["enabled_providers"] == [start._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == []
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert inline["small_model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"


def test_opencode_passthrough_flags_omit_model_flag(fake_studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--dir", "repo"])
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "--dir", "repo"]
    assert "--model" not in command
    assert (
        _opencode_inline_config(result.output)["model"]
        == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    )


def test_opencode_passthrough_subcommand_omits_model_flag(fake_studio):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "serve"])
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[0] == "opencode"
    assert command[1] == "serve"
    assert "--model" not in command


def test_connect_opencode_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "opencode" in result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    _assert_env_set(result.output, "OPENCODE_CONFIG", str(config_path))
    inline_config = _opencode_inline_config(result.output)
    config = json.loads(config_path.read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["options"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "disabled_providers" not in config
    assert "enabled_providers" not in config
    assert inline_config == {
        "model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "small_model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "enabled_providers": [start._OPENCODE_PROVIDER],
        "disabled_providers": [],
    }
    assert _launch_command(result.output) == ["opencode"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


def test_connect_opencode_v2_no_launch_uses_private_server(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "--standalone"]
    assert _opencode_inline_config(result.output)["model"] == (
        f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    )
    assert "enabled_providers" not in _opencode_inline_config(result.output)
    assert "disabled_providers" not in _opencode_inline_config(result.output)
    assert f"provider policies must allow '{start._OPENCODE_PROVIDER}'" in result.output


def test_connect_opencode_v2_no_launch_uses_resolved_off_path_binary(
    fake_studio, monkeypatch, tmp_path
):
    binary = tmp_path / ".opencode" / "bin" / "opencode2"
    monkeypatch.setattr(
        start,
        "_opencode_command",
        lambda: (str(binary), True),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == [str(binary), "--standalone"]


def test_connect_opencode_v2_models_uses_private_server(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "models"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "models", "--standalone"]


def test_connect_opencode_as_subagent_preserves_cloud_parent(fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(
        start, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
    )
    result = CliRunner().invoke(
        start.start_app,
        [
            "opencode",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"] + ":UD-Q4_K_XL",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode"]
    expected_model = f"{start._OPENCODE_PROVIDER}/{MODEL['id']}:UD-Q4_K_XL"
    assert _opencode_inline_config(result.output) == {
        "agent": {
            "unsloth": {
                "description": start._SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": expected_model,
                "prompt": start._SUBAGENT_INSTRUCTIONS,
            }
        }
    }
    path = tmp_path / "agents" / "opencode-subagent" / "opencode.json"
    config = json.loads(path.read_text())
    assert "model" not in config
    assert "small_model" not in config
    assert "compaction" not in config
    agent = config["agent"]["unsloth"]
    assert agent["model"] == expected_model
    assert "Unsloth is available as @unsloth and in /models." in result.output


def test_claude_subagent_allowed_tools_precede_forwarded_delimiter(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "--", "--resume", "abc123"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    allowed = next(arg for arg in command if arg.startswith("--allowedTools="))
    assert command.index(allowed) < command.index("--resume")


def test_claude_subagent_forwards_positional_prompt(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "fix the failing test"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-1] == "fix the failing test"
    assert "--allowedTools" not in command


def test_opencode_subagent_installs_binary_before_filter_inspection(fake_studio, monkeypatch):
    installed = {}
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode" if installed.get("done") else None,
    )

    def require(name, hint, launch):
        assert launch is True
        installed["done"] = True
        installed["name"] = name

    monkeypatch.setattr(start, "_require_agent_for_launch", require)
    inspected = {}

    def inline(path, permission, **kwargs):
        inspected["binary"] = start._which_with_install_dirs("opencode")
        return {}

    monkeypatch.setattr(start, "_opencode_subagent_inline_config", inline)
    monkeypatch.setattr(start, "_run", lambda *a, **k: None)

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 0, result.output
    assert installed["name"] == "opencode"
    assert inspected["binary"] == "/usr/local/bin/opencode"


def test_opencode_subagent_pins_agent_in_inline_overlay(fake_studio, monkeypatch):
    monkeypatch.setattr(
        start, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
    )
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--as-subagent", "--no-launch", "--model", MODEL["id"] + ":UD-Q4_K_XL"],
    )
    assert result.exit_code == 0, result.output
    agent = _opencode_inline_config(result.output)["agent"]["unsloth"]
    assert agent["mode"] == "subagent"
    assert agent["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}:UD-Q4_K_XL"
    assert agent["prompt"] == start._SUBAGENT_INSTRUCTIONS
    assert agent["description"] == start._SUBAGENT_DESCRIPTION


def test_connect_opencode_subagent_yolo_no_launch_stays_append_safe(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda *_: True)
    captured = {}

    def inline(path, permission, **kwargs):
        captured["permission"] = permission
        return {"permission": permission}

    monkeypatch.setattr(start, "_opencode_subagent_inline_config", inline)
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--as-subagent", "--no-launch", "--yolo"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode"]
    assert "--auto" not in result.output
    assert captured["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "task": "allow",
        "external_directory": {"*": "allow"},
    }
    assert _opencode_inline_config(result.output)["permission"] == captured["permission"]


@pytest.fixture()
def hermes_config(tmp_path):
    return tmp_path / "config.yaml"


def test_write_hermes_config_fresh(hermes_config):
    yaml = pytest.importorskip("yaml")
    start.write_hermes_config(BASE, MODEL, hermes_config)
    config = yaml.safe_load(hermes_config.read_text())
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["model"]["default"] == MODEL["id"]
    assert config["model"]["api_mode"] == "openai"
    assert config["model"]["context_length"] == MODEL["context_length"]
    assert config["compression"] == {"enabled": True, "threshold": 0.9}
    assert "auxiliary" not in config
    provider = config["providers"]["unsloth"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["api_mode"] == "openai"
    assert provider["key_env"] == "UNSLOTH_API_KEY"
    assert "sk-unsloth" not in hermes_config.read_text()


def test_write_hermes_config_small_window_claims_floor(hermes_config):
    yaml = pytest.importorskip("yaml")
    small = {"id": "unsloth/Qwen3-1.7B-GGUF", "context_length": 40960}
    start.write_hermes_config(BASE, small, hermes_config)
    config = yaml.safe_load(hermes_config.read_text())
    # 0.5625 = 0.9 * 40960 / 65536: 90% of the real window.
    assert config["model"]["context_length"] == 65536
    assert config["compression"] == {"enabled": True, "threshold": 0.5625}
    assert config["auxiliary"]["compression"]["context_length"] == 65536
    assert config["model"]["ollama_num_ctx"] == 65536


def test_write_hermes_config_large_window_drops_claimed_num_ctx(hermes_config):
    yaml = pytest.importorskip("yaml")
    start.write_hermes_config(BASE, {"id": "small", "context_length": 40960}, hermes_config)
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert "ollama_num_ctx" not in yaml.safe_load(hermes_config.read_text())["model"]


def test_write_hermes_config_keeps_user_num_ctx(hermes_config):
    yaml = pytest.importorskip("yaml")
    hermes_config.write_text(yaml.safe_dump({"model": {"ollama_num_ctx": 100000}}))
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert yaml.safe_load(hermes_config.read_text())["model"]["ollama_num_ctx"] == 100000


def test_write_hermes_config_preserves_and_idempotent(hermes_config):
    yaml = pytest.importorskip("yaml")
    hermes_config.write_text(
        yaml.safe_dump(
            {
                "terminal": {"backend": "local"},
                "model": {"temperature": 0.7},
                "providers": {"openrouter": {"base_url": "https://openrouter.ai/api/v1"}},
            }
        )
    )
    start.write_hermes_config(BASE, MODEL, hermes_config)
    config = yaml.safe_load(hermes_config.read_text())
    assert config["terminal"] == {"backend": "local"}
    assert config["model"]["temperature"] == 0.7
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    before = hermes_config.read_text()
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert hermes_config.read_text() == before


def test_write_hermes_config_preserves_non_mapping_file(hermes_config, capsys):
    pytest.importorskip("yaml")
    original = "- just\n- a\n- list\n"
    hermes_config.write_text(original)
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert hermes_config.read_text() == original
    assert "couldn't parse" in capsys.readouterr().err


def test_connect_hermes_no_launch(fake_studio, tmp_path):
    yaml = pytest.importorskip("yaml")
    result = CliRunner().invoke(start.start_app, ["hermes", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "UNSLOTH_API_KEY", "sk-unsloth-feedfacefeedface")
    home = tmp_path / "agents" / "hermes"
    _assert_env_set(result.output, "HERMES_HOME", str(home))
    assert "hermes" in result.output
    config = yaml.safe_load((home / "config.yaml").read_text())
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    assert config["model"]["default"] == MODEL["id"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


def test_write_pi_config_fresh(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["providers"]["unsloth"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-unsloth-abc"
    assert provider["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]


def test_write_pi_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"providers": {"google": {"api": "gemini"}}}))
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["providers"]["google"] == {"api": "gemini"}
    assert config["providers"]["unsloth"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == before


def _pi_user_agent_dir(tmp_path, monkeypatch) -> Path:
    user_home = tmp_path / "user-home"
    agent_dir = user_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising = False)
    return agent_dir


def test_connect_pi_no_launch(fake_studio, tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "extensions" / "mine.ts").write_text("export default () => {};\n")
    (user_agent_dir / "settings.json").write_text(json.dumps({"packages": ["npm:pi-mine"]}))
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "pi"
    _assert_env_set(result.output, "HOME", str(home))
    _assert_env_set(result.output, "PI_CODING_AGENT_DIR", str(home / ".pi" / "agent"))
    assert f"pi --provider unsloth --model {MODEL['id']}" in result.output
    config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
    assert config["providers"]["unsloth"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["providers"]["unsloth"]["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)
    assert (home / ".pi" / "agent" / "extensions" / "mine.ts").is_file()
    settings = json.loads((home / ".pi" / "agent" / "settings.json").read_text())
    assert settings == {"packages": ["npm:pi-mine"]}
    assert not (user_agent_dir / "models.json").exists()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_links_resources_and_keeps_config_private(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    user_home = user_agent_dir.parent.parent
    for name in ("extensions", "skills", "npm", "git"):
        (user_agent_dir / name).mkdir()
    (user_home / ".agents" / "skills").mkdir(parents = True)
    (user_agent_dir / "auth.json").write_text('{"google": "user-key"}\n')
    (user_agent_dir / "models.json").write_text('{"providers": {"mine": {}}}\n')
    (user_agent_dir / "sessions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultProvider": "google",
                "theme": "light",
                "packages": [
                    "npm:pi-mine@1.2.3",
                    "git:github.com/me/pi-tools",
                    "../../src/local-extension",
                    {"source": "~/other-extension", "extensions": ["index.ts"]},
                ],
                "extensions": [
                    "extensions/extra.ts",
                    "-extensions/off.ts",
                    "root.ts",
                    "/abs/ext.ts",
                    "!local/old.ts",
                    "-~/literal.ts",
                ],
                "skills": ["skills/*", "../shared/*"],
            }
        )
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, agent_dir / "models.json")

    start.write_pi_user_resources(agent_dir, session_home)

    for name in ("extensions", "skills", "npm", "git"):
        assert (agent_dir / name).is_symlink()
        assert (agent_dir / name).resolve() == (user_agent_dir / name).resolve()
    assert not (agent_dir / "prompts").exists()
    assert (session_home / ".agents" / "skills").resolve() == (
        user_home / ".agents" / "skills"
    ).resolve()
    for private in ("auth.json", "sessions"):
        assert not (agent_dir / private).exists()
    assert "mine" not in json.loads((agent_dir / "models.json").read_text())["providers"]
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == {
        "packages": [
            "npm:pi-mine@1.2.3",
            "git:github.com/me/pi-tools",
            str(user_home / "src" / "local-extension"),
            {"source": str(user_home / "other-extension"), "extensions": ["index.ts"]},
        ],
        "extensions": [
            "extensions/extra.ts",
            "-extensions/off.ts",
            str(user_agent_dir / "root.ts"),
            "/abs/ext.ts",
            "!local/old.ts",
            f"!{user_agent_dir / 'local' / 'old.ts'}",
            "-~/literal.ts",
        ],
        "skills": ["skills/*", "../shared/*", str(user_home / ".pi" / "shared" / "*")],
    }


def test_write_pi_user_resources_refreshes_a_persisted_session(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    user_settings = user_agent_dir / "settings.json"
    user_settings.write_text(json.dumps({"packages": ["npm:old", "npm:kept"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["packages"].append("npm:session-only")
    settings["theme"] = "dark"
    settings_path.write_text(json.dumps(settings))
    user_settings.write_text(json.dumps({"packages": ["npm:kept", "npm:new"]}))
    (user_agent_dir / "extensions").rmdir()

    start.write_pi_user_resources(agent_dir, session_home)

    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only", "npm:kept", "npm:new"],
        "theme": "dark",
    }
    assert not (agent_dir / "extensions").exists() and not (agent_dir / "extensions").is_symlink()

    user_settings.unlink()
    start.write_pi_user_resources(agent_dir, session_home)
    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only"],
        "theme": "dark",
    }
    assert not (agent_dir / start._PI_USER_RESOURCES_MANIFEST).exists()


def test_write_pi_user_resources_leaves_session_dirs_and_user_files_alone(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "npm" / "node_modules" / "pi-mine").mkdir(parents = True)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "extensions" / "mine.ts").write_text("mine\n")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    (agent_dir / "npm").mkdir(parents = True)
    (agent_dir / "npm" / "session.txt").write_text("session\n")

    start.write_pi_user_resources(agent_dir, session_home)

    assert not (agent_dir / "npm").is_symlink()
    assert (agent_dir / "npm" / "session.txt").read_text() == "session\n"
    assert (agent_dir / "extensions" / "mine.ts").read_text() == "mine\n"
    shutil.rmtree(session_home)
    assert (user_agent_dir / "extensions" / "mine.ts").read_text() == "mine\n"
    assert (user_agent_dir / "npm" / "node_modules" / "pi-mine").is_dir()


def test_write_pi_user_resources_uses_inherited_agent_dir(tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    configured = tmp_path / "custom-pi"
    (configured / "extensions").mkdir(parents = True)
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(configured))

    start.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()

    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    start.write_pi_user_resources(agent_dir, session_home)
    assert not (agent_dir / "extensions").exists()


def test_write_pi_user_resources_resolves_a_relative_agent_dir_from_the_launch_dir(
    tmp_path, monkeypatch
):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    configured = tmp_path / "launch" / "custom-pi"
    (configured / "extensions").mkdir(parents = True)
    (configured / "settings.json").write_text(json.dumps({"packages": ["../src/local-ext"]}))
    monkeypatch.chdir(tmp_path / "launch")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "custom-pi")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)

    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == {"packages": [str(tmp_path / "launch" / "src" / "local-ext")]}


def test_is_junction_reads_the_reparse_tag_before_python_3_12(tmp_path, monkeypatch):
    # Python 3.13 also defines is_junction on a pathlib base class.
    for cls in type(tmp_path).__mro__:
        if "is_junction" in vars(cls):
            monkeypatch.delattr(cls, "is_junction")
    tags = {"junction": 0xA0000003, "symlink": 0xA000000C}

    def lstat(path):
        name = Path(path).name
        if name not in tags:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_reparse_tag = tags[name])

    monkeypatch.setattr(start.os, "lstat", lstat)
    assert start._is_junction(tmp_path / "junction")
    assert not start._is_junction(tmp_path / "symlink")
    assert not start._is_junction(tmp_path / "missing")


def test_link_user_dir_replaces_a_junction_from_an_earlier_run(tmp_path, monkeypatch):
    target = tmp_path / "session" / "extensions"
    target.mkdir(parents = True)
    source = tmp_path / "user" / "extensions"
    source.mkdir(parents = True)
    monkeypatch.setattr(start, "_is_junction", lambda path: path == target and path.is_dir())

    start._link_user_dir(source, target)

    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_write_pi_user_resources_skips_a_windows_pi_under_wsl(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(json.dumps({"packages": ["npm:pi-mine"]}))
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)

    assert not agent_dir.exists()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_clears_a_session_a_linux_pi_prepared(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm:user-pkg"], "extensions": ["extensions/mine.ts"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    start.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").is_symlink()
    settings = json.loads((agent_dir / "settings.json").read_text())
    settings["packages"].append("npm:session-only")
    (agent_dir / "settings.json").write_text(json.dumps(settings))

    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    start.write_pi_user_resources(agent_dir, session_home)

    assert not (agent_dir / "extensions").exists()
    assert json.loads((agent_dir / "settings.json").read_text()) == {
        "packages": ["npm:session-only"],
    }
    assert not (agent_dir / start._PI_USER_RESOURCES_MANIFEST).exists()
    assert (user_agent_dir / "extensions").is_dir()


@pytest.mark.parametrize("session_command", [None, ["npm", "--silent"]])
def test_write_pi_user_resources_clears_npm_command_whole(tmp_path, monkeypatch, session_command):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"npmCommand": ["npm", "--registry=x"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    start.write_pi_user_resources(agent_dir, session_home)
    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == [
        "npm",
        "--registry=x",
    ]
    if session_command is not None:
        settings = json.loads((agent_dir / "settings.json").read_text())
        settings["npmCommand"] = session_command
        (agent_dir / "settings.json").write_text(json.dumps(settings))

    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    start.write_pi_user_resources(agent_dir, session_home)

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings.get("npmCommand") == session_command


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_reanchors_when_a_real_session_dir_blocks_the_link(
    tmp_path, monkeypatch
):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "npm").mkdir()
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm/pkg"], "extensions": ["extensions/mine.ts"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    (agent_dir / "npm").mkdir(parents = True)

    start.write_pi_user_resources(agent_dir, session_home)

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert not (agent_dir / "npm").is_symlink()
    assert settings["packages"] == [str(user_agent_dir / "npm" / "pkg")]
    assert settings["extensions"] == ["extensions/mine.ts"]


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_keeps_a_disabled_global_skill_disabled(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    user_home = user_agent_dir.parent.parent
    disabled = user_home / ".agents" / "skills" / "off" / "SKILL.md"
    disabled.parent.mkdir(parents = True)
    disabled.write_text("disabled\n")
    (user_agent_dir / "settings.json").write_text(json.dumps({"skills": [f"-{disabled}"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)

    session_skill = session_home / ".agents" / "skills" / "off" / "SKILL.md"
    assert session_skill.exists()
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["skills"] == [f"-{disabled}", f"-{session_skill}"]


def test_write_pi_user_resources_copies_the_npm_command(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    command = ["mise", "exec", "node@20", "--", "npm"]
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm:pi-mine"], "npmCommand": command})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)

    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == command
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["npmCommand"] = ["pnpm"]
    settings_path.write_text(json.dumps(settings))
    start.write_pi_user_resources(agent_dir, session_home)
    assert json.loads(settings_path.read_text())["npmCommand"] == ["pnpm"]


def test_write_pi_user_resources_warns_on_an_unusable_agent_dir_override(
    tmp_path, monkeypatch, capsys
):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    missing = tmp_path / "not-a-directory"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"  {missing}  ")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)

    start.write_pi_user_resources(agent_dir, session_home)

    assert "PI_CODING_AGENT_DIR" in capsys.readouterr().err


def test_write_pi_user_resources_keeps_a_non_list_setting(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "settings.json").write_text(json.dumps({"themes": ["npm:user-theme"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)
    (agent_dir / "settings.json").write_text(json.dumps({"themes": {"name": "dark"}}))

    start.write_pi_user_resources(agent_dir, session_home)

    assert json.loads((agent_dir / "settings.json").read_text())["themes"] == {"name": "dark"}


@pytest.mark.parametrize("entry", ["", "   ", "."])
def test_pi_local_entry_leaves_degenerate_entries_alone(tmp_path, entry):
    assert start._pi_local_entry(entry, tmp_path, tmp_path, frozenset()) == entry


@pytest.mark.parametrize("dangling", [False, True])
def test_remove_overlay_entry_rmdirs_a_windows_directory_symlink(tmp_path, monkeypatch, dangling):
    # DeleteFileW refuses a directory symlink (WinError 5); is_dir() would follow a dangling target.
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep\n")
    target = tmp_path / "link"
    target.symlink_to(source, target_is_directory = True)
    if dangling:
        shutil.rmtree(source)
        source.mkdir()
        (source / "keep.txt").write_text("keep\n")
        target.unlink()
        target.symlink_to(tmp_path / "gone", target_is_directory = True)
    # POSIX lstat has no st_file_attributes; stand in for the Windows link attributes.
    real_lstat = os.lstat
    monkeypatch.setattr(
        start.os,
        "lstat",
        lambda p: SimpleNamespace(
            st_file_attributes = 0x10,
            st_reparse_tag = 0,
            st_mode = real_lstat(p).st_mode,
        ),
    )
    # rmdir on a link is POSIX-invalid, so record the routing instead of running it.
    monkeypatch.setattr(start.os, "name", "nt")
    calls = []
    monkeypatch.setattr(start.Path, "unlink", lambda self, **kw: calls.append("unlink"))
    monkeypatch.setattr(start.Path, "rmdir", lambda self: calls.append("rmdir"))

    start._remove_overlay_entry(target)

    assert calls == ["rmdir"]
    assert (source / "keep.txt").read_text() == "keep\n"


@pytest.mark.parametrize("yolo", [False, True])
def test_connect_pi_as_subagent_preserves_cloud_parent(fake_studio, tmp_path, yolo):
    args = [
        "pi",
        "--as-subagent",
        "--no-launch",
        "--model",
        MODEL["id"] + ":UD-Q4_K_XL",
    ]
    if yolo:
        args.insert(2, "--yolo")
    result = CliRunner().invoke(
        start.start_app,
        args,
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:2] == ["pi", "--extension"]
    assert command[2].endswith("unsloth_cli/pi_subagent.ts")
    assert ("--approve" in command) is yolo
    assert "--provider" not in command
    assert "--model" not in command
    assert "PI_CODING_AGENT_DIR" not in result.output
    assert "export HOME=" not in result.output
    assert "UNSLOTH_PI_SUBAGENT_API_KEY" not in result.output
    assert "sk-unsloth-feedfacefeedface" not in result.output
    config_path = tmp_path / "agents" / "pi-subagent" / "subagent.json"
    _assert_env_set(result.output, "UNSLOTH_PI_SUBAGENT_CONFIG", str(config_path))
    assert json.loads(config_path.read_text()) == {
        "baseUrl": f"{BASE}/v1",
        "apiKey": "sk-unsloth-feedfacefeedface",
        "model": MODEL["id"] + ":UD-Q4_K_XL",
        "contextWindow": 4096,
        "maxTokens": 1024,
        "approve": yolo,
    }
    assert "Ask Pi to spawn an Unsloth or local agent." in result.output


def _pi_generated_model(tmp_path, as_subagent):
    if as_subagent:
        return json.loads((tmp_path / "agents" / "pi-subagent" / "subagent.json").read_text())
    return json.loads((tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json").read_text())[
        "providers"
    ]["unsloth"]["models"][0]


@pytest.mark.parametrize("as_subagent", [False, True])
def test_connect_pi_output_limit(fake_studio, tmp_path, as_subagent):
    args = ["pi", "--no-launch", "--max-tokens", "40000"]
    if as_subagent:
        args.append("--as-subagent")
    for _ in range(2):
        result = CliRunner().invoke(start.start_app, args)
        assert result.exit_code == 0, result.output
        assert "--max-tokens" not in _launch_command(result.output)
        config = _pi_generated_model(tmp_path, as_subagent)
        assert config["maxTokens"] == 40000
        assert config["contextWindow"] == MODEL["context_length"]


@pytest.mark.parametrize("as_subagent", [False, True])
def test_connect_pi_output_limit_capped_at_half_window(fake_studio, tmp_path, as_subagent):
    args = ["pi", "--no-launch", "--max-tokens", "200000"]
    if as_subagent:
        args.append("--as-subagent")
    result = CliRunner().invoke(start.start_app, args)
    assert result.exit_code == 0, result.output
    assert "leaves too little" in result.output
    assert _pi_generated_model(tmp_path, as_subagent)["maxTokens"] == MODEL["context_length"] // 2


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_connect_pi_invalid_output_limit(fake_studio, tmp_path, value):
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch", "--max-tokens", value])
    assert result.exit_code != 0
    assert "--max-tokens" in result.output
    assert not list((tmp_path / "agents").rglob("models.json"))


@pytest.mark.parametrize("context", [{}, {"max_context_length": 32768}])
def test_write_pi_output_limit_context_metadata(tmp_path, context):
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "test-key", {"id": "test-model", **context}, path, max_tokens = 10000)
    model = json.loads(path.read_text())["providers"]["unsloth"]["models"][0]
    assert model["maxTokens"] == 10000
    if context:
        assert model["contextWindow"] == 32768
    else:
        assert "contextWindow" not in model


def test_connect_pi_no_launch_windows_relocates_userprofile(fake_studio, tmp_path, monkeypatch):
    user_home = _pi_user_agent_dir(tmp_path, monkeypatch).parent.parent
    # Avoid pathlib selecting WindowsPath on this POSIX runner.
    monkeypatch.setattr(start.Path, "home", lambda: user_home)
    monkeypatch.setattr(start.os, "name", "nt")
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "pi"
    assert f'$env:HOME = "{home}"' in result.output
    assert f'$env:USERPROFILE = "{home}"' in result.output


@pytest.fixture()
def dsh_patch(tmp_path):
    return tmp_path / "unsloth.patch.yml"


def _dsh_entries(path):
    yaml = pytest.importorskip("yaml")
    entries = yaml.safe_load(path.read_text())
    assert isinstance(entries, list), entries
    return {entry["id"]: entry for entry in entries}


def test_write_dsh_patch_fresh(dsh_patch):
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
    entries = _dsh_entries(dsh_patch)
    assert set(entries) == {"llm-pi-ai", "agent-default-model"}
    assert entries["llm-pi-ai"]["name"] == "@deepseek-ai/dsh-llm-pi-ai"
    assert entries["agent-default-model"]["name"] == "@deepseek-ai/dsh-agent-default-model"
    provider = entries["llm-pi-ai"]["config"]["providers"]["unsloth"]
    assert provider["api"] == "openai-completions"
    assert provider["baseURL"] == f"{BASE}/v1"
    assert provider["apiKeyEnv"] == "UNSLOTH_API_KEY"
    assert "sk-unsloth" not in dsh_patch.read_text()
    assert provider["compat"] == {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"}
    assert provider["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]
    assert entries["agent-default-model"]["config"] == {"provider": "unsloth", "model": MODEL["id"]}


@pytest.mark.parametrize("window, expected", [(32_768, 8_192), (143_616, 32_000)])
def test_pi_and_dsh_output_limit_follows_the_context(tmp_path, window, expected):
    model = {**MODEL, "context_length": window}
    start.write_pi_config(BASE, "sk-unsloth-abc", model, tmp_path / "models.json")
    start.write_pi_subagent_config(BASE, "sk-unsloth-abc", model, tmp_path / "subagent.json")
    start.write_dsh_patch(BASE, model, tmp_path / "unsloth.patch.yml")
    pi = json.loads((tmp_path / "models.json").read_text())["providers"]["unsloth"]["models"][0]
    subagent = json.loads((tmp_path / "subagent.json").read_text())
    patch = _dsh_entries(tmp_path / "unsloth.patch.yml")
    dsh = patch["llm-pi-ai"]["config"]["providers"]["unsloth"]["models"][0]
    assert pi["maxTokens"] == subagent["maxTokens"] == dsh["maxTokens"] == expected


def test_write_dsh_patch_without_window_omits_limits(dsh_patch):
    start.write_dsh_patch(BASE, {"id": "unsloth/unknown-window"}, dsh_patch)
    provider = _dsh_entries(dsh_patch)["llm-pi-ai"]["config"]["providers"]["unsloth"]
    assert provider["models"] == [{"id": "unsloth/unknown-window"}]


def test_write_dsh_patch_is_idempotent_and_follows_the_server(dsh_patch, capsys):
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
    before = dsh_patch.read_text()
    capsys.readouterr()
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
    assert dsh_patch.read_text() == before
    assert "Updated" not in capsys.readouterr().out
    start.write_dsh_patch("http://127.0.0.1:9999", {"id": "other"}, dsh_patch)
    entries = _dsh_entries(dsh_patch)
    provider = entries["llm-pi-ai"]["config"]["providers"]["unsloth"]
    assert provider["baseURL"] == "http://127.0.0.1:9999/v1"
    assert provider["models"] == [{"id": "other"}]
    assert entries["agent-default-model"]["config"]["model"] == "other"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["dsh", "web"]),
        (["--no-open"], ["dsh", "web", "--no-open"]),
        (["--port", "3099"], ["dsh", "web", "--port", "3099"]),
        (["--profile", "headless", "fix the bug"], ["dsh", "--profile", "headless", "fix the bug"]),
        (["--profile=headless", "fix"], ["dsh", "--profile=headless", "fix"]),
        (
            ["plugin", "--profile", "web", "add", "x"],
            ["dsh", "plugin", "--profile", "web", "add", "x"],
        ),
        (["web", "--no-open"], ["dsh", "web", "--no-open"]),
    ],
)
def test_dsh_command_selects_web_only_for_app_arguments(args, expected):
    assert start._dsh_command(args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["dsh", "web", "--patch", "P"]),
        (["--no-open"], ["dsh", "web", "--patch", "P", "--no-open"]),
        (["web", "--no-open"], ["dsh", "web", "--patch", "P", "--no-open"]),
        (
            ["--profile", "headless", "fix the bug"],
            ["dsh", "--patch", "P", "--profile", "headless", "fix the bug"],
        ),
        (["--profile=headless", "fix"], ["dsh", "--patch", "P", "--profile=headless", "fix"]),
        (["--dump-config"], ["dsh", "--patch", "P", "--dump-config"]),
        (
            ["--patch", "mine.yml", "--profile", "headless"],
            ["dsh", "--patch", "P", "--patch", "mine.yml", "--profile", "headless"],
        ),
        (["--patch=mine.yml", "web"], ["dsh", "--patch", "P", "--patch=mine.yml", "web"]),
        (
            ["plugin", "--profile", "web", "add", "x"],
            ["dsh", "plugin", "--profile", "web", "add", "x"],
        ),
        (["-V"], ["dsh", "-V"]),
        (["--version"], ["dsh", "--version"]),
    ],
)
def test_dsh_command_places_the_patch_where_dsh_parses_it(args, expected):
    assert start._dsh_command(args, "P") == expected


def test_connect_dsh_no_launch(fake_studio, tmp_path):
    yaml = pytest.importorskip("yaml")
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "UNSLOTH_API_KEY", "sk-unsloth-feedfacefeedface")
    home = tmp_path / "agents" / "dsh"
    _assert_env_set(result.output, "DSH_HOME", str(home))
    _assert_env_set(result.output, "DSH_TELEMETRY_DISABLED", "1")
    patch = home / "unsloth.patch.yml"
    assert _launch_command(result.output) == ["dsh", "web", "--patch", str(patch)]
    entries = {entry["id"]: entry for entry in yaml.safe_load(patch.read_text())}
    assert entries["agent-default-model"]["config"] == {"provider": "unsloth", "model": MODEL["id"]}
    assert entries["llm-pi-ai"]["config"]["providers"]["unsloth"]["baseURL"] == f"{BASE}/v1"
    assert not (home / "settings.yaml").exists()


def _vibe_settings(env: dict) -> tuple:
    return json.loads(env["VIBE_PROVIDERS"]), json.loads(env["VIBE_MODELS"])


def test_vibe_env(tmp_path):
    env = start._vibe_env(BASE, MODEL)
    providers, models = _vibe_settings(env)
    assert providers == [
        {
            "name": start._VIBE_PROVIDER,
            "api_base": f"{BASE}/v1",
            "api_key_env_var": start._VIBE_ENV_KEY,
            "api_style": "openai",
            "backend": "generic",
        }
    ]
    assert models == [
        {
            "name": MODEL["id"],
            "provider": start._VIBE_PROVIDER,
            "alias": start._VIBE_MODEL_ALIAS,
            "auto_compact_threshold": int(MODEL["context_length"] * 0.9),
        }
    ]
    assert env["VIBE_ACTIVE_MODEL"] == start._VIBE_MODEL_ALIAS
    assert env["VIBE_ENABLE_TELEMETRY"] == "false"
    assert json.loads(env["VIBE_ALLOWED_MODELS"]) == ["re:" + re.escape(MODEL["id"])]


def test_vibe_env_carries_temperature_and_odd_model_ids():
    model = {"id": 'C:\\models\\q"\U0001f600.gguf', "max_context_length": 32768}
    _, models = _vibe_settings(start._vibe_env(BASE, model, {"temperature": 0.7}))
    assert models[0]["name"] == model["id"]
    assert models[0]["temperature"] == 0.7
    assert models[0]["auto_compact_threshold"] == int(32768 * 0.9)
    _, models = _vibe_settings(start._vibe_env(BASE, {"id": "m"}))
    assert set(models[0]) == {"name", "provider", "alias"}


def test_connect_vibe_no_launch_uses_env_only(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["vibe", "--no-launch", "--yolo"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "UNSLOTH_API_KEY", "sk-unsloth-feedfacefeedface")
    _assert_env_set(result.output, "VIBE_ACTIVE_MODEL", start._VIBE_MODEL_ALIAS)
    assert "VIBE_HOME" not in result.output
    assert _launch_command(result.output) == ["vibe", "--auto-approve"]
    assert not (tmp_path / "agents" / "vibe").exists()


def test_vibe_launch_keeps_user_home(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/vibe")
    captured = _capture_launch(monkeypatch, ["vibe", "--temperature", "0.6", "-p", "hi"])
    assert captured["command"][-3:] == ["/usr/local/bin/vibe", "-p", "hi"]
    assert captured["env"].get("HOME") == os.environ.get("HOME")
    assert "VIBE_HOME" not in captured["env"]
    assert _vibe_settings(captured["env"])[1][0]["temperature"] == 0.6


@pytest.mark.skipif(shutil.which("vibe") is None, reason = "needs the mistral-vibe CLI")
def test_vibe_real_cli_reaches_unsloth(monkeypatch, tmp_path):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.headers.get("Authorization"), body))
            chunk = {
                "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": "stop"}]
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(start, "_connect", lambda *args, **kwargs: (base, "sk-unsloth-e2e", MODEL))
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
    monkeypatch.chdir(tmp_path)
    try:
        result = CliRunner().invoke(
            start.start_app,
            ["vibe", "--yolo", "--temperature", "0.5", "--disabled-tools", "*", "-p", "Reply pong"],
        )
    finally:
        server.shutdown()
    assert result.exit_code == 0, result.output
    assert requests, "vibe never called the Unsloth /v1 endpoint"
    auth, body = requests[0]
    assert auth == "Bearer sk-unsloth-e2e"
    assert body["model"] == MODEL["id"]
    assert body["temperature"] == 0.5


@pytest.mark.parametrize(
    ("argv", "status", "expected"),
    [
        ([], {"model_identifier": MODEL["id"], "inference": {"temperature": 1.0}}, 1.0),
        (
            ["--temperature", "0.3"],
            {"model_identifier": MODEL["id"], "inference": {"temperature": 1.0}},
            0.3,
        ),
        ([], {"model_identifier": "other/model", "inference": {"temperature": 1.0}}, None),
        ([], {}, None),
    ],
)
def test_vibe_sends_recommended_temperature(fake_studio, monkeypatch, argv, status, expected):
    monkeypatch.setattr(start, "_inference_status", lambda *args: status)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/vibe")
    captured = _capture_launch(monkeypatch, ["vibe", *argv])
    assert json.loads(captured["env"]["VIBE_MODELS"])[0].get("temperature") == expected


def test_vibe_install_hint_per_os(monkeypatch):
    monkeypatch.setattr(start.os, "name", "nt")
    assert start._vibe_install_hint() == start._VIBE_WINDOWS_INSTALL_HINT
    monkeypatch.setattr(start.os, "name", "posix")
    assert start._vibe_install_hint() == start._VIBE_POSIX_INSTALL_HINT


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_dsh_under_wsl_gets_the_windows_patch_path(fake_studio, monkeypatch):
    windows_path = r"\\wsl.localhost\Ubuntu\tmp\unsloth.patch.yml"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    shim = "/mnt/c/Users/x/AppData/Roaming/npm/dsh"
    monkeypatch.setattr(start.shutil, "which", lambda _: shim)
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: windows_path)
    captured = _capture_launch(monkeypatch, ["dsh", "--profile", "headless", "hi"])
    command = captured["command"]
    assert command[command.index("--patch") + 1] == windows_path, command


def test_dsh_yolo_sets_permission_mode(fake_studio):
    result = CliRunner().invoke(start.start_app, ["dsh", "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "danger-full-access")


def test_dsh_without_yolo_pins_the_safe_permission_mode(fake_studio):
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "workspace-write")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["dsh"], "workspace-write"),
        (["dsh", "--yolo"], "danger-full-access"),
    ],
)
def test_dsh_permission_mode_overrides_an_inherited_bypass(
    argv, expected, fake_studio, monkeypatch
):
    monkeypatch.setenv("DSH_PERMISSION_MODE", "danger-full-access")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, argv)
    assert captured["env"]["DSH_PERMISSION_MODE"] == expected


def test_start_dsh_forwards_reasoning_effort(fake_studio, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    captured = {}
    fake = SimpleNamespace(pid = 1, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        captured["server_options"] = server_options
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["patch"] = Path(command[command.index("--patch") + 1]).read_text()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)

    result = CliRunner().invoke(
        start.start_app,
        ["dsh", "--model", "unsloth/gemma-4-E2B-it-GGUF", "--reasoning-effort", "high"],
    )
    assert result.exit_code == 0, result.output
    assert captured["server_options"].reasoning_effort is None
    assert "--reasoning-effort" not in captured["command"]
    provider = yaml.safe_load(captured["patch"])[0]["config"]["providers"][start._DSH_PROVIDER]
    assert provider["compat"]["chatTemplateKwargs"] == {"reasoning_effort": "high"}


def test_wsl_bridge_names_flags_paths_not_scalars():
    env = {
        "CODEX_HOME": "/tmp/sess/codex",
        "HOME": "/tmp/sess/pi",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "4096",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8888",
        "USERPROFILE": r"C:\Users\x",
    }
    names = start._wsl_bridge_names(env, ("ANTHROPIC_API_KEY",))
    assert "CODEX_HOME/p" in names
    assert "HOME/p" in names
    assert "USERPROFILE/p" in names
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" in names
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW/p" not in names
    assert "ANTHROPIC_BASE_URL" in names
    assert "ANTHROPIC_API_KEY" in names


def test_merge_wslenv_dedups_on_base_name():
    merged = start._merge_wslenv("CODEX_HOME/p:FOO", ("CODEX_HOME/p", "BAR/p"))
    parts = merged.split(":")
    assert parts.count("CODEX_HOME/p") == 1
    assert "FOO" in parts and "BAR/p" in parts


def test_merge_wslenv_upgrades_existing_unflagged_entry():
    merged = start._merge_wslenv("HOME:FOO", ("HOME/p", "CODEX_HOME/p"))
    parts = merged.split(":")
    assert "HOME/p" in parts and "HOME" not in parts
    assert parts.count("HOME/p") == 1
    assert "FOO" in parts
    assert "CODEX_HOME/p" in parts


def test_powershell_quote_single_quotes_json():
    assert start._powershell_quote("--settings") == "--settings"
    assert start._powershell_quote("unsloth/gemma-4-26B") == "unsloth/gemma-4-26B"
    overlay = start._claude_settings_overlay("unsloth/gemma-4-26B")
    quoted = start._powershell_quote(overlay)
    assert quoted == "'" + overlay + "'"
    assert "\\" not in quoted
    assert start._powershell_quote("a'b") == "'a''b'"


_NATIVE_YOLO = {
    "claude": "--dangerously-skip-permissions",
    "codex": "--dangerously-bypass-approvals-and-sandbox",
    "hermes": "--yolo",
    "pi": "--approve",
}


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_yolo_routes_to_native_flag(fake_studio, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert native in result.output


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_no_yolo_omits_native_flag(fake_studio, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command and command[0] == agent, result.output
    assert native not in command


@pytest.mark.parametrize(
    "alias",
    ["--yolo", "--dangerously-skip-permissions", "--dangerously-bypass-approvals-and-sandbox"],
)
def test_yolo_aliases_are_interchangeable(fake_studio, alias):
    claude = CliRunner().invoke(start.start_app, ["claude", alias, "--no-launch"])
    assert claude.exit_code == 0, claude.output
    assert "--dangerously-skip-permissions" in claude.output
    assert "--dangerously-bypass-approvals-and-sandbox" not in claude.output

    codex = CliRunner().invoke(start.start_app, ["codex", alias, "--no-launch"])
    assert codex.exit_code == 0, codex.output
    assert "--dangerously-bypass-approvals-and-sandbox" in codex.output
    assert "--dangerously-skip-permissions" not in codex.output

    opencode = CliRunner().invoke(
        start.start_app,
        ["opencode", alias, "--no-launch", "run", "hello"],
    )
    assert opencode.exit_code == 0, opencode.output
    assert _launch_command(opencode.output) == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(opencode.output)


def test_yolo_opencode_bare_no_launch_uses_permission_fallback(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_yolo_opencode_run_uses_native_auto(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_v2_run_uses_standalone_and_native_auto(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == [
        "opencode2",
        "run",
        "--standalone",
        "hello",
        "--auto",
    ]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_v2_mini_uses_permission_fallback(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "mini"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "mini", "--standalone"]
    assert _opencode_inline_config(result.output)["permission"]["edit"] == "allow"


def test_yolo_opencode_tui_resume_uses_native_auto(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "--session", "sid"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "--session", "sid", "--auto"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_no_yolo_opencode_run_omits_native_auto(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", "run", "hello"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_bare_launch_uses_native_auto(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda *_: True)
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])
    assert captured["command"][1:] == [
        "--model",
        f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "--auto",
    ]
    assert "permission" not in json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])


def test_yolo_opencode_v2_bare_launch_omits_root_model(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode2")
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])

    assert captured["command"][0].endswith("opencode2")
    assert captured["command"][1:] == ["--standalone", "--auto"]
    assert "--model" not in captured["command"]


def test_yolo_opencode_native_auto_clears_prior_config_fallback(fake_studio, tmp_path):
    fallback = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch"],
    )
    assert fallback.exit_code == 0, fallback.output

    native = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert native.exit_code == 0, native.output
    assert _launch_command(native.output) == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(native.output)
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.17.11", False),
        ("1.17.12", True),
        ("opencode 1.18.2", True),
        ("development build", False),
    ],
)
def test_opencode_native_auto_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._opencode_supports_native_auto() is expected


def test_opencode_native_auto_assumes_current_without_local_binary(monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    assert start._opencode_supports_native_auto() is True


def test_yolo_opencode_old_version_uses_config_fallback(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: "1.17.11")
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", "run", "hello"]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


@pytest.mark.parametrize(
    ("args", "expected", "native"),
    [
        ([], ["--auto"], True),
        (["run", "hello"], ["run", "hello", "--auto"], True),
        (
            ["run", "hello", "--", "--literal"],
            ["run", "hello", "--auto", "--", "--literal"],
            True,
        ),
        (["--print-logs", "run", "hello"], ["--print-logs", "run", "hello", "--auto"], True),
        (["--session", "serve"], ["--session", "serve", "--auto"], True),
        (["serve"], ["serve"], False),
        (["--print-logs", "serve"], ["--print-logs", "serve"], False),
        (["run", "--auto", "hello"], ["run", "--auto", "hello"], True),
        (["generate"], ["generate"], False),
        (["console", "login"], ["console", "login"], False),
        (["--mini"], ["--mini"], False),
        (["--session", "sid", "--mini"], ["--session", "sid", "--mini"], False),
    ],
)
def test_opencode_native_auto_args(args, expected, native):
    assert start._opencode_native_auto_args(args, True) == (expected, native)
    assert start._opencode_native_auto_args(args, False) == (args, False)


def test_yolo_opencode_non_agent_subcommand_uses_config_fallback(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "serve"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "serve"]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


@pytest.mark.parametrize("passthrough", (["generate"], ["console", "login"], ["--mini"]))
def test_yolo_opencode_no_auto_command_uses_config_fallback(fake_studio, passthrough):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", *passthrough],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", *passthrough]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_no_yolo_opencode_has_no_permission_block(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert "permission" not in config


def test_no_yolo_opencode_flips_prior_yolo_allow_to_ask(fake_studio, tmp_path):
    yolo = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert json.loads(config_path.read_text())["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }
    plain = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    assert json.loads(config_path.read_text())["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


def test_yolo_openclaw_writes_exec_policy(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    state = tmp_path / "agents" / "openclaw"
    config = json.loads((state / "openclaw.json").read_text())
    assert config["tools"]["exec"] == {"host": "gateway", "security": "full", "ask": "off"}
    approvals = json.loads((state / "exec-approvals.json").read_text())
    assert approvals["defaults"] == {"security": "full", "ask": "off", "askFallback": "full"}


def test_no_yolo_openclaw_leaves_fresh_config_untouched(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    state = tmp_path / "agents" / "openclaw"
    config = json.loads((state / "openclaw.json").read_text())
    assert "exec" not in config.get("tools", {})
    assert not (state / "exec-approvals.json").exists()


def test_write_opencode_config_yolo_unit(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    config = json.loads(path.read_text())
    assert config["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_write_openclaw_config_yolo_unit(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"host": "gateway", "security": "full", "ask": "off"}
    approvals = json.loads((path.parent / "exec-approvals.json").read_text())
    assert approvals == {
        "version": 1,
        "defaults": {"security": "full", "ask": "off", "askFallback": "full"},
    }


def test_no_launch_rerun_clears_stale_opencode_yolo_permissions(fake_studio, tmp_path):
    yolo = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert "permission" in json.loads(config_path.read_text())
    plain = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    config = json.loads(config_path.read_text())
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }
    assert start._OPENCODE_PROVIDER in config["provider"]


def test_no_launch_rerun_clears_stale_openclaw_yolo_state(fake_studio, tmp_path):
    yolo = CliRunner().invoke(start.start_app, ["openclaw", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    state = tmp_path / "agents" / "openclaw"
    assert (state / "exec-approvals.json").exists()
    plain = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    config = json.loads((state / "openclaw.json").read_text())
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}
    assert not (state / "exec-approvals.json").exists()
    assert "unsloth" in config["models"]["providers"]


def test_write_openclaw_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}
    assert not (path.parent / "exec-approvals.json").exists()


def test_write_opencode_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


def test_openclaw_non_yolo_keeps_runtime_approvals(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    approvals = path.parent / "exec-approvals.json"
    state = json.loads(approvals.read_text())
    state["agents"] = {"main": {"allowlist": ["git status"]}}
    approvals.write_text(json.dumps(state))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    remaining = json.loads(approvals.read_text())
    assert "defaults" not in remaining
    assert remaining["agents"] == {"main": {"allowlist": ["git status"]}}


def test_openclaw_non_yolo_keeps_mixed_approval_defaults(tmp_path):
    path = tmp_path / "openclaw.json"
    approvals = path.parent / "exec-approvals.json"
    mixed = {
        "version": 1,
        "defaults": {"security": "allowlist", "ask": "on-miss", "askFallback": "full"},
    }
    approvals.write_text(json.dumps(mixed))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(approvals.read_text()) == mixed


def test_openclaw_non_yolo_leaves_partial_policy_untouched(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"timeout": 30, "ask": "off"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"timeout": 30, "ask": "off"}


def test_openclaw_non_yolo_leaves_no_permissive_values(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    exec_policy = json.loads(path.read_text())["tools"]["exec"]
    assert exec_policy.get("security") != "full"
    assert exec_policy.get("ask") != "off"
    assert not (path.parent / "exec-approvals.json").exists()


def test_openclaw_non_yolo_preserves_stricter_exec_policy(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"security": "deny", "ask": "on"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"security": "deny", "ask": "on"}


def test_openclaw_non_yolo_preserves_stricter_approval_defaults(tmp_path):
    path = tmp_path / "openclaw.json"
    approvals = path.parent / "exec-approvals.json"
    approvals.write_text(
        json.dumps({"version": 1, "defaults": {"security": "allowlist", "ask": "on"}})
    )
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    state = json.loads(approvals.read_text())
    assert state["defaults"] == {"security": "allowlist", "ask": "on"}


def test_openclaw_non_yolo_leaves_unparseable_approvals(tmp_path):
    path = tmp_path / "openclaw.json"
    approvals = path.parent / "exec-approvals.json"
    approvals.write_text("{not json")
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert approvals.read_text() == "{not json"


def test_opencode_non_yolo_flips_only_explicit_allow(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"edit": "allow", "bash": "deny", "read": "ask"}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["permission"] == {"edit": "ask", "bash": "deny", "read": "ask"}
    assert session == {}


def test_opencode_subagent_non_yolo_clears_yolo_task_permission(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(
        BASE,
        "sk-unsloth-abc",
        MODEL,
        path,
        yolo = True,
        as_subagent = True,
    )
    start.write_opencode_config(
        BASE,
        "sk-unsloth-abc",
        MODEL,
        path,
        as_subagent = True,
    )

    assert json.loads(path.read_text())["permission"]["task"] == "ask"


def test_opencode_non_yolo_leaves_string_permission(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": "deny"}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == "deny"
    assert session == {}


def test_opencode_non_yolo_leaves_catch_all_and_flips_explicit_allow(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"*": "allow", "bash": "allow"}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == {"*": "allow", "bash": "ask"}
    assert session == {}


def test_opencode_non_yolo_leaves_granular_object(tmp_path):
    path = tmp_path / "opencode.json"
    obj = {"read *": "deny", "git *": "ask"}
    path.write_text(json.dumps({"permission": {"bash": dict(obj)}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"]["bash"] == obj
    assert session == {}


def test_openclaw_non_yolo_leaves_mode_policy(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"mode": "deny"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"mode": "deny"}


def test_openclaw_non_yolo_preserves_sandbox_host(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"host": "sandbox"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"host": "sandbox"}


def test_openclaw_non_yolo_preserves_node_host(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"host": "node"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"host": "node"}


def test_openclaw_non_yolo_preserves_auto_host_permissive(tmp_path):
    for exec_policy in (
        {"host": "auto", "security": "full", "ask": "off"},
        {"security": "full", "ask": "off"},
    ):
        path = tmp_path / "openclaw.json"
        path.write_text(json.dumps({"tools": {"exec": dict(exec_policy)}}))
        start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
        config = json.loads(path.read_text())
        assert config["tools"]["exec"] == exec_policy


def test_openclaw_non_yolo_resets_only_gateway_yolo_fingerprint(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(
        json.dumps({"tools": {"exec": {"host": "gateway", "security": "full", "ask": "off"}}})
    )
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}


def test_openclaw_non_yolo_preserves_full_mode(tmp_path):
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"mode": "full"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"mode": "full"}


def test_yolo_command_flags_unmapped_agent_is_empty():
    assert start._yolo_command_flags("opencode", True) == []
    assert start._yolo_command_flags("openclaw", True) == []
    assert start._yolo_command_flags("claude", True) == ["--dangerously-skip-permissions"]
    assert start._yolo_command_flags("claude", False) == []


def test_yolo_config_fallbacks_add_no_legacy_command_flag(fake_studio):
    for agent in ("opencode", "openclaw"):
        result = CliRunner().invoke(start.start_app, [agent, "--yolo", "--no-launch"])
        assert result.exit_code == 0, result.output
        command = _launch_command(result.output)
        assert command and command[0] == agent, result.output
        assert not any("--yolo" in arg or "--dangerous" in arg for arg in command)


def test_pi_launch_clears_screen_first(fake_studio, monkeypatch):
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/pi")

    def run(command, env):
        calls.append("exec")
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi"])
    assert result.exit_code == 0, result.output
    assert calls == ["clear", "exec"]


def test_pi_no_launch_does_not_clear(fake_studio, monkeypatch):
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert calls == []


def test_claude_launch_does_not_clear(fake_studio, monkeypatch):
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))
    result = CliRunner().invoke(start.start_app, ["claude"])
    assert result.exit_code == 0, result.output
    assert calls == []


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario: a Windows pi shim under /mnt called from WSL "
    "(os.name is 'posix' under WSL), so this can't run on a native Windows runner.",
)
def test_connect_pi_wsl_windows_shim_relocates_userprofile(fake_studio, monkeypatch):
    captured = {}
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/pi")

    def run(command, env):
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi"])
    assert result.exit_code == 0, result.output
    home = captured["env"]["HOME"]
    assert captured["env"]["USERPROFILE"] == home
    wslenv = captured["env"]["WSLENV"].split(":")
    assert "HOME/p" in wslenv
    assert "USERPROFILE/p" in wslenv


def test_agent_api_key_auto_started_rejected_env_key_falls_back(fake_studio, tmp_path, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models") and token == "sk-unsloth-other-server":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    key = start._agent_api_key(BASE, "sk-unsloth-other-server", auto_started = True)
    assert key == "sk-unsloth-feedfacefeedface"
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert "sk-unsloth-other-server" not in json.dumps(cached["servers"].get(BASE, {}))


def test_agent_api_key_auto_started_accepted_key_is_honored(fake_studio, tmp_path):
    key = start._agent_api_key(BASE, "sk-unsloth-deadbeefdeadbeef", auto_started = True)
    assert key == "sk-unsloth-deadbeefdeadbeef"
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["saved"] == ["sk-unsloth-deadbeefdeadbeef"]


def test_session_config_no_launch_preserves_existing_state(fake_studio, tmp_path):
    with start._session_config("codex", launch = False) as home:
        marker = home / "sessions" / "live.sqlite"
        marker.parent.mkdir(parents = True)
        marker.write_text("state")
    with start._session_config("codex", launch = False) as home2:
        assert home2 == home
        assert (home2 / "sessions" / "live.sqlite").read_text() == "state"


def test_session_config_persist_uses_stable_dir_and_survives(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    with start._session_config("codex", launch = True, persist = True) as home:
        assert home == tmp_path / "agents" / "codex"
        (home / "marker").write_text("kept")
    assert home.exists()
    assert (home / "marker").read_text() == "kept"


def test_session_config_default_launch_is_ephemeral(monkeypatch, tmp_path):
    agents_root = tmp_path / "agents"
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)
    with start._session_config("codex", launch = True) as home:
        assert home.exists()
        parent = start._ephemeral_session_parent("codex")
        assert home.name.startswith(start._ephemeral_session_prefix("codex", parent))
        if parent is None:
            assert home.parent == agents_root / ".tmp"
    assert not home.exists()


def test_session_config_codex_uses_short_ephemeral_parent(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    monkeypatch.setattr(
        start,
        "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with start._session_config("codex", launch = True) as home:
        assert home.parent == short_parent
        assert home.name.startswith("u-codex-")
        assert home.exists()
    assert not home.exists()


def test_locked_file_windows_blocking_retries_until_acquired(monkeypatch, tmp_path):
    attempts = []
    sleeps = []

    def locking(_fd, mode, _length):
        if mode == 1:
            attempts.append(mode)
            if len(attempts) < 3:
                raise PermissionError(start.errno.EACCES, "busy")

    fake_msvcrt = SimpleNamespace(LK_NBLCK = 1, LK_UNLCK = 2, locking = locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.time, "sleep", sleeps.append)

    with start._locked_file(tmp_path / "lock") as acquired:
        assert acquired
    assert len(attempts) == 3
    assert sleeps == [0.05, 0.05]


def test_session_config_reclaims_old_short_homes_but_keeps_recent_and_live(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    stale = short_parent / "u-codex-abandoned"
    stale.mkdir()
    (stale / ".active.lock").write_bytes(b"\0")
    (stale / "plugin-checkout").write_text("left behind")
    old = time.time() - start._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(stale / ".active.lock", (old, old))
    recent = short_parent / "u-codex-surviving-child"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")
    monkeypatch.setattr(
        start,
        "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with start._session_config("codex", launch = True) as first:
        assert not stale.exists()
        assert recent.exists()
        with start._session_config("codex", launch = True) as second:
            assert first.exists()
            assert second.exists()
            assert first != second
        assert first.exists()
        assert not second.exists()
    assert not first.exists()


@pytest.mark.parametrize("agent", ["codex", "codex-subagent"])
def test_windows_codex_homes_use_the_short_parent(monkeypatch, tmp_path, agent):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.Path, "home", staticmethod(lambda: tmp_path))

    assert start._ephemeral_session_parent(agent) == tmp_path / ".unsloth" / ".tmp"
    parent = start._ephemeral_session_parent(agent)
    assert start._ephemeral_session_prefix(agent, parent) == "u-codex-"


def test_non_codex_agents_keep_the_studio_private_root(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.Path, "home", staticmethod(lambda: tmp_path))

    assert start._ephemeral_session_parent("claude") is None
    assert start._ephemeral_session_prefix("claude", None) == "unsloth-claude-"


def test_session_config_falls_back_when_existing_temp_root_is_unwritable(monkeypatch, tmp_path):
    # mkdir(exist_ok = True) succeeds on an existing unwritable root, so the lock fails first.
    agents = tmp_path / "agents"
    temp_root = agents / ".tmp"
    temp_root.mkdir(parents = True)
    os.chmod(temp_root, 0o500)
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents)

    try:
        with start._session_config("claude", launch = True) as home:
            assert home.exists()
            assert temp_root not in home.parents
    finally:
        os.chmod(temp_root, 0o700)
    assert not home.exists()


def test_augment_path_keeps_managed_node_without_a_home(monkeypatch, tmp_path):
    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    monkeypatch.setattr(
        start,
        "_managed_node_tools",
        lambda: (managed_bin / "node", managed_bin / "npm", True),
    )
    monkeypatch.setattr(
        start.Path,
        "home",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home directory"))),
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    start._augment_path_with_install_dirs()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(managed_bin)


def test_augment_path_leaves_path_alone_when_nothing_to_add(monkeypatch):
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)
    monkeypatch.setattr(
        start.Path,
        "home",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home directory"))),
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    start._augment_path_with_install_dirs()

    assert os.environ["PATH"] == "/usr/bin"


def test_probe_env_carries_install_dirs_and_restores_path(monkeypatch, tmp_path):
    managed_bin = tmp_path / "node" / "bin"
    managed_bin.mkdir(parents = True)
    monkeypatch.setattr(
        start,
        "_managed_node_tools",
        lambda: (managed_bin / "node", managed_bin / "npm", True),
    )
    before = os.environ.get("PATH")

    env = start._probe_env(OPENCODE_CONFIG = "/tmp/cfg.json")

    assert str(managed_bin) in env["PATH"]
    assert env["OPENCODE_CONFIG"] == "/tmp/cfg.json"
    assert os.environ.get("PATH") == before


def test_session_config_falls_back_when_studio_auth_root_is_unwritable(monkeypatch, tmp_path):
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode = 0o500)
    monkeypatch.setattr(start, "_agents_config_root", lambda: readonly / "agents")

    with start._session_config("claude", launch = True) as home:
        assert home.exists()
        assert readonly not in home.parents
    assert not home.exists()


def test_session_config_reclaims_abandoned_homes_for_non_codex_agents(monkeypatch, tmp_path):
    agents_root = tmp_path / "agents"
    temp_root = agents_root / ".tmp"
    temp_root.mkdir(parents = True)
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)
    abandoned = temp_root / "unsloth-claude-abandoned"
    abandoned.mkdir()
    (abandoned / ".active.lock").write_bytes(b"\0")
    (abandoned / "state.json").write_text("left behind")
    old = time.time() - start._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(abandoned / ".active.lock", (old, old))
    recent = temp_root / "unsloth-claude-still-running"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")

    with start._session_config("claude", launch = True) as home:
        assert not abandoned.exists()
        assert recent.exists()
        assert home.parent == temp_root
    assert not home.exists()


def test_session_config_serializes_normal_short_home_deletion(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    monkeypatch.setattr(start, "_ephemeral_session_parent", lambda _agent: short_parent)
    original_rmtree = start.shutil.rmtree

    def checked_rmtree(path, *args, **kwargs):
        if path.parent == short_parent and path.name.startswith("u-codex-"):
            with start._locked_file(short_parent / ".cleanup.lock", blocking = False) as unlocked:
                assert not unlocked
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(start.shutil, "rmtree", checked_rmtree)
    with start._session_config("codex", launch = True) as home:
        assert home.exists()
    assert not home.exists()


_RESUME_ENV_VAR = {
    "codex": "CODEX_HOME",
    "openclaw": "OPENCLAW_STATE_DIR",
    "hermes": "HERMES_HOME",
    "pi": "HOME",
    "dsh": "DSH_HOME",
}


def _capture_launch(monkeypatch, argv):
    captured = {}
    monkeypatch.setattr(start, "_managed_node_tools", lambda: None)

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, argv)
    assert result.exit_code == 0, result.output
    return captured


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_resume_persists_agent_home_to_stable_dir(agent, fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent, "--persist"])
    stable = tmp_path / "agents" / agent
    assert captured["env"][_RESUME_ENV_VAR[agent]] == str(stable)
    assert stable.exists()


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_default_launch_home_is_ephemeral(agent, fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent])
    home = captured["env"][_RESUME_ENV_VAR[agent]]
    parent = start._ephemeral_session_parent(agent)
    assert start._ephemeral_session_prefix(agent, parent) in home
    if parent is None:
        assert Path(home).parent == tmp_path / "agents" / ".tmp"


def test_resume_opencode_config_in_stable_dir(fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    stable = tmp_path / "agents" / "opencode"
    assert captured["env"]["OPENCODE_CONFIG"] == str(stable / "opencode.json")
    assert stable.exists()


def test_persist_bare_codex_launch_has_no_resume_token(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist"])
    assert "resume" not in captured["command"]
    assert captured["command"][1:] == ["--oss", "--profile", start._CODEX_PROFILE]


def test_persist_bare_opencode_launch_has_no_resume_token(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    assert "--continue" not in captured["command"]
    assert captured["command"][1:] == ["--model", f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"]


def test_persist_bare_claude_launch_has_no_resume_token(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    captured = _capture_launch(monkeypatch, ["claude", "--persist"])
    assert "--continue" not in captured["command"]
    assert captured["command"][1:] == ["--model", MODEL["id"]]


def test_resume_with_passthrough_does_not_auto_append(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist", "exec", "hello"])
    assert "resume" not in captured["command"]
    assert captured["command"][-2:] == ["exec", "hello"]


def test_default_launch_has_no_resume_token(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex"])
    assert "resume" not in captured["command"]


def test_resume_persist_only_agents_have_no_resume_token(fake_studio, monkeypatch):
    for agent in ("openclaw", "hermes", "dsh"):
        monkeypatch.setattr(start.shutil, "which", lambda _, a = agent: f"/usr/local/bin/{a}")
        if agent == "dsh":
            monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
        captured = _capture_launch(monkeypatch, [agent, "--persist"])
        assert "resume" not in captured["command"]
        assert "--continue" not in captured["command"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--resume", "session-id", "-z", "follow up"],
            [
                "chat",
                "-Q",
                "--yolo",
                "--accept-hooks",
                "--resume",
                "session-id",
                "-q",
                "follow up",
            ],
        ),
        (
            ["-rsession-id", "-zfollow up"],
            ["chat", "-Q", "--yolo", "--accept-hooks", "-rsession-id", "-qfollow up"],
        ),
        (
            ["-c=project", "-z=follow up"],
            ["chat", "-Q", "--yolo", "--accept-hooks", "-c=project", "-q=follow up"],
        ),
        (
            ["-r", "session-id", "--oneshot=follow up"],
            [
                "chat",
                "-Q",
                "--yolo",
                "--accept-hooks",
                "-r",
                "session-id",
                "--query=follow up",
            ],
        ),
        (
            ["--continue", "project", "--oneshot", "follow up"],
            [
                "chat",
                "-Q",
                "--yolo",
                "--accept-hooks",
                "--continue",
                "project",
                "-q",
                "follow up",
            ],
        ),
        (
            ["--yolo", "--resume", "session-id", "-z", "follow up"],
            [
                "chat",
                "-Q",
                "--accept-hooks",
                "--yolo",
                "--resume",
                "session-id",
                "-q",
                "follow up",
            ],
        ),
        (
            ["--accept-hooks", "--resume", "session-id", "-z", "follow up"],
            [
                "chat",
                "-Q",
                "--yolo",
                "--accept-hooks",
                "--resume",
                "session-id",
                "-q",
                "follow up",
            ],
        ),
        (
            ["--resume", "chat", "-z", "follow up"],
            [
                "chat",
                "-Q",
                "--yolo",
                "--accept-hooks",
                "--resume",
                "chat",
                "-q",
                "follow up",
            ],
        ),
        (["--resume", "session-id"], ["--resume", "session-id"]),
        (["-z", "new session"], ["-z", "new session"]),
    ],
)
def test_hermes_resume_oneshot_args(args, expected):
    assert start._hermes_resume_oneshot_args(args) == expected


def test_hermes_resume_oneshot_uses_session_aware_chat(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/hermes")
    captured = _capture_launch(
        monkeypatch,
        ["hermes", "--persist", "--resume", "session-id", "-z", "follow up"],
    )
    assert captured["command"][1:] == [
        "chat",
        "-Q",
        "--yolo",
        "--accept-hooks",
        "--resume",
        "session-id",
        "-q",
        "follow up",
    ]


@pytest.mark.parametrize("usage_arg", ["--usage-file", "--usage-file=usage.json"])
def test_hermes_resume_oneshot_rejects_usage_file(monkeypatch, usage_arg):
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail("argument validation must run before connect"),
    )
    argv = ["hermes", "--resume", "session-id", "-z", "follow up", usage_arg]
    if usage_arg == "--usage-file":
        argv.append("usage.json")
    result = CliRunner().invoke(start.start_app, argv)
    assert result.exit_code == 2
    assert "cannot resume a one-shot session with --usage-file" in result.output


def test_native_resume_flag_passes_through_unchanged(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    captured = _capture_launch(monkeypatch, ["claude", "--resume", "some-session-guid"])
    resume = captured["command"].index("--resume")
    assert captured["command"][resume : resume + 2] == ["--resume", "some-session-guid"]
    assert captured["command"].index("--model") < resume
    assert captured["command"].count("--resume") == 1
    assert "--continue" not in captured["command"]


def _fake_hub_listing(monkeypatch, files_by_repo):
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    calls = []

    def fake(repo):
        calls.append(repo)
        return files_by_repo.get(repo)

    monkeypatch.setattr(start, "_hub_gguf_files", fake)
    return calls


def test_codex_preflight_rejects_non_gguf_repo(monkeypatch, capsys):
    _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    with pytest.raises(typer.Exit) as excinfo:
        start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "mlx-community/Qwen3-0.6B-4bit")
    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "Codex needs a GGUF model" in err
    assert "Try:" not in err


def test_codex_preflight_passes_gguf_repo_and_splits_variant(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {"unsloth/Qwen3-0.6B-GGUF": ["Qwen3-0.6B-Q4_K_M.gguf"]})
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    assert calls == ["unsloth/Qwen3-0.6B-GGUF"]


def test_codex_preflight_defers_when_listing_unavailable(monkeypatch):
    _fake_hub_listing(monkeypatch, {})
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "owner/private-model")


def test_codex_preflight_skips_paths_and_empty_model(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {})
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "./models/foo.gguf")
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, None)
    assert calls == []


def test_codex_preflight_skips_remote_studio(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {"models/qwen-finetune": []})
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://studio.example:8888")
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "models/qwen-finetune")
    assert calls == []


def test_codex_gguf_failure_suggests_only_a_verified_sibling(monkeypatch, capsys):
    _fake_hub_listing(monkeypatch, {"owner/model-GGUF": ["model-Q4_K_M.gguf"]})
    with pytest.raises(typer.Exit):
        start._fail_agent_needs_gguf(start._CODEX_GGUF_AGENT, "owner/model")
    assert "Try: unsloth start codex --model owner/model-GGUF" in capsys.readouterr().err


def test_hub_gguf_files_parses_listing(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    payload = {"siblings": [{"rfilename": "README.md"}, {"rfilename": "model-Q4_K_M.GGUF"}]}
    monkeypatch.setattr(
        start.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    assert start._hub_gguf_files("owner/model") == ["model-Q4_K_M.GGUF"]


def test_hub_gguf_files_unknown_on_error_or_empty_listing(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    def unauthorized(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", None, None)

    monkeypatch.setattr(start.urllib.request, "urlopen", unauthorized)
    assert start._hub_gguf_files("owner/missing") is None
    monkeypatch.setattr(
        start.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b'{"siblings": []}')
    )
    assert start._hub_gguf_files("owner/empty") is None


def test_codex_rejects_non_gguf_model_before_connect(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: [])
    monkeypatch.setattr(
        start, "_connect", lambda *a, **k: pytest.fail("preflight must run before connect")
    )
    result = CliRunner().invoke(
        start.start_app, ["codex", "--model", "mlx-community/Qwen3-0.6B-4bit"]
    )
    assert result.exit_code == 1
    assert "Codex needs a GGUF model" in result.output


def test_codex_preflight_normalizes_ownerless_shorthand(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {"unsloth/Qwen3-0.6B": []})
    with pytest.raises(typer.Exit):
        start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "Qwen3-0.6B")
    assert calls[0] == "unsloth/Qwen3-0.6B"


def test_codex_preflight_shorthand_skips_existing_local_dir(monkeypatch, tmp_path):
    calls = _fake_hub_listing(monkeypatch, {})
    (tmp_path / "Qwen3-0.6B").mkdir()
    monkeypatch.chdir(tmp_path)
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "Qwen3-0.6B")
    assert calls == []


def test_hub_gguf_files_ignores_auxiliary_ggufs(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    payload = {
        "siblings": [
            {"rfilename": "mmproj-F16.gguf"},
            {"rfilename": "mtp-gemma.gguf"},
            {"rfilename": "MTP/gemma-Q8_0-MTP.gguf"},
            {"rfilename": "README.md"},
        ]
    }
    monkeypatch.setattr(
        start.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    assert start._hub_gguf_files("owner/mmproj-pack") == []


def test_hub_gguf_files_ignores_prefixed_drafters(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    payload = {
        "siblings": [
            {"rfilename": "DSpark-drafter-Q2K-Q8.gguf"},
            {"rfilename": "dflash-drafter-Q8_0.gguf"},
            {"rfilename": "eagle3-gpt-oss-20b-Q8_0.gguf"},
            {"rfilename": "dspark/DeepSeek-V4-Flash-Q8_0.gguf"},
            {"rfilename": "Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"},
            {"rfilename": "DFlash/Qwen3.6-27B-DFlash-Q4_K_M.gguf"},
            {"rfilename": "Llama-3.1-8B-Eagle3-Q4_K_M.gguf"},
        ]
    }
    monkeypatch.setattr(
        start.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    assert start._hub_gguf_files("owner/dspark-pack") == [
        "Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
        "DFlash/Qwen3.6-27B-DFlash-Q4_K_M.gguf",
        "Llama-3.1-8B-Eagle3-Q4_K_M.gguf",
    ]


def test_hub_gguf_files_filters_root_big_endian_only(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    payload = {
        "siblings": [
            {"rfilename": "model-Q4_K_M-be.gguf"},
            {"rfilename": "model-Q4_K_M_be.gguf"},
            {"rfilename": "quants/model-be.gguf"},
            {"rfilename": "model-belle.gguf"},
        ]
    }
    monkeypatch.setattr(
        start.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    assert start._hub_gguf_files("owner/be-pack") == ["quants/model-be.gguf", "model-belle.gguf"]


def test_codex_preflight_defers_to_running_server(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://127.0.0.1:8888")
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "mlx-community/Qwen3-0.6B-4bit")
    assert calls == []


def _fake_variants(monkeypatch, responses):
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        if isinstance(responses, Exception):
            raise responses
        return responses

    monkeypatch.setattr(start, "_http_json", http_json)
    return urls


def test_codex_attach_check_rejects_on_empty_variants(monkeypatch, capsys):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    _fake_variants(monkeypatch, {"variants": [], "has_vision": False})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "mlx-community/Qwen3-0.6B-4bit"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err


def test_codex_attach_check_passes_on_variants(monkeypatch):
    urls = _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M"}]})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
    )
    assert "repo_id=unsloth%2FQwen3-0.6B-GGUF" in urls[0]


def test_codex_attach_check_rejects_unavailable_variant(monkeypatch, capsys):
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M"}, {"quant": "Q8_0"}]})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF:Q4_KM"
        )
    err = capsys.readouterr().err
    assert "no GGUF variant Q4_KM" in err
    assert "Q4_K_M, Q8_0" in err


@pytest.mark.parametrize(
    "requested,rows",
    [
        ("q4_k_m", [{"quant": "Q4_K_M"}]),
        ("Q4_K_XL", [{"quant": "UD-Q4_K_XL", "filename": "Qwen3-0.6B-UD-Q4_K_XL.gguf"}]),
        ("Q4_K_M", [{"size_bytes": 1}]),
    ],
)
def test_codex_attach_check_passes_resolvable_variants(monkeypatch, requested, rows):
    _fake_variants(monkeypatch, {"variants": rows})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", f"unsloth/Qwen3-0.6B-GGUF:{requested}"
    )


def test_codex_attach_check_takes_the_variant_from_the_caller(monkeypatch):
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q8_0"}]})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF", "Q4_K_M"
        )


def test_codex_attach_check_defers_on_server_error(monkeypatch):
    _fake_variants(
        monkeypatch,
        urllib.error.HTTPError(f"{BASE}/api/models/gguf-variants", 404, "nope", None, None),
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "mlx-community/Qwen3-0.6B-4bit"
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
    )


def test_codex_attach_check_skips_without_model(monkeypatch):
    urls = _fake_variants(monkeypatch, {"variants": []})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", None)
    assert urls == []


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_hub_gguf_files_skips_hub_when_offline(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    monkeypatch.setattr(
        start.urllib.request,
        "urlopen",
        lambda request, timeout: pytest.fail("offline mode must not call the hub"),
    )
    assert start._hub_gguf_files("owner/model") is None


def test_codex_preflight_defers_bare_names_to_attached_server(monkeypatch):
    calls = _fake_hub_listing(monkeypatch, {"unsloth/Qwen3-0.6B": []})
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://127.0.0.1:8888")
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: True)
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "Qwen3-0.6B")
    assert calls == []


def test_codex_gguf_failure_skips_hint_probe_for_non_hub_ids(monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_hub_gguf_files",
        lambda repo: pytest.fail("must not probe the hub for a non-hub id"),
    )
    with pytest.raises(typer.Exit):
        start._fail_agent_needs_gguf(start._CODEX_GGUF_AGENT, "models/Llama/customer-model")
    assert "Try:" not in capsys.readouterr().err


def test_codex_attach_rejects_before_load(fake_studio, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if "/api/models/gguf-variants" in url:
            return {"variants": []}
        if url.endswith("/api/inference/load"):
            pytest.fail("rejected model must not be loaded")
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["codex", "--model", "mlx-community/Qwen3-0.6B-4bit", "--no-launch"]
    )
    assert result.exit_code == 1
    assert "Codex needs a GGUF model" in result.output


def test_codex_attach_rejects_unavailable_variant_before_load(fake_studio, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if "/api/models/gguf-variants" in url:
            return {"variants": [{"quant": "Q4_K_M"}]}
        if url.endswith("/api/inference/load"):
            pytest.fail("a quant the repo does not have must not evict the resident model")
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--model", "unsloth/Qwen3-0.6B-GGUF:Q4_KM", "--no-launch"],
    )
    assert result.exit_code == 1
    assert "no GGUF variant Q4_KM" in result.output


def test_codex_attach_reuses_resident_model_without_preload_probe(fake_studio, monkeypatch):
    inner = start._http_json
    probes = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if "/api/models/gguf-variants" in url:
            probes.append(url)
            return {"variants": []}
        if url.endswith("/api/inference/load"):
            pytest.fail("the resident model already matches; no load is needed")
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["codex", "--model", MODEL["id"], "--no-launch"])
    assert result.exit_code == 0, result.output
    assert probes == []


@pytest.mark.parametrize("endpoint", ["", "huggingface.co", "not a url"])
def test_hub_gguf_files_unknown_on_malformed_endpoint(monkeypatch, endpoint):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    assert start._hub_gguf_files("owner/model") is None


def test_codex_attach_check_skips_direct_gguf_files(monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a direct .gguf file needs no variants probe"),
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/foo-Q4_K_M.gguf")
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./local/model.GGUF")
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/Q4_K_M/model-be.gguf"
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/mmproj-dumps/foo-Q4_K_M.gguf"
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/dflash/Qwen-DFlash-Q4_K_M.gguf"
    )


def test_codex_attach_check_direct_variant_always_asks_the_server(monkeypatch):
    probes = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        probes.append(url)
        return {
            "variants": [{"quant": "Q4_K_M"}],
            "resolved_locally": True,
            "loadable_variants": ["Q4_K_M", "q4_k_m"],
            "loadable": True,
        }

    monkeypatch.setattr(start, "_http_json", http_json)
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/foo-Q4_K_M.gguf", "q4_k_m"
    )
    assert probes, "an explicit variant must reach the server"

    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a variantless direct file needs no probe"),
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/foo-Q4_K_M.gguf")


def test_codex_attach_check_asks_server_for_foreign_direct_variant(monkeypatch, capsys):
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M"}, {"quant": "Q8_0"}]})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/m/model-Q4_K_M.gguf", "Q8_0"
    )
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M"}]})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/foo-Q4_K_M.gguf", "Q8_0"
        )
    assert "no GGUF variant Q8_0" in capsys.readouterr().err
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/Q8_0/foo-Q4_K_M.gguf", "Q8_0"
        )


def test_codex_attach_check_fails_live_empty_explicit_paths(tmp_path, monkeypatch, capsys):
    _fake_variants(monkeypatch, {"variants": []})
    target = tmp_path / "hf-dir"
    target.mkdir()
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(target))
    assert "Codex needs a GGUF model" in capsys.readouterr().err


def test_codex_attach_check_still_defers_existing_raw_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my-model-dir").mkdir()
    _fake_variants(monkeypatch, {"variants": []})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "my-model-dir")


def test_codex_attach_check_strictness_follows_the_server_answer(monkeypatch):
    rows = [{"quant": "Q4_K_M", "filename": "model-Q4_K_M.gguf"}]
    _fake_variants(monkeypatch, {"variants": rows, "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "models/qwen", "Q4")
    _fake_variants(monkeypatch, {"variants": rows})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "models/qwen", "Q4")


def test_codex_attach_check_strict_accepts_the_full_stem(monkeypatch):
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "Q4_K_M", "filename": "model-Q4_K_M.gguf"}],
            "resolved_locally": True,
        },
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "model-Q4_K_M")


def test_codex_attach_check_strict_rejects_nested_basename_labels(monkeypatch):
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "BF16", "filename": "BF16/model.gguf"}],
            "resolved_locally": True,
        },
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "model")


def test_codex_attach_check_rejects_torn_named_file_despite_sibling_variant(tmp_path, monkeypatch):
    shard = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (tmp_path / "model-Q8_0.gguf").write_bytes(b"GGUF")
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q8_0", "partial": False}], "resolved_locally": True},
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(shard), "Q8_0")


def test_codex_attach_check_rejects_a_requested_torn_local_variant(monkeypatch):
    rows = {
        "variants": [
            {"quant": "Q8_0", "partial": False},
            {"quant": "Q4_K_M", "partial": True},
        ],
        "resolved_locally": True,
    }
    _fake_variants(monkeypatch, rows)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "Q4_K_M")
    _fake_variants(monkeypatch, rows)
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "Q8_0")


def test_codex_attach_check_probes_direct_paths_on_remote_servers(monkeypatch, capsys):
    remote = "http://studio.example:8888"
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, remote, "sk-test", "/models/gone-Q4_K_M.gguf"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "partial": False}], "resolved_locally": True},
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, remote, "sk-test", "/models/foo-Q4_K_M.gguf")
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: pytest.fail("loopback needs no probe"))
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/foo-Q4_K_M.gguf")


@pytest.mark.skipif(os.name == "nt", reason = "every spelling is native on Windows")
def test_codex_attach_check_probes_non_native_direct_paths(monkeypatch, capsys):
    urls = _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", r"C:\models\typo-Q4_K_M.gguf"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    assert urls, "the server was never asked"
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "partial": False}], "resolved_locally": True},
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", r"C:\models\foo-Q4_K_M.gguf")


def test_codex_attach_check_honors_a_negative_verdict_without_an_allow_list(monkeypatch, capsys):
    negative = {
        "variants": [{"quant": "Q8_0", "partial": False}],
        "resolved_locally": True,
        "loadable": False,
    }
    _fake_variants(monkeypatch, negative)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/MTP/sub/foo-Q8_0.gguf", "Q8_0"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(monkeypatch, {**negative, "loadable": True})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/sub/foo-Q8_0.gguf", "Q8_0"
    )
    _fake_variants(monkeypatch, {k: v for k, v in negative.items() if k != "loadable"})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/sub/foo-Q8_0.gguf", "Q8_0"
    )


def test_codex_attach_check_refuses_companion_paths_with_a_variant(monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a refused direct file needs no variants probe"),
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/m/mmproj-F16.gguf", "Q4_K_M"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/m/mmproj-F16.gguf"
        )


def test_codex_attach_check_ignores_cleanable_only_answers(monkeypatch, capsys):
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "partial": True, "cleanable": True}]},
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "owner/no-gguf")
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(
        monkeypatch,
        {
            "variants": [
                {"quant": "Q8_0", "partial": True, "cleanable": True},
                {"quant": "Q4_K_M"},
            ]
        },
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "owner/has-gguf")


def test_codex_attach_check_follows_bare_names_the_server_calls_remote(monkeypatch):
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        if "unsloth" in url:
            return {"variants": [{"quant": "Q4_K_M"}], "resolved_locally": False}
        return {"variants": [], "resolved_locally": False}

    monkeypatch.setattr(start, "_http_json", http_json)
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "Qwen3-0.6B-GGUF")
    assert len(urls) == 2 and "unsloth%2FQwen3-0.6B-GGUF" in urls[1]


def test_codex_attach_check_trusts_the_servers_loadable_answer(monkeypatch, capsys):
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "BF16", "filename": "BF16/model.gguf"}],
            "resolved_locally": True,
            "loadable_variants": ["BF16"],
            "loadable": False,
        },
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "BF16")
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "BF16", "filename": "BF16/model.gguf"}],
            "resolved_locally": True,
            "loadable_variants": ["BF16"],
            "loadable": False,
        },
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m")
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "Q8_0", "filename": "m-Q8_0-00001-of-00002.gguf"}],
            "resolved_locally": True,
            "loadable_variants": [],
            "loadable": False,
        },
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "Q8_0")


def test_codex_attach_check_probes_missing_bare_gguf_shorthands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        return {"variants": []}

    monkeypatch.setattr(start, "_http_json", http_json)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "foo.gguf")
    assert any("repo_id=unsloth%2Ffoo.gguf" in url for url in urls)
    (tmp_path / "real-Q4_K_M.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(
        start, "_http_json", lambda *a, **k: pytest.fail("an existing file needs no probe")
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "real-Q4_K_M.gguf")


def test_codex_attach_check_treats_gguf_suffixed_hub_ids_as_remote(monkeypatch):
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "filename": "BF16/model-Q4_K_M.gguf"}]},
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "owner/model.gguf")
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "UD-Q4_K_XL", "filename": "m-UD-Q4_K_XL.gguf"}]},
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "owner/model.gguf", "Q4_K_XL"
    )


def test_codex_attach_check_accepts_bpw_qualified_local_requests(monkeypatch):
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "IQ4_XS", "filename": "model-IQ4_XS-3.53bpw.gguf"}],
            "resolved_locally": True,
        },
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "IQ4_XS-3.53bpw")


def test_codex_attach_check_strict_accepts_basename_quant_tokens(monkeypatch):
    _fake_variants(
        monkeypatch,
        {
            "variants": [{"quant": "F16", "filename": "F16-checkpoint-Q4_K_M.gguf"}],
            "resolved_locally": True,
        },
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/F16-checkpoint-Q4_K_M.gguf", "Q4_K_M"
    )


def test_codex_attach_check_honors_resolved_locally_empty_for_raw_names(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "qwen").mkdir(parents = True)
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "models/qwen")
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(monkeypatch, {"variants": []})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "models/qwen")


def test_codex_attach_check_requires_a_pickable_row_without_a_variant(monkeypatch, capsys):
    rows = {
        "variants": [{"quant": "BF16", "filename": "BF16/model.gguf"}],
        "resolved_locally": True,
    }
    _fake_variants(monkeypatch, rows)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m")
    err = capsys.readouterr().err
    assert "quant subdirectories" in err and "BF16" in err
    _fake_variants(monkeypatch, rows)
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "BF16")
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "filename": "m-Q4_K_M.gguf"}], "resolved_locally": True},
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m")


def test_codex_attach_check_probes_gguf_named_directories(tmp_path, monkeypatch, capsys):
    gguf_dir = tmp_path / "foo.gguf"
    gguf_dir.mkdir()
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(gguf_dir))
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q4_K_M", "filename": "m-Q4_K_M.gguf"}], "resolved_locally": True},
    )
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(gguf_dir))


def test_codex_attach_check_allows_short_shard_like_names(tmp_path, monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a direct .gguf file needs no variants probe"),
    )
    lone = tmp_path / "model-Q4_K_M-001-of-002.gguf"
    lone.write_bytes(b"GGUF")
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(lone))


def test_codex_attach_check_honors_loadable_on_an_empty_listing(monkeypatch):
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True, "loadable": True})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/MTP/foo-Q8_0.gguf")
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True, "loadable": False})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/MTP/foo-Q8_0.gguf"
        )


def test_codex_preload_gate_checks_direct_path_identity(fake_studio, monkeypatch, tmp_path):
    inner = start._http_json
    probed = []
    resident = tmp_path / "old" / "foo-Q4_K_M.gguf"
    resident.parent.mkdir()
    resident.write_bytes(b"GGUF")
    other = tmp_path / "new" / "foo-Q4_K_M.gguf"
    other.parent.mkdir()
    other.write_bytes(b"GGUF")

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/loaded-models"):
            return {"data": [{"id": "foo-Q4_K_M", "loaded": True}]}
        if url.endswith("/api/inference/status"):
            return {
                "is_gguf": True,
                "gguf_variant": "Q4_K_M",
                "model_identifier": os.fspath(resident),
            }
        if "/api/models/gguf-variants" in url:
            probed.append(url)
            return {"variants": [{"quant": "Q4_K_M"}], "resolved_locally": True, "loadable": True}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    CliRunner().invoke(
        start.start_app,
        ["codex", "--model", os.fspath(other), "--gguf-variant", "Q4_K_M", "--no-launch"],
    )
    assert probed, "a different path with the same basename is not the resident model"


def test_codex_preload_gate_runs_for_a_settings_reload(fake_studio, monkeypatch):
    inner = start._http_json
    probed = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if "/api/models/gguf-variants" in url:
            probed.append(url)
            return {"variants": [{"quant": "Q4_K_M"}], "resolved_locally": False}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    CliRunner().invoke(
        start.start_app,
        ["codex", "--model", MODEL["id"], "--max-seq-length", "8192", "--no-launch"],
    )
    assert probed, "a context-length change reloads, so the gate must check it"


def test_codex_preload_gate_runs_for_a_mistyped_resident_variant(fake_studio, monkeypatch):
    inner = start._http_json
    probed = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "gguf_variant": "Q4_K_M"}
        if "/api/models/gguf-variants" in url:
            probed.append(url)
            return {"variants": [{"quant": "Q4_K_M"}], "resolved_locally": False}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    CliRunner().invoke(
        start.start_app,
        ["codex", "--model", MODEL["id"], "--gguf-variant", "Q4KM", "--no-launch"],
    )
    assert probed, "a separator-mangled quant is not the resident one"


def test_codex_preload_gate_defers_to_the_resident_model(fake_studio, monkeypatch):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "gguf_variant": "Q4_K_M"}
        if "/api/models/gguf-variants" in url:
            pytest.fail("the resident model already serves this request")
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--model", MODEL["id"], "--gguf-variant", "Q4_K_M", "--no-launch"],
    )
    assert result.exit_code == 0, result.output


def test_codex_preload_gate_still_runs_for_a_different_variant(fake_studio, monkeypatch):
    inner = start._http_json
    probed = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "gguf_variant": "Q4_K_M"}
        if "/api/models/gguf-variants" in url:
            probed.append(url)
            return {"variants": [{"quant": "Q4_K_M"}], "resolved_locally": False}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    CliRunner().invoke(
        start.start_app,
        ["codex", "--model", MODEL["id"], "--gguf-variant", "Q8_0", "--no-launch"],
    )
    assert probed, "a different quant reloads, so the gate must check it"


def test_codex_attach_check_asks_about_nested_drafter_folders(monkeypatch, capsys):
    _fake_variants(monkeypatch, {"variants": [], "resolved_locally": True})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/MTP/copies/foo-Q8_0.gguf"
        )
    assert "Codex needs a GGUF model" in capsys.readouterr().err
    _fake_variants(
        monkeypatch,
        {"variants": [{"quant": "Q8_0"}], "resolved_locally": True, "loadable": True},
    )
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/MTP/copies/foo-Q8_0.gguf"
    )
    monkeypatch.setattr(
        start, "_http_json", lambda *a, **k: pytest.fail("an immediate companion needs no probe")
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", "/models/mtp-foo-Q8_0.gguf"
        )


def test_codex_attach_check_defers_foreign_path_syntax(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: {"variants": [{"quant": "Q4_K_M"}]})
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "C:\\models\\foo-Q4_K_M.gguf"
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(tmp_path / "gone-Q4_K_M.gguf")
        )


def test_codex_attach_check_defers_when_loopback_is_not_this_machine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "verify_studio_identity", lambda base, **_kw: False)
    probes = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        probes.append(url)
        return {"variants": [{"quant": "Q4_K_M"}]}

    monkeypatch.setattr(start, "_http_json", http_json)
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(tmp_path / "gone-Q4_K_M.gguf")
    )
    torn = tmp_path / "torn-Q4_K_M-00001-of-00002.gguf"
    torn.write_bytes(b"GGUF")
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(torn))
    assert probes, "the server must be asked when its filesystem is not ours"


def test_codex_attach_check_treats_both_spellings_as_native_on_windows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _simulate_windows(monkeypatch)
    assert start._path_syntax_is_native("C:\\models\\foo-Q4_K_M.gguf") is True
    assert start._path_syntax_is_native("/models/foo-Q4_K_M.gguf") is True
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a visible missing file needs no probe"),
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(tmp_path / "gone-Q4_K_M.gguf")
        )


def test_codex_attach_check_rejects_missing_direct_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a visible missing file needs no probe"),
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(
            start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(tmp_path / "typo-Q4_K_M.gguf")
        )
    assert "does not exist" in capsys.readouterr().err
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/nonexistent-root/dir/m-Q4_K_M.gguf"
    )


def test_codex_attach_check_rejects_broken_direct_symlinks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a direct .gguf file needs no variants probe"),
    )
    link = tmp_path / "gone-Q4_K_M.gguf"
    link.symlink_to(tmp_path / "missing-target.gguf")
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(link))
    assert "incomplete" in capsys.readouterr().err


def test_codex_attach_check_fails_all_partial_local_answers(monkeypatch, capsys):
    rows = {"variants": [{"quant": "Q4_K_M", "partial": True}], "resolved_locally": True}
    _fake_variants(monkeypatch, rows)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m")
    assert "incomplete GGUF weights" in capsys.readouterr().err
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M", "partial": True}]})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF")
    _fake_variants(monkeypatch, {"variants": [{"quant": "Q4_K_M", "partial": True}]})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m")


def test_codex_attach_check_local_answers_take_exact_labels_only(monkeypatch):
    rows = {"variants": [{"quant": "Q4_K_M", "filename": "model-Q4_K_M.gguf"}]}
    _fake_variants(monkeypatch, rows)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "./m", "Q4")
    _fake_variants(monkeypatch, rows)
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "unsloth/Qwen3-0.6B-GGUF", "Q4"
    )


def test_codex_attach_check_rejects_incomplete_direct_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a direct .gguf file needs no variants probe"),
    )
    empty = tmp_path / "zero-Q4_K_M.gguf"
    empty.write_bytes(b"")
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(empty))
    assert "incomplete" in capsys.readouterr().err

    shard = tmp_path / "m-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(shard))

    (tmp_path / "m-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(shard))
    start._attach_gguf_check(
        start._CODEX_GGUF_AGENT, BASE, "sk-test", "/nonexistent/other-Q4_K_M.gguf"
    )


def test_codex_attach_check_accepts_symlinked_split_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a direct .gguf file needs no variants probe"),
    )
    real = tmp_path / "real"
    real.mkdir()
    (real / "m-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")
    (real / "m-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    links = tmp_path / "links"
    links.mkdir()
    link = links / "m-Q4_K_M-00001-of-00002.gguf"
    link.symlink_to(real / "m-Q4_K_M-00001-of-00002.gguf")
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(link))

    (real / "m-Q4_K_M-00002-of-00002.gguf").unlink()
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", os.fspath(link))


@pytest.mark.parametrize(
    "path",
    [
        "/models/mmproj-F16.gguf",
        "./local/MMPROJ-F32.GGUF",
        "/models/mtp-Qwen3-Q8_0.gguf",
        "/models/dspark/dspark-DeepSeek-Q8_0.gguf",
        "C:\\models\\mmproj-F16.gguf",
    ],
)
def test_codex_attach_check_refuses_companion_gguf_files(monkeypatch, capsys, path):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: pytest.fail("a companion .gguf needs no variants probe"),
    )
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", path)
    assert "Codex needs a GGUF model" in capsys.readouterr().err


def test_codex_preflight_canonicalizes_missing_bare_gguf_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = _fake_hub_listing(monkeypatch, {"unsloth/foo.gguf": []})
    with pytest.raises(typer.Exit):
        start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "foo.gguf")
    assert calls == ["unsloth/foo.gguf"]


@pytest.mark.parametrize("kwargs", [{"serve": False}, {"launch": False}])
def test_codex_preflight_skips_when_autostart_impossible(monkeypatch, kwargs):
    calls = _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    start._preflight_agent_gguf(start._CODEX_GGUF_AGENT, "mlx-community/Qwen3-0.6B-4bit", **kwargs)
    assert calls == []


def test_codex_attach_check_normalizes_shorthand_after_raw_probe(monkeypatch, capsys):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        if "repo_id=Qwen3-0.6B" in url and "unsloth" not in url:
            raise urllib.error.HTTPError(url, 400, "invalid repo_id", None, None)
        return {"variants": []}

    monkeypatch.setattr(start, "_http_json", http_json)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "Qwen3-0.6B")
    assert len(urls) == 2
    assert "repo_id=unsloth%2FQwen3-0.6B" in urls[1]
    assert "unsloth/Qwen3-0.6B" in capsys.readouterr().err


def test_codex_attach_check_trusts_raw_server_dir_answer(monkeypatch):
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        return {"variants": [{"quant": "Q4_K_M"}]}

    monkeypatch.setattr(start, "_http_json", http_json)
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "local-gguf-dir")
    assert len(urls) == 1


def test_codex_attach_check_rejects_live_empty_raw_shorthand(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    urls = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        urls.append(url)
        if "unsloth" in url:
            return {"variants": [{"quant": "Q4_K_M"}]}
        return {"variants": []}

    monkeypatch.setattr(start, "_http_json", http_json)
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "Qwen3-0.6B-GGUF")
    assert len(urls) == 1
    assert "Qwen3-0.6B-GGUF" in capsys.readouterr().err


def test_codex_attach_check_defers_shorthand_when_canonical_probe_errors(monkeypatch):
    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        raise urllib.error.HTTPError(url, 404, "nope", None, None)

    monkeypatch.setattr(start, "_http_json", http_json)
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "Qwen3-0.6B")


def test_codex_attach_check_probes_hub_shaped_gguf_ids(monkeypatch, capsys):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    urls = _fake_variants(monkeypatch, {"variants": []})
    with pytest.raises(typer.Exit):
        start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "owner/model.gguf")
    assert len(urls) == 1
    assert "owner%2Fmodel.gguf" in urls[0]


def test_codex_attach_check_defers_when_raw_name_exists_locally(monkeypatch, tmp_path):
    (tmp_path / "models" / "qwen").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    _fake_variants(monkeypatch, {"variants": []})
    start._attach_gguf_check(start._CODEX_GGUF_AGENT, BASE, "sk-test", "models/qwen")


def test_gguf_agents_name_themselves_and_their_own_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(start, "_is_hub_model_id", lambda name: True)
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: ["model-Q4_K_M.gguf"])
    for agent, label, command in (
        (start._CODEX_GGUF_AGENT, "Codex", "codex"),
        (start._CLAUDE_GGUF_AGENT, "Claude Code", "claude"),
    ):
        with pytest.raises(typer.Exit) as excinfo:
            start._fail_agent_needs_gguf(agent, "unsloth/gemma-3-4b-it")
        assert excinfo.value.exit_code == 1
        assert capsys.readouterr().err.strip() == (
            f"{label} needs a GGUF model served by llama-server, but unsloth/gemma-3-4b-it "
            f"is not one. Try: unsloth start {command} --model unsloth/gemma-3-4b-it-GGUF"
        )


def test_require_gguf_for_agent_reads_the_servers_status(monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: {"is_gguf": False, "active_model": "unsloth/gemma-3-4b-it"},
    )
    with pytest.raises(typer.Exit):
        start._require_gguf_for_agent(
            start._CLAUDE_GGUF_AGENT, BASE, "sk-test", "unsloth/gemma-3-4b-it"
        )
    assert "Claude Code needs a GGUF model" in capsys.readouterr().err
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: {"is_gguf": True})
    start._require_gguf_for_agent(
        start._CLAUDE_GGUF_AGENT, BASE, "sk-test", "unsloth/gemma-3-4b-it-GGUF"
    )


def test_claude_preflight_rejects_non_gguf_repo(monkeypatch, capsys):
    _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    with pytest.raises(typer.Exit):
        start._preflight_agent_gguf(start._CLAUDE_GGUF_AGENT, "mlx-community/Qwen3-0.6B-4bit")
    assert "Claude Code needs a GGUF model" in capsys.readouterr().err


def test_claude_command_preflights_before_starting_a_server(fake_studio, monkeypatch, tmp_path):
    _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    started = {"called": False}
    monkeypatch.setattr(
        start, "_start_studio_server", lambda *a, **k: started.__setitem__("called", True)
    )
    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "mlx-community/Qwen3-0.6B-4bit"]
    )
    assert result.exit_code == 1
    assert "Claude Code needs a GGUF model" in result.output
    assert started["called"] is False


@pytest.mark.parametrize(
    ("subcommand", "label"),
    [("claude", "Claude Code"), ("codex", "Codex")],
)
def test_a_rejected_model_never_offers_to_install_the_agent(
    monkeypatch, tmp_path, subcommand, label
):
    # No fake_studio: it stubs _require_agent_for_launch, the call under test here.
    _fake_hub_listing(monkeypatch, {"mlx-community/Qwen3-0.6B-4bit": []})
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda name: None)
    monkeypatch.setattr(start, "_start_studio_server", lambda *a, **k: None)
    offered = []
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda name, hint: offered.append(name),
    )
    result = CliRunner().invoke(
        start.start_app,
        [subcommand, "--model", "mlx-community/Qwen3-0.6B-4bit", "--launch"],
    )
    assert result.exit_code == 1
    assert f"{label} needs a GGUF model" in result.output
    assert offered == []


def test_a_missing_agent_is_still_reported_for_a_model_that_passes(monkeypatch, tmp_path):
    _fake_hub_listing(monkeypatch, {"unsloth/Qwen3-0.6B-GGUF": ["Q4_K_M.gguf"]})
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda name: None)
    monkeypatch.setattr(start, "_start_studio_server", lambda *a, **k: None)
    offered = []
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda name, hint: offered.append(name),
    )
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--model", "unsloth/Qwen3-0.6B-GGUF", "--launch"],
    )
    assert offered == ["claude"]
    assert result.exit_code == 1
    assert "not found on PATH" in result.output


def test_claude_preload_gate_rejects_before_an_evicting_load(fake_studio, monkeypatch):
    inner = start._http_json
    probed = []

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if "/api/models/gguf-variants" in url:
            probed.append(url)
            return {"variants": []}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B", "--no-launch"]
    )
    assert result.exit_code == 1
    assert "Claude Code needs a GGUF model" in result.output
    assert probed, "the gate must probe the server before the load"
    assert not [call for call in fake_studio if call[1].endswith("/api/inference/load")]


def test_claude_post_connect_failure_tears_down_auto_served(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started.update(base = base, model = model)
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(
        start, "_shutdown_server", lambda server: started.__setitem__("down", server)
    )
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "model_identifier": "transformers-model"}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["claude", "--model", "unsloth/Qwen3-1.7B", "--launch"]
    )
    assert result.exit_code != 0, result.output
    assert "Claude Code needs a GGUF model" in result.output
    assert started.get("down") is fake


def test_claude_post_connect_failure_spares_an_attached_server(fake_studio, monkeypatch):
    down = []
    monkeypatch.setattr(start, "_shutdown_server", lambda server: down.append(server))
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"is_gguf": False, "model_identifier": "transformers-model"}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "Claude Code needs a GGUF model" in result.output
    assert down == []


def _status_raises(monkeypatch, exc):
    def http_json(*args, **kwargs):
        raise exc

    monkeypatch.setattr(start, "_http_json", http_json)


def _require_claude_gguf():
    start._require_gguf_for_agent(
        start._CLAUDE_GGUF_AGENT, BASE, "sk-test", "unsloth/gemma-3-4b-it"
    )


@pytest.mark.parametrize(
    "code,reason",
    [
        (404, "not found"),
        (401, "Unauthorized"),
        (403, "Forbidden"),
        (500, "Failed to get status"),
        (503, "Service Unavailable"),
        (302, "refusing redirect"),
    ],
)
def test_require_gguf_never_rejects_on_an_http_error(monkeypatch, capsys, code, reason):
    url = f"{BASE}/api/inference/status"
    _status_raises(monkeypatch, urllib.error.HTTPError(url, code, reason, None, None))
    assert _require_claude_gguf() is None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
        TimeoutError("timed out"),
        OSError(101, "Network is unreachable"),
        json.JSONDecodeError("Expecting value", "<html>not json</html>", 0),
        http.client.BadStatusLine("garbage"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_require_gguf_never_rejects_on_a_transport_failure(monkeypatch, capsys, exc):
    _status_raises(monkeypatch, exc)
    assert _require_claude_gguf() is None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"model_identifier": "unsloth/Qwen3-1.7B-GGUF"},
        {"error": "API endpoint not found"},
        {"is_gguf": None},
        ["not", "a", "dict"],
        None,
    ],
)
def test_require_gguf_treats_an_unreadable_answer_as_unknown(monkeypatch, capsys, body):
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: body)
    assert _require_claude_gguf() is None
    assert "needs a GGUF model" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "agent,label",
    [("_CODEX_GGUF_AGENT", "Codex"), ("_CLAUDE_GGUF_AGENT", "Claude Code")],
)
def test_require_gguf_still_rejects_a_definite_no(monkeypatch, capsys, agent, label):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: {"is_gguf": False, "model_identifier": "transformers-model"},
    )
    with pytest.raises(typer.Exit) as excinfo:
        start._require_gguf_for_agent(
            getattr(start, agent), BASE, "sk-test", "unsloth/gemma-3-4b-it"
        )
    assert excinfo.value.exit_code == 1
    assert capsys.readouterr().err.strip() == (
        f"{label} needs a GGUF model served by llama-server, "
        "but unsloth/gemma-3-4b-it is not one."
    )


@pytest.mark.parametrize(
    "body",
    [
        {"is_gguf": False},
        {"is_gguf": False, "active_model": None},
        {"is_gguf": False, "active_model": None, "model_identifier": None},
    ],
)
def test_require_gguf_treats_an_idle_server_as_unknown(monkeypatch, body):
    monkeypatch.setattr(start, "_http_json", lambda *a, **k: body)
    start._require_gguf_for_agent(
        start._CLAUDE_GGUF_AGENT, BASE, "sk-test", "unsloth/gemma-3-4b-it-GGUF"
    )


def test_require_gguf_still_rejects_a_named_non_gguf_model(monkeypatch, capsys):
    monkeypatch.setattr(
        start,
        "_http_json",
        lambda *a, **k: {"is_gguf": False, "model_identifier": "unsloth/gemma-3-4b-it"},
    )
    with pytest.raises(typer.Exit):
        start._require_gguf_for_agent(
            start._CLAUDE_GGUF_AGENT, BASE, "sk-test", "unsloth/gemma-3-4b-it"
        )
    assert "Claude Code needs a GGUF model" in capsys.readouterr().err


def test_require_gguf_does_not_swallow_a_real_exit(monkeypatch):
    _status_raises(monkeypatch, typer.Exit(code = 1))
    with pytest.raises(typer.Exit):
        _require_claude_gguf()


def test_require_gguf_does_not_swallow_a_bug(monkeypatch):
    _status_raises(monkeypatch, AttributeError("typo in the stub"))
    with pytest.raises(AttributeError):
        _require_claude_gguf()


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_an_unreadable_status_leaves_the_auto_served_server_alone(fake_studio, monkeypatch, agent):
    monkeypatch.setattr(start, "_hub_gguf_files", lambda repo: None)
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    started = {}
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(
        base,
        model,
        load,
        server_options = None,
    ):
        started.update(base = base, model = model)
        start._auto_served_server = fake
        return base, fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(
        start, "_shutdown_server", lambda server: started.__setitem__("down", server)
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            raise urllib.error.HTTPError(url, 500, "Failed to get status", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, [agent, "--model", "unsloth/Qwen3-1.7B-GGUF", "--launch"]
    )
    assert result.exit_code == 0, result.output
    assert "needs a GGUF model" not in result.output
    assert started.get("down") is None


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_a_status_body_without_is_gguf_still_launches(fake_studio, monkeypatch, agent):
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/api/inference/status"):
            return {"model_identifier": MODEL["id"]}
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "needs a GGUF model" not in result.output


@pytest.mark.parametrize(
    ("agent", "unset"),
    [
        ("codex", ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")),
        ("openclaw", ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")),
    ],
)
def test_launch_drops_provider_credentials(agent, unset, fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    for name in unset:
        monkeypatch.setenv(name, "sk-stale")
    captured = _capture_launch(monkeypatch, [agent])
    for name in unset:
        assert name not in captured["env"]


@pytest.mark.parametrize("enabled", ["1", "true", "yes", "on"])
def test_openclaw_launch_disables_the_login_shell_key_fallback(enabled, fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/openclaw")
    monkeypatch.setenv("OPENCLAW_LOAD_SHELL_ENV", enabled)
    captured = _capture_launch(monkeypatch, ["openclaw"])
    assert captured["env"]["OPENCLAW_LOAD_SHELL_ENV"] == "0"


def test_openclaw_no_launch_recipe_disables_the_login_shell_key_fallback(fake_studio):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "OPENCLAW_LOAD_SHELL_ENV", "0")


def test_openclaw_state_dir_is_a_real_path_not_a_blank(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/openclaw")
    captured = _capture_launch(monkeypatch, ["openclaw"])
    assert captured["env"]["OPENCLAW_STATE_DIR"].strip()


def test_openclaw_config_pins_the_shell_env_fallback_off(fake_studio, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents = True)
    config_path.write_text(json.dumps({"env": {"shellEnv": {"enabled": True}}}))
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads(config_path.read_text())
    assert config["env"]["shellEnv"]["enabled"] is False


def _openclaw_without_the_settings_route(monkeypatch, exc = RuntimeError("404")):
    real = start._http_json

    def older_server(method, url, *args, **kwargs):
        if url.endswith("/api/settings/embedding-model"):
            raise exc
        return real(method, url, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", older_server)


def test_openclaw_memory_search_asks_for_keyword_only_without_the_settings_route(
    fake_studio, tmp_path, monkeypatch
):
    _openclaw_without_the_settings_route(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "openclaw" / "openclaw.json").read_text())
    search = config["memory"]["search"]
    assert search == {"provider": "none", "fallback": "none"}


def test_openclaw_memory_search_drops_a_stale_remote_when_the_route_is_gone(
    fake_studio, tmp_path, monkeypatch
):
    _openclaw_without_the_settings_route(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents = True)
    config_path.write_text(
        json.dumps(
            {
                "memory": {
                    "search": {
                        "provider": "openai-compatible",
                        "model": "unsloth/bge-small-en-v1.5",
                        "fallback": "none",
                        "remote": {
                            "baseUrl": "http://127.0.0.1:8888/v1",
                            "apiKey": "sk-unsloth-old",
                        },
                    }
                }
            }
        )
    )
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    search = json.loads(config_path.read_text())["memory"]["search"]
    assert search == {"provider": "none", "fallback": "none"}


def test_openclaw_memory_search_model_is_stripped(fake_studio, tmp_path, monkeypatch):
    real = start._http_json

    def padded(method, url, *args, **kwargs):
        if url.endswith("/api/settings/embedding-model"):
            return {"embedding_model": "  unsloth/bge-small-en-v1.5\n"}
        return real(method, url, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", padded)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "openclaw" / "openclaw.json").read_text())
    assert config["memory"]["search"]["model"] == "unsloth/bge-small-en-v1.5"


def test_openclaw_memory_search_does_not_swallow_a_cli_abort(fake_studio, tmp_path, monkeypatch):
    _openclaw_without_the_settings_route(monkeypatch, exc = typer.Exit(2))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 2, result.output


def test_openclaw_memory_search_clears_a_stale_external_fallback(
    fake_studio, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents = True)
    config_path.write_text(json.dumps({"memory": {"search": {"fallback": "openai"}}}))
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads(config_path.read_text())
    assert config["memory"]["search"]["fallback"] == "none"
    assert config["memory"]["search"]["provider"] == "openai-compatible"


def test_direct_gguf_labels_keep_packed_and_grouped_quants():
    # Mirrors model_config._extract_quant_label: prism-ml/Ternary-Bonsai-*-gguf ships all three.
    for name, label in (
        ("Ternary-Bonsai-8B-PQ2_0.gguf", "PQ2_0"),
        ("Ternary-Bonsai-8B-Q2_0.gguf", "Q2_0"),
        ("Ternary-Bonsai-8B-Q2_0_g64.gguf", "Q2_0_g64"),
        ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", "PTQ1_0"),
    ):
        assert start._direct_gguf_variant_labels(name)[1] == label


GiB = 1024**3


class _LoadServer:
    """Fake Studio: `listing` answers active-downloads, `repos` maps repo -> iterator of progress readings."""

    def __init__(self, listing, repos):
        self.listing = listing
        self.repos = {repo: iter(readings) for repo, readings in repos.items()}
        self.urls = []

    def __call__(
        self,
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        self.urls.append(url)
        if url.endswith("/active-downloads"):
            if isinstance(self.listing, Exception):
                raise self.listing
            return {"downloads": self.listing}
        repo = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["repo_id"][0]
        reading = next(self.repos[repo])
        if isinstance(reading, Exception):
            raise reading
        return reading

    def count(self, fragment):
        return sum(fragment in url for url in self.urls)


def _reading(
    downloaded,
    expected = 4 * GiB,
    completed = 0,
):
    return {
        "downloaded_bytes": downloaded,
        "completed_bytes": completed,
        "expected_bytes": expected,
    }


def _load_job(repo, **extra):
    return {"repo_id": repo, "owner": "load", "state": "running", **extra}


def _progress(monkeypatch, server, clock):
    monkeypatch.setattr(start, "_http_json", server)
    monkeypatch.setattr(start.time, "monotonic", lambda: clock[0])
    return start._ModelDownloadProgress(BASE, "sk-test", "owner/adapter", None)


def test_model_download_progress_counts_the_base_a_load_reports(monkeypatch, capsys):
    adapter_done = _reading(8 * 1024**2, 8 * 1024**2, 8 * 1024**2)
    server = _LoadServer(
        [
            _load_job("owner/base"),
            {"repo_id": "someone/else", "state": "running"},
            {"repo_id": "owner/attached", "state": "running", "load_attached": True},
        ],
        {
            "owner/adapter": [adapter_done] * 3,
            "owner/base": [_reading(0), _reading(1 * GiB), _reading(2 * GiB)],
            "owner/attached": [_reading(3 * GiB)] + [TimeoutError("slow")] * 2,
        },
    )
    clock = [0.0]
    progress = _progress(monkeypatch, server, clock)

    for _ in range(3):
        progress.poll()
        clock[0] += 1.0

    assert progress.downloaded_bytes == 8 * 1024**2 + 2 * GiB + 3 * GiB
    assert server.count("someone%2Felse") == 0
    out = capsys.readouterr().out
    assert "1.0 GiB / 4.0 GiB" in out and "3.0 GiB" not in out


def test_model_download_progress_lists_load_jobs_at_most_every_interval(monkeypatch):
    done = _reading(4 * GiB, completed = 4 * GiB)
    server = _LoadServer(
        [_load_job("owner/base")],
        {
            "owner/adapter": [_reading(1024)] * 20,
            "owner/base": [_reading(GiB), done] + [AssertionError] * 20,
        },
    )
    clock = [0.0]
    progress = _progress(monkeypatch, server, clock)

    for _ in range(12):
        progress.poll()
        clock[0] += 1.0

    assert server.count("/active-downloads") == 3
    assert server.count("owner%2Fbase") == 2
    assert progress.downloaded_bytes == 1024 + 4 * GiB


def test_model_download_progress_stops_listing_on_a_server_without_the_route(monkeypatch):
    missing = urllib.error.HTTPError(BASE, 404, "Not Found", None, None)
    server = _LoadServer(missing, {"owner/adapter": [_reading(1024)] * 20})
    clock = [0.0]
    progress = _progress(monkeypatch, server, clock)

    for _ in range(12):
        progress.poll()
        clock[0] += 1.0

    assert server.count("/active-downloads") == 1
    assert progress.downloaded_bytes == 1024


def test_download_progress_display_restarts_when_the_repo_changes(monkeypatch, capsys):
    monkeypatch.setattr(start.sys.stdout, "isatty", lambda: False, raising = False)
    display = start._DownloadProgressDisplay()

    display.update(_reading(19 * 1024**2, 20 * 1024**2), "owner/adapter")
    display.update(_reading(GiB), "owner/base")

    out = capsys.readouterr().out
    assert "19.0 MiB / 20.0 MiB" in out
    assert "1.0 GiB / 4.0 GiB" in out


def test_model_download_progress_counts_the_remote_base_of_a_local_adapter(monkeypatch):
    server = _LoadServer(
        [_load_job("owner/base")], {"owner/base": [_reading(GiB), _reading(2 * GiB)]}
    )
    clock = [0.0]
    monkeypatch.setattr(start, "_http_json", server)
    monkeypatch.setattr(start.time, "monotonic", lambda: clock[0])
    progress = start._ModelDownloadProgress(BASE, "sk-test", "/models/my-lora", None)

    progress.poll()
    progress.poll()

    assert progress.downloaded_bytes == 2 * GiB
    assert server.count("my-lora") == 0
