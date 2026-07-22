# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth start` — config merging and launch env, no network."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "unsloth/gemma-4-26B-A4B-it-GGUF", "context_length": 131072}


# --no-launch prints shell setup as POSIX (export/unset) on Unix/WSL and
# PowerShell ($env:/Remove-Item) on native Windows; assert the host's form.
def _assert_env_set(output: str, name: str, value: str) -> None:
    needle = f'$env:{name} = "{value}"' if os.name == "nt" else f"export {name}={value}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_unset(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _launch_command(output: str) -> list:
    # The --no-launch recipe ends with a self-contained one-liner: inline NAME=value
    # assignments, then the command. Return just the command argv.
    last = [ln for ln in output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    for i, part in enumerate(parts):
        name = part.partition("=")[0]
        if "=" not in part or not name.replace("_", "").isalnum():
            return parts[i:]
    return []


def _fake_claude(monkeypatch, version_output: str) -> None:
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = version_output),
    )


def _path_aware_which(binaries: dict):
    # A shutil.which fake that resolves a name only when its directory is on PATH at call time.
    # Lets a test prove a version probe augments PATH before resolving: an agent present only in
    # an install dir (~/.local/bin, %APPDATA%\npm) must still be found and version-checked.
    def _which(name):
        directory = binaries.get(name)
        if directory is None:
            return None
        entries = os.environ.get("PATH", "").split(os.pathsep)
        # os.path.join (not Path()) so this works when a test has flipped os.name to "nt": under
        # a simulated os.name, pathlib would build the non-native flavour and raise.
        return os.path.join(str(directory), name) if str(directory) in entries else None

    return _which


def _simulate_windows(monkeypatch) -> None:
    # Exercise the `os.name == "nt"` branch on any host. Flipping os.name alone makes pathlib
    # pick the non-native flavour (WindowsPath on POSIX, PosixPath on Windows) when a Path is
    # constructed, which raises; pin Path to the host-native class (captured before the flip)
    # so the branch logic runs without that crash. Keeps these tests green on Linux/Mac/WSL too.
    monkeypatch.setattr(start, "Path", type(Path()))
    monkeypatch.setattr(start.os, "name", "nt")


def test_claude_flags_passed_to_supported_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.1.98 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_skipped_on_old_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.0.14 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == []


def test_claude_flags_skipped_on_unparseable_version(monkeypatch):
    _fake_claude(monkeypatch, "weird build string\n")
    assert start._claude_flags(MODEL["id"]) == []


def test_claude_flags_detected_when_version_not_first_token(monkeypatch):
    # The X.Y.Z is pulled from anywhere in the output, so a format change (version not
    # the first token) doesn't silently drop the optimization flags.
    _fake_claude(monkeypatch, "claude version 2.1.98\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_settings_overlay_pins_served_model():
    # The session overlay must pin availableModels to the served model: a user's allowlist
    # in ~/.claude/settings.json otherwise rejects the Unsloth --model ("restricted by your
    # organization's settings"), and no env var can bypass it. The override must be a
    # NON-EMPTY array to take effect (an empty [] is ignored and the user's list still
    # applies), so it lists exactly this model, for this session only.
    overlay = json.loads(start._claude_settings_overlay(MODEL["id"]))
    assert overlay["availableModels"] == [MODEL["id"]]
    # The attribution-header suppression is preserved alongside it.
    assert overlay["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    # Subagents fall through to the served model instead of a user's opus/sonnet pin.
    assert overlay["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"


def test_install_agent_prompts_then_installs(monkeypatch):
    # TTY + yes: run the documented install command, then re-resolve the now-present binary.
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    ran = []
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    # _install_agent only re-resolves after installing (the pre-install check is the
    # caller's job), so `which` reports the now-present binary.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    executable = start._install_agent("codex", "npm install -g @openai/codex")
    assert executable == "/usr/local/bin/codex"
    assert ran == [["/bin/sh", "-c", "npm install -g @openai/codex"]]


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
    # -ExecutionPolicy Bypass (process-scoped) lets npm's npm.ps1 wrapper and irm|iex
    # scripts run even when the machine policy is the Windows default Restricted.
    assert ran == [
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", install_hint]
    ]


def test_install_agent_windows_failure_hints_execution_policy(monkeypatch, capsys):
    # A failed install on Windows points the user at the per-user execution-policy fix:
    # our subprocess bypasses the policy, but their own shell may still block npm.ps1
    # (PSSecurityException) when they run the install by hand.
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
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


def test_install_agent_posix_failure_omits_execution_policy_hint(monkeypatch, capsys):
    # The execution-policy hint is Windows-only; a POSIX install failure must not mention it.
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


def test_install_agent_warns_remote_installer_is_unverified_third_party(monkeypatch, capsys):
    # Before the confirm, a remote installer must name the URL it fetches so the
    # user consents to a specific source rather than blindly accepting.
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)  # decline: nothing runs
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
    # An npm-style installer has no URL to fetch, but still runs with the user's
    # privileges, so the warning names the command instead.
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    assert start._install_agent("codex", "npm install -g @openai/codex") is None
    err = capsys.readouterr().err
    assert "npm install -g @openai/codex" in err
    assert "with your privileges" in err


def test_hermes_install_hint_is_windows_native_on_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "nt")

    # Scriptblock form so `-SkipSetup` reaches the installer and the interactive
    # setup wizard is skipped during the unattended `unsloth start hermes` run.
    assert start._hermes_install_hint() == (
        f"& ([scriptblock]::Create((irm {start._HERMES_INSTALL_BASE}/install.ps1)))"
        f" -SkipSetup -Commit {start._HERMES_INSTALL_COMMIT}"
    )


def test_hermes_install_hint_is_bash_on_posix(monkeypatch):
    monkeypatch.setattr(start.os, "name", "posix")

    # `bash -s -- --skip-setup` forwards the skip flag to the piped installer.
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
    # Fake Windows registry PATH values written after this process started.
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
    # Claude's installer drops its binary in ~/.local/bin but only *suggests* adding it to
    # PATH, so Unsloth appends it in-process to resolve the freshly installed agent.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(local_bin) in entries
    # Appended (lowest precedence), so it never shadows an existing PATH entry.
    assert entries[-1] == str(local_bin)


def test_augment_path_skips_missing_and_duplicate_dirs(monkeypatch, tmp_path):
    # A non-existent ~/.local/bin is not added; an already-present one is not duplicated.
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no .local/bin created yet
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"] == str(tmp_path / "existing")

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "existing"), str(local_bin)]))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"].split(os.pathsep).count(str(local_bin)) == 1


def test_augment_path_adds_npm_global_bin_on_windows(monkeypatch, tmp_path):
    # npm -g shims (codex/opencode/pi) land in %APPDATA%\npm on Windows; add it so a freshly
    # installed npm agent resolves even when that dir isn't on PATH yet.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no ~/.local/bin created
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert str(npm_dir) in os.environ["PATH"].split(os.pathsep)


def test_which_with_install_dirs_finds_agent_and_restores_path(monkeypatch, tmp_path):
    # The probe helper resolves against the augmented PATH but must NOT persist it: only
    # _launch() should mutate PATH for the child process. Here `claude` is only in ~/.local/bin.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    original = str(tmp_path / "existing")
    monkeypatch.setenv("PATH", original)  # local_bin NOT on PATH yet
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    assert start._which_with_install_dirs("claude") == str(local_bin / "claude")
    assert os.environ["PATH"] == original  # restored, no global pollution


def test_claude_flags_probes_old_agent_only_in_install_dir(monkeypatch, tmp_path):
    # Regression: the version probe must augment PATH before resolving, so an OLD claude present
    # only in ~/.local/bin (not yet on PATH) is detected as old and the unsupported flags are
    # dropped -- the same binary _launch() will run. Before the fix the probe saw no binary,
    # assumed a current build, and emitted flags the old claude rejects.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == []


def test_claude_flags_detects_supported_agent_only_in_install_dir(monkeypatch, tmp_path):
    # The counterpart: a SUPPORTED claude present only in ~/.local/bin is now resolved and gets
    # the flags, instead of being missed and (coincidentally) also assumed current.
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
    # npm -g shims land in %APPDATA%\npm on Windows; an old claude there (not on PATH) must still
    # be version-checked so the unsupported flags are dropped.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no ~/.local/bin created
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": npm_dir}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == []


def test_codex_catalog_probes_old_codex_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for codex: an old codex present only in an install dir is detected so the
    # model-catalog config is omitted (the old binary can't consume it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"codex": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "codex-cli 0.109.0")
    assert start._codex_supports_model_catalog() is False


def test_opencode_native_auto_probes_old_opencode_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for opencode: an old opencode present only in an install dir is detected
    # so native --auto is not assumed (the old binary rejects it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"opencode": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "1.17.11")
    assert start._opencode_supports_native_auto() is False


def test_augment_path_preserves_defpath_when_path_unset(monkeypatch, tmp_path):
    # PATH unset: shutil.which() and exec*p* fall back to os.defpath (e.g. /bin:/usr/bin), so the
    # augmentation must keep those default dirs instead of collapsing to just the install dir
    # (which would hide a system-installed agent and strip the launched child's normal PATH).
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
    # With PATH unset, a system agent on os.defpath (e.g. /usr/bin) must still resolve; the
    # install-dir augmentation must not drop the default search path. PATH is restored to unset.
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
    # TTY + no: never runs anything; caller falls back to the print-hint failure.
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install when declined")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def test_install_agent_non_interactive_returns_none(monkeypatch):
    # No TTY (piped stdin): cannot prompt, so don't install; return None silently.
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: False))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install without a TTY")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


def test_merge_codex_config_fresh():
    merged = start._merge_codex_config("", BASE)
    parsed = _parse_toml(merged)
    assert parsed["oss_provider"] == "unsloth_api"
    provider = parsed["model_providers"]["unsloth_api"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False


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
    start.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    assert profile["oss_provider"] == "unsloth_api"
    assert profile["model_provider"] == "unsloth_api"
    assert profile["model"] == MODEL["id"]
    assert profile["model_context_window"] == 131072

    catalog_path = Path(profile["model_catalog_json"])
    assert catalog_path == Path("model-catalog.json")
    catalog = json.loads((tmp_path / catalog_path).read_text())
    assert catalog["models"][0]["slug"] == MODEL["id"]
    assert catalog["models"][0]["context_window"] == 131072
    assert catalog["models"][0]["max_context_window"] == 131072
    assert catalog["models"][0]["supports_reasoning_summary_parameter"] is False
    assert catalog["models"][0]["supports_parallel_tool_calls"] is False

    assert catalog["models"][0]["base_instructions"] == start._CODEX_FALLBACK_PROMPT.read_text()
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


def test_write_codex_config_omits_catalog_for_old_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "unsloth_api.config.toml").read_text())
    assert "model_catalog_json" not in profile
    assert not (tmp_path / "model-catalog.json").exists()


def test_write_codex_subagent_config_keeps_parent_model_out(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    path = start.write_codex_subagent_config(BASE, "private-token", local, tmp_path)
    agent = _parse_toml(path.read_text())
    assert agent["name"] == "unsloth"
    assert "local agent" in agent["description"].lower()
    assert agent["model_provider"] == start._CODEX_PROFILE
    assert agent["model"] == local["id"]
    assert agent["model_context_window"] == MODEL["context_length"]
    assert agent["model_providers"][start._CODEX_PROFILE] == {
        "name": "Unsloth Studio",
        "base_url": f"{BASE}/v1",
        "wire_api": "responses",
        "auth": {
            "command": sys.executable,
            "args": [
                "-c",
                "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['token'])",
                str(tmp_path / "unsloth-auth.json"),
            ],
            "timeout_ms": 5000,
        },
    }
    assert json.loads((tmp_path / "unsloth-auth.json").read_text()) == {"token": "private-token"}
    catalog = json.loads((tmp_path / agent["model_catalog_json"]).read_text())
    assert catalog["models"][0]["slug"] == local["id"]


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_codex_subagent_auth_uses_wsl_for_windows_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex.exe",
    )

    path = start.write_codex_subagent_config(BASE, "private-token", MODEL, tmp_path)
    auth = _parse_toml(path.read_text())["model_providers"][start._CODEX_PROFILE]["auth"]

    assert auth["command"] == "wsl.exe"
    assert auth["args"][:5] == ["-d", "Ubuntu", "--", sys.executable, "-c"]
    assert auth["args"][-1] == str(tmp_path / "unsloth-auth.json")


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


@pytest.mark.parametrize("agent", ["openclaw", "hermes"])
def test_unsupported_agents_reject_as_subagent(agent):
    result = CliRunner().invoke(start.start_app, [agent, "--as-subagent"])
    assert result.exit_code == 1
    assert f"--as-subagent is not supported for {agent}." in result.output


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
        if url.endswith("/v1/models"):
            return {"object": "list", "data": state["models"]}
        if url.endswith("/api/inference/status"):
            return {"is_gguf": True, "model_identifier": state["models"][0]["id"]}
        if url.endswith("/api/auth/api-keys"):
            return {"key": "sk-unsloth-feedfacefeedface"}
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
    # Identity handshake has its own tests; trust the loopback server here.
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: True)
    # _studio_token / api-keys are faked so the mint flow stays offline.
    monkeypatch.setattr(start, "_studio_token", lambda: "jwt-token")
    monkeypatch.setattr(start, "_http_json", http_json)
    monkeypatch.setattr(start, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    # --no-launch session configs land under tmp instead of the real Unsloth dir.
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    # No `claude` on PATH, so _claude_flags never probes the real binary.
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    return calls


def test_connect_claude_no_launch(fake_studio):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_unset(result.output, "ANTHROPIC_API_KEY")
    _assert_env_unset(result.output, "CLAUDE_CODE_OAUTH_TOKEN")
    _assert_env_set(result.output, "ANTHROPIC_BASE_URL", BASE)
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    # Suppress the full-screen TUI redraw so a bursty local server doesn't flicker.
    _assert_env_set(result.output, "CLAUDE_CODE_NO_FLICKER", "1")
    # Attribution header is suppressed for the session via env + --settings, never
    # by writing the user's ~/.claude/settings.json.
    _assert_env_set(result.output, "CLAUDE_CODE_ATTRIBUTION_HEADER", "0")
    # Auto-compact window is sized to the loaded model's real context length so the
    # session compacts before it overflows the local server's (much smaller) window,
    # and compaction is forced at 90% of it for headroom.
    _assert_env_set(result.output, "CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(MODEL["context_length"]))
    _assert_env_set(result.output, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "90")
    assert f"claude --model {MODEL['id']} --exclude-dynamic-system-prompt-sections" in result.output
    # Overlay is passed inline (session-only), not a path into the user's ~/.claude.
    assert "--settings" in result.output
    assert ".claude/settings.json" not in result.output


def test_connect_claude_as_subagent_preserves_cloud_parent(fake_studio, tmp_path):
    variant = MODEL["id"] + ":UD-Q4_K_XL"
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "--model", variant, "hello"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    plugin = tmp_path / "agents" / "claude-subagent" / "unsloth-local-agent"
    assert command == [
        "claude",
        "--plugin-dir",
        str(plugin),
        "--allowedTools",
        start._CLAUDE_SUBAGENT_TOOL,
        "hello",
    ]
    assert "--model" not in command
    parent_base = "$env:ANTHROPIC_BASE_URL" if os.name == "nt" else "export ANTHROPIC_BASE_URL="
    parent_token = (
        "$env:ANTHROPIC_AUTH_TOKEN" if os.name == "nt" else "export ANTHROPIC_AUTH_TOKEN="
    )
    assert parent_base not in result.output
    assert parent_token not in result.output
    assert "UNSLOTH_CLAUDE_SUBAGENT_API_KEY" not in result.output
    assert "sk-unsloth-feedfacefeedface" not in result.output
    mcp = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["unsloth"]
    assert mcp["command"] == sys.executable
    assert mcp["args"] == ["-m", start._CLAUDE_SUBAGENT_MCP_MODULE]
    assert mcp["env"] == {
        "UNSLOTH_CLAUDE_SUBAGENT_BASE_URL": BASE,
        "UNSLOTH_CLAUDE_SUBAGENT_API_KEY": "sk-unsloth-feedfacefeedface",
        "UNSLOTH_CLAUDE_SUBAGENT_MODEL": variant,
        "UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS": "0",
        "UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW": "4096",
    }
    assert "Ask Claude to spawn an Unsloth or local agent." in result.output


def test_connect_claude_native_subagent_runs_local_parent(fake_studio, tmp_path):
    variant = MODEL["id"] + ":UD-Q4_K_XL"
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--native-subagent", "--no-launch", "--model", variant, "hello"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    plugin = tmp_path / "agents" / "claude-native-subagent" / "unsloth-local-agent"
    # Parent runs on the local endpoint so the in-process subagent can reach the model.
    assert command == [
        "claude",
        "--model",
        variant,
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(variant),
        "--plugin-dir",
        str(plugin),
        "hello",
    ]
    _assert_env_set(result.output, "ANTHROPIC_BASE_URL", BASE)
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    _assert_env_set(result.output, "ANTHROPIC_MODEL", variant)
    _assert_env_unset(result.output, "ANTHROPIC_API_KEY")
    # No MCP bridge: the plugin ships a native subagent, not an .mcp.json child process.
    assert not (plugin / ".mcp.json").exists()
    assert json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())["name"] == (
        "unsloth-local-agent"
    )
    agent = (plugin / "agents" / f"{start._CLAUDE_SUBAGENT_NAME}.md").read_text()
    assert f'model: "{variant}"' in agent
    assert f"'{start._CLAUDE_SUBAGENT_NAME}'" in result.output


def test_write_claude_native_subagent_plugin_pins_local_model(tmp_path):
    plugin = start.write_claude_native_subagent_plugin(tmp_path, {"id": MODEL["id"] + ":Q4_K_M"})
    agent = (plugin / "agents" / f"{start._CLAUDE_SUBAGENT_NAME}.md").read_text()
    # Frontmatter pins the local model and drops plan-mode tools; body is the instructions.
    assert f'name: "{start._CLAUDE_SUBAGENT_NAME}"' in agent
    assert f'model: "{MODEL["id"]}:Q4_K_M"' in agent
    assert 'disallowedTools: "EnterPlanMode, ExitPlanMode"' in agent
    assert start._SUBAGENT_INSTRUCTIONS in agent


def test_connect_claude_rejects_both_subagent_modes(fake_studio):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--native-subagent", "--no-launch"],
    )
    assert result.exit_code != 0
    assert "Choose either --as-subagent or --native-subagent, not both." in result.output


def test_connect_claude_compact_window_omitted_without_context(fake_studio, monkeypatch):
    # A model that doesn't report a context length -> leave Claude's default window
    # rather than guessing one.
    monkeypatch.setattr(start, "_resolve_model", lambda *a, **k: {"id": "local-model"})
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in result.output
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result.output


def test_connect_claude_launch_scrubs_conflicting_auth_env(fake_studio, monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
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
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in captured["env"]
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_windows_shim_from_wsl_bridges_env(fake_studio, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])

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
    ]
    assert captured["env"]["ANTHROPIC_API_KEY"] == ""
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-feedfacefeedface"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["PWD"] == str(tmp_path)
    assert "PWD/p" in captured["env"]["WSLENV"].split(":")
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        assert name in captured["env"]["WSLENV"].split(":")


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_no_launch_windows_shim_from_wsl_prints_wslenv(
    fake_studio, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )

    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert "export ANTHROPIC_API_KEY=" in result.output
    assert "export CLAUDE_CODE_OAUTH_TOKEN=" in result.output
    assert "export WSLENV=" in result.output
    # PWD must NOT be frozen into the recipe (no `export PWD=`): WSLENV PWD/p translates the
    # shell's live PWD at run time, so a recipe reused from another dir resolves the project root.
    assert "export PWD=" not in result.output
    assert "PWD/p" in result.output
    assert "ANTHROPIC_AUTH_TOKEN" in result.output
    assert "CLAUDE_CODE_OAUTH_TOKEN" in result.output


def test_connect_codex_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "UNSLOTH_STUDIO_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    assert "codex --oss --profile unsloth_api" in result.output
    # Config lands in the session-scoped CODEX_HOME, not the user's ~/.codex.
    home = tmp_path / "agents" / "codex"
    _assert_env_set(result.output, "CODEX_HOME", str(home))
    assert (home / "config.toml").exists()
    assert (home / "unsloth_api.config.toml").exists()


def test_connect_codex_as_subagent_preserves_cloud_parent(fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
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
    assert command[1:3] == ["--enable", "multi_agent"]
    assert "agents.max_depth=1" in command
    assert "--oss" not in command
    assert "--profile" not in command
    assert "--model" not in command
    assert "CODEX_HOME" not in result.output
    assert start._CODEX_ENV_KEY not in result.output
    assert "sk-unsloth-feedfacefeedface" not in result.output
    home = tmp_path / "agents" / "codex-subagent"
    agent_path = home / "unsloth.toml"
    agent = _parse_toml(agent_path.read_text())
    assert agent["model"] == MODEL["id"] + ":UD-Q4_K_XL"
    assert "env_key" not in agent["model_providers"][start._CODEX_PROFILE]
    assert f"agents.unsloth.config_file={json.dumps(str(agent_path))}" in command
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
        if url.endswith("/v1/models"):
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


def test_resolve_model_loads_when_catalog_hit_is_not_loaded(monkeypatch):
    # A cached-but-unloaded catalog entry (loaded == False) that only case-differs must
    # not be treated as ready; the load endpoint must still be called so the requested
    # model becomes resident instead of the agent preflighting a different backend.
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
        if url.endswith("/v1/models"):
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
        if url.endswith("/v1/models"):
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
    # The mirror case: a loaded entry (loaded == True) that case-matches attaches with
    # no /api/inference/load call.
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
        if url.endswith("/v1/models"):
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
    # Against a remote Unsloth the local existence probe cannot see server-side paths,
    # so a case-variant loaded id must NOT attach without a load: it could be a distinct
    # server-side path on a case-sensitive host. The load endpoint resolves the request.
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
        if url.endswith("/v1/models"):
            return {
                "data": [{"id": "unsloth/Gemma-4-GGUF", "loaded": True, "context_length": 131072}]
            }
        if url.endswith("/api/inference/load"):
            state["loaded"] = True
            return {"model": "unsloth/Gemma-4-GGUF"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(start, "_http_json", http_json)

    entry = start._resolve_model("http://10.0.0.5:8888", "sk-test", "unsloth/gemma-4-gguf")

    # The load endpoint was consulted (no casefold shortcut), and we still attach to the
    # server's canonical id it reports back.
    assert entry["id"] == "unsloth/Gemma-4-GGUF"
    assert any(u.endswith("/api/inference/load") for _, u in calls)


def test_model_id_matching_does_not_casefold_local_paths(tmp_path):
    existing_local = tmp_path / "Org" / "Foo"
    existing_local.mkdir(parents = True)

    assert start._model_id_matches("Org/Foo", "org/foo")
    assert not start._model_id_matches(str(existing_local), str(existing_local).lower())
    assert not start._model_id_matches("./Models/Foo", "./models/foo")
    assert not start._model_id_matches(r".\Models\Foo", r".\models\foo")
    # A server-side relative path (extra path segments) is not a hub id even when it
    # does not exist on the CLI host, so it must not casefold-match a differently
    # cased path on a case-sensitive server filesystem.
    assert not start._is_hub_model_id("models/Llama/Foo.gguf")
    assert not start._model_id_matches("models/Llama/Foo.gguf", "models/llama/foo.gguf")
    # A genuine two-segment hub id still matches case-insensitively.
    assert start._is_hub_model_id("unsloth/Gemma-3-4b-it-GGUF")
    assert start._model_id_matches("unsloth/Gemma-3-4b-it-GGUF", "unsloth/gemma-3-4b-it-gguf")
    # Casefolding is gated to loopback studios (allow_casefold). With it disabled (a
    # remote studio, where a two-segment string could be a server-side path), even a
    # genuine hub-id case variant must not match, so the load endpoint resolves it.
    assert not start._model_id_matches(
        "unsloth/Gemma-3-4b-it-GGUF", "unsloth/gemma-3-4b-it-gguf", allow_casefold = False
    )
    assert start._model_id_matches("unsloth/Foo", "unsloth/Foo", allow_casefold = False)


def test_connect_codex_launch_uses_ephemeral_home(fake_studio, monkeypatch):
    # Launch mode writes config to a throwaway temp CODEX_HOME and removes it after
    # the agent exits; the user's real ~/.codex is never the target.
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
    assert captured["config_present"]  # config existed while codex ran
    assert "unsloth-codex-" in home.name  # an ephemeral temp dir, not ~/.codex
    assert not home.exists()  # cleaned up after the agent exits


@pytest.mark.skipif(
    os.name == "nt",
    reason = "the #6547 CI parser is bash-only; on Windows --no-launch prints PowerShell",
)
def test_no_launch_output_is_parseable(fake_studio):
    # Mirror the #6547 CI parser: status lines, then `export`/`unset`, then exactly
    # one launch command on the last line (now an inline-env one-liner, so the parser
    # matches by substring rather than prefix).
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    skip = ("export ", "unset ", "Unsloth ", "Updated ", "Disabled ", "Warning", "Loading")
    body = [ln for ln in lines if not ln.startswith(skip)]
    assert "codex --oss --profile unsloth_api" in body[-1]
    assert any(ln.startswith("export CODEX_HOME=") for ln in lines)


def test_no_launch_last_line_is_self_contained(fake_studio, tmp_path):
    # People copy just the last line. A bare `codex` there would run against the user's
    # real ~/.codex (e.g. a pre-existing damaged state DB) with zero isolation, so the
    # last line must inline every session env var ahead of the command.
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
    # The unset vars must be neutralized inline too, or a partial copy would send the
    # user's own ANTHROPIC_API_KEY to the Unsloth base.
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    assert "ANTHROPIC_API_KEY= " in last
    assert "CLAUDE_CODE_OAUTH_TOKEN= " in last
    assert "ANTHROPIC_AUTH_TOKEN=" in last  # the real key still applied after the blanks


def test_opencode_inline_config_beats_project_config(fake_studio):
    # A project's opencode.json outranks OPENCODE_CONFIG, so the model pin (and --yolo
    # permissions) ride in OPENCODE_CONFIG_CONTENT, which outranks project config.
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
    assert "sk-unsloth" not in result.output  # key stays in the private file, not the env


def test_opencode_inline_config_omits_permission_without_yolo(fake_studio):
    # A non-yolo session carries no permission inline. OPENCODE_CONFIG_CONTENT outranks the
    # project opencode.json we cannot read, so forcing any value there would override the
    # user's project rules; clearing our own config is the fix, and the inline pins the model.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "permission" not in inline


def test_https_loopback_never_auto_serves(fake_studio, monkeypatch):
    # `unsloth run` serves plain HTTP; auto-serving behind an https:// target would poll
    # the wrong scheme until the startup timeout. Keep the plain "no server" error.
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
    # `unsloth connect` remains a compat alias for `unsloth start`.
    from unsloth_cli import app

    result = CliRunner().invoke(app, ["connect", "claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])


def test_connect_key_minted_once_then_cached(fake_studio, tmp_path):
    CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    # First run mints; second reuses the minted key cached for this server.
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert len(mints) == 1
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface"]


def test_connect_explicit_key_remembered_for_keyless_runs(fake_studio, tmp_path):
    CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    # Reused, not re-minted (a mint would return the feedface stand-in).
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    # An explicit key is remembered as "saved" so it replays without the handshake.
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
        if url.endswith("/v1/models") and token == "sk-unsloth-stale":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    # The working key moves to the front so the next run tries it first.
    cached = json.loads(cache.read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface", "sk-unsloth-stale"]


def test_connect_saved_key_server_outage_surfaces_not_reminted(fake_studio, tmp_path, monkeypatch):
    # A 5xx/timeout while checking a saved key is a server outage, not a rejected key:
    # surface it instead of discarding the key and minting a new one against a sick server.
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
        if url.endswith("/v1/models") and token == "sk-unsloth-saved":
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code != 0, result.output
    # The outage did not cause a fresh key to be minted.
    mints = [c for c in fake_studio if c[1].endswith("/api/auth/api-keys")]
    assert mints == []


def test_connect_legacy_unscoped_cache_not_replayed(fake_studio, tmp_path):
    # Legacy unscoped caches have no server binding (could leak across servers),
    # so they're ignored: a fresh key is minted and stored scoped to this server.
    (tmp_path / "agent_api_key.json").write_text(json.dumps({"key": "sk-unsloth-oldformat"}))
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-feedfacefeedface")
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["minted"] == ["sk-unsloth-feedfacefeedface"]
    assert "key" not in cached  # legacy field collapsed away


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


def test_connect_model_flag_forwards_load_options(fake_studio):
    # The model-load knobs mirrored from `unsloth run` reach /api/inference/load.
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
    # Unsloth registers a loaded model under a canonical id (resolved identifier
    # / casing) that can differ from the path we passed. The agent must connect
    # to that model, not silently fall through to the first loaded one.
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
        if url.endswith("/v1/models"):
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
        ("unsloth/Qwen3-1.7B-GGUF", ("unsloth/Qwen3-1.7B-GGUF", None)),  # no suffix
        ("/models/local.gguf", ("/models/local.gguf", None)),  # absolute path
        ("./rel.gguf", ("./rel.gguf", None)),  # relative path
        ("C:\\models\\x.gguf", ("C:\\models\\x.gguf", None)),  # Windows drive
        ("repo:with/slash", ("repo:with/slash", None)),  # slash in variant -> not a variant
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
        ("--continue", False),  # flag
        ("resume", False),  # single word, no slash
        ("/models/local.gguf", False),  # absolute path
        ("./rel.gguf", False),  # relative path
        ("C:\\models\\x.gguf", False),  # Windows drive
        ("my models/foo", False),  # has a space
        ("owner/repo/extra", False),  # too many segments
    ],
)
def test_looks_like_model(token, expected):
    assert start._looks_like_model(token) is expected


def test_consume_positional_model_leading_token():
    # A leading org/name positional routes to --model and is dropped from the passthrough.
    model, rest = start._consume_positional_model(None, ["unsloth/Model-GGUF", "--continue"])
    assert model == "unsloth/Model-GGUF"
    assert rest == ["--continue"]


def test_looks_like_model_leaves_existing_local_dir_for_agent(tmp_path, monkeypatch):
    # A relative `owner/repo` that actually exists (e.g. an OpenCode project dir) must
    # stay an agent argument, not be consumed as a model.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "repo").mkdir(parents = True)
    assert start._looks_like_model("owner/repo") is False
    model, rest = start._consume_positional_model(None, ["owner/repo"])
    assert model is None and rest == ["owner/repo"]
    # The same shape, when it does not exist locally, is still treated as a model.
    assert start._looks_like_model("owner/absent-repo") is True


def test_consume_positional_model_ignores_non_leading_and_explicit_model():
    # An org/name that is an option value (not leading) is never stolen.
    model, rest = start._consume_positional_model(None, ["--profile", "owner/repo"])
    assert model is None and rest == ["--profile", "owner/repo"]
    # An explicit --model always wins; the positional is left untouched.
    model, rest = start._consume_positional_model("explicit/model", ["owner/repo"])
    assert model == "explicit/model" and rest == ["owner/repo"]


def test_start_positional_model_routes_to_model_on_auto_serve(fake_studio, monkeypatch):
    # `unsloth start claude unsloth/Model-GGUF` (no --model): the positional becomes the
    # model; the GGUF variant is left unset so the server's own quant preference selects it.
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
        return fake

    monkeypatch.setattr(start, "_start_studio_server", fake_start)
    monkeypatch.setattr(start, "_shutdown_server", lambda server: None)
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))

    result = CliRunner().invoke(start.start_app, ["claude", "unsloth/gemma-4-E2B-it-GGUF"])
    assert result.exit_code == 0, result.output
    assert captured["model"] == "unsloth/gemma-4-E2B-it-GGUF"
    assert captured["load"].gguf_variant is None


def test_start_local_gguf_path_keeps_no_default_variant(fake_studio, monkeypatch, tmp_path):
    # A local GGUF dir/path ending in -GGUF must NOT get a forced default quant: the dir
    # may only hold a different quant, and pre-PR the server picked whatever was available.
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
        return fake

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
    # No inherited kill switches, so the omitted-flag default applies.
    monkeypatch.delenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", raising = False)
    monkeypatch.delenv("UNSLOTH_TOOL_CALL_NUDGE", raising = False)

    # Default start: tools off (passthrough), healing + nudging on.
    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    cmd, env = captured["command"], captured["kwargs"]["env"]
    assert "--disable-tools" in cmd and "--enable-tools" not in cmd
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "0"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "1"

    # Flipped: tools on, healing off, nudging off.
    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(enable_tools = True, tool_call_healing = False, tool_call_nudging = False),
    )
    cmd, env = captured["command"], captured["kwargs"]["env"]
    assert "--enable-tools" in cmd and "--disable-tools" not in cmd
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "1"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "0"


def test_start_studio_server_respects_inherited_tool_call_env(monkeypatch):
    # With the flags omitted, an operator's pre-exported kill switch must survive into the
    # child server instead of being overwritten with the start defaults.
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

    # Flags omitted -> inherited values are preserved.
    start._start_studio_server("http://127.0.0.1:8888", "unsloth/M-GGUF", start.LoadOptions())
    env = captured["kwargs"]["env"]
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "1"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "0"

    # An explicit flag still overrides the inherited env.
    start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start.LoadOptions(),
        start.ServerOptions(tool_call_healing = True, tool_call_nudging = True),
    )
    env = captured["kwargs"]["env"]
    assert env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "0"
    assert env["UNSLOTH_TOOL_CALL_NUDGE"] == "1"


def test_connect_model_bare_id_matches_loaded_without_reload(fake_studio):
    # A bare `--model <loaded repo>` (no load knobs) attaches to the already-loaded model
    # without touching /api/inference/load, so it can never evict another session.
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--model", MODEL["id"]])
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == []
    assert f"Reusing loaded model: {MODEL['id']}\n" in result.output
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])


def test_connect_model_variant_suffix_defers_to_server_dedup(fake_studio):
    # `--model repo:QUANT` splits into a VALID load payload (bare repo + gguf_variant),
    # never the `:`-suffixed repo id Unsloth rejects. The variant knob defers to
    # /api/inference/load, whose already-loaded dedup answers without reloading when the
    # active variant+settings match -- so a second session running the same command
    # attaches without evicting the first, while a genuinely different quant reloads.
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
    # /v1/models can't reveal the active quant, so an id match alone would silently keep
    # the wrong variant loaded. Explicit knobs must always consult the load endpoint.
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--model", MODEL["id"], "--gguf-variant", "Q8_0"],
    )
    assert result.exit_code == 0, result.output
    loads = [c for c in fake_studio if c[1].endswith("/api/inference/load")]
    assert loads == [
        ("POST", f"{BASE}/api/inference/load", {"model_path": MODEL["id"], "gguf_variant": "Q8_0"})
    ]


def test_connect_model_variant_suffix_loads_split_repo(fake_studio):
    # When the model is not already loaded, the `:QUANT` suffix becomes the gguf_variant
    # and the load uses the bare (valid) repo id, mirroring `unsloth run repo --gguf-variant`.
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
    # An explicit --gguf-variant takes precedence; the suffix is still stripped so the
    # repo id stays valid.
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
    # Unsloth never surfaces the requested model; fail loudly rather than
    # silently connecting to whatever else happens to be loaded.
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
        if url.endswith("/v1/models"):
            return {"object": "list", "data": [MODEL]}  # decoy; request never appears
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--model", "unsloth/Missing-7B"]
    )
    assert result.exit_code == 1
    assert "unsloth/Missing-7B" in result.output


def test_connect_codex_rejects_non_gguf_model(fake_studio, monkeypatch):
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
    assert "GGUF" in result.output
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output


def test_connect_nonloopback_keyless_refuses_to_send_credential(fake_studio, monkeypatch):
    # A server known only by URL + health check is unverified: keyless connect
    # must refuse and make no request at all.
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://studio.evil.example:8888")
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 1
    assert "Settings → API" in result.output
    assert "--api-key" in result.output
    assert fake_studio == []  # no HTTP request of any kind (no mint, no /v1/models)


def test_connect_nonloopback_explicit_key_is_allowed(fake_studio, monkeypatch):
    # User named both server and key, so it's their choice; only auto-send is blocked.
    monkeypatch.setattr(start, "find_studio_server", lambda: "http://studio.example:8888")
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--no-launch", "--api-key", "sk-unsloth-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output


def test_connect_nonloopback_replays_saved_key(fake_studio, tmp_path, monkeypatch):
    # A key saved for a remote (non-loopback) Unsloth is replayed on keyless runs;
    # auto-minting stays blocked for non-loopback.
    remote = "http://studio.example:8888"
    monkeypatch.setattr(start, "find_studio_server", lambda: remote)
    (tmp_path / "agent_api_key.json").write_text(
        json.dumps({"servers": {remote: {"saved": ["sk-unsloth-deadbeefdeadbeef"]}}})
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)  # never minted


def test_connect_studio_server_errors_on_explicit_remote(monkeypatch):
    # A user who pointed UNSLOTH_STUDIO_URL at a remote Unsloth should get an
    # error, not a silent local model load (which they did not ask for).
    import typer

    import unsloth_cli._inference as inference

    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://studio.example:8888")
    monkeypatch.setattr(
        inference, "find_studio_server", lambda *a, **k: "http://studio.example:8888"
    )
    with pytest.raises(typer.Exit):
        inference.connect_studio_server("m", hf_token = None, max_seq_length = 4096, load_in_4bit = False)


def test_connect_studio_server_falls_back_locally_on_default_discovery(monkeypatch):
    # Opportunistic local discovery (no UNSLOTH_STUDIO_URL): if the loopback
    # server can't be verified, fall back to a local load rather than erroring.
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
    # With no saved key, the next step would auto-mint; an unverified loopback
    # server (port squatter) must be refused, with nothing sent.
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "--api-key" in result.output
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)  # never minted


def test_connect_replays_saved_key_without_identity_check(fake_studio, tmp_path, monkeypatch):
    # A "saved" key (e.g. for an SSH-tunnelled Unsloth the handshake can't match)
    # replays on keyless runs without the handshake, scoped to its own base.
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"saved": ["sk-unsloth-deadbeefdeadbeef"]}}}))
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-unsloth-deadbeefdeadbeef")
    assert not any(c[1].endswith("/api/auth/api-keys") for c in fake_studio)  # reused, not minted


def test_connect_minted_cache_requires_identity_check(fake_studio, tmp_path, monkeypatch):
    # A "minted" key is NOT replayed to an unverified loopback server: minting and
    # minted-key replay both sit behind the handshake, so a squatter can't grab it.
    cache = tmp_path / "agent_api_key.json"
    cache.write_text(json.dumps({"servers": {BASE: {"minted": ["sk-unsloth-feedfacefeedface"]}}}))
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 1
    assert "--api-key" in result.output
    assert not any(c[1].endswith("/v1/models") for c in fake_studio)  # minted key never sent


def test_connect_explicit_key_skips_identity_check(fake_studio, monkeypatch):
    # An explicit key is the user's deliberate choice, so it does not require
    # the automatic identity handshake.
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
    # Real crypto end to end: verify_studio_identity reads the install secret from
    # an isolated DB; a "good" server proves the same secret, a spoofing one can't.
    import unsloth_cli._inference as inference

    inference.ensure_studio_backend_path()
    try:
        from studio.backend.auth import storage
    except Exception as exc:  # backend not importable here (e.g. missing deps)
        pytest.skip(f"studio backend not importable: {exc}")

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)

    good = lambda nonce, host, port: storage.compute_identity_proof(
        nonce, host, port
    )  # real secret
    bad = lambda nonce, host, port: "00" * 32  # spoofer without the secret
    base_ok, stop_ok = _serve_identity(good)
    base_bad, stop_bad = _serve_identity(bad)
    try:
        assert inference.verify_studio_identity(base_ok) is True
        assert inference.verify_studio_identity(base_bad) is False
    finally:
        stop_ok()
        stop_bad()


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
    # A squatter could 302 /api/auth/identity to the real Unsloth and relay its
    # proof; redirects must be refused so the squatter's base isn't accepted.
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
        assert inference.verify_studio_identity(real_base) is True  # direct: ok
        assert inference.verify_studio_identity(squatter_base) is False  # relayed: refused
    finally:
        stop_real()
        stop_squatter()


def test_verify_studio_identity_rejects_relayed_proof(tmp_path, monkeypatch):
    # A squatter that proxies the nonce to the real Unsloth on another port gets a
    # proof bound to *that* port; the client expects one bound to the port it
    # connected to, so the relayed proof is rejected.
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
    # The squatter answers on its own port but returns the proof for the real port.
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
        ("http://127.0.0.5:9001", True),  # SSH tunnels can land anywhere in 127/8
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
def _reset_auto_served():
    # Never let a test leave a fake server in the module slot (an atexit backstop would
    # otherwise try to signal it at interpreter shutdown).
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

    server = start._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL",
        start.LoadOptions(
            gguf_variant = "UD-Q4_K_XL", max_seq_length = 8192, load_in_4bit = True, tensor_parallel = True
        ),
    )
    cmd = captured["command"]
    assert cmd[1] == "run"
    assert "--disable-tools" in cmd and "--no-cloudflare" in cmd
    assert cmd[cmd.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL"
    assert cmd[cmd.index("--gguf-variant") + 1] == "UD-Q4_K_XL"
    assert cmd[cmd.index("--context-length") + 1] == "8192"
    assert "--tensor-parallel" in cmd
    assert "--start-api-key-marker" not in cmd
    assert captured["kwargs"]["env"][start._START_API_KEY_MARKER_ENV] == "1"
    assert start.os.environ[start._START_API_KEY_MARKER_ENV] == "parent"
    assert cmd[cmd.index("-p") + 1] == "8888"
    assert start.LoadOptions().load_in_4bit is True and "--no-load-in-4bit" not in cmd
    assert captured["kwargs"].get("start_new_session") is True  # own process group
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

    server = start._start_studio_server(
        BASE,
        "owner/model-GGUF",
        start.LoadOptions(gguf_variant = "Q4_K_M"),
    )

    assert server.pid == 4321
    assert (BASE, "sk-unsloth-early", "owner/model-GGUF", "Q4_K_M", "created") in created
    assert created.count("poll") == 2
    assert created[-2:] == ["complete", "close"]
    assert not any(isinstance(event, tuple) and "server ready" in event[-1] for event in created)


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
        return fake

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
    # The `:QUANT` suffix is split off into the gguf_variant so `unsloth run` gets a valid
    # repo id plus `--gguf-variant`, mirroring how `unsloth run` accepts either form.
    assert started["model"] == "unsloth/Qwen3-1.7B-GGUF"
    assert started["load"].gguf_variant == "UD-Q4_K_XL"
    assert started["base"] == BASE
    # A successful agent exit releases ownership and leaves the server available
    # for another terminal. Explicit startup failures still use the cleanup path.
    assert "down" not in started
    assert start._auto_served_server is None
    assert "is still running" in result.output
    assert "unsloth studio stop" in result.output


def test_auto_served_agent_launch_failure_stops_server(fake_studio, monkeypatch):
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    stopped = []
    fake = SimpleNamespace(pid = 999, poll = lambda: None)

    def fake_start(*_args):
        start._auto_served_server = fake
        return fake

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
        return fake

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
        # The child prints the early key marker, then dies before it is ready.
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
    # The Codex GGUF preflight runs after _connect may have auto-started a server but
    # before _run's teardown finally, so a preflight rejection must not leave the server
    # holding the port/GPU (waiting on the atexit backstop).
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
        return fake

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
    # Torn down at the point the preflight rejected the model, not only via atexit.
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
        ("http://127.0.0.1", "http://127.0.0.1:8888"),  # portless -> unsloth run's :8888
        ("http://127.0.0.1:8888", "http://127.0.0.1:8888"),  # explicit port kept
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("http://localhost", "http://localhost:8888"),
        ("http://[::1]", "http://[::1]:8888"),  # IPv6 literal stays bracketed
        ("http://[::1]:8888", "http://[::1]:8888"),
        # Paths are stripped: unsloth run serves at the root, so /studio would make the
        # health poll hit /studio/api/health (404) until the startup timeout.
        ("http://127.0.0.1:8888/studio", "http://127.0.0.1:8888"),
        ("http://127.0.0.1/studio", "http://127.0.0.1:8888"),
    ],
)
def test_effective_base(base, expected):
    assert start._effective_base(base) == expected


def test_auto_serve_normalizes_portless_url(fake_studio, monkeypatch):
    # A portless UNSLOTH_STUDIO_URL must launch AND poll :8888 (what unsloth run binds),
    # not port 80, or readiness never matches and we hit the startup timeout.
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
        return fake

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


# ── OpenClaw (Anthropic /v1/messages) ────────────────────────────────


def test_write_openclaw_config_fresh(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["models"]["providers"]["unsloth"]
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-unsloth-abc"
    assert provider["api"] == "openai-completions"
    assert provider["models"] == [
        {"id": MODEL["id"], "name": MODEL["id"], "contextWindow": MODEL["context_length"]}
    ]
    # The default model must be pinned or OpenClaw has nothing active.
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["agents"]["defaults"]["workspace"] == str(tmp_path / "workspace")
    assert (tmp_path / "workspace").is_dir()
    assert config["gateway"]["mode"] == "local"
    assert config["gateway"]["auth"]["mode"] == "none"  # unauth loopback gateway
    if os.name != "nt":  # the file holds an API key
        assert path.stat().st_mode & 0o777 == 0o600


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
    assert config["agents"]["defaults"]["temperature"] == 0.5  # other agent defaults kept
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["models"]["mode"] == "replace"  # user's mode is left as-is
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


def test_connect_openclaw_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "openclaw" in result.output
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    # Config + state are scoped to the session dir, not the user's ~/.openclaw.
    _assert_env_set(result.output, "OPENCLAW_CONFIG_PATH", str(config_path))
    _assert_env_set(result.output, "OPENCLAW_STATE_DIR", str(tmp_path / "agents" / "openclaw"))
    config = json.loads(config_path.read_text())
    assert config["models"]["providers"]["unsloth"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["agents"]["defaults"]["model"]["primary"] == f"unsloth/{MODEL['id']}"
    assert config["agents"]["defaults"]["workspace"] == str(
        tmp_path / "agents" / "openclaw" / "workspace"
    )
    assert _launch_command(result.output) == ["openclaw", "tui", "--local"]
    # OpenAI /v1/chat/completions works on either backend — no GGUF gate.
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_connect_openclaw_wsl_windows_shim_translates_workspace(fake_studio, tmp_path, monkeypatch):
    windows_workspace = r"\\wsl.localhost\Ubuntu\tmp\openclaw\workspace"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/openclaw"
    )
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: windows_workspace)

    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])

    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config = json.loads(config_path.read_text())
    assert config["agents"]["defaults"]["workspace"] == windows_workspace
    assert (config_path.parent / "workspace").is_dir()


def test_connect_openclaw_no_launch_keeps_explicit_subcommand(fake_studio):
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", "crestodian"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "crestodian"]


def test_connect_openclaw_no_launch_passes_global_flags_through(fake_studio):
    # OpenClaw globals (openclaw [--dev] [--profile <name>] <command>) precede the
    # command, and tui does not accept them, so any passthrough args must be forwarded
    # verbatim rather than rewritten into `openclaw tui --local <globals>`.
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", "--profile", "test"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "--profile", "test"]


def test_connect_openclaw_no_launch_keeps_explicit_tui(fake_studio):
    result = CliRunner().invoke(
        start.start_app, ["openclaw", "--no-launch", "tui", "--message", "hi"]
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["openclaw", "tui", "--message", "hi"]


# ── OpenCode (OpenAI /v1/chat/completions) ───────────────────────────


def test_write_opencode_config_fresh(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": "sk-unsloth-abc"}
    # Context limit must be declared, or OpenCode treats it as 0 and disables compaction.
    assert provider["models"] == {
        MODEL["id"]: {"name": MODEL["id"], "limit": {"context": 131072, "output": 8192}}
    }
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # Provider filters belong to the launch-time inline overlay, not this config writer.
    assert "disabled_providers" not in config
    # Compaction buffer scaled to ~10% of the window (compact near 90%).
    assert config["compaction"] == {"auto": True, "reserved": 131072 // 10}


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
    # The overlay no longer edits disabled_providers; re-enabling unsloth is done in
    # the inline layer, so an existing list here is preserved untouched.
    assert config["disabled_providers"] == ["ollama", "unsloth"]
    assert config["provider"]["anthropic"]["name"] == "Anthropic"
    assert config["provider"][start._OPENCODE_PROVIDER]["options"]["baseURL"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == before


def test_write_opencode_config_keeps_foreign_disabled_providers(tmp_path):
    # A user who disabled other providers (but not unsloth) must keep them disabled:
    # the overlay must not rewrite disabled_providers, or those providers get silently
    # re-enabled for the session.
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
    inherited = {"theme": "tokyonight"}
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
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
        "enabled_providers": ["opencode-go", start._OPENCODE_PROVIDER],
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


def _opencode_inline_config(output: str) -> dict:
    # --no-launch prints OPENCODE_CONFIG_CONTENT as a POSIX `export NAME=<shell-quoted>`
    # line on Unix/WSL and a PowerShell `$env:NAME = "<escaped>"` line on native Windows;
    # parse whichever the host emitted so the opencode tests are shell-agnostic.
    name = "OPENCODE_CONFIG_CONTENT"
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(f"export {name}="):
            return json.loads(shlex.split(line.removeprefix(f"export {name}="))[0])
        prefix = f'$env:{name} = "'
        if line.startswith(prefix) and line.endswith('"'):
            escaped = line[len(prefix) : -1]
            # Reverse _print_env's PowerShell escaping (backtick is the escape char).
            value = escaped.replace("`$", "$").replace('`"', '"').replace("``", "`")
            return json.loads(value)
    raise AssertionError(f"{name} not found in:\n{output}")


def test_opencode_inline_scopes_session_to_studio_provider(fake_studio):
    # opencode filters even config-defined providers through enabled/disabled_providers,
    # and a model pin does not bypass that gate. The inline overlay (session-only, highest
    # layer, arrays replace) allowlists our provider and clears the denylist so the Unsloth
    # model always loads regardless of the user's config, without reading or editing it.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["enabled_providers"] == [start._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == []
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # small_model stays on the enabled provider too, so lightweight tasks do not resolve a
    # filtered provider mid-session.
    assert inline["small_model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"


def test_opencode_passthrough_flags_omit_model_flag(fake_studio):
    # Any passthrough (top-level flags that may precede a subcommand, or a subcommand)
    # is left untouched; --model is not injected. The model is pinned by the inline
    # OPENCODE_CONFIG_CONTENT (highest layer) instead, so it is still forced.
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
    # A passthrough subcommand (e.g. `serve`) takes the model from the pinned config;
    # inserting --model before it would break opencode's arg parsing.
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
    # OPENCODE_CONFIG overlay points at the session file, not the user's global config.
    _assert_env_set(result.output, "OPENCODE_CONFIG", str(config_path))
    inline_config = _opencode_inline_config(result.output)
    config = json.loads(config_path.read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["options"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # The session config file (a throwaway overlay, not the user's real config) does not
    # carry provider filters; the session scoping rides in the inline env layer only.
    assert "disabled_providers" not in config
    assert "enabled_providers" not in config
    assert inline_config == {
        "model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "small_model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "enabled_providers": [start._OPENCODE_PROVIDER],
        "disabled_providers": [],
    }
    # --no-launch prints an append-safe base command (no --model before a subcommand a
    # driver may append); the model is forced by the inline pin above.
    assert _launch_command(result.output) == ["opencode"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


def test_connect_opencode_as_subagent_preserves_cloud_parent(fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_opencode_subagent_inline_config", lambda path, permission: {})
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
    # The agent rides in the inline overlay; nothing else comes from the empty base.
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


def test_claude_subagent_plugin_dir_precedes_forwarded_delimiter(fake_studio):
    # A forwarded `--` makes everything after it positional; the plugin flag must be
    # parsed as an option, so it rides before ctx.args.
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "--", "--resume", "abc123"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command.index("--plugin-dir") < command.index("--resume")


def test_opencode_subagent_installs_binary_before_filter_inspection(fake_studio, monkeypatch):
    # The effective-config inspection needs the opencode binary; a first launch must
    # offer the install before building the overlay, or a global allowlist read only
    # after _launch installs OpenCode would filter out the new provider.
    installed = {}
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode" if installed.get("done") else None,
    )

    def install(name, hint):
        installed["done"] = True
        installed["name"] = name
        return "/usr/local/bin/opencode"

    monkeypatch.setattr(start, "_install_agent", install)
    inspected = {}

    def inline(path, permission):
        inspected["binary"] = start._which_with_install_dirs("opencode")
        return {}

    monkeypatch.setattr(start, "_opencode_subagent_inline_config", inline)
    monkeypatch.setattr(start, "_run", lambda *a, **k: None)

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 0, result.output
    assert installed["name"] == "opencode"
    assert inspected["binary"] == "/usr/local/bin/opencode"


def test_opencode_subagent_pins_agent_in_inline_overlay(fake_studio, monkeypatch):
    # A project opencode.json outranks the session file, so the agent must ride in
    # OPENCODE_CONFIG_CONTENT where a repo's own agent.unsloth cannot field-merge over it.
    monkeypatch.setattr(start, "_opencode_subagent_inline_config", lambda path, permission: {})
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
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda: True)
    captured = {}

    def inline(path, permission):
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


# ── Hermes (OpenAI /v1/chat/completions, key via env) ────────────────


@pytest.fixture()
def hermes_config(tmp_path):
    return tmp_path / "config.yaml"


def test_write_hermes_config_fresh(hermes_config):
    yaml = pytest.importorskip("yaml")
    start.write_hermes_config(BASE, MODEL, hermes_config)
    config = yaml.safe_load(hermes_config.read_text())
    # Hermes only honors the key for a *named* custom provider, so the endpoint
    # is registered under providers.* and model.provider points at it.
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["model"]["default"] == MODEL["id"]
    assert config["model"]["api_mode"] == "openai"
    # Pin the real context window (top-level override) and compact at 90% of it.
    assert config["model"]["context_length"] == MODEL["context_length"]
    assert config["compression"] == {"enabled": True, "threshold": 0.9}
    # Windows at or above Hermes' floor need no auxiliary compression override.
    assert "auxiliary" not in config
    provider = config["providers"]["unsloth"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["api_mode"] == "openai"
    assert provider["key_env"] == "UNSLOTH_API_KEY"
    # The key is resolved from the launch env, never written to disk.
    assert "sk-unsloth" not in hermes_config.read_text()


def test_write_hermes_config_small_window_claims_floor(hermes_config):
    yaml = pytest.importorskip("yaml")
    small = {"id": "unsloth/Qwen3-1.7B-GGUF", "context_length": 40960}
    start.write_hermes_config(BASE, small, hermes_config)
    config = yaml.safe_load(hermes_config.read_text())
    # Hermes refuses to initialize below its 64,000-token floor, so the recipe
    # claims the floor and scales the compaction threshold so it still fires at
    # 90% of the REAL window: 0.9 * 40960 / 65536.
    assert config["model"]["context_length"] == 65536
    assert config["compression"] == {"enabled": True, "threshold": 0.5625}
    # The same floor check runs against the compression model mid-session.
    assert config["auxiliary"]["compression"]["context_length"] == 65536


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
    assert config["terminal"] == {"backend": "local"}  # unrelated sections kept
    assert config["model"]["temperature"] == 0.7  # unrelated model keys kept
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    before = hermes_config.read_text()
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert hermes_config.read_text() == before


def test_write_hermes_config_preserves_non_mapping_file(hermes_config, capsys):
    pytest.importorskip("yaml")
    original = "- just\n- a\n- list\n"  # valid YAML, but not a mapping
    hermes_config.write_text(original)
    start.write_hermes_config(BASE, MODEL, hermes_config)
    assert hermes_config.read_text() == original  # user-managed file left untouched
    assert "couldn't parse" in capsys.readouterr().err


def test_connect_hermes_no_launch(fake_studio, tmp_path):
    yaml = pytest.importorskip("yaml")
    result = CliRunner().invoke(start.start_app, ["hermes", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "UNSLOTH_API_KEY", "sk-unsloth-feedfacefeedface")
    # HERMES_HOME relocates the whole hermes home, so the user's ~/.hermes is untouched.
    home = tmp_path / "agents" / "hermes"
    _assert_env_set(result.output, "HERMES_HOME", str(home))
    assert "hermes" in result.output
    config = yaml.safe_load((home / "config.yaml").read_text())
    assert config["model"]["provider"] == "custom:unsloth"
    assert config["providers"]["unsloth"]["base_url"] == f"{BASE}/v1"
    assert config["model"]["default"] == MODEL["id"]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


# ── Pi (OpenAI-compatible /v1, key in config, ~/.pi relocated via HOME) ──


def test_write_pi_config_fresh(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["providers"]["unsloth"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-unsloth-abc"
    # Pin the loaded window (and a sane output cap) so Pi compacts instead of
    # overflowing; without it Pi assumes its 128000 default.
    assert provider["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 8192}
    ]


def test_write_pi_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"providers": {"google": {"api": "gemini"}}}))
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["providers"]["google"] == {"api": "gemini"}  # unrelated provider kept
    assert config["providers"]["unsloth"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    assert path.read_text() == before


def test_connect_pi_no_launch(fake_studio, tmp_path):
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    # Pi resolves its config dir from PI_CODING_AGENT_DIR first, so pin it at the session
    # dir (and relocate HOME) to keep the user's real ~/.pi untouched and their own
    # PI_CODING_AGENT_DIR from redirecting Pi away from our provider/key.
    home = tmp_path / "agents" / "pi"
    _assert_env_set(result.output, "HOME", str(home))
    _assert_env_set(result.output, "PI_CODING_AGENT_DIR", str(home / ".pi" / "agent"))
    # Provider/model pinned on the command (Pi defaults to google otherwise).
    assert f"pi --provider unsloth --model {MODEL['id']}" in result.output
    config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
    assert config["providers"]["unsloth"]["apiKey"] == "sk-unsloth-feedfacefeedface"
    assert config["providers"]["unsloth"]["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 8192}
    ]
    assert not any(c[1].endswith("/api/inference/status") for c in fake_studio)


def test_connect_pi_as_subagent_preserves_cloud_parent(fake_studio, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        [
            "pi",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"] + ":UD-Q4_K_XL",
        ],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:2] == ["pi", "--extension"]
    assert command[2].endswith("unsloth_cli/pi_subagent.ts")
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
    }
    assert "Ask Pi to spawn an Unsloth or local agent." in result.output


def test_connect_pi_no_launch_windows_relocates_userprofile(fake_studio, tmp_path, monkeypatch):
    # On native Windows Node resolves ~/.pi via USERPROFILE, not HOME, so the session
    # must point USERPROFILE at the relocated home or Pi reads the user's real ~/.pi.
    monkeypatch.setattr(start.os, "name", "nt")
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "pi"
    assert f'$env:HOME = "{home}"' in result.output
    assert f'$env:USERPROFILE = "{home}"' in result.output


# ── WSLENV path translation + PowerShell quoting (helper units) ──


def test_wsl_bridge_names_flags_paths_not_scalars():
    # WSLENV only translates a var to a Windows path when its entry carries /p.
    # Path-valued vars must get it; scalar knobs and URLs must not, or WSLENV would
    # mangle them when handing off to a Windows shim under /mnt.
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
    assert "USERPROFILE/p" in names  # drive-qualified Windows path
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" in names  # scalar: no /p
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW/p" not in names
    assert "ANTHROPIC_BASE_URL" in names  # URL is not a filesystem path
    assert "ANTHROPIC_API_KEY" in names  # cleared var carries no value to translate


def test_merge_wslenv_dedups_on_base_name():
    # An already-shared var must not be appended again just because the flag differs.
    merged = start._merge_wslenv("CODEX_HOME/p:FOO", ("CODEX_HOME/p", "BAR/p"))
    parts = merged.split(":")
    assert parts.count("CODEX_HOME/p") == 1
    assert "FOO" in parts and "BAR/p" in parts


def test_merge_wslenv_upgrades_existing_unflagged_entry():
    # A user's pre-existing bare "HOME" must be upgraded to "HOME/p" (not left bare or
    # duplicated), or the Windows shim gets the path without WSL translation.
    merged = start._merge_wslenv("HOME:FOO", ("HOME/p", "CODEX_HOME/p"))
    parts = merged.split(":")
    assert "HOME/p" in parts and "HOME" not in parts  # upgraded in place
    assert parts.count("HOME/p") == 1
    assert "FOO" in parts  # untouched user var preserved
    assert "CODEX_HOME/p" in parts


def test_powershell_quote_single_quotes_json():
    # Bare flags/paths pass through; JSON payloads get single-quoted so PowerShell
    # keeps the embedded double quotes literal (list2cmdline's backslashes would not).
    assert start._powershell_quote("--settings") == "--settings"
    assert start._powershell_quote("unsloth/gemma-4-26B") == "unsloth/gemma-4-26B"
    overlay = start._claude_settings_overlay("unsloth/gemma-4-26B")
    quoted = start._powershell_quote(overlay)
    assert quoted == "'" + overlay + "'"
    assert "\\" not in quoted  # no cmd.exe backslash escaping
    assert start._powershell_quote("a'b") == "'a''b'"  # embedded quote doubled


# ── --yolo: one switch routed to each agent's own auto-approve form ──

# The native "run tools without prompting" CLI flag each agent should receive.
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
    # pi's --approve is a real flag only added under --yolo; assert it's absent here.
    command = _launch_command(result.output)
    assert command and command[0] == agent, result.output
    assert native not in command


@pytest.mark.parametrize(
    "alias",
    ["--yolo", "--dangerously-skip-permissions", "--dangerously-bypass-approvals-and-sandbox"],
)
def test_yolo_aliases_are_interchangeable(fake_studio, alias):
    # Any spelling on any agent routes to that agent's own flag, even the "wrong" one.
    claude = CliRunner().invoke(start.start_app, ["claude", alias, "--no-launch"])
    assert claude.exit_code == 0, claude.output
    assert "--dangerously-skip-permissions" in claude.output
    # The codex spelling must not leak through to Claude's command line.
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
    # A bare --no-launch recipe stays append-safe (callers add a subcommand later);
    # `opencode --auto run ...` would select the TUI, not `run`, so keep the config fallback.
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
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda: True)
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])
    assert captured["command"][1:] == [
        "--model",
        f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "--auto",
    ]
    assert "permission" not in json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])


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
        # Hidden commands that reject --auto fall back like the visible utility ones.
        (["generate"], ["generate"], False),
        (["console", "login"], ["console", "login"], False),
        # --mini ignores --auto (runMini forces auto=false), so use the config fallback.
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
    # generate/console are hidden and reject --auto, --mini ignores it: none get --auto,
    # all keep the config permission fallback.
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
    # A non-yolo run on a fresh config writes no permission block; it only flips a prior
    # --yolo run's explicit allow back to ask (see the yolo-then-plain test below).
    assert "permission" not in config


def test_no_yolo_opencode_flips_prior_yolo_allow_to_ask(fake_studio, tmp_path):
    # The core reset: a --yolo run wrote explicit per-tool allow; a later non-yolo run
    # must flip exactly those back to ask so nothing stays auto-approved.
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
    # Both layers: the host approvals file in OPENCLAW_STATE_DIR must also be set, or
    # OpenClaw can still prompt/deny despite the config.
    approvals = json.loads((state / "exec-approvals.json").read_text())
    assert approvals["defaults"] == {"security": "full", "ask": "off", "askFallback": "full"}


def test_no_yolo_openclaw_leaves_fresh_config_untouched(fake_studio, tmp_path):
    # A fresh non-yolo run only undoes state a prior --yolo wrote; with no yolo
    # fingerprint present it must not synthesize an exec policy. An omitted policy can
    # resolve to a sandbox default of security=deny, so writing allowlist here would
    # BROADEN it. The reset is scoped to the exact yolo write, verified by the
    # yolo-then-plain round trip below.
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
    # The no-launch config dir is reused across runs, so a --yolo run persists its
    # auto-approve settings; a later run without --yolo must strip them, not leave
    # tool execution silently pre-approved.
    yolo = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert "permission" in json.loads(config_path.read_text())
    plain = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    config = json.loads(config_path.read_text())
    # The yolo allow policy is replaced by a prompting one, not deleted (which would
    # revert to OpenCode's permissive "allow" default).
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }
    # The session provider survives the cleanup.
    assert start._OPENCODE_PROVIDER in config["provider"]


def test_no_launch_rerun_clears_stale_openclaw_yolo_state(fake_studio, tmp_path):
    yolo = CliRunner().invoke(start.start_app, ["openclaw", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    state = tmp_path / "agents" / "openclaw"
    assert (state / "exec-approvals.json").exists()
    plain = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    config = json.loads((state / "openclaw.json").read_text())
    # The yolo policy is replaced by a prompting one, not deleted (which would revert
    # to OpenClaw's permissive default), and the yolo approvals file is gone.
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}
    assert not (state / "exec-approvals.json").exists()
    # The session provider survives the cleanup.
    assert "unsloth" in config["models"]["providers"]


def test_write_openclaw_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    # A plain rerun replaces the yolo policy with a prompting one (deleting it would
    # fall back to OpenClaw's permissive default) and removes the yolo approvals file.
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}
    assert not (path.parent / "exec-approvals.json").exists()


def test_write_opencode_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    # A plain rerun replaces the yolo allow policy with a prompting one.
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


def test_openclaw_non_yolo_keeps_runtime_approvals(tmp_path):
    # OpenClaw records its own entries in exec-approvals.json (OPENCLAW_STATE_DIR is
    # this dir); the non-yolo reset drops only the yolo defaults, not those.
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
    # A mixed user-managed defaults block that only shares a field with the yolo payload
    # (here askFallback=full, whose omitted default is deny) is not stale yolo state, so a
    # non-yolo run leaves it intact rather than stripping the shared field.
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
    # A policy that lacks the full yolo fingerprint (here no host and no security) is not
    # our --yolo write, so a non-yolo run leaves it as-is rather than assuming ask=off
    # means permissive: an omitted host/security can resolve to a sandbox deny default.
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"timeout": 30, "ask": "off"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"timeout": 30, "ask": "off"}


def test_openclaw_non_yolo_leaves_no_permissive_values(tmp_path):
    # The whole point of the reset: after a yolo run, a plain run must leave neither the
    # config nor the approvals file at OpenClaw's permissive (security=full, ask=off)
    # default, or exec still auto-approves.
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = True)
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    exec_policy = json.loads(path.read_text())["tools"]["exec"]
    assert exec_policy.get("security") != "full"
    assert exec_policy.get("ask") != "off"
    assert not (path.parent / "exec-approvals.json").exists()


def test_openclaw_non_yolo_preserves_stricter_exec_policy(tmp_path):
    # A policy that doesn't carry the yolo values (for example stricter security or
    # prompting turned on) was not written by --yolo and must survive a plain run.
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"security": "deny", "ask": "on"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"security": "deny", "ask": "on"}


def test_openclaw_non_yolo_preserves_stricter_approval_defaults(tmp_path):
    # exec-approvals.json defaults that don't match the yolo payload (stricter
    # settings from the user or the OpenClaw UI) are kept, and the file stays.
    path = tmp_path / "openclaw.json"
    approvals = path.parent / "exec-approvals.json"
    approvals.write_text(
        json.dumps({"version": 1, "defaults": {"security": "allowlist", "ask": "on"}})
    )
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    state = json.loads(approvals.read_text())
    assert state["defaults"] == {"security": "allowlist", "ask": "on"}


def test_openclaw_non_yolo_leaves_unparseable_approvals(tmp_path):
    # An unreadable approvals file is left in place rather than deleted, matching
    # how an unparseable config is handled.
    path = tmp_path / "openclaw.json"
    approvals = path.parent / "exec-approvals.json"
    approvals.write_text("{not json")
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert approvals.read_text() == "{not json"


def test_opencode_non_yolo_flips_only_explicit_allow(tmp_path):
    # Only a tool explicitly set to "allow" (what --yolo writes) is flipped to "ask". A
    # deny/ask a user set is kept, and an absent tool is not added.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"edit": "allow", "bash": "deny", "read": "ask"}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["permission"] == {"edit": "ask", "bash": "deny", "read": "ask"}
    assert session == {}  # a non-yolo session carries no permission inline


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
    # A global string rule ("deny") is a user-managed catch-all; leave it untouched and
    # carry no inline override.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": "deny"}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == "deny"
    assert session == {}


def test_opencode_non_yolo_leaves_catch_all_and_flips_explicit_allow(tmp_path):
    # A "*" catch-all is the user's own rule, never something --yolo writes (yolo sets
    # explicit per-tool allow), so it is left intact; an explicit per-tool "allow" is still
    # flipped to "ask", but an absent tool inheriting the catch-all is not touched.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"*": "allow", "bash": "allow"}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == {"*": "allow", "bash": "ask"}
    assert session == {}


def test_opencode_non_yolo_leaves_granular_object(tmp_path):
    # A granular object value is a user rule (yolo only ever writes a plain "allow" string),
    # so it is left in the file verbatim and never carried inline.
    path = tmp_path / "opencode.json"
    obj = {"read *": "deny", "git *": "ask"}
    path.write_text(json.dumps({"permission": {"bash": dict(obj)}}))
    session = start.write_opencode_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"]["bash"] == obj
    assert session == {}


def test_openclaw_non_yolo_leaves_mode_policy(tmp_path):
    # tools.exec.mode is OpenClaw's normalized knob and cannot be combined with explicit
    # security/ask (the config is rejected), so a mode-based policy must be left as-is.
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"mode": "deny"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"mode": "deny"}


def test_openclaw_non_yolo_preserves_sandbox_host(tmp_path):
    # host=sandbox defaults to security=deny (stricter than the gateway "full" default),
    # so a non-yolo run must not treat the missing security as permissive nor pop host
    # (which would broaden routing to the gateway).
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"host": "sandbox"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"host": "sandbox"}


def test_openclaw_non_yolo_preserves_node_host(tmp_path):
    # host=node routes to a paired node and is only ever set by the user (--yolo writes
    # host=gateway), so a non-yolo run must not pop it and reroute to the gateway.
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"host": "node"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"host": "node"}


def test_openclaw_non_yolo_preserves_auto_host_permissive(tmp_path):
    # host=auto (or omitted) with security=full/ask=off is NOT the --yolo write: under an
    # active sandbox, auto resolves to security=deny. --yolo only ever writes host=gateway,
    # so the reset must not treat auto/None as the permissive gateway default and broaden a
    # sandboxed deny to allowlist.
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
    # The reset fires on exactly the host=gateway + security=full + ask=off write --yolo
    # makes, and nothing else.
    path = tmp_path / "openclaw.json"
    path.write_text(
        json.dumps({"tools": {"exec": {"host": "gateway", "security": "full", "ask": "off"}}})
    )
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"security": "allowlist", "ask": "on-miss"}


def test_openclaw_non_yolo_preserves_full_mode(tmp_path):
    # OpenClaw never normalizes our security=full/ask=off yolo write into mode:"full"
    # (verified against the binary: doctor --fix and config get leave security/ask as-is),
    # so a mode:"full" is always a deliberate user policy, not stale yolo state; leave it.
    path = tmp_path / "openclaw.json"
    path.write_text(json.dumps({"tools": {"exec": {"mode": "full"}}}))
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["tools"]["exec"] == {"mode": "full"}


def test_yolo_command_flags_unmapped_agent_is_empty():
    # Placement-aware/config-based agents (and any typo) must yield no prefix flag.
    assert start._yolo_command_flags("opencode", True) == []
    assert start._yolo_command_flags("openclaw", True) == []
    assert start._yolo_command_flags("claude", True) == ["--dangerously-skip-permissions"]
    assert start._yolo_command_flags("claude", False) == []


def test_yolo_config_fallbacks_add_no_legacy_command_flag(fake_studio):
    # OpenClaw is config-only; OpenCode's append-safe bare recipe uses its config fallback.
    # Neither should leak a legacy yolo/dangerous alias onto argv.
    for agent in ("opencode", "openclaw"):
        result = CliRunner().invoke(start.start_app, [agent, "--yolo", "--no-launch"])
        assert result.exit_code == 0, result.output
        command = _launch_command(result.output)
        assert command and command[0] == agent, result.output
        assert not any("--yolo" in arg or "--dangerous" in arg for arg in command)


def test_pi_launch_clears_screen_first(fake_studio, monkeypatch):
    # Pi paints inline from the current cursor position (no alternate screen, no
    # clear on its first render), so the launcher hands it a clean screen. The
    # clear must come BEFORE the exec, and only on the launch path.
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
    # The --no-launch recipe is meant to be read (and piped); never wipe it.
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert calls == []


def test_claude_launch_does_not_clear(fake_studio, monkeypatch):
    # Alternate-screen agents manage the terminal themselves; leave it alone.
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
    # A Windows pi shim resolves ~/.pi via USERPROFILE, so it must match the session
    # HOME and ride the WSLENV bridge (with /p) so the path is translated for Windows.
    assert captured["env"]["USERPROFILE"] == home
    wslenv = captured["env"]["WSLENV"].split(":")
    assert "HOME/p" in wslenv
    assert "USERPROFILE/p" in wslenv


def test_agent_api_key_auto_started_rejected_env_key_falls_back(fake_studio, tmp_path, monkeypatch):
    # UNSLOTH_API_KEY exported for some OTHER server must not fail the launch
    # against a server this run just auto-started: validate, then fall back to
    # the local mint path, and never remember the foreign key for this base.
    inner = start._http_json

    def http_json(
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/v1/models") and token == "sk-unsloth-other-server":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)
        return inner(method, url, token, payload, timeout, error)

    monkeypatch.setattr(start, "_http_json", http_json)
    key = start._agent_api_key(BASE, "sk-unsloth-other-server", auto_started = True)
    assert key == "sk-unsloth-feedfacefeedface"  # minted for the fresh server
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert "sk-unsloth-other-server" not in json.dumps(cached["servers"].get(BASE, {}))


def test_agent_api_key_auto_started_accepted_key_is_honored(fake_studio, tmp_path):
    # An explicit key the fresh server accepts (e.g. persisted in this Unsloth
    # home's auth db across restarts) keeps working exactly as before.
    key = start._agent_api_key(BASE, "sk-unsloth-deadbeefdeadbeef", auto_started = True)
    assert key == "sk-unsloth-deadbeefdeadbeef"
    cached = json.loads((tmp_path / "agent_api_key.json").read_text())
    assert cached["servers"][BASE]["saved"] == ["sk-unsloth-deadbeefdeadbeef"]


def test_session_config_no_launch_preserves_existing_state(fake_studio, tmp_path):
    # A previously printed recipe may still be running an agent whose sessions
    # or sqlite state live in the stable home; a re-run must not wipe it.
    with start._session_config("codex", launch = False) as home:
        marker = home / "sessions" / "live.sqlite"
        marker.parent.mkdir(parents = True)
        marker.write_text("state")
    with start._session_config("codex", launch = False) as home2:
        assert home2 == home
        assert (home2 / "sessions" / "live.sqlite").read_text() == "state"


# ── --persist: persist the agent session so it can be resumed ────────────────
def test_session_config_persist_uses_stable_dir_and_survives(monkeypatch, tmp_path):
    # --persist routes a launch to the stable Unsloth agents dir (the one --no-launch
    # already uses) instead of a throwaway temp dir, and never wipes it on exit.
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    with start._session_config("codex", launch = True, persist = True) as home:
        assert home == tmp_path / "agents" / "codex"
        (home / "marker").write_text("kept")
    assert home.exists()
    assert (home / "marker").read_text() == "kept"


def test_session_config_default_launch_is_ephemeral():
    # Default launch (no --persist) still uses a throwaway temp dir wiped on exit.
    with start._session_config("codex", launch = True) as home:
        assert home.exists()
        assert "unsloth-codex-" in home.name
    assert not home.exists()


# The temp-dir agents: --persist points each one's home/state env at the stable dir;
# without it, at an ephemeral temp path. opencode is handled separately (only its
# config overlay is relocated; its session data was never in the temp dir).
_RESUME_ENV_VAR = {
    "codex": "CODEX_HOME",
    "openclaw": "OPENCLAW_STATE_DIR",
    "hermes": "HERMES_HOME",
    "pi": "HOME",
}


def _capture_launch(monkeypatch, argv):
    captured = {}

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
    captured = _capture_launch(monkeypatch, [agent, "--persist"])
    stable = tmp_path / "agents" / agent
    assert captured["env"][_RESUME_ENV_VAR[agent]] == str(stable)
    # The stable dir survives the agent exit, so the session can be resumed.
    assert stable.exists()


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_default_launch_home_is_ephemeral(agent, fake_studio, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    captured = _capture_launch(monkeypatch, [agent])
    home = captured["env"][_RESUME_ENV_VAR[agent]]
    assert f"unsloth-{agent}-" in home
    assert str(tmp_path / "agents") not in home


def test_resume_opencode_config_in_stable_dir(fake_studio, tmp_path, monkeypatch):
    # opencode's session data lives in ~/.local/share/opencode (never relocated), so
    # resume already survives exit; --persist also stabilizes its config overlay dir.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    stable = tmp_path / "agents" / "opencode"
    assert captured["env"]["OPENCODE_CONFIG"] == str(stable / "opencode.json")
    assert stable.exists()


def test_persist_bare_codex_launch_has_no_resume_token(fake_studio, monkeypatch):
    # A bare `--persist` only persists the session dir; it must NOT auto-append a native
    # resume token, or the very first launch (no session yet) would send codex down its
    # no-session error path. The user resumes explicitly: `unsloth start codex --persist resume`.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist"])
    assert "resume" not in captured["command"]
    # command[0] is the resolved executable path; assert the argv after it.
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
    # When the caller drives their own subcommand, --persist only persists the dir; it
    # must not inject a resume token that would collide with the user's command.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist", "exec", "hello"])
    assert "resume" not in captured["command"]
    assert captured["command"][-2:] == ["exec", "hello"]


def test_default_launch_has_no_resume_token(fake_studio, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex"])
    assert "resume" not in captured["command"]


def test_resume_persist_only_agents_have_no_resume_token(fake_studio, monkeypatch):
    # Persistence alone must not select a session.
    for agent in ("openclaw", "hermes"):
        monkeypatch.setattr(start.shutil, "which", lambda _, a = agent: f"/usr/local/bin/{a}")
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
    # The persistence flag is --persist, NOT --resume, so an agent's own
    # `--resume <id>` (e.g. `unsloth start claude --resume <guid>`) still flows
    # through to the agent verbatim and is not swallowed as an Unsloth option.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    captured = _capture_launch(monkeypatch, ["claude", "--resume", "some-session-guid"])
    assert captured["command"][-2:] == ["--resume", "some-session-guid"]
    # Unsloth never auto-appends its own resume token when the user drives resume.
    assert captured["command"].count("--resume") == 1
    assert "--continue" not in captured["command"]
