# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Persistent PATH writes must not demote an active conda env, or later conda loads the wrong DLLs."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
INSTALL_SH = REPO_ROOT / "install.sh"
SETUP_SH_POSIX = REPO_ROOT / "studio" / "setup.sh"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]

# install.ps1 nests helpers, setup.ps1 does not, so indentation is part of the pattern.
_PS_FILES = {
    "install.ps1": (INSTALL_PS1, "    "),
    "studio/setup.ps1": (SETUP_PS1, ""),
}


def _function(path: Path, indent: str, name: str) -> str:
    source = path.read_text(encoding = "utf-8")
    match = re.search(rf"{indent}function {name} \{{.*?\n{indent}\}}\n", source, flags = re.DOTALL)
    assert match is not None, f"{path.name} no longer defines {name}"
    return match.group(0)


def _run(
    shell: str,
    script: str,
    env: dict[str, str] | None = None,
) -> str:
    base = {k: v for k, v in os.environ.items() if not k.startswith("CONDA_")}
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "strict",
        env = {**base, **(env or {})},
        timeout = 60,
    )
    return result.stdout.strip()


_NO_CONDA = {}
_ACTIVE_CONDA = {"CONDA_PREFIX": "E:\\anaconda\\install", "CONDA_DEFAULT_ENV": "base"}
_PREFIX_ONLY = {"CONDA_PREFIX": "E:\\anaconda\\install"}
# Exporting only CONDA_DEFAULT_ENV still keeps conda's ordering, so it counts too.
_NAME_ONLY = {"CONDA_DEFAULT_ENV": "base"}
# Set but empty is not an activation: conda deactivate empties these.
_EMPTY = {"CONDA_PREFIX": "", "CONDA_DEFAULT_ENV": ""}


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
@pytest.mark.parametrize(
    "requested,env,expected",
    [
        ("Prepend", _NO_CONDA, "Prepend"),
        ("Prepend", _ACTIVE_CONDA, "Append"),
        ("Prepend", _PREFIX_ONLY, "Append"),
        ("Prepend", _NAME_ONLY, "Append"),
        ("Prepend", _EMPTY, "Prepend"),
        ("Append", _ACTIVE_CONDA, "Append"),
        ("Append", _NO_CONDA, "Append"),
    ],
    ids = [
        "prepend-no-conda",
        "prepend-active-conda",
        "prepend-prefix-only",
        "prepend-name-only",
        "prepend-empty-vars",
        "append-active-conda",
        "append-no-conda",
    ],
)
def test_a_prepend_becomes_an_append_inside_an_active_conda_environment(
    shell: str, path_label: str, requested: str, env: dict[str, str], expected: str
):
    path, indent = _PS_FILES[path_label]
    script = (
        _function(path, indent, "Test-ActiveCondaEnvironment")
        + _function(path, indent, "Resolve-UserPathPosition")
        + f"\nWrite-Host ('POSITION=' + (Resolve-UserPathPosition -Position '{requested}'))\n"
    )
    out = _run(shell, script, env = env)
    assert f"POSITION={expected}" in out


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_every_persistent_path_write_routes_through_the_guard(path_label: str):
    """Add-ToUserPath is the only persistent PATH writer, so every new call site inherits the guard."""
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Add-ToUserPath")
    assert (
        "Resolve-UserPathPosition" in body
    ), f"{path_label}: Add-ToUserPath must resolve its position through the guard"
    # Ahead of the registry read, or the decision lands after the value is used.
    assert body.index("Resolve-UserPathPosition") < body.index(
        "Registry"
    ), f"{path_label}: the guard has to run before the registry write is prepared"


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_a_downgraded_prepend_still_repositions_an_existing_front_entry(path_label: str):
    """An already-present early return must still reposition a front PATH entry, not leave it there."""
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Add-ToUserPath")
    match = re.search(r"\$alreadyPresent -and \$Position -eq 'Append'([^\)]*)\)", body)
    assert match is not None, f"{path_label}: no append early-return found in Add-ToUserPath"
    assert "-not $positionDowngraded" in match.group(1), (
        f"{path_label}: the append early-return must not fire for a prepend that conda "
        f"downgraded, or an existing front entry is never moved to the back"
    )
    assert re.search(
        r"\$positionDowngraded\s*=\s*\(", body
    ), f"{path_label}: $positionDowngraded must be computed from Resolve-UserPathPosition"


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_the_two_installers_agree_on_the_guard(path_label: str):
    """Both copies decide the same way. They are separate scripts, not a shared module."""
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Resolve-UserPathPosition")
    assert "Test-ActiveCondaEnvironment" in body
    assert "'Append'" in body
    conda = _function(path, indent, "Test-ActiveCondaEnvironment")
    assert "CONDA_PREFIX" in conda
    assert "CONDA_DEFAULT_ENV" in conda


@pytest.mark.parametrize(
    "path,funcs",
    [
        (INSTALL_SH, ("_persist_login_path_dir", "_persist_fish_path_dir")),
        (SETUP_SH_POSIX, ("_setup_persist_uv_path",)),
    ],
    ids = ["install.sh", "studio/setup.sh"],
)
def test_the_posix_installers_carry_the_same_guard(path: Path, funcs: tuple[str, ...]):
    """Only checks that the POSIX installers still carry the conda PATH guard; behaviour is in tests/sh/."""
    source = path.read_text(encoding = "utf-8")
    for func in funcs:
        body = re.search(rf"\n{func}\(\) \{{.*?\n\}}\n", source, flags = re.DOTALL)
        assert body is not None, f"{path.name} no longer defines {func}"
        assert "_unsloth_conda_env_active" in body.group(0), f"{func} must not prepend inside conda"


@pytest.mark.parametrize(
    "path", [INSTALL_SH, SETUP_SH_POSIX], ids = ["install.sh", "studio/setup.sh"]
)
def test_both_halves_call_the_same_environment_an_active_conda(path: Path):
    """The PowerShell and POSIX halves must agree on WHICH variables mean "inside conda"."""
    posix = path.read_text(encoding = "utf-8")
    body = re.search(r"\n_unsloth_conda_env_active\(\) \{.*?\n\}\n", posix, flags = re.DOTALL)
    assert body is not None, f"{path.name} no longer defines _unsloth_conda_env_active"
    posix_vars = set(re.findall(r"\bCONDA_[A-Z_]+", body.group(0)))

    windows = INSTALL_PS1.read_text(encoding = "utf-8")
    probe = re.search(
        r"function Test-ActiveCondaEnvironment \{.*?\n    \}", windows, flags = re.DOTALL
    )
    assert probe is not None, "install.ps1 no longer defines Test-ActiveCondaEnvironment"
    windows_vars = set(re.findall(r"\bCONDA_[A-Z_]+", probe.group(0)))

    assert posix_vars == windows_vars, (
        f"{path.name} reads {sorted(posix_vars)} but install.ps1 reads " f"{sorted(windows_vars)}"
    )


# `irm | iex` runs in the caller's process, so the live session PATH must keep conda first.


def _refresh_preamble(current_path: str) -> str:
    """Refresh-SessionPath with the two registry reads stubbed and $env:Path seeded."""
    body = (
        _function(INSTALL_PS1, "    ", "Test-ActiveCondaEnvironment")
        + _function(INSTALL_PS1, "    ", "Get-ActiveCondaPrefixes")
        + _function(INSTALL_PS1, "    ", "Test-PathUnderCondaPrefix")
        + _function(INSTALL_PS1, "    ", "Refresh-SessionPath")
    )
    return f"""
$ErrorActionPreference = "Stop"
{body}
# The suite itself runs inside a virtualenv, and Refresh-SessionPath puts $VIRTUAL_ENV
# first. That branch is not what these cases are about, and leaving it on would make the
# expected orderings depend on where the suite was launched from.
$env:VIRTUAL_ENV = ""
$env:Path = "{current_path}"
Refresh-SessionPath
Write-Host ("PATH=" + $env:Path)
"""


MACHINE = "C:\\Windows\\system32;C:\\Windows"
USER_PATH = "C:\\Users\\me\\.local\\bin;C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python312"
CONDA_ROOT = "E:\\anaconda\\install"
_CONDA_ENTRIES = (
    f"{CONDA_ROOT}\\envs\\ml;"
    f"{CONDA_ROOT}\\envs\\ml\\Library\\bin;"
    f"{CONDA_ROOT}\\envs\\ml\\Scripts;"
    f"{CONDA_ROOT}\\Scripts"
)


def _stub_registry(script: str) -> str:
    """Answer the two registry reads without touching the machine running the suite."""
    return script.replace(
        '$machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")',
        f'$machine = "{MACHINE}"',
    ).replace(
        '$user    = [System.Environment]::GetEnvironmentVariable("Path", "User")',
        f'$user    = "{USER_PATH}"',
    )


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_session_refresh_keeps_an_active_conda_ahead_of_the_user_path(shell: str):
    """The case #5871 is about, in the prompt the user is standing in.

    Without this the message that conda keeps priority is false for the rest of that
    session: the registry ordering is right and the live PATH is not.
    """
    script = _stub_registry(_refresh_preamble(f"{_CONDA_ENTRIES};{MACHINE}"))
    out = _run(
        shell,
        script,
        env = {
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
            "CONDA_EXE": f"{CONDA_ROOT}\\Scripts\\conda.exe",
        },
    )
    assert out.startswith("PATH="), out
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]

    conda_positions = [
        index for index, entry in enumerate(lowered) if entry.startswith(CONDA_ROOT.lower())
    ]
    assert conda_positions, f"the conda entries were dropped entirely: {entries}"
    user_first = min(
        index for index, entry in enumerate(lowered) if entry.startswith("c:\\users\\me")
    )
    assert (
        max(conda_positions) < user_first
    ), f"an activated conda environment was left behind the User PATH: {entries}"
    assert lowered[:4] == [
        f"{CONDA_ROOT}\\envs\\ml".lower(),
        f"{CONDA_ROOT}\\envs\\ml\\Library\\bin".lower(),
        f"{CONDA_ROOT}\\envs\\ml\\Scripts".lower(),
        f"{CONDA_ROOT}\\Scripts".lower(),
    ], entries
    assert len(lowered) == len(set(lowered)), entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_session_refresh_is_unchanged_outside_conda(shell: str):
    """No conda, no reordering. The refresh is on the hot path of the whole installer and
    must keep doing exactly what it did."""
    previous = "C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env = _NO_CONDA)
    entries = out[len("PATH=") :].split(";")
    assert entries == MACHINE.split(";") + USER_PATH.split(";") + [previous], entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_sibling_directory_is_not_dragged_forward_with_the_prefix(shell: str):
    """`C:\\conda-backup` is not inside `C:\\conda`. Matching on the raw string would move an
    unrelated directory to the front of PATH, which is the same class of defect this change
    exists to prevent, pointed the other way."""
    sibling = f"{CONDA_ROOT}-backup\\bin"
    script = _stub_registry(_refresh_preamble(f"{sibling};{CONDA_ROOT}\\Scripts"))
    out = _run(
        shell,
        script,
        env = {
            "CONDA_PREFIX": CONDA_ROOT,
            "CONDA_DEFAULT_ENV": "base",
        },
    )
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    assert lowered[0] == f"{CONDA_ROOT}\\scripts".lower(), entries
    assert lowered.index(sibling.lower()) > lowered.index(
        USER_PATH.split(";")[0].lower()
    ), f"a sibling of the conda prefix was promoted with it: {entries}"


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_stacked_conda_activation_keeps_both_prefixes_in_front(shell: str):
    """`conda activate` inside another environment stacks them and records the outer one in
    CONDA_PREFIX_1. Its entries are on PATH exactly as the inner one's are."""
    outer = f"{CONDA_ROOT}\\envs\\outer\\Scripts"
    inner = f"{CONDA_ROOT}\\envs\\inner\\Scripts"
    script = _stub_registry(_refresh_preamble(f"{inner};{outer}"))
    out = _run(
        shell,
        script,
        env = {
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\inner",
            "CONDA_PREFIX_1": f"{CONDA_ROOT}\\envs\\outer",
            "CONDA_DEFAULT_ENV": "inner",
        },
    )
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    assert lowered[:2] == [inner.lower(), outer.lower()], entries


def test_the_refresh_consults_the_conda_helper_at_all():
    """The three helpers are only worth anything if Refresh-SessionPath calls them, and a
    refresh that rebuilt PATH the old way would leave every case above untouched if they
    were driven directly instead."""
    body = _function(INSTALL_PS1, "    ", "Refresh-SessionPath")
    assert "Test-ActiveCondaEnvironment" in body
    assert "Get-ActiveCondaPrefixes" in body
    assert "Test-PathUnderCondaPrefix" in body
    assert body.index("$sources += $condaFront") < body.index(
        "$sources += @($machine"
    ), "the conda entries are appended after the User PATH, which changes nothing"


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_name_only_activation_keeps_the_session_it_already_had(shell: str):
    """A hook that exports CONDA_DEFAULT_ENV and nothing that names a directory.

    Test-ActiveCondaEnvironment accepts that shape on purpose, but nothing then identifies
    which PATH entries are conda's, so the refresh would rebuild PATH as Machine, User,
    previous and put the hook's entries behind the User PATH again. Guessing from the
    environment NAME would promote any directory containing it, so the caller's PATH is kept
    whole and in front instead.
    """
    previous = f"{CONDA_ROOT}\\envs\\ml\\Scripts;C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env = {"CONDA_DEFAULT_ENV": "ml"})
    entries = out[len("PATH=") :].split(";")
    assert entries[: len(previous.split(";"))] == previous.split(";"), entries
    assert USER_PATH.split(";")[0] in entries, entries
    assert MACHINE.split(";")[0] in entries, entries
    assert entries.index(USER_PATH.split(";")[0]) > entries.index("C:\\tools\\bin"), entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_prefix_activation_still_takes_the_precise_route(shell: str):
    """The whole-PATH fallback is for the case where nothing better can be known, so an
    activation that DOES name its prefix must keep the targeted behaviour: only the conda
    entries move, and an unrelated directory that happened to be on PATH does not."""
    previous = f"C:\\tools\\bin;{CONDA_ROOT}\\envs\\ml\\Scripts"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(
        shell,
        script,
        env = {
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
        },
    )
    entries = out[len("PATH=") :].split(";")
    assert entries[0] == f"{CONDA_ROOT}\\envs\\ml\\Scripts", entries
    assert entries.index("C:\\tools\\bin") > entries.index(USER_PATH.split(";")[0]), entries


# setup.ps1 also refreshes PATH in the caller's process, so it must keep conda first.


def _setup_refresh_preamble(current_path: str) -> str:
    """Refresh-Environment with the registry reads stubbed and $env:Path seeded."""
    body = (
        _function(SETUP_PS1, "", "Test-ActiveCondaEnvironment")
        + _function(SETUP_PS1, "", "Get-ActiveCondaPrefixes")
        + _function(SETUP_PS1, "", "Test-PathUnderCondaPrefix")
        + _function(SETUP_PS1, "", "Refresh-Environment")
    )
    return f"""
$ErrorActionPreference = "Stop"
{body}
$env:VIRTUAL_ENV = ""
$env:Path = "{current_path}"
Refresh-Environment
Write-Host ("PATH=" + $env:Path)
"""


def _stub_setup_registry(script: str) -> str:
    return script.replace(
        "$machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')",
        f'$machinePath = "{MACHINE}"',
    ).replace(
        "$userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')",
        f'$userPath = "{USER_PATH}"',
    )


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_setup_refresh_keeps_an_active_conda_ahead_of_the_user_path(shell: str):
    script = _stub_setup_registry(_setup_refresh_preamble(f"{_CONDA_ENTRIES};{MACHINE}"))
    assert "GetEnvironmentVariable('Path'" not in script, "the registry stub stopped matching"
    out = _run(
        shell,
        script,
        env = {
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
            "CONDA_EXE": f"{CONDA_ROOT}\\Scripts\\conda.exe",
        },
    )
    assert out.startswith("PATH="), out
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    conda_positions = [
        index for index, entry in enumerate(lowered) if entry.startswith(CONDA_ROOT.lower())
    ]
    assert conda_positions, f"the conda entries were dropped entirely: {entries}"
    user_first = min(
        index for index, entry in enumerate(lowered) if entry.startswith("c:\\users\\me")
    )
    assert (
        max(conda_positions) < user_first
    ), f"setup.ps1 left an activated conda environment behind the User PATH: {entries}"
    assert len(lowered) == len(set(lowered)), entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_setup_refresh_is_unchanged_outside_conda(shell: str):
    """Refresh-Environment runs on the hot path of the whole setup and must keep doing
    exactly what it did when no conda environment is active."""
    previous = "C:\\tools\\bin"
    script = _stub_setup_registry(_setup_refresh_preamble(previous))
    out = _run(shell, script, env = _NO_CONDA)
    entries = out[len("PATH=") :].split(";")
    assert entries == MACHINE.split(";") + USER_PATH.split(";") + [previous], entries


def test_the_setup_refresh_consults_the_conda_helper_at_all():
    """A static pair for the pwsh cases above, so a Windows-only regression is still caught
    on a runner without PowerShell."""
    body = _function(SETUP_PS1, "", "Refresh-Environment")
    assert "Test-ActiveCondaEnvironment" in body, body
    assert "Get-ActiveCondaPrefixes" in body, body
    assert "Test-PathUnderCondaPrefix" in body, body


# A prepend persisted by an earlier non-conda run must be repositioned, not accepted.


def _shell_function(path: Path, name: str) -> str:
    source = path.read_text(encoding = "utf-8")
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}\n", source, flags = re.DOTALL | re.M)
    assert match is not None, f"{path.name} no longer defines {name}"
    return match.group(0)


@pytest.mark.parametrize("path", [INSTALL_SH, SETUP_SH_POSIX], ids = ["install.sh", "setup.sh"])
def test_the_rc_repointer_rewrites_only_the_line_it_wrote(path: Path, tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text(
        "# Added by Unsloth installer\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "# a comment\n"
        'export PATH="/opt/mine:$PATH"\n'
        "alias ll='ls -l'\n",
        encoding = "utf-8",
    )
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(
        ["sh", "-c", script],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert "status=0" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding = "utf-8") == (
        "# Added by Unsloth installer\n"
        'export PATH="$PATH:$HOME/.local/bin"\n'
        "# a comment\n"
        'export PATH="/opt/mine:$PATH"\n'
        "alias ll='ls -l'\n"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["rc"]


def test_the_rc_repointer_keeps_a_symlinked_rc_a_symlink(tmp_path: Path):
    """An rc file managed by chezmoi or home-manager is a symlink into a repo. Replacing it
    with a fresh file would detach it from the thing that manages it."""
    real = tmp_path / "dotfiles" / "bashrc"
    real.parent.mkdir()
    real.write_text(
        '# Added by Unsloth installer\nexport PATH="$HOME/.local/bin:$PATH"\n',
        encoding = "utf-8",
    )
    link = tmp_path / ".bashrc"
    link.symlink_to(real)

    script = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{link}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
    )
    subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert link.is_symlink(), "the rc file was replaced instead of rewritten"
    assert real.read_text(encoding = "utf-8") == (
        '# Added by Unsloth installer\nexport PATH="$PATH:$HOME/.local/bin"\n'
    )


@pytest.mark.parametrize(
    "path,arms",
    [
        (INSTALL_SH, ("_persist_login_path_dir", "_persist_fish_path_dir")),
        (SETUP_SH_POSIX, ("_setup_persist_uv_path",)),
    ],
    ids = ["install.sh", "studio/setup.sh"],
)
def test_every_posix_writer_repositions_a_stale_prepend(path: Path, arms: tuple[str, ...]):
    """A guard that only applies to a FIRST run is not a guard: every one of these writers is
    reached again by the next `irm | iex`."""
    for name in arms:
        body = _shell_function(path, name)
        assert (
            "_unsloth_repoint_rc_line" in body
        ), f"{path.name}:{name} accepts a stale prepend as present instead of moving it"


def test_the_repoint_pass_runs_before_the_presence_guards():
    """The shim repoint must run before the _path_has_dir guard, since the stale prepend satisfies it."""
    source = INSTALL_SH.read_text(encoding = "utf-8")
    shim_guard = source.index('if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_LOCAL_BIN"')
    shim_repoint = source.index('"~/.local/bin" \'\\.local/bin\' "" repoint')
    assert shim_repoint < shim_guard, "the shim repoint is behind the guard that skips it"

    uv_guard = source.index('if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_UNSLOTH_UV_BIN_DIR"')
    uv_repoint = source.index('"$_UNSLOTH_UV_BIN_DIR" "" "$_uv_prof" repoint')
    assert uv_repoint < uv_guard, "the uv repoint is behind the guard that skips it"

    setup = SETUP_SH_POSIX.read_text(encoding = "utf-8")
    early_return = setup.index('_setup_path_has_dir "${_SETUP_LOGIN_PATH:-$PATH}" "$_supp_dir"')
    assert (
        "_unsloth_conda_env_active || return 0" in setup[early_return : early_return + 300]
    ), "studio/setup.sh still returns before it can reposition anything"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("repoint", '# Added by Unsloth installer\nexport PATH="$PATH:/opt/unsloth/bin"\n'),
        ("", '# Added by Unsloth installer\nexport PATH="$PATH:/opt/unsloth/bin"\n'),
    ],
    ids = ["repoint-only", "ordinary"],
)
def test_the_repoint_only_mode_adds_nothing(tmp_path: Path, mode: str, expected: str):
    """Repointing must not put a line in the rc file of someone whose PATH comes from
    somewhere else: the caller's guard already decided against adding one."""
    rc = tmp_path / "rc"
    rc.write_text(
        '# Added by Unsloth installer\nexport PATH="/opt/unsloth/bin:$PATH"\n',
        encoding = "utf-8",
    )
    empty = tmp_path / "empty"
    empty.write_text("# nothing here\n", encoding = "utf-8")
    source = INSTALL_SH.read_text(encoding = "utf-8")
    body = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + _shell_function(INSTALL_SH, "_unsloth_conda_env_active")
        + _shell_function(INSTALL_SH, "_persist_fish_path_dir")
        + re.search(r"^_PATH_LINE_RE=.*$", source, flags = re.M).group(0)
        + "\n"
        + _shell_function(INSTALL_SH, "_persist_login_path_dir")
    )
    script = f"""
CONDA_PREFIX=/opt/conda
step() {{ :; }}
substep() {{ :; }}
C_WARN=""
{body}
_persist_login_path_dir "/opt/unsloth/bin" "/opt/unsloth/bin" "/opt/unsloth/bin" \\
    "/opt/unsloth/bin" "{rc}" "{mode}"
_persist_login_path_dir "/opt/unsloth/bin" "/opt/unsloth/bin" "/opt/unsloth/bin" \\
    "/opt/unsloth/bin" "{empty}" "{mode}"
"""
    subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert rc.read_text(encoding = "utf-8") == expected
    added = "/opt/unsloth/bin" in empty.read_text(encoding = "utf-8")
    assert added is (mode != "repoint"), empty.read_text(encoding = "utf-8")


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids = ["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_never_exposes_a_truncated_rc_file(path: Path, tmp_path: Path):
    """Repoint rc files by atomic rename; never cat staged > rc, which truncates the profile first."""
    rc = tmp_path / "rc"
    original = 'export PATH="$HOME/.local/bin:$PATH"\n# keep me\n'
    rc.write_text(original, encoding = "utf-8")
    script = (
        # `mv` commits the rewrite, so a failing shadow `mv` simulates an incomplete commit.
        "mv() { return 1; }\n"
        + _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert "status=1" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding = "utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["rc"], "a staging file was left behind"


def test_the_rc_repointer_keeps_the_permission_bits_it_found(tmp_path: Path):
    """The staged rc is a copy, so the rename keeps the original's mode bits instead of the umask."""
    rc = tmp_path / "rc"
    rc.write_text(
        '# Added by Unsloth installer\nexport PATH="$HOME/.local/bin:$PATH"\n',
        encoding = "utf-8",
    )
    rc.chmod(0o600)
    script = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
    )
    subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert rc.read_text(encoding = "utf-8") == (
        '# Added by Unsloth installer\nexport PATH="$PATH:$HOME/.local/bin"\n'
    )
    assert stat.S_IMODE(rc.stat().st_mode) == 0o600, oct(rc.stat().st_mode)


@pytest.mark.parametrize(
    "path,indent",
    [(INSTALL_PS1, "    "), (SETUP_PS1, "")],
    ids = ["install.ps1", "studio/setup.ps1"],
)
def test_every_stacked_conda_prefix_is_enumerated(path: Path, indent: str):
    """Count stacked conda prefixes by CONDA_SHLVL: a fixed CONDA_PREFIX_1 to _3 list drops the rest."""
    body = _function(path, indent, "Get-ActiveCondaPrefixes")
    assert "CONDA_SHLVL" in body, body
    assert 'GetEnvironmentVariable("CONDA_PREFIX_$level")' in body, body
    for stale in ("CONDA_PREFIX_1", "CONDA_PREFIX_2", "CONDA_PREFIX_3"):
        assert f"$env:{stale}" not in body, (stale, body)
    assert "$levels -gt 64" in body, body
    assert "$env:CONDA_PREFIX)" in body, body


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids = ["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_rewrites_a_line_holding_a_backslash(path: Path, tmp_path: Path):
    """awk -v decodes backslash escapes, so an escaped line never matched; the helper reported success."""
    rc = tmp_path / "rc"
    old_line = 'export PATH="/opt/od\\\\d/bin:$PATH"'
    new_line = 'export PATH="$PATH:/opt/od\\\\d/bin"'
    rc.write_text("# Added by Unsloth installer\n" + old_line + "\n# keep me\n", encoding = "utf-8")
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + "_unsloth_repoint_rc_line "
        + f"'{rc}' '{old_line}' '{new_line}'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert "status=0" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding = "utf-8") == (
        "# Added by Unsloth installer\n" + new_line + "\n# keep me\n"
    ), rc.read_text(encoding = "utf-8")


def test_the_uv_repoint_pass_moves_the_home_relative_spelling_too():
    """The default uv destination IS ~/.local/bin, and the shim block writes that line
    unexpanded as `$HOME/.local/bin`."""
    source = INSTALL_SH.read_text(encoding = "utf-8")
    start = source.index("_uv_repoint_literal=")
    block = source[start : source.index("_persist_fish_path_dir", start)]
    assert "_uv_repoint_home_literal" in block, block
    assert "'$HOME'" in block, block
    assert block.count("_persist_login_path_dir") == 2, block
    assert "${_UNSLOTH_UV_BIN_DIR#$HOME}" in block, block


def test_the_standalone_setup_repoints_the_home_relative_prepend_too():
    """`studio/setup.sh` runs on its own during an update, after install.sh wrote the shim line
    as `export PATH="$HOME/.local/bin:$PATH"`."""
    body = _shell_function(SETUP_SH_POSIX, "_setup_persist_uv_path")
    assert "_supp_export_home_prepend" in body, body
    assert "'$HOME'" in body, body
    assert "${_supp_dir#$HOME}" in body, body
    assert body.count('_supp_export_home_prepend"') >= 2, body


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids = ["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_leaves_a_line_the_user_wrote(path: Path, tmp_path: Path):
    """Rewrite a PATH line only under the Added by Unsloth marker; hand-written copies are the user's."""
    rc = tmp_path / "rc"
    original = "# my own path\n" 'export PATH="$HOME/.local/bin:$PATH"\n' 'alias ll="ls -l"\n'
    rc.write_text(original, encoding = "utf-8")
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output = True, text = True, timeout = 60)
    assert "status=1" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding = "utf-8") == original, rc.read_text(encoding = "utf-8")
