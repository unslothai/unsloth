# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh / setup.ps1 must install the managed runtimes at the same root the resolvers read.

`llama.cpp`, `node` and `whisper.cpp` are siblings of `studio/` under `UNSLOTH_HOME`, and
`storage_roots.unsloth_home()` sends every runtime resolver there. The CLI exports
`UNSLOTH_STUDIO_HOME=<root>/studio` before running setup, so a setup that derived the runtime
parent from the Studio home alone would install them one level too deep and no GGUF model,
managed Node or dictation engine would be found. Companion to
tests/test_managed_tools_master_root.py, which holds the resolvers to each other.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# tests/conftest.py puts tests/_shared on sys.path; see unsloth_pwsh_runner for why a bare
# subprocess call to pwsh is not enough under xdist.
from unsloth_pwsh_runner import PWSH, run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


# These execute blocks of the POSIX installer, so they need bash AND a POSIX filesystem. Git
# Bash puts bash on PATH on a Windows runner, where the blocks run against Windows path
# semantics and hand back UTF-16 (`assert '\x00' == 'ALLOWED'`). setup.ps1 is covered separately
# here and in tests/studio/, so skipping loses no coverage.
NEEDS_POSIX_BASH = pytest.mark.skipif(
    shutil.which("bash") is None or os.name == "nt",
    reason = "runs studio/setup.sh blocks: needs bash on a POSIX filesystem",
)


def _ps_quote(value: str) -> str:
    """A PowerShell single-quoted literal: only the quote itself needs escaping."""
    return "'" + value.replace("'", "''") + "'"


def _slice(src: str, start: str, end: str) -> str:
    begin = src.index(start)
    return src[begin : src.index(end, begin)]


def _runtime_parent(env: dict[str, str]) -> tuple[str, str]:
    """Run the shipped setup.sh derivations and report (node parent, llama.cpp parent).

    The two blocks are executed rather than pattern-matched, so a later edit that keeps the
    words and changes the order still fails here.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    master = _slice(src, "# Stripped before anything else", "# Directory-local evidence")
    node = _slice(src, "# Mirror the llama.cpp UNSLOTH_HOME derivation", "NODE_DIR=")
    llama = _slice(src, 'if [ -n "$STAGE_ROOT" ]; then\n    UNSLOTH_HOME=', "LLAMA_CPP_DIR=")
    script = "\n".join(
        (
            "set -u",
            master,
            node,
            llama,
            'printf "%s\\n%s\\n" "$_NODE_PARENT" "$UNSLOTH_HOME"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    node_parent, llama_parent = completed.stdout.splitlines()
    return node_parent, llama_parent


def _env(home: Path, **overrides: str) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "STAGE_ROOT": "",
        "RUNTIME_ROOT": "",
        "STUDIO_HOME": str(home / ".unsloth" / "studio"),
        "_STUDIO_HOME_IS_CUSTOM": "false",
    }
    env.update(overrides)
    return env


def _master_root_answer(tmp_path: Path, environment: str) -> str:
    """What uninstall.ps1's own _MasterRoot answers, with *environment* run before it.

    The functions are Invoke-Expression'd straight out of the shipped script, so a rewrite that
    keeps the words and loses the behaviour fails here instead of passing against a copy.
    """
    script = tmp_path / "probe.ps1"
    script.write_text(
        f"""$txt = Get-Content -Raw "{UNINSTALL_PS1}"
foreach ($n in @("_ExpandTilde", "_MasterRoot")) {{
    $m = [regex]::Match($txt, "(?ms)^    function $n \\{{.*?^    \\}}")
    if (-not $m.Success) {{ Write-Output "EXTRACT-FAILED:$n"; exit 1 }}
    Invoke-Expression $m.Value
}}
{environment}
""",
        encoding = "utf-8",
    )
    out = run_pwsh(
        [PWSH, "-NoProfile", "-File", str(script)],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    assert "EXTRACT-FAILED" not in out, out
    return out


@NEEDS_POSIX_BASH
def test_a_master_root_puts_the_runtimes_beside_studio(tmp_path):
    root = tmp_path / "portable"
    (root / "studio").mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = str(root),
        # What unsloth_cli/commands/studio.py exports before it runs setup.
        STUDIO_HOME = str(root / "studio"),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == root
    assert Path(llama_parent) == root


@NEEDS_POSIX_BASH
def test_no_master_root_keeps_the_studio_home_layout(tmp_path):
    """The pre-existing custom-root behaviour, which the new branch must not disturb."""
    studio = tmp_path / "elsewhere" / "studio"
    studio.mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        STUDIO_HOME = str(studio),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == studio
    assert Path(llama_parent) == studio


@NEEDS_POSIX_BASH
def test_a_default_install_still_uses_the_legacy_root(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    node_parent, llama_parent = _runtime_parent(_env(home))
    assert Path(node_parent) == home / ".unsloth"
    assert Path(llama_parent) == home / ".unsloth"


@NEEDS_POSIX_BASH
def test_a_staging_root_still_outranks_the_master_root(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = str(tmp_path / "portable"),
        STAGE_ROOT = str(stage),
        RUNTIME_ROOT = str(stage),
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == stage
    assert Path(llama_parent) == stage


def test_setup_ps1_derives_both_runtimes_from_the_master_root():
    """No PowerShell on the Linux runners, so the Windows half is held structurally: both
    derivations must go through the one helper, and it must read UNSLOTH_HOME."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    assert "function Get-MasterRootOverride" in src
    helper = _slice(src, "function Get-MasterRootOverride", "function Get-ManagedLlamaCppDir")
    assert "$env:UNSLOTH_HOME" in helper
    llama = _slice(src, "function Get-ManagedLlamaCppDir", "\n# Failure reason when the managed")
    assert "Get-MasterRootOverride" in llama
    # The master root has to lose to an explicit staging root, as it does in setup.sh.
    assert llama.index("$StagingRoot") < llama.index("Get-MasterRootOverride")
    node = _slice(src, "    $_masterRoot = Get-MasterRootOverride", "    $NodeDir = Join-Path")
    assert "$NodeParent = $_masterRoot" in node
    assert node.index("$StageRoot") < node.index("$NodeParent = $_masterRoot")


@NEEDS_POSIX_BASH
def test_a_padded_master_root_names_the_same_directory(tmp_path):
    """storage_roots.unsloth_home() and the CLI both .strip(), so setup has to as well: an
    unstripped value installs under a directory whose name carries the whitespace."""
    root = tmp_path / "portable"
    (root / "studio").mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = f"  {root}  ",
        STUDIO_HOME = str(root / "studio"),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == root
    assert Path(llama_parent) == root


@NEEDS_POSIX_BASH
def test_a_blank_master_root_counts_as_unset(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    node_parent, llama_parent = _runtime_parent(_env(home, UNSLOTH_HOME = "   "))
    assert Path(node_parent) == home / ".unsloth"
    assert Path(llama_parent) == home / ".unsloth"


BUILD_WHISPER = REPO_ROOT / "scripts" / "build_whisper_cpp.sh"


def _whisper_root_block() -> str:
    """The shipped root selection, from the normalizer down to the INSTALL_DIR it decides."""
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    return _slice(src, "_root_value() {", "STUDIO_OWNED_MARKER=")


@pytest.fixture
def whisper_path(tmp_path_factory):
    """PATH for the extracted whisper block, with git and cmake satisfied by stubs.

    The slice above is the shipped file verbatim, so it carries the builder's `command -v`
    preflight along with the root selection these tests are about. On a runner without cmake the
    block exits 1 before choosing anything and four tests fail for a reason that has nothing to
    do with what they assert. Stubbed rather than excised: cutting the preflight out of the slice
    would mean the tests no longer run the shipped text, which is the whole point of lifting it.
    """
    stub_bin = tmp_path_factory.mktemp("stubbin")
    for tool in ("git", "cmake"):
        path = stub_bin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
        path.chmod(0o755)
    return f"{stub_bin}:/usr/bin:/bin"


@NEEDS_POSIX_BASH
def test_the_whisper_builder_installs_under_the_master_root(tmp_path, whisper_path):
    """setup.sh runs this with UNSLOTH_STUDIO_HOME=<root>/studio still inherited, so a builder
    that preferred it would install a level below _managed_whisper_cpp_dir()."""
    root = tmp_path / "portable"
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    block = _whisper_root_block()
    script = "\n".join(("set -eu", block, 'printf "%s\\n" "$INSTALL_DIR"'))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": whisper_path,
            "UNSLOTH_HOME": str(root),
            "UNSLOTH_STUDIO_HOME": str(root / "studio"),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == root / "whisper.cpp"


@NEEDS_POSIX_BASH
def test_the_whisper_builder_still_honours_a_studio_home_alone(tmp_path, whisper_path):
    studio = tmp_path / "elsewhere" / "studio"
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    block = _whisper_root_block()
    script = "\n".join(("set -eu", block, 'printf "%s\\n" "$INSTALL_DIR"'))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": whisper_path,
            "UNSLOTH_STUDIO_HOME": str(studio),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == studio / "whisper.cpp"


@NEEDS_POSIX_BASH
def test_a_blank_master_root_does_not_outrank_the_whisper_studio_home(tmp_path, whisper_path):
    """${VAR:-} only treats the EMPTY string as unset, so an unstripped whitespace value would
    win the new precedence and name a relative "   /whisper.cpp"."""
    studio = tmp_path / "elsewhere" / "studio"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -eu",
                    _whisper_root_block(),
                    'printf "%s\\n" "$INSTALL_DIR"',
                )
            ),
        ],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": whisper_path,
            "UNSLOTH_HOME": "   ",
            "UNSLOTH_STUDIO_HOME": str(studio),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == studio / "whisper.cpp"


@NEEDS_POSIX_BASH
def test_a_tilde_master_root_expands_for_the_whisper_builder(tmp_path, whisper_path):
    home = tmp_path / "home"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -eu",
                    _whisper_root_block(),
                    'printf "%s\\n" "$INSTALL_DIR"',
                )
            ),
        ],
        env = {"HOME": str(home), "PATH": whisper_path, "UNSLOTH_HOME": "~/portable"},
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == home / "portable" / "whisper.cpp"


@NEEDS_POSIX_BASH
def test_a_relative_master_root_resolves_against_the_caller(tmp_path):
    """Node is chosen before setup.sh's first `cd "$SCRIPT_DIR"` and llama.cpp after it, so a
    value left relative names two different directories and the backend's neither."""
    caller = tmp_path / "caller"
    caller.mkdir()
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s\\n" "$_MASTER_ROOT"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd = str(caller),
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "PWD": str(caller),
            "UNSLOTH_HOME": "not-created-yet",
            "_STUDIO_HOME_IS_CUSTOM": "false",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == caller / "not-created-yet"


@NEEDS_POSIX_BASH
def test_a_master_root_alone_still_asserts_ownership_of_the_runtimes(tmp_path):
    """With only UNSLOTH_HOME set, STUDIO_HOME stays legacy and _STUDIO_HOME_IS_CUSTOM stays
    false, which is what licenses the installers to os.replace() and rm -rf without checking
    the Unsloth-owned marker. The runtimes moved, so the guard has to follow them."""
    root = tmp_path / "portable"
    root.mkdir()
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s %s\\n" "$_STUDIO_HOME_IS_CUSTOM" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            "UNSLOTH_HOME": str(root),
            "_STUDIO_HOME_IS_CUSTOM": "false",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["false", "true"]


@NEEDS_POSIX_BASH
@pytest.mark.parametrize(
    "environment, expected",
    [
        # Staging: the placement below gives STAGE_ROOT precedence over the master root, so the
        # master root is not where anything lands and must not decide ownership. Raised anyway,
        # it demanded an owner marker from a markerless tree the old updater had staged, and an
        # update that worked on the merge base exited instead.
        pytest.param(
            {"UNSLOTH_STUDIO_STAGE_ROOT": "STAGE", "_STUDIO_HOME_IS_CUSTOM": "false"},
            ["false", "false"],
            id = "a staged update ignores the master root",
        ),
        # And the flag has to be ASSIGNED, not only raised: a custom STUDIO_HOME whose runtimes
        # land in the legacy root kept it true and demanded markers from exactly the pre-marker
        # ~/.unsloth/llama.cpp this comparison exists to spare.
        pytest.param(
            {"UNSLOTH_HOME": "LEGACY", "_STUDIO_HOME_IS_CUSTOM": "true"},
            ["true", "false"],
            id = "a custom studio home whose runtimes stay legacy",
        ),
    ],
)
def test_the_ownership_flag_follows_where_the_runtimes_land(tmp_path, environment, expected):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    stage = tmp_path / "stage"
    stage.mkdir()
    resolved = {
        key: str(stage)
        if value == "STAGE"
        else str(home / ".unsloth")
        if value == "LEGACY"
        else value
        for key, value in environment.items()
    }
    if "UNSLOTH_HOME" not in resolved:
        resolved["UNSLOTH_HOME"] = str(tmp_path / "portable")
        (tmp_path / "portable").mkdir()
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            f'STAGE_ROOT="{resolved.pop("UNSLOTH_STUDIO_STAGE_ROOT", "")}"',
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s %s\\n" "$_STUDIO_HOME_IS_CUSTOM" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {"HOME": str(home), "PATH": "/usr/bin:/bin", **resolved},
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == expected


@NEEDS_POSIX_BASH
def test_no_master_root_leaves_the_ownership_flag_alone(tmp_path):
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s %s\\n" "$_STUDIO_HOME_IS_CUSTOM" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )
    for flag in ("false", "true"):
        completed = subprocess.run(
            ["bash", "-c", script],
            env = {
                "HOME": str(tmp_path / "home"),
                "PATH": "/usr/bin:/bin",
                "_STUDIO_HOME_IS_CUSTOM": flag,
            },
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.split() == [flag, flag]


def test_every_runtime_ownership_guard_uses_the_runtime_flag():
    """The Studio home and its venvs keep _STUDIO_HOME_IS_CUSTOM; every guard that names a
    runtime child has to move, or a master-root install loses the marker check on it."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    for line in src.splitlines():
        if "_STUDIO_HOME_IS_CUSTOM" not in line:
            continue
        assert not any(
            name in line for name in ("$NODE_DIR", "$LLAMA_CPP_DIR", "$WHISPER_CPP_DIR")
        ), line
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    # Seeded from the Studio flag and raised only when the master root is elsewhere, as setup.sh
    # does; test_the_windows_legacy_root_named_explicitly_is_not_custom runs the divergence.
    assert "$RuntimeRootIsCustom = $StudioHomeIsCustom\n" in ps
    assert "$_masterRootForOwnership -ine $_legacyRuntimeRoot" in ps
    for line in ps.splitlines():
        if "$StudioHomeIsCustom" not in line:
            continue
        assert not any(
            name in line for name in ("$LlamaCppDir", "$WhisperCppDir", "$NodeDir")
        ), line


@NEEDS_POSIX_BASH
def test_the_legacy_root_named_explicitly_is_not_custom(tmp_path):
    """UNSLOTH_HOME=$HOME/.unsloth names the root a default install already uses.

    Nothing moves, so demanding an owner marker there rejects an update that used to reuse a
    legacy source-built ~/.unsloth/llama.cpp, which predates the marker entirely. Customness has
    to come from where the runtimes LAND, not from whether the variable was set.
    """
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "# Stripped before anything else", "# Directory-local evidence")
    script = "\n".join(
        (
            "set -u",
            "_STUDIO_HOME_IS_CUSTOM=false",
            block,
            'printf "%s\\n" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )

    def flag(master: str) -> str:
        done = subprocess.run(
            ["bash", "-c", script],
            env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "UNSLOTH_HOME": master},
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    assert flag(str(home / ".unsloth")) == "false"
    # Non-vacuity: a root that really is elsewhere still takes the strict path.
    assert flag(str(tmp_path / "portable")) == "true"
    assert flag("") == "false"


@pytest.mark.skipif(PWSH is None, reason = "needs pwsh")
def test_the_windows_legacy_root_named_explicitly_is_not_custom(tmp_path):
    """The Windows half of the test above, run rather than pattern-matched.

    setup.ps1 read `$StudioHomeIsCustom -or [bool](Get-MasterRootOverride)`, so
    UNSLOTH_HOME=%USERPROFILE%\\.unsloth turned the ownership guard on for a root a default
    install is already using. Nothing moves there, and the guard then demands an owner marker
    from a markerless source-built .unsloth\\llama.cpp, so Assert-StudioOwnedOrAbsent exits an
    update that used to reuse that build. setup.sh adopts it, which made the two halves answer
    differently for one environment.
    """
    profile = tmp_path / "profile"
    (profile / ".unsloth" / "studio").mkdir(parents = True)
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    # The derivation itself, from the flag it seeds to the line after it.
    block = _slice(ps, "$RuntimeRootIsCustom = $StudioHomeIsCustom", "$LlamaCppDir = ")
    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            (
                f"$txt = Get-Content -Raw '{SETUP_PS1}'",
                # The shipped helpers, so a rewrite that keeps the words and changes the
                # canonicalisation still fails here.
                'foreach ($n in @("Get-PathState", "Test-AccessDeniedError", "Get-CanonicalDir",'
                ' "Get-MasterRootOverride")) {',
                '    $m = [regex]::Match($txt, "(?ms)^function $n \\{.*?^\\}")',
                '    if (-not $m.Success) { Write-Output "EXTRACT-FAILED:$n"; exit 1 }',
                "    Invoke-Expression $m.Value",
                "}",
                # Test-StudioHomeIsCustom closes over $StudioHome, so it is stated here rather
                # than extracted: the point of this probe is the master-root comparison.
                "$StudioHomeIsCustom = $false",
                block,
                "Write-Output $RuntimeRootIsCustom",
            )
        ),
        encoding = "utf-8",
    )

    def flag(master: str) -> str:
        out = run_pwsh(
            [PWSH, "-NoProfile", "-File", str(script)],
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(profile),
                "USERPROFILE": str(profile),
                "UNSLOTH_HOME": master,
            },
            capture_output = True,
            text = True,
            check = True,
        ).stdout.strip()
        assert "EXTRACT-FAILED" not in out, out
        return out.splitlines()[-1].strip()

    assert flag(str(profile / ".unsloth")) == "False"
    # Non-vacuity, and the same two controls the bash test uses.
    assert flag(str(tmp_path / "portable")) == "True"
    assert flag("") == "False"


@pytest.mark.skipif(PWSH is None, reason = "needs pwsh")
def test_the_windows_uninstaller_only_removes_a_root_that_is_really_empty(tmp_path):
    """The master root and its .staging go only when empty, and by an operation that cannot do
    anything else.

    Windows asked with Get-ChildItem -ErrorAction SilentlyContinue and then called _RemovePath,
    which is Remove-Item -Recurse -Force, in the one directory this branch teaches users to
    point at their own files. Two separate calls, so a file arriving between the two was taken
    by the -Recurse; and a swallowed enumeration error read as an empty directory.
    uninstall.sh uses rmdir for both paths, which is one call that refuses a non-empty
    directory, and this file's own %TEMP% prune already asks with -ErrorAction Stop.

    The behavioural halves below are what can be shown on a POSIX host: a recursive delete needs
    to enumerate too, so an unlistable directory survives either version here and the ACL shape
    that separates DELETE from LIST is Windows-only. What is checked instead is the mechanism,
    which is what removes the race as well.
    """
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    assert "function _RemoveDirIfEmpty" in ps
    helper = _slice(ps, "    function _RemoveDirIfEmpty {", "\n    # The exact shape")
    # One atomic call: no -Recurse anywhere in it, and no emptiness question asked separately.
    assert "[System.IO.Directory]::Delete($Path, $false)" in helper
    assert "-Recurse" not in helper, helper
    assert "Get-ChildItem" not in helper, helper
    # And both master-root sites go through it rather than keeping their own version.
    master_block = _slice(ps, "$masterRoot = $masterRootToStop", "if ($defaultLlamaCpp)")
    assert master_block.count("_RemoveDirIfEmpty") == 2, master_block
    assert "-ErrorAction SilentlyContinue)) {\n            _RemovePath" not in master_block

    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            (
                f"$txt = Get-Content -Raw '{UNINSTALL_PS1}'",
                '$m = [regex]::Match($txt, "(?ms)^    function _RemoveDirIfEmpty \\{.*?^    \\}")',
                'if (-not $m.Success) { Write-Output "EXTRACT-FAILED"; exit 1 }',
                "Invoke-Expression $m.Value",
                "function _Substep { param($a, $b) }",
                "_RemoveDirIfEmpty $args[0]",
                'Write-Output ("EXISTS=" + (Test-Path -LiteralPath $args[0]))',
            )
        ),
        encoding = "utf-8",
    )

    def still_there(path: Path) -> bool:
        out = run_pwsh(
            [PWSH, "-NoProfile", "-File", str(script), str(path)],
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
            capture_output = True,
            text = True,
            check = True,
        ).stdout
        assert "EXTRACT-FAILED" not in out, out
        assert "EXISTS=" in out, out
        return out.strip().splitlines()[-1] == "EXISTS=True"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert not still_there(empty), "an empty root is the one case that may be removed"

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "theirs.txt").write_text("mine", encoding = "utf-8")
    assert still_there(occupied)
    assert (occupied / "theirs.txt").is_file(), "a non-empty root must keep its contents"

    # A directory holding only a subdirectory: rmdir refuses this too, and a -Recurse would not.
    nested = tmp_path / "nested"
    (nested / "child").mkdir(parents = True)
    assert still_there(nested)
    assert (nested / "child").is_dir()


@NEEDS_POSIX_BASH
def test_only_a_directory_can_be_adopted_at_a_runtime_path(tmp_path):
    """A dangling symlink and a regular file are both things the user put there.

    -d alone read either as "nothing is here", and the caller then rm -rf'd the path or let
    install_*_prebuilt.py os.replace() over it. A dangling link is the ordinary case: its target
    volume is simply not mounted, and resolving it later installs onto somebody's other disk.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "_studio_path_shape() {", "\n_packaged_frontend_available")
    script = "\n".join(
        (
            "set -u",
            "_STUDIO_OWNED_MARKER=.unsloth-studio-owned",
            "_studio_owned_adoptable() { return 1; }",
            "_studio_dir_unsearchable() { return 1; }",
            "_path_access_denied() { echo DENIED; exit 9; }",
            "setup_fail() { echo REFUSED; exit 1; }",
            block,
            '_assert_studio_owned_or_absent "$1" llama.cpp true',
            "echo ALLOWED",
        )
    )

    def verdict(path: pathlib.Path) -> str:
        done = subprocess.run(
            ["bash", "-c", script, "_", str(path)],
            env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
            capture_output = True,
            text = True,
            timeout = 60,
        )
        return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-120:]

    assert verdict(tmp_path / "absent") == "ALLOWED"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "no-such-volume" / "llama.cpp")
    assert verdict(dangling) == "REFUSED"

    plain = tmp_path / "afile"
    plain.write_text("mine")
    assert verdict(plain) == "REFUSED"

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / ".unsloth-studio-owned").touch()
    assert verdict(owned) == "ALLOWED"


def test_the_windows_inductor_cache_agrees_with_the_resolver():
    """setup.ps1 persists TORCHINDUCTOR_CACHE_DIR to the USER environment.

    Every later Studio process inherits it, so _setup_cache_env's fill-if-unset default never
    applies on Windows and the containment this branch is for does not happen there. It has to
    name the directory the resolver would have chosen. Two things still outrank that, and both
    are recorded here so a later edit cannot quietly drop them: long paths off keeps the short
    drive-root directory for MAX_PATH headroom, and a path holding whitespace or an apostrophe is
    refused for the same reason storage_roots.toolchain_path_unparseable refuses one, since the
    C++ builders paste it in unquoted and reparse it with shlex in POSIX mode.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(ps, "$TorchCacheDir = $null", "$env:TORCHINDUCTOR_CACHE_DIR = $TorchCacheDir")
    assert 'Join-Path (Join-Path $StudioHome "cache") "torchinductor"' in block
    assert "$LongPathsEnabled" in block
    refusal = re.search(r"-notmatch\s+'(\[[^']*(?:''[^']*)*\])'", block)
    assert refusal, "the unparseable-path refusal is gone"
    assert "\\s" in refusal.group(1), "the whitespace refusal is gone"
    assert "''" in refusal.group(1), "the apostrophe refusal is gone"
    # The two refusals do NOT share a destination. A spaced path keeps the drive-root directory
    # it has always used; an apostrophe-only path publishes nothing, because C:\tc is shared and
    # predictable and _setup_cache_env honours an inherited value without applying its own
    # per-account rule to it.
    assert "$TorchCacheUnparseable" in block, "the apostrophe case lost its separate route"
    assert "-not $TorchCacheUnparseable" in block
    assert '"C:\\tc"' in block

    roots = (REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py").read_text(
        encoding = "utf-8"
    )
    # The same key, named by both sides, so the two cannot drift apart silently.
    assert '"TORCHINDUCTOR_CACHE_DIR",' in roots
    assert 'str(root / "torchinductor")' in roots


def test_the_windows_node_guard_covers_a_master_root():
    """setup.ps1's Node ownership guard and its marker both hung off $NodeOverride alone.

    $NodeOverride is set only in the UNSLOTH_STUDIO_HOME / STUDIO_HOME branch. The master-root
    branch sets $NodeParent and leaves it null, so <master>\\node reached the whole-directory
    os.replace() in install_node_prebuilt.py with no ownership evidence at all, and the tree the
    run then created stayed unmarked, which makes the uninstaller decline to remove it later.
    setup.sh had already moved these two sites to _RUNTIME_ROOT_IS_CUSTOM.

    The sibling test above only rejects $StudioHomeIsCustom beside $NodeDir, which this bug
    never wrote: it named a third variable. So the rule here is positive, not a denial.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    # Every site in the Node install section that decides whether the Node path is the user's.
    # Keyed on $NodeOverride, the variable the bug named, not $NodeDir: the guard's probe moved
    # to its own line, so a $NodeDir test would pass by seeing less. The section start drops the
    # parsing block far above, which rightly says nothing about the runtime root.
    lines = ps.splitlines()
    start = next(i for i, line in enumerate(lines) if "$nodeEntry = if (" in line)
    guards = [line for line in lines[start:] if "$NodeOverride" in line]
    # The guard's probe, and the marker's `if`. Fewer means the block was restructured and this
    # test would otherwise pass by finding nothing.
    assert len(guards) >= 2, guards
    for line in guards:
        assert "$RuntimeRootIsCustom" in line, line


UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"
UNINSTALL_PS1 = REPO_ROOT / "scripts" / "uninstall.ps1"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh not available")
def test_the_windows_uninstaller_resolves_a_relative_root_like_setup(tmp_path):
    """setup.ps1 resolves through PowerShell's own location; uninstall.ps1 used
    [IO.Path]::GetFullPath, which anchors at [Environment]::CurrentDirectory.

    PowerShell does not keep those two in step, so after a Set-Location a relative UNSLOTH_HOME
    named one install to setup and a different one to the uninstaller. The owner marker does not
    save you there: it spares trees that are not Unsloth's, and the wrongly resolved path is
    another Unsloth install, marker and all.

    Runs the shipped function rather than matching its text, so a rewrite that keeps the words
    and loses the behaviour still fails.
    """
    initial = tmp_path / "initial"
    chosen = tmp_path / "chosen"
    (chosen / "portable").mkdir(parents = True)
    initial.mkdir()
    out = _master_root_answer(
        tmp_path,
        f"""[System.Environment]::CurrentDirectory = "{initial}"
Set-Location "{chosen}"
$env:UNSLOTH_HOME = "portable"
$env:USERPROFILE = "{tmp_path}/profile"
Write-Output (_MasterRoot)
""",
    )
    assert out == str(chosen / "portable"), out


def test_neither_uninstaller_recurses_into_an_install_lock_path():
    """A lock is always a regular file, so a directory at one of those fixed names is the user's.

    prebuilt_core.install_lock creates it with os.open(O_CREAT | O_EXCL). Both uninstallers
    reached the lock names through their recursive remover, which in a user-chosen master root
    deletes a whole tree with none of the owner-marker proof the runtime children require.
    Behaviour is covered by tests/sh/test_uninstall_master_root.sh for the POSIX half; this
    holds the PowerShell twin, which has no runner here.
    """
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")

    assert "_remove_lock_file() {" in sh
    assert "function _RemoveLockFile" in ps

    # The call sites do not all spell the lock out (PowerShell iterates $lockName, both sweep a
    # $stale), so matching only "install.lock" let the master-root loop keep the recursive
    # remover and still pass.
    lockish = ("install.lock", "lockName", "$stale", "_stale", "_mr_lock")
    for text, remover, shape in (
        (sh, "_remove_path ", "_remove_lock_file"),
        (ps, "_RemovePath ", "_RemoveLockFile"),
    ):
        hits = 0
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if remover not in line and shape not in line:
                continue
            if not any(token in line for token in lockish):
                continue
            hits += 1
            assert shape in line, line
        # Counts the call sites found, not the ones left wrong: once they are all correct the
        # remover no longer appears beside a lock at all, and a count of zero would mean the
        # tokens above had stopped matching and this loop proved nothing.
        assert hits >= 4, (remover, hits)

    # The rename keeps the leading dot, and neither a glob without one nor the dot alone is
    # enough, so the sweep matches the installer's full shape; held by
    # test_neither_uninstaller_sweeps_a_stale_lock_name_it_did_not_make below.
    assert "install.lock.stale" in ps
    assert '"*.install.lock.stale.*"' not in ps


def test_both_uninstallers_clear_the_master_root_children():
    """setup installs llama.cpp, node and whisper.cpp as children of the master root, so an
    uninstaller that only knows the legacy siblings and the Studio root strands them. Behaviour
    is covered by tests/sh/test_uninstall_master_root.sh; this holds the PowerShell twin, which
    the Linux runners cannot execute, and pins the marker gate on both."""
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    assert "_master_root() {" in sh
    assert "function _MasterRoot" in ps
    for src, marker in ((sh, ".unsloth-studio-owned"), (ps, ".unsloth-studio-owned")):
        block = _slice(src, "master root's own children", "llama.cpp build + cache")
        assert marker in block, block
        for child in ("llama.cpp", "node", "whisper.cpp"):
            assert child in block, (child, block)
    # A user-chosen root reaches the deny list on both sides before anything is removed.
    assert '_is_unsafe_root "$_mr_root"' in sh
    assert "_IsUnsafeRoot $masterRoot" in ps
    # And it only contributes its studio child when neither exact override is set, as in
    # storage_roots.studio_root().
    assert '_emit "$(_master_root)/studio"' in sh
    assert '$envRoot = (Join-Path $master "studio")' in ps


def test_the_stop_pass_covers_the_master_root_runtimes():
    """Windows locks a loaded executable, so a runtime still running under the master root has
    to be stopped before its tree is removed or the delete exhausts its retries. _MasterRoot is
    resolved before the stop pass, and only marker-owned children join it, so an unmarked
    neighbour's process is never killed."""
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    stop_line = next(l for l in ps.splitlines() if l.strip().startswith("$stopRoots = "))
    assert "$masterChildrenToStop" in stop_line, stop_line
    block = _slice(ps, "$masterRootToStop = _MasterRoot", "$stopRoots = ")
    assert ".unsloth-studio-owned" in block, block
    assert "_IsUnsafeRoot $masterRootToStop" in block, block
    assert ps.index("$masterRootToStop = _MasterRoot") < ps.index(
        "_StopProcessesLockingRoots -Roots"
    )


def test_a_shared_staging_directory_is_pruned_not_deleted():
    """The prebuilt installers share <root>/.staging and prune it only when empty, so anything
    left in a user-chosen root is not ours to delete recursively."""
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    assert 'rmdir "$_mr_root/.staging"' in sh
    assert '_remove_path "$_mr_root/.staging"' not in sh
    # The Windows equivalent of rmdir, for the same reason. Asking with Get-ChildItem and then
    # calling _RemovePath is not it: see
    # test_the_windows_uninstaller_only_removes_a_root_that_is_really_empty.
    staging = _slice(
        ps, '_RemoveDirIfEmpty (Join-Path $masterRoot ".staging")', "# Shared llama.cpp build"
    )
    assert "_RemovePath" not in staging, staging


def test_the_windows_uninstaller_refuses_a_note_carried_in_from_elsewhere(tmp_path):
    """A Studio tree copied from master root A to B keeps a note naming A.

    A's llama.cpp, node, whisper.cpp and sd.cpp carry exactly the owner markers B's would, so
    the marker gates cannot tell them apart: accepting the copied note has the uninstall of B
    delete the ORIGINAL install's runtimes. The note has to describe the tree it was found in,
    which means the Studio directory it was read from lying inside the root it names.
    """
    original = tmp_path / "original"
    (original / "studio").mkdir(parents = True)
    copied = tmp_path / "copied"
    (copied / "studio" / "share").mkdir(parents = True)
    (copied / "studio" / "share" / ".unsloth-master-root").write_text(
        f"{original}\n", encoding = "utf-8"
    )
    profile = tmp_path / "profile"
    (profile / ".unsloth" / "studio" / "share").mkdir(parents = True)

    script = tmp_path / "probe.ps1"
    script.write_text(
        f"""$txt = Get-Content -Raw "{UNINSTALL_PS1}"
foreach ($n in @("_ExpandTilde", "_RootFromConf", "_MasterRoot")) {{
    $m = [regex]::Match($txt, "(?ms)^    function $n \\{{.*?^    \\}}")
    if (-not $m.Success) {{ Write-Output "EXTRACT-FAILED:$n"; exit 1 }}
    Invoke-Expression $m.Value
}}
$env:UNSLOTH_HOME = ""
$env:STUDIO_HOME = ""
$env:UNSLOTH_STUDIO_HOME = "{copied}/studio"
$env:USERPROFILE = "{profile}"
$answer = _MasterRoot
Write-Output "ANSWER:$answer"
""",
        encoding = "utf-8",
    )
    out = run_pwsh(
        [PWSH, "-NoProfile", "-File", str(script)],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    assert "EXTRACT-FAILED" not in out, out
    assert out == "ANSWER:", out


def test_the_windows_uninstaller_does_not_borrow_another_installs_note(tmp_path):
    """Two installs on one box: the legacy tree carries a note, the named one does not.

    _MasterRoot probed the legacy path first, so its note won here while the removal still
    worked on the tree UNSLOTH_STUDIO_HOME names. The run deleted the named Studio and then
    followed the OTHER install's master root and took its marked runtime children with it.
    Runs the shipped function, as the test below does.
    """
    profile = tmp_path / "profile"
    (profile / ".unsloth" / "studio" / "share").mkdir(parents = True)
    borrowed = tmp_path / "borrowed"
    (borrowed / "studio").mkdir(parents = True)
    (profile / ".unsloth" / "studio" / "share" / ".unsloth-master-root").write_text(
        f"{borrowed}\n", encoding = "utf-8"
    )
    named = tmp_path / "named"
    (named / "share").mkdir(parents = True)

    out = _master_root_answer(
        tmp_path,
        f"""$env:UNSLOTH_HOME = ""
$env:STUDIO_HOME = ""
$env:UNSLOTH_STUDIO_HOME = "{named}"
$env:USERPROFILE = "{profile}"
$answer = _MasterRoot
Write-Output "ANSWER:$answer"
""",
    )
    assert out == "ANSWER:", out


def test_the_windows_uninstaller_finds_a_master_root_from_the_note(tmp_path):
    """`$env:UNSLOTH_HOME = 'D:\\portable'; unsloth studio update` names the root for one command.

    setup.ps1 puts node\\, llama.cpp\\ and whisper.cpp\\ under it and marks them, and a later
    uninstall run from an ordinary shell has no UNSLOTH_HOME at all. Reading only the current
    environment, _MasterRoot answered null there, so the run removed <master>\\studio and left
    multi-gigabyte runtimes beside it. setup.sh already wrote the note; setup.ps1 did not, and
    the PowerShell uninstaller did not read it.

    Runs the shipped function, so a rewrite that keeps the words and loses the behaviour fails.
    """
    master = tmp_path / "portable"
    (master / "studio" / "share").mkdir(parents = True)
    (master / "studio" / "share" / ".unsloth-master-root").write_text(
        f"{master}\n", encoding = "utf-8"
    )
    profile = tmp_path / "profile"
    (profile / ".unsloth" / "studio" / "share").mkdir(parents = True)

    out = _master_root_answer(
        tmp_path,
        f"""$env:UNSLOTH_HOME = ""
$env:STUDIO_HOME = ""
$env:UNSLOTH_STUDIO_HOME = "{master}/studio"
$env:USERPROFILE = "{profile}"
Write-Output (_MasterRoot)
""",
    )
    assert out == str(master), out


def test_the_windows_setup_records_the_master_root_for_the_uninstaller():
    """The note the test above reads has to be written, and only where it is true.

    A staging run installs to a throwaway root, and every non-master branch derives the root
    from paths the uninstaller already knows, so a note there would only ever be able to go
    stale. This is the same rule setup.sh applies.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, "# Record the master root inside the Studio tree", "$WithLlamaCppDir = $null"
    )
    assert "(Get-MasterRootOverride)" in block
    assert "-not $StageRoot" in block
    assert '".unsloth-master-root"' in block
    # Staged then renamed: a reader catching a half-written note would name a truncated path,
    # and this note licenses deletions.
    assert "$noteTmp" in block and "Move-Item" in block
    # The 3-argument overwrite overload is .NET Core only, and setup.ps1 runs under 5.1.
    assert "[System.IO.File]::Move(" not in block


def test_the_refused_windows_cache_path_is_taken_away_on_upgrade():
    """Declining to write a value is not enough: the old one is already persisted.

    Every setup before the refusal existed sent an apostrophe-named account's contained path
    straight to $TorchCacheDir and wrote it to the USER environment, so on an upgrade the value is
    sitting there and this process inherited it. _setup_cache_env honours any inherited value as
    the caller's own choice and never applies its per-account rule, which would leave the one
    account this branch exists for as the one account the backend never gets to help. So the
    refusal has to clear both copies, and only when the value is one the builders cannot read: a
    parseable path is somebody's decision.

    And it has to clear them on EVERY launch. The refusal itself lives inside
    `if (-not $SkipPythonDeps)`, which a current core package and a verified UV_OFFLINE tree both
    skip, so a cleanup nested in there would never reach the account that is already in this
    state. The clear needs nothing that block computes, so it is hoisted out of it.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    body = _slice(ps, "function Clear-UnparseableTorchCacheEnv {", "\nClear-UnparseableTorchCacheEnv")
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    assert "GetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', 'User')" in code, code
    assert "[NullString]::Value, 'User'" in code, code
    assert "Remove-Item -LiteralPath Env:TORCHINDUCTOR_CACHE_DIR" in code, code
    # Guarded on the same characters the refusal itself uses, both times. An unguarded clear
    # would throw away a directory the user chose on purpose.
    assert code.count("-match '[\\s'']'") == 2, code
    # A staged run never writes the real account's environment, as the persist below does not.
    assert "-not $StageRoot" in code, code

    # The call is outside the dependency block, measured by brace depth rather than by reading.
    lines = ps.splitlines()
    gate = lines.index("if (-not $SkipPythonDeps) {")
    depth = 0
    for close, line in enumerate(lines[gate:], start = gate):
        depth += line.count("{") - line.count("}")
        if depth == 0:
            break
    calls = [i for i, line in enumerate(lines) if line.strip() == "Clear-UnparseableTorchCacheEnv"]
    assert calls, "nothing calls the cleanup"
    assert any(i < gate for i in calls), (
        "the cleanup only runs on a dependency pass, which the fast paths skip"
    )


def test_the_windows_uninstaller_clears_the_inductor_path_it_persisted():
    """setup.ps1 writes TORCHINDUCTOR_CACHE_DIR to the USER environment, so it outlives the
    install. Every later PyTorch process on the account inherits it, including ones unrelated to
    Unsloth, and they compile into the deleted tree and rebuild part of it.

    Only a value inside a root this run owned: a directory the user chose is theirs, and the
    shared C:\\tc fallback is not install specific and is not deleted here either.
    """
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, "# Clear the persisted Inductor cache path", "# Remove HKCU\\Software\\Unsloth"
    )
    assert "GetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', 'User')" in block
    assert "[NullString]::Value, 'User'" in block
    # Scoped to what this run owns. $knownRoots includes roots the gates refused to delete.
    assert "$ownedRoots" in block and "$knownRoots" not in block
    # Comments stripped first: this block explains why C:\tc is spared, and a comment saying so
    # is not the same thing as code naming it.
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    assert "C:\\tc" not in code


def test_the_windows_node_guard_treats_a_file_as_occupied():
    """install_node_prebuilt's _swap_into_place renames whatever it finds out of the way.

    Guarding on -PathType Container meant a regular file or a symlink named `node` under a
    user-selected master root was invisible to the check and got displaced. Ownership evidence
    lives inside a directory, so a non-directory can never carry it and is refused outright.
    setup.sh's _assert_studio_owned_or_absent takes the same view of -d against -e and -L.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps,
        "        $nodeEntry = if ($NodeOverride -or $RuntimeRootIsCustom) {",
        "install_node_prebuilt.py",
    )
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    # Get-Item -Force, not Test-Path: under 5.1 Test-Path answers for the link TARGET, so a
    # dangling `node` link read as absent and install_node_prebuilt.py installed through it.
    assert "Get-Item -LiteralPath $NodeDir -Force" in code, code
    assert "Test-Path -LiteralPath $NodeDir" not in code, code
    # A reparse point is not a directory here either, however it resolves.
    assert "$nodeEntry.PSIsContainer" in code and "ReparsePoint" in code, code
    # And a non-directory must fail rather than fall through to the marker questions.
    assert "$nodeIsDir" in code and "-not $nodeIsDir -or" in code


def test_the_windows_runtime_guard_treats_a_file_as_occupied():
    """The Node guard above is one of three: llama.cpp and whisper.cpp share
    Assert-StudioOwnedOrAbsent, which was still container-only.

    install_llama_prebuilt's activate_install_tree moves aside whatever Path.exists() finds, so
    a regular file named `llama.cpp` under a master root was renamed to a rollback name and
    replaced. The behaviour is run for real in tests/studio/test_path_probe_access_denied.ps1,
    which the Linux runners cannot execute; this holds the shape there too.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(ps, "function Assert-StudioOwnedOrAbsent", "function Mark-StudioOwned")
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    # Get-Item -Force, not another Test-Path: 5.1 answers false for a dangling link.
    assert "Get-Item -LiteralPath $Path -Force" in code
    # A non-directory is refused, and only under a root the user chose.
    assert "if (-not $isCustomRoot) { return }" in code
    assert "already exists and is not a directory" in block
    # Fatal even in -NonFatal mode: that mode rescues a denial, not somebody else's file.
    refusal = code[
        code.index("Get-Item -LiteralPath $Path -Force") : code.index("Exit-SetupFailure")
    ]
    assert "$NonFatal" not in refusal, refusal


def test_neither_uninstaller_takes_a_studio_root_that_is_also_the_master_root():
    """The flat layout, UNSLOTH_HOME and UNSLOTH_STUDIO_HOME naming one directory.

    A Studio root is removed WHOLE once it carries the ownership marker, so in the flat case
    anything else the user keeps in that directory goes with the install. Every other
    master-root child is individually marker-gated precisely so a user-chosen root is never
    removed wholesale; this was the one hole in that rule. Kept rather than pruned: data left
    behind is recoverable and printed, a deleted file is not.
    """
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")

    sh_block = _slice(sh, "_crf_canon=", '_remove_root_recording_db "$_custom_root"')
    sh_code = "\n".join(l for l in sh_block.splitlines() if not l.lstrip().startswith("#"))
    assert "_MASTER_ROOT_SAVED" in sh_code and "continue" in sh_code
    # Canonicalised on both sides, or a symlinked path compares unequal to itself and the guard
    # never fires on the very layout it is for.
    assert "cd -P --" in sh_code

    ps_block = _slice(ps, "$flatMaster = $masterRootToStop", "_RemoveRootRecordingDb $r")
    ps_code = "\n".join(l for l in ps_block.splitlines() if not l.lstrip().startswith("#"))
    assert "$masterRootToStop" in ps_code and "continue" in ps_code
    # Case-insensitive: Windows paths differing only in case are the same directory.
    assert "-ieq" in ps_code


def test_neither_uninstaller_re_resolves_the_master_root_after_deleting_it():
    """_master_root can read its answer from a note INSIDE a Studio tree.

    The custom-root loop removes that tree, so a second call after it returns nothing and the
    marked llama.cpp, Node and whisper.cpp siblings are stranded: the very failure the note was
    added to prevent, reintroduced by asking too late. Both scripts resolve it once, before
    anything is deleted, and every later use takes that value.
    """
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")

    sh_code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    saved = sh_code.index('_MASTER_ROOT_SAVED="$(_master_root)"')
    removal = sh_code.index("_custom_studio_roots | while")
    # Resolved before the first deletion, and nothing asks again after it. Calls BEFORE the loop
    # are fine: those enumerate while every tree is still on disk.
    assert saved < removal
    after = sh_code[removal:]
    assert "_master_root" not in after, after[: after.index("_master_root") + 200]
    # Both consumers take the saved value: the flat-layout guard and the children block.
    assert after.count('"$_MASTER_ROOT_SAVED"') >= 2

    ps_code = "\n".join(l for l in ps.splitlines() if not l.lstrip().startswith("#"))
    ps_saved = ps_code.index("$masterRootToStop = _MasterRoot")
    ps_removal = ps_code.index("_RemoveRootRecordingDb $r")
    assert ps_saved < ps_removal
    ps_after = ps_code[ps_removal:]
    assert "_MasterRoot" not in ps_after, ps_after[: ps_after.index("_MasterRoot") + 200]
    assert ps_after.count("$masterRootToStop") >= 1


def test_neither_uninstaller_sweeps_a_stale_lock_name_it_did_not_make():
    """prebuilt_core.py leaves <name>.stale.<pid>. A dotted glob also matched a user's own
    ".backup.install.lock.stale.copy", and in a root the user chose that file is theirs.
    Behaviour for the POSIX half is in tests/sh/test_uninstall_master_root.sh."""
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")

    sh_code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert ".*.install.lock.stale.*" not in sh_code
    # A numeric pid, checked at both sweeps: the master root and the default home.
    assert sh_code.count("*[!0-9]*) continue ;;") >= 2

    ps_code = "\n".join(l for l in ps.splitlines() if not l.lstrip().startswith("#"))
    assert '-like ".*.install.lock.stale.*"' not in ps_code
    assert "$script:StaleLockPattern" in ps_code
    assert "[0-9]+$" in ps_code


@NEEDS_POSIX_BASH
def test_the_master_root_note_does_not_write_through_a_planted_link(tmp_path):
    """The note is staged under an unpredictable name, so a link left in share/ is not followed.

    The staging name used to be "$notePath.$$", and a shell redirection into an existing symlink
    truncates its target. Anyone who could write share/ -- a portable volume mounted for two
    accounts is the shape that matters, since a master root is what this note records -- could
    therefore point that name at a file of theirs and have setup empty it. The attacker's
    capability is simulated exactly rather than approximated: the link is created by the same
    shell that then runs the shipped block, so "$$" matches the way a guessed pid would.

    setup.sh already states this rule for the uv cache probe ("mktemp (O_EXCL) not $$"); the note
    writer is now held to it too.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "_master_root_note_is_honoured() {", "LLAMA_CPP_DIR=")

    # Contained, so the honoured-note gate passes and the writer below actually runs. The gate
    # itself is covered by test_the_master_root_note_is_only_written_when_a_reader_honours_it.
    master = tmp_path / "portable"
    studio_home = master / "studio"
    (studio_home / "share").mkdir(parents = True)
    victim = tmp_path / "victim.txt"
    victim.write_text("do not truncate me\n", encoding = "utf-8")

    script = "\n".join(
        (
            "set -u",
            # The predictable name the old block would have opened.
            'ln -s "$VICTIM" "$STUDIO_HOME/share/.unsloth-master-root.$$"',
            block,
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "STUDIO_HOME": str(studio_home),
            "UNSLOTH_HOME": str(master),
            "_MASTER_ROOT": str(master),
            "STAGE_ROOT": "",
            "VICTIM": str(victim),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr

    assert victim.read_text(encoding = "utf-8") == "do not truncate me\n"
    note = studio_home / "share" / ".unsloth-master-root"
    assert note.read_text(encoding = "utf-8").strip() == str(master)
    # The rename replaces a link at the final path rather than writing through it.
    assert not note.is_symlink()
    # No staging file survives the run, whatever name it was given.
    leftovers = sorted(
        p.name
        for p in (studio_home / "share").iterdir()
        if p.name.startswith(".unsloth-master-root.") and not p.is_symlink()
    )
    assert leftovers == [], leftovers


def test_the_windows_note_writer_creates_its_staging_file_exclusively():
    """The setup.ps1 half of the test above. Held structurally: the Linux runners have no
    Windows filesystem to plant a link on, and CreateNew is the thing that must not regress."""
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(ps, "if ((Get-MasterRootOverride) -and -not $StageRoot -and", "\n# ")
    # Comments stripped, as the other structural checks here do: the block explains the old
    # staging name in prose, and the name is only a finding when something executes it.
    block = "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("#"))

    assert "$notePath.$PID" not in block, "predictable staging name is back"
    assert "GetRandomFileName()" in block
    assert "[System.IO.FileMode]::CreateNew" in block
    # WriteAllText opens an existing path rather than failing on it.
    assert "WriteAllText($noteTmp" not in block


def _run_note_block(
    tmp_path,
    studio_home,
    master,
    *,
    existing_note = None,
    home = None,
):
    """Run the shipped note gate + writer + legacy sweep against a fixture, and report the note.

    The whole region is executed, not pattern-matched: the gate, the writer it guards and the
    sweep below it are one decision, and slicing them apart is how a gate that never runs passes.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "_master_root_note_is_honoured() {", "LLAMA_CPP_DIR=")
    note = studio_home / "share" / ".unsloth-master-root"
    note.parent.mkdir(parents = True, exist_ok = True)
    if existing_note is not None:
        note.write_text(str(existing_note) + "\n", encoding = "utf-8")
    completed = subprocess.run(
        ["bash", "-c", "set -u\n" + block],
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home if home is not None else tmp_path / "home"),
            "STUDIO_HOME": str(studio_home),
            "UNSLOTH_HOME": str(master),
            "_MASTER_ROOT": str(master),
            "STAGE_ROOT": "",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    value = note.read_text(encoding = "utf-8").strip() if note.exists() else None
    return value, completed.stderr


@NEEDS_POSIX_BASH
def test_the_master_root_note_is_only_written_when_a_reader_honours_it(tmp_path):
    """A note whose Studio tree is not INSIDE the root it names is refused by every reader --
    storage_roots.py, the CLI and both uninstallers all require containment -- so writing one
    records nothing and hides the fact that nothing was recorded.

    This is not a corner. install.sh does not read UNSLOTH_HOME yet, so
    `UNSLOTH_HOME=/mnt/portable unsloth studio update` on an ordinary install is exactly this
    shape: Studio stays at ~/.unsloth/studio while llama.cpp, node and whisper.cpp go to
    /mnt/portable. That is the case the note was added for, and the case in which it was written
    and then thrown away by every reader. Warn instead.
    """
    home = tmp_path / "home"
    studio_home = home / ".unsloth" / "studio"
    master = tmp_path / "portable"
    master.mkdir()
    value, stderr = _run_note_block(tmp_path, studio_home, master, home = home)
    assert value is None, f"wrote a note no reader will honour: {value}"
    assert "cannot be recorded" in stderr, stderr

    # The contained layout, which every reader does honour, still records.
    contained = master / "studio"
    value, _ = _run_note_block(tmp_path, contained, master, home = home)
    assert value == str(master)


@NEEDS_POSIX_BASH
def test_the_legacy_default_root_is_never_recorded_and_an_old_note_is_cleared(tmp_path):
    """`UNSLOTH_HOME=$HOME/.unsloth` names the directory the install is already in. Both
    uninstallers refuse that value outright, so a note carrying it licenses nothing -- but
    storage_roots honours it, and honouring it flips portable_mode() on for every later BARE
    launch, which moves HF_HUB_CACHE off ~/.cache/huggingface. One command that named the default
    root therefore turned a shared model cache into a private one, permanently, with no variable
    set at the time.

    The write is refused, and an existing note carrying that value is cleared on the next run --
    without the sweep the damage survives the fix, since the writer only runs when UNSLOTH_HOME
    is set again and the user who hit this set it once.
    """
    home = tmp_path / "home"
    studio_home = home / ".unsloth" / "studio"
    legacy_master = home / ".unsloth"
    studio_home.mkdir(parents = True)

    # Contained (studio IS inside ~/.unsloth), so only the legacy-root rule can refuse it.
    value, _ = _run_note_block(tmp_path, studio_home, legacy_master, home = home)
    assert value is None, f"recorded the legacy default root: {value}"

    # An install that already carries one from an earlier build is repaired in place.
    value, _ = _run_note_block(
        tmp_path,
        studio_home,
        legacy_master,
        existing_note = legacy_master,
        home = home,
    )
    assert value is None, "a stale legacy-root note survived"

    # A note naming any other root is left alone: it may describe an install this run cannot see.
    other = tmp_path / "portable"
    other.mkdir()
    value, _ = _run_note_block(
        tmp_path,
        studio_home,
        legacy_master,
        existing_note = other,
        home = home,
    )
    assert value == str(other)


def test_the_windows_note_gate_holds_the_same_two_rules():
    """The setup.ps1 half of the two tests above. Structural: no PowerShell-on-Linux run can
    exercise %USERPROFILE% canonicalisation the way a real Windows profile does."""
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    gate = _slice(ps, "function Test-MasterRootNoteIsHonoured {", "\nif ((Get-MasterRootOverride)")
    assert "$legacy = Get-CanonicalDir" in gate, "the legacy default root is not refused"
    assert "StartsWith($norm + $sep" in gate, "containment is not checked"
    # The gate has to be wired into the writer, not merely defined beside it.
    assert "Test-MasterRootNoteIsHonoured -Root $UnslothHome -StudioRoot $StudioHome" in ps
    sweep = _slice(ps, "$staleNote = Join-Path", "} catch {")
    assert "Remove-Item -LiteralPath $staleNote" in sweep


def test_the_windows_note_writer_is_a_no_op_when_the_note_already_says_this():
    """Move-Item -Force is not an atomic replace: PowerShell's provider deletes the destination
    and then moves the source, so an interruption in between leaves NO note, and a backend that
    looks next caches "no master root" from it. Re-entering that window on every update to
    rewrite bytes that are already there is the avoidable half, and the staging file has to go
    whether the move threw or not.

    Run under pwsh rather than asserted from source text: the comparison is the point, and a
    structural check would pass on a version that compares the wrong two things.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, '        $notePath = Join-Path $noteDir ".unsloth-master-root"', "\n    } catch {"
    )
    assert "Remove-Item -LiteralPath $noteTmp" in block, "the staging file is never cleaned up"
    assert "$noteSame" in block, "the note is rewritten even when it already says this"


@pytest.mark.skipif(PWSH is None, reason = "needs pwsh")
def test_the_windows_note_writer_rewrites_only_when_the_value_changed(tmp_path):
    """The behaviour, measured: same value twice leaves the file untouched; a different value
    replaces it; and no staging file survives either way."""
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, '        $notePath = Join-Path $noteDir ".unsloth-master-root"', "\n    } catch {"
    )
    note_dir = tmp_path / "share"
    note_dir.mkdir()
    script = tmp_path / "write_note.ps1"
    script.write_text(
        f"$noteDir = {_ps_quote(str(note_dir))}\n"
        f"$UnslothHome = {_ps_quote(str(tmp_path / 'master'))}\n"
        + block
        + "\n$final = Join-Path $noteDir '.unsloth-master-root'\n"
        "$stamp = (Get-Item -LiteralPath $final).LastWriteTimeUtc.Ticks\n"
        "Write-Output ((Get-Content -LiteralPath $final -Raw) + '|' + $stamp)\n",
        encoding = "utf-8",
    )
    argv = [PWSH, "-NoProfile", "-File", str(script)]
    first = run_pwsh(argv, capture_output = True, text = True, check = True)
    note = note_dir / ".unsloth-master-root"

    # Back-dated so a rewrite is unmistakable: two runs a millisecond apart land on one
    # filesystem tick, which is how the first version of this test passed against the unfixed
    # writer.
    old_stamp = 1_000_000_000
    os.utime(note, (old_stamp, old_stamp))

    second = run_pwsh(argv, capture_output = True, text = True, check = True)

    body_1, _ = first.stdout.strip().rsplit("|", 1)
    body_2, _ = second.stdout.strip().rsplit("|", 1)
    assert body_1 == body_2 == str(tmp_path / "master") + "\n"
    assert (
        int(note.stat().st_mtime) == old_stamp
    ), "an unchanged note was rewritten, re-entering the delete-then-move window"
    leftovers = [p.name for p in note_dir.iterdir() if p.name != ".unsloth-master-root"]
    assert leftovers == [], leftovers

    # And a CHANGED value must still be written, or the check above passes by doing nothing.
    script.write_text(
        script.read_text(encoding = "utf-8").replace(
            _ps_quote(str(tmp_path / "master")), _ps_quote(str(tmp_path / "moved"))
        ),
        encoding = "utf-8",
    )
    run_pwsh(argv, capture_output = True, text = True, check = True)
    assert note.read_text(encoding = "utf-8").strip() == str(tmp_path / "moved")
    assert int(note.stat().st_mtime) != old_stamp
