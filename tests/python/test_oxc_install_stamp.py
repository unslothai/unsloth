# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup skips the OXC validator `npm ci` only when node_modules exactly matches the last stamped install."""

from __future__ import annotations

import sys
import functools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[2]
OXC_DIR = REPO_ROOT / "studio" / "backend" / "core" / "data_recipe" / "oxc-validator"
HELPER = OXC_DIR / "install-stamp.mjs"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
NODE = shutil.which("node")
BASH = shutil.which("bash")
PWSH = shutil.which("pwsh") or shutil.which("powershell")
STAMP = Path("node_modules") / ".unsloth-install-stamp.json"

needs_node = pytest.mark.skipif(NODE is None, reason = "needs node")


@functools.cache
def _host() -> tuple[str, str, str]:
    out = subprocess.run(
        [
            NODE,
            "-e",
            "process.report.excludeNetwork = true; const r = process.report.getReport().header;"
            "console.log(process.platform, process.arch,"
            " process.platform !== 'linux' ? '' : r.glibcVersionRuntime ? 'glibc' : 'musl')",
        ],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.split()
    return out[0], out[1], out[2] if len(out) > 2 else ""


def _installs_here(entry: dict) -> bool:
    plat, arch, libc = _host()

    def allows(value, wanted):
        if not wanted:
            return True
        if f"!{value}" in wanted:
            return False
        plain = [w for w in wanted if not w.startswith("!")]
        return not plain or value in plain

    return (
        allows(plat, entry.get("os"))
        and allows(arch, entry.get("cpu"))
        and (not entry.get("libc") or allows(libc, entry.get("libc")))
    )


def _template(root: Path) -> Path:
    """What `npm ci` leaves for the committed lockfile, minus the real binaries."""
    lock = json.loads((OXC_DIR / "package-lock.json").read_text(encoding = "utf-8"))
    wanted = {
        key: entry
        for key, entry in lock["packages"].items()
        if key.startswith("node_modules/") and _installs_here(entry)
    }
    modules = root / "node_modules"
    for key, entry in wanted.items():
        pkg = root / key
        pkg.mkdir(parents = True)
        name = key[len("node_modules/") :]
        (pkg / "package.json").write_text(
            json.dumps({"name": name, "version": entry["version"]}), encoding = "utf-8"
        )
        (pkg / "binding.node").write_bytes(name.encode() * 64)
    (modules / "oxlint" / "AGENTS.md").write_text("pruned by setup\n", encoding = "utf-8")
    (modules / ".package-lock.json").write_text(
        json.dumps({"name": lock["name"], "lockfileVersion": 3, "packages": wanted}),
        encoding = "utf-8",
    )
    return modules


def _oxc(tmp_path: Path) -> Path:
    oxc = tmp_path / "studio" / "backend" / "core" / "data_recipe" / "oxc-validator"
    oxc.mkdir(parents = True)
    for name in ("package.json", "package-lock.json", "install-stamp.mjs"):
        if (OXC_DIR / name).exists():
            shutil.copy2(OXC_DIR / name, oxc / name)
    return oxc


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def _helper(
    oxc: Path,
    mode: str,
    env: dict | None = None,
) -> int:
    return subprocess.run(
        [NODE, str(oxc / "install-stamp.mjs"), mode],
        cwd = str(oxc.parent),
        env = env,
        capture_output = True,
    ).returncode


@needs_node
def test_helper_writes_and_confirms_an_unchanged_tree(tmp_path):
    oxc = _oxc(tmp_path)
    assert _helper(oxc, "check") == 1
    _template(oxc)
    assert _helper(oxc, "check") == 1
    assert _helper(oxc, "write") == 0
    before = _snapshot(oxc)
    assert _helper(oxc, "check") == 0
    assert _helper(oxc, "check") == 0
    assert _snapshot(oxc) == before
    # Setup prunes agent files after install; still the same tree.
    (oxc / "node_modules" / "oxlint" / "AGENTS.md").unlink()
    assert _helper(oxc, "check") == 0


def _tamper_byte(oxc):
    p = oxc / "node_modules" / "oxlint" / "binding.node"
    data = bytearray(p.read_bytes())
    data[3] ^= 1
    p.write_bytes(bytes(data))


def _stamp_field(field, value):
    def edit(oxc):
        p = oxc / STAMP
        doc = json.loads(p.read_text(encoding = "utf-8"))
        doc[field] = value
        p.write_text(json.dumps(doc), encoding = "utf-8")

    return edit


TAMPERS = {
    "binary byte flipped": _tamper_byte,
    "binary deleted": lambda oxc: (oxc / "node_modules" / "oxlint" / "binding.node").unlink(),
    "package dir deleted": lambda oxc: shutil.rmtree(oxc / "node_modules" / "oxc-parser"),
    "extra file": lambda oxc: (oxc / "node_modules" / "oxlint" / "extra.js").write_text("x"),
    "hidden lockfile deleted": lambda oxc: (oxc / "node_modules" / ".package-lock.json").unlink(),
    "hidden lockfile edited": lambda oxc: (oxc / "node_modules" / ".package-lock.json").write_text(
        (oxc / "node_modules" / ".package-lock.json").read_text() + " "
    ),
    "lockfile changed": lambda oxc: (oxc / "package-lock.json").write_text(
        (oxc / "package-lock.json").read_text() + "\n"
    ),
    "package.json changed": lambda oxc: (oxc / "package.json").write_text(
        (oxc / "package.json").read_text() + "\n"
    ),
    "stamp truncated": lambda oxc: (oxc / STAMP).write_text('{"version": 1, "node"'),
    "stamp empty": lambda oxc: (oxc / STAMP).write_text(""),
    "stamp deleted": lambda oxc: (oxc / STAMP).unlink(),
    "stamp from another Node": _stamp_field("node", "v20.19.0"),
    "stamp from another arch": _stamp_field("arch", "ia32"),
    "stamp from a future format": _stamp_field("version", 2),
    "user npmrc changed": lambda oxc: (oxc.parent / "home" / ".npmrc").write_text(
        "omit=optional\n"
    ),
    "project npmrc added": lambda oxc: (oxc / ".npmrc").write_text("omit=optional\n"),
}


@needs_node
@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_helper_fails_open_on_any_doubt(tmp_path, name):
    oxc = _oxc(tmp_path)
    home = oxc.parent / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    env = {k: v for k, v in env.items() if not k.lower().startswith("npm_config_")}
    _template(oxc)
    assert _helper(oxc, "write", env) == 0
    assert _helper(oxc, "check", env) == 0
    TAMPERS[name](oxc)
    assert _helper(oxc, "check", env) == 1


@needs_node
@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX exec bits")
def test_helper_fails_open_when_an_exec_bit_is_lost(tmp_path):
    oxc = _oxc(tmp_path)
    _template(oxc)
    binary = oxc / "node_modules" / "oxlint" / "binding.node"
    binary.chmod(0o755)
    assert _helper(oxc, "write") == 0
    assert _helper(oxc, "check") == 0
    binary.chmod(0o644)
    assert _helper(oxc, "check") == 1


@needs_node
def test_helper_refuses_a_tree_missing_this_platforms_native_binding(tmp_path):
    oxc = _oxc(tmp_path)
    modules = _template(oxc)
    hidden = json.loads((modules / ".package-lock.json").read_text(encoding = "utf-8"))
    bindings = [k for k, e in hidden["packages"].items() if e.get("optional")]
    assert bindings, "host has no native binding in the lockfile"
    del hidden["packages"][bindings[0]]
    shutil.rmtree(oxc / bindings[0])
    (modules / ".package-lock.json").write_text(json.dumps(hidden), encoding = "utf-8")
    assert _helper(oxc, "write") == 1
    assert not (oxc / STAMP).exists()


@needs_node
def test_stamp_never_stores_npm_config_values(tmp_path):
    oxc = _oxc(tmp_path)
    _template(oxc)
    secret = "http://user:hunter2@proxy.invalid:8080"
    assert _helper(oxc, "write", {**os.environ, "npm_config_https_proxy": secret}) == 0
    assert "hunter2" not in (oxc / STAMP).read_text(encoding = "utf-8")


@needs_node
def test_helper_fails_open_when_npm_settings_could_change_the_tree(tmp_path):
    oxc = _oxc(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith("npm_config_")}
    _template(oxc)
    assert _helper(oxc, "write", env) == 0
    assert _helper(oxc, "check", {**env, "npm_config_registry": "https://mirror.invalid/"}) == 0
    assert _helper(oxc, "check", {**env, "NPM_CONFIG_PREFIX": str(tmp_path)}) == 0
    assert _helper(oxc, "check", {**env, "npm_config_omit": "optional"}) == 1
    assert _helper(oxc, "check", {**env, "NPM_CONFIG_CPU": "arm64"}) == 1
    globalrc = tmp_path / "globalrc"
    genv = {**env, "npm_config_globalconfig": str(globalrc)}
    assert _helper(oxc, "write", genv) == 0
    assert _helper(oxc, "check", genv) == 0
    globalrc.write_text("omit=optional\n", encoding = "utf-8")
    assert _helper(oxc, "check", genv) == 1


@needs_node
def test_helper_refuses_to_stamp_an_incomplete_tree(tmp_path):
    oxc = _oxc(tmp_path)
    _template(oxc)
    shutil.rmtree(oxc / "node_modules" / "oxlint")
    assert _helper(oxc, "write") == 1
    assert not (oxc / STAMP).exists()
    assert _helper(oxc, "check") == 1


@needs_node
def test_the_tree_check_covers_the_real_lockfile_dependencies():
    declared = json.loads((OXC_DIR / "package.json").read_text(encoding = "utf-8"))["dependencies"]
    assert {"oxlint", "oxc-parser"} <= set(declared)
    lock = json.loads((OXC_DIR / "package-lock.json").read_text(encoding = "utf-8"))
    # No install scripts: the tree npm ci builds is fixed by the lockfile and the platform alone.
    assert not [k for k, v in lock["packages"].items() if v.get("hasInstallScript")]


def _bash_block() -> str:
    text = SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(r"^_install_oxc_runtime\(\) \{\n.*?^\}\n", text, re.M | re.S)
    assert m, "no _install_oxc_runtime in setup.sh"
    return m.group(0)


def _ps1_block() -> str:
    lines = SETUP_PS1.read_text(encoding = "utf-8").splitlines()
    parts = []
    if "function Invoke-OxcInstallStamp {" in lines:
        start = lines.index("function Invoke-OxcInstallStamp {")
        end = next(i for i in range(start, len(lines)) if lines[i] == "}")
        parts.append("\n".join(lines[start : end + 1]))
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("if ((Test-Path $OxcValidatorDir) -and $NodeSource -ne")
    )
    end = next(
        i for i in range(start, len(lines)) if lines[i].startswith("Remove-AgentInstructionFiles")
    )
    parts.append("\n".join(lines[start:end]))
    return "\n".join(parts)


def _fake_npm(bindir: Path, template: Path, calls: Path, fail_file: Path) -> None:
    """npm stub: like `npm ci`, wipe node_modules and lay the locked tree down; or fail half way."""
    bindir.mkdir()
    npm = bindir / "npm"
    npm.write_text(
        f"#!{BASH}\n"
        f'echo "$1" >> "{calls}"\n'
        "rm -rf node_modules\n"
        f'if [ -f "{fail_file}" ]; then mkdir -p node_modules/oxlint; exit 1; fi\n'
        f'cp -R "{template}" node_modules\n',
        encoding = "utf-8",
    )
    npm.chmod(0o755)
    (bindir / "node").symlink_to(NODE)


def _run_block(shell: str, tmp_path: Path, oxc: Path, bindir: Path) -> subprocess.CompletedProcess:
    env = {"PATH": f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin", "HOME": str(tmp_path / "home")}
    if shell == "bash":
        script = "\n".join(
            [
                "set -euo pipefail",
                'verbose_substep() { echo "verbose: $1"; }',
                'substep() { echo "substep: $1"; }',
                'run_quiet_no_exit() { local label=$1; shift; "$@"; }',
                "_npm_mirror_retry() { return 1; }",
                "_suggest_npm_registry() { :; }",
                'setup_fail() { echo "FAILED: $2"; exit "$1"; }',
                f'SCRIPT_DIR="{oxc.parents[3]}"',
                f'_OXC_DIR="{oxc}"',
                "NODE_SOURCE=bundled",
                "_NPM_REGISTRY_ARGS=()",
                _bash_block(),
                "_install_oxc_runtime",
                'echo "done in $PWD"',
            ]
        )
        return subprocess.run([BASH, "-c", script], env = env, capture_output = True, text = True)
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            'function step { param($a, $b) Write-Output "step: $a $b" }',
            'function substep { param($a, $b) Write-Output "substep: $a" }',
            "function Write-StudioLine { param($a) Write-Output $a }",
            "function Invoke-SetupCommand { param([scriptblock]$Command) & $Command | Out-Null; return $LASTEXITCODE }",
            "function Invoke-NpmMirrorRetry { return $false }",
            "function Show-NpmRegistryHint { param($FailureOutput) }",
            'function Exit-SetupFailure { param($m) Write-Output "FAILED: $m"; exit 1 }',
            f"$OxcValidatorDir = '{oxc}'",
            "$NodeSource = 'bundled'",
            "$NpmRegistryArgs = @()",
            f"Set-Location '{tmp_path}'",
            _ps1_block(),
            '"done in $((Get-Location).Path)"',
        ]
    )
    return run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        env = env,
        capture_output = True,
        text = True,
    )


SHELLS = [
    pytest.param(
        "bash", marks = pytest.mark.skipif(BASH is None or os.name == "nt", reason = "needs bash")
    ),
    pytest.param(
        "pwsh",
        marks = pytest.mark.skipif(
            PWSH is None or BASH is None or os.name == "nt", reason = "needs pwsh"
        ),
    ),
]


@needs_node
@pytest.mark.parametrize("shell", SHELLS)
def test_setup_runs_npm_ci_once_then_skips_until_the_tree_changes(tmp_path, shell):
    oxc = _oxc(tmp_path)
    (tmp_path / "home").mkdir()
    template = _template(tmp_path / "template")
    calls = tmp_path / "npm_calls"
    fail = tmp_path / "npm_fails"
    bindir = tmp_path / "bin"
    _fake_npm(bindir, template, calls, fail)

    def run():
        result = _run_block(shell, tmp_path, oxc, bindir)
        n = len(calls.read_text().splitlines()) if calls.exists() else 0
        return result, n

    result, n = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert n == 1
    if shell == "pwsh":
        assert "step: oxc runtime installed" in result.stdout
    installed = _snapshot(oxc)

    # The re-run (or a no-op update) leaves the same tree alone and ends where it did before.
    result, n = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert n == 1, "setup reinstalled an unchanged OXC tree"
    assert (oxc / STAMP).is_file()
    assert _snapshot(oxc) == installed
    assert result.stdout.strip().splitlines()[-1] == (
        f"done in {oxc.parents[3]}" if shell == "bash" else f"done in {tmp_path}"
    )
    if shell == "pwsh":
        assert "step: oxc runtime up to date" in result.stdout

    # A damaged tree is reinstalled, and the next run trusts the fresh stamp again.
    _tamper_byte(oxc)
    result, n = run()
    assert result.returncode == 0 and n == 2
    assert _snapshot(oxc) == installed
    result, n = run()
    assert result.returncode == 0 and n == 2

    # npm ci dying half way leaves no stamp, so the run after it reinstalls.
    _tamper_byte(oxc)
    fail.touch()
    result, n = run()
    assert result.returncode != 0 and n == 3
    assert "FAILED: OXC validator dependency installation failed" in result.stdout
    assert not (oxc / STAMP).exists()
    fail.unlink()
    result, n = run()
    assert result.returncode == 0 and n == 4
    assert _snapshot(oxc) == installed


@needs_node
@pytest.mark.parametrize("shell", SHELLS)
def test_setup_never_skips_npm_install_without_a_lockfile(tmp_path, shell):
    oxc = _oxc(tmp_path)
    (tmp_path / "home").mkdir()
    (oxc / "package-lock.json").unlink()
    template = _template(tmp_path / "template")
    calls = tmp_path / "npm_calls"
    bindir = tmp_path / "bin"
    _fake_npm(bindir, template, calls, tmp_path / "npm_fails")
    for _ in range(2):
        result = _run_block(shell, tmp_path, oxc, bindir)
        assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().split() == ["install", "install"]
    assert not (oxc / STAMP).exists()


@needs_node
@pytest.mark.parametrize("shell", SHELLS)
def test_setup_without_the_helper_runs_npm_ci_as_before(tmp_path, shell):
    oxc = _oxc(tmp_path)
    (tmp_path / "home").mkdir()
    template = _template(tmp_path / "template")
    calls = tmp_path / "npm_calls"
    bindir = tmp_path / "bin"
    _fake_npm(bindir, template, calls, tmp_path / "npm_fails")
    assert _run_block(shell, tmp_path, oxc, bindir).returncode == 0
    (oxc / "install-stamp.mjs").unlink(missing_ok = True)
    assert _run_block(shell, tmp_path, oxc, bindir).returncode == 0
    assert calls.read_text().split() == ["ci", "ci"]
