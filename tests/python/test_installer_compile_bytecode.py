# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup precompiles the Studio environment's bytecode after a dependency pass, and it can never
fail an install.

uv installs without .pyc, so the first launch paid for compiling every module `import main`
loads. studio/precompile_bytecode.py runs compileall over site-packages (and an editable checkout)
after setup.sh / setup.ps1 changed the venv. uv itself is left alone: no installer sets
UV_COMPILE_BYTECODE, because a uv compile failure or timeout fails the install command (#5983).

On Windows it is opt-in (UNSLOTH_STUDIO_COMPILE_BYTECODE=1): Defender scans every .pyc the update
writes, which made updates slower than Python's lazy compile on first import.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

from unsloth_pwsh_runner import run_pwsh

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
HELPER = REPO_ROOT / "studio" / "precompile_bytecode.py"
BASH = shutil.which("bash")
PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding = "utf-8").splitlines()


def _index(
    lines: list[str],
    pattern: str,
    start: int = 0,
) -> int:
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]) and not lines[i].lstrip().startswith("#"):
            return i
    raise AssertionError(f"no line matches {pattern!r}")


def _helper():
    spec = importlib.util.spec_from_file_location("precompile_bytecode_under_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_env(monkeypatch):
    for name in (
        "UV_COMPILE_BYTECODE",
        "UNSLOTH_STUDIO_COMPILE_BYTECODE",
        "UNSLOTH_STUDIO_COMPILE_BYTECODE_TIMEOUT",
        "UNSLOTH_VERBOSE",
        "SOURCE_DATE_EPOCH",
    ):
        monkeypatch.delenv(name, raising = False)
    # The helper only compiles with the opt-in.
    monkeypatch.setenv("UNSLOTH_STUDIO_COMPILE_BYTECODE", "1")


def _tree(root: Path) -> Path:
    pkg = root / "pkg"
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    (pkg / "mod.py").write_text("def f():\n    return 2\n", encoding = "utf-8")
    return pkg


def _pycs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pyc"))


def _snapshot(root: Path) -> dict:
    """Every path under root, and each file's size and mtime: equal snapshots = nothing changed.
    A directory is compared by its entries, not its mtime, which any create-then-delete moves."""
    return {
        str(p.relative_to(root)): ("dir",)
        if p.is_dir()
        else (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


def test_setup_sh_marks_the_deps_pass_and_compiles_before_the_footer():
    lines = _lines(SETUP_SH)
    stack = _index(lines, r"^    install_python_stack$")
    # The handoff stays directly behind the dependency pass (tests/sh/test_setup_rerun_after_replace.sh);
    # the flag follows it inside the same branch, and a replaced copy compiles on UNSLOTH_SETUP_RERUN.
    assert lines[stack + 1] == "    _setup_rerun_if_replaced"
    flag = _index(lines, r"^    _SETUP_VENV_CHANGED=true$", stack)
    assert flag <= stack + 3 and lines[flag - 1].lstrip().startswith(
        ("#", "_setup_rerun_if_replaced")
    )
    assert _index(lines, r"^_SETUP_VENV_CHANGED=false$") < stack
    block = _index(lines, r"_SETUP_VENV_CHANGED:-false")
    call = _index(lines, r'"\$VENV_DIR/bin/python" -I "\$SCRIPT_DIR/precompile_bytecode.py"', block)
    assert lines[call].rstrip().endswith("|| true")
    assert "UNSLOTH_SETUP_RERUN" in lines[block]
    assert '"${_COLAB_NO_VENV:-false}" != true' in lines[block + 1]
    # After every install phase and the frontend join, before the footer.
    assert _index(lines, r"^_frontend_bg_join$", stack) < block
    assert call < next(i for i, line in enumerate(lines) if line.startswith("# ── Footer ──"))
    assert block > _index(lines, r"install_audio_cpp_prebuilt\.py")


def test_setup_ps1_marks_the_deps_pass_and_compiles_before_the_footer():
    lines = _lines(SETUP_PS1)
    assert _index(lines, r"^\$script:SetupVenvChanged = \$false$") < 200
    stack = _index(lines, r'^python "\$PSScriptRoot\\install_python_stack\.py"$')
    assert lines[stack + 1] == "$stackExit = $LASTEXITCODE"
    assert lines[stack + 2] == "$script:SetupVenvChanged = $true"
    block = _index(lines, r"\$script:SetupVenvChanged -or \$env:UNSLOTH_SETUP_RERUN -eq '1'")
    assert block > stack
    footer = next(i for i, line in enumerate(lines) if line == "# Footer")
    assert block < footer
    body = "\n".join(lines[block:footer])
    assert "$ErrorActionPreference = 'Continue'" in body
    assert "catch { }" in body
    assert "$_pycPrevExit = $global:LASTEXITCODE" in body
    assert "$global:LASTEXITCODE = $_pycPrevExit" in body


def test_uv_bytecode_behaviour_is_left_alone():
    for path in (INSTALL_SH, INSTALL_PS1, SETUP_SH, SETUP_PS1):
        text = path.read_text(encoding = "utf-8")
        assert "UV_COMPILE_BYTECODE:=" not in text, path
        assert "$env:UV_COMPILE_BYTECODE =" not in text, path
        assert "export UV_COMPILE_BYTECODE\n" not in text, path
        assert "--compile-bytecode" not in text, path


def test_no_installer_disables_pip_compile():
    for path in (
        INSTALL_SH,
        INSTALL_PS1,
        SETUP_SH,
        SETUP_PS1,
        REPO_ROOT / "studio" / "install_python_stack.py",
    ):
        text = path.read_text(encoding = "utf-8")
        assert "--no-compile" not in text, path
        assert "PIP_NO_COMPILE" not in text, path


def _setup_sh_block() -> str:
    lines = _lines(SETUP_SH)
    start = _index(lines, r"^_setup_bytecode_opted_out\(\) \{$")
    gate = _index(lines, r"_SETUP_VENV_CHANGED:-false", start)
    end = next(i for i in range(gate, len(lines)) if lines[i] == "fi")
    return "\n".join(lines[start : end + 1])


@pytest.mark.skipif(BASH is None, reason = "needs bash")
@pytest.mark.parametrize(
    "changed, rerun, opt_out, fake_exit, expect_call",
    [
        # Off by default on every OS: compiling the whole venv each update cost more than it saved.
        ("true", "", None, 1, False),
        ("false", "1", None, 1, False),
        ("true", "", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "1"), 1, True),
        ("true", "", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "1"), 0, True),
        ("false", "1", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", " On "), 1, True),
        ("false", "", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "1"), 0, False),
        ("true", "", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "maybe"), 0, False),
        ("true", "", ("UV_COMPILE_BYTECODE", " FALSE "), 0, False),
        ("true", "", ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "0"), 0, False),
        ("true", "", ("UV_COMPILE_BYTECODE", "1"), 0, True),
    ],
)
def test_setup_sh_block_never_fails_and_runs_only_after_a_deps_pass(
    tmp_path, changed, rerun, opt_out, fake_exit, expect_call
):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents = True)
    marker = tmp_path / "called"
    fake = venv / "bin" / "python"
    fake.write_text(
        f'#!/bin/sh\n: > "{marker}"\necho noisy >&2\nexit {fake_exit}\n', encoding = "utf-8"
    )
    fake.chmod(0o755)
    script_dir = tmp_path / "studio"
    script_dir.mkdir()
    (script_dir / "precompile_bytecode.py").write_text("", encoding = "utf-8")
    script = "\n".join(
        [
            "set -euo pipefail",
            'step() { printf "%s %s\\n" "$1" "$2"; }',
            f'VENV_DIR="{venv}"',
            f'SCRIPT_DIR="{script_dir}"',
            "_COLAB_NO_VENV=false",
            f"_SETUP_VENV_CHANGED={changed}",
            _setup_sh_block(),
            "echo reached-the-footer",
        ]
    )
    env = {"PATH": "/usr/bin:/bin"}
    if rerun:
        env["UNSLOTH_SETUP_RERUN"] = rerun
    if opt_out:
        env[opt_out[0]] = opt_out[1]
    result = subprocess.run([BASH, "-c", script], env = env, capture_output = True, text = True)
    assert result.returncode == 0, result.stderr
    expected = (["python precompiling bytecode..."] if expect_call else []) + ["reached-the-footer"]
    assert result.stdout.splitlines() == expected
    assert marker.exists() is expect_call


@pytest.mark.skipif(PWSH is None or BASH is None, reason = "needs PowerShell")
@pytest.mark.parametrize(
    "switches, expect_call",
    [
        # Off by default: Defender scans every .pyc the update writes.
        ({}, False),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "1"}, True),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": " On "}, True),
        ({"UV_COMPILE_BYTECODE": "true"}, True),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "maybe"}, False),
        ({"UV_COMPILE_BYTECODE": "False"}, False),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "off"}, False),
        # A false always wins over a true.
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "1", "UV_COMPILE_BYTECODE": "0"}, False),
    ],
)
def test_setup_ps1_block_never_fails(tmp_path, switches, expect_call):
    lines = _lines(SETUP_PS1)
    start = _index(lines, r"^\$_pycPython = ")
    footer = next(i for i, line in enumerate(lines) if line == "# Footer")
    block = "\n".join(lines[start : footer - 1])
    venv = tmp_path / "venv"
    marker = tmp_path / "called"
    studio = tmp_path / "studio"
    studio.mkdir()
    if os.name == "nt":
        # Windows runs only real executables: a real venv python, and a helper that marks, is noisy and exits 7.
        import venv as venv_module
        venv_module.EnvBuilder(with_pip = False).create(venv)
        (studio / "precompile_bytecode.py").write_text(
            f"import pathlib, sys\npathlib.Path(r'{marker}').touch()\n"
            "sys.stderr.write('noisy\\n')\nsys.exit(7)\n",
            encoding = "utf-8",
        )
    else:
        (venv / "Scripts").mkdir(parents = True)
        fake = venv / "Scripts" / "python.exe"
        fake.write_text(f'#!{BASH}\n: > "{marker}"\necho noisy >&2\nexit 7\n', encoding = "utf-8")
        fake.chmod(0o755)
        (studio / "precompile_bytecode.py").write_text("", encoding = "utf-8")
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            'function step { param($a, $b) Write-Output "$a $b" }',
            f"$VenvDir = '{venv}'",
            f"$PSScriptRoot = '{studio}'",
            "$script:SetupVenvChanged = $true",
            "$global:LASTEXITCODE = 5",
            block,
            '"reached-the-footer exit=$LASTEXITCODE"',
        ]
    )
    if os.name == "nt":
        # PowerShell on Windows needs SystemRoot and friends; only the switches under test are cleared.
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in ("UV_COMPILE_BYTECODE", "UNSLOTH_STUDIO_COMPILE_BYTECODE", "UNSLOTH_SETUP_RERUN")
        }
    else:
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    env.update(switches)
    result = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        env = env,
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, result.stderr
    # $LASTEXITCODE is left as the step found it, whatever the helper returned (the fake exits 7).
    expected = (["python precompiling bytecode..."] if expect_call else []) + [
        "reached-the-footer exit=5"
    ]
    assert [line for line in result.stdout.splitlines() if line.strip()] == expected
    assert marker.exists() is expect_call


def test_helper_is_off_by_default(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.delenv("UNSLOTH_STUDIO_COMPILE_BYTECODE", raising = False)
    helper = _helper()
    _tree(tmp_path)
    before = _snapshot(tmp_path)
    assert helper.main([str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before, "no OS may precompile without the opt-in"


@pytest.mark.parametrize(
    "switches, compiles",
    [
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "1"}, True),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "TRUE"}, True),
        ({"UV_COMPILE_BYTECODE": "yes"}, True),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": ""}, False),
        ({"UNSLOTH_STUDIO_COMPILE_BYTECODE": "1", "UV_COMPILE_BYTECODE": "0"}, False),
    ],
)
def test_helper_opt_in(tmp_path, monkeypatch, switches, compiles):
    _clean_env(monkeypatch)
    monkeypatch.delenv("UNSLOTH_STUDIO_COMPILE_BYTECODE", raising = False)
    for name, value in switches.items():
        monkeypatch.setenv(name, value)
    helper = _helper()
    pkg = _tree(tmp_path)
    assert helper.main([str(tmp_path)]) == 0
    assert len(_pycs(pkg)) == (2 if compiles else 0)


def test_helper_compiles_a_tree_and_is_idempotent(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    pkg = _tree(tmp_path)
    assert _helper().main([str(tmp_path)]) == 0
    assert len(_pycs(pkg)) == 2
    before = _snapshot(tmp_path)
    time.sleep(0.05)
    assert _helper().main([str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before, "a re-run over a compiled tree must write nothing"
    # The installed interpreter's view: the .pyc are current, so an import writes nothing either.
    subprocess.run([sys.executable, "-c", "import pkg.mod"], cwd = tmp_path, check = True)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "name, value",
    [
        ("UV_COMPILE_BYTECODE", "0"),
        ("UV_COMPILE_BYTECODE", "false"),
        ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "0"),
        ("UNSLOTH_STUDIO_COMPILE_BYTECODE", "off"),
    ],
)
def test_helper_is_skipped_when_disabled(tmp_path, monkeypatch, name, value):
    _clean_env(monkeypatch)
    monkeypatch.setenv(name, value)
    _tree(tmp_path)
    assert _helper().main([str(tmp_path)]) == 0
    assert _pycs(tmp_path) == []


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason = "root ignores the read-only bit",
)
def test_helper_succeeds_on_a_read_only_tree(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    pkg = _tree(tmp_path)
    before = _snapshot(tmp_path)
    pkg.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = subprocess.run(
            [sys.executable, str(HELPER), str(tmp_path)], capture_output = True, text = True
        )
    finally:
        pkg.chmod(0o755)
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert _snapshot(tmp_path) == before


def test_helper_succeeds_when_compileall_exits_1(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    pkg = _tree(tmp_path)
    (pkg / "broken.py").write_text("def (:\n", encoding = "utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(tmp_path)], capture_output = True
    )
    assert result.returncode == 1, "the control must really fail"
    for name in _pycs(tmp_path):
        name.unlink()
    (pkg / "warns.py").write_text('X = "\\_"\n', encoding = "utf-8")  # SyntaxWarning on 3.12+
    result = subprocess.run(
        [sys.executable, str(HELPER), str(tmp_path)],
        capture_output = True,
        text = True,
        env = {**os.environ, "UNSLOTH_VERBOSE": "1"},
    )
    assert result.returncode == 0
    # Silent in verbose mode too: setup's progress line is the only output.
    assert result.stdout == "" and result.stderr == ""
    assert len(_pycs(pkg)) == 3


def test_helper_survives_a_failing_or_missing_command(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    _tree(tmp_path)
    before = _snapshot(tmp_path)
    helper = _helper()
    for child in ([sys.executable, "-c", "raise SystemExit(1)"], ["/nonexistent/python-for-test"]):
        monkeypatch.setattr(helper, "_child_command", lambda site, checkout, child = child: child)
        assert helper.main([str(tmp_path)]) == 0
        assert _snapshot(tmp_path) == before
    monkeypatch.setattr(helper, "plan", lambda site, checkout: 1 / 0)
    assert helper.main([str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before


def _slow_compile(tree: Path, pidfile: Path) -> str:
    """Writes what a compile cut short leaves behind (a finished .pyc, a temp file, a new
    __pycache__ dir), then hangs."""
    target = importlib.util.cache_from_source(str(tree / "pkg" / "mod.py"))
    fresh = importlib.util.cache_from_source(str(tree / "other" / "x.py"))
    return (
        "import os, time\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        f"os.makedirs(os.path.dirname({target!r}), exist_ok=True)\n"
        f"open({target!r}, 'wb').write(b'x')\n"
        f"open({target!r} + '.140234', 'wb').write(b'x')\n"
        f"os.makedirs(os.path.dirname({fresh!r}), exist_ok=True)\n"
        f"open({fresh!r} + '.99', 'wb').write(b'x')\n"
        "time.sleep(120)\n"
    )


def test_helper_timeout_bounds_the_compile_and_leaves_the_tree_as_it_was(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("UNSLOTH_STUDIO_COMPILE_BYTECODE_TIMEOUT", "1")
    tree = tmp_path / "site"
    pkg = _tree(tree)
    (pkg / "__pycache__").mkdir()
    (tree / "other").mkdir()
    (tree / "other" / "x.py").write_text("Y = 1\n", encoding = "utf-8")
    old = Path(importlib.util.cache_from_source(str(pkg / "__init__.py")))
    old.write_bytes(b"from an earlier run")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    before = _snapshot(tree)
    pidfile = tmp_path / "pid"
    helper = _helper()
    monkeypatch.setattr(
        helper,
        "_child_command",
        lambda site, checkout: [sys.executable, "-c", _slow_compile(tree, pidfile)],
    )
    started = time.monotonic()
    assert helper.main([str(tree)]) == 0
    assert time.monotonic() - started < 30
    pid = int(pidfile.read_text())
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert _snapshot(tree) == before


@pytest.mark.skipif(os.name == "nt", reason = "POSIX signal semantics")
@pytest.mark.parametrize("signame", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_a_stop_signal_kills_the_compile_and_undoes_it(tmp_path, signame):
    """bash only stops on a child's Ctrl+C when the child dies of SIGINT, so the helper must; the
    desktop's Stop sends SIGTERM to setup's group, which the compile's own session does not get."""
    signum = getattr(__import__("signal"), signame)
    tree = tmp_path / "site"
    _tree(tree)
    (tree / "other").mkdir()
    (tree / "other" / "x.py").write_text("Y = 1\n", encoding = "utf-8")
    before = _snapshot(tree)
    pidfile = tmp_path / "pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('h', {str(HELPER)!r})\n"
        "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
        f"h._child_command = lambda s, c: [sys.executable, '-c', {_slow_compile(tree, pidfile)!r}]\n"
        f"sys.exit(h.main([{str(tree)!r}]))\n",
        encoding = "utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        env = {**os.environ, "UNSLOTH_STUDIO_COMPILE_BYTECODE": "1"},
    )
    deadline = time.monotonic() + 30
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    proc.send_signal(signum)
    out, err = proc.communicate(timeout = 30)
    assert proc.returncode == -signum
    assert out == b"" and err == b""
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert _snapshot(tree) == before


def test_helper_skips_when_disk_is_low(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    helper = _helper()
    _tree(tmp_path)
    before = _snapshot(tmp_path)

    class Usage:
        free = helper._MIN_FREE_BYTES - 1

    monkeypatch.setattr(helper.shutil, "disk_usage", lambda path: Usage())
    assert helper.main([str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before


def test_default_timeout_stays_under_the_desktop_silence_report():
    """install_watchdog.rs reports "still waiting" after REPORT_INTERVAL of silence: the one
    progress line must stay the only output, so the default bound has to end the step first."""
    rust = (REPO_ROOT / "studio" / "src-tauri" / "src" / "install_watchdog.rs").read_text(
        encoding = "utf-8"
    )
    match = re.search(r"REPORT_INTERVAL: Duration = Duration::from_secs\(([0-9 *]+)\)", rust)
    assert match, "REPORT_INTERVAL moved"
    report_interval = eval(match.group(1), {"__builtins__": {}})
    assert _helper()._DEFAULT_TIMEOUT + 30 <= report_interval


def test_helper_default_dirs_add_an_editable_checkout_once(tmp_path, monkeypatch):
    helper = _helper()
    site = tmp_path / "site-packages"
    (site / "unsloth").mkdir(parents = True)
    checkout = tmp_path / "checkout" / "studio"
    checkout.mkdir(parents = True)
    (checkout / "__init__.py").write_text("", encoding = "utf-8")
    monkeypatch.setattr(
        helper.sysconfig, "get_paths", lambda: {"purelib": str(site), "platlib": str(site)}
    )

    class Spec:
        def __init__(self, path):
            self.submodule_search_locations = [path]

    locations = {"studio": str(checkout), "unsloth": str(site / "unsloth"), "unsloth_cli": None}
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: Spec(locations[name]) if locations.get(name) else None,
    )
    got_site, got_checkout = helper.default_dirs()
    assert got_site == [str(site)]
    assert got_checkout == [str(checkout)]

    def src(path):
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text("Z = 1\n", encoding = "utf-8")
        return str(path)

    want = [
        str(checkout / "__init__.py"),
        src(site / "unsloth" / "a.py"),
        src(site / "pkg" / "tests" / "test_kept_in_site.py"),
        src(checkout / "top.py"),
        src(checkout / "backend" / "main.py"),
    ]
    compiled = src(site / "unsloth" / "done.py")
    Path(importlib.util.cache_from_source(compiled)).parent.mkdir(exist_ok = True)
    Path(importlib.util.cache_from_source(compiled)).write_bytes(b"stale or not, never replaced")
    for skipped in (
        "frontend/node_modules/x.py",
        "src-tauri/y.py",
        "tests/test_z.py",
        "backend/tests/test_w.py",
        ".git/hooks/h.py",
    ):
        src(checkout / skipped)
    sources, targets, new_dirs = helper.plan(got_site, got_checkout)
    assert sorted(sources) == sorted(want)
    assert targets == [importlib.util.cache_from_source(s) for s in sources]
    assert new_dirs == {os.path.dirname(t) for t in targets} - {
        os.path.dirname(importlib.util.cache_from_source(compiled))
    }


def test_helper_ignores_pythonpath_and_namespace_dirs(tmp_path, monkeypatch):
    """A PYTHONPATH tree or a bare namespace dir is not the app's code; a .pth-added checkout is."""
    helper = _helper()
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(helper.sysconfig, "get_paths", lambda: {"purelib": str(site)})
    on_pythonpath = tmp_path / "pp"
    (on_pythonpath / "ib_fake_a").mkdir(parents = True)
    (on_pythonpath / "ib_fake_a" / "__init__.py").write_text("", encoding = "utf-8")
    editable = tmp_path / "checkout"
    (editable / "ib_fake_b").mkdir(parents = True)
    (editable / "ib_fake_b" / "__init__.py").write_text("", encoding = "utf-8")
    (editable / "ib_fake_c").mkdir()  # namespace only: no __init__.py
    monkeypatch.setenv("PYTHONPATH", str(on_pythonpath))
    monkeypatch.syspath_prepend(str(on_pythonpath))
    monkeypatch.syspath_prepend(str(editable))
    monkeypatch.setattr(helper, "_PACKAGES", ("ib_fake_a", "ib_fake_b", "ib_fake_c"))
    before = list(sys.path)
    got_site, got_checkout = helper.default_dirs()
    assert sys.path == before
    assert got_site == [str(site)]
    assert got_checkout == [str(editable / "ib_fake_b")]
