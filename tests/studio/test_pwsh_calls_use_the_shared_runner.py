# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Direct pwsh spawns must go through the shared runner; xdist workers race on one startup cache."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

# Both test trees: studio/backend/tests also runs under xdist and reaches tests/_shared.
_SCAN_ROOTS = (TESTS_ROOT, REPO_ROOT / "studio" / "backend" / "tests")

_RUNNER = TESTS_ROOT / "_shared" / "unsloth_pwsh_runner.py"

_SPAWNERS = frozenset({"run", "Popen", "call", "check_call", "check_output"})

_PWSH_EXECUTABLES = frozenset({"pwsh", "powershell", "powershell_ise"})

# Switches only PowerShell takes: an argv carrying one is a PowerShell launch whatever argv0 is.
_PWSH_ONLY_SWITCHES = frozenset({"-noninteractive", "-executionpolicy", "-noprofile"})

# Files allowed to spawn PowerShell directly, keyed by repo-root path since two trees are scanned.
_ALLOWED_DIRECT_PWSH_CALLS = {
    "tests/test_windows_amd_gpu_scan_fallback.py": (
        "hands the child a hermetic env whose HOME is the per-test tmp_path, so "
        "XDG_CACHE_HOME resolves inside tmp_path and the startup cache is already "
        "private per test -- this is the one pwsh-heavy file with zero failures in "
        "backend CI run 32341628757"
    ),
}


def _scanned_files() -> list[Path]:
    """Every .py file, not just test_*.py: a conftest or helper that spawns pwsh reopens the race."""
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        found = sorted(p for p in root.rglob("*.py") if p != _RUNNER)
        assert found, f"no Python files under {root} -- did the directory move?"
        files.extend(found)
    return sorted(files)


def _is_pwsh_executable(text: str) -> bool:
    """True for a string that names the PowerShell binary, path or bare name."""
    name = text.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".exe"):
        name = name[: -len(".exe")]
    return name in _PWSH_EXECUTABLES


class _PwshCallFinder(ast.NodeVisitor):
    """Resolves PowerShell names bound by assignment, so a renamed constant cannot hide a spawn."""

    def __init__(self, pwsh_names: set[str], private_env_names: set[str]) -> None:
        self.pwsh_names = pwsh_names
        self.private_env_names = private_env_names
        self.found: list[tuple[int, str]] = []
        self._subprocess_aliases = {"subprocess"}
        self._bare_spawners: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "subprocess":
                self._subprocess_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "subprocess":
            for alias in node.names:
                if alias.name in _SPAWNERS:
                    self._bare_spawners.add(alias.asname or alias.name)
        self.generic_visit(node)

    def _is_spawner(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Attribute):
            return (
                func.attr in _SPAWNERS
                and isinstance(func.value, ast.Name)
                and func.value.id in self._subprocess_aliases
            )
        return isinstance(func, ast.Name) and func.id in self._bare_spawners

    def _mentions_pwsh(self, node: ast.expr) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if _is_pwsh_executable(child.value) or child.value.lower() in _PWSH_ONLY_SWITCHES:
                    return True
            elif isinstance(child, ast.Name) and child.id in self.pwsh_names:
                return True
        return False

    def _has_private_cache(self, node: ast.Call) -> bool:
        """True if the spawn's env comes from pwsh_env; checked on the AST, so it vanishes with the
        argument."""
        for kw in node.keywords:
            if kw.arg != "env":
                continue
            return any(
                isinstance(child, ast.Name)
                and (child.id == "pwsh_env" or child.id in self.private_env_names)
                for child in ast.walk(kw.value)
            )
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_spawner(node.func):
            argv = node.args[0] if node.args else None
            if argv is None:
                for kw in node.keywords:
                    if kw.arg == "args":
                        argv = kw.value
            if argv is not None and self._mentions_pwsh(argv) and not self._has_private_cache(node):
                self.found.append((node.lineno, ast.unparse(node.func)))
        self.generic_visit(node)


def _pwsh_bound_names(tree: ast.AST) -> set[str]:
    """Matches any right-hand side naming a PowerShell binary, but only plain Name targets on the left."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets, value = [node.target], node.iter
        else:
            continue
        names_a_binary = any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and _is_pwsh_executable(child.value)
            for child in ast.walk(value)
        )
        # A Call on the right is excluded, or `proc = subprocess.run([PWSH, ...])` reads as a shell.
        aliases_a_known_name = isinstance(
            value, (ast.Name, ast.Tuple, ast.List, ast.Set, ast.BoolOp, ast.IfExp)
        ) and any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(value))
        if not names_a_binary and not aliases_a_known_name:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _private_env_names(tree: ast.AST) -> set[str]:
    """Names bound to a `pwsh_env(...)` result, so a hoisted env still counts as private."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(child, ast.Name) and (child.id == "pwsh_env" or child.id in names)
            for child in ast.walk(node.value)
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def direct_pwsh_calls(path: Path) -> list[tuple[int, str]]:
    """Every subprocess spawn of PowerShell in `path` that bypasses the shared runner."""
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    finder = _PwshCallFinder(_pwsh_bound_names(tree), _private_env_names(tree))
    finder.visit(tree)
    return sorted(finder.found)


def scan_tests() -> dict[str, list[tuple[int, str]]]:
    """{path relative to the repo root: [(lineno, callee), ...]} over every scanned tree."""
    offenders = {}
    for path in _scanned_files():
        calls = direct_pwsh_calls(path)
        if calls:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = calls
    return offenders


class TestEveryPwshCallUsesTheSharedRunner:
    def test_no_test_file_spawns_powershell_directly(self):
        """The guard itself. A new direct call site fails here with its own line number,
        rather than as an unrelated FileLoadException in someone else's test a month
        later."""
        offenders = {
            rel: calls
            for rel, calls in scan_tests().items()
            if rel not in _ALLOWED_DIRECT_PWSH_CALLS
        }
        assert not offenders, (
            "these test files start PowerShell through subprocess instead of "
            "run_pwsh from tests/_shared/unsloth_pwsh_runner.py, so they share one "
            "$XDG_CACHE_HOME/powershell startup cache with every other xdist worker "
            "and can die at startup with `Stack overflow.` or "
            "`System.IO.FileLoadException: The given assembly name was invalid`:\n"
            + "\n".join(
                f"  {rel}:{lineno}: {callee}(...)"
                for rel, calls in sorted(offenders.items())
                for lineno, callee in calls
            )
            + "\n\nUse run_pwsh(argv, ...) -- it takes the argv you already built. If "
            "the call site really is safe (its own private HOME for the child, say), "
            "add it to _ALLOWED_DIRECT_PWSH_CALLS with the reason."
        )

    def test_the_scan_reaches_the_backend_test_tree(self):
        """Checks that a real backend test file is scanned; listing the root alone can pass with no
        files."""
        backend = REPO_ROOT / "studio" / "backend" / "tests"
        assert backend in _SCAN_ROOTS, "the backend test tree dropped off the scan roots"
        scanned = _scanned_files()
        assert any(
            p.is_relative_to(backend) for p in scanned
        ), f"no file under {backend} was scanned"

    @pytest.mark.parametrize("rel", sorted(_ALLOWED_DIRECT_PWSH_CALLS))
    def test_every_allowlist_entry_is_still_needed(self, rel):
        """An allowlist that outlives its call sites is a claim nobody rechecks. This
        fails once the file stops spawning PowerShell directly, or moves away."""
        path = REPO_ROOT / rel
        assert path.is_file(), f"allowlisted {rel} does not exist; drop the entry"
        assert direct_pwsh_calls(path), (
            f"{rel} no longer spawns PowerShell directly, so its "
            "_ALLOWED_DIRECT_PWSH_CALLS entry is stale -- remove it"
        )

    @pytest.mark.parametrize("rel", sorted(_ALLOWED_DIRECT_PWSH_CALLS))
    def test_every_allowlist_entry_carries_a_reason(self, rel):
        reason = _ALLOWED_DIRECT_PWSH_CALLS[rel]
        assert (
            isinstance(reason, str) and len(reason.split()) >= 5
        ), f"{rel} needs a reason saying why the startup-cache race cannot reach it"

    def test_the_scanner_sees_a_direct_call_it_is_shown(self, tmp_path):
        """Non-vacuity. Every shape this guard claims to resolve, against a scanner
        that is only ever exercised on a suite it currently passes on."""
        cases = {
            'import subprocess\nsubprocess.run(["pwsh", "-Command", "echo hi"])\n': 2,
            'import subprocess\nsubprocess.run([r"C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-c"])\n': 2,
            'from subprocess import run\nrun(["powershell", "-NoProfile"])\n': 2,
            'import subprocess as sp\nPWSH = shutil.which("pwsh")\nsp.Popen([PWSH, "-c", "x"])\n': 3,
            'import subprocess\nsubprocess.check_output(args = ["pwsh", "-c", "x"])\n': 2,
            'import subprocess\nsubprocess.run(["pwsh", "-c", "x"], env = os.environ.copy())\n': 2,
            # The interpreter arrives as a parameter, named only by the switches it is given.
            'import subprocess\ndef start(shell):\n    subprocess.Popen([shell, "-NoLogo", "-NonInteractive", "-File", "x.ps1"])\n': 3,
        }
        for source, lineno in cases.items():
            path = tmp_path / "test_probe.py"
            path.write_text(source, encoding = "utf-8")
            found = direct_pwsh_calls(path)
            assert [line for line, _ in found] == [lineno], f"missed: {source!r} -> {found!r}"

    def test_the_scanner_does_not_flag_the_shared_runner_or_plain_shells(self, tmp_path):
        """The other half: a run through run_pwsh, and a subprocess spawn of something
        that is not PowerShell, must both stay clean or the guard is noise."""
        cases = [
            'from unsloth_pwsh_runner import run_pwsh\nrun_pwsh(["pwsh", "-c", "x"])\n',
            'import subprocess\nsubprocess.run(["bash", "-c", "echo hi"])\n',
            'import subprocess\nsubprocess.run([sys.executable, "-c", "print(1)"])\n',
            'import subprocess\ndef start(shell):\n    subprocess.Popen([shell, "-NoLogo", "-NonInteractive"], env = pwsh_env(env))\n',
            # "pwsh" as prose, not as an argv0.
            'import subprocess\nsubprocess.run(["bash", "-c", "which pwsh"])\n',
            'import subprocess\nsubprocess.run(["pwsh", "-c", "x"], env = pwsh_env())\n',
            'import subprocess\nsubprocess.Popen(["pwsh", "-c", "x"], env = pwsh_env(env))\n',
            'import subprocess\ne = pwsh_env()\nsubprocess.run(["pwsh", "-c", "x"], env = e)\n',
        ]
        for source in cases:
            path = tmp_path / "test_probe.py"
            path.write_text(source, encoding = "utf-8")
            assert direct_pwsh_calls(path) == [], f"false positive on {source!r}"
