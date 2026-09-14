# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that every test that starts PowerShell goes through the shared runner.

tests/_shared/unsloth_pwsh_runner.py exists because a `pwsh -NonInteractive` startup
reads and rewrites $XDG_CACHE_HOME/powershell/StartupProfileData-NonInteractive, and
under xdist every worker shares one $HOME, so the whole job races on one ~83 KB file.
A startup that deserialises a half-written one dies before it reaches our script, as
`Stack overflow.` + SIGABRT or as `System.IO.FileLoadException: The given assembly name
was invalid`. Measured at 7/4000 startups on a shared cache against 0/4000 with one
cache directory per worker; the runner's own header records both arms.

That protection is opt-in: it lives in `run_pwsh`, so a test file that calls
`subprocess.run(["pwsh", ...])` directly silently opts out and rejoins the race. This
happened to tests/studio/install/test_installed_release_backend_line.py, whose 498
parametrised `test_ps1_printer` cases fail ~9 at a time at `-n 16` with exactly that
FileLoadException. Grepping for "pwsh" does not catch it -- the string sits in an argv
list, and every file that DOES use the runner mentions pwsh in prose too -- so this
walks the AST of each test file and reports the call nodes themselves, with line
numbers.

A direct call is not always wrong. A test that hands its child a private HOME has
already left the race by another route, which is why tests/test_windows_amd_gpu_scan_
fallback.py was the one pwsh-heavy file with zero failures in backend CI run
32341628757. Those files are listed in _ALLOWED_DIRECT_PWSH_CALLS with the reason,
in the style of _EXPECTED_CI_SKIPS in tests/studio/test_ci_shell_suite_coverage.py:
an entry is a claim someone made and can be checked, a silent exemption is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

# The runner itself calls subprocess.run on a pwsh argv -- that is the whole point of it.
_RUNNER = TESTS_ROOT / "_shared" / "unsloth_pwsh_runner.py"

# subprocess entry points that start a process. `subprocess.run(...)`, and the
# from-import spellings, are both resolved; nothing else in subprocess spawns.
_SPAWNERS = frozenset({"run", "Popen", "call", "check_call", "check_output"})

# Interpreter names, lowercased and stripped of a .exe suffix, that mean PowerShell.
_PWSH_EXECUTABLES = frozenset({"pwsh", "powershell", "powershell_ise"})

# Files allowed to spawn PowerShell without the shared runner, each with the reason the
# startup-cache race does not reach them. Keep this keyed on the path relative to
# tests/, and keep the reason specific enough to re-check.
_ALLOWED_DIRECT_PWSH_CALLS = {
    "test_windows_amd_gpu_scan_fallback.py": (
        "hands the child a hermetic env whose HOME is the per-test tmp_path, so "
        "XDG_CACHE_HOME resolves inside tmp_path and the startup cache is already "
        "private per test -- this is the one pwsh-heavy file with zero failures in "
        "backend CI run 32341628757"
    ),
}


def _scanned_files() -> list[Path]:
    """Every Python file under tests/ except the runner.

    All of tests/, not just test_*.py: a conftest or a tests/_shared helper that spawns
    pwsh puts every file that imports it back in the race, and would be invisible to a
    scan keyed on the filename.
    """
    files = sorted(p for p in TESTS_ROOT.rglob("*.py") if p != _RUNNER)
    assert files, f"no Python files under {TESTS_ROOT} -- did the directory move?"
    return files


def _is_pwsh_executable(text: str) -> bool:
    """True for a string that names the PowerShell binary, path or bare name."""
    name = text.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".exe"):
        name = name[: -len(".exe")]
    return name in _PWSH_EXECUTABLES


class _PwshCallFinder(ast.NodeVisitor):
    """Collects (lineno, rendered_call) for subprocess spawns of PowerShell.

    Two ways a call site names the interpreter, both seen in this repo:

      * a literal, `subprocess.run(["pwsh", "-NoLogo", ...])`;
      * a module constant, `subprocess.run([PWSH, "-Command", ...])` or a parametrised
        `[shell, ...]`, where the name is bound elsewhere to shutil.which("pwsh").

    The second is resolved by collecting, per module, every name assigned from a
    shutil.which()/`os.environ`-style expression that mentions a PowerShell binary, so
    a rename of the constant does not silently drop the file off this guard. A name is
    only treated as PowerShell if some assignment in the file ties it to one.
    """

    def __init__(self, pwsh_names: set[str], private_env_names: set[str]) -> None:
        self.pwsh_names = pwsh_names
        self.private_env_names = private_env_names
        self.found: list[tuple[int, str]] = []
        self._subprocess_aliases = {"subprocess"}
        self._bare_spawners: set[str] = set()

    # -- import bookkeeping ---------------------------------------------------------
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

    # -- the check ------------------------------------------------------------------
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
                if _is_pwsh_executable(child.value):
                    return True
            elif isinstance(child, ast.Name) and child.id in self.pwsh_names:
                return True
        return False

    def _has_private_cache(self, node: ast.Call) -> bool:
        """True if this spawn's `env` comes from the runner's `pwsh_env`.

        The second supported way through the module, for the call sites `run_pwsh` cannot
        own -- a Popen holder, a deliberate control, a caller with its own crash policy.
        Matched on the AST rather than trusted from a comment, so the exemption is a fact
        about the call and disappears the moment the argument does. Accepted inline
        (`env = pwsh_env(env)`) or through a name the file bound to it earlier, which is
        what a call site inside a retry loop naturally writes.
        """
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
    """Module-level names whose assigned value names a PowerShell binary.

    Deliberately generous on the right-hand side -- `shutil.which("pwsh")`,
    `shutil.which("pwsh") or shutil.which("powershell")`, a bare `"pwsh"`, a list of
    them -- and deliberately narrow on the left: only plain `Name` targets, so nothing
    is inferred about attributes or subscripts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, (ast.For, ast.comprehension)):
            # `for shell in POWERSHELLS:` / `[... for shell in (PWSH, PS5)]`
            targets, value = [node.target], node.iter
        else:
            continue
        names_a_binary = any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and _is_pwsh_executable(child.value)
            for child in ast.walk(value)
        )
        # Aliases of an already-known name, and only in the forms that really are aliases:
        # `PWSH_OR_NONE = PWSH`, `for shell in POWERSHELLS:`. A Call on the right-hand side
        # is excluded on purpose, or `proc = subprocess.run([PWSH, ...])` would make `proc`
        # itself read as an interpreter.
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
    """{path relative to tests/: [(lineno, callee), ...]} over the whole suite."""
    offenders = {}
    for path in _scanned_files():
        calls = direct_pwsh_calls(path)
        if calls:
            offenders[path.relative_to(TESTS_ROOT).as_posix()] = calls
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
                f"  tests/{rel}:{lineno}: {callee}(...)"
                for rel, calls in sorted(offenders.items())
                for lineno, callee in calls
            )
            + "\n\nUse run_pwsh(argv, ...) -- it takes the argv you already built. If "
            "the call site really is safe (its own private HOME for the child, say), "
            "add it to _ALLOWED_DIRECT_PWSH_CALLS with the reason."
        )

    @pytest.mark.parametrize("rel", sorted(_ALLOWED_DIRECT_PWSH_CALLS))
    def test_every_allowlist_entry_is_still_needed(self, rel):
        """An allowlist that outlives its call sites is a claim nobody rechecks. This
        fails once the file stops spawning PowerShell directly, or moves away."""
        path = TESTS_ROOT / rel
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
            # An env is given, but not the runner's one.
            'import subprocess\nsubprocess.run(["pwsh", "-c", "x"], env = os.environ.copy())\n': 2,
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
            # "pwsh" as prose, not as an argv0.
            'import subprocess\nsubprocess.run(["bash", "-c", "which pwsh"])\n',
            # The pwsh_env route, inline and through a hoisted name.
            'import subprocess\nsubprocess.run(["pwsh", "-c", "x"], env = pwsh_env())\n',
            'import subprocess\nsubprocess.Popen(["pwsh", "-c", "x"], env = pwsh_env(env))\n',
            'import subprocess\ne = pwsh_env()\nsubprocess.run(["pwsh", "-c", "x"], env = e)\n',
        ]
        for source in cases:
            path = tmp_path / "test_probe.py"
            path.write_text(source, encoding = "utf-8")
            assert direct_pwsh_calls(path) == [], f"false positive on {source!r}"
