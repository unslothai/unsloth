# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every test that crashes a process on purpose must suppress its core dump.

Some tests need a child that dies of a real fatal signal, usually to prove that a
supervisor treats a hard fault differently from a clean non-zero exit. The signal is
legitimate. The core dump that follows is not: `/proc/sys/kernel/core_pattern` on a
stock Ubuntu box pipes it to apport, which reads the WHOLE core before the child is
reaped. Measured locally that is 123ms and a multi-MB write per fault, against 30ms
with the dump suppressed.

At CI volume that is a slow suite. The reason it is worth a guard is that the idiom
gets copied. The same `ctypes.string_at(0)` line, lifted into a local reproduction
harness wrapped in `for _ in range(trials)`, produced roughly 240 deliberate faults in
46 seconds on a shared build box and the resulting apport storm took down every tmux
session for that user. The tests here are where people learn the pattern, so this is
where the rule belongs.

The fix is one call before the fault:

    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE = 0

A non-dumpable process still dies of the same signal, so nothing a test asserts
changes, and no core is written. Note that `RLIMIT_CORE = 0` is NOT a substitute: a
piped core_pattern ignores it, measured at 117ms and still dumping. If the test only
needs "the child vanished" rather than a specific signal, prefer SIGKILL, which never
produces a core.

Three things this deliberately gets right, each of which it got wrong first:

  * Detection is AST-based. Roughly forty places in this suite mention SIGSEGV or
    SIGABRT in a comment, a docstring, or a return-code assertion such as
    `assert f(-11) is True`, and none of those crash anything. A textual scan would be
    almost entirely false positives, and comments could satisfy the suppression side.
  * A crash written as ordinary code counts, not only one inside a `-c` script string.
    A `multiprocessing` target that calls `ctypes.string_at(0)` dumps exactly the same
    core as a script string that does.
  * Suppression is matched to the crash it is meant to cover, not to the file. A file
    that defines a suppressed helper must not thereby bless a naked crash added to it
    later, which is the most likely way this regresses given the files that now
    contain such helpers.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "temp", "build", "dist"}

# Calls that can only be a deliberate fatal fault, keyed by the trailing attribute.
# `arg0` is a required literal first argument, `owners` a set of acceptable receivers.
# `abort` needs the receiver check: Playwright's `route.abort()` and a thread's
# `.abort()` are ordinary calls that share the name and crash nothing.
_CRASH_CALLS = {
    "string_at": {"arg0": 0},  # ctypes.string_at(0) -> strlen(NULL) -> SIGSEGV
    "abort": {"owners": ("os", "ctypes", "libc", "CDLL")},  # -> SIGABRT
    "_sigsegv": {},  # faulthandler._sigsegv()
}

# Textual form of the same, for snippets handed to a child interpreter.
_CRASH_MARKERS = ("string_at(0)", "os.abort()", "_sigsegv(")

# Only a deliberate crash when aimed at a core-dumping signal. `raise_signal(signum)`
# re-raising SIGINT after restoring the default handler is the normal terminal-prompt
# idiom and must not be flagged.
_SIGNAL_DIRECTED = ("raise_signal(", "os.kill(", ".kill(")
_DIRECTED_NAMES = {"raise_signal", "kill"}

# A signal aimed at self dumps core only for these. SIGKILL, SIGTERM and SIGINT do not,
# which is why SIGKILL is the recommended way to make a child vanish.
_FATAL_SIGNALS = ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGTRAP")

_PR_SET_DUMPABLE = 4

# Raw-text prefilter, run before any parsing. The AST is a subset of the source, so a
# file whose text holds none of these cannot match and parsing it is wasted work. This
# is what keeps the check cheap: 32 of ~1050 files are actually parsed, 496ms rather
# than 9s.
_PREFILTER = _CRASH_MARKERS + _SIGNAL_DIRECTED + ("abort(",)


def _test_roots():
    """Every directory named `tests`, found rather than listed, so a suite added later
    is guarded without anyone remembering to update a list here."""
    roots = []
    for path in REPO_ROOT.rglob("tests"):
        if not path.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        roots.append(path)
    return sorted(roots)


def _iter_test_files():
    """Every Python file under a test root, not only `test_*.py`.

    conftest.py, shared harnesses and the `_*_shim.py` files CI invokes directly can
    all spawn children, so restricting this to `test_*.py` would leave a real hole.
    This module is excluded: it necessarily contains every marker it looks for.
    """
    seen = set()
    for root in _test_roots():
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            if resolved == _SELF or resolved in seen:
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
                continue
            seen.add(resolved)
            yield path


def _called_name(node):
    """Trailing attribute of a call, without unparsing it.

    `ast.unparse` on every Call in the suite is what made an earlier version of this
    check cost 9s. Almost every call is ruled out by its name alone, so unparse only
    the handful that survive.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _prctl_clears_dumpable(node) -> bool:
    """A prctl(PR_SET_DUMPABLE, 0, ...) call. The value argument matters: prctl(4, 1)
    re-enables dumps, so treating any PR_SET_DUMPABLE call as suppression would bless a
    crash that still dumps."""
    if _called_name(node) != "prctl" or len(node.args) < 2:
        return False
    cmd, value = node.args[0], node.args[1]
    cmd_ok = (isinstance(cmd, ast.Constant) and cmd.value == _PR_SET_DUMPABLE) or (
        isinstance(cmd, ast.Name) and cmd.id.endswith("PR_SET_DUMPABLE")
    )
    return cmd_ok and isinstance(value, ast.Constant) and value.value == 0


def _scope_suppresses(scope) -> bool:
    """Whether this function (or module) clears the dumpable flag anywhere in it."""
    return any(_prctl_clears_dumpable(n) for n in ast.walk(scope))


def _snippet_suppresses(snippet: str) -> bool:
    """Whether a child-script snippet clears the dumpable flag.

    Snippets are Python source, so parse them and ask the same question. Fragments that
    do not parse fall back to a textual check.
    """
    try:
        return _scope_suppresses(ast.parse(snippet))
    except SyntaxError:
        return "prctl(4, 0" in snippet or "PR_SET_DUMPABLE, 0" in snippet


def _snippet_crashes(snippet: str) -> bool:
    if any(marker in snippet for marker in _CRASH_MARKERS):
        return True
    return any(m in snippet for m in _SIGNAL_DIRECTED) and any(s in snippet for s in _FATAL_SIGNALS)


def _fold(node, env):
    """Constant-fold a string expression. Returns (value, consumed literal ids).

    Needed so `_SAFE = _SUPPRESS + "ctypes.string_at(0)"` is judged as the script it
    actually becomes, rather than as a bare literal with no suppression next to it.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, {id(node)}
    if isinstance(node, ast.Name):
        return env.get(node.id), set()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, lids = _fold(node.left, env)
        right, rids = _fold(node.right, env)
        if left is not None and right is not None:
            return left + right, lids | rids
    return None, set()


def _snippets(tree):
    """Candidate child scripts: folded module constants, then unconsumed literals."""
    env, consumed = {}, set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value, ids = _fold(node.value, env)
        if value is not None:
            env[target.id] = value
            consumed |= ids

    docstrings = _docstring_ids(tree)
    out = list(env.values())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in consumed or id(node) in docstrings:
            continue
        out.append(node.value)
    return out


def _docstring_ids(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _is_crash_call(node) -> bool:
    """A call that deliberately takes a core-dumping signal, written as code."""
    if not isinstance(node, ast.Call):
        return False
    name = _called_name(node)
    if name is None:
        return False
    if name in _CRASH_CALLS:
        func = ast.unparse(node.func)
        rule = _CRASH_CALLS[name]
        if "arg0" in rule:
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant) and first.value == rule["arg0"]):
                return False
        if "owners" in rule:
            receiver = func[: -len(name)]
            if not any(owner in receiver for owner in rule["owners"]):
                return False
        return True
    if name not in _DIRECTED_NAMES:
        return False
    rendered = ast.unparse(node)
    return any(s in rendered for s in _FATAL_SIGNALS)


def _enclosing_scopes(tree):
    """Map each node id to the nearest enclosing function, else the module."""
    owner = {}

    def walk(scope, node):
        for child in ast.iter_child_nodes(node):
            next_scope = (
                child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            )
            owner[id(child)] = next_scope
            walk(next_scope, child)

    walk(tree, tree)
    return owner


@lru_cache(maxsize = None)
def _analyze(path):
    """`(crashes_on_purpose, unsuppressed_reasons)` for one file.

    One pass, cached, because both tests below ask about every file and parsing twice
    doubles the cost of the check for nothing.
    """
    source = path.read_text(encoding = "utf-8", errors = "replace")
    if not any(marker in source for marker in _PREFILTER):
        return False, ()

    try:
        tree = ast.parse(source, str(path))
    except SyntaxError:
        # A fixture that is deliberately unparseable is not running anything.
        return False, ()

    crashes, out = False, []
    for snippet in _snippets(tree):
        if not _snippet_crashes(snippet):
            continue
        crashes = True
        if not _snippet_suppresses(snippet):
            out.append("a child script that crashes on purpose")

    owner = None
    for node in ast.walk(tree):
        if not _is_crash_call(node):
            continue
        crashes = True
        if owner is None:
            owner = _enclosing_scopes(tree)
        scope = owner.get(id(node), tree)
        if not _scope_suppresses(scope):
            where = getattr(scope, "name", "module level")
            out.append(f"{ast.unparse(node)} at line {node.lineno} (in {where})")
    return crashes, tuple(dict.fromkeys(out))


def _violations(path):
    """Deliberate crashes in this file whose core dump is not suppressed."""
    return _analyze(path)[1]


def _crashes(path) -> bool:
    """Whether this file crashes on purpose at all, suppressed or not."""
    return _analyze(path)[0]


_FIX = """\
Add this to the child before it faults:

    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE = 0

The child still dies of the same signal, so nothing you assert changes.

Three traps:
  * prctl is Linux-only, so guard the call. It is a no-op elsewhere.
  * The value argument matters. prctl(4, 1) re-enables dumps.
  * RLIMIT_CORE = 0 looks like the fix and is NOT: a piped core_pattern ignores
    it, measured at 117ms and still dumping.

If you only need the child to vanish rather than to take a specific signal, use
SIGKILL instead. SIGKILL never produces a core."""


# Each of these is a way the detector was fooled before, kept as a fixture so the fix
# stays fixed. `want_violations` is whether the file should be reported.
_FIXTURES = {
    "crash_written_as_real_code": (
        "import ctypes\ndef child():\n    ctypes.string_at(0)\n",
        True,  # not inside a script string, and still dumps a core
    ),
    "suppression_in_the_same_function": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,
    ),
    "suppression_in_a_different_function": (
        "import ctypes\n"
        "def suppressed():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n"
        "def naked():\n"
        "    ctypes.string_at(0)\n",
        True,  # the first function must not bless the second
    ),
    "helper_then_a_naked_script": (
        'SUPPRESS = "ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"\n'
        'SAFE = SUPPRESS + "ctypes.string_at(0)\\n"\n'
        'NAKED = "ctypes.string_at(0)\\n"\n',
        True,  # SAFE is fine, NAKED is not, and the file must not pass on SAFE
    ),
    "concatenated_script_is_folded": (
        'SUPPRESS = "ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"\n'
        'SAFE = SUPPRESS + "ctypes.string_at(0)\\n"\n',
        False,  # judged as the script it becomes, not as a bare literal
    ),
    "dumpable_set_back_to_one": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        True,  # prctl(4, 1) re-enables dumps, so this still dumps
    ),
    "signal_named_in_a_comment_only": (
        "# this used to raise SIGSEGV, now it returns\ndef child():\n    return -11\n",
        False,
    ),
    "reraise_of_a_variable_signal": (
        "import signal\n"
        "def restore(signum):\n"
        "    signal.signal(signum, signal.SIG_DFL)\n"
        "    signal.raise_signal(signum)\n",
        False,  # the terminal-prompt idiom, no fatal signal named
    ),
    "unrelated_abort_methods": (
        "def go(route, task):\n    route.abort('failed')\n    task.abort()\n",
        False,  # Playwright and friends share the name and crash nothing
    ),
}


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_the_detector_is_not_fooled(tmp_path, name):
    source, want_violations = _FIXTURES[name]
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding = "utf-8")
    crashes, violations = _analyze(path)
    assert (
        bool(violations) is want_violations
    ), f"{name}: expected violations={want_violations}, got {violations or '()'}"
    if want_violations:
        assert crashes, f"{name}: a reported file must also count as crashing"


def test_the_scan_finds_the_files_it_is_meant_to_guard():
    """A silent scan matching nothing would pass forever and protect nothing.

    This module excludes itself, so the match has to come from a real test.
    """
    crashing = [p.relative_to(REPO_ROOT) for p in _iter_test_files() if _crashes(p)]
    assert crashing, (
        "no deliberate-crash files matched, so this check is guarding an empty set. "
        "Either the markers or the test roots are wrong."
    )


def test_every_deliberate_crash_suppresses_its_core():
    offenders = {
        p.relative_to(REPO_ROOT): _violations(p) for p in _iter_test_files() if _violations(p)
    }
    report = "".join(
        f"    {path}\n" + "".join(f"        {why}\n" for why in whys)
        for path, whys in offenders.items()
    )
    assert not offenders, (
        "These crash a process on purpose without suppressing the core dump:\n\n"
        + report
        + "\nA fatal signal is piped to the host core_pattern handler (apport on "
        "Ubuntu), which reads the WHOLE core before the child is reaped: about 4x the "
        "wall time and a multi-MB write per fault, every run.\n\n" + _FIX
    )
