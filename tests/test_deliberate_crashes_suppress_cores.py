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

Detection is AST-based on purpose. Roughly forty places in this suite mention SIGSEGV
or SIGABRT in a comment, a docstring, or a return-code assertion such as
`assert f(-11) is True`, and none of those crash anything. A textual scan would be
almost entirely false positives, so only real calls and real child-script strings are
considered, and comments cannot satisfy the suppression side either.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "temp", "build", "dist"}

# Unconditional: these can only be a deliberate fatal fault.
_CRASH_MARKERS = (
    "string_at(0)",  # ctypes.string_at(0) -> strlen(NULL) -> SIGSEGV
    "os.abort()",  # SIGABRT
    "_sigsegv(",  # faulthandler._sigsegv()
)

# Conditional: these are only a deliberate crash when aimed at a core-dumping
# signal. `signal.raise_signal(signum)` re-raising SIGINT after restoring the
# default handler is the normal terminal-prompt idiom and must not be flagged.
_SIGNAL_DIRECTED = ("raise_signal(", "os.kill(", ".kill(")

# A signal aimed at self dumps core only for these. SIGKILL, SIGTERM and SIGINT
# do not, which is why SIGKILL is the recommended way to make a child vanish.
_FATAL_SIGNALS = ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGTRAP")

# Either spelling counts as suppressing the dump.
_SUPPRESS_MARKERS = ("PR_SET_DUMPABLE", "prctl(4")

_PRCTL_SET_DUMPABLE = 4


def _test_roots():
    """Every directory named `tests` in the repo, found rather than listed.

    Hardcoding the roots means a suite added later is silently unguarded, which is
    how this check would quietly stop being worth running.
    """
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
    """
    seen = set()
    for root in _test_roots():
        for path in sorted(root.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
                continue
            if path.resolve() not in seen:
                seen.add(path.resolve())
                yield path


def _docstring_nodes(tree):
    """Constant nodes that are docstrings, which must not count as either signal."""
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


def _code_strings(tree):
    """Every string literal that is not a docstring."""
    skip = _docstring_nodes(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


def _directs_a_fatal_signal(tree) -> bool:
    """A call that aims a core-dumping signal at a process, as real code.

    Only fires when a fatal signal is named in the call. `raise_signal(signum)` with
    a variable, the re-raise-after-restoring-the-default-handler idiom, is not a
    deliberate crash and must not be flagged.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        rendered = ast.unparse(node)
        if not any(marker in rendered for marker in _SIGNAL_DIRECTED):
            continue
        if any(sig in rendered for sig in _FATAL_SIGNALS):
            return True
    return False


def _calls_prctl_set_dumpable(tree) -> bool:
    """A real prctl(PR_SET_DUMPABLE, 0, ...) call in this module's code."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not ast.unparse(node.func).endswith("prctl"):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == _PRCTL_SET_DUMPABLE:
            return True
        if isinstance(first, ast.Name) and first.id.endswith("PR_SET_DUMPABLE"):
            return True
    return False


# Anything that could possibly make _classify say "crashes". Used as a raw-text
# prefilter: the AST is a subset of the source, so a file whose text contains none of
# these cannot match, and parsing it would be wasted work. This is what keeps the
# check cheap. Without it, unparsing every Call node in ~1000 files costs ~9s; with
# it, only a handful of files are ever parsed.
_PREFILTER = _CRASH_MARKERS + _SIGNAL_DIRECTED


@lru_cache(maxsize = None)
def _classify(path):
    """Return (crashes_on_purpose, suppresses_core) for one file."""
    source = path.read_text(encoding = "utf-8", errors = "replace")
    if not any(marker in source for marker in _PREFILTER):
        return False, False

    try:
        tree = ast.parse(source, str(path))
    except SyntaxError:
        # Fixture files that are deliberately unparseable are not running anything.
        return False, False
    strings = _code_strings(tree)

    crashes = _directs_a_fatal_signal(tree) or any(
        marker in text for text in strings for marker in _CRASH_MARKERS
    )
    # The same, written inside a script string handed to a child interpreter.
    for text in strings:
        if any(marker in text for marker in _SIGNAL_DIRECTED) and any(
            sig in text for sig in _FATAL_SIGNALS
        ):
            crashes = True

    suppresses = _calls_prctl_set_dumpable(tree) or any(
        marker in text for text in strings for marker in _SUPPRESS_MARKERS
    )
    return crashes, suppresses


_FIX = """\
Add this to the child before it faults:

    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE = 0

The child still dies of the same signal, so nothing you assert changes.

Two traps:
  * prctl is Linux-only, so guard the call. It is a no-op elsewhere.
  * RLIMIT_CORE = 0 looks like the fix and is NOT: a piped core_pattern ignores
    it, measured at 117ms and still dumping.

If you only need the child to vanish rather than to take a specific signal, use
SIGKILL instead. SIGKILL never produces a core."""


def test_the_scan_finds_the_files_it_is_meant_to_guard():
    """A silent scan matching nothing would pass forever and protect nothing."""
    crashing = [p for p in _iter_test_files() if _classify(p)[0]]
    assert crashing, (
        "no deliberate-crash files matched, so this check is guarding an empty set. "
        "Either the markers or the test roots are wrong."
    )


def test_every_deliberate_crash_suppresses_its_core():
    offenders = [
        p.relative_to(REPO_ROOT) for p in _iter_test_files() if _classify(p) == (True, False)
    ]
    assert not offenders, (
        "These files crash a process on purpose without suppressing the core dump:\n\n"
        + "".join(f"    {p}\n" for p in offenders)
        + "\nA fatal signal is piped to the host core_pattern handler (apport on "
        "Ubuntu), which reads the WHOLE core before the child is reaped: about 4x the "
        "wall time and a multi-MB write per fault, every run.\n\n" + _FIX
    )
