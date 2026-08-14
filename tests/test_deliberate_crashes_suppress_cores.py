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
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories holding test suites. Anything under these is scanned.
_TEST_ROOTS = (
    REPO_ROOT / "tests",
    REPO_ROOT / "studio" / "backend" / "tests",
)

# Substrings that mean "this code, or this string handed to a child interpreter,
# deliberately takes a fatal signal that dumps core".
_CRASH_MARKERS = (
    "string_at(0)",          # ctypes.string_at(0) -> strlen(NULL) -> SIGSEGV
    "os.abort()",            # SIGABRT
    "raise_signal(",         # signal.raise_signal(signal.SIGSEGV) and friends
    "_sigsegv(",             # faulthandler._sigsegv()
)

# A kill aimed at self only dumps core for these. SIGKILL and SIGTERM do not.
_FATAL_SIGNALS = ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGTRAP")

# Either spelling counts as suppressing the dump.
_SUPPRESS_MARKERS = ("PR_SET_DUMPABLE", "prctl(4")

_PRCTL_SET_DUMPABLE = 4


def _iter_test_files():
    for root in _TEST_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
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


def _self_kill_with_fatal_signal(tree) -> bool:
    """os.kill(os.getpid(), signal.SIGSEGV) and similar, as real code."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func) if node.args else ""
        if not func.endswith("kill"):
            continue
        rendered = ast.unparse(node)
        if "getpid()" in rendered and any(sig in rendered for sig in _FATAL_SIGNALS):
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


def _classify(path):
    """Return (crashes_on_purpose, suppresses_core) for one test file."""
    tree = ast.parse(path.read_text(encoding = "utf-8", errors = "replace"), str(path))
    strings = _code_strings(tree)

    crashes = _self_kill_with_fatal_signal(tree) or any(
        marker in text for text in strings for marker in _CRASH_MARKERS
    )
    # A self-kill written inside a child-script string counts too.
    for text in strings:
        if "getpid()" in text and any(sig in text for sig in _FATAL_SIGNALS):
            crashes = True

    suppresses = _calls_prctl_set_dumpable(tree) or any(
        marker in text for text in strings for marker in _SUPPRESS_MARKERS
    )
    return crashes, suppresses


def test_the_scan_finds_the_files_it_is_meant_to_guard():
    """A silent scan that matches nothing would pass forever and protect nothing."""
    crashing = [p for p in _iter_test_files() if _classify(p)[0]]
    assert crashing, "no deliberate-crash tests found; the scan or the markers are wrong"


@pytest.mark.parametrize(
    "path",
    [pytest.param(p, id = str(p.relative_to(REPO_ROOT))) for p in _iter_test_files()],
)
def test_deliberate_crash_suppresses_its_core(path):
    crashes, suppresses = _classify(path)
    if not crashes:
        return
    assert suppresses, (
        f"{path.relative_to(REPO_ROOT)} crashes a process on purpose but does not "
        f"suppress the core dump.\n\n"
        f"A fatal signal here is piped to the host core_pattern handler (apport on "
        f"Ubuntu), which reads the whole core before the child is reaped: about 4x the "
        f"wall time and a multi-MB write per fault, on every run of this suite.\n\n"
        f"Add this to the child before it faults:\n"
        f"    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE = 0\n\n"
        f"The child still dies of the same signal, so nothing you assert changes. Guard "
        f"the call, since prctl is Linux-only. RLIMIT_CORE = 0 does NOT work: a piped "
        f"core_pattern ignores it. If you only need the child to vanish rather than to "
        f"take a specific signal, use SIGKILL, which never dumps core."
    )
