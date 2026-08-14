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
import warnings
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
# Which argument carries the signal. Checking every argument read the PID in
# `os.kill(11, signal.SIGKILL)` as SIGSEGV and failed CI over a call that cannot dump.
_SIGNAL_ARG_INDEX = {"raise_signal": 0, "kill": 1}

# A signal aimed at self dumps core only for these. SIGKILL, SIGTERM and SIGINT do not,
# which is why SIGKILL is the recommended way to make a child vanish.
_FATAL_SIGNALS = ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGTRAP")
_FATAL_SIGNAL_NUMBERS = {4, 5, 6, 7, 8, 11}  # SIGILL SIGTRAP SIGABRT SIGBUS SIGFPE SIGSEGV

_PR_SET_DUMPABLE = 4

# Raw-text prefilter, run before any parsing. The AST is a subset of the source, so a
# file whose text holds none of these cannot match and parsing it is wasted work. This
# is what keeps the check cheap: 32 of ~1050 files are actually parsed, 496ms rather
# than 9s.
# Deliberately looser than `_CRASH_MARKERS`: matching `string_at(0)` exactly meant
# `string_at( 0)` was skipped before it was ever parsed. The prefilter only decides what
# to parse, so it should over-match and let the AST checks be the precise ones.
_PREFILTER = ("string_at(", "abort(", "_sigsegv(") + _SIGNAL_DIRECTED


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


def _assigned_pair(node):
    """`(target, value)` for a single-name assignment, else None."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target, value = node.target, node.value
    else:
        return None
    return (target, value) if isinstance(target, ast.Name) else None


def _string_env(tree):
    """Name to foldable string, per scope, so same-named locals cannot collide.

    One flat map let an unrelated function's `SCRIPT = "print(1)"` overwrite a
    module-level `SCRIPT` that really does crash, and the executed script then went
    unchecked. Functions are seeded from the module bindings and shadow them locally.
    Annotated constants and strings assembled inside a test are folded either way,
    so `_SAFE = _SUPPRESS + "ctypes.string_at(0)"` is still judged as what it becomes.
    """
    owner = _enclosing_scopes(tree)
    module_env, scoped = {}, {}
    for module_pass in (True, False):
        for node in ast.walk(tree):
            pair = _assigned_pair(node)
            if pair is None:
                continue
            scope = owner.get(id(node), tree)
            if (scope is tree) != module_pass:
                continue
            env = module_env if module_pass else scoped.setdefault(id(scope), dict(module_env))
            folded, _ = _fold(pair[1], env)
            if folded is not None:
                env[pair[0].id] = folded
    return owner, module_env, scoped


def _iter_executable(scope):
    """Nodes that run when `scope` runs, skipping bodies that need a separate call.

    Walking everything treated a call inside an uninvoked nested `def` as having
    already run, so a helper that suppresses cores blessed a crash it never covered.
    """
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from _iter_executable(child)


def _rebound_names(scope):
    """Names this scope binds itself, which therefore no longer mean the import."""
    out = set()
    for node in ast.walk(scope):
        pair = _assigned_pair(node)
        if pair is not None:
            out.add(pair[0].id)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


def _snippets(tree):
    """Strings that actually reach a child interpreter.

    A string only counts if it is passed as a call argument, directly or through a
    name, a list/tuple, or a concatenation. That is what `subprocess.run([exe, "-c",
    SCRIPT])` looks like. Treating every string literal as a candidate script meant an
    ordinary expectation such as `assert "ctypes.string_at(0)" in text` was reported as
    a deliberate crash, which would fail CI over a string nothing executes.
    """
    owner, module_env, scoped = _string_env(tree)
    out = []

    def collect(node, env):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            # An f-string reaches the child as a script like any other. Keep its
            # literal parts: the interpolations cannot be known here, and dropping the
            # whole thing let `f"import os; os.abort(); print({v})"` through unread.
            out.append("".join(
                part.value for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ))
        elif isinstance(node, ast.Name):
            if node.id in env:
                out.append(env[node.id])
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                collect(element, env)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            folded, _ = _fold(node, env)
            if folded is not None:
                out.append(folded)
            else:
                collect(node.left, env)
                collect(node.right, env)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        env = scoped.get(id(owner.get(id(node), tree)), module_env)
        for argument in list(node.args) + [kw.value for kw in node.keywords]:
            collect(argument, env)
    return out


def _crash_aliases(tree):
    """Names imported directly from a crashing module, e.g. `from os import abort`.

    Needed so an aliased call is still recognised. A bare `abort()` is only fatal if it
    came from `os` or `ctypes`; Playwright's `route.abort()` keeps its receiver and is
    still correctly ignored.
    """
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.split(".")[0] not in ("os", "ctypes", "signal", "faulthandler"):
            continue
        for alias in node.names:
            if alias.name in _CRASH_CALLS:
                out.add(alias.asname or alias.name)
    return frozenset(out)


def _is_crash_call(node, aliases = frozenset()) -> bool:
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
            bare = isinstance(node.func, ast.Name) and node.func.id in aliases
            if not bare and not any(owner in receiver for owner in rule["owners"]):
                return False
        return True
    if name not in _DIRECTED_NAMES:
        return False
    index = _SIGNAL_ARG_INDEX[name]
    if len(node.args) <= index:
        return False
    # Only the signal argument, so a PID never reads as a signal.
    signal_argument = node.args[index]
    rendered = ast.unparse(signal_argument)
    if any(s in rendered for s in _FATAL_SIGNALS):
        return True
    # Numeric signals. `signal.raise_signal(11)` and `os.kill(pid, 6)` dump exactly the
    # same core as the named forms, so matching only symbolic names missed them.
    return (
        isinstance(signal_argument, ast.Constant)
        and signal_argument.value in _FATAL_SIGNAL_NUMBERS
    )


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


def _functions_by_name(tree):
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _position(node):
    """Source order of a node. Line alone ties on a one-line `-c` script, where
    `prctl(4, 0, ...); ctypes.string_at(0)` really does suppress the crash after it."""
    return (node.lineno, node.col_offset)


_AFTER_EVERYTHING = (float("inf"), 0)


def _clears_dumpable_before(scope, position) -> bool:
    """A prctl(4, 0, ...) on this scope's own path that runs before `position`.

    Order matters. Suppression placed after the fault does nothing, so accepting it
    anywhere in the scope blessed a child that still dumps.
    """
    return any(
        _prctl_clears_dumpable(node) and _position(node) < position
        for node in _iter_executable(scope)
    )


def _suppressed(node, scope, functions) -> bool:
    """Whether this crash call is covered, directly or by a helper it calls first."""
    position = _position(node)
    if _clears_dumpable_before(scope, position):
        return True
    # Following one level of local helper covers `suppress_core()` then the fault,
    # which is the natural shape once more than one test needs this.
    for called in _iter_executable(scope):
        if not isinstance(called, ast.Call) or _position(called) >= position:
            continue
        target = functions.get(_called_name(called))
        if target is not None and _clears_dumpable_before(target, _AFTER_EVERYTHING):
            return True
    return False


def _live_aliases(aliases, scope, rebound):
    """The imported crash names still in force where this call sits.

    A scope that binds the name itself no longer means the import, so a test that
    does `abort = mock` before calling it is not crashing anything.
    """
    if not aliases or scope is None:
        return aliases
    if id(scope) not in rebound:
        rebound[id(scope)] = _rebound_names(scope)
    return aliases - rebound[id(scope)]


def _unsuppressed_crashes(tree):
    """Crash calls in this tree whose core dump is not suppressed first."""
    owner = _enclosing_scopes(tree)
    functions = None
    aliases = _crash_aliases(tree)
    rebound = {}
    out = []
    for node in ast.walk(tree):
        scope = owner.get(id(node), tree)
        if not _is_crash_call(node, _live_aliases(aliases, scope, rebound)):
            continue
        if functions is None:
            functions = _functions_by_name(tree)
        if not _suppressed(node, scope, functions):
            out.append((node, scope))
    return out


def _tree_crashes(tree) -> bool:
    aliases = _crash_aliases(tree)
    owner = _enclosing_scopes(tree)
    rebound = {}
    return any(
        _is_crash_call(node, _live_aliases(aliases, owner.get(id(node), tree), rebound))
        for node in ast.walk(tree)
    )


_NESTED_EXEC = {"exec", "eval"}
_MAX_SNIPPET_DEPTH = 5


def _nested_scripts(tree):
    """Source a snippet hands to exec/eval, or on to another child interpreter.

    `SCRIPT = 'exec("import os; os.abort()")'` parses cleanly and holds no crash call
    of its own, so without this the crash one level down was never looked at.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node) in _NESTED_EXEC:
            argument = node.args[0] if node.args else None
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                yield argument.value
        else:
            yield from _snippets_of_call(node)


def _snippets_of_call(node):
    """Script strings passed to a nested subprocess call, e.g. a `-c` argument."""
    for argument in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(argument, (ast.List, ast.Tuple)):
            for element in argument.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    yield element.value


def _snippet_state(snippet: str, depth: int = 0):
    """`(crashes, violates)` for a child script.

    The snippet is Python, so parse it and reuse the same call detector rather than
    matching marker substrings. Textual matching missed aliased forms such as
    `from os import abort; abort()` and any spacing the markers did not anticipate.
    """
    try:
        with warnings.catch_warnings():
            # Snippets are other people's source. An unrelated escape-sequence warning
            # from one of them must not show up as noise in this suite's output.
            warnings.simplefilter("ignore")
            tree = ast.parse(snippet)
    except (SyntaxError, ValueError):
        crashes = any(marker in snippet for marker in _CRASH_MARKERS) or (
            any(m in snippet for m in _SIGNAL_DIRECTED)
            and any(s in snippet for s in _FATAL_SIGNALS)
        )
        suppressed = "prctl(4, 0" in snippet or "PR_SET_DUMPABLE, 0" in snippet
        return crashes, crashes and not suppressed
    crashes, violates = _tree_crashes(tree), bool(_unsuppressed_crashes(tree))
    if depth < _MAX_SNIPPET_DEPTH:
        for nested in _nested_scripts(tree):
            inner_crashes, inner_violates = _snippet_state(nested, depth + 1)
            crashes = crashes or inner_crashes
            violates = violates or inner_violates
    return crashes, violates


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
        snippet_crashes, violates = _snippet_state(snippet)
        crashes = crashes or snippet_crashes
        if violates:
            out.append("a child script that crashes on purpose")

    for node, scope in _unsuppressed_crashes(tree):
        where = getattr(scope, "name", "module level")
        out.append(f"{ast.unparse(node)} at line {node.lineno} (in {where})")
    crashes = crashes or _tree_crashes(tree)
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
        "import subprocess, sys\n"
        'SUPPRESS = "import ctypes\\nctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"\n'
        'SAFE = SUPPRESS + "ctypes.string_at(0)\\n"\n'
        'NAKED = "import ctypes\\nctypes.string_at(0)\\n"\n'
        'subprocess.run([sys.executable, "-c", SAFE])\n'
        'subprocess.run([sys.executable, "-c", NAKED])\n',
        True,  # SAFE is fine, NAKED is not, and the file must not pass on SAFE
    ),
    "concatenated_script_is_folded": (
        "import subprocess, sys\n"
        'SUPPRESS = "import ctypes\\nctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"\n'
        'SAFE = SUPPRESS + "ctypes.string_at(0)\\n"\n'
        'subprocess.run([sys.executable, "-c", SAFE])\n',
        False,  # judged as the script it becomes, not as a bare literal
    ),
    # Each of the following was a live hole found in review of the guard itself.
    "spacing_the_markers_did_not_anticipate": (
        "import ctypes\ndef child():\n    ctypes.string_at( 0)\n",
        True,  # the prefilter must not decide precision, only what to parse
    ),
    "suppression_placed_after_the_fault": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.string_at(0)\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n",
        True,  # too late, the core is already written
    ),
    "numeric_fatal_signal": (
        "import signal\ndef child():\n    signal.raise_signal(11)\n",
        True,  # 11 is SIGSEGV and dumps the same core as the symbolic name
    ),
    "annotated_script_constant": (
        "import subprocess, sys\n"
        'SUP: str = "import ctypes\\nctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"\n'
        'SCRIPT: str = SUP + "ctypes.string_at(0)\\n"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # an annotated assignment folds like any other
    ),
    "expectation_string_is_not_a_script": (
        'def test_x(script):\n    assert "ctypes.string_at(0)" in script\n',
        False,  # nothing executes it, so failing CI on it would be wrong
    ),
    "aliased_crash_inside_a_script": (
        "import subprocess, sys\n"
        'SCRIPT = "from os import abort\\nabort()\\n"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # same SIGABRT, spelled differently
    ),
    "suppression_via_a_local_helper": (
        "import ctypes\n"
        "def suppress_core():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "def child():\n"
        "    suppress_core()\n"
        "    ctypes.string_at(0)\n",
        False,  # factoring the suppression out is good practice, not a violation
    ),
    "sigkill_is_not_a_deliberate_crash": (
        "import os, signal\ndef stop(pid):\n    os.kill(pid, signal.SIGKILL)\n",
        False,  # SIGKILL never dumps, which is why it is the recommended alternative
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
    # Seven more the detector got wrong, each found by reading it rather than by a
    # failing run. Four let a real core dump through; three failed CI on safe code.
    "a_pid_that_looks_like_a_signal": (
        "import os, signal\ndef stop():\n    os.kill(11, signal.SIGKILL)\n",
        False,  # 11 is the PID here, and SIGKILL never dumps
    ),
    "same_name_bound_in_two_scopes": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes\\nctypes.string_at(0)\\n"\n'
        "def unrelated():\n"
        '    SCRIPT = "print(1)"\n'
        "    return SCRIPT\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # a local of the same name must not overwrite the script that runs
    ),
    "suppression_earlier_on_the_same_line": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        'ctypes.string_at(0)"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # a one-line -c script still has an order, given by the column
    ),
    "helper_called_only_from_an_uninvoked_def": (
        "import ctypes\n"
        "def suppress_core():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "def child():\n"
        "    def configure():\n"
        "        suppress_core()\n"
        "    ctypes.string_at(0)\n",
        True,  # configure() is never called, so nothing suppressed the fault
    ),
    "script_built_as_an_f_string": (
        "import subprocess, sys\n"
        'subprocess.run([sys.executable, "-c", f"import os; os.abort(); '
        'print({sys.argv})"])\n',
        True,  # an f-string reaches the child as a script like any other
    ),
    "imported_name_rebound_before_the_call": (
        "from os import abort\ndef test_it(mock):\n    abort = mock\n    abort()\n",
        False,  # only the mock runs, so there is no crash to suppress
    ),
    "crash_nested_inside_an_exec": (
        "import subprocess, sys\n"
        "SCRIPT = \"exec('import os; os.abort()')\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # the SIGABRT is one level down, and dumps just the same
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
