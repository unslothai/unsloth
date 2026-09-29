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
import re
import warnings
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

_MAX_COLLECT_DEPTH = 25

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "temp", "build", "dist"}

# Calls that can only be a deliberate fatal fault, keyed by the trailing attribute.
# `arg0` is a required literal first argument, `owners` a set of acceptable receivers.
# `abort` needs the receiver check: Playwright's `route.abort()` and a thread's `.abort()` are ordinary calls that share
# the name and crash nothing.
_CRASH_CALLS = {
    "string_at": {"arg0": 0},  # ctypes.string_at(0) -> strlen(NULL) -> SIGSEGV
    "abort": {"owners": ("os", "ctypes", "libc", "CDLL")},  # -> SIGABRT
    "_sigsegv": {},  # faulthandler._sigsegv()
}

_CRASH_MARKERS = ("string_at(0)", "os.abort()", "_sigsegv(")

# Only a deliberate crash when aimed at a core-dumping signal.
# `raise_signal(signum)` re-raising SIGINT after restoring the default handler is the normal terminal-prompt idiom and
# must not be flagged.
_SIGNAL_DIRECTED = ("raise_signal(", "os.kill(", ".kill(")
_DIRECTED_NAMES = {"raise_signal", "kill"}
# Which argument carries the signal.
# Checking every argument read the PID in `os.kill(11, signal.SIGKILL)` as SIGSEGV and failed CI over a call that cannot
# dump.
_SIGNAL_ARG_INDEX = {"raise_signal": 0, "kill": 1}

# A signal aimed at self dumps core only for these. SIGKILL, SIGTERM and SIGINT do not,
# which is why SIGKILL is the recommended way to make a child vanish.
_FATAL_SIGNALS = ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGTRAP", "SIGQUIT")
# Linux dump-core defaults: 3 QUIT, 4 ILL, 5 TRAP, 6 ABRT, 7 BUS, 8 FPE, 11 SEGV.
_FATAL_SIGNAL_NUMBERS = {3, 4, 5, 6, 7, 8, 11}

# Whole names only: a variable merely spelled `SIGQUIT_HANDLER` names no signal.
_FATAL_SIGNAL_RE = re.compile(r"\b(?:" + "|".join(_FATAL_SIGNALS) + r")\b")

_PR_SET_DUMPABLE = 4

# Raw-text prefilter, so only files that could match are parsed (32 of ~1050, ~1.5s against ~9s).
# Deliberately looser than `_CRASH_MARKERS`: it only decides what to parse, so it should over-match and leave
# precision to the AST checks. Matching `string_at(0)` exactly once skipped `string_at( 0)` before it was ever parsed.
_PREFILTER = ("string_at(", "abort(", "_sigsegv(") + _SIGNAL_DIRECTED
# An aliased import carries none of the shapes above: `from os import abort as die` then `die()` has no "abort("
# anywhere. The import spelling is the one text such a file must contain, so match that too.
_PREFILTER += ("import abort", "import string_at", "import raise_signal", "import kill")


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


# Matched after stripping underscores and case, so `_libc`, `LibC` and `libc` are one name.
_LIBC_NAMES = ("cdll", "ctypes", "libc")


def _names_libc(name) -> bool:
    return name.strip("_").lower() in _LIBC_NAMES


def _libc_aliases(tree):
    """Names bound to a real libc handle, e.g. `lib = ctypes.CDLL(None)`."""
    out = set()
    for node in ast.walk(tree):
        pair = _assigned_pair(node)
        if pair is None:
            continue
        target, value = pair
        if isinstance(value, ast.Call) and _is_libc_handle(value.func):
            out.add(target.id)
    return out


def _is_libc_handle(node, aliases = ()) -> bool:
    """Whether this expression is a real libc handle, not a stand-in named like one."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and (_names_libc(inner.id) or inner.id in aliases):
            return True
        if isinstance(inner, ast.Attribute) and _names_libc(inner.attr):
            return True
    return False


def _prctl_dumpable_value(node, libc = ()):
    """The value a prctl(PR_SET_DUMPABLE, v, ...) call sets, else None.

    The value argument matters: prctl(4, 1) re-enables dumps, so treating any
    PR_SET_DUMPABLE call as suppression would bless a crash that still dumps.
    """
    if _called_name(node) != "prctl" or len(node.args) < 2:
        return None
    # The receiver matters: a test's `fake.prctl(4, 1)` mock touches no kernel state, and crediting it let a mock
    # override the real suppression on the line above.
    if not isinstance(node.func, ast.Attribute) or not _is_libc_handle(node.func.value, libc):
        return None
    cmd, value = node.args[0], node.args[1]
    cmd_ok = (isinstance(cmd, ast.Constant) and cmd.value == _PR_SET_DUMPABLE) or (
        isinstance(cmd, ast.Name) and cmd.id.endswith("PR_SET_DUMPABLE")
    )
    if not cmd_ok or not isinstance(value, ast.Constant) or value.value not in (0, 1):
        return None
    return value.value


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

    One flat map let an unrelated `SCRIPT = "print(1)"` overwrite a module-level
    `SCRIPT` that really crashes. Functions are seeded from the module bindings and
    shadow them locally; concatenations fold, so `_SUPPRESS + "string_at(0)"` is
    judged as what it becomes.
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


def _iter_executable(scope, enter_classes = True):
    """Nodes that run when `scope` runs, skipping bodies that need a separate call.

    Walking everything treated a call inside an uninvoked nested `def` as having
    already run, so a helper that suppresses cores blessed a crash it never covered.
    """
    for child in ast.iter_child_nodes(scope):
        # Not a class body: it runs the moment the class is defined.
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if not enter_classes and isinstance(child, ast.ClassDef):
            continue
        yield child
        yield from _iter_executable(child, enter_classes)


def _rebound_names(scope):
    """Names this scope binds itself, which therefore no longer mean the import."""
    out = set()
    # Class bodies excluded: `class C: abort = ...` binds C.abort, not abort.
    for node in _iter_executable(scope, enter_classes = False):
        pair = _assigned_pair(node)
        if pair is not None:
            out.add(pair[0].id)
    for node in ast.walk(scope.args) if hasattr(scope, "args") else ():
        if isinstance(node, ast.arg):
            out.add(node.arg)
    return out


def _sequence_env(tree, owner):
    """Name to the elements of a list/tuple it is bound to.

    Separate from `_string_env`, which folds to a string; a command vector does not.
    Flat by name rather than per scope: a collision only adds a candidate string to
    read, which is cheaper than missing the script a child runs.
    """
    out = {}
    for node in ast.walk(tree):
        pair = _assigned_pair(node)
        if pair is None:
            continue
        target, value = pair
        if isinstance(value, (ast.List, ast.Tuple)):
            out.setdefault(target.id, []).extend(value.elts)
    return out


def _snippets(tree):
    """Strings that actually reach a child interpreter.

    Only a call argument counts, directly or via a name, list/tuple or concatenation,
    as in `subprocess.run([exe, "-c", SCRIPT])`. Counting every literal flagged
    ordinary assertions over a string nothing executes.
    """
    owner, module_env, scoped = _string_env(tree)
    sequences = _sequence_env(tree, owner)
    out = []

    def collect(
        node,
        env,
        depth = 0,
    ):
        # A deeply nested literal is not a command vector, and recursing all the way into one raised RecursionError out
        # of a file the scan only wanted to skim.
        if depth > _MAX_COLLECT_DEPTH:
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            # An f-string reaches the child as a script like any other. Keep its literal parts: the interpolations
            # cannot be known here, and dropping the whole thing let `f"import os; os.abort(); print({v})"` through
            # unread.
            out.append(
                "".join(
                    part.value
                    for part in node.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
            )
        elif isinstance(node, ast.Name):
            if node.id in env:
                out.append(env[node.id])
            elif node.id in sequences:
                # `CMD = [sys.executable, "-c", SCRIPT]` then `subprocess.run(CMD)`.
                for element in sequences[node.id]:
                    collect(element, env, depth + 1)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                collect(element, env, depth + 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            folded, _ = _fold(node, env)
            if folded is not None:
                out.append(folded)
            else:
                collect(node.left, env, depth + 1)
                collect(node.right, env, depth + 1)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        env = scoped.get(id(owner.get(id(node), tree)), module_env)
        for argument in list(node.args) + [kw.value for kw in node.keywords]:
            collect(argument, env)
    return out


def _crash_aliases(tree):
    """Names imported directly from a crashing module, e.g. `from os import abort`.

    A bare `abort()` is only fatal from `os` or `ctypes`; Playwright's `route.abort()`
    keeps its receiver and stays ignored. Keyed by the call-site name and valued by the
    name the rules use, since `from os import abort as die` binds `die`, which finds
    nothing in `_CRASH_CALLS`.
    """
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.split(".")[0] not in ("os", "ctypes", "signal", "faulthandler"):
            continue
        for alias in node.names:
            if alias.name in _CRASH_CALLS or alias.name in _DIRECTED_NAMES:
                out[alias.asname or alias.name] = alias.name
    return out


def _is_crash_call(node, aliases = None) -> bool:
    """A call that deliberately takes a core-dumping signal, written as code."""
    aliases = aliases or {}
    if not isinstance(node, ast.Call):
        return False
    name = _called_name(node)
    if name is None:
        return False
    # `from os import abort as die` binds `die`; the rules are written against `abort`.
    if isinstance(node.func, ast.Name) and name in aliases:
        name = aliases[name]
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
    # A string delivers nothing: `raise_signal("SIGQUIT")` is a TypeError, and the quoted name survives unparsing and
    # matched as though it were the symbol.
    if isinstance(signal_argument, ast.Constant) and isinstance(signal_argument.value, str):
        return False
    rendered = ast.unparse(signal_argument)
    if _FATAL_SIGNAL_RE.search(rendered):
        return True
    # Numeric signals. `signal.raise_signal(11)` and `os.kill(pid, 6)` dump exactly the same core as the named forms,
    # so matching only symbolic names missed them.
    return (
        isinstance(signal_argument, ast.Constant) and signal_argument.value in _FATAL_SIGNAL_NUMBERS
    )


def _enclosing_scopes(tree):
    """Map each node id to the nearest enclosing function, else the module."""
    owner = {}

    def walk(scope, node):
        for child in ast.iter_child_nodes(node):
            is_scope = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            next_scope = child if is_scope else scope
            owner[id(child)] = next_scope
            walk(next_scope, child)
            if not is_scope:
                continue
            # Defaults and annotations run where the def sits, so they belong to the enclosing scope.
            # A further scope nested inside one of them keeps its own.
            for part in _definition_time(child):
                for inner in ast.walk(part):
                    if owner.get(id(inner)) is child:
                        owner[id(inner)] = scope

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


_BRANCHING = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
    ast.BoolOp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
)


def _child_paths(scope, certain):
    """`(child, certain)`. A branch's body may not run, but its `finally` always does."""
    branching = isinstance(scope, _BRANCHING)
    if isinstance(scope, ast.Try):
        always = scope.finalbody  # a finally runs whatever the try did
    elif isinstance(scope, ast.BoolOp):
        always = scope.values[:1]  # only the first operand of a short circuit
    elif isinstance(scope, (ast.If, ast.IfExp, ast.While)):
        always = (scope.test,)  # the condition is evaluated before either branch
    elif isinstance(scope, (ast.For, ast.AsyncFor, ast.comprehension)):
        always = (scope.iter,)  # the iterable is evaluated before the first pass
    elif isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        always = scope.generators[:1]  # the outermost clause runs; its body may not
    else:
        always = ()
    for child in ast.iter_child_nodes(scope):
        yield child, certain and (not branching or any(child is node for node in always))


def _definition_time(node):
    """Parts of a def that run where it sits: defaults and decorators, not the body."""
    defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
    # Annotations run here too, unless `from __future__ import annotations` is on.
    annotations = [
        argument.annotation
        for argument in ast.walk(node.args)
        if isinstance(argument, ast.arg) and argument.annotation is not None
    ]
    returns = [node.returns] if getattr(node, "returns", None) is not None else []
    return defaults + annotations + returns + list(getattr(node, "decorator_list", ()))


def _dumpable_writes(
    scope,
    certain = True,
    functions = None,
    shadowed = (),
    libc = (),
):
    """`(position, value, certain)` for each prctl dumpability write on this path.

    With `functions`, a call to a local helper counts too, at the call's position, as
    whatever that helper leaves dumpability set to.
    """

    def written(node):
        value = _prctl_dumpable_value(node, libc)
        if value is None and functions is not None:
            value = _helper_leaves_dumpable(node, scope, functions, shadowed, libc)
        return value

    for child, child_certain in _child_paths(scope, certain):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # The body waits for a call, but defaults and decorators run right here.
            for part in _definition_time(child):
                if isinstance(part, ast.Call) and written(part) is not None:
                    yield _position(part), written(part), child_certain
                yield from _dumpable_writes(part, child_certain, functions, shadowed, libc)
            continue
        if isinstance(child, ast.Call):
            value = written(child)
            if value is not None:
                yield _position(child), value, child_certain
        yield from _dumpable_writes(child, child_certain, functions, shadowed, libc)


def _helper_leaves_dumpable(
    call,
    scope,
    functions,
    shadowed = (),
    libc = (),
):
    """What a bare call to a local helper leaves dumpability at, else None."""
    # Bare calls only, as in _suppressed: `obj.restore()` shares its trailing name with a local `def restore` but need
    # not be it.
    if not isinstance(call.func, ast.Name):
        return None
    target = functions.get(call.func.id)
    if target is None:
        return None
    # A local `restore = lambda: None` before the call is not the module-level helper.
    if call.func.id in shadowed:
        return None
    # Calling an `async def` builds a coroutine and runs none of its body.
    if isinstance(target, ast.AsyncFunctionDef) and not _is_awaited(call, scope):
        return None
    # Body only: a default or decorator on the helper ran at definition time, so it is not something calling the helper
    # does again.
    writes = [
        w
        for statement in target.body
        for w in _dumpable_writes(statement, libc = libc)
        if w[2] or w[1] == 0
    ]
    return writes[-1][1] if writes else None


def _clears_dumpable_before(
    scope,
    position,
    inherited = False,
    functions = None,
    libc = (),
) -> bool:
    """A prctl(4, 0, ...) on this scope's own path that runs before `position`.

    Order matters. Suppression placed after the fault does nothing, so accepting it
    anywhere in the scope blessed a child that still dumps.
    """
    # A name this scope rebinds itself is not the module-level helper of that name.
    shadowed = _rebound_names(scope) if hasattr(scope, "body") else ()
    writes = sorted(
        w
        for w in _dumpable_writes(scope, functions = functions, shadowed = shadowed, libc = libc)
        if w[0] < position
    )
    # Only a write that certainly runs decides: a conditional restore may never run.
    # A conditional clear still counts: platform-guarded prctl is the documented shape.
    decisive = [w for w in writes if w[2] or w[1] == 0]
    if decisive:
        return decisive[-1][1] == 0
    return inherited


def _suppressed(
    node,
    scope,
    functions,
    inherited = False,
    libc = (),
) -> bool:
    """Whether this crash call is covered, directly or by a helper it calls first."""
    position = _position(node)
    if _clears_dumpable_before(scope, position, inherited, functions, libc):
        return True
    # Following one level of local helper covers `suppress_core()` then the fault, which is the natural shape once more
    # than one test needs this.
    for called in _iter_executable(scope):
        if not isinstance(called, ast.Call) or _position(called) >= position:
            continue
        # Bare calls only. `obj.suppress_core()` shares its trailing name with a local `def suppress_core`, and
        # crediting the local one there means an object method that may clear nothing at all is taken as proof the fault
        # is covered.
        if not isinstance(called.func, ast.Name):
            continue
        target = functions.get(called.func.id)
        if target is None:
            continue
        # Calling an `async def` builds a coroutine and runs none of its body, so the
        # prctl never happens. Only an awaited one has actually cleared dumpability.
        if isinstance(target, ast.AsyncFunctionDef) and not _is_awaited(called, scope):
            continue
        if _clears_dumpable_before(target, _AFTER_EVERYTHING, libc = libc):
            return True
    return False


def _is_awaited(call, scope) -> bool:
    """Whether `call` is the operand of an `await`."""
    for node in ast.walk(scope):
        if isinstance(node, ast.Await) and node.value is call:
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
    shadowed = rebound[id(scope)]
    return {bound: original for bound, original in aliases.items() if bound not in shadowed}


def _unsuppressed_crashes(tree, inherited = False):
    """Crash calls in this tree whose core dump is not suppressed first."""
    owner = _enclosing_scopes(tree)
    functions = None
    libc = _libc_aliases(tree)
    aliases = _crash_aliases(tree)
    rebound = {}
    out = []
    for node in ast.walk(tree):
        scope = owner.get(id(node), tree)
        if not _is_crash_call(node, _live_aliases(aliases, scope, rebound)):
            continue
        if functions is None:
            functions = _functions_by_name(tree)
        if not _suppressed(node, scope, functions, inherited, libc):
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


def _is_builtin_exec(func) -> bool:
    if isinstance(func, ast.Name):
        return func.id in _NESTED_EXEC
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _NESTED_EXEC
        and isinstance(func.value, ast.Name)
        and func.value.id == "builtins"
    )


_MAX_SNIPPET_DEPTH = 5


def _bindings_before(tree, scope, position):
    """`(env, maybe)` for `scope` at `position`.

    `env` holds what a name is definitely bound to there. `maybe` collects values a
    name might still hold, because a rebind under a branch may not have run: dropping
    the old value on `if False: INNER = "pass"` lost the crash it replaced.
    """
    env, maybe = {}, {}
    # Python binds a name locally for the whole function if it is assigned anywhere in it, so a global of that name is
    # never what the body reads, even above the assign.
    shadowed = _rebound_names(scope) if scope is not tree else ()
    for owner_scope in (tree, scope) if scope is not tree else (tree,):
        # A nested scope runs after the module body, so a global assigned below the `def` is still bound by the time the
        # call gets there.
        limit = _AFTER_EVERYTHING if owner_scope is tree and scope is not tree else position
        for node, certain in _assignments_before(owner_scope, limit):
            pair = _assigned_pair(node)
            if owner_scope is tree and pair[0].id in shadowed:
                continue
            folded, _ids = _fold(pair[1], env)
            if not certain:
                if folded is not None:
                    maybe.setdefault(pair[0].id, []).append(folded)
                continue
            # A certain rebind definitely replaces what was there, foldable or not.
            env.pop(pair[0].id, None)
            maybe.pop(pair[0].id, None)
            if folded is not None:
                env[pair[0].id] = folded
    return env, maybe


def _assignments_before(
    scope,
    position,
    certain = True,
):
    """`(assignment, certain)` in source order, before `position`."""
    for child, child_certain in _child_paths(scope, certain):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if _assigned_pair(child) is not None and _position(child) < position:
            yield child, child_certain
        yield from _assignments_before(child, position, child_certain)


def _nested_scripts(tree, inherited = False):
    """Source a snippet hands to exec/eval, or on to another child interpreter.

    `SCRIPT = 'exec("import os; os.abort()")'` parses cleanly and holds no crash call
    of its own, so without this the crash one level down was never looked at.
    """
    owner = _enclosing_scopes(tree)
    functions = _functions_by_name(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scope = owner.get(id(node), tree)
        # The builtin only, bare or via `builtins`: `Runner().exec(...)` is not it.
        if _is_builtin_exec(node.func):
            argument = node.args[0] if node.args else None
            # Bindings AT the exec, so a name reused afterwards is not what runs.
            env, maybe = _bindings_before(tree, scope, _position(node))
            # Same interpreter, so dumpability carries into the payload, including a restore a helper made between the
            # clear and the exec.
            carried = _clears_dumpable_before(
                scope,
                _position(node),
                inherited = inherited,
                functions = functions,
                libc = _libc_aliases(tree),
            )
            payloads = []
            if argument is not None:
                folded, _ids = _fold(argument, env)
                if folded is not None:
                    payloads.append(folded)
                if isinstance(argument, ast.Name):
                    # A rebind that may not have run leaves the old value possible.
                    payloads.extend(maybe.get(argument.id, ()))
            for payload in payloads:
                yield payload, carried
        else:
            for nested in _snippets_of_call(node):
                yield nested, False


def _snippets_of_call(node):
    """Script strings passed to a nested subprocess call, e.g. a `-c` argument."""
    for argument in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(argument, (ast.List, ast.Tuple)):
            for element in argument.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    yield element.value


def _snippet_state(
    snippet: str,
    depth: int = 0,
    inherited: bool = False,
):
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
            any(m in snippet for m in _SIGNAL_DIRECTED) and _FATAL_SIGNAL_RE.search(snippet)
        )
        suppressed = inherited or "prctl(4, 0" in snippet or "PR_SET_DUMPABLE, 0" in snippet
        return crashes, crashes and not suppressed
    crashes, violates = _tree_crashes(tree), bool(_unsuppressed_crashes(tree, inherited))
    if depth < _MAX_SNIPPET_DEPTH:
        for nested, carried in _nested_scripts(tree, inherited):
            inner_crashes, inner_violates = _snippet_state(nested, depth + 1, carried)
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


# Each of these is a way the detector was fooled before, kept as a fixture so the fix stays fixed.
# `want_violations` is whether the file should be reported.
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
    "command_vector_bound_to_a_name": (
        "import subprocess, sys\n"
        'CMD = [sys.executable, "-c", "import ctypes; ctypes.string_at(0)"]\n'
        "subprocess.run(CMD)\n",
        True,  # a command list built once and passed by name is ordinary subprocess use
    ),
    "crash_imported_under_an_alias": (
        "from os import abort as die\ndef child():\n    die()\n",
        True,  # the bound name is `die`; the rules are written against `abort`
    ),
    "aliased_string_at": (
        "from ctypes import string_at as boom\ndef child():\n    boom(0)\n",
        True,
    ),
    "method_sharing_a_helper_name_is_not_suppression": (
        "import ctypes\n"
        "def suppress_core():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "def child(obj):\n"
        "    obj.suppress_core()\n"
        "    ctypes.string_at(0)\n",
        True,  # the object's method may clear nothing; only a bare local call counts
    ),
    "async_suppressor_must_be_awaited": (
        "import ctypes\n"
        "async def suppress_core():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "def child():\n"
        "    suppress_core()\n"
        "    ctypes.string_at(0)\n",
        True,  # calling it only builds a coroutine, so the prctl never runs
    ),
    "awaited_async_suppressor_counts": (
        "import ctypes\n"
        "async def suppress_core():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "async def child():\n"
        "    await suppress_core()\n"
        "    ctypes.string_at(0)\n",
        False,
    ),
    "rebound_alias_is_not_a_crash": (
        "from os import abort\ndef child():\n    abort = lambda: None\n    abort()\n",
        False,
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
    # Seven more the detector got wrong, each found by reading it rather than by a failing run. Four let a real core
    # dump through; three failed CI on safe code.
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
    # Six the guard still gets wrong, each verified against the current file.
    "sigquit_is_a_core_dumping_signal": (
        "import signal\ndef child():\n    signal.raise_signal(3)\n",
        True,  # SIGQUIT dumps core by default
    ),
    "class_body_runs_with_its_enclosing_scope": (
        "import ctypes\n"
        "class Probe:\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # unlike a def, a class body executes immediately
    ),
    "rebinding_inside_an_unrelated_function": (
        "from os import abort\n"
        "abort()\n"
        "def unrelated(mock):\n    abort = mock\n    return abort\n",
        True,  # a nested local must not disarm the module-level alias
    ),
    "dumpability_restored_before_the_crash": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        True,  # the setting nearest the crash is the one that counts
    ),
    "exec_payload_reached_by_name": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'import os; os.abort()'\\nexec(INNER)\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # the payload is one name away, and still runs
    ),
    "suppression_above_a_nested_exec": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        "exec('import os; os.abort()')\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # same interpreter, so the prctl above covers the nested crash
    ),
    # Six more, each one a way the fixes above were themselves wrong.
    "class_attribute_shadowing_a_crash_alias": (
        "from os import abort\nclass C:\n    abort = lambda: None\nabort()\n",
        True,  # that binds C.abort; the module name still crashes
    ),
    "restore_inside_a_branch_that_may_not_run": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    if False:\n        ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # a conditional restore cannot be assumed to have run
    ),
    "platform_guarded_suppression": (
        "import ctypes, sys\n"
        "def child():\n"
        '    if sys.platform == "linux":\n'
        "        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # the documented shape, since prctl is Linux-only
    ),
    "exec_payload_restores_dumpability": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        "exec('import ctypes, os; ctypes.CDLL(None).prctl(4, 1, 0, 0, 0); os.abort()')\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # inherited suppression is a starting state, not a blanket pass
    ),
    "helper_restores_dumpability_before_the_exec": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes\\n'
        "def restore():\\n    ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\\n"
        "ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"
        "restore()\\n"
        "exec('import os; os.abort()')\\n\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # the helper put dumping back, so the payload's abort does dump
    ),
    "exec_payload_name_reused_afterwards": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'pass'\\nexec(INNER)\\nINNER = 'import os; os.abort()'\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # exec runs what the name held at the time
    ),
    "signal_name_only_as_a_substring": (
        "import signal\n"
        "SIGQUIT_HANDLER = signal.SIGTERM\n"
        "def child():\n    signal.raise_signal(SIGQUIT_HANDLER)\n",
        False,  # SIGTERM does not dump, whatever the variable is called
    ),
    "prctl_value_the_kernel_rejects": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.CDLL(None).prctl(4, 2, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # PR_SET_DUMPABLE takes 0 or 1; 2 is EINVAL and changes nothing
    ),
    "restore_inside_an_unmatched_case": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    match 0:\n        case 1:\n"
        "            ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # a match arm is a branch, so it may never run
    ),
    "method_named_exec_is_not_the_builtin": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'import os; os.abort()'\\nRunner().exec(INNER)\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # only a bare exec/eval runs the string it is handed
    ),
    "payload_rebound_to_something_unfoldable": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'import os; os.abort()'\\nINNER = str('pass')\\nexec(INNER)\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # the rebind wins even when its value cannot be folded
    ),
    "payload_rebound_only_in_a_branch": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'import os; os.abort()'\\nif False:\\n"
        "    INNER = 'pass'\\nexec(INNER)\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # a rebind that may not run leaves the old payload possible
    ),
    "guarded_clear_after_a_definite_restore": (
        "import ctypes, sys\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        '    if sys.platform == "linux":\n'
        "        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # the guarded clear is the documented shape and comes last
    ),
    "global_assigned_below_the_exec": (
        "import subprocess, sys\n"
        'SCRIPT = "def run():\\n    exec(INNER)\\n'
        "INNER = 'import os; os.abort()'\\nrun()\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # the body runs after the module, so the later global is bound
    ),
    "branch_candidate_then_a_definite_rebind": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'pass'\\nif cond:\\n    INNER = 'import os; os.abort()'\\n"
        "INNER = 'pass'\\nexec(INNER)\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # the definite rebind rules out the branch value it replaced
    ),
    "helper_guarded_clear_before_a_nested_exec": (
        "import subprocess, sys\n"
        "SCRIPT = \"import ctypes, sys\\ndef clear():\\n    if sys.platform == 'linux':\\n"
        "        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\nclear()\\n"
        "exec('import os; os.abort()')\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # a guarded clear counts the same inside a helper as inline
    ),
    "restore_in_a_finally_before_a_later_crash": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    try:\n        pass\n"
        "    finally:\n        ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        True,  # reaching the crash proves the finally ran and undid the clear
    ),
    "inherited_dumpability_through_two_execs": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        'exec(\\"exec(\'import os; os.abort()\')\\")"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # one interpreter throughout, so the clear reaches both levels
    ),
    "builtins_exec_payload": (
        "import subprocess, sys\n"
        "SCRIPT = \"import builtins\\nbuiltins.exec('import os; os.abort()')\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # builtins.exec is the builtin, spelled out
    ),
    "payload_name_is_local_to_the_function": (
        "import subprocess, sys\n"
        "SCRIPT = \"INNER = 'import os; os.abort()'\\ndef run():\\n    exec(INNER)\\n"
        "    INNER = 'pass'\\nrun()\"\n"
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        False,  # assigning INNER anywhere makes it local, so the global never applies
    ),
    "restore_in_a_short_circuited_operand": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    False and ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # the operand never evaluates, so the clear still stands
    ),
    "restore_in_a_method_default": (
        "import ctypes\n"
        "class C:\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    def f(x = ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)):\n        pass\n"
        "    ctypes.string_at(0)\n",
        True,  # a default runs where the def sits, so the restore beats the crash
    ),
    "crash_alias_as_a_lambda_parameter": (
        "from os import abort\nf = lambda abort: abort()\nf(mock)\n",
        False,  # the parameter shadows the import inside the lambda
    ),
    "signal_name_passed_as_a_string": (
        "import signal\n" "def child():\n" '    signal.raise_signal("SIGQUIT")\n',
        False,  # a string is a TypeError, and delivers no signal
    ),
    "helper_restore_inside_an_inherited_payload": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        'exec(\\"import ctypes, os\\\\ndef restore():\\\\n    '
        'ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\\\\nrestore()\\\\nos.abort()\\")"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # an inherited clear does not outrank a helper restore in the payload
    ),
    "restore_in_a_parameter_annotation": (
        "import ctypes\n"
        "class C:\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    def f(x: ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)):\n        pass\n"
        "    ctypes.string_at(0)\n",
        True,  # an annotation evaluates where the def sits, like a default
    ),
    "lambda_default_runs_in_the_enclosing_scope": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    f = lambda x = ctypes.string_at(0): None\n",
        False,  # the default belongs to the caller, which cleared first
    ),
    "restore_in_an_if_condition": (
        "import subprocess, sys\n"
        'SCRIPT = "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); '
        'exec(\\"import ctypes, os\\\\nif ctypes.CDLL(None).prctl(4, 1, 0, 0, 0):'
        '\\\\n    pass\\\\nos.abort()\\")"\n'
        'subprocess.run([sys.executable, "-c", SCRIPT])\n',
        True,  # the condition runs before either branch, so the restore is certain
    ),
    "restore_inside_a_generator_expression": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    gen = (ctypes.CDLL(None).prctl(4, 1, 0, 0, 0) for _ in range(1))\n"
        "    ctypes.string_at(0)\n",
        False,  # a generator body does not run at construction
    ),
    "mocked_prctl_on_an_unrelated_object": (
        "import ctypes\n"
        "def child(fake):\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    fake.prctl(4, 1)\n"
        "    ctypes.string_at(0)\n",
        False,  # a mock named prctl touches no kernel state
    ),
    "local_rebinding_of_a_helper_name": (
        "import ctypes\n"
        "def restore():\n"
        "    ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    restore = lambda: None\n"
        "    restore()\n"
        "    ctypes.string_at(0)\n",
        False,  # the local binding is not the module-level helper
    ),
    "aliased_libc_handle_still_suppresses": (
        "import ctypes\n"
        "lib = ctypes.CDLL(None)\n"
        "def child():\n"
        "    lib.prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # a handle bound to any name is still the real libc
    ),
    "outer_comprehension_iterable_is_certain": (
        "import ctypes\n"
        "def child():\n"
        "    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\n"
        "    xs = [i for i in [ctypes.CDLL(None).prctl(4, 1, 0, 0, 0)]]\n"
        "    ctypes.string_at(0)\n",
        True,  # the outermost iterable is evaluated immediately
    ),
    "imported_libc_handle_attribute": (
        "import ctypes, tools\n"
        "def child():\n"
        "    tools._libc.prctl(4, 0, 0, 0, 0)\n"
        "    ctypes.string_at(0)\n",
        False,  # tools._libc.prctl is the convention in test_bypass_permissions.py
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
