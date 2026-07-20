#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Deterministic, scope-aware verifier for import-hoisting / alias-rename refactors.

The risk when moving `from a import b as _b` (or `import b as _b`) to module top
and normalizing `_b` -> `b` is twofold:

  1. DANGLING ALIAS  - a `_b` reference is left un-normalized; it now resolves to
     nothing (NameError) or, worse, to some *other* module-level `_b`.
  2. RENAME CLASH    - `_b` was an alias on purpose because `b` already meant
     something else in that scope; normalizing `_b` -> `b` silently re-points the
     reference at the wrong object (no NameError, no pyflakes warning).

This tool parses BEFORE (a git ref, default origin/main) and AFTER (default HEAD)
for each file, builds a real LEGB scope model (functions, classes, lambdas,
comprehensions, global/nonlocal, args, walrus, star-imports), and resolves every
Name load to its binding. It then compares, PER SCOPE:

  * UNRESOLVED-NEW   : loads that resolve to nothing in AFTER but did in BEFORE
                       (or are newly present) -> catches dangling aliases.
  * TARGET-MISSING   : an import *target* (e.g. module `glob`, or
                       `importlib.metadata.version`) that a function resolved to
                       in BEFORE but no longer resolves to in AFTER -> catches a
                       function that lost access to a module it still uses.
                       Robust to alias renames because it compares the *target*,
                       not the local name.
  * TARGET-CHANGED   : a load whose resolved import target differs BEFORE vs
                       AFTER -> catches a rename that re-points to a different
                       module (the clash case).
  * AMBIGUOUS-BIND   : a name bound by BOTH an import and a non-import in the same
                       scope in AFTER (and not in BEFORE) -> the "alias was on
                       purpose / now collides" smell.
  * MODULE-DUP-IMPORT: a module-level name imported and also defined/assigned at
                       module level (introduced by the change).
  * NEW-UNUSED-IMPORT: a module-level import added in AFTER that nothing resolves
                       to (informational; re-exports are a known false positive).

Usage:
  verify_import_hoist.py [--before REF] [--after REF] <file>...   # compare
  verify_import_hoist.py --self-test                              # prove it catches bugs
Exit code 1 if any non-informational finding.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re as _re_mod
import subprocess
import sys
from dataclasses import dataclass, field

_BUILTINS = set(dir(builtins)) | {
    "__file__",
    "__name__",
    "__doc__",
    "__package__",
    "__spec__",
    "__loader__",
    "__builtins__",
    "__class__",
    "__annotations__",
    "__dict__",
    "__qualname__",
    "__module__",
    "__path__",
    "__debug__",
    "__import__",
    "NotImplemented",
    "Ellipsis",
    "copyright",
    "credits",
    "license",
    "help",
    "exit",
    "quit",
    "__build_class__",
    "__cached__",
    "reveal_type",
    "reveal_locals",
}


# ---------------------------------------------------------------- scope model


@dataclass
class Binding:
    kind: str  # 'import' | 'importfrom' | 'def' | 'class' | 'other'
    target: str | None = None  # canonical import target id, else None


@dataclass
class Scope:
    kind: str  # 'module' | 'function' | 'class' | 'lambda' | 'comp'
    qualname: str
    parent: "Scope | None"
    bindings: dict[str, list[Binding]] = field(default_factory = dict)
    globals: set[str] = field(default_factory = set)
    nonlocals: set[str] = field(default_factory = set)
    star_import: bool = False

    def add(self, name: str, b: Binding) -> None:
        self.bindings.setdefault(name, []).append(b)


def _import_target(node: ast.AST, alias: ast.alias) -> tuple[str, str]:
    """Return (bound_name, canonical_target_id) for one import alias."""
    if isinstance(node, ast.Import):
        bound = alias.asname or alias.name.split(".")[0]
        return bound, f"import:{alias.name}"
    # ImportFrom
    bound = alias.asname or alias.name
    mod = ("." * (node.level or 0)) + (node.module or "")
    return bound, f"from:{mod}:{alias.name}"


class _Builder(ast.NodeVisitor):
    """Builds the scope tree + bindings, and records every (scope, Name-load)."""

    def __init__(self):
        self.module = Scope("module", "<module>", None)
        self.uses: list[tuple[Scope, str, int]] = []  # hard loads
        # annotations: count as "used" but never as "unresolved" (forward refs)
        self.soft_uses: list[tuple[Scope, str, int]] = []

    def _visit_annotation(self, node, scope: Scope) -> None:
        """Record annotation names as SOFT uses: an import used only in an annotation
        counts as used, but a forward-ref name is never 'unresolved'."""
        if node is None:
            return
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                self.soft_uses.append((scope, n.id, n.lineno))

    # -- binding helpers --
    def _bind_targets(self, scope: Scope, target: ast.AST) -> None:
        for n in ast.walk(target):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                self._bind_name(scope, n.id, Binding("other"))
            elif isinstance(n, ast.Starred):
                pass

    def _bind_name(self, scope: Scope, name: str, b: Binding) -> None:
        if name in scope.globals:
            self.module.add(name, b)
        elif name in scope.nonlocals:
            p = scope.parent
            while p is not None and p.kind not in ("function", "lambda"):
                p = p.parent
            (p or self.module).add(name, b)
        else:
            scope.add(name, b)

    # -- generic dispatch within a scope --
    def _visit_body(self, stmts, scope: Scope) -> None:
        for s in stmts:
            self._visit_stmt(s, scope)

    def _visit_stmt(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            star = isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
            if star:
                scope.star_import = True
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound, target = _import_target(node, alias)
                kind = "import" if isinstance(node, ast.Import) else "importfrom"
                self._bind_name(scope, bound, Binding(kind, target))
            return
        if isinstance(node, ast.Global):
            scope.globals.update(node.names)
            return
        if isinstance(node, ast.Nonlocal):
            scope.nonlocals.update(node.names)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._bind_name(scope, node.name, Binding("def"))
            # decorators / defaults evaluate in the ENCLOSING scope
            for d in node.decorator_list:
                self._visit_expr(d, scope)
            self._visit_arg_defaults(node.args, scope)
            child = Scope("function", f"{scope.qualname}.{node.name}", scope)
            self._bind_type_params(node, child)
            self._bind_args(node.args, child)
            # arg + return annotations: soft uses
            for a in self._all_args(node.args):
                self._visit_annotation(a.annotation, child)
            self._visit_annotation(getattr(node, "returns", None), child)
            self._visit_body(node.body, child)
            return
        if isinstance(node, ast.ClassDef):
            self._bind_name(scope, node.name, Binding("class"))
            for d in node.decorator_list:
                self._visit_expr(d, scope)
            for b in node.bases:
                self._visit_expr(b, scope)
            for kw in node.keywords:
                self._visit_expr(kw.value, scope)
            child = Scope("class", f"{scope.qualname}.{node.name}", scope)
            self._bind_type_params(node, child)
            self._visit_body(node.body, child)
            return
        if isinstance(node, ast.Match):
            self._visit_expr(node.subject, scope)
            for case in node.cases:
                self._bind_pattern(case.pattern, scope)
                if case.guard is not None:
                    self._visit_expr(case.guard, scope)
                self._visit_body(case.body, scope)
            return
        if isinstance(node, getattr(ast, "TryStar", ())):  # py3.11 except*
            self._visit_body(node.body, scope)
            for h in node.handlers:
                if h.type is not None:
                    self._visit_expr(h.type, scope)
                if h.name:
                    self._bind_name(scope, h.name, Binding("other"))
                self._visit_body(h.body, scope)
            self._visit_body(node.orelse, scope)
            self._visit_body(node.finalbody, scope)
            return
        if isinstance(node, getattr(ast, "TypeAlias", ())):  # py3.12 `type X = ...`
            if isinstance(node.name, ast.Name):
                self._bind_name(scope, node.name.id, Binding("other"))
            self._visit_annotation(node.value, scope)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = node.value
            if val is not None:
                self._visit_expr(val, scope)
            if isinstance(node, ast.AnnAssign) and node.annotation is not None:
                self._visit_annotation(node.annotation, scope)
            for t in targets:
                self._bind_targets(scope, t)
                # AugAssign target is also a load
                if isinstance(node, ast.AugAssign):
                    self._record_loads(t, scope)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self._visit_expr(node.iter, scope)
            self._bind_targets(scope, node.target)
            self._visit_body(node.body, scope)
            self._visit_body(node.orelse, scope)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self._visit_expr(item.context_expr, scope)
                if item.optional_vars is not None:
                    self._bind_targets(scope, item.optional_vars)
            self._visit_body(node.body, scope)
            return
        if isinstance(node, ast.Try):
            self._visit_body(node.body, scope)
            for h in node.handlers:
                if h.type is not None:
                    self._visit_expr(h.type, scope)
                if h.name:
                    self._bind_name(scope, h.name, Binding("other"))
                self._visit_body(h.body, scope)
            self._visit_body(node.orelse, scope)
            self._visit_body(node.finalbody, scope)
            return
        # generic statement: visit all child expressions/stmts in same scope
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._visit_stmt(child, scope)
            else:
                self._visit_expr(child, scope)

    # -- expressions --
    def _visit_arg_defaults(self, args: ast.arguments, scope: Scope) -> None:
        for d in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            self._visit_expr(d, scope)

    def _all_args(self, args: ast.arguments) -> list[ast.arg]:
        out = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        if args.vararg:
            out.append(args.vararg)
        if args.kwarg:
            out.append(args.kwarg)
        return out

    def _bind_args(self, args: ast.arguments, scope: Scope) -> None:
        for a in self._all_args(args):
            scope.add(a.arg, Binding("other"))

    def _bind_type_params(self, node, scope: Scope) -> None:
        for tp in getattr(node, "type_params", []) or []:
            name = getattr(tp, "name", None)
            if isinstance(name, str):
                scope.add(name, Binding("other"))
            self._visit_annotation(getattr(tp, "bound", None), scope)
            self._visit_annotation(getattr(tp, "default_value", None), scope)

    def _bind_pattern(self, pat, scope: Scope) -> None:
        if pat is None:
            return
        if isinstance(pat, ast.MatchValue):
            self._visit_expr(pat.value, scope)
        elif isinstance(pat, ast.MatchSingleton):
            pass
        elif isinstance(pat, ast.MatchSequence):
            for p in pat.patterns:
                self._bind_pattern(p, scope)
        elif isinstance(pat, ast.MatchStar):
            if pat.name:
                self._bind_name(scope, pat.name, Binding("other"))
        elif isinstance(pat, ast.MatchMapping):
            for k in pat.keys:
                self._visit_expr(k, scope)
            for p in pat.patterns:
                self._bind_pattern(p, scope)
            if pat.rest:
                self._bind_name(scope, pat.rest, Binding("other"))
        elif isinstance(pat, ast.MatchClass):
            self._visit_expr(pat.cls, scope)
            for p in pat.patterns:
                self._bind_pattern(p, scope)
            for p in pat.kwd_patterns:
                self._bind_pattern(p, scope)
        elif isinstance(pat, ast.MatchAs):
            self._bind_pattern(pat.pattern, scope)
            if pat.name:
                self._bind_name(scope, pat.name, Binding("other"))
        elif isinstance(pat, ast.MatchOr):
            for p in pat.patterns:
                self._bind_pattern(p, scope)

    def _record_loads(self, node: ast.AST, scope: Scope) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                self.uses.append((scope, n.id, n.lineno))

    def _visit_expr(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.uses.append((scope, node.id, node.lineno))
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                self._bind_name(scope, node.id, Binding("other"))
            return
        if isinstance(node, ast.Lambda):
            self._visit_arg_defaults(node.args, scope)
            child = Scope("lambda", f"{scope.qualname}.<lambda>", scope)
            self._bind_args(node.args, child)
            self._visit_expr(node.body, child)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            child = Scope("comp", f"{scope.qualname}.<comp>", scope)
            for i, gen in enumerate(node.generators):
                # first iterable evaluates in the enclosing scope
                self._visit_expr(gen.iter, scope if i == 0 else child)
                self._bind_targets(child, gen.target)
                for cond in gen.ifs:
                    self._visit_expr(cond, child)
            if isinstance(node, ast.DictComp):
                self._visit_expr(node.key, child)
                self._visit_expr(node.value, child)
            else:
                self._visit_expr(node.elt, child)
            return
        if isinstance(node, ast.NamedExpr):  # walrus binds in enclosing scope
            self._visit_expr(node.value, scope)
            if isinstance(node.target, ast.Name):
                self._bind_name(scope, node.target.id, Binding("other"))
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._visit_stmt(child, scope)
            else:
                self._visit_expr(child, scope)

    def run(self, tree: ast.Module) -> None:
        self._visit_body(tree.body, self.module)


# ---------------------------------------------------------------- resolution


def _any_star(scope: Scope) -> bool:
    c = scope
    while c is not None:
        if c.star_import:
            return True
        c = c.parent
    return False


def _resolve(scope: Scope, name: str):
    """LEGB resolution. Returns (status, bindings); status in
    {'local','import','other','builtin','star','unresolved'}."""
    start = scope
    if name in scope.globals:
        chain = [_module_of(scope)]
    elif name in scope.nonlocals:
        chain = _enclosing_functions(scope)
    else:
        chain = _legb_chain(scope)
    for i, sc in enumerate(chain):
        if sc is None:
            continue
        if name in sc.bindings:
            binds = sc.bindings[name]
            if any(b.kind in ("import", "importfrom") for b in binds):
                return "import", binds
            return "other", binds
    if name in _BUILTINS:
        return "builtin", []
    if _any_star(start):
        return "star", []
    return "unresolved", []


def _module_of(scope: Scope) -> Scope:
    while scope.parent is not None:
        scope = scope.parent
    return scope


def _enclosing_functions(scope: Scope) -> list[Scope]:
    out = []
    p = scope.parent
    while p is not None:
        if p.kind in ("function", "lambda"):
            out.append(p)
        p = p.parent
    out.append(_module_of(scope))
    return out


def _legb_chain(scope: Scope) -> list[Scope]:
    """Immediate scope, then enclosing scopes skipping class scopes, then module."""
    chain = [scope]
    p = scope.parent
    while p is not None:
        if p.kind != "class" or p.parent is None:  # skip class scopes, keep module
            if p.kind != "class":
                chain.append(p)
        p = p.parent
    return chain


# ---------------------------------------------------------------- analysis


def _collect_dunder_all(tree: ast.Module) -> tuple[set[str], bool]:
    """The FINAL module-level ``__all__`` string entries plus an ``opaque`` flag
    (``= [...]``, ``+= [...]``, or an annotated assign).

    A name listed in ``__all__`` is a public re-export, which is a real use of the
    import that binds it. ``__all__`` entries are string constants, not ``Name``
    loads, so the load-based use scan never sees them -- without this a package
    ``__init__`` that adds ``from .x import y`` purely to re-export ``y`` looks
    like an unused hoist and trips ``HOISTED-IMPORT-UNUSED``.

    Only the final value counts, so assignments are replayed in order: a plain ``=``
    REPLACES the list and ``+=`` extends it. Unioning every assignment instead would
    let ``__all__ = ["y"]`` followed by ``__all__ = []`` still mark ``y`` used, so a
    genuinely unused hoist would slip through.

    ``opaque`` is True when ``__all__`` is rebound OR extended by a value we cannot read
    statically (a call, a name, a comprehension, a spread). The static entry set is then
    not known to be exhaustive, so the caller must NOT treat a name's absence from it as
    proof the import is unexported -- a re-export supplied dynamically
    (``__all__ += _exports()``) would otherwise trip ``HOISTED-IMPORT-UNUSED``. An
    unreadable ``=`` and an unreadable ``+=`` set it alike, so the two stay consistent.
    """
    names: set[str] = set()
    opaque = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, replaces = node.targets, True
        elif isinstance(node, ast.AnnAssign):
            targets, replaces = [node.target], True
        elif isinstance(node, ast.AugAssign):
            targets, replaces = [node.target], False
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        entries: set[str] = set()
        readable = True  # every element is a statically-known string constant
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    entries.add(elt.value)
                else:
                    readable = False  # a spread / computed element: unknown extra exports
        else:
            readable = False  # a call, a name, a comprehension: contents unknown
        if not readable:
            # Contents partly/wholly unknown. Keep the names we can see, but mark the set
            # non-exhaustive so the caller preserves uncertainty rather than flagging a
            # dynamically-supplied re-export as unused. Applies to both `=` and `+=`.
            opaque = True
        if replaces:
            names = entries
        else:
            names |= entries
    return names, opaque


def _analyze(src: str):
    tree = ast.parse(src)
    b = _Builder()
    b.run(tree)
    # Per-scope: unresolved load names + import targets it resolves to.
    unresolved: dict[str, set[str]] = {}
    targets_by_scope: dict[str, set[str]] = {}
    target_by_use: dict[tuple[str, str], set[str]] = {}
    for scope, name, _ln in b.uses:
        status, binds = _resolve(scope, name)
        if status == "unresolved":
            unresolved.setdefault(scope.qualname, set()).add(name)
        elif status == "import":
            tids = {bd.target for bd in binds if bd.target}
            targets_by_scope.setdefault(scope.qualname, set()).update(tids)
            target_by_use.setdefault((scope.qualname, name), set()).update(tids)
    # soft uses (annotations): contribute to "used" only, never "unresolved"
    for scope, name, _ln in b.soft_uses:
        status, binds = _resolve(scope, name)
        if status == "import":
            tids = {bd.target for bd in binds if bd.target}
            targets_by_scope.setdefault(scope.qualname, set()).update(tids)
    # module-level binding info for clash checks
    module = b.module
    module_imports = {
        n: bs
        for n, bs in module.bindings.items()
        if any(x.kind in ("import", "importfrom") for x in bs)
    }
    # Re-exports count as uses: a name listed in module-level __all__ that is bound
    # by a module import is deliberately exported, not a dangling hoist. Fold its
    # targets into the used set so HOISTED-IMPORT-UNUSED does not fire on a
    # legitimately-added `from .x import y` in a package __init__. When __all__ is
    # opaque (a dynamically-computed part), the static list is not known to be
    # exhaustive, so credit EVERY module import rather than risk flagging a
    # dynamically-exported hoist as unused.
    _all_names, _all_opaque = _collect_dunder_all(tree)
    _reexported = module_imports.values() if _all_opaque else (
        _bs for _n in _all_names if (_bs := module_imports.get(_n))
    )
    for _bs in _reexported:
        targets_by_scope.setdefault(module.qualname, set()).update(
            x.target for x in _bs if x.target
        )
    module_dup = {
        n
        for n, bs in module.bindings.items()
        if any(x.kind in ("import", "importfrom") for x in bs)
        and any(x.kind not in ("import", "importfrom") for x in bs)
    }
    # ambiguous: any scope where a name is bound by import AND non-import
    ambiguous: dict[str, set[str]] = {}

    def walk_scopes(scope: Scope):
        for n, bs in scope.bindings.items():
            if any(x.kind in ("import", "importfrom") for x in bs) and any(
                x.kind not in ("import", "importfrom") for x in bs
            ):
                ambiguous.setdefault(scope.qualname, set()).add(n)
        # scope tree isn't stored; approximate with module only.

    walk_scopes(module)
    return {
        "unresolved": unresolved,
        "targets_by_scope": targets_by_scope,
        "target_by_use": target_by_use,
        "module_import_targets": {
            n: {x.target for x in bs if x.target} for n, bs in module_imports.items()
        },
        "module_dup": module_dup,
        "ambiguous": ambiguous,
    }


def _git_show(ref: str, path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{path}"], capture_output = True, text = True, check = True
        ).stdout
    except subprocess.CalledProcessError:
        return None


def compare(before_src: str, after_src: str, path: str) -> list[tuple[str, str]]:
    """Return list of (severity, message). severity in BLOCKER/WARN/INFO.

    Blocker signals (precise, no relocation false-positives):
      UNRESOLVED-NEW    - a load became undefined (dangling alias / removed import).
      NEW-UNUSED-HOIST  - a module-level import added by this change is resolved by
                          NO load (un-normalized alias or wrong rename target).
      TARGET-CHANGED    - same (scope, name) load resolves to a different import
                          target before vs after (a same-name re-point).
    """
    a = _analyze(before_src)
    b = _analyze(after_src)
    findings: list[tuple[str, str]] = []

    def used_targets(analysis) -> set[str]:
        out: set[str] = set()
        for tids in analysis["targets_by_scope"].values():
            out |= tids
        return out

    before_used = used_targets(a)
    after_used = used_targets(b)
    before_module_targets: set[str] = set()
    for tids in a["module_import_targets"].values():
        before_module_targets |= tids
    after_module_targets: set[str] = set()
    for tids in b["module_import_targets"].values():
        after_module_targets |= tids
    added_module_targets = after_module_targets - before_module_targets

    # 1. UNRESOLVED-NEW
    for scope, names in b["unresolved"].items():
        new = names - a["unresolved"].get(scope, set())
        for n in sorted(new):
            findings.append(
                (
                    "BLOCKER",
                    f"{path}: UNRESOLVED-NEW '{n}' in scope {scope} "
                    f"(undefined after change -> dangling alias / removed import)",
                )
            )

    # 2. HOISTED-IMPORT-UNUSED  (core botched-hoist / wrong-rename signal)
    #    A module-level import in AFTER that NO load resolves to, that was either
    #    newly added by this change OR actually used before. Excludes relocation
    #    (import removed) and stable pre-existing re-exports.
    for n, tids in b["module_import_targets"].items():
        if tids & after_used:
            continue  # resolved -> fine
        # `from __future__ import ...` is a compiler directive, not a runtime
        # binding: the name (`annotations`, ...) is never loaded, so it can never
        # "resolve" to a use. Skip it so a legitimately-added future import
        # (e.g. `annotations` for lazy PEP 604 `X | None` on py3.9) is not flagged.
        if all(t.startswith("from:__future__:") for t in tids):
            continue
        newly_added = bool(tids - before_module_targets)
        was_used_before = bool(tids & before_used)
        if newly_added or was_used_before:
            why = (
                "added but unused"
                if newly_added
                else "was used before, now unused (references re-pointed)"
            )
            findings.append(
                (
                    "BLOCKER",
                    f"{path}: HOISTED-IMPORT-UNUSED '{n}' ({sorted(tids)}) "
                    f"{why} -> un-normalized alias or wrong rename target?",
                )
            )

    # 3. TARGET-CHANGED (same scope+name resolves to a different import target)
    #    Only a *swap* is dangerous: a BEFORE target that is no longer reachable in
    #    AFTER means a reference was silently re-pointed. A pure superset growth
    #    (tbefore <= tafter) is the benign `import pkg.subA` + `import pkg.subB`
    #    case: both statements bind the same top-level name `pkg` to the same
    #    package object and only *add* submodule attributes (e.g. adding
    #    `import urllib.error` next to `import urllib.request`). Nothing the name
    #    resolved to before is lost, so no reference is re-pointed -- skip it.
    #
    #    A deliberate *relocation* is also benign and must not block: when a name
    #    keeps its spelling but its import source is moved A -> B in THIS diff (the
    #    old `from A import x` is removed at module level and a new `from B import x`
    #    is added), the swap is intentional, not a silent re-point to a pre-existing
    #    different object. This mirrors the relocation tolerance already applied to
    #    TARGET-MISSING. The dangerous case -- the name now resolving to a target
    #    that already existed before (shadow/clash) -- is NOT exempted.
    removed_module_targets = before_module_targets - after_module_targets
    for key, tafter in b["target_by_use"].items():
        tbefore = a["target_by_use"].get(key)
        if tbefore and tbefore != tafter and (tbefore - tafter):
            lost = tbefore - tafter
            gained = tafter - tbefore
            relocated = lost <= removed_module_targets and gained <= added_module_targets
            if relocated:
                continue
            findings.append(
                (
                    "BLOCKER",
                    f"{path}: TARGET-CHANGED name '{key[1]}' in {key[0]} "
                    f"{sorted(tbefore)} -> {sorted(tafter)} (rename re-points module)",
                )
            )

    # 4. MODULE-DUP-IMPORT introduced
    for n in sorted(b["module_dup"] - a["module_dup"]):
        findings.append(
            (
                "WARN",
                f"{path}: MODULE-DUP-IMPORT '{n}' bound by import AND non-import "
                f"at module level (possible clash)",
            )
        )

    # 5. AMBIGUOUS-BIND introduced (module scope)
    for scope, names in b["ambiguous"].items():
        new = names - a["ambiguous"].get(scope, set())
        for n in sorted(new):
            findings.append(("WARN", f"{path}: AMBIGUOUS-BIND '{n}' import+non-import in {scope}"))

    # 6. TARGET-MISSING (informational): a scope stopped resolving to an import
    #    target. Real bugs are covered above; remaining cases are relocated code.
    for scope, tbefore in a["targets_by_scope"].items():
        tafter = b["targets_by_scope"].get(scope, set())
        for t in sorted(tbefore - tafter):
            relocated = (
                ""
                if t in added_module_targets
                else "  [target not re-added here -> likely relocated/deleted]"
            )
            findings.append(("INFO", f"{path}: TARGET-MISSING {t} in scope {scope}{relocated}"))
    return findings


# ---------------------------------------------------------------- self-test

_SELF_TESTS = {
    "dangling_alias": (
        # before: inline aliased import, used as _b
        "import os\ndef f():\n    import glob as _b\n    return _b.glob('*')\n",
        # after: hoisted to canonical, but reference NOT normalized -> _b dangles
        "import os\nimport glob\ndef f():\n    return _b.glob('*')\n",
        "BLOCKER",
    ),
    "rename_clash": (
        # before: _b is a deliberate alias; `b` already means something else
        "import re as _b\nb = 123\ndef f():\n    return _b.compile('x'), b\n",
        # after: someone normalized _b -> b ; now f().b is the int, re is lost
        "import re\nb = 123\ndef f():\n    return b.compile('x'), b\n",
        "BLOCKER",  # TARGET-MISSING from:.. or import:re in f
    ),
    "clean_rename": (
        "def f():\n    import glob as _g\n    return _g.glob('*')\n",
        "import glob\ndef f():\n    return glob.glob('*')\n",
        None,  # expect NO blocker
    ),
    "clean_dedup_redundant": (
        "import sys\ndef f():\n    import sys\n    return sys.argv\n",
        "import sys\ndef f():\n    return sys.argv\n",
        None,
    ),
    "from_import_dangling": (
        # from-import alias left un-normalized
        "def f():\n    from importlib.metadata import version as _v\n    return _v('x')\n",
        "from importlib.metadata import version\ndef f():\n    return _v('x')\n",
        "BLOCKER",
    ),
    "local_var_clash": (
        # _b renamed to b, but b is a LOCAL var in f -> import silently unused
        "def f(b):\n    import re as _b\n    return _b.compile(b)\n",
        "import re\ndef f(b):\n    return b.compile(b)\n",  # 'b' is the param, not the module
        "BLOCKER",
    ),
    "substring_safe": (
        # correct _copy->copy rename while config_copy var exists: NO false positive
        "def f(config):\n"
        "    import copy as _copy\n"
        "    config_copy = _copy.deepcopy(config)\n"
        "    return config_copy\n",
        "import copy\n"
        "def f(config):\n"
        "    config_copy = copy.deepcopy(config)\n"
        "    return config_copy\n",
        None,
    ),
    "attr_access_not_a_use": (
        # x._b is attribute access, not a use of name _b; removing import _b is fine
        "import os\ndef f(x):\n    import sys as _b\n    return x._b + _b.argv[0]\n",
        "import os\nimport sys\ndef f(x):\n    return x._b + sys.argv[0]\n",
        None,
    ),
    "reexport_in_all_is_used": (
        # a new re-export added to a package __init__ (name in __all__, no load) is
        # a deliberate export, NOT a botched hoist -> must not block
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a", "b"]\n',
        None,
    ),
    "unused_import_not_in_all_still_blocks": (
        # the fix is precise: a newly-added module import that is neither loaded nor
        # listed in __all__ is still a dangling/unused hoist
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a"]\n',
        "BLOCKER",
    ),
    "reassigned_all_drops_the_reexport": (
        # only the FINAL __all__ exports anything: a later plain "=" REPLACES the list,
        # so b is not re-exported and its import is a genuine unused hoist. Unioning
        # every assignment would have marked it used and let this through.
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a", "b"]\n__all__ = ["a"]\n',
        "BLOCKER",
    ),
    "augmented_all_extends_the_reexport": (
        # "+=" extends rather than replaces, so b IS re-exported -> must not block
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a"]\n__all__ += ["b"]\n',
        None,
    ),
    "unreadable_all_keeps_earlier_reexports": (
        # rebound to something not statically readable: contents unknown, so keep what
        # we had rather than flag a real re-export as unused
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a", "b"]\n__all__ = sorted(__all__)\n',
        None,
    ),
    "unreadable_augmented_all_keeps_the_reexport": (
        # "+=" extended by a value we cannot read statically: the augmented contents are
        # unknown, so a dynamically-supplied re-export must not be flagged as unused
        # (mirrors the unreadable "=" case, which also preserves uncertainty)
        'from pkg import a\n__all__ = ["a"]\n',
        'from pkg import a\nfrom pkg import b\n__all__ = ["a"]\n__all__ += sorted(["b"])\n',
        None,
    ),
}


def _self_test() -> int:
    ok = True
    for name, (before, after, expect) in _SELF_TESTS.items():
        findings = compare(before, after, f"<{name}>")
        blockers = [m for sev, m in findings if sev == "BLOCKER"]
        got = "BLOCKER" if blockers else None
        passed = got == expect
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: expect={expect} got={got}")
        for sev, m in findings:
            print(f"        ({sev}) {m}")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


def _pyflakes_undefined(path: str) -> set[str] | None:
    """Return the set of names pyflakes reports as 'undefined name' for `path`,
    or None if pyflakes failed to run/parse the file."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pyflakes", path], capture_output = True, text = True
        )
    except Exception:
        return None
    if "syntax error" in (proc.stdout + proc.stderr).lower():
        return None
    names = set()
    for line in proc.stdout.splitlines():
        m = _re_mod.search(r"undefined name '([^']+)'", line)
        if m:
            names.add(m.group(1))
    return names


def audit_files(paths: list[str]) -> int:
    """Single-version robustness audit: confirm the analyzer doesn't crash, then
    cross-check its 'unresolved' names against pyflakes. A name the resolver flags
    that pyflakes accepts is a tool false positive."""
    n_files = n_err = n_fp = n_syntax = 0
    fp_detail: dict[str, set[str]] = {}
    err_detail: dict[str, str] = {}
    for path in paths:
        n_files += 1
        try:
            src = open(path, encoding = "utf-8").read()
        except Exception as e:  # unreadable
            n_err += 1
            err_detail[path] = f"read: {e}"
            continue
        try:
            res = _analyze(src)
        except SyntaxError:
            n_syntax += 1
            continue
        except Exception as e:  # analyzer crash -> robustness bug
            n_err += 1
            err_detail[path] = f"{type(e).__name__}: {e}"
            continue
        tool_unresolved = set()
        for names in res["unresolved"].values():
            tool_unresolved |= names
        if not tool_unresolved:
            continue
        pf = _pyflakes_undefined(path)
        if pf is None:
            continue  # pyflakes couldn't adjudicate; skip cross-check
        false_pos = tool_unresolved - pf
        if false_pos:
            n_fp += 1
            fp_detail[path] = false_pos
    print(f"audited files      : {n_files}")
    print(f"syntax-skipped     : {n_syntax}")
    print(f"analyzer errors    : {n_err}")
    for p, e in sorted(err_detail.items()):
        print(f"    ERROR {p}: {e}")
    print(f"false-positive files: {n_fp}  (resolver flagged a name pyflakes accepts)")
    for p, names in sorted(fp_detail.items()):
        print(f"    FP {p}: {sorted(names)}")
    ok = n_err == 0 and n_fp == 0
    print(
        "\nAUDIT:",
        "ROBUST (no crashes, no false positives vs pyflakes)" if ok else "NEEDS WORK (see above)",
    )
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default = "origin/main")
    ap.add_argument("--after", default = "HEAD")
    ap.add_argument("--self-test", action = "store_true")
    ap.add_argument(
        "--audit",
        action = "store_true",
        help = "single-version robustness audit on filesystem paths",
    )
    ap.add_argument("files", nargs = "*")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.audit:
        return audit_files(args.files)

    any_blocker = False
    for path in args.files:
        before = _git_show(args.before, path)
        after = _git_show(args.after, path)
        if after is None:
            print(f"SKIP {path}: not found at {args.after}")
            continue
        if before is None:
            before = ""  # new file
        findings = compare(before, after, path)
        blockers = [f for f in findings if f[0] == "BLOCKER"]
        warns = [f for f in findings if f[0] == "WARN"]
        infos = [f for f in findings if f[0] == "INFO"]
        status = "CLEAN" if not blockers and not warns else ("BLOCKERS" if blockers else "WARNINGS")
        print(f"\n=== {path}: {status} ===")
        for sev, m in blockers + warns + infos:
            print(f"  [{sev}] {m}")
        any_blocker = any_blocker or bool(blockers)
    print("\nOVERALL:", "FAIL (blockers found)" if any_blocker else "PASS (no blockers)")
    return 1 if any_blocker else 0


if __name__ == "__main__":
    sys.exit(main())
