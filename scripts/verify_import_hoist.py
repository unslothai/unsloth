#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Scope-aware check for dangling aliases and silent rename clashes in import hoists (_b -> b)."""

from __future__ import annotations

import argparse
import ast
from pathlib import PurePosixPath
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


@dataclass
class Binding:
    kind: str  # 'import' | 'importfrom' | 'def' | 'class' | 'other'
    target: str | None = None


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
    bound = alias.asname or alias.name
    mod = ("." * (node.level or 0)) + (node.module or "")
    return bound, f"from:{mod}:{alias.name}"


def _is_literal_ref(node: ast.AST) -> bool:
    """Whether `node` names `Literal`, however it was imported (`Literal`, `t.Literal`)."""
    if isinstance(node, ast.Name):
        return node.id == "Literal"
    return isinstance(node, ast.Attribute) and node.attr == "Literal"


class _Builder(ast.NodeVisitor):
    """Builds the scope tree + bindings, and records every (scope, Name-load)."""

    def __init__(self):
        self.module = Scope("module", "<module>", None)
        self.uses: list[tuple[Scope, str, int]] = []
        # Annotations count as used but never unresolved (forward refs).
        self.soft_uses: list[tuple[Scope, str, int]] = []

    # `Optional["Dict[str, 'T']"]` is two deep; nothing real goes further.
    _FORWARD_REF_DEPTH = 3

    def _visit_annotation(
        self,
        node,
        scope: Scope,
        _depth: int = 0,
        _lineno: int = 0,
    ) -> None:
        """Record annotation names as SOFT uses: an import used only in an annotation
        counts as used, but a forward-ref name is never 'unresolved'.

        A QUOTED annotation is one too: `Optional["T"]` keeps the name in an ast.Constant,
        invisible to a Name walk, so a TYPE_CHECKING import reached only that way read as
        unused and blocked correct code. Parse the string and walk what it denotes.
        """
        if node is None:
            return
        # Literal[...] holds values, not type names; crediting them would hide a real finding.
        if isinstance(node, ast.Subscript) and _is_literal_ref(node.value):
            self._visit_annotation(node.value, scope, _depth, _lineno)
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, str) or _depth >= self._FORWARD_REF_DEPTH:
                return
            try:
                inner = ast.parse(node.value.strip(), mode = "eval").body
            except (SyntaxError, ValueError):
                return
            self._visit_annotation(inner, scope, _depth + 1, _lineno or node.lineno)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.soft_uses.append((scope, node.id, _lineno or node.lineno))
        for child in ast.iter_child_nodes(node):
            self._visit_annotation(child, scope, _depth, _lineno)

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
            # Decorators and defaults evaluate in the enclosing scope.
            for d in node.decorator_list:
                self._visit_expr(d, scope)
            self._visit_arg_defaults(node.args, scope)
            child = Scope("function", f"{scope.qualname}.{node.name}", scope)
            self._bind_type_params(node, child)
            self._bind_args(node.args, child)
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
        if isinstance(node, getattr(ast, "TryStar", ())):
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
        if isinstance(node, getattr(ast, "TypeAlias", ())):
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
                if isinstance(node, ast.AugAssign):
                    self._record_loads(t, scope)
                else:
                    # `overrides[dependency] = fn` loads both names.
                    if not isinstance(t, ast.Name):
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
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._visit_stmt(child, scope)
            else:
                self._visit_expr(child, scope)

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
                # First iterable evaluates in the enclosing scope.
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
        if p.kind != "class" or p.parent is None:
            if p.kind != "class":
                chain.append(p)
        p = p.parent
    return chain


# ---------------------------------------------------------------- analysis


def _analyze(src: str):
    tree = ast.parse(src)
    b = _Builder()
    b.run(tree)
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
    for scope, name, _ln in b.soft_uses:
        status, binds = _resolve(scope, name)
        if status == "import":
            tids = {bd.target for bd in binds if bd.target}
            targets_by_scope.setdefault(scope.qualname, set()).update(tids)
    module = b.module
    module_imports = {
        n: bs
        for n, bs in module.bindings.items()
        if any(x.kind in ("import", "importfrom") for x in bs)
    }
    module_dup = {
        n
        for n, bs in module.bindings.items()
        if any(x.kind in ("import", "importfrom") for x in bs)
        and any(x.kind not in ("import", "importfrom") for x in bs)
    }
    ambiguous: dict[str, set[str]] = {}

    def walk_scopes(scope: Scope):
        for n, bs in scope.bindings.items():
            if any(x.kind in ("import", "importfrom") for x in bs) and any(
                x.kind not in ("import", "importfrom") for x in bs
            ):
                ambiguous.setdefault(scope.qualname, set()).add(n)

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


def _is_package_init(path: str) -> bool:
    """True for a package __init__.py, where __all__ means "public re-export"."""
    return PurePosixPath(str(path).replace("\\", "/")).name == "__init__.py"


def _from_parts(target: str) -> tuple[str, str] | None:
    """("module", "name") for a `from:module:name` target id, else None."""
    if not target.startswith("from:"):
        return None
    mod, _, name = target[len("from:") :].rpartition(":")
    return (mod, name) if mod and name else None


def _is_definition_move(lost: set[str], gained: set[str], moved) -> bool:
    """Every lost `from:old:N` pairs with one gained `from:new:N` that `moved` confirms.

    A function-local `from old import N` re-pointed to `from new import N` after N itself moved to
    `new` binds the same object; the module-level relocation check above cannot see that because
    neither import is at module level.
    """
    if moved is None or not lost or len(lost) != len(gained):
        return False
    gained_by_name: dict[str, list[str]] = {}
    for target in gained:
        parts = _from_parts(target)
        if parts is None:
            return False
        gained_by_name.setdefault(parts[1], []).append(target)
    used: set[str] = set()
    for target in lost:
        parts = _from_parts(target)
        candidates = gained_by_name.get(parts[1], []) if parts else []
        if len(candidates) != 1 or candidates[0] in used or not moved(target, candidates[0]):
            return False
        used.add(candidates[0])
    # One-to-one: a gained target no lost one paired with is an unexplained re-point.
    return used == set(gained)


def compare(
    before_src: str,
    after_src: str,
    path: str,
    moved = None,
) -> list[tuple[str, str]]:
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
    after_exported = _dunder_all_names(after_src)

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

    # Module-level import no load resolves to, that is new or was used before.
    for n, tids in b["module_import_targets"].items():
        if tids & after_used:
            continue
        # __future__ imports are compiler directives and never loaded.
        if all(t.startswith("from:__future__:") for t in tids):
            continue
        # __all__ re-exports are exempt only in package __init__.py; broader would hide rename clashes.
        if n in after_exported and _is_package_init(path):
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

    # Only a swap is dangerous; superset growth and an in-diff relocation are benign.
    removed_module_targets = before_module_targets - after_module_targets
    for key, tafter in b["target_by_use"].items():
        tbefore = a["target_by_use"].get(key)
        if tbefore and tbefore != tafter and (tbefore - tafter):
            lost = tbefore - tafter
            gained = tafter - tbefore
            relocated = lost <= removed_module_targets and gained <= added_module_targets
            if relocated:
                continue
            if _is_definition_move(lost, gained, moved):
                findings.append(
                    (
                        "INFO",
                        f"{path}: MOVED name '{key[1]}' in {key[0]} "
                        f"{sorted(tbefore)} -> {sorted(tafter)} (definition moved to the new module)",
                    )
                )
                continue
            findings.append(
                (
                    "BLOCKER",
                    f"{path}: TARGET-CHANGED name '{key[1]}' in {key[0]} "
                    f"{sorted(tbefore)} -> {sorted(tafter)} (rename re-points module)",
                )
            )

    for n in sorted(b["module_dup"] - a["module_dup"]):
        findings.append(
            (
                "WARN",
                f"{path}: MODULE-DUP-IMPORT '{n}' bound by import AND non-import "
                f"at module level (possible clash)",
            )
        )

    for scope, names in b["ambiguous"].items():
        new = names - a["ambiguous"].get(scope, set())
        for n in sorted(new):
            findings.append(("WARN", f"{path}: AMBIGUOUS-BIND '{n}' import+non-import in {scope}"))

    # Informational: remaining cases are relocated code.
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
        "import os\ndef f():\n    import glob as _b\n    return _b.glob('*')\n",
        "import os\nimport glob\ndef f():\n    return _b.glob('*')\n",
        "BLOCKER",
    ),
    "rename_clash": (
        "import re as _b\nb = 123\ndef f():\n    return _b.compile('x'), b\n",
        "import re\nb = 123\ndef f():\n    return b.compile('x'), b\n",
        "BLOCKER",
    ),
    "clean_rename": (
        "def f():\n    import glob as _g\n    return _g.glob('*')\n",
        "import glob\ndef f():\n    return glob.glob('*')\n",
        None,
    ),
    "clean_dedup_redundant": (
        "import sys\ndef f():\n    import sys\n    return sys.argv\n",
        "import sys\ndef f():\n    return sys.argv\n",
        None,
    ),
    "from_import_dangling": (
        "def f():\n    from importlib.metadata import version as _v\n    return _v('x')\n",
        "from importlib.metadata import version\ndef f():\n    return _v('x')\n",
        "BLOCKER",
    ),
    "local_var_clash": (
        "def f(b):\n    import re as _b\n    return _b.compile(b)\n",
        "import re\ndef f(b):\n    return b.compile(b)\n",
        "BLOCKER",
    ),
    "substring_safe": (
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
        "import os\ndef f(x):\n    import sys as _b\n    return x._b + _b.argv[0]\n",
        "import os\nimport sys\ndef f(x):\n    return x._b + sys.argv[0]\n",
        None,
    ),
    "reexport_in_package_init_is_allowed": (
        'from .a import A\n__all__ = ["A"]\n',
        'from .a import A\nfrom .b import B\n__all__ = ["A", "B"]\n',
        None,
        "pkg/__init__.py",
    ),
    "reexport_in_ordinary_module_is_still_blocked": (
        'from .a import A\n__all__ = ["A"]\n',
        'from .a import A\nfrom .b import B\n__all__ = ["A", "B"]\n',
        "BLOCKER",
        "pkg/helpers.py",
    ),
    "unexported_new_import_in_init_is_still_blocked": (
        'from .a import A\n__all__ = ["A"]\n',
        'from .a import A\nfrom .b import B\n__all__ = ["A"]\n',
        "BLOCKER",
        "pkg/__init__.py",
    ),
    "forward_ref_string_annotation_counts_as_a_use": (
        "from typing import TYPE_CHECKING, Optional\ndef f(x) -> Optional[int]:\n    return x\n",
        "from typing import TYPE_CHECKING, Optional\n"
        "if TYPE_CHECKING:\n"
        "    from .m import T\n"
        'def f(x) -> Optional["T"]:\n'
        "    return x\n",
        None,
    ),
    "nested_forward_ref_counts_as_a_use": (
        "from typing import TYPE_CHECKING, Optional\ndef f(x) -> Optional[int]:\n    return x\n",
        "from typing import TYPE_CHECKING, Optional\n"
        "if TYPE_CHECKING:\n"
        "    from .m import T\n"
        "def f(x) -> Optional[\"Optional['T']\"]:\n"
        "    return x\n",
        None,
    ),
    "a_subscript_key_on_the_left_hand_side_is_a_use": (
        "app = {}\n",
        "from .deps import dependency\napp = {}\napp[dependency] = 1\n",
        None,
    ),
    "a_subscripted_object_on_the_left_hand_side_is_a_use": (
        "def f(k, v):\n    return k, v\n",
        "from .deps import registry\ndef f(k, v):\n    registry[k] = v\n",
        None,
    ),
    "an_attribute_target_loads_the_object": (
        "def f(v):\n    return v\n",
        "from .deps import settings\ndef f(v):\n    settings.value = v\n",
        None,
    ),
    "a_literal_value_is_not_a_use_of_that_name": (
        "from typing import TYPE_CHECKING, Literal\ndef f(x) -> Literal['a']:\n    return x\n",
        "from typing import TYPE_CHECKING, Literal\n"
        "if TYPE_CHECKING:\n"
        "    from .m import T\n"
        "def f(x) -> Literal['T']:\n"
        "    return x\n",
        "BLOCKER",
    ),
    "unparseable_annotation_string_is_ignored": (
        "from typing import Annotated\ndef f(x: Annotated[int, 'ok']) -> int:\n    return x\n",
        "from typing import Annotated\n"
        "def f(x: Annotated[int, 'not a type at all']) -> int:\n"
        "    return x\n",
        None,
    ),
    # A function-local import re-pointed after the definition itself moved binds the same object.
    "local_import_follows_a_moved_definition": (
        "def f():\n    from pkg.old import helper\n    return helper()\n",
        "def f():\n    from pkg.new import helper\n    return helper()\n",
        None,
        "<local_import_follows_a_moved_definition>",
        lambda old, new: (old, new) == ("from:pkg.old:helper", "from:pkg.new:helper"),
    ),
    # The same re-point when the definition did not move is still a clash.
    "local_import_re_pointed_without_a_move_is_blocked": (
        "def f():\n    from pkg.old import helper\n    return helper()\n",
        "def f():\n    from pkg.new import helper\n    return helper()\n",
        "BLOCKER",
        "<local_import_re_pointed_without_a_move_is_blocked>",
        lambda old, new: False,
    ),
    # A rename (helper -> other_helper's module binding under a new name) is never a move.
    "local_import_renamed_across_modules_is_blocked": (
        "def f():\n    from pkg.old import helper\n    return helper()\n",
        "def f():\n    from pkg.new import other as helper\n    return helper()\n",
        "BLOCKER",
        "<local_import_renamed_across_modules_is_blocked>",
        lambda old, new: True,
    ),
    "two_lost_imports_cannot_share_one_gained_target": (
        "def f(x):\n    if x:\n        from pkg.old1 import helper\n    else:\n        from pkg.old2 import helper\n    return helper()\n",
        "def f(x):\n    if x:\n        from pkg.shared import helper\n    else:\n        from pkg.other import other as helper\n    return helper()\n",
        "BLOCKER",
        "<two_lost_imports_cannot_share_one_gained_target>",
        lambda old, new: new == "from:pkg.shared:helper",
    ),
}


def _self_test() -> int:
    ok = True
    for name, case in _SELF_TESTS.items():
        before, after, expect = case[0], case[1], case[2]
        path = case[3] if len(case) > 3 else f"<{name}>"
        moved = case[4] if len(case) > 4 else None
        findings = compare(before, after, path, moved = moved)
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
        except Exception as e:
            n_err += 1
            err_detail[path] = f"read: {e}"
            continue
        try:
            res = _analyze(src)
        except SyntaxError:
            n_syntax += 1
            continue
        except Exception as e:
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
            continue
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


def _dunder_all_names(src: str) -> set[str]:
    """Names a module publishes via __all__, i.e. deliberate re-exports."""
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = getattr(node, "value", None)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _top_level_binding(src: str | None, name: str):
    """How a module binds `name` at top level: "def", ("import", module, original_name), None, or "ambiguous"
    when it is bound more than once (a later binding can shadow the definition) or by a plain `import`."""
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    found = []

    def targets(node) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [n for elt in node.elts for n in targets(elt)]
        if isinstance(node, ast.Starred):
            return targets(node.value)
        return []

    def visit(body) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    found.append("def")
                continue
            if isinstance(node, ast.Assign):
                found.extend("def" for t in node.targets if name in targets(t))
            elif isinstance(node, ast.AnnAssign):
                # A bare annotation (`helper: int`) declares without binding.
                if node.value is not None and name in targets(node.target):
                    found.append("def")
            elif isinstance(node, ast.AugAssign):
                if name in targets(node.target):
                    found.append("ambiguous")
            elif isinstance(node, ast.Delete):
                if any(name in targets(t) for t in node.targets):
                    found.append("ambiguous")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    # `import *` may rebind any name; its exports are not resolved here.
                    if alias.name == "*":
                        found.append("ambiguous")
                    elif (alias.asname or alias.name) == name:
                        found.append(
                            ("import", "." * (node.level or 0) + (node.module or ""), alias.name)
                        )
            elif isinstance(node, ast.Import):
                found.extend(
                    "ambiguous" for a in node.names if (a.asname or a.name.split(".")[0]) == name
                )
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                for item in getattr(node, "items", []):
                    if item.optional_vars is not None and name in targets(item.optional_vars):
                        found.append("ambiguous")
                if isinstance(node, (ast.For, ast.AsyncFor)) and name in targets(node.target):
                    found.append("ambiguous")
            # Conditional and guarded blocks (`if TYPE_CHECKING:`, `try:`) may not bind at runtime.
            try_like = isinstance(node, ast.Try) or isinstance(node, getattr(ast, "TryStar", ()))
            if try_like or isinstance(
                node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
            ):
                for field in ("body", "orelse", "finalbody"):
                    guarded(getattr(node, field, []) or [])
            if try_like:
                for handler in node.handlers:
                    guarded(handler.body)
            if isinstance(node, getattr(ast, "Match", ())):
                for case in node.cases:
                    guarded(case.body)

    def guarded(body) -> None:
        start = len(found)
        visit(body)
        found[start:] = ["ambiguous"] * (len(found) - start)

    visit(tree.body)
    if not found:
        return None
    # The scan above classifies the common shapes; any second binding site, in whatever form, makes it ambiguous.
    if len(found) > 1 or _module_binding_sites(tree, name) > 1:
        return "ambiguous"
    return found[0]


def _module_binding_sites(tree: ast.Module, name: str) -> int:
    """Every place that binds or unbinds `name` in module scope: assignment / del / for / with / walrus targets,
    `except ... as`, match captures, imports (and `import *`), def / class, plus `global name` anywhere."""
    count = 0

    def visit(node) -> None:
        nonlocal count
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                count += child.name == name
                # Decorators, defaults, bases and annotations run in module scope; the body does not.
                for sub in ast.iter_child_nodes(child):
                    if sub not in child.body:
                        visit_one(sub)
                count += sum(isinstance(n, ast.Global) and name in n.names for n in ast.walk(child))
                continue
            if isinstance(child, ast.Lambda):
                # Defaults are evaluated where the lambda is; only its body is a new scope.
                visit_one(child.args)
                continue
            visit_one(child)

    def visit_one(node) -> None:
        nonlocal count
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            count += 1
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            count += 1
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                bound = alias.asname or (
                    alias.name if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
                )
                count += alias.name == "*" or bound == name
        elif (
            type(node).__name__ in ("MatchAs", "MatchStar") and getattr(node, "name", None) == name
        ):
            count += 1
        elif type(node).__name__ == "MatchMapping" and getattr(node, "rest", None) == name:
            count += 1
        visit(node)

    visit(tree)
    return count


def _module_paths(module: str, importer: str) -> list[str]:
    """Candidate repo paths for `module` as imported from the file `importer`."""
    importer_dir = PurePosixPath(str(importer).replace("\\", "/")).parent
    if module.startswith("."):
        level = len(module) - len(module.lstrip("."))
        base = importer_dir
        for _ in range(level - 1):
            base = base.parent
        # The repo root is not a package: a relative import that reaches it (or climbs past it) cannot resolve.
        if str(base) in ("", "."):
            return []
        rest = module.lstrip(".")
        stem = base / rest.replace(".", "/") if rest else base
        return [f"{stem}.py", f"{stem}/__init__.py"]
    rel = module.replace(".", "/")
    roots = [importer_dir, *importer_dir.parents]
    out = []
    for root in roots:
        prefix = "" if str(root) in ("", ".") else f"{root}/"
        out += [f"{prefix}{rel}.py", f"{prefix}{rel}/__init__.py"]
    return out


def _git_moved(before_ref: str, after_ref: str, importer: str):
    """moved(old_target, new_target): the name's definition left `old` for `new` in this diff."""
    cache: dict[tuple[str, str], str | None] = {}

    def show(ref: str, path: str) -> str | None:
        if (ref, path) not in cache:
            cache[(ref, path)] = _git_show(ref, path)
        return cache[(ref, path)]

    def locate(module: str, ref: str, near: str) -> str | None:
        # Only a unique match: which root an absolute import reaches depends on sys.path, not on nearness.
        found = [p for p in _module_paths(module, near) if show(ref, p) is not None]
        return found[0] if len(found) == 1 else None

    def origin(
        ref: str,
        module: str,
        name: str,
        near: str,
        hops: int = 0,
    ):
        """(path, name) of the definition `from module import name` reaches at `ref`, following re-exports."""
        path = locate(module, ref, near)
        binding = _top_level_binding(show(ref, path), name) if path else None
        if binding == "def":
            return (path, name)
        if isinstance(binding, tuple) and hops < 8:
            return origin(ref, binding[1], binding[2], path, hops + 1)
        return None

    def moved(old_target: str, new_target: str) -> bool:
        old, new = _from_parts(old_target), _from_parts(new_target)
        if old is None or new is None or old[1] != new[1]:
            return False
        name = old[1]
        # Both imports reach one definition (the old module and the new one re-export it).
        before_origin = origin(before_ref, old[0], name, importer)
        if before_origin is not None and before_origin == origin(after_ref, new[0], name, importer):
            return True
        old_path = locate(old[0], before_ref, importer)
        new_path = locate(new[0], after_ref, importer)
        if not old_path or not new_path or old_path == new_path:
            return False
        # Defined in the new module now, and was defined in the old one before.
        if _top_level_binding(show(after_ref, new_path), name) != "def":
            return False
        if _top_level_binding(show(before_ref, old_path), name) != "def":
            return False
        # Arrived with this change: a destination that already had its own `name` is a different object.
        if _top_level_binding(show(before_ref, new_path), name) is not None:
            return False
        # The old module no longer defines it; at most it re-imports it from the new module.
        left = _top_level_binding(show(after_ref, old_path), name)
        if left is None:
            return True
        if not isinstance(left, tuple):
            return False
        _, module, original = left
        return original == name and locate(module, after_ref, old_path) == new_path

    return moved


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
            before = ""
        findings = compare(before, after, path, moved = _git_moved(args.before, args.after, path))
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
