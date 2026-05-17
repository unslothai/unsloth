"""Static-analysis regression test: callback signature drift.

Catches the class of bug where a producer (e.g. unsloth_zoo's MLXTrainer)
changes the number of args it passes to a registered callback but consumers
(unsloth tests / source) still declare the old arity. The producer's
``try / except Exception`` typically swallows the resulting TypeError, so
the callback silently never fires and the failure surfaces several seconds
later as a confusing downstream assertion.

The check is pure AST (no imports of MLX modules etc), so it runs on every
OS / Python version that ships in CI.

Pattern detected:
  * Producer side: a class with ``self._<name>_callbacks`` list, populated
    via ``self._<name>_callbacks.append(...)`` from an ``add_<name>_callback``
    method, and invoked via ``for cb in self._<name>_callbacks: cb(arg1, ...)``.
    The arity at the call site is the canonical expected arity.
  * Consumer side: any ``<obj>.add_<name>_callback(fn)`` call where ``fn``
    resolves to a ``def`` or ``async def`` in the same file. Consumer arity
    must equal canonical arity (or be variadic).

Consumers handled tolerantly:
  * ``*args`` / ``**kwargs``: accept any canonical arity.
  * Methods (``self.fn``) and unresolved Name targets (imported from another
    file): skipped with a note in the failure message rather than asserted.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Skip noisy paths during file discovery.
SKIP_PARTS = {
    ".git", ".out", "temp", "node_modules", "build", "dist",
    ".venv", "venv", ".pytest_cache", "__pycache__",
    # Frontend tree under studio is JS/TS plus a few stub .py files; not worth walking.
    "frontend",
}


def _iter_py(root: pathlib.Path):
    root = pathlib.Path(root).resolve()
    for p in root.rglob("*.py"):
        try:
            rel_parts = p.resolve().relative_to(root).parts
        except ValueError:
            rel_parts = p.parts
        if any(part.startswith(".") and part not in (".", "..") for part in rel_parts):
            continue
        if any(part in SKIP_PARTS for part in rel_parts):
            continue
        yield p


# Module-level parse cache so discover_producers + check_registrations only
# pay the parse cost once per file across the whole test run.
_PARSE_CACHE: dict[pathlib.Path, ast.AST | None] = {}


def _safe_parse(path: pathlib.Path):
    key = path.resolve()
    if key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    try:
        import warnings as _w
        with _w.catch_warnings():
            # Suppress SyntaxWarning emitted while parsing third-party files
            # that contain invalid escape sequences in regex / docstrings.
            _w.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        tree = None
    _PARSE_CACHE[key] = tree
    return tree


def _callback_list_attrs_in_class(cls: ast.ClassDef) -> set[str]:
    """Find self._<name>_callbacks attributes assigned or appended-to inside cls."""
    found = set()
    for node in ast.walk(cls):
        # self._x_callbacks = [...]
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr.startswith("_")
                    and t.attr.endswith("_callbacks")
                ):
                    found.add(t.attr)
        # self._x_callbacks.append(fn)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            and node.func.value.attr.startswith("_")
            and node.func.value.attr.endswith("_callbacks")
        ):
            found.add(node.func.value.attr)
    return found


def _producer_arities(tree: ast.AST) -> dict[str, int]:
    """For each ``for cb in self._x_callbacks: cb(...)`` in the AST, return
    {cb_list_attr: max_arity}. Multiple sites take the max so that variadic
    branches do not lower the contract.
    """
    out: dict[str, int] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        cb_lists = _callback_list_attrs_in_class(cls)
        for cb_list in cb_lists:
            for node in ast.walk(cls):
                if not isinstance(node, ast.For):
                    continue
                if not (
                    isinstance(node.iter, ast.Attribute)
                    and isinstance(node.iter.value, ast.Name)
                    and node.iter.value.id == "self"
                    and node.iter.attr == cb_list
                ):
                    continue
                if not isinstance(node.target, ast.Name):
                    continue
                cb_name = node.target.id
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == cb_name
                    ):
                        arity = len(inner.args)
                        out[cb_list] = max(out.get(cb_list, 0), arity)
    return out


def _registration_attr_to_list(attr: str) -> str | None:
    """add_step_callback -> _step_callbacks. Returns None if pattern doesn't match."""
    if attr.startswith("add_") and attr.endswith("_callback"):
        middle = attr[len("add_") : -len("_callback")]
        if middle:
            return f"_{middle}_callbacks"
    if attr.startswith("register_") and attr.endswith("_callback"):
        middle = attr[len("register_") : -len("_callback")]
        if middle:
            return f"_{middle}_callbacks"
    return None


def _func_arity(node: ast.AST) -> tuple[int, bool] | None:
    """Return (positional_arity, accepts_var_positional). None if not a function def."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return None
    args = node.args
    arity = len(args.posonlyargs) + len(args.args)
    accepts_var = args.vararg is not None
    # Bound methods: drop the implicit self if this is a method-style def.
    # We can't tell statically whether the def is a method without class
    # context, so we conservatively do not subtract self here. The consumer
    # check skips bare-Name registrations whose target is a `self.fn` attr
    # anyway.
    return arity, accepts_var


def discover_producers(roots: list[pathlib.Path]) -> dict[str, list[tuple[pathlib.Path, int]]]:
    """Walk every .py under each root and return {cb_list_attr: [(file, arity), ...]}."""
    producers: dict[str, list[tuple[pathlib.Path, int]]] = {}
    for root in roots:
        if not root or not root.exists():
            continue
        for src in _iter_py(root):
            tree = _safe_parse(src)
            if tree is None:
                continue
            for cb_list, arity in _producer_arities(tree).items():
                producers.setdefault(cb_list, []).append((src, arity))
    return producers


def check_registrations(roots: list[pathlib.Path], producers: dict[str, list[tuple[pathlib.Path, int]]]):
    """Walk every .py under each root, find <x>.add_*_callback(fn) where fn is a
    bare Name resolvable to a def in the same file, and assert its arity
    matches the producer's canonical arity. Returns (issues, skipped, ok_count).
    """
    issues: list[str] = []
    skipped: list[str] = []
    ok_count = 0
    for root in roots:
        if not root or not root.exists():
            continue
        for src in _iter_py(root):
            tree = _safe_parse(src)
            if tree is None:
                continue
            # All function/lambda defs in this file by name (and by id for lambdas via assignment).
            defs_by_name: dict[str, ast.AST] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs_by_name[node.name] = node
                if isinstance(node, ast.Assign):
                    if isinstance(node.value, ast.Lambda) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        defs_by_name[node.targets[0].id] = node.value
            # Find <x>.add_*_callback(fn) sites
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                if not isinstance(call.func, ast.Attribute):
                    continue
                cb_list = _registration_attr_to_list(call.func.attr)
                if cb_list is None:
                    continue
                if cb_list not in producers:
                    skipped.append(
                        f"{src}:{call.lineno}: {call.func.attr}(...) but no producer "
                        f"defines {cb_list} (third-party API?)"
                    )
                    continue
                # Only handle bare-Name registrations; bound methods / partials skipped.
                if not (len(call.args) == 1 and isinstance(call.args[0], ast.Name)):
                    skipped.append(
                        f"{src}:{call.lineno}: {call.func.attr}(...) registers a "
                        f"non-Name callback (lambda/method/partial); arity not statically checkable"
                    )
                    continue
                cb_name = call.args[0].id
                fn = defs_by_name.get(cb_name)
                if fn is None:
                    skipped.append(
                        f"{src}:{call.lineno}: {call.func.attr}({cb_name}) but {cb_name} "
                        f"is not defined as a function/lambda in this file (imported?)"
                    )
                    continue
                arity_info = _func_arity(fn)
                if arity_info is None:
                    continue
                consumer_arity, accepts_var = arity_info
                expected_arity = max(a for _, a in producers[cb_list])
                if accepts_var:
                    ok_count += 1
                    continue
                if consumer_arity != expected_arity:
                    issues.append(
                        f"{src}:{call.lineno}: {cb_name} declared with {consumer_arity} "
                        f"positional arg(s), but producer calls {cb_list} entries with "
                        f"{expected_arity} arg(s) "
                        f"({', '.join(str(p) for p, _ in producers[cb_list])})"
                    )
                else:
                    ok_count += 1
    return issues, skipped, ok_count


def _zoo_roots() -> list[pathlib.Path]:
    """Where to look for unsloth_zoo source. We try, in order:
      1. ``UNSLOTH_ZOO_SRC`` env var (a local git checkout).
      2. ``../unsloth-zoo`` next to this repo (common monorepo-style layout).
      3. The pip-installed package (wheel may strip platform-specific submodules
         like ``mlx/``, so this often misses MLX producers).
    Every root that exists is scanned; duplicates are fine.
    """
    roots: list[pathlib.Path] = []
    env_src = os.environ.get("UNSLOTH_ZOO_SRC")
    if env_src:
        p = pathlib.Path(env_src).expanduser().resolve()
        if p.exists():
            roots.append(p)
    sibling = (REPO_ROOT.parent / "unsloth-zoo").resolve()
    if sibling.exists():
        roots.append(sibling)
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is not None and spec.origin is not None:
        # spec.origin -> .../site-packages/unsloth_zoo/__init__.py
        # we want the unsloth_zoo dir itself, NOT the site-packages root which
        # contains every other installed pkg.
        roots.append(pathlib.Path(spec.origin).resolve().parent)
    return roots


def test_no_callback_signature_drift():
    roots = [REPO_ROOT, *_zoo_roots()]
    producers = discover_producers(roots)
    if not producers:
        import pytest

        pytest.skip(
            "no callback producer pattern (self._*_callbacks + cb(...)) found in "
            "unsloth or unsloth_zoo. Set UNSLOTH_ZOO_SRC=<path-to-unsloth-zoo-git-checkout> "
            "(the pip wheel strips platform-specific submodules like mlx/) to enable "
            "the detector locally."
        )
    issues, skipped, ok_count = check_registrations(roots, producers)
    msg_parts = [
        f"producers discovered: {len(producers)} ({sorted(producers)})",
        f"registrations matched: {ok_count}",
        f"registrations skipped: {len(skipped)}",
    ]
    if issues:
        msg_parts.append("")
        msg_parts.append("Callback signature drift detected:")
        msg_parts.extend("  " + i for i in issues)
        raise AssertionError("\n".join(msg_parts))
    if "-v" in sys.argv or "--verbose" in sys.argv:
        print("\n".join(msg_parts))


if __name__ == "__main__":
    # Allow running directly as a script for fast feedback.
    sys.argv.append("-v")
    test_no_callback_signature_drift()
    print("PASS")
