"""The package must stay importable on the Python floor ``pyproject.toml`` declares.

It was not. ``requires-python`` says ``>=3.9`` while ``registry/registry.py`` annotated a
dataclass field ``list[QuantType] | dict[str, list[QuantType]]``, which evaluates at class
creation, so importing ``unsloth.registry.registry`` died with ``TypeError: unsupported
operand type(s) for |``. Nothing caught it: every CI job here pins 3.12.

The floor is read from ``pyproject.toml`` rather than hardcoded, so raising
``requires-python`` relaxes these checks instead of turning them into a false alarm.

This is a static AST check. It imports nothing from the package, so it needs no torch, no
GPU and no network, and it sees files that are never imported at test time.
"""

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "unsloth"

# studio/ ships under the same requires-python, so it is held to the syntax check. Its
# evaluated-union debt is ratcheted rather than fixed here: the files involved include FastAPI routers and pydantic
# models, where `from __future__ import annotations` is supported but has real failure modes around class
# dependencies, so converting them needs Unsloth actually booted and its routes exercised.
#
# The SET, not just a count, so a breach can name the files it added. Shrinking is the only edit that should ever be
# made here: a new entry means a new file evaluates a union on the floor, and the fix is the future import in that
# file, not a longer list.
STUDIO_UNION_DEBT_FILES = frozenset(
    {
        "studio/backend/auth/hashing.py",
        "studio/backend/core/inference/external_provider.py",
        "studio/backend/core/inference/key_exchange.py",
        "studio/backend/core/inference/providers.py",
        "studio/backend/core/inference/tools.py",
        "studio/backend/core/training/trainer.py",
        "studio/backend/hub/tests/test_model_services.py",
        "studio/backend/main.py",
        "studio/backend/routes/auth.py",
        "studio/backend/routes/inference.py",
        "studio/backend/routes/mcp_servers.py",
        "studio/backend/routes/models.py",
        "studio/backend/routes/providers.py",
        "studio/backend/routes/settings.py",
        "studio/backend/storage/rag_db.py",
        "studio/backend/storage/studio_db.py",
        "studio/backend/tests/test_anthropic_citations_edge.py",
        "studio/backend/tests/test_cached_gguf_routes.py",
        "studio/backend/tests/test_chat_attachments.py",
        "studio/backend/tests/test_chat_history_storage.py",
        "studio/backend/tests/test_export_absolute_paths.py",
        "studio/backend/tests/test_index_bootstrap_origin.py",
        "studio/backend/tests/test_kv_cache_estimation.py",
        "studio/backend/tests/test_middleware.py",
        "studio/backend/tests/test_openai_citation_markers.py",
        "studio/backend/tests/test_openai_citation_markers_edge.py",
        "studio/backend/tests/test_setup_llama_cpp_backend.py",
        "studio/backend/tests/test_training_history_update.py",
        "studio/backend/tests/test_transformers_version.py",
        "studio/backend/utils/datasets/dataset_utils.py",
        "studio/backend/utils/datasets/format_conversion.py",
        "studio/backend/utils/datasets/format_detection.py",
        "studio/backend/utils/models/model_config.py",
        "studio/backend/utils/transformers_latest.py",
        "studio/backend/utils/transformers_version.py",
    }
)


def declared_floor():
    """The ``>=X.Y`` in requires-python, as a tuple for ast.parse(feature_version=)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def guarded_floor(init_path):
    """The floor an ``__init__.py`` refuses to import below, if it declares one.

    The shape a vendored package uses to state its own requirement:

        if sys.version_info < (3, 10):
            raise ImportError("truststore requires Python 3.10 or later")

    Returns ``(3, 10)`` there, ``None`` when no such guard exists.
    """
    try:
        tree = ast.parse(init_path.read_text(encoding = "utf-8"), filename = str(init_path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt)
            and "version_info" in ast.dump(test.left)
            and isinstance(test.comparators[0], ast.Tuple)
        ):
            continue
        if not any(isinstance(stmt, ast.Raise) for stmt in node.body):
            continue
        parts = [e.value for e in test.comparators[0].elts if isinstance(e, ast.Constant)]
        if len(parts) >= 2 and all(isinstance(v, int) for v in parts[:2]):
            return (parts[0], parts[1])
    return None


def floor_guarded_dirs(root):
    """Package directories that refuse to import below a floor above ours.

    `studio/backend/vendor/truststore` is vendored third-party code whose
    `__init__.py` raises on anything under 3.10, so the PEP 604 type aliases in
    its `_api.py` can never evaluate on our 3.9 floor: the package is gone
    before that module is reached, and its one caller wraps `import truststore`
    in try/except. Scanning those files reports a break that cannot happen.

    Keyed on the guard, not the path, so unguarded code dropped into the same
    vendor directory is still scanned. A blanket `vendor/` exclusion would have
    covered that silently.
    """
    guarded = []
    for init in root.rglob("__init__.py"):
        if "__pycache__" in init.parts:
            continue
        floor = guarded_floor(init)
        if floor and floor > declared_floor():
            guarded.append(init.parent)
    return guarded


def package_files(root = PACKAGE_ROOT, minimum = 50):
    skip = floor_guarded_dirs(root)
    files = sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
        and "node_modules" not in p.parts
        and not any(d in p.parents for d in skip)
    )
    assert len(files) >= minimum, f"only found {len(files)} files under {root}, glob is wrong"
    return files


def packaged_roots():
    """Top-level directories setuptools ships, from the `include` list in pyproject.

    Read rather than hardcoded so a newly packaged directory cannot silently escape the
    floor guarantee.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    block = re.search(r"^include\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert block, "no packages.find include list in pyproject.toml"
    roots = []
    for pattern in re.findall(r"[\"']([^\"']+)[\"']", block.group(1)):
        top = pattern.split(".")[0].rstrip("*")
        candidate = REPO_ROOT / top
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    assert roots, "no packaged directories resolved from pyproject"
    return roots


def evaluated_union_files(root):
    """Files under `root` that would raise on the floor, i.e. need the future import."""
    offenders = set()
    for path in package_files(root, minimum = 1):
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        if has_future_annotations(tree):
            continue
        for expression in evaluated_annotations(tree):
            for inner in ast.walk(expression):
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                    offenders.add(path)
    return offenders


def signature_annotations(node):
    """Parameter and return annotations, evaluated when the ``def`` executes."""
    args = node.args
    out = [
        a.annotation
        for a in [*args.args, *args.kwonlyargs, *args.posonlyargs, args.vararg, args.kwarg]
        if a is not None and a.annotation is not None
    ]
    if node.returns:
        out.append(node.returns)
    return out


def evaluated_annotations(tree):
    """Annotations Python evaluates, so the future import is what defers them.

    Variable annotations are evaluated at module and class scope only; inside a function
    body they are never evaluated, so flagging those is a false positive. Signature
    annotations are evaluated wherever their ``def`` is.
    """
    out = []

    def walk(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AnnAssign) and child.annotation and not in_function:
                out.append(child.annotation)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(signature_annotations(child))
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, True)
            elif isinstance(child, ast.ClassDef):
                walk(child, False)  # a class body is class scope wherever it sits
            else:
                walk(child, in_function)

    walk(tree, False)
    return out


# A `|` between these is a union, not arithmetic. These are the builtin types; `None` and whatever the module pulled
# in from typing count as anchors too, and are handled in the check below.
TYPE_ANCHORS = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "object",
        "type",
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
    }
)
TYPING_MODULES = frozenset({"typing", "typing_extensions", "collections.abc"})


def typing_names(tree):
    """Names this module bound with ``from typing import X`` and friends."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in TYPING_MODULES
        for alias in node.names
    }


def union_operands(node):
    """Operands of an ``A | B | C`` chain, or None if any part is not name-shaped."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = union_operands(node.left), union_operands(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Subscript):
        return union_operands(node.value)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return [node]
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return [node]
    return None


def looks_like_a_type_alias(node, known_typing_names):
    """``PathLike = str | Path`` yes; ``defaults | extra`` and ``re.A | re.M`` no.

    Every operand must be name-shaped and at least one must be a recognisable type.
    Without that anchor a ``|`` between plain names is far more likely to be a dict merge
    (PEP 584, valid on 3.9), a set union or flag arithmetic, and failing the gate on those
    would block code that runs perfectly well on the floor.
    """
    operands = union_operands(node)
    if not operands:
        return False
    for operand in operands:
        if isinstance(operand, ast.Constant):
            if operand.value is None:
                return True
            continue  # a bare string is a forward reference, never an anchor alone
        name = operand.id if isinstance(operand, ast.Name) else operand.attr
        if name in TYPE_ANCHORS or name in known_typing_names:
            return True
    return False


def evaluated_values(tree):
    """Expressions that run at import, at module or class scope.

    Type aliases are the case that matters: ``PathLike = str | Path`` raises below 3.10
    and, unlike an annotation, the future import does not defer it. Decorator expressions
    and parameter defaults are evaluated the same way, so they belong here too. Control
    flow is descended into (a branch that runs, runs) but function bodies are not, since
    those only execute when called.
    """
    out = []
    scoped = (
        ast.If,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.ClassDef,
    )

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(child.decorator_list)
                out.extend(d for d in child.args.defaults if d is not None)
                out.extend(d for d in child.args.kw_defaults if d is not None)
                continue  # the body only runs when called
            if isinstance(child, ast.ClassDef):
                out.extend(child.decorator_list)
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                out.append(child.value)
            if isinstance(child, scoped):
                walk(child)

    walk(tree)
    return out


def has_future_annotations(tree):
    return any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(alias.name == "annotations" for alias in n.names)
        for n in tree.body
    )


def test_every_packaged_module_parses_on_the_declared_floor():
    """Every directory pyproject ships, not just unsloth/ - studio/ is packaged too."""
    floor = declared_floor()
    broken = []
    for root in packaged_roots():
        for path in package_files(root, minimum = 1):
            try:
                ast.parse(
                    path.read_text(encoding = "utf-8"),
                    filename = str(path),
                    feature_version = floor,
                )
            except SyntaxError as error:
                broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, (
        "syntax newer than the declared floor "
        f"{floor[0]}.{floor[1]} in a packaged directory; either rewrite it or raise "
        "requires-python:\n  " + "\n  ".join(broken)
    )


def test_every_packaged_module_compiles():
    """``ast.parse`` accepts things ``compile`` rejects, and only compile runs on import.

    A misplaced ``from __future__ import annotations`` is the case that bites here: it
    parses, it makes ``has_future_annotations`` suppress the union check, and it still
    raises SyntaxError on import.
    """
    broken = []
    for root in packaged_roots():
        for path in package_files(root, minimum = 1):
            try:
                compile(path.read_text(encoding = "utf-8"), str(path), "exec", dont_inherit = True)
            except SyntaxError as error:
                broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, "these modules do not compile:\n  " + "\n  ".join(broken)


def test_studio_evaluated_unions_do_not_grow():
    """studio/ is shipped on the same floor but still carries unions that raise there.

    A ratchet, not a pass: converting those files needs Unsloth booted and its routes
    exercised, because FastAPI resolves annotations when it builds each endpoint. This
    keeps the debt from growing in the meantime.
    """
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    studio = REPO_ROOT / "studio"
    if not studio.is_dir():
        pytest.skip("no studio/ directory in this checkout")
    offenders = sorted(str(p.relative_to(REPO_ROOT)) for p in evaluated_union_files(studio))
    # Name the files that are NEW against the recorded set, not the whole list.
    # A bare count told you only that 37 exceeded 35, and the full list was truncated at 2000 chars, so finding the two
    # additions meant re-running the scan on an older checkout and diffing by hand.
    added = [p for p in offenders if p not in STUDIO_UNION_DEBT_FILES]
    removed = [p for p in sorted(STUDIO_UNION_DEBT_FILES) if p not in offenders]
    assert len(offenders) <= len(STUDIO_UNION_DEBT_FILES), (
        f"{len(offenders)} studio files now evaluate PEP 604 unions on the floor, up from "
        f"{len(STUDIO_UNION_DEBT_FILES)}. Add `from __future__ import annotations` to "
        "these, do not raise the ratchet:\n  "
        + "\n  ".join(added or offenders)
        + (
            "\n\nalso no longer offending (drop them from STUDIO_UNION_DEBT_FILES):\n  "
            + "\n  ".join(removed)
            if removed
            else ""
        )
    )


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """``X | Y`` is a TypeError below 3.10 unless the module defers it."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    scanned = [p for root in packaged_roots() for p in package_files(root, minimum = 1)]
    for path in scanned:
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        where = path.relative_to(REPO_ROOT)
        known_typing_names = typing_names(tree)
        # The future import defers annotations only; an assigned value still runs.
        deferred = has_future_annotations(tree)
        in_unsloth = PACKAGE_ROOT in path.parents
        annotation_sources = [] if (deferred or not in_unsloth) else evaluated_annotations(tree)
        for expression in annotation_sources:
            for inner in ast.walk(expression):
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                    offenders.append(f"{where}:{inner.lineno}: {ast.unparse(inner)} (annotation)")
        for expression in evaluated_values(tree):
            for inner in ast.walk(expression):
                if (
                    isinstance(inner, ast.BinOp)
                    and isinstance(inner.op, ast.BitOr)
                    and looks_like_a_type_alias(inner, known_typing_names)
                ):
                    offenders.append(f"{where}:{inner.lineno}: {ast.unparse(inner)} (type alias)")
    assert not offenders, (
        "PEP 604 unions evaluated below 3.10. Annotations are deferred by `from __future__ "
        "import annotations`; type aliases need typing.Union, which that import does NOT "
        "defer:\n  " + "\n  ".join(sorted(set(offenders)))
    )


# ---- the exemption itself ------------------------------------------------
#
# The skip above is the kind of thing that rots into a blanket `vendor/`
# exclusion. These pin it to the guard.
def test_the_truststore_guard_is_what_exempts_it():
    """Not the path. If upstream drops the version guard, the files come back
    into the scan and this gate goes red again -- which is correct, because at
    that point `import truststore` really can reach `_api.py` on 3.9."""
    init = REPO_ROOT / "studio/backend/vendor/truststore/__init__.py"
    if not init.exists():
        pytest.skip("truststore is not vendored in this checkout")
    assert guarded_floor(init) == (3, 10)
    assert init.parent in floor_guarded_dirs(REPO_ROOT / "studio")


def test_unguarded_code_in_the_same_vendor_directory_is_still_scanned(tmp_path):
    vendor = tmp_path / "vendor"
    (vendor / "guarded").mkdir(parents = True)
    (vendor / "plain").mkdir()
    (vendor / "guarded" / "__init__.py").write_text(
        "import sys\nif sys.version_info < (3, 10):\n    raise ImportError('nope')\n"
    )
    (vendor / "plain" / "__init__.py").write_text("")
    skipped = floor_guarded_dirs(tmp_path)
    assert vendor / "guarded" in skipped
    assert vendor / "plain" not in skipped


def test_a_guard_at_or_below_our_floor_does_not_exempt(tmp_path):
    """A package that merely requires 3.9 is a package we ship on. Exempting it
    would hide real breakage."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "import sys\nif sys.version_info < (3, 9):\n    raise ImportError('nope')\n"
    )
    assert floor_guarded_dirs(tmp_path) == []


def test_a_guard_that_does_not_raise_does_not_exempt(tmp_path):
    """`if sys.version_info < (3, 10): warnings.warn(...)` still imports."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "import sys, warnings\nif sys.version_info < (3, 10):\n    warnings.warn('old')\n"
    )
    assert floor_guarded_dirs(tmp_path) == []
