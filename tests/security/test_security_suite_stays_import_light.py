# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`workflow-trigger-lint.yml` runs `tests/security` on a four-package runner.

That job installs only pyyaml, pytest, pytest-xdist and vermin. It has no paths
filter, which is the point: it is the one job that still runs when a PR touches
nothing but workflow files. Keeping it that cheap is what lets it stay unfiltered.

So a file under `tests/security` may not import torch, transformers, numpy,
unsloth or unsloth_zoo at module scope unless the workflow explicitly ignores it.
Three files legitimately need those deps, and a torch-installing job runs them; the
workflow names them in `--ignore` so collection does not error.

The failure this guards against is quiet rather than loud. Under `-n 4` xdist does
not abort the session on a collection error, so an unlisted heavy import does not
turn the job red in an obvious way while its own tests stop running entirely. A
test that is silently never executed is worse than one that fails.

Pure AST and text: no imports of the modules under discussion, so this file is
itself safe to collect on the light runner.
"""

import ast
import pathlib
import re


HEAVY = {"torch", "transformers", "numpy", "unsloth", "unsloth_zoo", "peft", "trl"}

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_WORKFLOW = _REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"


def _constant_truth(test):
    """True/False for a statically decidable condition, else None."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _constant_truth(test.operand)
        return None if inner is None else not inner
    return None


def _own_expressions(node):
    """Every expression node belonging to `node` itself, not to a body it holds.

    `ast.walk` on the statement descends into the suites too, so a test FUNCTION
    defined inside a module-level `if` had its `pytest.importorskip("torch")` counted
    as a module-level import - the guard then demanded a workflow ignore entry for a
    file that is genuinely import-light. The suites are already visited separately by
    `_module_level_statements`, with function bodies excluded, so only the statement's
    own expressions are read here. A `lambda` body is deferred as well and is pruned
    for the same reason.
    """
    pending = [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]
    while pending:
        current = pending.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, ast.expr):
                pending.append(child)


def _module_level_statements(body):
    """Statements that run at import time, including inside module-level control flow.

    `try: import torch / except ImportError: ...` is the ordinary way to write an
    optional dependency, and it executes during collection exactly like a bare import.
    Looking only at direct children of the module missed it, so a file written that way
    was reported as light, never added to the workflow's ignore list, and would stop
    collecting on the light runner - the failure this guard exists to prevent.

    Function bodies are still excluded, since an import in there is paid lazily. A
    CLASS body is not: it executes while the class is built, which happens at import
    time, so `class C: import torch` costs exactly as much as a bare import.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # The BODY is deferred; the header is not. Decorators, parameter defaults,
            # runtime annotations and class bases are all evaluated while the module is
            # imported, so `def f(x = pytest.importorskip("torch"))` loads torch at
            # collection time. Skipping the statement wholesale missed every one of
            # them. Wrapped in an `Expr` so the walker below sees ordinary expressions.
            for part in _definition_time_expressions(node):
                yield ast.Expr(value = part)
            if isinstance(node, ast.ClassDef):
                yield from _module_level_statements(node.body)
            continue
        yield node
        if isinstance(node, ast.If) and isinstance(node.test, (ast.Constant, ast.UnaryOp)):
            decided = _constant_truth(node.test)
            if decided is not None:
                # `if False: import torch` never runs, so it does not make the file
                # need the heavy runner, and requiring an ignore entry for it would be
                # an ignore entry for nothing.
                yield from _module_level_statements(node.body if decided else node.orelse)
                continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            for child in getattr(node, field, []) or []:
                if isinstance(child, ast.ExceptHandler):
                    yield from _module_level_statements(child.body)
                elif isinstance(child, ast.stmt):
                    yield from _module_level_statements([child])
        # `match` keeps its suites under `cases`, not under any of the fields above, so
        # a module-level `match ...: case _: import torch` walked straight past.
        for case in getattr(node, "cases", []) or []:
            yield from _module_level_statements(case.body)


def _definition_time_expressions(node):
    """The parts of a `def`/`class` header that run at import time."""
    yield from node.decorator_list
    if isinstance(node, ast.ClassDef):
        yield from node.bases
        for keyword in node.keywords:
            yield keyword.value
        return
    arguments = node.args
    yield from arguments.defaults
    yield from (default for default in arguments.kw_defaults if default is not None)
    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ):
        if argument is not None and argument.annotation is not None:
            yield argument.annotation
    if node.returns is not None:
        yield node.returns


# The helpers that load a dependency with no `ast.Import` node of their own, and the
# module each one is imported FROM. `from importlib import import_module as load`
# renames the call without changing what it does, so the spelling at the call site is
# not enough to recognise it.
_LOADER_ORIGINS = {
    "import_module": "importlib",
    "__import__": "builtins",
    "importorskip": "pytest",
}


def _loader_aliases(tree):
    """Names bound to one of those helpers, by an import or by an assignment.

    Two spellings, both of which state outright what the name holds:
    `from <origin> import <helper> as <alias>`, and `alias = <helper>` where the right
    hand side is the helper written out (`importlib.import_module`, a bare name already
    recorded here, or the helper under its own name). Nothing is guessed at: a name
    assigned from anything else is not recorded.
    """
    # alias -> the helper it names, so a renamed `import_module` is not mistaken for
    # a renamed `importorskip`. A bare set lost that and `_guarded_roots` then read an
    # importlib call as a pytest skip guard.
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if _LOADER_ORIGINS.get(alias.name) == node.module:
                    aliases[alias.asname or alias.name] = alias.name
    # A second pass, because `load = importlib.import_module` may sit above the
    # `from ... import ... as ...` that named another helper, and one pass in source
    # order would then miss a chain. Repeated until nothing new is bound.
    while True:
        added = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Name) or target.id in aliases:
                continue
            if isinstance(value, ast.Attribute):
                held = value.attr
                owner = value.value.id if isinstance(value.value, ast.Name) else None
                if _LOADER_ORIGINS.get(held) != owner:
                    continue
            elif isinstance(value, ast.Name):
                if value.id in _LOADER_ORIGINS:
                    held = value.id
                elif value.id in aliases:
                    held = aliases[value.id]
                else:
                    continue
            else:
                continue
            aliases[target.id] = held
            added = True
        if not added:
            return aliases


def _loader_call_names(call, loaders):
    """The module names a dynamic loader call names outright, else an empty list.

    `importlib.import_module("torch")`, `__import__("torch")` and
    `pytest.importorskip("torch")`, under their own names or under an alias this file
    recorded. Only literal arguments are read, which is what these calls look like in
    practice, and the module name is taken positionally or by the keyword each helper
    documents.
    """
    if not isinstance(call, ast.Call):
        return []
    function = call.func
    attribute = (
        function.attr
        if isinstance(function, ast.Attribute)
        else (function.id if isinstance(function, ast.Name) else "")
    )
    if attribute not in _LOADER_ORIGINS and attribute not in loaders:
        return []
    candidates = list(call.args[:1])
    for keyword in call.keywords:
        if keyword.arg in ("modname", "name"):
            candidates.append(keyword.value)
    return [
        first.value
        for first in candidates
        if isinstance(first, ast.Constant) and isinstance(first.value, str)
    ]


def _is_importorskip(call, loaders):
    """Whether a call is `pytest.importorskip`, under any recorded spelling."""
    function = call.func
    attribute = (
        function.attr
        if isinstance(function, ast.Attribute)
        else (function.id if isinstance(function, ast.Name) else "")
    )
    return _is_pytest_skip(attribute, loaders)


def _is_pytest_skip(attribute, loaders) -> bool:
    """Whether this callee is `pytest.importorskip`, under any recorded spelling."""
    if attribute == "importorskip":
        return True
    return loaders.get(attribute) == "importorskip"


def _module_level_heavy_imports(path):
    """Top-level import names only. An import inside a function is paid lazily."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    loaders = _loader_aliases(tree)
    found = set()
    for node in _module_level_statements(tree.body):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module of its own to blame.
            names = [node.module or ""] if node.level == 0 else []
        else:
            # `torch = pytest.importorskip("torch")` and
            # `importlib.import_module("torch")` load the dependency with no `Import`
            # node at all. The first one is the dangerous spelling: on the light runner
            # it SKIPS the module during collection, so its tests silently disappear
            # while this guard reports the file as import-light. Only literal argument
            # forms are read, which is what these calls look like in practice.
            for call in _own_expressions(node):
                names.extend(_loader_call_names(call, loaders))
        for name in names:
            root = name.split(".")[0]
            if root in HEAVY:
                found.add(root)
    return found


def _body_level_heavy_imports(path):
    """Heavy dependencies a test BODY imports unconditionally.

    A module-scope import errors during COLLECTION on the light runner; one inside a
    test body errors when that test RUNS there, which is just as red and just as much
    a reason for the file to be redirected to the torch-dependent job. A
    `pytest.importorskip` inside a body is deliberate and skips only that test, so it
    does not count - unlike at module scope, where it silently skips the whole file.
    """
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    loaders = _loader_aliases(tree)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        skipped = _guarded_roots(node, loaders)
        for child in ast.walk(node):
            names = []
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and not child.level:
                names = [child.module or ""]
            elif isinstance(child, ast.Call):
                # `importlib.import_module("torch")` inside a body loads the
                # dependency with no `Import` node at all, and the light runner runs
                # that call and fails just the same. The module-scope scanner already
                # read these; the body one saw only the statement spellings and called
                # the file light. `importorskip` stays exempt here: it skips this one
                # test rather than erroring, which is the allowance the docstring
                # above already states.
                if not _is_importorskip(child, loaders):
                    names = _loader_call_names(child, loaders)
            for name in names:
                root = name.split(".")[0]
                if root in HEAVY and root not in skipped:
                    found.add(root)
    return found


def _reaches_a_heavy_dependency(statement, loaders):
    """Whether a statement loads a heavy dependency, however it spells the load."""
    for child in ast.walk(statement):
        names = []
        if isinstance(child, ast.Import):
            names = [alias.name for alias in child.names]
        elif isinstance(child, ast.ImportFrom) and not child.level:
            names = [child.module or ""]
        elif isinstance(child, ast.Call) and not _is_importorskip(child, loaders):
            names = _loader_call_names(child, loaders)
        if any(name.split(".")[0] in HEAVY for name in names):
            return True
    return False


def _guarded_roots(node, loaders):
    """Roots this body skips on before it does anything else with them.

    `def t(): pytest.importorskip("torch"); import torch` never reaches the import on
    the light runner - the skip fires first - so redirecting the whole file for it
    hides the unrelated light tests sitting next to it. That is the same allowance the
    docstring above already states for a bare `importorskip`; it just was not applied
    when a plain import followed.

    Only an UNCONDITIONAL call at the top level of the body counts. One inside an `if`
    or a `try` may not run, and then the import really is reached.
    """
    roots = set()
    for statement in node.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            # An import BEFORE the guard is reached first, and the light runner fails
            # there: `def t(): import torch; pytest.importorskip("torch")` never gets
            # to the skip. Scanning the whole body regardless of order recorded the
            # later call as a guard and let the file stay on the light runner, so the
            # scan stops at the first statement that loads anything heavy.
            if _reaches_a_heavy_dependency(statement, loaders):
                break
            continue
        call = statement.value
        function = call.func
        attribute = (
            function.attr
            if isinstance(function, ast.Attribute)
            else (function.id if isinstance(function, ast.Name) else "")
        )
        # `from importlib import import_module as load` puts `load` in `loaders` too,
        # and `_LOADER_ORIGINS.get("load")` is None, so an importlib call was being
        # read as a pytest skip guard - which then subtracted the dependency and left
        # the file on the runner where that call fails. The alias's ORIGIN decides,
        # not its presence in the table.
        if not _is_pytest_skip(attribute, loaders):
            continue
        candidates = list(call.args[:1])
        for keyword in call.keywords:
            if keyword.arg == "modname":
                candidates.append(keyword.value)
        for first in candidates:
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                roots.add(first.value.split(".")[0])
    return roots


def _needs_the_heavy_runner(path):
    """Either kind of import: one breaks collection there, the other breaks the run."""
    return _module_level_heavy_imports(path) | _body_level_heavy_imports(path)


def _ignored_by_the_workflow():
    text = _WORKFLOW.read_text(encoding = "utf-8")
    return {
        pathlib.PurePosixPath(m).name
        for m in re.findall(r"--ignore=(tests/security/[\w./-]+\.py)", text)
    }


def test_the_workflow_still_runs_the_security_suite():
    """Guards the guard: if the step stops naming the suite, the rest is vacuous."""
    assert _WORKFLOW.exists(), _WORKFLOW
    assert "tests/security" in _WORKFLOW.read_text(encoding = "utf-8")


def test_heavy_imports_are_declared_to_the_light_runner():
    ignored = _ignored_by_the_workflow()
    offenders = {}
    # conftest.py and __init__.py as well: pytest imports them during COLLECTION, so
    # a heavy import there breaks or skips the whole suite, and no per-file --ignore
    # can cover them. They are checked unconditionally for that reason.
    support = [p for p in (_HERE / "conftest.py", _HERE / "__init__.py") if p.exists()]
    for path in sorted(_HERE.glob("test_*.py")) + support:
        heavy = _module_level_heavy_imports(path)
        if heavy and (path.name not in ignored or path in support):
            offenders[path.name] = sorted(heavy)
    assert not offenders, (
        "these tests/security files import a runtime dependency at module scope but "
        f"are not in the workflow's --ignore list, so they will error during collection "
        f"on the four-package runner and then silently not run: {offenders}. Either "
        "move the import inside the test, or add --ignore for the file in "
        ".github/workflows/workflow-trigger-lint.yml."
    )


def test_the_ignore_list_has_no_stale_entries():
    """An ignore for a file that no longer needs it hides the file for no reason."""
    stale = []
    for name in sorted(_ignored_by_the_workflow()):
        path = _HERE / name
        if not path.exists():
            stale.append(f"{name} (no such file)")
        elif not _needs_the_heavy_runner(path):
            stale.append(f"{name} (no longer imports anything heavy)")
    assert not stale, f"drop these from --ignore: {stale}"


def test_a_body_level_heavy_import_also_needs_the_redirect():
    """The light runner RUNS these tests, so a body import is red there too.

    Pinned because it is the failure that got past the earlier version of this guard:
    a file whose module scope is clean but whose test bodies import `unsloth` was
    reported as import-light and left on the four-package runner.
    """
    sample = _HERE / "test_inherited_custom_dtype_is_neutralized.py"
    if not sample.exists():
        pytest.skip("the suite this pins has moved")
    assert not _module_level_heavy_imports(sample), "module scope should stay clean"
    assert _body_level_heavy_imports(sample), "its bodies do import a heavy dependency"
    assert sample.name in _ignored_by_the_workflow()


def test_the_guard_is_not_vacuous():
    """The list must actually be non-empty, or the two tests above prove nothing."""
    ignored = _ignored_by_the_workflow()
    assert ignored, "the workflow names no --ignore, so this suite is not being checked"
    with_heavy = [path.name for path in _HERE.glob("test_*.py") if _needs_the_heavy_runner(path)]
    assert with_heavy, "no file imports a heavy dep, so the ignore list should be empty"


def test_a_renamed_loader_helper_is_still_recognised(tmp_path):
    """`from importlib import import_module as load` is the same import.

    The helper is recognised by the name it was IMPORTED under, not by the spelling at
    the call site, so renaming it cannot hide an undeclared heavy dependency whose
    tests would then vanish from the light run.
    """
    spellings = {
        "aliased importlib helper": 'from importlib import import_module as load\ntorch = load("torch")\n',
        "aliased builtin": 'from builtins import __import__ as bring\ntorch = bring("torch")\n',
        "aliased importorskip": 'from pytest import importorskip as need\ntorch = need("torch")\n',
    }
    for description, source in spellings.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _module_level_heavy_imports(sample) == {"torch"}, description


def test_an_assignment_alias_of_a_loader_is_still_recognised(tmp_path):
    """`load = importlib.import_module` is the same helper under another name."""
    spellings = {
        "attribute assignment": 'import importlib\nload = importlib.import_module\ntorch = load("torch")\n',
        "importorskip by assignment": 'import pytest\nneed = pytest.importorskip\ntorch = need("torch")\n',
        "chained through an import alias": 'from importlib import import_module as first\nsecond = first\ntorch = second("torch")\n',
    }
    for description, source in spellings.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _module_level_heavy_imports(sample) == {"torch"}, description


def test_a_body_that_skips_first_does_not_redirect_the_file(tmp_path):
    """`importorskip` then `import` never reaches the import on the light runner.

    Redirecting the whole file for it would hide the unrelated light tests next to it.
    A guard that may not run does not count, which is what the last case pins.
    """
    cases = {
        'import pytest\n\n\ndef t():\n    pytest.importorskip("torch")\n    import torch\n': set(),
        'from pytest import importorskip as need\n\n\ndef t():\n    need("torch")\n    import torch\n': set(),
        'import pytest\n\n\ndef t():\n    if 0:\n        pytest.importorskip("torch")\n    import torch\n': {
            "torch"
        },
        "def t():\n    import torch\n": {"torch"},
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_a_guard_after_the_import_does_not_count(tmp_path):
    """The light runner reaches the import first and fails there.

    Order matters: `import torch` then `pytest.importorskip("torch")` is not guarded,
    though scanning the whole body regardless of position recorded it as one.
    """
    cases = {
        'import pytest\n\n\ndef t():\n    import torch\n    pytest.importorskip("torch")\n': {
            "torch"
        },
        'import pytest\n\n\ndef t():\n    pytest.importorskip("torch")\n    import torch\n': set(),
        # A statement that reaches nothing heavy does not end the guard run.
        'import pytest\n\n\ndef t():\n    x = 1\n    pytest.importorskip("torch")\n    import torch\n': set(),
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_a_dynamic_import_inside_a_body_also_needs_the_redirect(tmp_path):
    """A body can load a dependency with no `Import` node at all.

    The light runner runs the call and fails just as hard. `importorskip` stays
    exempt: it skips that one test rather than erroring.
    """
    cases = {
        'import importlib\n\n\ndef t():\n    importlib.import_module("torch")\n': {"torch"},
        'from importlib import import_module\n\n\ndef t():\n    import_module("torch")\n': {
            "torch"
        },
        'def t():\n    __import__("torch")\n': {"torch"},
        'import importlib\n\n\ndef t():\n    importlib.import_module(name = "torch")\n': {"torch"},
        'import pytest\n\n\ndef t():\n    pytest.importorskip("torch")\n': set(),
        "import importlib\n\n\ndef t():\n    importlib.import_module(picked)\n": set(),
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_a_renamed_importlib_alias_is_not_a_skip_guard(tmp_path):
    """`load("torch")` is an import, not a skip, however `load` was named.

    Both helpers can be renamed, and only the pytest one suppresses the dependency:
    the light runner runs an `import_module` call and fails.
    """
    cases = {
        'from importlib import import_module as load\n\n\ndef t():\n    load("torch")\n': {"torch"},
        'from pytest import importorskip as load\n\n\ndef t():\n    load("torch")\n': set(),
        'from importlib import import_module as load\n\n\ndef t():\n    load("torch")\n    import torch\n': {"torch"},
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_every_ignored_suite_runs_somewhere_else():
    """An --ignore on the light runner must be a redirect, not a deletion.

    The three torch-dependent suites are excluded from the only step that names
    `tests/security`, and for a while that was the whole story: nothing else ran them,
    so a regression in the hardening they cover could merge behind a green tick. This
    pins the other half - each ignored file has to be named by some workflow that is
    not the ignoring one.
    """
    ignored = _ignored_by_the_workflow()
    assert ignored, "no suite is ignored any more, so this guard has nothing to check"

    workflows = sorted(_WORKFLOW.parent.glob("*.yml"))
    elsewhere = {}
    for path in workflows:
        if path == _WORKFLOW:
            continue
        text = path.read_text(encoding = "utf-8")
        for name in ignored:
            if name in text:
                elsewhere.setdefault(name, []).append(path.name)

    missing = sorted(name for name in ignored if name not in elsewhere)
    assert not missing, (
        f"ignored by {_WORKFLOW.name} and run by no other workflow: {missing}. "
        f"An --ignore has to move a suite to a job that can install its "
        f"dependencies, not drop it out of CI."
    )


def test_the_redirect_job_is_triggered_by_what_it_protects():
    """A job that only runs after the merge is not a gate.

    The three suites moved off the light runner because they need torch. The workflow
    they moved to is path-filtered, and a filter that does not name the production code
    they cover means a PR touching only that code skips the job: the tests run for the
    first time on the push event, after the regression has merged.
    """
    audit = _WORKFLOW.parent / "security-audit.yml"
    assert audit.exists(), audit
    text = audit.read_text(encoding = "utf-8")

    # What the ignored suites actually read, taken from the suites rather than listed
    # here, so adding a module to one of them cannot leave this stale.
    protected = set()
    for name in sorted(_ignored_by_the_workflow()):
        path = _HERE / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "unsloth.models":
                    # `from unsloth.models import loader_utils` names the module in the
                    # alias, not in `node.module`.
                    protected.update(f"{alias.name}.py" for alias in node.names)
                elif node.module.startswith("unsloth.models."):
                    protected.add(node.module.split(".")[-1] + ".py")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("unsloth.models."):
                        protected.add(alias.name.split(".")[-1] + ".py")
            # `__import__("unsloth.models.loader", fromlist = ...)` too: one of these
            # suites reads the loader's source that way, and it is the module the test
            # is about.
            if isinstance(node, ast.Call) and node.args:
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else (function.id if isinstance(function, ast.Name) else "")
                )
                if name in ("__import__", "import_module"):
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        if first.value.startswith("unsloth.models."):
                            protected.add(first.value.split(".")[-1] + ".py")

    assert protected, "the ignored suites no longer import unsloth.models, so this is vacuous"
    missing = sorted(name for name in protected if f"unsloth/models/{name}" not in text)
    assert not missing, (
        f"security-audit.yml runs the suites that cover these modules but is not "
        f"triggered by changes to them: {missing}. A PR touching only that code would "
        f"skip the job and the suites would run only after merge."
    )
