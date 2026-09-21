# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`workflow-trigger-lint.yml` runs `tests/security` on a four-package runner.

That job has no paths filter, and keeping it cheap is what lets it stay unfiltered, so
a file there may not import anything in HEAVY at module scope unless the workflow
`--ignore`s it. The failure is quiet: under `-n 4` xdist does not abort on a collection
error, so the job stays green while that file's tests stop running entirely.
"""

import ast
import pathlib
import re

import yaml


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
    Lambdas are pruned by EXPANSION, since a lambda DEFAULT really does load."""

    def _expansion(current):
        if isinstance(current, ast.Lambda):
            arguments = current.args
            return arguments.defaults + [d for d in arguments.kw_defaults if d is not None]
        return list(ast.iter_child_nodes(current))

    pending = [child for child in _expansion(node) if isinstance(child, ast.expr)]
    while pending:
        current = pending.pop()
        yield current
        for child in _expansion(current):
            if isinstance(child, ast.expr):
                pending.append(child)


def _module_level_statements(body):
    """Statements that run at import time, including inside module-level control flow.
    Function bodies are excluded; a CLASS body is not, since it runs at import."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # The BODY is deferred; decorators, defaults, annotations and bases are evaluated at import.
            for part in _definition_time_expressions(node):
                yield ast.Expr(value = part)
            if isinstance(node, ast.ClassDef):
                yield from _module_level_statements(node.body)
            continue
        yield node
        if isinstance(node, ast.If) and isinstance(node.test, (ast.Constant, ast.UnaryOp)):
            decided = _constant_truth(node.test)
            if decided is not None:
                # `if False: import torch` never runs, so it needs no ignore entry.
                yield from _module_level_statements(node.body if decided else node.orelse)
                continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            for child in getattr(node, field, []) or []:
                if isinstance(child, ast.ExceptHandler):
                    yield from _module_level_statements(child.body)
                elif isinstance(child, ast.stmt):
                    yield from _module_level_statements([child])
        # `match` keeps its suites under `cases`, not under the fields above.
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


# Helpers that load with no `ast.Import` node, and the module each is imported FROM, since a rename makes the spelling
# at the call site insufficient.
_LOADER_ORIGINS = {
    "import_module": "importlib",
    "__import__": "builtins",
    "importorskip": "pytest",
}


_ALIAS_ROUNDS = 8


_ALIAS_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _scope_bindings(scope):
    """The import and assignment nodes written directly in this scope, not `ast.walk`:
    a nested `def`'s bindings belong to that scope rather than to this one."""
    found = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        current = pending.pop()
        if isinstance(current, _ALIAS_SCOPES):
            continue
        if isinstance(current, (ast.ImportFrom, ast.Assign)):
            found.append(current)
        pending.extend(ast.iter_child_nodes(current))
    return found


def _settled_aliases(bindings, inherited):
    """Names bound to one of those helpers, by `from <origin> import <helper> as
    <alias>` or `alias = <helper>`. Nothing else is guessed at."""
    # alias -> the helper it NAMES, so a renamed `import_module` is not read as a renamed `importorskip`.
    aliases: dict = dict(inherited)
    for _round in range(_ALIAS_ROUNDS):
        previous = aliases
        aliases = dict(inherited)
        for node in sorted(
            bindings,
            key = lambda child: (getattr(child, "lineno", 0), getattr(child, "col_offset", 0)),
        ):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if _LOADER_ORIGINS.get(alias.name) == node.module:
                        aliases[bound] = alias.name
                    else:
                        aliases.pop(bound, None)
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target, value = node.targets[0], node.value
            held = None
            if isinstance(value, ast.Attribute):
                owner = value.value.id if isinstance(value.value, ast.Name) else None
                if _LOADER_ORIGINS.get(value.attr) == owner:
                    held = value.attr
            elif isinstance(value, ast.Name):
                if value.id in _LOADER_ORIGINS:
                    held = value.id
                elif value.id in previous:
                    held = previous[value.id]
            if held is None:
                aliases.pop(target.id, None)
            else:
                aliases[target.id] = held
        if aliases == previous:
            break
    return aliases


def _alias_scopes(tree):
    """One alias table per lexical scope, each built on its enclosing one: a file-wide
    table let a binding in one function decide how an unrelated call was read."""
    scopes: dict = {}

    def visit(scope, inherited):
        table = _settled_aliases(_scope_bindings(scope), inherited)
        scopes[id(scope)] = table
        # A class body's names are invisible to its methods.
        handed_down = inherited if isinstance(scope, ast.ClassDef) else table
        pending = list(ast.iter_child_nodes(scope))
        while pending:
            current = pending.pop()
            if isinstance(current, _ALIAS_SCOPES):
                visit(current, handed_down)
                continue
            pending.extend(ast.iter_child_nodes(current))

    visit(tree, {})
    return scopes


def _loader_aliases(tree):
    """The module-scope alias table, for callers that read module level only."""
    return _alias_scopes(tree)[id(tree)]


def _loader_call_names(call, loaders):
    """The module names a dynamic loader call names outright, else an empty list.
    Literal arguments only, positionally or by the keyword each helper documents."""
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
            names = [node.module or ""] if node.level == 0 else []
        else:
            # `importorskip` SKIPS the whole file during collection at module scope.
            for call in _own_expressions(node):
                names.extend(_loader_call_names(call, loaders))
        for name in names:
            root = name.split(".")[0]
            if root in HEAVY:
                found.add(root)
    return found


def _body_level_heavy_imports(path):
    """Heavy dependencies a test BODY imports unconditionally. A body-level
    `importorskip` skips only that test, so unlike at module scope it does not count."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    scopes = _alias_scopes(tree)
    module_loaders = scopes[id(tree)]
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # This body's OWN table; another function's `load` says nothing about it.
        loaders = scopes.get(id(node), module_loaders)
        skipped = _guarded_roots(node, loaders)
        for child in ast.walk(node):
            names = []
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and not child.level:
                names = [child.module or ""]
            elif isinstance(child, ast.Call):
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
    """Roots this body skips on before it does anything else with them. Only an
    UNCONDITIONAL top-level call counts, since one inside an `if` may not run."""
    roots = set()
    for statement in node.body:
        held = None
        if isinstance(statement, ast.Expr):
            held = statement.value
        elif isinstance(statement, ast.Assign):
            held = statement.value
        elif isinstance(statement, ast.AnnAssign):
            held = statement.value
        if not isinstance(held, ast.Call):
            if _reaches_a_heavy_dependency(statement, loaders):
                break
            continue
        call = held
        function = call.func
        attribute = (
            function.attr
            if isinstance(function, ast.Attribute)
            else (function.id if isinstance(function, ast.Name) else "")
        )
        # The alias's ORIGIN decides: an importlib alias lives in `loaders` too.
        if not _is_pytest_skip(attribute, loaders):
            # An import ABOVE the guard is reached first, so the scan stops here.
            # The same stop, for calls: a non-guard call may be the load itself.
            if _reaches_a_heavy_dependency(statement, loaders):
                break
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
    # conftest.py and __init__.py too: no per-file --ignore can cover them.
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
    """The light runner RUNS these tests, so a body import is red there too."""
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
    """Recognised by the name it was IMPORTED under, not the spelling at the call."""
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
    """`importorskip` then `import` never reaches the import on the light runner; the
    third case pins that a guard which may not run does not count."""
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
    """The light runner reaches the import first and fails there."""
    cases = {
        'import pytest\n\n\ndef t():\n    import torch\n    pytest.importorskip("torch")\n': {
            "torch"
        },
        'import pytest\n\n\ndef t():\n    pytest.importorskip("torch")\n    import torch\n': set(),
        'import pytest\n\n\ndef t():\n    x = 1\n    pytest.importorskip("torch")\n    import torch\n': set(),
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_a_dynamic_import_inside_a_body_also_needs_the_redirect(tmp_path):
    """A body can load a dependency with no `Import` node at all; `importorskip` is
    still exempt, since it skips that one test rather than erroring."""
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
    """Both helpers can be renamed, and only the pytest one suppresses the dep."""
    cases = {
        'from importlib import import_module as load\n\n\ndef t():\n    load("torch")\n': {"torch"},
        'from pytest import importorskip as load\n\n\ndef t():\n    load("torch")\n': set(),
        'from importlib import import_module as load\n\n\ndef t():\n    load("torch")\n    import torch\n': {
            "torch"
        },
    }
    for source, expected in cases.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _body_level_heavy_imports(sample) == expected, source


def test_every_ignored_suite_runs_somewhere_else():
    """An --ignore must be a redirect, not a deletion: each ignored file has to be
    named by some workflow that is not the ignoring one."""
    ignored = _ignored_by_the_workflow()
    assert ignored, "no suite is ignored any more, so this guard has nothing to check"

    # By STEP: the redirect job sits in this same workflow.
    elsewhere = {}
    for path in sorted(_WORKFLOW.parent.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        for job in (document.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                command = str((step or {}).get("run", ""))
                for name in ignored:
                    if name in command and f"--ignore=tests/security/{name}" not in command:
                        elsewhere.setdefault(name, []).append(path.name)

    missing = sorted(name for name in ignored if name not in elsewhere)
    assert not missing, (
        f"ignored by {_WORKFLOW.name} and run by no other step: {missing}. "
        f"An --ignore has to move a suite to a job that can install its "
        f"dependencies, not drop it out of CI."
    )


def _redirect_workflow() -> pathlib.Path:
    """The workflow file whose steps run the ignored suites."""
    ignored = _ignored_by_the_workflow()
    for path in sorted(_WORKFLOW.parent.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        for job in (document.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                command = str((step or {}).get("run", ""))
                if any(
                    name in command and f"--ignore=tests/security/{name}" not in command
                    for name in ignored
                ):
                    return path
    raise AssertionError("no workflow step runs the ignored suites")


def test_the_redirect_job_is_triggered_by_what_it_protects():
    """A job that only runs after the merge is not a gate: if the redirect workflow's
    path filter omits the code the suites cover, a PR touching it skips the job."""
    host = _redirect_workflow()
    document = yaml.safe_load(host.read_text(encoding = "utf-8")) or {}
    triggers = document.get(True, document.get("on")) or {}
    on_pull_request = triggers.get("pull_request") if isinstance(triggers, dict) else None
    if not (
        isinstance(on_pull_request, dict)
        and (on_pull_request.get("paths") or on_pull_request.get("paths-ignore"))
    ):
        return
    text = host.read_text(encoding = "utf-8")

    # Taken from the suites, so adding a module to one cannot leave this stale.
    protected = set()
    for name in sorted(_ignored_by_the_workflow()):
        path = _HERE / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "unsloth.models":
                    protected.update(f"{alias.name}.py" for alias in node.names)
                elif node.module.startswith("unsloth.models."):
                    protected.add(node.module.split(".")[-1] + ".py")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("unsloth.models."):
                        protected.add(alias.name.split(".")[-1] + ".py")
            # `__import__("unsloth.models.loader", ...)` too: a suite reads it so.
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
        f"{host.name} runs the suites that cover these modules but is not "
        f"triggered by changes to them: {missing}. A PR touching only that code would "
        f"skip the job and the suites would run only after merge."
    )


def test_a_lambda_body_is_not_a_module_level_import(tmp_path):
    """A `lambda` runs when it is CALLED, and collection never calls this one."""
    spellings = {
        "assigned lambda": 'import pytest\nunused = lambda: pytest.importorskip("torch")\n',
        "lambda in a list": 'import pytest\nunused = [lambda: pytest.importorskip("torch")]\n',
        "lambda returning a lambda": (
            'import pytest\nunused = lambda: (lambda: pytest.importorskip("torch"))\n'
        ),
        "default of a lambda inside a lambda body": (
            "import pytest\n" 'unused = lambda: (lambda x = pytest.importorskip("torch"): x)\n'
        ),
    }
    for description, source in spellings.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _module_level_heavy_imports(sample) == set(), description


def test_a_lambda_default_is_still_a_module_level_import(tmp_path):
    """The half of a lambda that is NOT deferred: a default is evaluated where the
    lambda is written, so it loads torch during collection."""
    spellings = {
        "positional default": 'import pytest\nunused = lambda x = pytest.importorskip("torch"): x\n',
        "keyword-only default": (
            'import pytest\nunused = lambda *, x = pytest.importorskip("torch"): x\n'
        ),
        "lambda as another lambda's default": (
            "import pytest\n"
            'unused = lambda x = (lambda y = pytest.importorskip("torch"): y): x\n'
        ),
    }
    for description, source in spellings.items():
        sample = tmp_path / "sample.py"
        sample.write_text(source)
        assert _module_level_heavy_imports(sample) == {"torch"}, description


def test_a_later_assignment_replaces_an_imported_alias(tmp_path):
    """`load` stops being a skip guard once it is assigned the real importer. An alias recorded from an import could never be replaced, so a call through the reassigned name was still read as a pytest skip and the file stayed on the runner where that call fails."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from pytest import importorskip as load\n"
        "import importlib\n"
        "load = importlib.import_module\n"
        "def test_it():\n"
        '    torch = load("torch")\n'
    )
    assert _body_level_heavy_imports(sample) == {"torch"}


def test_an_imported_alias_still_wins_where_it_is_written_last(tmp_path):
    """The other direction: source order decides, so the import can replace too."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import importlib\n"
        "load = importlib.import_module\n"
        "from pytest import importorskip as load\n"
        "def test_it():\n"
        '    load("torch")\n'
        "    import torch\n"
    )
    assert _body_level_heavy_imports(sample) == set()


def test_an_assigned_skip_guard_still_guards(tmp_path):
    """`torch = pytest.importorskip("torch")` skips exactly as the bare call does."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import pytest\n"
        "def test_it():\n"
        '    torch = pytest.importorskip("torch")\n'
        "    from torch import nn\n"
        "    assert nn is not None\n"
    )
    assert _body_level_heavy_imports(sample) == set()


def test_an_import_before_the_assigned_guard_is_still_reported(tmp_path):
    """Order still decides: an import above the guard is reached first."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import pytest\n"
        "def test_it():\n"
        "    from torch import nn\n"
        '    torch = pytest.importorskip("torch")\n'
        "    assert nn is not None\n"
    )
    assert _body_level_heavy_imports(sample) == {"torch"}


def test_a_dynamic_load_before_the_guard_is_still_reported(tmp_path):
    """A dynamic load above the skip is reached first, exactly as an `import` is."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import importlib\n"
        "import pytest\n"
        "def test_it():\n"
        '    importlib.import_module("torch")\n'
        '    pytest.importorskip("torch")\n'
    )
    assert _body_level_heavy_imports(sample) == {"torch"}


def test_a_local_alias_does_not_reclassify_another_scope(tmp_path):
    """A binding inside one function must not relabel a call in an unrelated one."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from importlib import import_module as load\n"
        "def test_it():\n"
        '    load("torch")\n'
        "def helper():\n"
        "    from pytest import importorskip as load\n"
        '    return load("torch")\n'
    )
    assert _body_level_heavy_imports(sample) == {"torch"}
