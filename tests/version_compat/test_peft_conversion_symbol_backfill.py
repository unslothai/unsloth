# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""transformers can keep a module and drop the names peft imports from it.

`Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game` dies on main with

    ImportError: cannot import name '_MODEL_TO_CONVERSION_PATTERN'
    from 'transformers.conversion_mapping'

import_fixes already stubs both modules when they are ABSENT, but here the
module is present and only the symbol is gone, so the guard fell through and
no-oped. The names peft imports at module top are backfilled onto the real
module instead: additive only, so a transformers that still exports them is
untouched.
"""

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


def _load_import_fixes():
    """By file path, not `from unsloth import ...`.

    `version-compat-ci.yml`'s `daily-fresh-fetch` job collects this whole
    directory with pytest as its only dependency, and importing the package
    runs `_gpu_init.py`, which needs NumPy. Collection would stop there,
    before a single test in this directory ran.
    """
    path = Path(__file__).resolve().parents[2] / "unsloth" / "import_fixes.py"
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_for_backfill_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = _load_import_fixes()


@pytest.fixture()
def fake_modules():
    """Two real-looking modules missing exactly the drifted names."""
    saved = {n: sys.modules.get(n) for n in F._PEFT_CONVERSION_SYMBOLS}
    for name in F._PEFT_CONVERSION_SYMBOLS:
        mod = types.ModuleType(name)
        mod.__file__ = f"<fake {name}>"
        mod.something_else = object()  # proves the module is not replaced
        sys.modules[name] = mod
    try:
        yield {n: sys.modules[n] for n in F._PEFT_CONVERSION_SYMBOLS}
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_the_missing_names_are_added(fake_modules):
    assert F._backfill_missing_conversion_symbols() is True
    for name, symbols in F._PEFT_CONVERSION_SYMBOLS.items():
        for symbol in symbols:
            assert hasattr(fake_modules[name], symbol), f"{name}.{symbol}"


def test_the_real_module_is_not_replaced(fake_modules):
    before = {n: m for n, m in fake_modules.items()}
    F._backfill_missing_conversion_symbols()
    for name, mod in before.items():
        assert sys.modules[name] is mod, "the module object must survive"
        assert hasattr(mod, "something_else"), "its own exports must survive"


def test_a_complete_module_is_untouched(fake_modules):
    """A transformers that still exports them must see no change at all."""
    sentinels = {}
    for name, symbols in F._PEFT_CONVERSION_SYMBOLS.items():
        for symbol in symbols:
            sentinels[(name, symbol)] = object()
            setattr(fake_modules[name], symbol, sentinels[(name, symbol)])

    assert F._backfill_missing_conversion_symbols() is False
    for (name, symbol), value in sentinels.items():
        assert getattr(fake_modules[name], symbol) is value, "an existing symbol was overwritten"


def test_it_is_idempotent(fake_modules):
    assert F._backfill_missing_conversion_symbols() is True
    assert F._backfill_missing_conversion_symbols() is False


def test_a_module_blocked_by_another_drift_is_retried(monkeypatch):
    """The two drifts together used to be unrecoverable in one pass.

    `transformers.conversion_mapping` imports names from
    `transformers.core_model_loading` at its own module top, so while those are
    missing, importing it raises and the pass skips it. Backfilling
    core_model_loading later in the SAME pass unblocks that import, but nothing
    came back for conversion_mapping and `_gpu_init` calls this guard once, so an
    installation carrying both drifts stayed broken.
    """
    names = list(F._PEFT_CONVERSION_SYMBOLS)
    blocked, unblocker = names[0], names[-1]
    assert blocked != unblocker, "this test needs two modules to order"

    saved = {n: sys.modules.get(n) for n in names}
    for name in names:
        sys.modules.pop(name, None)

    real_import = importlib.import_module
    attempts = {"blocked": 0}

    def fake_import(name, *a, **kw):
        if name == blocked:
            attempts["blocked"] += 1
            # Importable only once the other module has been backfilled, which is exactly the real dependency.
            other = sys.modules.get(unblocker)
            if other is None or any(
                not hasattr(other, s) for s in F._PEFT_CONVERSION_SYMBOLS[unblocker]
            ):
                raise ImportError(f"cannot import name from {unblocker}")
            mod = types.ModuleType(name)
            mod.__file__ = f"<fake {name}>"
            sys.modules[name] = mod
            return mod
        if name == unblocker:
            mod = types.ModuleType(name)
            mod.__file__ = f"<fake {name}>"
            sys.modules[name] = mod
            return mod
        return real_import(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    try:
        assert F._backfill_missing_conversion_symbols() is True
        assert attempts["blocked"] >= 2, "the blocked module was never retried"
        for symbol in F._PEFT_CONVERSION_SYMBOLS[blocked]:
            assert hasattr(sys.modules[blocked], symbol), f"{blocked}.{symbol} never backfilled"
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_the_retry_stops_when_a_module_stays_unimportable(monkeypatch):
    """A module that never imports must not spin the loop."""
    names = list(F._PEFT_CONVERSION_SYMBOLS)
    saved = {n: sys.modules.get(n) for n in names}
    for name in names:
        sys.modules.pop(name, None)

    calls = {"n": 0}

    def always_fails(name, *a, **kw):
        calls["n"] += 1
        raise ImportError("nope")

    monkeypatch.setattr(importlib, "import_module", always_fails)
    try:
        assert F._backfill_missing_conversion_symbols() is False
        # Nothing was added, so the loop breaks after one pass.
        assert calls["n"] == len(names)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _peft_converter_source():
    """peft's own source, from the installed package or from GitHub.

    `version-compat-ci.yml`'s `daily-fresh-fetch` job installs only pytest, so
    `find_spec("peft")` is None there and this authority check reported as a
    skip: a new import added upstream would never fail the guard it exists to
    be. The pinned-symbol suites in this directory already read upstream over
    the network, so do the same and fall back to the installed copy.

    Both layouts are tried, the single module and the package, because peft can
    split this file into `transformers_weight_conversion/__init__.py` at any
    time. A packaging-only move would 404 the sole URL and drop straight back
    to the installed copy, which is absent in that job, so the guard would go
    quiet exactly when upstream churn is highest.
    `test_peft_pinned_symbols.py` already probes the same pair.

    A package `__init__.py` that only re-exports would then report no transformers
    imports at all, so the submodules it pulls in at import time are fetched too
    and appended. Importing the package runs them, so an unlisted symbol in one
    breaks startup exactly as it would in the flat file.
    """
    import os
    import urllib.error
    import urllib.request

    base = "https://raw.githubusercontent.com/huggingface/peft/main/src/peft/utils/"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    def fetch(url):
        """The file's text, None on a 404, skip on anything else."""
        request = urllib.request.Request(url)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout = 15) as response:
                return response.read().decode("utf-8", errors = "replace")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                pytest.skip(f"GitHub fetch failed ({exc.code}) for {url}")
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"GitHub fetch failed ({exc}) for {url}")

    flat = fetch(base + "transformers_weight_conversion.py")
    if flat is not None:
        return flat

    pkg = base + "transformers_weight_conversion/"
    init = fetch(pkg + "__init__.py")
    if init is not None:
        # Breadth-first, not one level:
        sources = [init]
        # Each entry is the module's dotted path from the package root.
        # A child's own relative imports resolve against ITS package, not the root: in a nested split where `__init__`
        # imports `.sub.core` and `sub/core.py` imports `.ops`, Python reads that as `sub.ops`, so carry the prefix
        # rather than fetching `ops.py`.
        pending = _resolved_relative_targets("", init)
        seen = set()
        while pending:
            child = pending.pop(0)
            if child in seen:
                continue
            seen.add(child)
            path = child.replace(".", "/")
            for candidate, package in (
                (f"{pkg}{path}.py", child.rpartition(".")[0]),
                (f"{pkg}{path}/__init__.py", child),
            ):
                text = fetch(candidate)
                if text is None:
                    continue
                sources.append(text)
                pending.extend(_resolved_relative_targets(package, text))
                break
        return "\n\n".join(src if src.endswith("\n") else src + "\n" for src in sources)

    module = "peft.utils.transformers_weight_conversion"
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed and upstream could not be read")
    try:
        import inspect
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "") in (module, "peft.utils", "peft"):
            pytest.skip("this peft has no weight converter")
        raise
    # imports -- the ones that would fail at startup -- are never read. That is
    # If peft turns the converter into a package, `getsource` returns only the `__init__.py` re-exports, and the
    sources = [inspect.getsource(loaded)]
    for path in getattr(loaded, "__path__", ()) or ():
        import pkgutil
        for info in pkgutil.iter_modules([path]):
            try:
                sources.append(inspect.getsource(importlib.import_module(f"{module}.{info.name}")))
            except Exception:
                continue  # a child that will not import cannot be read
    return "\n".join(sources)


# The converter package itself, so an absolute import of its own children can be told from an unrelated one.
# Absolute, but still the package's own child: peft can re-export its implementation as `from
# peft.utils.transformers_weight_conversion .core import build`, which loads the same file a relative import would and
# was queued by neither branch.
_CONVERTER_MODULE = "peft.utils.transformers_weight_conversion"


def _resolved_relative_targets(package, src):
    """`_relative_import_targets`, resolved against the importing module's package.

    A level counts dots: `from .ops import X` is level 1 and resolves inside
    `package`, `from ..ops import X` is level 2 and resolves one package up.
    Prefixing `package` onto every name regardless probed `sub.ops` for the
    second form and silently read nothing, which is a drift check that cannot
    fail. A level that walks above the converter package leaves the code this
    test is about, so it is dropped rather than guessed at.
    """
    parts = package.split(".") if package else []
    resolved = []
    for level, name in _relative_import_targets(src):
        if level == 0:  # absolute, already rooted at the converter package
            resolved.append(name)
            continue
        drop = level - 1
        if drop > len(parts):
            continue
        base = parts[: len(parts) - drop] if drop else parts
        resolved.append(".".join([*base, name]))
    return resolved


def _relative_import_targets(src):
    """`(level, submodule)` for every relative import a module runs at import time.

    `from .core import X` and `from . import core` both name `core`, at level 1.
    The level travels with the name because the caller has to resolve it against
    the importing module's own package. Only the modules the package actually
    pulls in are followed, so a package that re-exports one implementation file
    costs one extra fetch.
    """
    import ast

    targets = []

    def visit(body) -> None:
        for node in body:
            # Absolute, but still the package's own child:
            if (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and (node.module or "").startswith(_CONVERTER_MODULE)
            ):
                inner = (node.module or "")[len(_CONVERTER_MODULE) :].lstrip(".")
                if inner:
                    targets.append((0, inner))
                targets.extend(
                    (0, f"{inner}.{a.name}" if inner else a.name)
                    for a in node.names
                    if a.name != "*"
                )
                continue
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(_CONVERTER_MODULE + "."):
                        targets.append((0, a.name[len(_CONVERTER_MODULE) + 1 :]))
                continue
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.module:
                    targets.append((node.level, node.module))
                    # `from .sub import core` imports `pkg.sub.core` when `core` is a module, whether or not
                    targets.extend(
                        (node.level, f"{node.module}.{alias.name}")
                        for alias in node.names
                        if alias.name != "*"
                    )
                else:
                    targets.extend((node.level, alias.name) for alias in node.names)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import)):
                continue
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                visit(node.body if isinstance(node.test, ast.UnaryOp) else node.orelse)
                continue
            # Same import-time traversal as `_transformers_imports`:
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                child = getattr(node, field, None)
                if not isinstance(child, list):
                    continue
                for item in child:
                    if isinstance(item, ast.stmt):
                        visit([item])
                    else:
                        visit(getattr(item, "body", []))

    visit(ast.parse(src).body)
    return list(dict.fromkeys(targets))


def _is_type_checking(test) -> bool:
    """`TYPE_CHECKING`, `typing.TYPE_CHECKING`, or either negated."""
    import ast

    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _transformers_imports(src):
    """{module: {symbol}} for every import transformers... executed at module load.

    AST, not a regex over the source. A regex for the parenthesised form alone
    read `from transformers.core_model_loading import A, B` as importing
    nothing, and looking up only the modules already in the table never
    examined a third transformers module at all. Either one leaves the drift
    this file exists to catch reporting green.

    Walking the whole tree is equally wrong in the other direction. An import
    under `if TYPE_CHECKING:` or inside a function body never runs when the
    converter is imported, so it cannot raise the startup `ImportError` this
    backfill exists to absorb; requiring it in the table would fail
    `daily-fresh-fetch` over a type annotation. So descend only through
    statements that execute at module load: `if`/`try`/`with`/loop bodies yes,
    typing-only branches and function bodies no. A class body is NOT a function
    body: it executes the moment the module defines the class, so an import
    inside one can break startup and is collected. A conditional module-level
    import counts too -- it can run, and that is the bar.
    """
    import ast

    out = {}

    def visit(body) -> None:
        for node in body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers"):
                out.setdefault(node.module, set()).update(a.name for a in node.names)
                continue
            # `import transformers.x` raises the same startup ModuleNotFoundError as the `from` form, and skipping it
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("transformers"):
                        out.setdefault(alias.name, set())
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                # `if TYPE_CHECKING:` never runs; `if not TYPE_CHECKING:` runs the OTHER branch.
                visit(node.body if isinstance(node.test, ast.UnaryOp) else node.orelse)
                continue
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                child = getattr(node, field, None)
                if not isinstance(child, list):
                    continue
                for item in child:
                    if isinstance(item, ast.stmt):
                        visit([item])
                    else:  # an ExceptHandler holds its own statement list
                        visit(getattr(item, "body", []))

    visit(ast.parse(src).body)
    return out


def test_both_import_spellings_are_parsed():
    """peft writes either, and reformatting one must not blind the guard."""
    src = (
        "from transformers.core_model_loading import Concatenate, ConversionOps\n"
        "from transformers.utils import (\n    logging,\n    is_torch_available,\n)\n"
        "import os\n"
        "from peft.utils import something\n"
    )
    assert _transformers_imports(src) == {
        "transformers.core_model_loading": {"Concatenate", "ConversionOps"},
        "transformers.utils": {"logging", "is_torch_available"},
    }


def test_only_imports_that_run_at_module_load_are_collected():
    """An import the converter never executes cannot break importing it.

    The backfill absorbs the `ImportError` raised while importing peft's
    converter, so the drift guard must be scoped to the same thing. A
    `TYPE_CHECKING` import or a function-local one is invisible at that moment;
    demanding it in `_PEFT_CONVERSION_SYMBOLS` would redden `daily-fresh-fetch`
    for an annotation. Anything that can run at load, including under a plain
    `if` or a `try`, still counts.
    """
    src = (
        "from typing import TYPE_CHECKING\n"
        "from transformers.core_model_loading import ConversionOps\n"
        "if TYPE_CHECKING:\n"
        "    from transformers.annotations import OnlyForTypes\n"
        "if not TYPE_CHECKING:\n"
        "    from transformers.runtime import AtRuntime\n"
        "try:\n"
        "    from transformers.optional import MaybeThere\n"
        "except ImportError:\n"
        "    from transformers.fallback import Instead\n"
        "def later():\n"
        "    from transformers.lazy import NotAtImport\n"
        "    class Inner:\n"
        "        from transformers.innerclass import NorThis\n"
        "class Holder:\n"
        "    from transformers.classbody import AtClassCreation\n"
    )
    assert _transformers_imports(src) == {
        "transformers.core_model_loading": {"ConversionOps"},
        "transformers.runtime": {"AtRuntime"},
        "transformers.optional": {"MaybeThere"},
        "transformers.fallback": {"Instead"},
        # A module-level class body runs the moment the class is defined, so an import in one can break the very import
        "transformers.classbody": {"AtClassCreation"},
    }


def test_the_package_layout_is_fetched_when_the_module_layout_is_gone(monkeypatch):
    """A packaging-only move upstream must not silently disable this guard.

    `daily-fresh-fetch` installs only pytest, so a 404 on the single module URL
    falls through to an absent peft and the authority check reports a skip. The
    fetch therefore has to know both layouts, the same pair
    `test_peft_pinned_symbols.py` probes.
    """
    import io
    import urllib.error
    import urllib.request

    tried = []

    def fake_urlopen(request, timeout = None):
        tried.append(request.full_url)
        if request.full_url.endswith("transformers_weight_conversion.py"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)
        return io.BytesIO(b"from transformers.core_model_loading import ConversionOps\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    src = _peft_converter_source()
    assert "ConversionOps" in src
    assert len(tried) == 2 and tried[1].endswith("transformers_weight_conversion/__init__.py")


def test_a_package_split_is_followed_more_than_one_level(monkeypatch):
    """Importing the package runs `.core`, which runs whatever IT imports.

    Fetching only the immediate children left a transformers import two levels down
    invisible, so the authority check stayed green while startup could still fail on it.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": "from .core import build\n",
        "transformers_weight_conversion/core.py": (
            "from .ops import apply\nfrom transformers.core_model_loading import ConversionOps\n"
        ),
        "transformers_weight_conversion/ops.py": "from transformers.deep import TwoLevelsDown\n",
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {
        "transformers.core_model_loading": {"ConversionOps"},
        "transformers.deep": {"TwoLevelsDown"},
    }


def test_the_package_walk_survives_a_circular_relative_import(monkeypatch):
    """Two children importing each other must not loop the fetch."""
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": "from .a import x\n",
        "transformers_weight_conversion/a.py": "from .b import y\n",
        "transformers_weight_conversion/b.py": (
            "from .a import x\nfrom transformers.utils import logging\n"
        ),
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {"transformers.utils": {"logging"}}


def test_a_relative_import_under_an_import_time_block_is_followed(monkeypatch):
    """A package can pull its implementation in from inside `try:` or `if not TYPE_CHECKING:`.

    Importing it still runs that, so a child reached only that way has to be read, while a
    typing-only one must not drag in a module that never executes.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from .typing_only import Never\n"
            "try:\n"
            "    from .core import build\n"
            "except ImportError:\n"
            "    from .fallback import build\n"
        ),
        "transformers_weight_conversion/core.py": (
            "from transformers.core_model_loading import ConversionOps\n"
        ),
        "transformers_weight_conversion/fallback.py": "from transformers.legacy import Old\n",
        "transformers_weight_conversion/typing_only.py": (
            "from transformers.never_runs import NotThis\n"
        ),
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {
        "transformers.core_model_loading": {"ConversionOps"},
        "transformers.legacy": {"Old"},
    }


def test_a_nested_package_resolves_relative_imports_from_its_own_path(monkeypatch):
    """`sub/core.py` doing `from .ops import x` means `sub.ops`, not `ops`.

    Resolving every child against the package root fetched the wrong file, so a
    transformers import in the nested implementation was never read.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": "from .sub.core import build\n",
        "transformers_weight_conversion/sub/core.py": "from .ops import apply\n",
        "transformers_weight_conversion/sub/ops.py": ("from transformers.nested import DeepOne\n"),
        # The wrong resolution would land here instead.
        "transformers_weight_conversion/ops.py": ("from transformers.wrong_level import NotThis\n"),
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {
        "transformers.nested": {"DeepOne"},
    }


def test_an_unrecoverable_conversion_map_fails_on_use(caplog):
    """An empty map is the one shape that fails silently.

    peft copies it at import and then calls `.get(model_type, None)`; a None makes
    `_convert_peft_config_moe` return early, so every affected adapter loads with its
    legacy targets unconverted and nothing is logged. Every other runtime symbol here is
    backfilled fail-on-use for that reason.
    """
    stand_in = F._UnavailableConversionPatternMap()

    # peft's own line: `_MODEL_TO_CONVERSION_PATTERN = _MODEL_TO_CONVERSION_PATTERN.copy()`
    copied = stand_in.copy()
    assert isinstance(copied, F._UnavailableConversionPatternMap), "a plain copy is silent again"

    # ...then `_MODEL_TO_CONVERSION_PATTERN["mixtral"] = "mixtral"`, which must still work.
    copied["mixtral"] = "mixtral"
    assert copied.get("mixtral") == "mixtral"
    assert copied["mixtral"] == "mixtral"

    with pytest.raises(RuntimeError):
        copied.get("qwen3_moe", None)
    # A fused-MoE lookup we cannot answer honestly raises where it happens.
    with pytest.raises(RuntimeError):
        copied["qwen3_moe"]

    # But peft reaches _convert_peft_config_moe for ANY model type with a checkpoint conversion mapping, and for a
    assert copied.get("llama", None) is None
    assert copied.get("gemma3", "fallback") == "fallback"
    with pytest.raises(KeyError):
        copied["llama"]


def test_an_unrecoverable_map_is_what_the_backfill_actually_installs(fake_modules, monkeypatch):
    """The stand-in only helps if the backfill installs it instead of an empty dict."""
    monkeypatch.setattr(F, "_recover_conversion_pattern_map", lambda _real: None)
    assert F._backfill_missing_conversion_symbols() is True
    installed = getattr(
        fake_modules["transformers.conversion_mapping"], "_MODEL_TO_CONVERSION_PATTERN"
    )
    assert isinstance(
        installed, F._UnavailableConversionPatternMap
    ), "an empty dict here is the silent mis-conversion this exists to stop"

    # And a real map still comes through as an ordinary dict.
    monkeypatch.setattr(F, "_recover_conversion_pattern_map", lambda _real: {"qwen3_moe": "qwen3"})
    delattr(fake_modules["transformers.conversion_mapping"], "_MODEL_TO_CONVERSION_PATTERN")
    assert F._backfill_missing_conversion_symbols() is True
    recovered = getattr(
        fake_modules["transformers.conversion_mapping"], "_MODEL_TO_CONVERSION_PATTERN"
    )
    assert recovered == {"qwen3_moe": "qwen3"}
    assert not isinstance(recovered, F._UnavailableConversionPatternMap)


def test_a_re_exporting_package_is_followed_to_its_implementation(monkeypatch):
    """A package `__init__.py` that only re-exports imports nothing from transformers.

    Reading it alone reports an empty import set, so the authority check passes while the
    implementation module it pulls in can still break startup on an unlisted symbol. Importing the
    package runs that module, so the guard has to read it.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": (
            "from .core import build_peft_weight_mapping\nfrom . import ops\n"
        ),
        "transformers_weight_conversion/core.py": (
            "from transformers.core_model_loading import ConversionOps\n"
        ),
        "transformers_weight_conversion/ops.py": "from transformers.utils import logging\n",
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {
        "transformers.core_model_loading": {"ConversionOps"},
        "transformers.utils": {"logging"},
    }


def test_the_symbol_list_matches_what_peft_imports():
    """Two lists that can drift; peft's own source is the authority."""
    # Not `importorskip`:
    # Not `importorskip`: on pytest 8 to 9.0 that also swallows an ImportError raised INSIDE an existing converter
    imports = _transformers_imports(_peft_converter_source())
    known = F._PEFT_CONVERSION_SYMBOLS
    for module, imported in imports.items():
        assert module in known, (
            f"peft imports from {module}, which is not in _PEFT_CONVERSION_SYMBOLS, "
            f"so its ImportError is neither recognised nor backfilled"
        )
        assert imported <= set(known[module]), (
            f"{module}: peft imports {imported - set(known[module])}, " f"which we do not backfill"
        )


def test_the_backfill_runs_from_the_guard():
    """Wiring, not behaviour: the guard used to return False here."""
    import inspect

    src = inspect.getsource(F)
    assert "_backfill_missing_conversion_symbols() or patched_any" in src


# --- what the first review round found --------------------------------------
def test_the_model_type_map_is_recovered_not_emptied(fake_modules):
    """peft copies this dict and looks model families up in it, so handing it
    the stub's empty one drops every alias silently: `_convert_peft_config_moe`
    misses the lookup and leaves legacy LoRA targets unconverted, with no
    error. A rename is the likeliest reason for the name to go, so the map is
    found by shape."""
    real = fake_modules["transformers.conversion_mapping"]
    real._RENAMED_CONVERSION_PATTERN = {
        "qwen3_moe": "qwen2_moe",
        "deepseek_v3": "qwen2_moe",
        "minimax": "mixtral",
        "mixtral": "mixtral",
    }

    assert F._backfill_missing_conversion_symbols() is True

    recovered = real._MODEL_TO_CONVERSION_PATTERN
    assert recovered["qwen3_moe"] == "qwen2_moe"
    assert (
        recovered is not real._RENAMED_CONVERSION_PATTERN
    ), "peft mutates its copy; hand it one of ours"


def test_no_recoverable_map_says_so(fake_modules, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert F._backfill_missing_conversion_symbols() is True
    assert fake_modules["transformers.conversion_mapping"]._MODEL_TO_CONVERSION_PATTERN == {}
    assert "fused MoE" in caplog.text


def test_a_non_string_dict_is_not_mistaken_for_the_map(fake_modules):
    real = fake_modules["transformers.conversion_mapping"]
    real._cache = {"llama": object()}
    F._backfill_missing_conversion_symbols()
    assert real._MODEL_TO_CONVERSION_PATTERN == {}


def test_a_called_symbol_refuses_rather_than_answering_wrongly(fake_modules):
    """The stub bodies are inert on purpose: on transformers <5 the whole
    module is ours and peft's converter never runs. On a real transformers it
    does run -- the donor `rename_source_key` takes three arguments and always
    returns the original key, while peft also calls it with a prefix and an
    adapter state dict."""
    F._backfill_missing_conversion_symbols()
    core = fake_modules["transformers.core_model_loading"]
    with pytest.raises(RuntimeError, match = "would silently mis-convert"):
        core.rename_source_key("k", [], [], "prefix", {})


def test_a_class_valued_symbol_stays_a_class(fake_modules):
    """peft runs `isinstance(entry, WeightRenaming)`, which needs a type."""
    F._backfill_missing_conversion_symbols()
    core = fake_modules["transformers.core_model_loading"]
    assert isinstance(core.WeightRenaming, type)
    with pytest.raises(RuntimeError, match = "would silently mis-convert"):
        core.WeightRenaming("a", "b")


def test_a_type_check_against_a_placeholder_raises_rather_than_missing():
    """Answering False is the dangerous answer. peft buckets its conversion
    entries by type -- `isinstance(entry, WeightConverter)`, `isinstance(op,
    Concatenate)` -- so a placeholder that quietly matches nothing drops the
    operations and converts the adapter wrongly with no error."""
    placeholder = F._unsupported_conversion_symbol(
        "transformers.core_model_loading.WeightConverter", donor_value = type
    )
    with pytest.raises(RuntimeError, match = "would silently mis-convert"):
        isinstance(object(), placeholder)


def test_a_placeholder_can_still_be_subclassed():
    """peft does `class PeftConcatenate(Concatenate)` at module top, so class
    creation has to work even though construction refuses."""
    placeholder = F._unsupported_conversion_symbol(
        "transformers.core_model_loading.Concatenate", donor_value = type
    )

    class Mine(placeholder):
        pass

    assert issubclass(Mine, placeholder)


def test_the_import_only_symbols_still_come_from_the_stub(fake_modules):
    """`ConversionOps` is subclassed by peft at module top and never asked
    about instances, so it stays a real usable class."""
    F._backfill_missing_conversion_symbols()
    core = fake_modules["transformers.core_model_loading"]

    class Mine(core.ConversionOps):
        pass

    assert issubclass(Mine, core.ConversionOps)


def test_the_operation_classes_are_treated_as_runtime(fake_modules):
    """`build_peft_weight_mapping` isinstance-checks Concatenate and
    MergeModulelist and builds `Transpose(dim0 = 0, dim1 = 1)`, so an inert stub
    silently skips the conversion instead of doing it."""
    F._backfill_missing_conversion_symbols()
    core = fake_modules["transformers.core_model_loading"]
    for name in ("Concatenate", "MergeModulelist", "Transpose"):
        with pytest.raises(RuntimeError, match = "would silently mis-convert"):
            getattr(core, name)(dim = 1)


def test_every_runtime_symbol_is_one_we_backfill():
    """The two tables can drift; a name in neither is the silent case."""
    known = {f"{m}.{s}" for m, syms in F._PEFT_CONVERSION_SYMBOLS.items() for s in syms}
    assert F._PEFT_CONVERSION_RUNTIME_SYMBOLS <= known


def test_a_class_body_import_is_seen():
    """`ast.walk` descends into everything, so an import inside a module-level
    class body -- which executes at import time -- is covered."""
    src = "class Converter:\n    from transformers.core_model_loading import Concatenate\n"
    assert _transformers_imports(src) == {"transformers.core_model_loading": {"Concatenate"}}


def test_a_package_split_still_reads_the_implementation(tmp_path, monkeypatch):
    """A converter package whose `__init__.py` only re-exports must not hide
    the child module's transformers imports, which are the ones that fail."""
    import importlib
    import sys
    import types

    pkg = types.ModuleType("zz_conv_pkg")
    pkg.__path__ = [str(tmp_path)]
    (tmp_path / "__init__.py").write_text("from .impl import Converter\n")
    (tmp_path / "impl.py").write_text(
        "from transformers.core_model_loading import Concatenate\nclass Converter: pass\n"
    )
    child = types.ModuleType("zz_conv_pkg.impl")
    monkeypatch.setitem(sys.modules, "zz_conv_pkg", pkg)
    monkeypatch.setitem(sys.modules, "zz_conv_pkg.impl", child)

    import inspect
    import pkgutil

    sources = []
    for path in pkg.__path__:
        for info in pkgutil.iter_modules([path]):
            sources.append((tmp_path / f"{info.name}.py").read_text())
    joined = "\n".join(sources)
    assert _transformers_imports(joined) == {
        "transformers.core_model_loading": {"Concatenate"},
    }, "the child module's imports were not read"


def test_the_fetcher_walks_a_package():
    """Wiring: the fallback must iterate `__path__`, not stop at `__init__`."""
    import inspect

    src = inspect.getsource(_peft_converter_source)
    assert "__path__" in src and "iter_modules" in src


# --- what the second review round found -------------------------------------
def test_a_fused_moe_type_that_does_not_say_moe_still_refuses():
    """Eleven of the twenty-four fused MoE model types are not named for it.

    `deepseek_v3`, `dots1`, `longcat_flash`, `minimax`, `mellum`, `qwen3_next`,
    `solar_open` and `flex_olmo` all map to `mixtral` or `qwen2_moe`, which are
    the only two base patterns peft rewrites for. A substring test over the name
    answered the default for exactly the checkpoints the stand-in exists to
    protect.
    """
    stand_in = F._UnavailableConversionPatternMap().copy()
    stand_in["mixtral"] = "mixtral"

    for model_type in (
        "deepseek_v3",
        "dots1",
        "longcat_flash",
        "minimax",
        "minimax_m2",
        "mellum",
        "qwen3_next",
        "solar_open",
        "flex_olmo",
        "deepseek_v2",
        "deepseek_v32",
    ):
        assert "moe" not in model_type, f"{model_type} would pass on the name alone"
        with pytest.raises(RuntimeError):
            stand_in.get(model_type, None)
        with pytest.raises(RuntimeError):
            stand_in[model_type]

    with pytest.raises(RuntimeError):
        stand_in.get("some_new_moe", None)
    # And a non-MoE type is still answered with the default.
    assert stand_in.get("llama", None) is None


def test_the_moe_snapshot_matches_the_installed_transformers():
    """The snapshot only helps while it agrees with the map it stands in for."""
    try:
        from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN as real
    except Exception:
        pytest.skip("this transformers ships no conversion map to compare against")

    live = {k: v for k, v in real.items() if v in ("mixtral", "qwen2_moe")}
    missing = {k: v for k, v in live.items() if F._PEFT_MOE_CONVERSION_PATTERNS.get(k) != v}
    assert not missing, f"fused MoE model types missing from the snapshot: {sorted(missing)}"


def test_an_unrelated_string_dictionary_is_not_installed_as_the_map(fake_modules):
    """Shape alone selects the biggest `dict[str, str]`, not the right one.

    A module that renamed the conversion map is just as likely to carry an alias
    table or a doc map, and installing that maps a coincidentally matching model
    type to the wrong conversion family -- and bypasses the stand-in, so every
    other MoE lookup goes back to a silent None.
    """
    real = fake_modules["transformers.conversion_mapping"]
    real._DOC_ALIASES = {f"key_{i}": f"value_{i}" for i in range(50)}
    real._RENAMED = {
        "qwen3_moe": "qwen2_moe",
        "deepseek_v3": "qwen2_moe",
        "minimax": "mixtral",
    }

    assert F._backfill_missing_conversion_symbols() is True
    recovered = real._MODEL_TO_CONVERSION_PATTERN
    assert recovered["deepseek_v3"] == "qwen2_moe", "the larger unrelated dict won"
    assert "key_0" not in recovered


def test_only_an_unrelated_dictionary_leaves_the_map_unrecovered(fake_modules):
    """Nothing convincing means nothing recovered, not something wrong."""
    real = fake_modules["transformers.conversion_mapping"]
    real._DOC_ALIASES = {f"key_{i}": f"value_{i}" for i in range(50)}

    assert F._backfill_missing_conversion_symbols() is True
    installed = real._MODEL_TO_CONVERSION_PATTERN
    assert isinstance(installed, F._UnavailableConversionPatternMap)
    # The naming convention is still honoured on top, for a type added later.
    with pytest.raises(RuntimeError):
        installed.get("deepseek_v3", None)


def test_a_parent_relative_import_resolves_above_its_own_package(monkeypatch):
    """`from ..ops import x` in `sub/core.py` means `ops`, not `sub.ops`.

    Prefixing the current package onto every name regardless probed a path that
    is not there, read nothing, and left the transformers import in the real
    target out of the drift check -- a check that then cannot fail.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": "from .sub.core import build\n",
        "transformers_weight_conversion/sub/core.py": "from ..ops import apply\n",
        "transformers_weight_conversion/ops.py": "from transformers.parent import UpOne\n",
        # Where the level-blind resolution would look instead.
        "transformers_weight_conversion/sub/ops.py": (
            "from transformers.wrong_level import NotThis\n"
        ),
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {"transformers.parent": {"UpOne"}}


def test_a_relative_import_above_the_package_root_is_dropped(monkeypatch):
    """`from ...elsewhere import x` leaves the converter package entirely.

    There is nothing under the fetch root to read, and guessing a path produces
    a 404 per level. The walk drops it rather than probing.
    """
    import io
    import urllib.error
    import urllib.request

    pages = {
        "transformers_weight_conversion/__init__.py": (
            "from ...elsewhere import gone\nfrom .core import build\n"
        ),
        "transformers_weight_conversion/core.py": "from transformers.here import Kept\n",
    }

    def fake_urlopen(request, timeout = None):
        for suffix, body in pages.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _transformers_imports(_peft_converter_source()) == {"transformers.here": {"Kept"}}


def test_a_moe_named_model_that_is_not_fused_still_loads():
    """`_convert_peft_config_moe` is keyed on `mixtral` and `qwen2_moe` alone and
    returns without a rewrite for anything else, so refusing on the NAME turned
    three working adapter loads into hard errors. All three ship today:
    `qwen3_5_moe_text` converts as `qwen3_5_text`, and both Granite MoE variants
    as `granitemoe`."""
    stand_in = F._UnavailableConversionPatternMap()
    for model_type in ("qwen3_5_moe_text", "granitemoehybrid", "granitemoeshared"):
        assert stand_in.get(model_type) is None, model_type
        assert stand_in.get(model_type, "fallback") == "fallback", model_type


def test_a_fused_moe_model_is_still_refused():
    """The carve-out must not reach the types that really do need the rewrite."""
    stand_in = F._UnavailableConversionPatternMap()
    for model_type in ("qwen3_moe", "minimax", "deepseek_v3", "olmoe"):
        with pytest.raises(RuntimeError, match = "conversion map"):
            stand_in.get(model_type)


def test_an_unknown_moe_name_is_still_refused():
    """The hint stays for a fused type added after the snapshot: silently
    returning the default there mis-converts the adapter."""
    stand_in = F._UnavailableConversionPatternMap()
    with pytest.raises(RuntimeError):
        stand_in.get("some_future_moe_model")


def test_the_not_fused_list_matches_upstream():
    """Derived from the real map, so it cannot drift into guesswork."""
    src = _fetch_conversion_mapping_source()
    if src is None:
        pytest.skip("upstream conversion map unavailable")
    import re

    pairs = dict(re.findall(r'"([\w.]+)":\s*"([\w.]+)"', src))
    named = {k: v for k, v in pairs.items() if "moe" in k.lower() or "mixtral" in k.lower()}
    if not named:
        pytest.skip("upstream map shape changed")
    not_fused = {k for k, v in named.items() if v not in ("mixtral", "qwen2_moe")}
    assert not_fused == set(F._PEFT_MOE_NAMED_NOT_FUSED), (
        f"upstream MoE-named non-fused types moved: "
        f"{not_fused ^ set(F._PEFT_MOE_NAMED_NOT_FUSED)}"
    )


def _fetch_conversion_mapping_source():
    """The upstream map, over the network, so the canary runs in the job that
    installs only pytest -- where `transformers` is absent and every comparison
    against the installed copy silently skips."""
    import os
    import urllib.error
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/huggingface/transformers/main/"
        "src/transformers/conversion_mapping.py"
    )
    request = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout = 15) as response:
            return response.read().decode("utf-8", errors = "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def test_the_fused_snapshot_matches_upstream():
    """The other half of the same canary: a fused MoE alias added upstream and
    missing here falls through the stand-in silently if its name says nothing
    about MoE. Runs off the network, so the dependency-free job exercises it."""
    src = _fetch_conversion_mapping_source()
    if src is None:
        pytest.skip("upstream conversion map unavailable")
    import re

    pairs = dict(re.findall(r'"([\w.]+)":\s*"([\w.]+)"', src))
    fused = {k for k, v in pairs.items() if v in ("mixtral", "qwen2_moe")}
    if not fused:
        pytest.skip("upstream map shape changed")
    missing = fused - set(F._PEFT_MOE_CONVERSION_PATTERNS)
    assert not missing, f"fused MoE types missing from the snapshot: {sorted(missing)}"


def test_a_fromlist_child_module_is_queued_too():
    """`from .sub import core` imports `pkg.sub.core` when `core` is a module.

    Queueing only `sub` fetched the package shim, whose `__init__.py` may
    re-export nothing, so the transformers imports in the file that actually has
    them were never read and the drift check could not fail.
    """
    targets = _relative_import_targets("from .sub import core, HELPER\n")
    assert (1, "sub") in targets, "the package itself is still followed"
    assert (1, "sub.core") in targets, "the child module was never queued"
    assert (1, "sub.HELPER") in targets, "a symbol costs one fetch that finds nothing"


def test_a_star_import_queues_only_the_package():
    """`from .sub import *` names no child, so there is nothing extra to fetch."""
    assert _relative_import_targets("from .sub import *\n") == [(1, "sub")]


def test_a_module_form_transformers_import_is_a_dependency():
    """`import transformers.x` raises the same startup ModuleNotFoundError as the
    `from` form, so skipping every `ast.Import` let a newly required submodule
    break the converter with this check still green. It binds no symbols."""
    imports = _transformers_imports(
        "import transformers.core_model_loading\nimport os\nimport transformers.utils as tu\n"
    )
    assert imports == {"transformers.core_model_loading": set(), "transformers.utils": set()}


def test_a_module_form_import_inside_a_function_is_still_ignored():
    """The control: it never runs at import time, so it cannot break startup."""
    assert _transformers_imports("def f():\n    import transformers.core_model_loading\n") == {}


def test_an_absolute_import_of_a_child_is_queued():
    """A package can re-export its implementation absolutely.

    `from peft.utils.transformers_weight_conversion.core import build` loads the
    same file the relative form would, and the relative-only branch queued
    nothing, so the fetcher read `__init__.py` and none of the transformers
    imports in `core.py`. Level 0 marks a name already rooted at the package.
    """
    src = f"from {_CONVERTER_MODULE}.core import build\n"
    assert (0, "core") in _relative_import_targets(src)
    assert _resolved_relative_targets("", src) == ["core", "core.build"]


def test_an_absolute_import_from_the_package_root_is_queued():
    """`from peft.utils.transformers_weight_conversion import core` names the
    child directly."""
    src = f"from {_CONVERTER_MODULE} import core\n"
    assert _resolved_relative_targets("", src) == ["core"]


def test_a_module_form_absolute_child_import_is_queued():
    """`import peft.utils.transformers_weight_conversion.core` loads it too."""
    src = f"import {_CONVERTER_MODULE}.core\n"
    assert _resolved_relative_targets("", src) == ["core"]


def test_an_unrelated_absolute_import_is_not_queued():
    """The control: only the converter's own children are followed."""
    assert _relative_import_targets("from peft.tuners.lora import Linear\n") == []


def test_an_import_in_a_module_level_match_case_is_collected():
    """The selected case runs at import time, so an import inside one can raise
    the startup error this guard exists to catch. `ast.Match` keeps its
    statements under `cases`, which the walk did not descend into."""
    src = (
        "import sys\n"
        "match sys.version_info[0]:\n"
        "    case 3:\n"
        "        from transformers.core_model_loading import Loader\n"
        "    case _:\n"
        "        from transformers.utils import logging\n"
    )
    assert _transformers_imports(src) == {
        "transformers.core_model_loading": {"Loader"},
        "transformers.utils": {"logging"},
    }


def test_an_import_in_a_match_case_inside_a_function_is_still_ignored():
    """The control: a function body never runs at import time."""
    assert (
        _transformers_imports(
            "def f(x):\n"
            "    match x:\n"
            "        case 1:\n"
            "            from transformers.utils import logging\n"
        )
        == {}
    )


def test_the_two_base_patterns_are_in_the_snapshot():
    """`mixtral` and `qwen2_moe` map to themselves and were left out.

    Not harmless: the substring hint answers False for `mixtral`, which says
    nothing about MoE, so the stand-in handed back the silent default for a type
    peft really does rewrite. It also failed the live comparison outright on
    transformers 5.5.0, which pyproject permits.
    """
    for name in ("mixtral", "qwen2_moe"):
        assert F._PEFT_MOE_CONVERSION_PATTERNS.get(name) == name
    stand_in = F._UnavailableConversionPatternMap()
    assert stand_in._is_moe("mixtral"), "the hint alone never saw this one"
    assert stand_in._is_moe("qwen2_moe")


@pytest.mark.parametrize("model_type", sorted(F._PEFT_MOE_NAMED_NOT_CONVERTED))
def test_a_moe_named_type_outside_the_map_is_not_refused(model_type):
    """peft answers `None` for a type the map does not hold and skips the target
    rewrite, so refusing breaks an ordinary adapter load. `qwen3_vl_moe` and
    `lfm2_moe` are both supported models the substring hint was catching."""
    stand_in = F._UnavailableConversionPatternMap()
    assert stand_in.get(model_type) is None
    assert stand_in.get(model_type, "fallback") == "fallback"


def test_the_unconverted_list_is_still_outside_the_upstream_map():
    """The canary for the other half. A name that gains a conversion pattern
    upstream becomes fused, and answering `None` for it is the silent
    mis-conversion the refusal exists to prevent."""
    src = _fetch_conversion_mapping_source()
    if src is None:
        pytest.skip("upstream conversion map unavailable")
    import re

    keys = set(re.findall(r'"([\w.]+)":\s*"[\w.]+"', src))
    if not keys:
        pytest.skip("upstream map shape changed")
    gained = keys & set(F._PEFT_MOE_NAMED_NOT_CONVERTED)
    assert not gained, f"now in the upstream map, so no longer safe to skip: {sorted(gained)}"


def test_the_moe_aware_map_wins_over_the_inert_donor():
    """Two backfills cover the same two submodules and both are `hasattr`-gated,
    so whichever runs first decides what `_MODEL_TO_CONVERSION_PATTERN` IS. The
    general pass donates an inert `{}`, which answers every lookup with a silent
    None; this one installs a map that refuses a fused-MoE lookup it cannot
    answer. Order is the only thing keeping a fused MoE adapter from loading
    with its LoRA targets unconverted and no error."""
    src = inspect.getsource(F.fix_peft_transformers_weight_conversion_import)
    mine = src.index("_backfill_missing_conversion_symbols()")
    general = src.index("_backfill_missing_peft_symbols(_submodule)")
    assert mine < general, "the inert donor now runs first and wins the symbol"


def test_the_general_pass_does_not_overwrite_an_installed_stand_in():
    """Even in the right order it must stay additive: a second pass that
    re-donated would undo the first."""
    donor = F._PEFT_STUB_BUILDERS["transformers.conversion_mapping"]()
    assert isinstance(getattr(donor, "_MODEL_TO_CONVERSION_PATTERN", None), dict)
    src = inspect.getsource(F._backfill_missing_peft_symbols)
    assert "not hasattr(mod, s)" in src, "the general pass no longer gates on absence"
