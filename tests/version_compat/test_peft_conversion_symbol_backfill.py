# Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

import importlib.util
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


def _peft_converter_source():
    """peft's own source, from the installed package or from GitHub.

    `version-compat-ci.yml`'s `daily-fresh-fetch` job installs only pytest, so
    `find_spec("peft")` is None there and this authority check reported as a
    skip: a new import added upstream would never fail the guard it exists to
    be. The pinned-symbol suites in this directory already read upstream over
    the network, so do the same and fall back to the installed copy.
    """
    import os
    import urllib.error
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/huggingface/peft/main/"
        "src/peft/utils/transformers_weight_conversion.py"
    )
    request = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout = 15) as response:
            return response.read().decode("utf-8", errors = "replace")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            pytest.skip(f"GitHub fetch failed ({exc.code}) for {url}")
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"GitHub fetch failed ({exc}) for {url}")

    module = "peft.utils.transformers_weight_conversion"
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed and upstream could not be read")
    try:
        import inspect
        return inspect.getsource(importlib.import_module(module))
    except ModuleNotFoundError as exc:
        if (exc.name or "") in (module, "peft.utils", "peft"):
            pytest.skip("this peft has no weight converter")
        raise


def _transformers_imports(src):
    """{module: {symbol}} for every `from transformers... import ...` in `src`.

    AST, not a regex over the source. A regex for the parenthesised form alone
    read `from transformers.core_model_loading import A, B` as importing
    nothing, and looking up only the modules already in the table never
    examined a third transformers module at all. Either one leaves the drift
    this file exists to catch reporting green.
    """
    import ast

    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers"):
            out.setdefault(node.module, set()).update(a.name for a in node.names)
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


def test_the_symbol_list_matches_what_peft_imports():
    """Two lists that can drift; peft's own source is the authority."""
    # Not `importorskip`: on pytest 8 to 9.0 that also swallows an ImportError
    # raised INSIDE an existing converter, which is exactly the drift this test
    # exists to catch -- peft adding an import we do not backfill would report
    # as a skip. Skip only when the module is genuinely absent.
    imports = _transformers_imports(_peft_converter_source())
    known = F._PEFT_CONVERSION_SYMBOLS
    for module, imported in imports.items():
        assert module in known, (
            f"peft imports from {module}, which is not in _PEFT_CONVERSION_SYMBOLS, "
            f"so its ImportError is neither recognised nor backfilled"
        )
        assert imported <= set(known[module]), (
            f"{module}: peft imports {imported - set(known[module])}, "
            f"which we do not backfill"
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
    real._RENAMED_CONVERSION_PATTERN = {"qwen3_moe": "qwen2_moe", "mixtral": "mixtral"}

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
