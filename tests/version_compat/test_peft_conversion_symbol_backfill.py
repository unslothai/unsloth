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


def test_the_symbol_list_matches_what_peft_imports():
    """Two lists that can drift; peft's own source is the authority."""
    # Not `importorskip`: on pytest 8 to 9.0 that also swallows an ImportError
    # raised INSIDE an existing converter, which is exactly the drift this test
    # exists to catch -- peft adding an import we do not backfill would report
    # as a skip. Skip only when the module is genuinely absent.
    module = "peft.utils.transformers_weight_conversion"
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed")
    try:
        peft = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "") in (module, "peft.utils", "peft"):
            pytest.skip("this peft has no weight converter")
        raise
    import inspect
    import re

    src = inspect.getsource(peft)
    for name, symbols in F._PEFT_CONVERSION_SYMBOLS.items():
        block = re.search(rf"from {re.escape(name)} import \(([^)]*)\)", src)
        if block is None:
            continue
        imported = {s.strip().rstrip(",") for s in block.group(1).split() if s.strip().rstrip(",")}
        assert imported <= set(
            symbols
        ), f"{name}: peft imports {imported - set(symbols)}, which we do not backfill"


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
    assert isinstance(object(), core.WeightRenaming) is False
    with pytest.raises(RuntimeError, match = "would silently mis-convert"):
        core.WeightRenaming("a", "b")


def test_the_import_only_symbols_still_come_from_the_stub(fake_modules):
    """`Concatenate`/`ConversionOps` are subclassed by peft at module top, so
    they must be real usable classes, not refusals."""
    F._backfill_missing_conversion_symbols()
    core = fake_modules["transformers.core_model_loading"]

    class Mine(core.ConversionOps):
        pass

    assert core.Concatenate(dim = 1).dim == 1
    assert issubclass(Mine, core.ConversionOps)


def test_every_runtime_symbol_is_one_we_backfill():
    """The two tables can drift; a name in neither is the silent case."""
    known = {f"{m}.{s}" for m, syms in F._PEFT_CONVERSION_SYMBOLS.items() for s in syms}
    assert F._PEFT_CONVERSION_RUNTIME_SYMBOLS <= known
