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

import sys
import types

import pytest

from unsloth import import_fixes as F


@pytest.fixture()
def fake_modules():
    """Two real-looking modules missing exactly the drifted names."""
    saved = {n: sys.modules.get(n) for n in F._PEFT_CONVERSION_SYMBOLS}
    for name in F._PEFT_CONVERSION_SYMBOLS:
        mod = types.ModuleType(name)
        mod.__file__ = f"<fake {name}>"
        mod.something_else = object()      # proves the module is not replaced
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
        assert getattr(fake_modules[name], symbol) is value, \
            "an existing symbol was overwritten"


def test_it_is_idempotent(fake_modules):
    assert F._backfill_missing_conversion_symbols() is True
    assert F._backfill_missing_conversion_symbols() is False


def test_the_symbol_list_matches_what_peft_imports():
    """Two lists that can drift; peft's own source is the authority."""
    peft = pytest.importorskip("peft.utils.transformers_weight_conversion",
                               reason = "this peft has no weight converter")
    import inspect
    import re
    src = inspect.getsource(peft)
    for name, symbols in F._PEFT_CONVERSION_SYMBOLS.items():
        block = re.search(rf"from {re.escape(name)} import \(([^)]*)\)", src)
        if block is None:
            continue
        imported = {s.strip().rstrip(",") for s in block.group(1).split()
                    if s.strip().rstrip(",")}
        assert imported <= set(symbols), \
            f"{name}: peft imports {imported - set(symbols)}, which we do not backfill"


def test_the_backfill_runs_from_the_guard():
    """Wiring, not behaviour: the guard used to return False here."""
    import inspect
    src = inspect.getsource(F)
    assert "_backfill_missing_conversion_symbols() or patched_any" in src
