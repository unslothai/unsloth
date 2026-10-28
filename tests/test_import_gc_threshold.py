# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`import unsloth` raises the GC threshold for its own imports and hands the caller's back."""

import ast
import gc
import importlib
import importlib.machinery
import importlib.util
import pathlib
import sys
import types

import pytest

_UNSLOTH = pathlib.Path(__file__).resolve().parents[1] / "unsloth"


@pytest.fixture
def import_gc():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_gc_under_test", _UNSLOTH / "_import_gc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = gc.get_threshold()
    try:
        yield module
    finally:
        gc.set_threshold(*original)
        sys.meta_path[:] = [
            f for f in sys.meta_path if not isinstance(f, module._RestoreAfterFailedImport)
        ]


def _stored(*thresholds):
    """What gc.get_threshold() reports for these: Python 3.14's incremental GC keeps no threshold2."""
    before = gc.get_threshold()
    gc.set_threshold(*thresholds)
    try:
        return gc.get_threshold()
    finally:
        gc.set_threshold(*before)


def test_raise_then_restore_returns_the_callers_thresholds(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    assert gc.get_threshold() == _stored(import_gc.IMPORT_THRESHOLD, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(700, 11, 12)
    # Idempotent: a second restore has nothing to undo.
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(700, 11, 12)


def test_a_failed_import_retry_still_restores_the_first_thresholds(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    # The first `import unsloth` raised before its restore; the retry raises again.
    import_gc.raise_gc_threshold()
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(700, 11, 12)


def test_a_threshold_changed_after_a_failed_import_is_the_new_original(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    gc.set_threshold(900, 11, 12)
    import_gc.raise_gc_threshold()
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(900, 11, 12)


@pytest.mark.parametrize("threshold", [0, 200_000])
def test_disabled_or_higher_thresholds_are_left_alone(import_gc, threshold):
    gc.set_threshold(threshold, 11, 12)
    import_gc.raise_gc_threshold()
    assert gc.get_threshold() == _stored(threshold, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(threshold, 11, 12)


def test_a_threshold_set_during_the_import_is_kept(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    gc.set_threshold(5000, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(5000, 11, 12)


def test_generations_1_and_2_changed_during_the_import_are_kept(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    gc.set_threshold(import_gc.IMPORT_THRESHOLD, 20, 30)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == _stored(700, 20, 30)


def _in_flight(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def test_a_failed_import_is_restored_by_the_next_import(import_gc):
    gc.set_threshold(700, 11, 12)
    module = _in_flight("_unsloth_gc_probe_pkg")
    import_gc.raise_gc_threshold(module)
    # Still in flight: imports during it keep the raised threshold.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("_unsloth_gc_missing_one")
    assert gc.get_threshold()[0] == import_gc.IMPORT_THRESHOLD
    # The import raised: the package left sys.modules; the next import gives the threshold back.
    del sys.modules["_unsloth_gc_probe_pkg"]
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("_unsloth_gc_missing_two")
    assert gc.get_threshold() == _stored(700, 11, 12)
    # Inert until the next raise / restore removes it outside finder iteration.
    assert import_gc._hook.module is None


def test_the_restoring_import_still_reaches_the_next_finder(import_gc):
    # sys.meta_path is iterated live: a hook removing itself would skip BuiltinImporter here.
    gc.set_threshold(700, 11, 12)
    module = _in_flight("_unsloth_gc_probe_builtin")
    import_gc.raise_gc_threshold(module)
    hook_at = sys.meta_path.index(import_gc._hook)
    builtin_at = sys.meta_path.index(importlib.machinery.BuiltinImporter)
    sys.meta_path.insert(hook_at + 1, sys.meta_path.pop(builtin_at))
    del sys.modules["_unsloth_gc_probe_builtin"]
    saved = sys.modules.pop("_string", None)
    try:
        importlib.import_module("_string")
    finally:
        if saved is not None:
            sys.modules["_string"] = saved
        sys.meta_path.insert(builtin_at, sys.meta_path.pop(hook_at + 1))
    assert gc.get_threshold() == _stored(700, 11, 12)


def test_a_retry_after_a_failed_import_rearms_and_then_removes_the_hook(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold(_in_flight("_unsloth_gc_probe_retry"))
    del sys.modules["_unsloth_gc_probe_retry"]
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("_unsloth_gc_missing_three")
    module = _in_flight("_unsloth_gc_probe_retry")
    try:
        import_gc.raise_gc_threshold(module)
        assert gc.get_threshold()[0] == import_gc.IMPORT_THRESHOLD
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("_unsloth_gc_missing_four")
        assert gc.get_threshold()[0] == import_gc.IMPORT_THRESHOLD
        import_gc.restore_gc_threshold()
    finally:
        sys.modules.pop("_unsloth_gc_probe_retry", None)
    assert gc.get_threshold() == _stored(700, 11, 12)
    assert not any(isinstance(f, import_gc._RestoreAfterFailedImport) for f in sys.meta_path)


def test_a_successful_import_removes_the_hook(import_gc):
    gc.set_threshold(700, 11, 12)
    module = _in_flight("_unsloth_gc_probe_ok")
    try:
        import_gc.raise_gc_threshold(module)
        assert any(isinstance(f, import_gc._RestoreAfterFailedImport) for f in sys.meta_path)
        import_gc.restore_gc_threshold()
    finally:
        sys.modules.pop("_unsloth_gc_probe_ok", None)
    assert gc.get_threshold() == _stored(700, 11, 12)
    assert not any(isinstance(f, import_gc._RestoreAfterFailedImport) for f in sys.meta_path)


def test_init_raises_first_and_restores_last():
    body = ast.parse((_UNSLOTH / "__init__.py").read_text(encoding = "utf-8")).body
    calls = [
        node.value.func.id
        for node in body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert calls[0] == "_raise_gc_threshold"
    assert calls[-1] == "_restore_gc_threshold"
    assert isinstance(body[-1], ast.Delete) and body[-1].targets[0].id == "_restore_gc_threshold"
