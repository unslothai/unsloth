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
import importlib.util
import pathlib

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


def test_raise_then_restore_returns_the_callers_thresholds(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    assert gc.get_threshold() == (import_gc.IMPORT_THRESHOLD, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (700, 11, 12)
    # Idempotent: a second restore has nothing to undo.
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (700, 11, 12)


def test_a_failed_import_retry_still_restores_the_first_thresholds(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    # The first `import unsloth` raised before its restore; the retry raises again.
    import_gc.raise_gc_threshold()
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (700, 11, 12)


def test_a_threshold_changed_after_a_failed_import_is_the_new_original(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    gc.set_threshold(900, 11, 12)
    import_gc.raise_gc_threshold()
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (900, 11, 12)


@pytest.mark.parametrize("threshold", [0, 200_000])
def test_disabled_or_higher_thresholds_are_left_alone(import_gc, threshold):
    gc.set_threshold(threshold, 11, 12)
    import_gc.raise_gc_threshold()
    assert gc.get_threshold() == (threshold, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (threshold, 11, 12)


def test_a_threshold_set_during_the_import_is_kept(import_gc):
    gc.set_threshold(700, 11, 12)
    import_gc.raise_gc_threshold()
    gc.set_threshold(5000, 11, 12)
    import_gc.restore_gc_threshold()
    assert gc.get_threshold() == (5000, 11, 12)


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
