# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/_shared/torch_config_guard.py: a deleted dynamo / inductor config entry is found, named and restored."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
import torch._dynamo.config  # noqa: E402
import torch._inductor.config  # noqa: E402

import torch_config_guard as guard  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_LIMIT = (
    "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
)


def test_a_deleted_entry_is_found_named_and_restored(monkeypatch):
    assert guard.install_delete_recorder()
    monkeypatch.setattr(torch._dynamo.config, _LIMIT, 37)
    delattr(torch._dynamo.config, _LIMIT)
    with pytest.raises(AttributeError, match = "does not exist"):
        getattr(torch._dynamo.config, _LIMIT)

    hidden = guard.hidden_entries()
    assert ("torch._dynamo.config", _LIMIT) in hidden
    report = guard.restore(hidden)

    assert f"torch._dynamo.config.{_LIMIT} was deleted" in report
    assert "test_a_deleted_entry_is_found_named_and_restored" in report
    assert getattr(torch._dynamo.config, _LIMIT) == 37
    assert guard.hidden_entries() == []


def test_check_after_test_raises_and_leaves_the_process_clean():
    guard.install_delete_recorder()
    before = torch._inductor.config.emulate_precision_casts
    delattr(torch._inductor.config, "emulate_precision_casts")
    with pytest.raises(AssertionError, match = "emulate_precision_casts was deleted"):
        guard.check_after_test()
    assert torch._inductor.config.emulate_precision_casts == before
    guard.check_after_test()


def test_mock_patch_round_trip_is_not_flagged():
    # mock.patch deletes on exit and re-creates right after; only a delete left in place is a leak.
    with mock.patch.object(torch._dynamo.config, _LIMIT, 5):
        assert getattr(torch._dynamo.config, _LIMIT) == 5
    assert guard.hidden_entries() == []


@pytest.mark.parametrize("conftest", ["tests/conftest.py", "studio/backend/tests/conftest.py"])
def test_both_suites_run_the_guard_after_every_test(conftest):
    tree = ast.parse((REPO / conftest).read_text(encoding = "utf-8"))
    fixture = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_no_deleted_torch_config_entries"
        ),
        None,
    )
    assert fixture is not None, f"{conftest} lost the torch config guard fixture"
    decorator = ast.unparse(fixture.decorator_list[0])
    assert "autouse = True" in decorator.replace("autouse=True", "autouse = True")
    calls = [ast.unparse(n.func) for n in ast.walk(fixture) if isinstance(n, ast.Call)]
    assert "check_after_test" in calls and "install_delete_recorder" in calls
    assert any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Yield) for n in fixture.body)


@pytest.mark.parametrize(
    "target, name",
    [
        ("torch._dynamo.config", "suppress_errors"),
        ("torch._inductor.config", "emulate_precision_casts"),
        ("torch._inductor.config.triton", "cudagraphs"),
    ],
)
def test_monkeypatch_undo_restores_a_config_entry_instead_of_deleting_it(target, name):
    # pytest 9.2 records a non-class target's old value from its __dict__, which a config entry is never in.
    owner = torch
    for part in target.split(".")[1:]:
        owner = getattr(owner, part)
    before = getattr(owner, name)
    mp = pytest.MonkeyPatch()
    mp.setattr(owner, name, not before)
    assert getattr(owner, name) is (not before)
    mp.undo()
    assert guard.hidden_entries() == []
    assert getattr(owner, name) == before


def test_monkeypatch_undo_restores_a_config_entry_named_by_import_path():
    before = torch._dynamo.config.suppress_errors
    mp = pytest.MonkeyPatch()
    mp.setattr("torch._dynamo.config.suppress_errors", not before)
    mp.undo()
    assert guard.hidden_entries() == []
    assert torch._dynamo.config.suppress_errors == before


def test_monkeypatch_on_an_ordinary_object_is_unchanged():
    class Target:
        pass

    target = Target()
    mp = pytest.MonkeyPatch()
    mp.setattr(target, "flag", True, raising = False)
    mp.undo()
    assert not hasattr(target, "flag")
