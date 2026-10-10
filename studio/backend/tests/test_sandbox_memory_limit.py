# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox memory limit: env first, then the owner's saved value, then 8 GiB."""

from pathlib import Path
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from storage import studio_db
from utils import sandbox_memory_limit as limit
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
GIB = 1024**3


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(limit.MEMORY_LIMIT_ENV, raising = False)


def test_default_is_eight_gib():
    assert limit.saved_memory_limit_gb() == 8
    assert limit.memory_limit_bytes() == 8 * GIB


def test_a_saved_value_applies_and_saving_again_is_a_no_op():
    for _ in range(2):
        assert limit.set_memory_limit_gb(32) == 32
    assert limit.memory_limit_bytes() == 32 * GIB


@pytest.mark.parametrize(
    "value, expected",
    [("16", 16 * GIB), ("abc", None), ("16.5", None)],
)
def test_a_present_environment_variable_decides_as_before(monkeypatch, value, expected):
    limit.set_memory_limit_gb(32)
    monkeypatch.setenv(limit.MEMORY_LIMIT_ENV, value)
    assert limit.locked_by_environment() is True
    assert limit.memory_limit_bytes() == expected


@pytest.mark.parametrize("value", [0, 4097, True, None, "x"])
def test_setter_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        limit.set_memory_limit_gb(value)


def test_an_unreadable_store_keeps_the_default(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", boom)
    assert limit.memory_limit_bytes() == 8 * GIB


def test_a_managed_account_reads_the_owner_value():
    limit.set_memory_limit_gb(24)
    token = bind_account(ALICE)
    try:
        assert limit.memory_limit_bytes() == 24 * GIB
    finally:
        reset_account(token)
