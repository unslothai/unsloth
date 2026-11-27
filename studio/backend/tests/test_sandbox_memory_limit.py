# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox memory limit: env first, then the owner's saved value, then half the RAM (at least 8 GiB)."""

from pathlib import Path
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from storage import studio_db
from utils import host_memory, sandbox_memory_limit as limit
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
GIB = 1024**3


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(limit.MEMORY_LIMIT_ENV, raising = False)
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: 16 * 1024)


def test_default_is_eight_gib_on_a_small_machine():
    assert limit.saved_memory_limit_gb() == 8
    assert limit.memory_limit_bytes() == 8 * GIB


@pytest.mark.parametrize(
    "total_mib, expected",
    [(32 * 1024, 16), (64 * 1024 - 300, 31), (4 * 1024, 8), (None, 8), (10**10, 4096)],
)
def test_default_is_half_the_ram(monkeypatch, total_mib, expected):
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: total_mib)
    assert limit.default_memory_limit_gb() == expected
    assert limit.memory_limit_bytes() == expected * GIB


def test_the_default_follows_the_ram_and_a_saved_value_does_not(monkeypatch):
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: 32 * 1024)
    assert limit.saved_memory_limit_gb() == 16
    limit.set_memory_limit_gb(12)
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: 64 * 1024)
    assert limit.saved_memory_limit_gb() == 12


def test_auto_clears_a_saved_value_and_follows_the_ram_again(monkeypatch):
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: 32 * 1024)
    assert limit.is_auto() is True
    limit.set_memory_limit_gb(16)
    assert limit.is_auto() is False
    limit.use_auto_memory_limit()
    assert limit.is_auto() is True
    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", lambda: 64 * 1024)
    assert limit.saved_memory_limit_gb() == 32


def test_an_unreadable_host_keeps_eight_gib(monkeypatch):
    def boom():
        raise OSError("no /proc")

    monkeypatch.setattr(host_memory, "host_ram_capacity_mib", boom)
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
