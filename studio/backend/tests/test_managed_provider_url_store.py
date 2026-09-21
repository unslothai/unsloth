# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installation-wide switch for managed-account private provider URLs.

The switch lives in the owner's app_settings, but every interesting read happens
while a MANAGED account is bound: ``workspace_root`` sends a managed caller to
``accounts/<id>/`` and ``account_path`` raises on anything outside it, so the
read has to cross back into the owner's tree deliberately. If it did not, the
feature would fail closed forever and the owner's switch would do nothing.
"""

from pathlib import Path
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from storage import studio_db
from utils import managed_provider_url_settings as mpu
from utils.account_context import OWNER, AccountContext, bind_account, reset_account
from utils.paths import storage_roots

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield


@pytest.fixture
def as_account():
    tokens = []

    def _bind(account):
        tokens.append(bind_account(account))

    yield _bind
    for token in reversed(tokens):
        reset_account(token)


def test_default_is_off():
    assert mpu.get_managed_private_provider_urls_allowed() is False


def test_owner_write_then_owner_read():
    assert mpu.set_managed_private_provider_urls_allowed(True) is True
    assert mpu.get_managed_private_provider_urls_allowed() is True


def test_managed_account_can_read_the_owners_switch(as_account):
    """The load-bearing one: a managed caller must see what the owner set."""
    mpu.set_managed_private_provider_urls_allowed(True)
    as_account(ALICE)
    assert mpu.get_managed_private_provider_urls_allowed() is True


def test_managed_read_does_not_create_a_per_account_copy(as_account, tmp_path):
    """A read from Alice must not land in, or be answered by, Alice's own DB."""
    mpu.set_managed_private_provider_urls_allowed(True)
    as_account(ALICE)
    assert mpu.get_managed_private_provider_urls_allowed() is True
    alice_db = tmp_path / "accounts" / ALICE.account_id / "studio.db"
    assert not alice_db.exists(), "the managed read opened a per-account settings DB"


def test_two_managed_accounts_see_one_installation_wide_answer(as_account):
    mpu.set_managed_private_provider_urls_allowed(True)
    for account in (ALICE, BOB):
        token = bind_account(account)
        try:
            assert mpu.get_managed_private_provider_urls_allowed() is True
        finally:
            reset_account(token)


def test_managed_account_write_still_targets_the_owner_db(as_account):
    """The route is owner-only, but the setter must not silently write the wrong DB
    if it is ever called from a managed context."""
    as_account(ALICE)
    mpu.set_managed_private_provider_urls_allowed(True)
    reset_account(bind_account(OWNER))
    token = bind_account(OWNER)
    try:
        assert mpu.get_managed_private_provider_urls_allowed() is True
    finally:
        reset_account(token)


def test_read_failure_fails_closed(monkeypatch):
    mpu.set_managed_private_provider_urls_allowed(True)

    def boom(*args, **kwargs):
        raise sqlite_error()

    def sqlite_error():
        import sqlite3

        return sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(studio_db, "get_app_setting", boom)
    assert mpu.get_managed_private_provider_urls_allowed() is False


def test_account_path_would_have_raised_without_the_owner_hop(as_account):
    """Documents why the run_as hop exists: the owner's DB path is not reachable
    from a managed context by the ordinary accessor."""
    as_account(ALICE)
    with pytest.raises(ValueError):
        storage_roots.account_path("../../studio.db")


@pytest.mark.parametrize(
    "stored,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("off", False),
        ("", False),
        (None, False),
        # Junk is not a yes.
        ("maybe", False),
        (1, False),
    ],
)
def test_stored_value_coercion(monkeypatch, stored, expected):
    monkeypatch.setattr(studio_db, "get_app_setting", lambda key, fallback = None: stored)
    assert mpu.get_managed_private_provider_urls_allowed() is expected


@pytest.mark.parametrize("value", ["maybe", 1, 0, None, [], {}])
def test_setter_rejects_non_boolean(value):
    if isinstance(value, str) and value in {"1", "true", "yes", "on"}:
        pytest.skip("accepted spelling")
    with pytest.raises(ValueError):
        mpu.set_managed_private_provider_urls_allowed(value)
