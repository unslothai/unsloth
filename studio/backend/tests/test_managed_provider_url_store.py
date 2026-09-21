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
    # The helper holds its answer briefly, and each test has its own home.
    mpu.forget_cached_setting()
    yield
    mpu.forget_cached_setting()


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


def test_the_held_answer_is_dropped_by_a_write(as_account):
    """Turning the switch off has to bite the next request, not the next second."""
    as_account(OWNER)
    mpu.set_managed_private_provider_urls_allowed(True)
    assert mpu.get_managed_private_provider_urls_allowed() is True

    mpu.set_managed_private_provider_urls_allowed(False)
    # No sleep: if this read came from the held answer it would still say True.
    assert mpu.get_managed_private_provider_urls_allowed() is False


def test_the_environment_lock_is_not_answered_from_the_cache(monkeypatch, as_account):
    """The strict answer must never be the held one, whatever was read a moment ago."""
    as_account(OWNER)
    mpu.set_managed_private_provider_urls_allowed(True)
    assert mpu.get_managed_private_provider_urls_allowed() is True

    monkeypatch.setenv(mpu.BLOCK_PRIVATE_ENV, "1")
    assert mpu.get_managed_private_provider_urls_allowed() is False
    monkeypatch.delenv(mpu.BLOCK_PRIVATE_ENV)
    assert mpu.get_managed_private_provider_urls_allowed() is True


def test_a_read_failure_is_never_remembered(monkeypatch, as_account):
    """A transient error fails closed per call; remembering it would pin the refusal."""
    as_account(OWNER)
    mpu.set_managed_private_provider_urls_allowed(True)
    mpu.forget_cached_setting()

    def _explode(*args, **kwargs):
        raise OSError("settings db is gone")

    import storage.studio_db as studio_db_module

    real = studio_db_module.get_app_setting
    studio_db_module.get_app_setting = _explode
    try:
        assert mpu.get_managed_private_provider_urls_allowed() is False
        assert mpu._remembered() is None
    finally:
        # By hand: monkeypatch.undo() would also undo the isolated-home fixture.
        studio_db_module.get_app_setting = real
    assert mpu.get_managed_private_provider_urls_allowed() is True


def test_the_held_answer_expires(monkeypatch, as_account):
    """A flip made in another process converges within the TTL rather than never."""
    as_account(OWNER)
    mpu.set_managed_private_provider_urls_allowed(False)
    assert mpu.get_managed_private_provider_urls_allowed() is False

    # Written behind this module's back, the way a second process would.
    from storage.studio_db import upsert_app_settings
    from utils.account_context import run_as

    run_as(
        OWNER,
        upsert_app_settings,
        {mpu.MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY: True},
    )
    assert mpu.get_managed_private_provider_urls_allowed() is False  # still held

    # Age the entry, not the clock: patching time.monotonic recurses through the cache's own call.
    with mpu._cache_lock:
        expiry, value = mpu._cached
        mpu._cached = (expiry - mpu._CACHE_TTL_SECONDS - 1.0, value)
    assert mpu.get_managed_private_provider_urls_allowed() is True
