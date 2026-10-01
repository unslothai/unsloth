# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``login_mode`` / ``installation_is_multi_user`` follow the ACTIVE count; ``installation_has_managed_accounts`` and the full-access gate follow whether any managed account exists at all."""

import secrets

import pytest

from auth import policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _state() -> tuple[str, bool, bool, bool]:
    policy.invalidate_account_cache()
    return (
        policy.login_mode(),
        policy.installation_is_multi_user(),
        policy.installation_has_managed_accounts(),
        policy.full_access_permitted(),
    )


def test_owner_only(auth_db):
    assert _state() == ("single", False, False, True)


def test_owner_plus_one_active_managed_account(auth_db):
    storage.issue_account_setup_code(username = "alice")
    assert _state() == ("multi", True, True, False)


def test_owner_plus_one_deactivated_managed_account(auth_db):
    account = storage.issue_account_setup_code(username = "alice")["account"]
    storage.set_account_active(account["account_id"], False)
    assert _state() == ("single", False, True, False)


def test_an_unreadable_auth_database(auth_db, monkeypatch):
    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    assert _state() == ("single", False, True, False)


def test_a_count_read_failure_keeps_a_bound_managed_account_isolated(auth_db, monkeypatch):
    from utils.account_context import AccountContext, run_as

    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    policy.invalidate_account_cache()
    assert run_as(AccountContext("a" * 32, "alice"), policy.installation_is_multi_user) is True


def test_deactivating_the_last_managed_account_keeps_a_bound_request_isolated(
    auth_db, monkeypatch, tmp_path
):
    from fastapi import HTTPException
    from hub.services.models import account_access as access
    from utils.account_context import AccountContext, run_as

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "installation-token")
    account = storage.issue_account_setup_code(username = "alice")["account"]
    alice = AccountContext(account["account_id"], "alice")
    assert run_as(alice, access.managed_account) is True

    storage.set_account_active(account["account_id"], False)
    assert policy.installation_is_multi_user() is False
    assert run_as(alice, access.managed_account) is True
    assert run_as(alice, access.ambient_hf_token) is False
    with pytest.raises(HTTPException) as raised:
        run_as(alice, access.require_model_access, "/owner/private/checkpoint.gguf")
    assert raised.value.status_code == 404


def test_deactivating_the_last_account_does_not_hand_its_download_to_the_owner(
    auth_db, monkeypatch, tmp_path
):
    """Deactivation does not cancel downloads, so the owner must stay scoped."""
    import logging

    from fastapi import HTTPException
    from hub.services import download_lifecycle
    from hub.services.models import account_access as access
    from hub.utils import download_registry
    from utils.account_context import OWNER, AccountContext, run_as

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(download_lifecycle, "_job_accounts", {})
    assert access.account_scope() is None
    record = storage.issue_account_setup_code(username = "alice")["account"]
    alice = AccountContext(record["account_id"], "alice")
    registry = download_registry.DownloadRegistry()
    key = "alice-org/private-secret::"
    registry.claim(key, "http", repo_type = "model", repo_id = "alice-org/private-secret")
    run_as(alice, download_lifecycle.record_download_account, registry, key)
    storage.set_account_active(record["account_id"], False)
    assert policy.installation_is_multi_user() is False
    assert run_as(OWNER, download_lifecycle.download_belongs_to_account, registry, key) is False
    assert (
        run_as(
            OWNER,
            lambda: download_lifecycle.active_download_refs(registry, None, with_variant = True),
        )
        == []
    )
    with pytest.raises(HTTPException) as exc:
        run_as(
            OWNER,
            lambda: download_lifecycle.cancel_worker(
                registry, key, generation = None, label = "model", logger = logging.getLogger(__name__)
            ),
        )
    assert exc.value.status_code == 404 and registry.get_job(key).state == "running"
    legacy = "legacy-org/model::"
    registry.claim(legacy, "http", repo_type = "model", repo_id = "legacy-org/model")
    assert run_as(OWNER, download_lifecycle.download_belongs_to_account, registry, legacy) is True
