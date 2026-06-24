# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration coverage: provisioning a real admin password before exposure.

Drives ``ensure_admin_password_before_exposure`` against a temp auth DB and
asserts that, for an exposed web UI, the first-run ``must_change_password``
state is cleared and the on-disk ``.bootstrap_password`` file is removed -- so
``_inject_bootstrap`` has nothing left to embed. Skips on loopback, api-only,
and Colab.
"""

from pathlib import Path

import pytest

from auth import storage
from auth.secure_admin_prompt import (
    ADMIN_PASSWORD_ENV_VAR,
    ensure_admin_password_before_exposure,
)


@pytest.fixture
def temp_auth_db(tmp_path, monkeypatch):
    """Point the storage module at a throwaway auth DB + bootstrap file."""
    db_path = tmp_path / "auth.db"
    bootstrap_path = tmp_path / ".bootstrap_password"
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", bootstrap_path)
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.delenv(ADMIN_PASSWORD_ENV_VAR, raising = False)
    return bootstrap_path


def _admin():
    return storage.DEFAULT_ADMIN_USERNAME


def test_env_password_clears_bootstrap_state(temp_auth_db, monkeypatch):
    monkeypatch.setenv(ADMIN_PASSWORD_ENV_VAR, "env-chosen-password")

    assert storage.ensure_default_admin() is True
    assert storage.requires_password_change(_admin()) is True
    assert temp_auth_db.is_file()

    ensure_admin_password_before_exposure(
        storage = storage,
        host = "0.0.0.0",
        secure = False,
        api_only = False,
        frontend_served = True,
        is_colab = False,
    )

    assert storage.requires_password_change(_admin()) is False
    assert not temp_auth_db.exists()


def test_secure_loopback_bind_still_provisions(temp_auth_db, monkeypatch):
    # --secure forces a loopback bind but exposes a public tunnel, so it must
    # still provision a real password.
    monkeypatch.setenv(ADMIN_PASSWORD_ENV_VAR, "env-chosen-password")
    storage.ensure_default_admin()

    ensure_admin_password_before_exposure(
        storage = storage,
        host = "127.0.0.1",
        secure = True,
        api_only = False,
        frontend_served = True,
        is_colab = False,
    )

    assert storage.requires_password_change(_admin()) is False


def test_short_env_password_refuses(temp_auth_db, monkeypatch):
    monkeypatch.setenv(ADMIN_PASSWORD_ENV_VAR, "short")
    storage.ensure_default_admin()

    with pytest.raises(SystemExit):
        ensure_admin_password_before_exposure(
            storage = storage,
            host = "0.0.0.0",
            secure = False,
            api_only = False,
            frontend_served = True,
            is_colab = False,
        )
    # The bootstrap state is untouched on refusal.
    assert storage.requires_password_change(_admin()) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(host = "127.0.0.1", secure = False, api_only = False, frontend_served = True),  # loopback
        dict(host = "0.0.0.0", secure = True, api_only = True, frontend_served = False),  # api-only
    ],
)
def test_skip_paths_leave_bootstrap_state(temp_auth_db, monkeypatch, kwargs):
    monkeypatch.setenv(ADMIN_PASSWORD_ENV_VAR, "env-chosen-password")
    storage.ensure_default_admin()

    ensure_admin_password_before_exposure(storage = storage, is_colab = False, **kwargs)

    # Nothing provisioned: the first-run state is preserved.
    assert storage.requires_password_change(_admin()) is True


def test_colab_skips_provisioning(temp_auth_db, monkeypatch):
    monkeypatch.setenv(ADMIN_PASSWORD_ENV_VAR, "env-chosen-password")
    storage.ensure_default_admin()

    ensure_admin_password_before_exposure(
        storage = storage,
        host = "0.0.0.0",
        secure = False,
        api_only = False,
        frontend_served = True,
        is_colab = True,
    )

    assert storage.requires_password_change(_admin()) is True


def test_backstop_when_no_tty_and_no_env(temp_auth_db, monkeypatch):
    # No env var and pytest has no interactive TTY: must not raise and must not
    # change the password (the main.py injection gate handles exposure safety).
    storage.ensure_default_admin()

    ensure_admin_password_before_exposure(
        storage = storage,
        host = "0.0.0.0",
        secure = False,
        api_only = False,
        frontend_served = True,
        is_colab = False,
    )

    assert storage.requires_password_change(_admin()) is True
