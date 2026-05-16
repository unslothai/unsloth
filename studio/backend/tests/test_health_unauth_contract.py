# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the /api/health unauthenticated contract.

PR 5375 stripped the legacy identity / capability fields from unauthenticated
``/api/health`` responses. That broke:

* ``install.sh::_check_health`` which matches on ``service`` + ``studio_root_id``.
* ``studio/src-tauri/src/preflight/backend.rs`` which reads
  ``service`` / ``desktop_protocol_version`` / ``studio_root_id``.
* ``run_studio_browser_test.preflight`` which mirrors the install.sh
  matcher.

The follow-up fix re-publishes the launcher contract (status / timestamp
/ service / studio_root_id / desktop protocol bits / supports_desktop_*)
unauthenticated, and keeps the sensitive diagnostic fields (version /
device_type / chat_only / desktop_owner / native_path_leases_supported)
gated on a valid bearer.

This module pins the contract in both directions so regressions show up
before they ship.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


LAUNCHER_KEYS = {
    "status",
    "timestamp",
    "service",
    "studio_root_id",
    "chat_only",
    "desktop_protocol_version",
    "desktop_manageability_version",
    "supports_desktop_auth",
    "supports_desktop_backend_ownership",
}

GATED_KEYS = {
    "version",
    "studio_version",
    "device_type",
    "native_path_leases_supported",
}


@pytest.fixture()
def fastapi_client(tmp_path, monkeypatch):
    """Boot the FastAPI app against a tmp Studio home and return a TestClient."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_API_ONLY", "1")
    # Reset any cached module state so DB_PATH / install root resolve
    # to the tmp directory.
    for name in list(sys.modules):
        if name.startswith(("auth.", "main", "models.", "routes.", "loggers.")):
            del sys.modules[name]
    main = importlib.import_module("main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        yield client, main


class TestUnauthHealth:
    def test_status_healthy(self, fastapi_client):
        client, _ = fastapi_client
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"

    def test_includes_launcher_contract(self, fastapi_client):
        client, _ = fastapi_client
        body = client.get("/api/health").json()
        missing = LAUNCHER_KEYS - set(body)
        assert not missing, f"unauth /api/health missing {sorted(missing)}"

    def test_does_not_leak_gated_fields(self, fastapi_client):
        client, _ = fastapi_client
        body = client.get("/api/health").json()
        leaked = GATED_KEYS & set(body)
        assert not leaked, f"unauth /api/health leaked {sorted(leaked)}"

    def test_invalid_bearer_drops_back_to_unauth(self, fastapi_client):
        client, _ = fastapi_client
        body = client.get(
            "/api/health", headers = {"Authorization": "Bearer not-real"}
        ).json()
        leaked = GATED_KEYS & set(body)
        assert not leaked, f"invalid-bearer health leaked {sorted(leaked)}"
        missing = LAUNCHER_KEYS - set(body)
        assert not missing, f"invalid-bearer health missing {sorted(missing)}"

    def test_coroutine_truthy_does_not_skip_auth(self, fastapi_client):
        """Regression: a bare coroutine is truthy.

        Before the fix the handler called ``get_current_subject(creds)``
        without ``await``, so any header starting with ``Bearer `` would
        be seen as a valid principal and produce the full payload.
        """
        client, _ = fastapi_client
        body = client.get(
            "/api/health", headers = {"Authorization": "Bearer x.y.z"}
        ).json()
        # Verify gated fields are NOT leaked when the token cannot be
        # decoded (which would have been the case for the original bug).
        assert "version" not in body
        assert "device_type" not in body


class TestAuthedHealth:
    """Authenticated health should expose the diagnostic dict."""

    def test_valid_bearer_exposes_diagnostic(self, fastapi_client):
        client, main = fastapi_client
        # Bypass HTTP login -- mint a token through auth.authentication
        # so the test does not depend on the bootstrap password file
        # existing in this tmp install. The default-admin row was
        # created during lifespan startup so the subject is valid.
        from auth.authentication import create_access_token
        from auth import storage

        # Ensure the unsloth admin user exists so its jwt_secret is on
        # disk and the subsequent get_current_subject(...) accepts the
        # token. The fixture's lifespan call usually seeds it, but
        # when this test runs after a prior fixture's module reload
        # the auth.storage module may have been re-imported and lost
        # its in-process DB connection. Idempotently re-seed here.
        if storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME) is None:
            # ensure_default_admin is idempotent (no-op when the row
            # already exists). The fixture's lifespan usually seeds the
            # default admin, but module reloads between tests can leave
            # the in-process state inconsistent; re-seeding here keeps
            # the test order-independent.
            seed = getattr(storage, "ensure_default_admin", None)
            if seed is not None:
                seed()
            else:
                pytest.skip("storage has no ensure_default_admin entrypoint")
        assert (
            storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME) is not None
        ), "could not seed default admin"
        # Clear the must_change_password flag so /api/health's
        # get_current_subject dependency accepts the token. Fresh installs
        # block diagnostic access until the first-boot password change,
        # which is the production contract but inconvenient for this test.
        conn = storage.get_connection()
        try:
            conn.execute(
                "UPDATE auth_user SET must_change_password = 0 WHERE username = ?",
                (storage.DEFAULT_ADMIN_USERNAME,),
            )
            conn.commit()
        finally:
            conn.close()
        token = create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)
        body = client.get(
            "/api/health", headers = {"Authorization": f"Bearer {token}"}
        ).json()
        # Diagnostic keys are present.
        for k in ("version", "device_type"):
            assert k in body, f"authed health missing {k!r}"
        # Launcher contract still present (authed payload is a superset).
        missing = LAUNCHER_KEYS - set(body)
        assert not missing, f"authed health missing launcher keys {sorted(missing)}"
