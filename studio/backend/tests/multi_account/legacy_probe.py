# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fresh-process upgrade simulation; main is deliberately imported after seeding."""

import importlib
import json
import os
from contextlib import closing
from pathlib import Path

from .seed import (
    MESSAGE_ID,
    PASSWORD,
    SENTINEL,
    THREAD_ID,
    legacy_studio_rows,
    old_auth_row,
    seed_legacy_install,
)


def main() -> None:
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    original = seed_legacy_install(home)
    auth_row = old_auth_row(home / "auth" / "auth.db")
    studio_rows = legacy_studio_rows(home / "studio.db")
    application = importlib.import_module("main")
    for name, payload in original.items():
        assert (home / name).read_bytes() == payload, f"Import changed {name}"
    assert not (home / "accounts").exists()

    from fastapi.testclient import TestClient
    from auth import policy, storage
    from storage import credential_secrets, rag_db, studio_db
    from core.rag import store
    from utils.account_context import OWNER
    from utils.paths import workspace_root
    from core.inference.tools import sandbox_root

    assert storage.get_account("unsloth") == OWNER
    assert old_auth_row(home / "auth" / "auth.db") == auth_row
    assert workspace_root() == home
    assert Path(sandbox_root()) == home / "sandbox"
    assert studio_db.get_chat_thread(THREAD_ID)["title"] == SENTINEL
    assert studio_db.get_chat_message(THREAD_ID, MESSAGE_ID)["content"][0]["text"] == SENTINEL
    assert studio_db.list_chat_settings()["theme"] == "dark"
    assert studio_db.get_app_setting("legacy-owner-setting") == {"preserve": True}
    assert credential_secrets.get_hf_token() == "hf_legacy_private"
    with closing(rag_db.get_metadata_connection()) as conn:
        assert store.get_kb(conn, "legacy-kb")["name"] == SENTINEL
    assert policy.login_mode() == "single"

    client = TestClient(application.app)
    try:
        response = client.get("/api/auth/status")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "initialized": True,
            "default_username": "unsloth",
            "requires_password_change": False,
            "bootstrap_deadline_seconds": None,
            "login_mode": "single",
            "full_access": True,
        }
        login = client.post("/api/auth/login", json = {"username": "unsloth", "password": PASSWORD})
        assert login.status_code == 200, login.text
        assert login.json()["access_token"]
    finally:
        client.close()
    # auth.db and studio.db gain columns when a current build opens them, so both are compared by what
    # the old build wrote rather than byte for byte. Everything else must be untouched.
    for name, payload in original.items():
        if name not in ("auth/auth.db", "studio.db"):
            assert (home / name).read_bytes() == payload, f"Owner read changed {name}"
    assert old_auth_row(home / "auth" / "auth.db") == auth_row
    assert (
        legacy_studio_rows(home / "studio.db") == studio_rows
    ), "Owner read changed legacy studio.db rows"
    assert not (home / "accounts").exists()
    print(json.dumps({"preserved_files": len(original), "owner_login": True}))


if __name__ == "__main__":
    main()
