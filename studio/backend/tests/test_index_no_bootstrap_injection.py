# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The seeded bootstrap password is never embedded in the served index.html.

The seed is delivered only via the startup log and ``.bootstrap_password``; the
served page must never carry it, for any caller.
"""

from fastapi import FastAPI
from starlette.testclient import TestClient


def _client(tmp_path, monkeypatch):
    import main

    # Force the conditions that previously triggered injection so this proves the
    # seed is withheld even with a pending password change and a seed present.
    monkeypatch.setattr(main.storage, "requires_password_change", lambda *a, **k: True)
    build = tmp_path / "build"
    build.mkdir()
    (build / "index.html").write_text("<html><head></head><body>ok</body></html>")
    app = FastAPI()
    app.state.bootstrap_password = "SEED-DO-NOT-LEAK"
    assert main.setup_frontend(app, build) is True
    return TestClient(app)


def test_index_never_contains_bootstrap_seed(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    # root, SPA fallback, and a same-origin request all get a clean page.
    for path, headers in (
        ("/", {}),
        ("/some/spa/route", {}),
        ("/", {"origin": "http://testserver"}),
    ):
        r = client.get(path, headers = headers)
        assert r.status_code == 200, (path, r.status_code)
        assert "SEED-DO-NOT-LEAK" not in r.text, path
        assert "__UNSLOTH_BOOTSTRAP__" not in r.text, path
        # no per-request injection means no Origin-varying and no script nonce
        assert "x-internal-script-nonce" not in {k.lower() for k in r.headers}


def test_seed_pointer_survives_bootstrap_suppression(tmp_path, monkeypatch, capsys):
    """A public launch that suppressed injection must still say WHERE the seed is.

    With the in-page credential gone, .bootstrap_password is the only way to
    learn the seeded password. suppress_bootstrap_injection means "do not hand
    the secret to a public page"; the pointer is a PATH, not the secret, so
    gating it on that flag hid the file on exactly the launch that needs it --
    a public launch restarted before first login.
    """
    import asyncio
    import threading

    from fastapi import FastAPI

    import main as studio_main
    from auth import storage
    from core.rag import folder_sync

    # Keep lifespan shutdown from latching folder-sync's global events for later tests.
    monkeypatch.setattr(folder_sync, "_stop", threading.Event())
    monkeypatch.setattr(folder_sync, "_wake", threading.Event())

    authdir = tmp_path / "auth"
    authdir.mkdir(parents = True)
    monkeypatch.setattr(storage, "DB_PATH", authdir / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", authdir / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None, raising = False)
    seed = storage.generate_bootstrap_password()
    storage.ensure_default_admin()  # first run already happened
    assert storage.ensure_default_admin() is False  # so this is the restart branch

    app = FastAPI()
    app.state.suppress_bootstrap_injection = True

    async def _drive():
        async with studio_main.lifespan(app):
            pass

    asyncio.run(_drive())
    out = capsys.readouterr().out
    assert ".bootstrap_password" in out, (
        "a suppressed public launch printed no pointer to the seed file, so the "
        "operator has no way to learn the password"
    )
    assert seed not in out, "the seed itself must never reach stdout (CWE-532)"
    # Suppression still does its job: the secret is not parked in app.state.
    assert getattr(app.state, "bootstrap_password", None) is None
