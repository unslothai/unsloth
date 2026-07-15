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
        r = client.get(path, headers=headers)
        assert r.status_code == 200, (path, r.status_code)
        assert "SEED-DO-NOT-LEAK" not in r.text, path
        assert "__UNSLOTH_BOOTSTRAP__" not in r.text, path
        # no per-request injection means no Origin-varying and no script nonce
        assert "x-internal-script-nonce" not in {k.lower() for k in r.headers}
