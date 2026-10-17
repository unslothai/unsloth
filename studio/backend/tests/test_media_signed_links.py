# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Key-free signed links for gallery images, audio and video: mint with the bearer, fetch with
nothing but the link, and only for the minting account while it is active and the link is fresh."""

from __future__ import annotations

import sqlite3
import time
import types
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage as auth_storage
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes import inference, video
from utils.account_context import bind_account, reset_account, run_as

from .test_account_media_isolation import ALICE, BOB, _save, isolated  # noqa: F401

# kind -> (module holding the signer and its clock, signer, mount the links are fetched on)
KINDS = {
    "images": (inference, "_sign_image_id", "time", "/api/inference/images/gallery"),
    "audio": (inference, "_sign_audio_id", "time", "/v1/audio/gallery"),
    "video": (video, "_sign_video_id", "_time", "/api/inference/video/gallery"),
}


def _app(account = None) -> TestClient:
    """Mounted like main.py; with no account, the real auth dependency runs (no overrides)."""
    app = FastAPI()
    if account is not None:

        async def subject():
            token = bind_account(account)
            try:
                yield account.username
            finally:
                reset_account(token)

        app.dependency_overrides[get_current_subject] = subject
        app.dependency_overrides[authenticated_via_api_key] = lambda: False
    for prefix in ("/api/inference", "/v1"):
        app.include_router(inference.router, prefix = prefix)
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    return TestClient(app)


def _mint(
    kind,
    media_id,
    account = ALICE,
    **params,
):
    with _app(account) as client:
        return client.get(f"{KINDS[kind][3]}/{media_id}/signed-url", params = params)


def _exp_left(url: str) -> int:
    return int(url.split("token=", 1)[1].rsplit(".", 2)[1]) - int(time.time())


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_minted_link_fetches_with_no_credential_at_all(kind):
    media_id = run_as(ALICE, _save, kind)["id"]
    root = KINDS[kind][3]
    minted = _mint(kind, media_id)
    assert minted.status_code == 200, minted.text
    url = minted.json()["url"]
    assert url.startswith(f"{root}/{media_id}/file-signed?token=")
    with _app() as bare:
        fetched = bare.get(url)
        assert fetched.status_code == 200, fetched.text
        assert "immutable" in fetched.headers["cache-control"]
        # Control: the same client is refused on the bearer route, so the link is the credential.
        assert bare.get(f"{root}/{media_id}/file").status_code in (401, 403)
    with _app(ALICE) as client:
        assert fetched.content == client.get(f"{root}/{media_id}/file").content


def test_audio_link_serves_ranges_and_follows_the_mount_called():
    audio_id = run_as(ALICE, _save, "audio")["id"]
    with _app(ALICE) as client:
        url = client.get(f"/api/inference/audio/gallery/{audio_id}/signed-url").json()["url"]
    assert url.startswith(f"/api/inference/audio/gallery/{audio_id}/file-signed?token=")
    ranged = _app().get(_mint("audio", audio_id).json()["url"], headers = {"Range": "bytes=0-3"})
    assert ranged.status_code == 206
    assert ranged.headers["content-type"] == "audio/wav"
    assert len(ranged.content) == 4


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_link_works_until_its_expiry_and_not_after(kind, monkeypatch):
    module, sign, clock, root = KINDS[kind]
    media_id = run_as(ALICE, _save, kind)["id"]
    token = run_as(ALICE, getattr(module, sign), media_id)
    exp = int(token.rsplit(".", 2)[1])
    url = f"{root}/{media_id}/file-signed"
    with _app() as bare:
        for now, status in ((exp - 1, 200), (exp + 1, 401)):
            monkeypatch.setattr(module, clock, types.SimpleNamespace(time = lambda now = now: now))
            response = bare.get(url, params = {"token": token})
            assert response.status_code == status, (now, response.text)
    assert response.json()["detail"].startswith("Invalid or expired")


@pytest.mark.parametrize("kind", list(KINDS))
def test_links_serve_only_the_minting_account_and_die_with_it(kind):
    module, sign, _clock, root = KINDS[kind]
    media_id = run_as(ALICE, _save, kind)["id"]
    assert _mint(kind, media_id, BOB).status_code == 404
    assert _mint(kind, "missing", ALICE).status_code == 404
    token = run_as(ALICE, getattr(module, sign), media_id)
    url = f"{root}/{media_id}/file-signed"
    with _app() as bare:
        tampered = token.replace(ALICE.account_id, BOB.account_id)
        assert bare.get(url, params = {"token": tampered}).status_code == 401
        assert bare.get(f"{root}/guessed/file-signed", params = {"token": token}).status_code == 401
        assert bare.get(url, params = {"token": token[:-1] + "x"}).status_code == 401
        assert bare.get(url, params = {"token": token}).status_code == 200
        with closing(sqlite3.connect(auth_storage.DB_PATH)) as conn:
            conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
            conn.commit()
        policy.invalidate_account_cache()
        assert bare.get(url, params = {"token": token}).status_code == 401


@pytest.mark.parametrize("kind", list(KINDS))
def test_mcp_links_last_an_hour_and_the_ui_video_link_twelve(kind):
    media_id = run_as(ALICE, _save, kind)["id"]
    params = {"ttl_s": 3600} if kind == "video" else {}
    assert 3590 <= _exp_left(_mint(kind, media_id, **params).json()["url"]) <= 3600
    if kind == "video":
        assert 43190 <= _exp_left(_mint(kind, media_id).json()["url"]) <= 43200
        for ttl_s in (59, 43201):
            assert _mint(kind, media_id, ttl_s = ttl_s).status_code == 422


@pytest.mark.parametrize("kind", ["images", "audio"])
def test_keyless_callers_cannot_mint_links(kind, monkeypatch):
    import auth.authentication

    media_id = run_as(ALICE, _save, kind)["id"]
    monkeypatch.setattr(auth.authentication, "request_admitted_without_credential", lambda r: True)
    response = _mint(kind, media_id)
    assert response.status_code == 403
    assert "API key" in response.json()["detail"]
