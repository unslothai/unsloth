# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /video/generate/cancel with a ``video_id``: only that render stops, any other id is a 409."""

from __future__ import annotations

import threading

import pytest

import core.inference.video as video_module
from core.inference.video_families import VIDEO_CANCELLED_MSG

from .test_video_routes import _healthy_diffusers, _wait_terminal, client  # noqa: F401

CANCEL = "/api/inference/video/generate/cancel"


@pytest.fixture
def rendering(client, monkeypatch):  # noqa: F811
    """A render of /v1 job ``vid-1`` that runs until it is cancelled."""
    backend = video_module.get_video_backend()
    backend.loaded = True
    started = threading.Event()

    def wait_for_cancel(*, cancel_event = None, **kwargs):
        started.set()
        assert cancel_event is not None and cancel_event.wait(5)
        raise RuntimeError(VIDEO_CANCELLED_MSG)

    begin = backend.begin_generate
    monkeypatch.setattr(
        backend, "begin_generate", lambda **kwargs: begin(**{**kwargs, "video_id": "vid-1"})
    )
    monkeypatch.setattr(backend, "generate", wait_for_cancel)
    assert client.post("/api/inference/video/generate", json = {"prompt": "p"}).status_code == 200
    assert started.wait(5)
    return backend


def test_a_wrong_id_is_refused_and_the_render_keeps_going(client, rendering):  # noqa: F811
    resp = client.post(CANCEL, json = {"video_id": "vid-2"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "That video is not rendering."
    assert client.get("/api/inference/video/generate-progress").json()["active"] is True
    assert client.post(CANCEL).json() == {"cancelled": True}
    assert _wait_terminal(client)["error"] == VIDEO_CANCELLED_MSG


def test_the_matching_id_cancels(client, rendering):  # noqa: F811
    resp = client.post(CANCEL, json = {"video_id": "vid-1"})
    assert resp.status_code == 200 and resp.json() == {"cancelled": True}
    progress = _wait_terminal(client)
    assert progress["phase"] == "failed" and progress["error"] == VIDEO_CANCELLED_MSG


@pytest.mark.parametrize("body", [None, {}, {"video_id": None}])
def test_without_an_id_an_idle_cancel_still_answers_false(client, body):  # noqa: F811
    resp = client.post(CANCEL, **({} if body is None else {"json": body}))
    assert resp.status_code == 200 and resp.json() == {"cancelled": False}


def test_an_id_with_nothing_rendering_is_a_409(client):  # noqa: F811
    resp = client.post(CANCEL, json = {"video_id": "vid-1"})
    assert resp.status_code == 409


@pytest.mark.parametrize("content", [b"{not json", b"[1]", b'"vid-1"', b'{"video_id": 7}'])
def test_a_bad_body_is_a_422(client, content):  # noqa: F811
    resp = client.post(CANCEL, content = content, headers = {"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert "no body" in resp.json()["detail"]


@pytest.mark.parametrize("video_id", ["", "  "])
def test_an_empty_id_is_a_422_and_the_render_keeps_going(client, rendering, video_id):  # noqa: F811
    resp = client.post(CANCEL, json = {"video_id": video_id})
    assert resp.status_code == 422
    assert "leave it out" in resp.json()["detail"]
    assert client.get("/api/inference/video/generate-progress").json()["active"] is True
    assert client.post(CANCEL).json() == {"cancelled": True}
    assert _wait_terminal(client)["error"] == VIDEO_CANCELLED_MSG
