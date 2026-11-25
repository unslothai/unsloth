# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/images/generate refuses a run while an image model load or switch is in flight (409, problem code loading), so
it never renders the outgoing model with the incoming pick's settings. No GPU: the backend is faked."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import core.inference.diffusion as diffusion_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_routes import client  # noqa: E402,F401 -- the route fixture with a faked backend


def _arm(monkeypatch, phase):
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    calls = []

    def _generate(**_kwargs):
        calls.append(1)
        return {"images": [], "seed": 1, "seeds": []}

    monkeypatch.setattr(backend, "generate", _generate)
    monkeypatch.setattr(
        backend,
        "load_progress",
        lambda: {
            "phase": phase,
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "fraction": 0.0,
            "error": None,
        },
    )
    return calls


@pytest.mark.parametrize("phase", ["downloading", "finalizing"])
def test_generate_during_a_load_is_refused(client, monkeypatch, phase):
    from utils.api_errors import PROBLEM_HEADER

    calls = _arm(monkeypatch, phase)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409, resp.text
    assert "loading" in resp.json()["detail"]
    assert resp.headers.get(PROBLEM_HEADER, "").startswith("loading")
    assert calls == []


@pytest.mark.parametrize("phase", ["ready", None, "error"])
def test_generate_after_a_load_runs(client, monkeypatch, phase):
    calls = _arm(monkeypatch, phase)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 200, resp.text
    assert calls == [1]


def test_a_load_post_still_in_preflight_counts(client, monkeypatch):
    from core.inference import chat_swap_kv

    calls = _arm(monkeypatch, "ready")
    monkeypatch.setattr(chat_swap_kv, "load_preparing", lambda media: media == "image")
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409 and calls == []
