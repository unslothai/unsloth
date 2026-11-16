# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /images/generate (and every internal caller of the same run: chat generate_image tool, queue, XY) keeps the
LoRAs a load baked in when ``loras`` is omitted; ``[]`` renders without any. The engine reads a missing selection as
"none", which on an int8 / fp8 bake zeroes the adapters, so the route resolves an omitted list first. No GPU."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import core.inference.diffusion as diffusion_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_routes import _FakeBackend, client  # noqa: E402,F401

BAKED = [{"id": "tarot-lora", "weight": 0.8}]


class _BakedLoad(_FakeBackend):
    """An int8 load that baked ``tarot-lora`` at 0.8; generate records the selection it was handed."""

    def __init__(
        self,
        baked = BAKED,
        dense = None,
    ):
        super().__init__()
        self.loaded = True
        self._baked = baked
        self._dense = dense or []
        self.seen = []

    def status(self):
        return {
            **super().status(),
            "loaded": True,
            "repo_id": "x/z-image",
            "family": "z-image",
            "transformer_quant": "int8" if self._baked else None,
            "baked_lora_specs": list(self._baked),
            "load_lora_specs": list(self._baked or self._dense),
            "generation_defaults": {"steps": 8.0, "guidance": 1.0},
        }

    def generate(self, **kwargs):
        self.seen.append(kwargs.get("loras"))
        return super().generate(**kwargs)


@pytest.fixture
def baked(client, monkeypatch):  # noqa: F811
    backend = _BakedLoad()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    client.backend = backend  # type: ignore[attr-defined]
    return client


def _gen(c, **body):
    return c.post(
        "/api/inference/images/generate", json = {"prompt": "trtcrd a sloth", "seed": 7, **body}
    )


def test_omitted_loras_render_with_the_baked_set(baked):
    resp = _gen(baked)
    assert resp.status_code == 200, resp.text
    assert baked.backend.seen == [[("tarot-lora", 0.8)]]
    assert resp.json()["images"][0]["loras"] == ["tarot-lora:0.8"]


def test_empty_loras_render_without_adapters(baked):
    resp = _gen(baked, loras = [])
    assert resp.status_code == 200, resp.text
    assert baked.backend.seen == [None]


def test_explicit_loras_are_sent_as_given(baked):
    resp = _gen(baked, loras = [{"id": "tarot-lora", "weight": 0.5}])
    assert resp.status_code == 200, resp.text
    assert baked.backend.seen == [[("tarot-lora", 0.5)]]


def test_a_dense_load_keeps_the_adapters_it_named(client, monkeypatch):  # noqa: F811
    backend = _BakedLoad(baked = [], dense = [{"id": "tarot-lora", "weight": 1.0}])
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    assert _gen(client).status_code == 200
    assert backend.seen == [[("tarot-lora", 1.0)]]


def test_a_load_without_adapters_keeps_the_old_call(client, monkeypatch):  # noqa: F811
    backend = _BakedLoad(baked = [])
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    assert _gen(client).status_code == 200
    assert backend.seen == [None]


def test_the_chat_generate_image_tool_keeps_the_baked_set(baked):
    """The tool builds a request with no loras and runs the same core as the route."""
    from core.inference.generate_image_tool import _build_request
    from routes.inference import _generate_diffusion_image_core

    request = _build_request({}, "trtcrd a sloth", baked.backend.status())
    assert request.loras is None
    asyncio.run(_generate_diffusion_image_core(request))
    assert baked.backend.seen == [[("tarot-lora", 0.8)]]
