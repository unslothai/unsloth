# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Omitted ``loras`` on /images/generate (and its tool / queue / XY callers) keeps the load's adapters; ``[]`` none."""

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
    from core.inference.generate_image_tool import _build_request
    from routes.inference import _generate_diffusion_image_core

    request = _build_request({}, "trtcrd a sloth", baked.backend.status())
    assert request.loras is None
    asyncio.run(_generate_diffusion_image_core(request))
    assert baked.backend.seen == [[("tarot-lora", 0.8)]]


def test_an_xy_cell_with_every_weight_at_0_renders_without_the_baked_set(baked):
    """A cell whose only adapter is swept to 0 must stay adapter-free."""
    from core.inference import diffusion_xy
    from models.inference import DiffusionGenerateRequest
    from routes.inference import _generate_diffusion_image_core

    plan = diffusion_xy.expand(
        DiffusionGenerateRequest(
            prompt = "trtcrd a sloth",
            seed = 7,
            loras = [{"id": "tarot-lora", "weight": 0.8}],
            xy = {"x": {"field": "lora_weight", "index": 0, "values": [0.8, 0]}},
        )
    )
    for cell in plan.cells:
        asyncio.run(_generate_diffusion_image_core(cell.request))
    assert baked.backend.seen == [[("tarot-lora", 0.8)], None]


def test_a_compiled_dense_load_omitting_loras_renders_without_its_named_set(client, monkeypatch):  # noqa: F811
    """supports_lora false after Speed Auto compiled it: the named set must not come back and be refused."""

    class _Compiled(_BakedLoad):
        def status(self):
            return {
                **super().status(),
                "supports_lora": False,
                "lora_unavailable_reason": "Speed compiled this model, so LoRAs are off.",
            }

    backend = _Compiled(baked = [], dense = [{"id": "tarot-lora", "weight": 1.0}])
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    assert _gen(client).status_code == 200
    assert backend.seen == [None]


def test_a_load_swapped_in_before_the_slot_retries_with_its_own_set(client, monkeypatch):  # noqa: F811
    """Resolved to none, then a baked load lands before the slot: refuse as replaced and retry with its set."""
    from core.inference.diffusion_families import DiffusionModelReplacedError, load_identity

    class _Swapping(_BakedLoad):
        def __init__(self):
            super().__init__(baked = [])
            self.repo = "x/plain"

        def status(self):
            return {**super().status(), "repo_id": self.repo}

        def generate(self, **kwargs):
            if self.repo == "x/plain":
                self.repo, self._baked = "x/baked", list(BAKED)
            loaded = load_identity(self.repo, None, "z-image")
            expected = kwargs.get("expected_load")
            if expected is not None and expected != loaded:
                raise DiffusionModelReplacedError(expected, loaded)
            return super().generate(**kwargs)

    backend = _Swapping()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _gen(client, steps = 8, guidance = 1.0, width = 1024, height = 1024)
    assert resp.status_code == 200, resp.text
    assert backend.seen == [[("tarot-lora", 0.8)]]


def test_a_same_model_reload_with_other_adapters_retries_with_its_own_set(client, monkeypatch):  # noqa: F811
    """The identity pin still matches, so the adapter signature must refuse it and retry with the new set."""
    from core.inference.diffusion_families import (
        DiffusionModelReplacedError,
        load_identity,
        load_lora_signature,
    )

    class _SameModelReload(_BakedLoad):
        def __init__(self):
            super().__init__(baked = [])
            self.swapped = False

        def generate(self, **kwargs):
            if not self.swapped:
                self.swapped, self._baked = True, list(BAKED)
            loaded = load_identity("x/z-image", None, "z-image")
            expected = kwargs.get("expected_load")
            if expected is not None and expected != loaded:
                raise DiffusionModelReplacedError(expected, loaded)
            pinned = kwargs.get("expected_load_loras")
            if pinned is not None and tuple(pinned) != load_lora_signature(
                self.status()["load_lora_specs"]
            ):
                raise DiffusionModelReplacedError(expected or loaded, loaded)
            return super().generate(**kwargs)

    backend = _SameModelReload()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    resp = _gen(client, steps = 8, guidance = 1.0, width = 1024, height = 1024)
    assert resp.status_code == 200, resp.text
    assert backend.seen == [[("tarot-lora", 0.8)]]


def test_the_diffusers_backend_checks_the_adapter_pin_in_its_slot():
    """The diffusers backend takes the adapter pin and compares it inside the generation slot."""
    import inspect

    from core.inference.diffusion import DiffusionBackend

    assert "expected_load_loras" in inspect.signature(DiffusionBackend.generate).parameters
    src = inspect.getsource(DiffusionBackend.generate)
    assert "expected_load_loras" in src and "load_lora_signature(" in src
