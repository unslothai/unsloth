# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A LoRA selection an INT8 / FP8 image load cannot take without a reload is refused with the remedy and a code the
Images page and API clients can act on: X-Unsloth-Refusal: lora-reload on /images/generate, error.code
lora_reload_required on /v1/images/generations. No GPU: the backend is faked."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

import core.inference.diffusion as diffusion_module
import core.inference.image_gallery as gallery_module
from core.inference.diffusion_lora import (
    LORA_RELOAD_OPENAI_CODE,
    LORA_RELOAD_REFUSAL,
    LoraReloadRequiredError,
    LoraSelectionError,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_routes import client  # noqa: E402,F401 -- the route fixture with a faked backend


def _state(quant = "int8"):
    pipe = types.SimpleNamespace(_unsloth_loras = (), _unsloth_loras_baked = False)
    return types.SimpleNamespace(
        pipe = pipe,
        transformer_quant = quant,
        family = types.SimpleNamespace(name = "moss-image"),
        hf_token = None,
        cuda_graphs = (),
    )


@pytest.mark.parametrize("quant, label", [("int8", "INT8"), ("fp8", "FP8"), (None, "quantized")])
def test_an_unbaked_quantized_load_names_the_reload(quant, label):
    state = _state(quant)
    backend = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    with pytest.raises(LoraReloadRequiredError) as caught:
        backend._adjust_baked_loras(
            state, state.pipe, [("tarot", 1.0)], (), False, threading.Event()
        )
    msg = str(caught.value)
    assert f"This {label} build was loaded without LoRA adapters" in msg
    # The exact remedy: the page's buttons and the load field.
    assert "Reapply" in msg and "Generate" in msg
    assert '"loras" in POST /api/inference/images/load' in msg
    assert isinstance(caught.value, LoraSelectionError) and isinstance(caught.value, ValueError)


def test_a_new_adapter_on_a_baked_load_names_the_reload(monkeypatch):
    state = _state("fp8")
    baked = (("tarot", "/loras/tarot.safetensors", 1.0),)
    state.pipe._unsloth_loras = baked
    state.pipe._unsloth_loras_baked = True
    backend = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    monkeypatch.setattr(
        backend,
        "_resolve_lora_set",
        lambda specs, **_kw: tuple((i, f"/loras/{i}.safetensors", w) for (i, w) in specs),
    )
    with pytest.raises(LoraReloadRequiredError) as caught:
        backend._adjust_baked_loras(
            state, state.pipe, [("tarot", 1.0), ("other", 1.0)], baked, True, threading.Event()
        )
    assert "FP8 build was not loaded with" in str(caught.value)
    # A subset of the bake still renders without a reload.
    backend._adjust_baked_loras(state, state.pipe, [("tarot", 1.0)], baked, True, threading.Event())


def test_the_refusal_codes_are_stable():
    # The Images page matches these literally (frontend lib/lora-reload.ts).
    assert LORA_RELOAD_REFUSAL == "lora-reload"
    assert LORA_RELOAD_OPENAI_CODE == "lora_reload_required"
    exc = LoraReloadRequiredError("x")
    assert exc.refusal == LORA_RELOAD_REFUSAL and exc.openai_code == LORA_RELOAD_OPENAI_CODE


def test_images_generate_tags_the_reload_refusal(client, monkeypatch):
    from core.inference.diffusion_memory import IMAGE_REFUSAL_HEADER

    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True

    def _refuse(**_kwargs):
        raise LoraReloadRequiredError("This INT8 build was loaded without LoRA adapters.")

    monkeypatch.setattr(backend, "generate", _refuse)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 400
    assert resp.headers.get(IMAGE_REFUSAL_HEADER) == LORA_RELOAD_REFUSAL
    assert "loaded without LoRA adapters" in resp.json()["detail"]

    def _plain(**_kwargs):
        raise LoraSelectionError("LoRA 'x' is for flux.1, not the loaded 'moss-image' model family")

    monkeypatch.setattr(backend, "generate", _plain)
    other = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert other.status_code == 400
    assert IMAGE_REFUSAL_HEADER not in other.headers


def test_openai_images_carry_the_reload_code(monkeypatch):
    from test_openai_images_baked_loras import _BakedBackend
    from test_openai_images_generations_route import _make_client

    backend = _BakedBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)

    def _refuse(**_kwargs):
        raise LoraReloadRequiredError("This INT8 build was loaded without LoRA adapters.")

    monkeypatch.setattr(backend, "generate", _refuse)
    body = {"size": "256x256", "prompt": "a sloth", "loras": [{"id": "other", "weight": 1}]}
    resp = cli.post("/v1/images/generations", json = body)
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["code"] == LORA_RELOAD_OPENAI_CODE and err["param"] == "loras"

    def _plain(**_kwargs):
        raise LoraSelectionError("Duplicate LoRA ids")

    monkeypatch.setattr(backend, "generate", _plain)
    resp = cli.post("/v1/images/generations", json = body)
    assert resp.status_code == 400 and resp.json()["error"]["code"] is None
