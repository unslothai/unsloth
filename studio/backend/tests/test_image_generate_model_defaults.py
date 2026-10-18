# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/images/generate fills omitted steps and guidance from the loaded model, like /v1/images."""

from __future__ import annotations

from fastapi import HTTPException

import core.inference.diffusion as diffusion_module
import routes.inference as inference_routes
from core.inference.diffusion_families import DiffusionModelReplacedError, load_identity

from .test_diffusion_routes import _FakeBackend, _unloaded_status, client  # noqa: F401  (fixture)

TURBO = {"repo_id": "unsloth/Z-Image-Turbo", "generation_defaults": {"steps": 8, "guidance": 0.0}}
DEV = {"repo_id": "unsloth/FLUX.1-dev", "generation_defaults": {"steps": 28, "guidance": 3.5}}


class _ModelBackend(_FakeBackend):
    """Loaded with ``models[0]``; each generate swaps in the next one first, as a replacement landing mid-request would."""

    def __init__(
        self,
        *models,
        loaded = True,
    ) -> None:
        super().__init__()
        self.loaded = loaded
        self.models = [dict(m) for m in models]
        self.calls: list[dict] = []
        self.status_calls = 0

    def _identity(self):
        model = self.models[0]
        return load_identity(model.get("repo_id"), model.get("base_repo"), "z-image")

    def status(self):
        self.status_calls += 1
        if not self.loaded:
            return _unloaded_status()
        return {**_unloaded_status(), "loaded": True, "family": "z-image", **self.models[0]}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.loaded:
            return super().generate(**kwargs)
        if len(self.models) > 1:
            self.models.pop(0)
        expected = kwargs.get("expected_load")
        if expected is not None and expected != self._identity():
            raise DiffusionModelReplacedError(expected, self._identity())
        return super().generate(**kwargs)


def _use(monkeypatch, backend):
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    return backend


def _recipe(response):
    assert response.status_code == 200, response.text
    image = response.json()["images"][0]
    return image["steps"], image["guidance"]


def test_omitted_steps_and_guidance_follow_the_status_recipe(client, monkeypatch):
    backend = _use(monkeypatch, _ModelBackend(DEV))
    response = client.post("/api/inference/images/generate", json = {"prompt": "a sloth"})
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (28, 3.5)
    assert _recipe(response) == (28, 3.5)
    assert backend.calls[0]["expected_load"] == load_identity("unsloth/FLUX.1-dev", None, "z-image")


def test_without_a_status_recipe_the_family_table_decides(client, monkeypatch):
    # sd.cpp and MLX report no generation_defaults; base_repo names FLUX.1-dev for a local path.
    flux = {"repo_id": "/models/my-flux", "base_repo": "black-forest-labs/FLUX.1-dev"}
    backend = _use(monkeypatch, _ModelBackend(flux))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert _recipe(response) == (20, 3.5)
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (20, 3.5)

    backend = _use(monkeypatch, _ModelBackend({"repo_id": "/models/opaque"}))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert _recipe(response) == (9, 0.0)


def test_only_the_omitted_value_is_filled(client, monkeypatch):
    backend = _use(monkeypatch, _ModelBackend(DEV))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p", "steps": 12})
    assert _recipe(response) == (12, 3.5)
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (12, 3.5)


def test_explicit_values_equal_to_the_schema_defaults_stay_and_skip_the_status_read(
    client, monkeypatch
):
    backend = _use(monkeypatch, _ModelBackend(DEV))
    response = client.post(
        "/api/inference/images/generate", json = {"prompt": "p", "steps": 9, "guidance": 0.0}
    )
    assert _recipe(response) == (9, 0.0)
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (9, 0.0)
    assert backend.status_calls == 0
    assert backend.calls[0]["expected_load"] is None


def test_a_replacement_retry_uses_the_new_models_defaults_and_pins_it(client, monkeypatch):
    # Turbo is swapped for FLUX.1-dev between the status read and generate (#9448).
    backend = _use(monkeypatch, _ModelBackend(TURBO, DEV))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert [(c["steps"], c["guidance"]) for c in backend.calls] == [(8, 0.0), (28, 3.5)]
    assert backend.calls[0]["expected_load"] == load_identity(
        "unsloth/Z-Image-Turbo", None, "z-image"
    )
    assert backend.calls[1]["expected_load"] == load_identity("unsloth/FLUX.1-dev", None, "z-image")
    assert _recipe(response) == (28, 3.5)


def test_external_provider_requests_never_read_status(client, monkeypatch):
    backend = _use(monkeypatch, _ModelBackend(DEV))

    async def external(request, http_request):
        raise HTTPException(status_code = 418, detail = (request.steps, request.guidance))

    monkeypatch.setattr(inference_routes, "_external_image_generation", external)
    response = client.post(
        "/api/inference/images/generate", json = {"prompt": "p", "provider_id": "openai-1"}
    )
    assert response.status_code == 418
    assert response.json()["detail"] == [9, 0.0]
    assert backend.status_calls == 0 and backend.calls == []


def test_an_unloaded_model_is_still_a_409_with_no_fill(client, monkeypatch):
    backend = _use(monkeypatch, _ModelBackend(DEV, loaded = False))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert response.status_code == 409
    assert (backend.calls[0]["steps"], backend.calls[0]["guidance"]) == (9, 0.0)
    assert backend.calls[0]["expected_load"] == load_identity(None, None, None)


class _LoadingBackend(_ModelBackend):
    """Reads unloaded while another client's load runs; generate is admitted once it commits."""

    def generate(self, **kwargs):
        self.loaded = True
        return super().generate(**kwargs)


def test_a_load_in_flight_hands_the_retry_the_new_models_defaults(client, monkeypatch):
    backend = _use(monkeypatch, _LoadingBackend(DEV, loaded = False))
    response = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert [(c["steps"], c["guidance"]) for c in backend.calls] == [(9, 0.0), (28, 3.5)]
    assert backend.calls[0]["expected_load"] == load_identity(None, None, None)
    assert backend.calls[1]["expected_load"] == load_identity("unsloth/FLUX.1-dev", None, "z-image")
    assert _recipe(response) == (28, 3.5)
