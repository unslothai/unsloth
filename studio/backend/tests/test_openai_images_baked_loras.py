# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /v1/images/generations keeps the LoRAs a quantized load baked in; backend faked, so no GPU."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

import core.inference.diffusion as diffusion_module
import core.inference.image_gallery as gallery_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_openai_images_generations_route import _FakeBackend, _make_client  # noqa: E402

BAKED = [{"id": "tarot-lora", "weight": 0.8}]


class _BakedBackend(_FakeBackend):
    """A diffusers status with adapters baked at load; generate records the LoRA set it was given."""

    def __init__(
        self,
        baked = BAKED,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._baked = baked
        self.loras_seen = []

    def status(self):
        out = super().status()
        out["baked_lora_specs"] = list(self._baked)
        return out

    def generate(
        self,
        *,
        loras = None,
        # The diffusers engine takes the adapter pin; the strict base fake does not.
        expected_load_loras = None,
        **kwargs,
    ):
        self.loras_seen.append(loras)
        return super().generate(**kwargs)


@pytest.fixture
def baked_client(monkeypatch):
    backend = _BakedBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)
    cli.backend, cli.store = backend, store  # type: ignore[attr-defined]
    return cli


def _post(client, body):
    return client.post("/v1/images/generations", json = {"size": "256x256", **body})


def test_omitted_loras_render_with_the_baked_set(baked_client):
    resp = _post(baked_client, {"prompt": "trtcrd a sloth", "response_format": "b64_json"})
    assert resp.status_code == 200, resp.text
    assert baked_client.backend.loras_seen == [[("tarot-lora", 0.8)]]
    (record,) = baked_client.store.values()
    assert record["loras"] == ["tarot-lora:0.8"]


def test_explicit_empty_loras_render_without_adapters(baked_client):
    resp = _post(baked_client, {"prompt": "a sloth", "loras": [], "response_format": "b64_json"})
    assert resp.status_code == 200, resp.text
    assert baked_client.backend.loras_seen == [None]
    (record,) = baked_client.store.values()
    assert record["loras"] == []


def test_explicit_loras_override_the_baked_set(baked_client):
    body = {"prompt": "a sloth", "loras": [{"id": "tarot-lora", "weight": 0.5}]}
    resp = _post(baked_client, {**body, "response_format": "b64_json"})
    assert resp.status_code == 200, resp.text
    assert baked_client.backend.loras_seen == [[("tarot-lora", 0.5)]]


def test_duplicate_lora_ids_are_refused(baked_client):
    body = {"prompt": "a sloth", "loras": [{"id": "a"}, {"id": "a"}]}
    assert _post(baked_client, body).status_code in (400, 422)


def test_load_without_baked_adapters_keeps_the_old_call_shape(monkeypatch):
    backend = _BakedBackend(baked = [])
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)
    resp = cli.post(
        "/v1/images/generations",
        json = {"prompt": "a sloth", "size": "256x256", "response_format": "b64_json"},
    )
    assert resp.status_code == 200, resp.text
    assert backend.loras_seen == [None]


def test_status_reports_the_load_request_behind_a_bake():
    pipe = types.SimpleNamespace()
    diffusion_module._record_baked_lora_specs(pipe, [("tarot-lora", 0.8), ("off", 0.0)])
    assert diffusion_module._baked_lora_specs(pipe) == []
    pipe._unsloth_loras_baked = True
    assert diffusion_module._baked_lora_specs(pipe) == [{"id": "tarot-lora", "weight": 0.8}]


def _baked_quant_state():
    pipe = types.SimpleNamespace(
        _unsloth_loras = (("tarot", "/loras/tarot.safetensors", 0.8),),
        _unsloth_loras_baked = True,
    )
    diffusion_module._record_baked_lora_specs(pipe, [("tarot-lora", 0.8)])
    return types.SimpleNamespace(pipe = pipe, transformer_quant = "int8", family = None, hf_token = None)


def test_reporting_the_baked_specs_back_keeps_the_bake_on(monkeypatch):
    # Omitted loras: the route sends the baked set unchanged.
    backend = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    monkeypatch.setattr(
        backend,
        "_resolve_lora_set",
        lambda specs, **_: tuple(("tarot", "/loras/tarot.safetensors", w) for (_i, w) in specs),
        raising = False,
    )
    monkeypatch.setattr(diffusion_module, "_refuse_lora_without_loader", lambda *a, **k: None)
    state = _baked_quant_state()
    pairs = [(s["id"], s["weight"]) for s in diffusion_module._baked_lora_specs(state.pipe)]
    backend._apply_loras(state, pairs, threading.Event())
    assert state.pipe._unsloth_loras == (("tarot", "/loras/tarot.safetensors", 0.8),)


def test_a_selection_refusal_is_a_400_and_other_value_errors_stay_500(baked_client, monkeypatch):
    from core.inference.diffusion_lora import LoraSelectionError

    def refuse(**_kwargs):
        raise LoraSelectionError(
            "The LoRA selection changed. Reload the model with the new adapter selection."
        )

    monkeypatch.setattr(baked_client.backend, "generate", refuse)
    resp = _post(baked_client, {"prompt": "a sloth", "loras": [{"id": "other", "weight": 1}]})
    assert resp.status_code == 400, resp.text
    assert "Reload the model" in resp.text and "loras" in resp.text

    def broken(**_kwargs):
        raise ValueError("internal detail")

    monkeypatch.setattr(baked_client.backend, "generate", broken)
    resp = _post(baked_client, {"prompt": "a sloth", "loras": [{"id": "other", "weight": 1}]})
    assert resp.status_code == 500 and "internal detail" not in resp.text


def test_managed_accounts_authorize_explicit_adapters(baked_client, monkeypatch):
    from fastapi import HTTPException

    from hub.services.models import account_access

    calls = []

    def deny(request):
        calls.append([spec.id for spec in request.loras])
        raise HTTPException(status_code = 403, detail = "adapter not allowed")

    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "require_media_generation_access", lambda *_a, **_k: None)
    monkeypatch.setattr(account_access, "require_media_adapters", deny)
    resp = _post(
        baked_client, {"prompt": "a sloth", "loras": [{"id": "private-hub-lora", "weight": 1}]}
    )
    assert resp.status_code == 403 and calls == [["private-hub-lora"]]
    assert baked_client.backend.loras_seen == []
    # The baked set was authorized by its load; omitting loras does not re-check it.
    resp = _post(baked_client, {"prompt": "a sloth", "response_format": "b64_json"})
    assert resp.status_code == 200 and calls == [["private-hub-lora"]]


class _DenseLoadBackend(_BakedBackend):
    def status(self):
        out = super().status()
        out["load_lora_specs"] = [{"id": "tarot-lora", "weight": 1.0}]
        return out


def test_omitted_loras_keep_a_dense_deploys_adapters(monkeypatch):
    backend = _DenseLoadBackend(baked = [])
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    cli, store, _save = _make_client(backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)
    resp = cli.post(
        "/v1/images/generations",
        json = {"prompt": "trtcrd a sloth", "size": "256x256", "response_format": "b64_json"},
    )
    assert resp.status_code == 200, resp.text
    assert backend.loras_seen == [[("tarot-lora", 1.0)]]
    resp = cli.post(
        "/v1/images/generations",
        json = {"prompt": "a sloth", "size": "256x256", "loras": [], "response_format": "b64_json"},
    )
    assert resp.status_code == 200, resp.text
    assert backend.loras_seen[-1] is None


def test_load_lora_specs_report_the_load_request():
    pipe = types.SimpleNamespace()
    assert diffusion_module._load_lora_specs(pipe) == []
    diffusion_module._record_load_lora_specs(pipe, [("tarot-lora", 1.0), ("off", 0.0)])
    assert diffusion_module._load_lora_specs(pipe) == [{"id": "tarot-lora", "weight": 1.0}]
    pipe._unsloth_loras_baked = True
    diffusion_module._record_baked_lora_specs(pipe, [("baked-lora", 0.5)])
    assert diffusion_module._load_lora_specs(pipe) == [{"id": "baked-lora", "weight": 0.5}]


def _diffusion_functions():
    import ast
    src = Path(diffusion_module.__file__).read_text(encoding = "utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_every_bake_records_the_specs_v1_reads_back():
    """Unrecorded baked specs make /v1/images/generations send an empty selection (refused on MiniMax-H3-Image)."""
    import ast
    for node in _diffusion_functions():
        marks = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            and any(ast.unparse(t).endswith("._unsloth_loras_baked") for t in n.targets)
            and isinstance(n.value, ast.Constant)
            and n.value.value is True
        ]
        if not marks:
            continue
        calls = {ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
        assert (
            "_record_baked_lora_specs" in calls
        ), f"{node.name} bakes LoRAs without recording their specs"


def test_dense_loads_record_their_requested_adapters():
    import ast
    for node in _diffusion_functions():
        if node.name == "load_pipeline":
            calls = {ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
            assert "_record_load_lora_specs" in calls
            return
    raise AssertionError("load_pipeline not found")
