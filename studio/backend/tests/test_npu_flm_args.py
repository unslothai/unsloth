# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM run settings: built into Lemonade's ``flm_args``, sent with the load, and compared on reload."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from core.inference import npu_backend as nb
from core.inference.npu_flm_args import build_flm_args, flm_args_status

_TEXT = nb.NpuModel(
    id = "qwen3-0.6b-FLM",
    checkpoint = "qwen3:0.6b",
    size_gb = 0.66,
    downloaded = True,
    labels = ("chat",),
    max_context_length = 40960,
)
_VISION = nb.NpuModel(
    id = "qwen3.5-9b-FLM",
    checkpoint = "qwen3.5:9b",
    size_gb = 6.0,
    downloaded = True,
    labels = ("chat", "vision"),
    max_context_length = 32768,
)


def _flm(**fields):
    return build_flm_args(
        power_mode = fields.get("power_mode"),
        prefill_chunk_len = fields.get("prefill_chunk_len"),
        img_pre_resize = fields.get("img_pre_resize"),
        ctx_size = fields.get("ctx_size", 8192),
    )


def test_defaults_send_nothing():
    assert _flm() is None
    assert _flm(power_mode = "default") is None


def test_flags_use_the_canonical_form_lemonade_accepts():
    assert _flm(power_mode = "turbo") == "--pmode turbo"
    assert (
        _flm(power_mode = "powersaver", prefill_chunk_len = 1024, img_pre_resize = 0)
        == "--pmode powersaver --prefill-chunk-len 1024 --img-pre-resize 0"
    )


def test_a_prefill_chunk_past_the_context_is_clamped_to_it():
    assert _flm(prefill_chunk_len = 16384, ctx_size = 4096) == "--prefill-chunk-len 4096"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"power_mode": "ludicrous"}, "Power mode"),
        ({"power_mode": "turbo --socket 9"}, "Power mode"),
        ({"prefill_chunk_len": 0}, "Prefill chunk length"),
        ({"prefill_chunk_len": True}, "Prefill chunk length"),
        ({"img_pre_resize": 5}, "Image pre-resize"),
        ({"img_pre_resize": -1}, "Image pre-resize"),
    ],
)
def test_invalid_values_are_refused(fields, message):
    with pytest.raises(ValueError, match = message):
        _flm(**fields)


def test_status_reads_the_flags_back():
    assert flm_args_status(None) == {
        "npu_power_mode": None,
        "npu_prefill_chunk_len": None,
        "npu_img_pre_resize": None,
    }
    assert flm_args_status("--pmode turbo --prefill-chunk-len 512 --img-pre-resize 2") == {
        "npu_power_mode": "turbo",
        "npu_prefill_chunk_len": 512,
        "npu_img_pre_resize": 2,
    }


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"status": "success"}


class _Server:
    port = 1
    api_key = "key"

    def __init__(self):
        self.bodies = []

    def request(
        self,
        method,
        path,
        json_body = None,
        timeout = None,
    ):
        if path == "/v1/load":
            self.bodies.append(json_body)
        return _Response()

    def is_alive(self):
        return True

    def stop(self):
        pass


@pytest.fixture
def backend(tmp_path, monkeypatch):
    npu = nb.LemonadeNpuBackend(root = tmp_path)
    server = _Server()
    monkeypatch.setattr(npu, "_ensure_running", lambda: server)
    monkeypatch.setattr(npu, "_model", lambda model_id: _VISION)
    monkeypatch.setattr(nb.LemonadeNpuBackend, "_resident_context", staticmethod(lambda s, m: 8192))
    npu._server = server
    return npu, server


def test_the_load_body_carries_flm_args_only_when_set(backend):
    npu, server = backend
    npu.load(_VISION.id, 8192)
    assert server.bodies[-1] == {"model_name": _VISION.id, "ctx_size": 8192, "save_options": False}
    assert npu._loaded.flm_args is None

    npu.load(_VISION.id, 8192, flm_args = "--pmode turbo")
    assert server.bodies[-1] == {
        "model_name": _VISION.id,
        "ctx_size": 8192,
        "save_options": False,
        "flm_args": "--pmode turbo",
    }
    assert npu._loaded.flm_args == "--pmode turbo"


def _route_npu(
    model,
    *,
    resident_flm_args = None,
    resident_ctx = None,
):
    class _Npu:
        requested = resident_ctx
        loads: list = []

        def __init__(self):
            self._loaded = nb._Loaded(
                model = model,
                context_length = 8192,
                requested_context_length = resident_ctx,
                flm_args = resident_flm_args,
            )

        def resident(self):
            return nb.NpuResident(
                model = model,
                context_length = 8192,
                requested_context_length = self.requested,
                base_url = "http://127.0.0.1:1",
                api_key = "key",
            )

        def loadable_model(self, model_id):
            return model

        def load(
            self,
            model_id,
            ctx,
            flm_args = None,
        ):
            self.loads.append((ctx, flm_args))
            self.requested = ctx
            self._loaded = nb._Loaded(
                model = model,
                context_length = 8192,
                requested_context_length = ctx,
                flm_args = flm_args,
            )

    return _Npu()


def _load(monkeypatch, npu, **fields):
    from models.inference import LoadRequest
    from routes import inference as routes

    async def _nothing_loaded(_backend):
        return None

    monkeypatch.setattr(nb, "get_npu_backend", lambda: npu)
    monkeypatch.setattr(routes, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(routes, "release_chat_gpu_claim", lambda: True)
    monkeypatch.setattr(routes, "_unload_llama_before_standard_load", _nothing_loaded)
    return asyncio.run(
        routes._load_npu_model(
            LoadRequest(model_path = npu.resident().model.model_path, **fields),
            current_request_counted = False,
            on_reload_confirmed = None,
            load_cancel_event = None,
        )
    )


@pytest.mark.parametrize(
    "resident, fields, expected",
    [
        (None, {}, "already_loaded"),
        ("--pmode turbo", {"npu_power_mode": "turbo"}, "already_loaded"),
        (None, {"npu_power_mode": "default"}, "already_loaded"),
        (None, {"npu_power_mode": "turbo"}, "loaded"),
        ("--pmode turbo", {"npu_power_mode": "balanced"}, "loaded"),
        ("--pmode turbo", {}, "loaded"),
        (None, {"npu_prefill_chunk_len": 1024}, "loaded"),
    ],
)
def test_a_changed_flm_setting_is_a_different_load(monkeypatch, resident, fields, expected):
    npu = _route_npu(_TEXT, resident_flm_args = resident)
    response = _load(monkeypatch, npu, **fields)
    assert response.status == expected
    # The load response reports the settings the resident load runs with, as status does.
    assert {
        field: getattr(response, field)
        for field in ("npu_power_mode", "npu_prefill_chunk_len", "npu_img_pre_resize")
    } == flm_args_status(npu._loaded.flm_args)
    if expected == "loaded":
        assert npu.loads[-1][1] == build_flm_args(
            power_mode = fields.get("npu_power_mode"),
            prefill_chunk_len = fields.get("npu_prefill_chunk_len"),
            img_pre_resize = None,
            ctx_size = 8192,
        )


def test_the_prefill_chunk_is_clamped_to_the_context_the_load_uses(monkeypatch):
    npu = _route_npu(_TEXT, resident_ctx = 2048)
    _load(monkeypatch, npu, max_seq_length = 4096, npu_prefill_chunk_len = 100_000)
    assert npu.loads[-1] == (4096, "--prefill-chunk-len 4096")


def test_image_pre_resize_reaches_vision_models_only(monkeypatch):
    vision = _route_npu(_VISION)
    _load(monkeypatch, vision, npu_img_pre_resize = 1)
    assert vision.loads[-1][1] == "--img-pre-resize 1"
    text = _route_npu(_TEXT, resident_flm_args = "--pmode turbo")
    _load(monkeypatch, text, npu_img_pre_resize = 1)
    assert text.loads[-1][1] is None


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"npu_power_mode": "ludicrous"}, "Power mode"),
        ({"npu_prefill_chunk_len": 0}, "Prefill chunk length"),
        ({"npu_img_pre_resize": 9}, "Image pre-resize"),
        ({"npu_prefill_chunk_len": True}, "Prefill chunk length"),
        ({"npu_img_pre_resize": True}, "Image pre-resize"),
    ],
)
def test_an_invalid_flm_setting_is_a_400_before_any_load(monkeypatch, fields, message):
    npu = _route_npu(_VISION, resident_flm_args = "--pmode turbo")
    with pytest.raises(HTTPException) as exc:
        _load(monkeypatch, npu, **fields)
    assert exc.value.status_code == 400
    assert message in exc.value.detail
    assert npu.loads == []


def test_the_saved_override_keeps_valid_npu_settings_only():
    from utils.openai_auto_switch_settings import normalize_model_override
    assert normalize_model_override(
        {"npu_power_mode": "turbo", "npu_prefill_chunk_len": 2048, "npu_img_pre_resize": 0}
    ) == {"npu_power_mode": "turbo", "npu_prefill_chunk_len": 2048, "npu_img_pre_resize": 0}
    assert (
        normalize_model_override(
            {"npu_power_mode": "default", "npu_prefill_chunk_len": 0, "npu_img_pre_resize": 7}
        )
        == {}
    )


@pytest.mark.parametrize("field", ["npu_prefill_chunk_len", "npu_img_pre_resize"])
def test_the_override_payload_refuses_a_boolean(field):
    from pydantic import ValidationError
    from routes.settings import ModelOverridePayload
    with pytest.raises(ValidationError):
        ModelOverridePayload(model_id = "lemonade:qwen3.5-9b-FLM", **{field: True})


def test_a_save_that_omits_the_npu_settings_keeps_them(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import utils.openai_auto_switch_settings as settings
    from test_openai_auto_switch import _mock_override_store, _put

    _mock_override_store(monkeypatch)
    npu = "lemonade:qwen3.5-9b-FLM"
    _put(npu, npu_power_mode = "turbo", npu_prefill_chunk_len = 2048, npu_img_pre_resize = 0)
    # An older client sends no NPU keys at all.
    _put(npu, max_seq_length = 8192)
    assert settings.get_model_override(npu) == {
        "max_seq_length": 8192,
        "npu_power_mode": "turbo",
        "npu_prefill_chunk_len": 2048,
        "npu_img_pre_resize": 0,
    }
    _put(npu, npu_power_mode = None, npu_prefill_chunk_len = None, npu_img_pre_resize = None)
    assert settings.get_model_override(npu) == {}
