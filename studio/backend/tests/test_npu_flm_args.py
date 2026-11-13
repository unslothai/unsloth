# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM run settings: built into Lemonade's ``flm_args``, sent with the load, and compared on reload."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from core.inference import npu_backend as nb
from core.inference.npu_flm_args import build_flm_args, clamp_prefill_chunk, flm_args_status

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
    flm = "--pmode turbo --prefill-chunk-len 16384"
    assert clamp_prefill_chunk(flm, 4096) == "--pmode turbo --prefill-chunk-len 4096"
    assert clamp_prefill_chunk(flm, 65536) == flm
    assert clamp_prefill_chunk(flm, None) == flm
    assert clamp_prefill_chunk(None, 4096) is None


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
    text = ""

    def __init__(
        self,
        payload = None,
        status_code = 200,
    ):
        self.payload = {"status": "success"} if payload is None else payload
        self.status_code = status_code

    def json(self):
        return self.payload


class _Server:
    port = 1
    api_key = "key"

    def __init__(self):
        self.bodies = []
        self.options = None

    def note(self, message):
        pass

    def open_window(self):
        return None

    def backend_output(self, window):
        import contextlib
        return contextlib.nullcontext()

    def output_mark(self):
        return 0

    def output_since(
        self,
        mark,
        lines = 40,
        settle = 0.5,
    ):
        return []

    def request(
        self,
        method,
        path,
        json_body = None,
        timeout = None,
    ):
        if path == "/v1/load":
            self.bodies.append(json_body)
        if method == "GET" and path.endswith("/options"):
            if self.options is None:
                return _Response({"error": "not found"}, 404)
            return _Response({"effective": {"flm_args": self.options}})
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
    assert npu.resident().flm_args == "--pmode turbo"


def test_the_load_holds_the_prefill_chunk_to_the_context_it_resolves(backend):
    npu, server = backend
    npu.load(_VISION.id, 4096, flm_args = "--prefill-chunk-len 100000")
    assert server.bodies[-1]["flm_args"] == "--prefill-chunk-len 4096"
    assert npu._loaded.flm_args == "--prefill-chunk-len 4096"


def test_a_saved_flag_survives_a_load_with_an_unsloth_studio_setting(backend):
    npu, server = backend
    server.options = "--q-len 4"
    npu.load(_VISION.id, 8192, flm_args = "--pmode turbo")
    assert server.bodies[-1]["flm_args"] == "--pmode turbo --q-len 4"
    # The resident's own settings stay what Unsloth Studio asked for, so a reload compares equal.
    assert npu._loaded.flm_args == "--pmode turbo"


def test_unsloth_studio_wins_where_it_sets_the_same_flag(backend):
    npu, server = backend
    server.options = "--pmode balanced --q-len 4 --prefill-chunk-len 100000"
    npu.load(_VISION.id, 4096, flm_args = "--pmode turbo")
    # The saved chunk is held to the context too: Lemonade refuses a longer one.
    assert server.bodies[-1]["flm_args"] == "--pmode turbo --q-len 4 --prefill-chunk-len 4096"


def test_without_saved_options_the_load_sends_unsloth_studio_flags_only(backend):
    npu, server = backend
    npu.load(_VISION.id, 8192, flm_args = "--pmode turbo")
    assert server.bodies[-1]["flm_args"] == "--pmode turbo"


def test_saved_flags_alone_are_left_to_lemonade(backend):
    npu, server = backend
    server.options = "--q-len 4"
    npu.load(_VISION.id, 8192)
    assert "flm_args" not in server.bodies[-1]


def test_merge_keeps_flag_values_together():
    from core.inference.npu_flm_args import merge_flm_args

    assert merge_flm_args("--q-len 4 --pmode=balanced --x -1", "--pmode turbo") == (
        "--pmode turbo --q-len 4 --x -1"
    )
    assert merge_flm_args("--q-len 4", None) is None
    assert merge_flm_args("", "--pmode turbo") == "--pmode turbo"


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
                flm_args = self._loaded.flm_args,
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
    assert {
        field: getattr(response, field)
        for field in ("npu_power_mode", "npu_prefill_chunk_len", "npu_img_pre_resize")
    } == flm_args_status(npu._loaded.flm_args)
    if expected == "loaded":
        assert npu.loads[-1][1] == build_flm_args(
            power_mode = fields.get("npu_power_mode"),
            prefill_chunk_len = fields.get("npu_prefill_chunk_len"),
            img_pre_resize = None,
        )


def test_the_route_leaves_the_prefill_chunk_to_the_load(monkeypatch):
    # Auto can resolve past 8192, so the route cannot know the context the chunk is held to.
    npu = _route_npu(_TEXT, resident_ctx = 2048)
    _load(monkeypatch, npu, npu_prefill_chunk_len = 16384)
    assert npu.loads[-1] == (None, "--prefill-chunk-len 16384")


def test_a_chunk_the_resident_holds_to_its_context_is_already_loaded(monkeypatch):
    # The resident runs at 8192, so asking for 16384 again is the load it already is.
    npu = _route_npu(_TEXT, resident_flm_args = "--prefill-chunk-len 8192")
    assert _load(monkeypatch, npu, npu_prefill_chunk_len = 16384).status == "already_loaded"
    assert _load(monkeypatch, npu, npu_prefill_chunk_len = 4096).status == "loaded"


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
    _put(npu, max_seq_length = 8192)
    assert settings.get_model_override(npu) == {
        "max_seq_length": 8192,
        "npu_power_mode": "turbo",
        "npu_prefill_chunk_len": 2048,
        "npu_img_pre_resize": 0,
    }
    _put(npu, npu_power_mode = None)
    assert settings.get_model_override(npu) == {
        "npu_prefill_chunk_len": 2048,
        "npu_img_pre_resize": 0,
    }
    _put(npu, npu_power_mode = None, npu_prefill_chunk_len = None, npu_img_pre_resize = None)
    assert settings.get_model_override(npu) == {}
