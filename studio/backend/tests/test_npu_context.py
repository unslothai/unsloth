# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto context for NPU loads. Free RAM and config.json are pinned: no test reads this host."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.inference import npu_backend as nb
from core.inference import npu_context as nc
from core.inference.flm_files import FlmFile, FlmModelFiles

_GIB = 1024

# The text fields of FastFlowLM's own config.json for each model (FastFlowLM 1.0.12 packages).
_QWEN35_9B = {
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "num_attention_heads": 16,
    "head_dim": 256,
    "full_attention_interval": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
}
_QWEN35_9B["layer_types"] = _QWEN35_9B["layer_types"] * 8
_QWEN36_35B = {
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "num_attention_heads": 16,
    "head_dim": 256,
    "full_attention_interval": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
}
_QWEN38_27B = {
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "num_attention_heads": 24,
    "head_dim": 256,
    "full_attention_interval": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
}


@pytest.fixture(autouse = True)
def free_ram(monkeypatch):
    """Pin the free-RAM reading; a test that never sets it fails if sizing reads it."""
    state = {"mib": None, "reads": 0}

    def _read():
        state["reads"] += 1
        if state["mib"] is None:
            raise AssertionError("sizing read the host's RAM")
        return state["mib"]

    monkeypatch.setattr(nc, "_available_mib", _read)
    return state


def _model(model_id, checkpoint, size_gb, reported):
    return nb.NpuModel(
        id = model_id,
        checkpoint = checkpoint,
        size_gb = size_gb,
        downloaded = True,
        labels = ("chat",),
        max_context_length = nb._max_context_length(checkpoint, reported),
    )


def _files(tmp_path: Path, config) -> FlmModelFiles:
    folder = tmp_path / "model"
    folder.mkdir()
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return FlmModelFiles(folder = folder, files = (FlmFile("model.q4nx", "u", 1, "d", True),))


QWEN35_9B = ("qwen3.5-9b-FLM", "qwen3.5:9b", 7.6)
QWEN36_35B = ("qwen3.6-moe-35b-a3b-FLM", "qwen3.6-moe:35b-a3b", 20.6)
QWEN38_27B = ("qwen3.8-mtp-27b-FLM", "qwen3.8-mtp:27b", 16.8)
QWEN38_DFLASH2 = ("qwen3.8-dflash2-27b-FLM", "qwen3.8-dflash2:27b", 18.1)


@pytest.mark.parametrize(
    "config, expected",
    [(_QWEN35_9B, 32768), (_QWEN36_35B, 20480), (_QWEN38_27B, 65536)],
)
def test_kv_counts_only_full_attention_layers_in_bf16(tmp_path, config, expected):
    # FastFlowLM logs "a saved row costs 32 KiB" for qwen3.5:4b: 8 full-attention layers, 4 KV heads.
    assert nc.kv_bytes_per_token(_files(tmp_path, config).folder / "config.json") == expected


def test_kv_reads_the_interval_or_every_layer_when_types_are_absent(tmp_path):
    hybrid = {k: v for k, v in _QWEN36_35B.items() if k != "layer_types"}
    assert nc.kv_bytes_per_token(_files(tmp_path, hybrid).folder / "config.json") == 20480
    dense = {
        "text_config": {
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 2048,
        }
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(dense), encoding = "utf-8")
    assert nc.kv_bytes_per_token(path) == 28 * 2 * 8 * 128 * 2


@pytest.mark.parametrize("text", [None, "not json", "[]", '{"num_hidden_layers": 0}'])
def test_kv_is_unknown_for_a_missing_or_unreadable_config(tmp_path, text):
    path = tmp_path / "config.json"
    if text is not None:
        path.write_text(text, encoding = "utf-8")
    assert nc.kv_bytes_per_token(path) is None


def test_a_9b_on_a_roomy_laptop_gets_far_more_than_8k(tmp_path, free_ram):
    free_ram["mib"] = int(23.9 * _GIB)
    model = _model(*QWEN35_9B, 262144)
    plan = nc.plan_context(model, _files(tmp_path, _QWEN35_9B), None)
    assert plan.context_length == nc.AUTO_CONTEXT_CEILING == 65536
    assert plan.fits == 262144


def test_auto_stops_at_the_models_own_limit(tmp_path, free_ram):
    free_ram["mib"] = 100 * _GIB
    model = _model(*QWEN35_9B, 32768)
    assert nc.plan_context(model, _files(tmp_path, _QWEN35_9B), None).context_length == 32768


@pytest.mark.parametrize("free_gib", [20.0, 23.9])
def test_the_35b_on_a_32gb_laptop_is_not_sized_to_32k(tmp_path, free_ram, free_gib):
    # Field report: 31.8 GiB visible, qwen3.6-moe-35b-a3b failed to load at 32k and loaded at 8k.
    free_ram["mib"] = int(free_gib * _GIB)
    model = _model(*QWEN36_35B, 262144)
    plan = nc.plan_context(model, _files(tmp_path, _QWEN36_35B), None)
    assert plan.context_length == 8192
    assert plan.fits == 8192


def test_the_35b_grows_once_the_ram_is_there(tmp_path, free_ram):
    free_ram["mib"] = 60 * _GIB
    model = _model(*QWEN36_35B, 262144)
    plan = nc.plan_context(model, _files(tmp_path, _QWEN36_35B), None)
    assert plan.context_length == 65536


def test_a_missing_config_keeps_8192_without_reading_ram(tmp_path, free_ram):
    model = _model(*QWEN35_9B, 262144)
    plan = nc.plan_context(model, _files(tmp_path, None), None)
    assert plan == nc.ContextPlan(context_length = 8192, fits = None)
    plan = nc.plan_context(model, None, None)
    assert plan == nc.ContextPlan(context_length = 8192, fits = None)
    assert free_ram["reads"] == 0


def test_unreadable_ram_keeps_8192(tmp_path, free_ram, monkeypatch):
    monkeypatch.setattr(nc, "_available_mib", lambda: None)
    model = _model(*QWEN35_9B, 262144)
    assert nc.plan_context(model, _files(tmp_path, _QWEN35_9B), None).context_length == 8192


@pytest.mark.parametrize("free_gib", [23.9, 200.0])
@pytest.mark.parametrize("entry", [QWEN38_27B, QWEN38_DFLASH2])
def test_the_qwen38_engine_cap_holds(tmp_path, free_ram, free_gib, entry):
    free_ram["mib"] = int(free_gib * _GIB)
    model = _model(*entry, 262144)
    assert model.max_context_length == 8192
    files = _files(tmp_path, _QWEN38_27B)
    assert nc.plan_context(model, files, None).context_length == 8192
    assert nc.plan_context(model, files, 16384).context_length == 8192


def _files_again(tmp_path: Path, config) -> FlmModelFiles:
    sub = tmp_path / "again"
    sub.mkdir(exist_ok = True)
    return _files(sub, config)


def test_a_pin_wins_and_a_pin_that_will_not_fit_is_logged(tmp_path, free_ram, caplog):
    free_ram["mib"] = int(23.9 * _GIB)
    model = _model(*QWEN36_35B, 262144)
    with caplog.at_level(logging.WARNING, logger = nc.__name__):
        plan = nc.plan_context(model, _files(tmp_path, _QWEN36_35B), 32768)
    assert plan.context_length == 32768
    assert plan.fits == 8192
    assert "32768" in caplog.text and "8192" in caplog.text


def test_a_pin_that_fits_is_not_logged_and_is_capped_at_the_limit(tmp_path, free_ram, caplog):
    free_ram["mib"] = int(23.9 * _GIB)
    model = _model(*QWEN35_9B, 32768)
    with caplog.at_level(logging.WARNING, logger = nc.__name__):
        plan = nc.plan_context(model, _files(tmp_path, _QWEN35_9B), 100_000)
    assert plan.context_length == 32768
    assert caplog.text == ""


def test_the_resident_models_weights_count_as_free(tmp_path, free_ram):
    # Loading over a resident NPU model: lemond frees it first, so its weights are not lost to the budget.
    free_ram["mib"] = int((23.9 - 7.6) * _GIB)
    model = _model(*QWEN35_9B, 262144)
    previous = _model(*QWEN35_9B, 262144)
    alone = nc.plan_context(model, _files(tmp_path, _QWEN35_9B), None)
    swapped = nc.plan_context(model, _files_again(tmp_path, _QWEN35_9B), None, previous)
    assert swapped.fits > alone.fits
    assert swapped.context_length == 65536


class _Response:
    def __init__(
        self,
        body,
        status_code = 200,
    ):
        self._body, self.status_code = body, status_code

    def json(self):
        return self._body


class _Server:
    def __init__(self):
        self.loads: list[dict] = []
        self.ctx = None

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
            self.loads.append(json_body)
            self.ctx = json_body["ctx_size"]
            return _Response({"status": "success"})
        rows = [
            {
                "model_name": self.loads[-1]["model_name"],
                "device": "npu",
                "recipe_options": {"ctx_size": self.ctx},
            }
        ]
        return _Response({"all_models_loaded": rows})

    def stop(self):
        pass


@pytest.fixture
def backend(tmp_path, monkeypatch):
    npu = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    server = _Server()
    monkeypatch.setattr(npu, "_ensure_running", lambda: server)
    monkeypatch.setattr(npu, "_flm_binary", lambda: None)
    npu.server = server
    return npu


def _serve(backend, monkeypatch, tmp_path, spec, reported, config):
    model = _model(*spec, reported)
    files = _files(tmp_path, config)
    monkeypatch.setattr(backend, "_model", lambda model_id: model)
    monkeypatch.setattr(backend, "_model_files", lambda checkpoint, binary: files)
    return model


def test_load_sends_the_auto_context_and_records_the_fit(backend, monkeypatch, tmp_path, free_ram):
    free_ram["mib"] = int(23.9 * _GIB)
    _serve(backend, monkeypatch, tmp_path, QWEN35_9B, 262144, _QWEN35_9B)
    backend.load("qwen3.5-9b-FLM")
    assert backend.server.loads[-1]["ctx_size"] == 65536
    assert backend._loaded.context_length == 65536
    assert backend._loaded.requested_context_length is None
    assert backend._loaded.fitted_context_length == 262144


def test_load_keeps_a_pin_and_8192_without_a_config(backend, monkeypatch, tmp_path, free_ram):
    _serve(backend, monkeypatch, tmp_path, QWEN36_35B, 262144, None)
    backend.load("qwen3.6-moe-35b-a3b-FLM")
    assert backend.server.loads[-1]["ctx_size"] == 8192
    backend.load("qwen3.6-moe-35b-a3b-FLM", 16384)
    assert backend.server.loads[-1]["ctx_size"] == 16384
    assert backend._loaded.requested_context_length == 16384
    assert free_ram["reads"] == 0


@pytest.mark.parametrize("fitted, expected", [(8192, 8192), (None, 262144)])
def test_the_load_response_reports_the_fit_as_the_ceiling(fitted, expected):
    from routes import inference as routes

    model = _model(*QWEN36_35B, 262144)
    resident = nb.NpuResident(
        model = model,
        context_length = 8192,
        requested_context_length = None,
        base_url = "http://127.0.0.1:1",
        api_key = "key",
        fitted_context_length = fitted,
    )
    response = routes._npu_load_response(resident, "loaded", None)
    assert response.context_length == 8192
    assert response.max_context_length == expected
    assert response.native_context_length == 262144
