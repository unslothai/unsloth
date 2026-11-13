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
from core.inference import npu_memory
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
    monkeypatch.setattr(nc, "_npu_window_gb", lambda: state.get("window_gb"))
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
    assert nc._kv_layout(_files(tmp_path, config).folder / "config.json").full_row_bytes == expected


def test_kv_reads_the_interval_or_every_layer_when_types_are_absent(tmp_path):
    hybrid = {k: v for k, v in _QWEN36_35B.items() if k != "layer_types"}
    assert nc._kv_layout(_files(tmp_path, hybrid).folder / "config.json").full_row_bytes == 20480
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
    assert nc._kv_layout(path).full_row_bytes == 28 * 2 * 8 * 128 * 2


@pytest.mark.parametrize("text", [None, "not json", "[]", '{"num_hidden_layers": 0}'])
def test_kv_is_unknown_for_a_missing_or_unreadable_config(tmp_path, text):
    path = tmp_path / "config.json"
    if text is not None:
        path.write_text(text, encoding = "utf-8")
    assert nc._kv_layout(path) is None


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
    if free_gib == 20.0:
        assert plan.fits == 0
    else:
        assert 0 < plan.fits < 8192


def test_the_35b_grows_once_the_ram_is_there(tmp_path, free_ram):
    free_ram["mib"] = 60 * _GIB
    model = _model(*QWEN36_35B, 262144)
    plan = nc.plan_context(model, _files(tmp_path, _QWEN36_35B), None)
    assert plan.context_length == 65536


def test_rounding_prices_the_active_allocation_for_auto_and_pins(tmp_path, free_ram, caplog):
    # Qwen3 0.6B: 112 KiB per token. 17,408 rows would cost exactly 6,000 MiB
    # including the prefix cache, but the active allocation jumps to 32,768.
    config = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    model = _model("qwen3-0.6b-FLM", "qwen3:0.6b", 1, 40960)
    files = _files(tmp_path, config)
    free_ram["mib"] = 1024 + 2048 + 6000
    assert nc.plan_context(model, files, None) == nc.ContextPlan(16384, 16384)
    with caplog.at_level(logging.WARNING, logger = nc.__name__):
        assert nc.plan_context(model, files, 16385) == nc.ContextPlan(16385, 16384)
    assert "above the 16384 tokens" in caplog.text


# Every chat tag whose v1.0.12 model card (or qwen3_core_backend.hpp) caps it at 32k.
@pytest.mark.parametrize(
    "tag,reported",
    [
        ("qwen3:0.6b", 40960),
        ("qwen3-core:8b", 40960),
        ("lfm2:1.2b", 128000),
        ("lfm2:2.6b", 128000),
        ("lfm2-trans:2.6b", 128000),
        ("lfm2.5-it:1.2b", 128000),
        ("lfm2.5-tk:1.2b", 128000),
        ("lfm3:1.7b", 128000),
    ],
)
def test_roomy_machine_and_explicit_pin_respect_the_engine_limit(tmp_path, free_ram, tag, reported):
    model = _model("model-FLM", tag, 1, reported)
    files = _files(tmp_path, _QWEN35_9B)
    free_ram["mib"] = 100 * _GIB
    for request in (None, 65536):
        assert nc.plan_context(model, files, request) == nc.ContextPlan(32768, 32768)
    assert nb._max_context_length(tag, 16384) == 16384


@pytest.mark.parametrize("explicit_types", [False, True])
def test_gemma_sliding_window_stops_growing(tmp_path, free_ram, explicit_types):
    config = {
        "num_hidden_layers": 34,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
    }
    if explicit_types:
        config["layer_types"] = [
            "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(34)
        ]
    files = _files(tmp_path, config)
    kv = nc._kv_layout(files.folder / "config.json")
    assert kv.bytes_for(65536) == 20 * 1024 * 65536 + 116 * 1024**2
    assert nc._cost(65536, kv) == kv.bytes_for(65536) + 4096 * 1024**2
    model = _model("gemma3-4b-FLM", "gemma3:4b", 3, 131072)
    free_ram["mib"] = (8 + 3 + 2) * _GIB
    assert nc.plan_context(model, files, None).context_length == 65536


def test_gemma4_shared_layers_reuse_kv_and_global_heads_can_be_wider(tmp_path):
    # Gemma 4 E2B: three global and twelve sliding layers own buffers; twenty reuse them.
    config = {
        "model_type": "gemma4_text",
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_kv_shared_layers": 20,
        "sliding_window": 512,
        "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
    }
    kv = nc._kv_layout(_files(tmp_path, config).folder / "config.json")
    assert kv.full_row_bytes == 6 * 1024
    assert kv.bytes_for(65536) == 65536 * 6 * 1024 + 6 * 1024**2
    assert nc._cost(1024, kv) == kv.bytes_for(4096) + 8 * kv.bytes_for(1024)


def test_lfm_only_charges_its_explicit_attention_layers(tmp_path):
    config = {
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "full_attn_idxs": [2, 5, 8, 10, 12, 14],
    }
    assert (
        nc._kv_layout(_files(tmp_path, config).folder / "config.json").full_row_bytes == 12 * 1024
    )


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


def test_a_pin_wins_and_a_pin_that_will_not_fit_is_logged(tmp_path, free_ram, caplog):
    free_ram["mib"] = int(23.9 * _GIB)
    model = _model(*QWEN36_35B, 262144)
    with caplog.at_level(logging.WARNING, logger = nc.__name__):
        plan = nc.plan_context(model, _files(tmp_path, _QWEN36_35B), 32768)
    assert plan.context_length == 32768
    assert 0 < plan.fits < 8192
    assert f"above the {plan.fits} tokens" in caplog.text


def test_a_pin_that_fits_is_not_logged_and_is_capped_at_the_limit(tmp_path, free_ram, caplog):
    free_ram["mib"] = int(23.9 * _GIB)
    model = _model(*QWEN35_9B, 32768)
    with caplog.at_level(logging.WARNING, logger = nc.__name__):
        plan = nc.plan_context(model, _files(tmp_path, _QWEN35_9B), 100_000)
    assert plan.context_length == 32768
    assert caplog.text == ""


@pytest.mark.parametrize(
    "entry, config, total_gb, free_gib, expected, fits",
    [
        # 24 GB window less 21.6 GB of weights and load margin leaves 2.4 GB for KV.
        (QWEN36_35B, _QWEN36_35B, 48, 40.0, 13312, 13680),
        # 12 GB window less 8.6 GB leaves 3.4 GB.
        (QWEN35_9B, _QWEN35_9B, 24, 18.0, 11264, 11878),
    ],
)
def test_windows_auto_fits_the_npu_window_not_all_free_ram(
    tmp_path, free_ram, entry, config, total_gb, free_gib, expected, fits
):
    free_ram["mib"] = int(free_gib * _GIB)
    model = _model(*entry, 262144)
    files = _files(tmp_path, config)
    assert nc.plan_context(model, files, None).context_length == 65536
    free_ram["window_gb"] = npu_memory.npu_memory_window_gb(total_gb, "win32")
    assert nc.plan_context(model, files, None) == nc.ContextPlan(expected, fits)
    budget = nc._npu_window_bytes(int(entry[2] * nc._GIB))
    kv = nc._kv_layout(files.folder / "config.json")
    assert nc._cost(fits, kv) <= budget < nc._cost(fits + 1, kv)


def test_strix_halo_linux_keeps_the_65536_ceiling(tmp_path, free_ram):
    # Measured: Linux Strix Halo, 125 GiB, qwen3.6-moe-35b-a3b loaded at the 65,536 Auto ceiling.
    free_ram["mib"] = 110 * _GIB
    free_ram["window_gb"] = npu_memory.npu_memory_window_gb(125, "linux")
    model = _model(*QWEN36_35B, 262144)
    assert nc.plan_context(model, _files(tmp_path, _QWEN36_35B), None).context_length == 65536


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

    def is_alive(self):
        return True

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
        if path == "/v1/unload":
            self.ctx = None
            return _Response({"status": "success"})
        if path == "/v1/load":
            self.loads.append(json_body)
            self.ctx = json_body["ctx_size"]
            return _Response({"status": "success"})
        if not self.loads:
            return _Response({"all_models_loaded": []})
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
    npu._server = server
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
    assert backend._server.loads[-1]["ctx_size"] == 65536
    assert backend._loaded.context_length == 65536
    assert backend._loaded.requested_context_length is None
    assert backend._loaded.fitted_context_length == 262144


def test_auto_reload_samples_free_ram_after_releasing_weights_and_caches(
    backend, monkeypatch, tmp_path
):
    model = _serve(backend, monkeypatch, tmp_path, QWEN36_35B, 262144, _QWEN36_35B)
    # With an active 64k KV buffer and a populated prefix cache the same host has
    # only 3.15 GiB available until Lemonade unloads this 20.6 GiB model.
    monkeypatch.setattr(
        nc, "_available_mib", lambda: int((29 if backend._server.ctx is None else 3.15) * _GIB)
    )
    backend.load(model.id)
    backend.load(model.id)
    assert [load["ctx_size"] for load in backend._server.loads] == [65536, 65536]


def test_a_windows_reload_releases_the_resident_and_still_caps_to_the_npu_window(
    backend, monkeypatch, tmp_path
):
    model = _serve(backend, monkeypatch, tmp_path, QWEN36_35B, 262144, _QWEN36_35B)
    monkeypatch.setattr(
        nc, "_available_mib", lambda: int((29 if backend._server.ctx is None else 3.15) * _GIB)
    )
    monkeypatch.setattr(nc, "_npu_window_gb", lambda: npu_memory.npu_memory_window_gb(48, "win32"))
    backend.load(model.id)
    backend.load(model.id)
    assert [load["ctx_size"] for load in backend._server.loads] == [13312, 13312]


def test_load_keeps_a_pin_and_8192_without_a_config(backend, monkeypatch, tmp_path, free_ram):
    _serve(backend, monkeypatch, tmp_path, QWEN36_35B, 262144, None)
    backend.load("qwen3.6-moe-35b-a3b-FLM")
    assert backend._server.loads[-1]["ctx_size"] == 8192
    backend.load("qwen3.6-moe-35b-a3b-FLM", 16384)
    assert backend._server.loads[-1]["ctx_size"] == 16384
    assert backend._loaded.requested_context_length == 16384
    assert free_ram["reads"] == 0


@pytest.mark.parametrize("fitted, expected", [(0, 0), (4096, 4096), (None, 262144)])
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
    response = routes._npu_load_response(resident, "loaded")
    assert response.context_length == 8192
    assert response.npu_context_fit == fitted
    assert response.max_context_length == expected
    assert response.native_context_length == 262144
