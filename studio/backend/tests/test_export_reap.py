# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in REAP expert pruning on the export page: MoE detection, the worker step, request models.

Core's FastLanguageModel.reap_prune is stubbed; Core has its own GPU tests for the pruning itself.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import tempfile
import types

import pytest
from pydantic import ValidationError

from core.export import reap as reap_mod
from models.export import (
    ExportBaseModelRequest,
    ExportGGUFRequest,
    ExportMergedModelRequest,
    ReapOptions,
)


def test_moe_info_from_config_layouts():
    assert reap_mod.moe_info_from_config({"num_experts": 128, "num_experts_per_tok": 8}) == {
        "num_experts": 128,
        "top_k": 8,
    }
    # Mixtral / DeepSeek / multimodal text_config / per-layer top_k lists.
    assert (
        reap_mod.moe_info_from_config({"num_local_experts": 8, "num_experts_per_tok": 2})[
            "num_experts"
        ]
        == 8
    )
    assert (
        reap_mod.moe_info_from_config({"n_routed_experts": 64, "num_experts_per_tok": 6})["top_k"]
        == 6
    )
    assert reap_mod.moe_info_from_config(
        {"model_type": "x_vl", "text_config": {"num_experts": 16, "top_k": 2}}
    ) == {"num_experts": 16, "top_k": 2}
    assert reap_mod.moe_info_from_config({"moe_num_experts": 32, "moe_k": [4, 8]})["top_k"] == 8


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"model_type": "llama", "hidden_size": 64},
        {"num_experts": 0, "num_experts_per_tok": 2},
        {"num_experts": 1, "num_experts_per_tok": 1},
        {"num_experts": True, "num_experts_per_tok": 1},
        {"num_experts": 8},
    ],
)
def test_moe_info_from_config_rejects_dense(config):
    assert reap_mod.moe_info_from_config(config) is None


def test_moe_export_info_local_folder_and_lora(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    assert reap_mod.moe_export_info(str(base)) == {"num_experts": 8, "top_k": 2}
    # A LoRA checkpoint resolves to its base model's config.
    assert reap_mod.moe_export_info(str(lora)) == {"num_experts": 8, "top_k": 2}
    # Unreadable or missing never raises.
    assert reap_mod.moe_export_info(str(tmp_path / "missing")) is None
    (base / "config.json").write_text("{not json")
    assert reap_mod.moe_export_info(str(base)) is None


class _Model:
    def __init__(self, four_bit = False):
        self.is_loaded_in_4bit = four_bit
        self.config = types.SimpleNamespace(_name_or_path = "org/base-model")
        self.merged = False

    def merge_and_unload(self):
        merged = _Model()
        merged.merged = True
        return merged


class _Backend:
    def __init__(
        self,
        model,
        is_peft = False,
    ):
        self.current_model = model
        self.current_tokenizer = object()
        self.is_peft = is_peft


@pytest.fixture
def fake_reap(monkeypatch):
    calls = []

    def reap_prune(model, tokenizer, **kwargs):
        calls.append(kwargs)
        model._unsloth_reap_report = {
            "model.layers.0.mlp": [0, 2, 3],
            "model.layers.1.mlp": [1, 2, 3],
        }
        return model

    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = types.SimpleNamespace(reap_prune = reap_prune)
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    return calls


def test_apply_reap_prunes_once_and_reuses(fake_reap, tmp_path):
    backend = _Backend(_Model())
    options = {"sparsity": 0.25, "num_calibration_samples": 64, "dataset": " "}
    ok, message = reap_mod.apply_reap(backend, options, str(tmp_path))
    assert ok and "3 experts" in message and "2 layers" in message
    assert len(fake_reap) == 1
    kwargs = fake_reap[0]
    assert kwargs["sparsity"] == 0.25 and kwargs["num_calibration_samples"] == 64
    assert kwargs["dataset"] is None
    # Staged where it fits (export dir or the temp dir), discarded with the loaded model; the folder
    # carries the model name, which GGUF file names come from.
    staging = backend._reap_staging.name
    import os

    assert os.path.isdir(staging) and kwargs["save_directory"] == os.path.join(
        staging, "base-model-REAP"
    )
    # A multi-format export calls the worker again with the same settings: no second pruning.
    assert reap_mod.apply_reap(backend, dict(options), str(tmp_path)) == (True, "")
    assert len(fake_reap) == 1
    ok, message = reap_mod.apply_reap(backend, {**options, "sparsity": 0.5}, str(tmp_path))
    assert not ok and "different REAP settings" in message
    reap_mod.discard_reap_staging(backend)
    assert backend._reap_key is None
    assert not os.path.exists(staging)


def test_apply_reap_merges_lora_and_refuses_4bit(fake_reap, tmp_path):
    assert reap_mod.apply_reap(_Backend(_Model()), None, str(tmp_path)) == (True, "")
    ok, message = reap_mod.apply_reap(
        _Backend(_Model(four_bit = True)), {"sparsity": 0.25}, str(tmp_path)
    )
    assert not ok and "16-bit" in message and not fake_reap
    backend = _Backend(_Model(), is_peft = True)
    ok, _ = reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    assert ok and backend.current_model.merged and backend.is_peft is False
    reap_mod.discard_reap_staging(backend)


def test_write_reap_report(fake_reap, tmp_path):
    backend = _Backend(_Model())
    reap_mod.apply_reap(backend, {"sparsity": 0.25, "num_calibration_samples": 64}, str(tmp_path))
    out = tmp_path / "export"
    out.mkdir()
    reap_mod.write_reap_report(backend, str(out))
    report = json.loads((out / reap_mod.REPORT_NAME).read_text())
    assert report["sparsity"] == 0.25 and report["dataset"] == "mlabonne/open-perfectblend"
    assert report["kept_experts"]["model.layers.1.mlp"] == [1, 2, 3]
    # A Hub-only export (no local folder) writes nothing and does not raise.
    reap_mod.write_reap_report(backend, None)
    reap_mod.discard_reap_staging(backend)


class _ExportBackend:
    def __init__(self):
        self.calls = []

    def export_merged_model(self, **kwargs):
        self.calls.append(("merged", kwargs))
        return True, "ok", "/out"

    def export_gguf(self, **kwargs):
        self.calls.append(("gguf", kwargs))
        return True, "ok", "/out"

    def export_lora_adapter(self, **kwargs):
        self.calls.append(("lora", kwargs))
        return True, "ok", "/out"


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get())
    return items


def test_worker_runs_reap_before_export(monkeypatch):
    from core.export.worker import _handle_export

    seen = []
    monkeypatch.setattr(reap_mod, "apply_reap", lambda b, o, p: (seen.append(o) or (True, "kept")))
    monkeypatch.setattr(
        reap_mod, "write_reap_report", lambda b, path: seen.append(("report", path))
    )
    backend, q = _ExportBackend(), queue.Queue()
    _handle_export(
        backend, {"export_type": "gguf", "save_directory": "/tmp/o", "reap": {"sparsity": 0.5}}, q
    )
    assert seen == [{"sparsity": 0.5}, ("report", "/out")]
    assert backend.calls[0][0] == "gguf"
    final = _drain(q)[-1]
    assert final["success"] is True

    # Off by default: no REAP step at all.
    seen.clear()
    _handle_export(backend, {"export_type": "merged", "save_directory": "/tmp/o"}, q)
    assert seen == []


def test_worker_reap_failure_skips_export(monkeypatch):
    from core.export.worker import _handle_export

    monkeypatch.setattr(reap_mod, "apply_reap", lambda b, o, p: (False, "needs 16-bit"))
    backend, q = _ExportBackend(), queue.Queue()
    _handle_export(
        backend,
        {"export_type": "merged", "save_directory": "/tmp/o", "reap": {"sparsity": 0.25}},
        q,
    )
    assert backend.calls == []
    final = _drain(q)[-1]
    assert final["success"] is False and final["message"] == "needs 16-bit"


def test_request_models_reap_is_optional_and_bounded():
    for cls in (ExportMergedModelRequest, ExportBaseModelRequest, ExportGGUFRequest):
        assert cls(save_directory = "/tmp/o").reap is None
        req = cls(save_directory = "/tmp/o", reap = {"sparsity": 0.5})
        assert req.reap.num_calibration_samples == 512 and req.reap.dataset is None
    assert ReapOptions(dataset = "mlabonne/open-perfectblend").dataset == "mlabonne/open-perfectblend"
    for bad in (
        {"sparsity": 0},
        {"sparsity": 1},
        {"num_calibration_samples": 4},
        {"dataset": "../etc"},
        {"dataset": "a b"},
        {"dataset": "a/b/c"},
    ):
        with pytest.raises(ValidationError):
            ReapOptions(**bad)


def test_moe_info_route_local_folder(monkeypatch):
    from routes import export as export_route
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/config.json", "w") as f:
            json.dump({"num_local_experts": 8, "num_experts_per_tok": 2}, f)
        res = asyncio.run(
            export_route.get_moe_info(
                model = d, hf_token = None, current_subject = "t", allow_ambient = False
            )
        )
        assert (res.num_experts, res.top_k) == (8, 2)
        with open(f"{d}/config.json", "w") as f:
            json.dump({"model_type": "llama"}, f)
        res = asyncio.run(
            export_route.get_moe_info(
                model = d, hf_token = None, current_subject = "t", allow_ambient = False
            )
        )
        assert res.num_experts is None and res.top_k is None
