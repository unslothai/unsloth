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
    assert reap_mod.moe_info_from_config({"moe_num_experts": 32, "top_k": [4, 8]})["top_k"] == 8


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
    monkeypatch.setattr(
        reap_mod, "apply_reap", lambda b, o, p, **k: (seen.append(o) or (True, "kept"))
    )
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

    monkeypatch.setattr(reap_mod, "apply_reap", lambda b, o, p, **k: (False, "needs 16-bit"))
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


def test_moe_info_reports_router_groups():
    dsv3 = {"n_routed_experts": 256, "num_experts_per_tok": 8, "n_group": 8, "topk_group": 4}
    assert reap_mod.moe_info_from_config(dsv3) == {
        "num_experts": 256,
        "top_k": 8,
        "n_group": 8,
        "topk_group": 4,
    }
    # Core groups only when both are set.
    assert "n_group" not in reap_mod.moe_info_from_config({**dsv3, "topk_group": None})


def test_staging_parent_follows_the_chosen_drive(tmp_path, monkeypatch):
    from utils.paths import storage_roots

    exports = tmp_path / "exports"
    exports.mkdir()
    other = tmp_path / "other_drive"
    other.mkdir()
    monkeypatch.setattr(storage_roots, "resolve_export_dir", lambda *a, **k: exports)
    monkeypatch.setattr(
        storage_roots,
        "resolve_export_write_dir",
        lambda p = None: (other / "new" / "folder") if p else exports,
    )
    # An absolute save folder on another drive stages there (its nearest existing ancestor).
    assert reap_mod.reap_staging_parent(str(other / "new" / "folder")) == other
    assert reap_mod.reap_staging_parent("") == exports


def test_cancel_sweeps_a_killed_workers_staging(tmp_path, monkeypatch):
    registry = tmp_path / "reap_staging.json"
    monkeypatch.setattr(reap_mod, "_registry", lambda: registry)
    owned = tmp_path / "unsloth-hub-upload-a"
    owned.mkdir()
    (owned / reap_mod._OWNER_FILE).write_text("4242")
    (owned / "model.safetensors").write_bytes(b"x" * 16)
    foreign = tmp_path / "not-ours"
    foreign.mkdir()
    live = tmp_path / "unsloth-hub-upload-live"
    live.mkdir()
    import os

    (live / reap_mod._OWNER_FILE).write_text(str(os.getpid()))
    registry.write_text(
        json.dumps(
            [
                {"path": str(owned), "pid": 4242},
                # A path without its owner marker is never deleted.
                {"path": str(foreign), "pid": 4242},
                {"path": str(live), "pid": os.getpid()},
            ]
        )
    )
    reap_mod.sweep_reap_staging(dead_pid = 4242)
    assert not owned.exists() and foreign.exists() and live.exists()
    # The live worker's copy stays registered; the next sweep without a dead pid keeps it too.
    reap_mod.sweep_reap_staging()
    assert live.exists()
    assert {e["path"] for e in json.loads(registry.read_text())} == {str(foreign), str(live)}


def test_moe_info_route_uses_the_authorized_adapter_base(monkeypatch, tmp_path):
    from routes import export as export_route

    lora = tmp_path / "lora"
    lora.mkdir()
    # The adapter names one base; the authorizer resolves (and checks) the one that is used.
    (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/elsewhere"}))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    checked = []
    monkeypatch.setattr(export_route, "_authorized_adapter_base", lambda p: str(base))
    monkeypatch.setattr(
        export_route, "validate_job_paths", lambda values, **k: checked.append(values)
    )
    res = asyncio.run(
        export_route.get_moe_info(
            model = str(lora), hf_token = None, current_subject = "t", allow_ambient = False
        )
    )
    assert (res.num_experts, res.top_k) == (8, 2)
    assert {"base_model": str(base), "hf_token": None} in checked


def test_probe_keys_match_core():
    # A key Core cannot read would advertise REAP for a model the export then refuses. Studio may
    # only add raw config.json names that a transformers config maps onto one of Core's keys.
    import ast
    from pathlib import Path

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    core = Path(__file__).resolve().parents[3] / "unsloth" / "models" / "reap.py"
    if not core.is_file():
        pytest.skip("Core reap.py is not in this tree")
    tree = ast.parse(core.read_text(encoding = "utf-8"))
    consts = {
        t.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id in ("_NUM_EXPERTS_KEYS", "_TOP_K_KEYS")
    }
    mapped = {}
    for config_class in CONFIG_MAPPING.values():
        for alias, raw in (getattr(config_class, "attribute_map", None) or {}).items():
            mapped.setdefault(raw, set()).add(alias)
    for name in ("_NUM_EXPERTS_KEYS", "_TOP_K_KEYS"):
        core_keys, studio_keys = consts[name], getattr(reap_mod, name)
        assert set(core_keys) <= set(studio_keys)
        for extra in set(studio_keys) - set(core_keys):
            assert mapped.get(extra, set()) & set(core_keys), f"{extra} is not read by Core"
    # HunYuan-A13B and Ernie 4.5 as their config.json files spell them.
    assert reap_mod.moe_info_from_config({"num_experts": 64, "moe_topk": [8] * 32})["top_k"] == 8
    assert reap_mod.moe_info_from_config({"moe_num_experts": 64, "moe_k": 6})["top_k"] == 6


def test_custom_calibration_dataset_uses_the_request_token(fake_reap, tmp_path, monkeypatch):
    import datasets

    opened = []
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: opened.append((a, k)) or ["rows"])
    backend = _Backend(_Model())
    reap_mod.apply_reap(
        backend, {"sparsity": 0.25, "dataset": "org/private"}, str(tmp_path), hf_token = "hf_x"
    )
    assert opened == [(("org/private",), {"split": "train", "streaming": True, "token": "hf_x"})]
    assert fake_reap[0]["dataset"] == ["rows"]
    reap_mod.discard_reap_staging(backend)
    # The default dataset is left to Core.
    backend = _Backend(_Model())
    reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path), hf_token = "hf_x")
    assert fake_reap[1]["dataset"] is None and len(opened) == 1
    reap_mod.discard_reap_staging(backend)


def test_worker_forwards_the_export_token_to_reap(monkeypatch):
    from core.export.worker import _handle_export

    seen = {}
    monkeypatch.setattr(
        reap_mod,
        "apply_reap",
        lambda b, o, p, hf_token = None: (seen.update(token = hf_token) or (True, "")),
    )
    monkeypatch.setattr(reap_mod, "write_reap_report", lambda b, path: None)
    backend, q = _ExportBackend(), queue.Queue()
    backend._checkpoint_token = "hf_load"
    _handle_export(
        backend, {"export_type": "merged", "save_directory": "/tmp/o", "reap": {"sparsity": 0.5}}, q
    )
    assert seen["token"] == "hf_load"
    _handle_export(
        backend,
        {
            "export_type": "merged",
            "save_directory": "/tmp/o",
            "hf_token": "hf_push",
            "reap": {"sparsity": 0.5},
        },
        q,
    )
    assert seen["token"] == "hf_push"


def test_moe_info_route_follows_a_hub_lora_to_its_authorized_base(monkeypatch):
    from routes import export as export_route
    from utils.models import model_config

    checked, probed = [], []

    async def require(model, token):
        checked.append((model, token))

    monkeypatch.setattr(
        export_route, "_require_model_access_or_caller_token", require, raising = False
    )
    import routes.models as models_route

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", require)
    monkeypatch.setattr(
        model_config, "get_base_model_from_lora_identifier", lambda m, t = None: "org/moe-base"
    )

    def probe(model, token, local, **kwargs):
        probed.append(model)
        return None if model == "me/lora" else {"num_experts": 8, "top_k": 2}

    monkeypatch.setattr(reap_mod, "moe_export_info", probe)
    res = asyncio.run(
        export_route.get_moe_info(
            model = "me/lora", hf_token = "hf_x", current_subject = "t", allow_ambient = False
        )
    )
    assert (res.num_experts, res.top_k) == (8, 2)
    assert probed == ["me/lora", "org/moe-base"]
    # The base is authorized with the caller's token before its config is read.
    assert checked == [("me/lora", "hf_x"), ("org/moe-base", "hf_x")]


def test_local_adapter_with_a_hub_base_reads_the_base_online(tmp_path, monkeypatch):
    from utils.models import model_config

    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/moe"}))
    calls = []

    def read(model, token, local_files_only, revision):
        calls.append((model, local_files_only))
        return {"num_experts": 8, "num_experts_per_tok": 2}

    monkeypatch.setattr(model_config, "_read_raw_config_json", read)
    assert reap_mod.moe_export_info(str(lora)) == {"num_experts": 8, "top_k": 2}
    # The adapter is local, its Hub base is not: an uncached base must still be fetched.
    assert calls == [("org/moe", False)]


def test_attached_mtp_head_is_refused_before_merging(fake_reap, tmp_path):
    model = _Model()
    model._unsloth_mtp_head = object()
    backend = _Backend(model, is_peft = True)
    ok, message = reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    assert not ok and "MTP head" in message
    assert not fake_reap and backend.is_peft is True and not model.merged


def test_hub_cards_name_the_model_reap_started_from(fake_reap, tmp_path):
    from core.export.export import _card_base_model

    backend = _Backend(_Model())
    reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    model = backend.current_model
    # Core repoints the config at the staging copy; the card keeps the original id.
    model.config._name_or_path = str(tmp_path / "staging" / "base-model-REAP")
    assert _card_base_model(model) == "org/base-model"
    plain = _Model()
    assert _card_base_model(plain) == "org/base-model"
    reap_mod.discard_reap_staging(backend)


def test_base_export_of_a_lora_keeps_its_refusal(monkeypatch):
    from core.export.worker import _handle_export

    seen = []
    monkeypatch.setattr(reap_mod, "apply_reap", lambda *a, **k: seen.append(a) or (True, ""))

    class _PeftBackend(_ExportBackend):
        is_peft = True

        def export_base_model(self, **kwargs):
            return False, "This is a LoRA checkpoint; use the merged export.", None

    backend, q = _PeftBackend(), queue.Queue()
    _handle_export(
        backend, {"export_type": "base", "save_directory": "/tmp/o", "reap": {"sparsity": 0.5}}, q
    )
    assert seen == []
    final = _drain(q)[-1]
    assert final["success"] is False and "LoRA" in final["message"]
