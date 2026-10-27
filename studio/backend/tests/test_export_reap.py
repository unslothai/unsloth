# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in REAP expert pruning on the export page: MoE detection, the worker step, request models.

Core's FastLanguageModel.reap_prune is stubbed; Core has its own GPU tests for the pruning itself.
"""

from __future__ import annotations

import weakref

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
    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda b, path: seen.append(("report", path)))
    backend, q = _ExportBackend(), queue.Queue()
    _handle_export(
        backend, {"export_type": "gguf", "save_directory": "/tmp/o", "reap": {"sparsity": 0.5}}, q
    )
    # GGUF exports keep one report per file themselves, before their upload.
    assert seen == [{"sparsity": 0.5}]
    assert backend.calls[0][0] == "gguf"
    final = _drain(q)[-1]
    assert final["success"] is True

    # Off by default: no pruning; only the report sync, which clears a stale report.
    seen.clear()
    _handle_export(backend, {"export_type": "merged", "save_directory": "/tmp/o"}, q)
    assert seen == [("report", "/out")]


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
    # Grouped routing without topk_group: Core refuses it, so the card is hidden.
    assert reap_mod.moe_info_from_config({**dsv3, "topk_group": None}) is None


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


def _core_keys():
    import ast
    from pathlib import Path

    core = Path(__file__).resolve().parents[3] / "unsloth" / "models" / "reap.py"
    if not core.is_file():
        pytest.skip("Core reap.py is not in this tree")
    tree = ast.parse(core.read_text(encoding = "utf-8"))
    return {
        t.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id in ("_NUM_EXPERTS_KEYS", "_TOP_K_KEYS")
    }


def test_probe_keys_cover_core():
    # A key Core reads but the probe does not would hide REAP for a model Core can prune.
    consts = _core_keys()
    for name in ("_NUM_EXPERTS_KEYS", "_TOP_K_KEYS"):
        assert set(consts[name]) <= set(getattr(reap_mod, name))
    # HunYuan-A13B and Ernie 4.5 as their config.json files spell them.
    assert reap_mod.moe_info_from_config({"num_experts": 64, "moe_topk": [8] * 32})["top_k"] == 8
    assert reap_mod.moe_info_from_config({"moe_num_experts": 64, "moe_k": 6})["top_k"] == 6


def test_probe_extra_keys_are_raw_names_core_reads():
    # Studio may only add raw config.json names that a transformers config maps onto a Core key
    # (Core reads them through attribute_map); anything else would advertise a model Core refuses.
    pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    consts = _core_keys()
    mapped = {}
    for config_class in CONFIG_MAPPING.values():
        for alias, raw in (getattr(config_class, "attribute_map", None) or {}).items():
            mapped.setdefault(raw, set()).add(alias)
    for name in ("_NUM_EXPERTS_KEYS", "_TOP_K_KEYS"):
        core_keys, studio_keys = consts[name], getattr(reap_mod, name)
        for extra in set(studio_keys) - set(core_keys):
            assert mapped.get(extra, set()) & set(core_keys), f"{extra} is not read by Core"


def test_custom_calibration_dataset_uses_the_request_token(fake_reap, tmp_path, monkeypatch):
    import datasets

    opened = []
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: opened.append((a, k)) or {"train": ["rows"]}
    )
    backend = _Backend(_Model())
    reap_mod.apply_reap(
        backend, {"sparsity": 0.25, "dataset": "org/private"}, str(tmp_path), hf_token = "hf_x"
    )
    assert opened == [(("org/private",), {"streaming": True, "token": "hf_x"})]
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
    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda b, path: None)
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
    # No network: the up-front adapter check finds nothing, so the config-less fallback runs.
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)
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
    # The Hub base has no hf_quant_config.json (checked without network here).
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)
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


def test_sweep_writes_nothing_for_users_who_never_used_reap(tmp_path, monkeypatch):
    registry = tmp_path / "cache" / "reap_staging.json"
    monkeypatch.setattr(reap_mod, "_registry", lambda: registry)
    reap_mod.sweep_reap_staging()
    reap_mod.sweep_reap_staging(dead_pid = 12345)
    reap_mod.discard_reap_staging(_Backend(_Model()))
    assert not registry.exists() and not registry.parent.exists()


def test_failed_pruning_unloads_the_partly_pruned_model(tmp_path, monkeypatch):
    def reap_prune(model, tokenizer, **kwargs):
        raise RuntimeError("out of memory in layer 7")

    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = types.SimpleNamespace(reap_prune = reap_prune)
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    monkeypatch.setattr(reap_mod, "_registry", lambda: tmp_path / "reap_staging.json")
    unloaded = []

    class _Loaded(_Backend):
        def cleanup_memory(self):
            unloaded.append(True)
            reap_mod.discard_reap_staging(self)
            self.current_model = None

    backend = _Loaded(_Model(), is_peft = True)
    with pytest.raises(RuntimeError, match = "unloaded"):
        reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    # Merged and possibly part pruned: unloaded, so no later export (REAP or not) can use it.
    assert unloaded and backend.current_model is None
    assert backend._reap_key is None and backend._reap_staging is None
    import glob

    assert not glob.glob(str(tmp_path / "unsloth-hub-upload-*"))


def test_probe_matches_core_on_identity_experts_and_split_levels():
    # LongCat style identity experts: Core refuses, so the card must not appear.
    assert (
        reap_mod.moe_info_from_config(
            {"n_routed_experts": 8, "num_experts_per_tok": 2, "zero_expert_num": 2}
        )
        is None
    )
    assert (
        reap_mod.moe_info_from_config(
            {"num_experts": 8, "top_k": 2, "use_bidirectional_attention": "all"}
        )
        is None
    )
    # Each field from the first level that has it, as Core's _first_attr reads them.
    split = {"num_experts": 8, "text_config": {"top_k": 2}}
    assert reap_mod.moe_info_from_config(split) == {"num_experts": 8, "top_k": 2}


def test_validated_sparsity_reaches_core_unrounded(fake_reap, tmp_path):
    backend = _Backend(_Model())
    reap_mod.apply_reap(
        backend, {"sparsity": 0.33337, "num_calibration_samples": 64}, str(tmp_path)
    )
    assert fake_reap[0]["sparsity"] == 0.33337
    reap_mod.discard_reap_staging(backend)


def test_report_follows_the_exported_model(fake_reap, tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    backend = _Backend(_Model())
    reap_mod.apply_reap(backend, {"sparsity": 0.25, "num_calibration_samples": 64}, str(tmp_path))
    reap_mod.sync_reap_report(backend, str(out))
    assert (out / reap_mod.REPORT_NAME).is_file()
    reap_mod.discard_reap_staging(backend)
    # The same folder reused for an export of an unpruned model: the old report goes.
    reap_mod.sync_reap_report(_Backend(_Model()), str(out))
    assert not (out / reap_mod.REPORT_NAME).exists()


def test_probe_hides_layouts_core_refuses():
    base = {"n_routed_experts": 16, "num_experts_per_tok": 2}
    # Grouped routing without topk_group, and model-parallel layouts saved in the config.
    assert reap_mod.moe_info_from_config({**base, "n_group": 4}) is None
    assert reap_mod.moe_info_from_config({**base, "n_group": 1})["num_experts"] == 16
    for layout in (
        {"tp_size": 2},
        {"pp_size": 4},
        {"fsdp_size": 2},
        {"enable_expert_parallel": True},
    ):
        assert reap_mod.moe_info_from_config({**base, "distributed_config": layout}) is None
    assert (
        reap_mod.moe_info_from_config({**base, "distributed_config": {"tp_size": 1}})["top_k"] == 2
    )


def test_probe_hides_stateful_decoders_core_refuses():
    moe = {"num_experts": 64, "num_experts_per_tok": 4}
    for model_type in ("glm_moe_dsa", "hy_v4", "zaya"):
        assert reap_mod.moe_info_from_config({**moe, "model_type": model_type}) is None
    assert (
        reap_mod.moe_info_from_config(
            {"model_type": "glm5_next", "text_config": {**moe, "model_type": "glm5_next_text"}}
        )
        is None
    )
    assert reap_mod.moe_info_from_config({**moe, "model_type": "qwen3_moe"})["num_experts"] == 64


def test_stateful_model_types_match_transformers():
    # Each listed type really has decoder layers taking prev_* state, and no other type does.
    pytest.importorskip("transformers")
    # The modeling modules import torch (absent on the CPU-only CI runners).
    pytest.importorskip("torch")
    import importlib
    import inspect

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    for model_type in sorted(reap_mod._STATEFUL_MODEL_TYPES - {"glm5_next_text"}):
        if model_type not in CONFIG_MAPPING_NAMES:
            continue
        module = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        layers = [
            c
            for n, c in vars(module).items()
            if inspect.isclass(c) and n.endswith("DecoderLayer") and hasattr(c, "forward")
        ]
        assert any(
            any(p.startswith("prev_") for p in inspect.signature(c.forward).parameters)
            for c in layers
        ), model_type


def test_probe_hides_hash_routed_moe():
    dsv4 = {"model_type": "deepseek_v4", "n_routed_experts": 256, "num_experts_per_tok": 6}
    assert reap_mod.moe_info_from_config(dsv4) is None


def test_hash_routed_model_types_match_transformers():
    # The listed types are exactly the installed transformers models whose code builds tid2eid.
    pytest.importorskip("transformers")
    from pathlib import Path

    import transformers

    models = Path(transformers.__file__).parent / "models"
    found = {
        f.parent.name
        for f in models.glob("*/modeling_*.py")
        if "tid2eid" in f.read_text(encoding = "utf-8", errors = "ignore")
    }
    assert found <= reap_mod._HASH_ROUTED_MODEL_TYPES, found - reap_mod._HASH_ROUTED_MODEL_TYPES


@pytest.mark.parametrize("dies_on", ["graceful", "terminate", "kill", "already_dead"])
def test_every_confirmed_worker_death_sweeps_its_reap_staging(monkeypatch, dies_on):
    # Server shutdown with REAP pruning: the worker cannot remove its staging copy itself, so the
    # orchestrator sweeps for that pid once it knows the worker is dead, not only on a cancel.
    from core.export import orchestrator as orchestrator_module

    class Worker:
        pid = 4321

        def __init__(self):
            self.alive = dies_on != "already_dead"

        def is_alive(self):
            return self.alive

        def join(self, timeout = None):
            if dies_on == "graceful":
                self.alive = False

        def terminate(self):
            if dies_on == "terminate":
                self.alive = False

        def kill(self):
            self.alive = False

    swept = []
    monkeypatch.setattr(
        reap_mod, "sweep_reap_staging", lambda dead_pid = None: swept.append(dead_pid)
    )
    orch = orchestrator_module.ExportOrchestrator()
    orch._proc = Worker()
    monkeypatch.setattr(orch, "_drain_queue", lambda: None)
    orch._cmd_queue = type("Q", (), {"put": lambda self, item: None})()
    try:
        assert orch._shutdown_subprocess(timeout = 0) is True
    finally:
        orch._proc = None
    assert swept == [4321]


def test_a_worker_that_survives_kill_is_not_swept(monkeypatch):
    from core.export import orchestrator as orchestrator_module

    class Wedged:
        pid = 4321
        is_alive = staticmethod(lambda: True)
        join = terminate = kill = staticmethod(lambda *a, **k: None)

    swept = []
    monkeypatch.setattr(
        reap_mod, "sweep_reap_staging", lambda dead_pid = None: swept.append(dead_pid)
    )
    orch = orchestrator_module.ExportOrchestrator()
    orch._proc = Wedged()
    monkeypatch.setattr(orch, "_drain_queue", lambda: None)
    orch._cmd_queue = type("Q", (), {"put": lambda self, item: None})()
    try:
        assert orch._shutdown_subprocess(timeout = 0) is False
    finally:
        orch._proc = None
    assert swept == []


@pytest.mark.parametrize(
    "result",
    [
        (False, "GGUF files were saved to /out, but the Hub upload failed: 503", "/out"),
        (False, "GGUF export failed: out of disk", None),
        (True, "ok", "/out"),
    ],
)
def test_the_worker_leaves_gguf_reports_to_the_gguf_export(monkeypatch, result):
    # export_gguf writes one report per GGUF before its upload; a folder-level one would describe
    # every GGUF a reused folder holds.
    from core.export.worker import _handle_export

    seen = []
    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda b, path, *a: seen.append(path))
    backend, q = _ExportBackend(), queue.Queue()
    backend.export_gguf = lambda **kwargs: result
    _handle_export(backend, {"export_type": "gguf", "save_directory": "/tmp/o"}, q)
    assert seen == []


def test_gguf_report_names_follow_each_file():
    assert (
        reap_mod.gguf_report_name("/x/model-REAP.Q4_K_M.gguf")
        == "model-REAP.Q4_K_M.reap_report.json"
    )
    assert reap_mod.gguf_report_name(".Q8_0.GGUF") == ".Q8_0.reap_report.json"


def test_cleanup_after_a_worker_crash_sweeps_its_staging(monkeypatch):
    # The worker died on its own (CUDA crash, OOM kill) after staging: cleanup sees it dead.
    from core.export import orchestrator as orchestrator_module

    class Dead:
        pid = 777
        is_alive = staticmethod(lambda: False)

    swept = []
    monkeypatch.setattr(
        reap_mod, "sweep_reap_staging", lambda dead_pid = None: swept.append(dead_pid)
    )
    orch = orchestrator_module.ExportOrchestrator()
    orch._proc = Dead()
    try:
        assert orch.cleanup_memory() is True
    finally:
        orch._proc = None
    assert swept == [777]


@pytest.mark.parametrize("export_type", ["merged", "base"])
@pytest.mark.parametrize("saved_locally", [True, False])
def test_a_failed_upload_after_a_complete_local_save_still_syncs_the_report(
    monkeypatch, export_type, saved_locally
):
    # merged / base return no path when the Hub upload raises; the local save is complete by then.
    from core.export.worker import _handle_export

    seen = []
    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda b, path: seen.append(path))
    backend, q = _ExportBackend(), queue.Queue()
    backend._local_export_path = "/stale"  # from an earlier export: never reused

    def export(**kwargs):
        if saved_locally:
            backend._local_export_path = "/out"
        return False, "Export failed: 503 from the Hub", None

    setattr(
        backend, "export_merged_model" if export_type == "merged" else "export_base_model", export
    )
    _handle_export(backend, {"export_type": export_type, "save_directory": "/tmp/o"}, q)
    assert seen == (["/out"] if saved_locally else [])
    assert _drain(q)[-1]["success"] is False


@pytest.mark.parametrize(
    "method, offered",
    [
        ("fp8", False),
        ("compressed-tensors", False),
        ("sparseml", False),
        ("modelopt", False),
        ("gptq", False),
        ("awq", False),
        ("nvfp4", False),
        ("some-future-format", False),
        # Packed unless the Hub id maps back to 16-bit (below); gpt-oss MXFP4 dequantizes to
        # bf16 without native kernels.
        ("bitsandbytes", False),
        ("mxfp4", True),
    ],
)
def test_probe_offers_reap_only_for_quantization_a_16_bit_load_unpacks(method, offered):
    config = {"model_type": "qwen3_moe", "num_experts": 128, "num_experts_per_tok": 8}
    for holder in ("top", "text_config"):
        quant = {"quant_method": method}
        cfg = (
            dict(config, quantization_config = quant)
            if holder == "top"
            else dict(config, text_config = {"quantization_config": quant})
        )
        assert (reap_mod.moe_info_from_config(cfg) is not None) is offered, (method, holder)
    # A bitsandbytes config from before quant_method existed reads the same.
    old_bnb = dict(config, quantization_config = {"load_in_4bit": True})
    assert reap_mod.moe_info_from_config(old_bnb) is None


def test_probe_hides_reap_for_an_older_modelopt_export(tmp_path, monkeypatch):
    # Declared in hf_quant_config.json, not config.json, as the loader reads it.
    folder = tmp_path / "moe-modelopt"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    assert reap_mod.moe_export_info(str(folder))["num_experts"] == 8
    (folder / "hf_quant_config.json").write_text('{"quantization": {"quant_algo": "FP8"}}')
    assert reap_mod.moe_export_info(str(folder)) is None
    # A Hub id is checked for the same file.
    import utils.models.model_config as model_config

    monkeypatch.setattr(
        model_config,
        "_read_raw_config_json",
        lambda *a, **k: {"num_experts": 8, "num_experts_per_tok": 2},
    )
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: True)
    assert reap_mod.moe_export_info("org/moe-modelopt", local_files_only = False) is None
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)
    assert reap_mod.moe_export_info("org/moe-modelopt", local_files_only = False)["num_experts"] == 8


@pytest.mark.parametrize(
    "route, request_cls",
    [
        ("export_merged_model", ExportMergedModelRequest),
        ("export_base_model", ExportBaseModelRequest),
        ("export_gguf", ExportGGUFRequest),
    ],
)
def test_custom_reap_dataset_is_validated_with_the_callers_token(route, request_cls):
    # Managed accounts refuse a remote dataset without the caller's own token (403), so the
    # dataset check must carry the token the request supplied.
    from routes import export as export_route

    class Stop(Exception):
        pass

    checked = []

    def validate(values, **kwargs):
        checked.append(values)
        if "hf_dataset" in values:
            raise Stop

    request = request_cls(
        save_directory = "out",
        hf_token = "hf_caller",
        reap = {"sparsity": 0.25, "dataset": "org/calib"},
    )
    original = export_route.validate_job_paths
    export_route.validate_job_paths = validate
    try:
        with pytest.raises(Stop):
            asyncio.run(
                getattr(export_route, route)(
                    request = request, current_subject = "t", allow_ambient = False
                )
            )
    finally:
        export_route.validate_job_paths = original
    assert {"hf_dataset": "org/calib", "hf_token": "hf_caller"} in checked


@pytest.mark.parametrize(
    "splits, expected",
    [
        ({"train": ["t"], "validation": ["v"]}, ["t"]),
        ({"validation": ["v"]}, ["v"]),
        ({"test": ["x"]}, ["x"]),
        (["plain"], ["plain"]),
    ],
)
def test_calibration_dataset_takes_train_or_its_only_split(monkeypatch, splits, expected):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: splits)
    assert reap_mod._calibration_dataset("org/calib", None) == expected


def test_calibration_dataset_with_several_splits_and_no_train_is_refused(monkeypatch):
    import datasets
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: {"validation": ["v"], "test": ["x"]}
    )
    with pytest.raises(ValueError, match = r"no train split .*\(test, validation\)"):
        reap_mod._calibration_dataset("org/calib", None)


def test_probe_follows_an_adapter_even_beside_a_dense_config(tmp_path):
    # A LoRA folder may also hold a config.json (dense, stale or the base's); the export loader
    # follows adapter_config.json, so the probe reads the base it names.
    base = tmp_path / "moe_base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    (lora / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64}))
    info = reap_mod.moe_export_info(str(lora))
    assert (info["num_experts"], info["top_k"]) == (8, 2)


def test_probe_reads_a_folders_own_config_when_its_adapter_names_no_base(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    (folder / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    assert reap_mod.moe_export_info(str(folder))["num_experts"] == 8


def test_failed_pruning_drops_the_model_even_when_cleanup_reports_failure(tmp_path, monkeypatch):
    # cleanup_memory catches its own errors and returns False (e.g. an inference unload failing
    # first): the merged, partly pruned model must still never be exported again.
    def reap_prune(model, tokenizer, **kwargs):
        raise RuntimeError("out of memory in layer 7")

    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = types.SimpleNamespace(reap_prune = reap_prune)
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    monkeypatch.setattr(reap_mod, "_registry", lambda: tmp_path / "reap_staging.json")

    class _Stuck(_Backend):
        current_checkpoint = "/ckpt"

        def cleanup_memory(self):
            return False

    backend = _Stuck(_Model(), is_peft = True)
    with pytest.raises(RuntimeError, match = "unloaded"):
        reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    assert backend.current_model is None and backend.current_tokenizer is None
    assert backend.current_checkpoint is None
    assert backend._reap_key is None and backend._reap_staging is None


def test_reap_merge_marks_the_lora_load_until_the_next_load(fake_reap, tmp_path):
    backend = _Backend(_Model(), is_peft = True)
    reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    assert backend.is_peft is False and backend._reap_merged_peft is True
    # An unload / new load clears it with the staging copy.
    reap_mod.discard_reap_staging(backend)
    assert backend._reap_merged_peft is False


def test_worker_skips_reap_for_a_base_export_after_a_reap_merge(monkeypatch):
    from core.export.worker import _handle_export

    applied = []
    monkeypatch.setattr(reap_mod, "apply_reap", lambda *a, **k: applied.append(a) or (True, ""))
    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda b, path: None)
    backend, q = _ExportBackend(), queue.Queue()
    backend.is_peft, backend._reap_merged_peft = False, True
    backend.export_base_model = lambda **kwargs: (False, "This is a PEFT model.", None)
    _handle_export(
        backend, {"export_type": "base", "save_directory": "/tmp/o", "reap": {"sparsity": 0.5}}, q
    )
    assert applied == []


@pytest.mark.parametrize("export_type", ["merged", "base", "gguf"])
def test_an_export_without_reap_after_a_pruned_one_on_the_same_load_is_refused(export_type):
    # The loaded weights are the pruned ones now; omitting `reap` means "not pruned".
    from core.export.worker import _handle_export

    backend, q = _ExportBackend(), queue.Queue()
    backend._reap_key = (0.25, 512, None)
    backend.current_model = _Model()
    backend._reap_model = weakref.ref(backend.current_model)
    backend.export_base_model = lambda **kwargs: backend.calls.append(("base", kwargs))
    _handle_export(backend, {"export_type": export_type, "save_directory": "/tmp/o"}, q)
    assert backend.calls == []
    final = _drain(q)[-1]
    assert final["success"] is False and "Load it again" in final["message"]


@pytest.mark.parametrize("strategy, offered", [("mtp", False), ("eagle3", True), (None, True)])
def test_probe_hides_reap_for_a_full_checkpoint_carrying_a_trained_mtp_head(
    tmp_path, strategy, offered
):
    # The export attaches that head on load, and REAP refuses an attached MTP head.
    folder = tmp_path / "checkpoint-30"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    if strategy:
        (folder / "drafter.json").write_text(json.dumps({"strategy": strategy}))
        (folder / "eagle3").mkdir()
        (folder / "eagle3" / "config.json").write_text("{}")
    assert (reap_mod.moe_export_info(str(folder)) is not None) is offered


@pytest.mark.parametrize("with_head", [True, False])
def test_probe_hides_reap_for_a_lora_folder_with_a_trained_mtp_head(tmp_path, with_head):
    # Unsloth's adapter loader re-attaches mtp_head.safetensors, and REAP refuses an attached head.
    base = tmp_path / "moe_base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    if with_head:
        (lora / "mtp_head.safetensors").write_bytes(b"head")
    assert (reap_mod.moe_export_info(str(lora)) is None) is with_head


def test_bitsandbytes_is_offered_only_for_hub_ids_unsloth_maps_to_16_bit(tmp_path, monkeypatch):
    config = {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
    }
    assert reap_mod.moe_info_from_config(config, bnb_maps_to_16_bit = True)["num_experts"] == 8
    assert reap_mod.moe_info_from_config(config) is None
    # A local folder never maps: loading it in 16-bit keeps the packed weights.
    folder = tmp_path / "Qwen3-30B-A3B-bnb-4bit"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(reap_mod, "_maps_to_16_bit", lambda model: True)
    assert reap_mod.moe_export_info(str(folder)) is None


def test_unsloth_mapper_is_read_without_importing_unsloth():
    import sys

    pytest.importorskip("importlib.util")
    if __import__("importlib.util").util.find_spec("unsloth") is None:
        pytest.skip("unsloth is not installed")
    reap_mod._int_to_float_mapper.cache_clear()
    already = "unsloth" in sys.modules
    mapper = reap_mod._int_to_float_mapper()
    assert any(k.endswith("-bnb-4bit") for k in mapper)
    assert ("unsloth" in sys.modules) is already


def test_moe_info_route_skips_a_hub_lora_with_an_mtp_head(monkeypatch):
    from routes import export as export_route

    import routes.models as models_route

    # Imported inside the route from routes.models.
    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", _async_none)
    monkeypatch.setattr(
        reap_mod,
        "moe_export_info",
        lambda model, *a, **k: None if model == "org/lora" else {"num_experts": 8, "top_k": 2},
    )
    monkeypatch.setattr(reap_mod, "hub_adapter_has_mtp_head", lambda repo, token = None: True)
    # No network: the repo is not seen as an adapter up front, so the fallback path runs.
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)
    import utils.models.model_config as model_config

    monkeypatch.setattr(
        model_config, "get_base_model_from_lora_identifier", lambda m, t = None: "org/moe-base"
    )
    res = asyncio.run(
        export_route.get_moe_info(
            model = "org/lora", hf_token = None, current_subject = "t", allow_ambient = False
        )
    )
    assert res.num_experts is None
    monkeypatch.setattr(reap_mod, "hub_adapter_has_mtp_head", lambda repo, token = None: False)
    res = asyncio.run(
        export_route.get_moe_info(
            model = "org/lora", hf_token = None, current_subject = "t", allow_ambient = False
        )
    )
    assert res.num_experts == 8


async def _async_none(*args, **kwargs):
    return None


@pytest.mark.parametrize("mtp_head", [True, False])
def test_moe_info_route_probes_a_hub_adapter_through_its_base_even_with_a_config(
    monkeypatch, mtp_head
):
    # A Hub LoRA repo can ship a config.json too; the loader follows adapter_config.json.
    from routes import export as export_route
    import routes.models as models_route
    import utils.models.model_config as model_config

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", _async_none)
    probed = []

    def info(model, *a, **k):
        probed.append(model)
        return {"num_experts": 4 if model == "org/lora" else 8, "top_k": 2}

    monkeypatch.setattr(reap_mod, "moe_export_info", info)
    monkeypatch.setattr(
        reap_mod,
        "hub_file_exists",
        lambda repo, name, token = None: name == "adapter_config.json"
        or (mtp_head and name == "mtp_head.safetensors"),
    )
    monkeypatch.setattr(reap_mod, "hub_adapter_has_mtp_head", lambda repo, token = None: mtp_head)
    monkeypatch.setattr(
        model_config, "get_base_model_from_lora_identifier", lambda m, t = None: "org/base"
    )
    res = asyncio.run(
        export_route.get_moe_info(
            model = "org/lora", hf_token = None, current_subject = "t", allow_ambient = False
        )
    )
    assert "org/lora" not in probed
    assert res.num_experts == (None if mtp_head else 8)


def test_probe_tells_an_unreadable_config_from_a_dense_or_missing_one(tmp_path, monkeypatch):
    import utils.models.model_config as model_config

    # Read failure (network, Hub 5xx): unknown, raised for the route to report.
    def flaky(*a, **k):
        raise OSError("Connection reset by peer")

    monkeypatch.setattr(model_config, "_read_raw_config_json", flaky)
    with pytest.raises(reap_mod.MoeProbeError, match = "Connection reset"):
        reap_mod.moe_export_info("org/moe", local_files_only = False)
    # No config.json at all (a GGUF repo): a definite answer.
    monkeypatch.setattr(model_config, "_read_raw_config_json", lambda *a, **k: None)
    assert reap_mod.moe_export_info("org/gguf-only", local_files_only = False) is None
    # A malformed config.json stays malformed: hidden, not retried.
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "config.json").write_text("{not json")
    monkeypatch.undo()
    assert reap_mod.moe_export_info(str(folder)) is None


def test_moe_info_route_answers_503_when_the_config_cannot_be_read(monkeypatch):
    # The page then keeps a REAP export blocked with a retry instead of exporting unpruned.
    from fastapi import HTTPException

    from routes import export as export_route
    import routes.models as models_route

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", _async_none)
    monkeypatch.setattr(reap_mod, "hub_file_exists", lambda repo, name, token = None: False)

    def unreadable(*a, **k):
        raise reap_mod.MoeProbeError("Could not read the config of 'org/moe': timeout")

    monkeypatch.setattr(reap_mod, "moe_export_info", unreadable)
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            export_route.get_moe_info(
                model = "org/moe", hf_token = None, current_subject = "t", allow_ambient = False
            )
        )
    assert err.value.status_code == 503


def test_hub_file_checks_tell_a_missing_file_from_an_unknown_one(monkeypatch):
    import huggingface_hub

    class Api:
        answer = False

        def __init__(self, token = None):
            pass

        def file_exists(self, repo, filename):
            if isinstance(Api.answer, Exception):
                raise Api.answer
            return Api.answer

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    # file_exists answers False only for a definite absence.
    assert reap_mod.hub_file_exists("org/lora", "mtp_head.safetensors") is False
    Api.answer = True
    assert reap_mod.hub_adapter_has_mtp_head("org/lora") is True
    # A timeout or gated repo is unknown: never assumed absent.
    Api.answer = TimeoutError("read timed out")
    with pytest.raises(reap_mod.MoeProbeError, match = "timed out"):
        reap_mod.hub_adapter_has_mtp_head("org/lora")


def test_moe_info_route_answers_503_when_the_adapter_check_fails(monkeypatch):
    from fastapi import HTTPException

    from routes import export as export_route
    import routes.models as models_route

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", _async_none)

    def unknown(
        repo,
        name,
        token = None,
    ):
        raise reap_mod.MoeProbeError(f"Could not check '{repo}' for {name}: timeout")

    monkeypatch.setattr(reap_mod, "hub_file_exists", unknown)
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            export_route.get_moe_info(
                model = "org/lora", hf_token = None, current_subject = "t", allow_ambient = False
            )
        )
    assert err.value.status_code == 503


def test_a_reap_key_left_by_an_older_model_is_never_reused(fake_reap, tmp_path):
    # An unload that failed part way, then a new load: the old settings must not read as
    # "already pruned" for the unpruned new model.
    backend = _Backend(_Model())
    backend._reap_key = (0.25, 512, None)
    backend._reap_model = weakref.ref(_Model())  # some other, earlier model
    assert reap_mod.current_reap_key(backend) is None
    ok, _ = reap_mod.apply_reap(backend, {"sparsity": 0.25}, str(tmp_path))
    assert ok and len(fake_reap) == 1  # pruned again, this time for the loaded model
    assert reap_mod.current_reap_key(backend) == (0.25, 512, None)
    reap_mod.discard_reap_staging(backend)


def test_a_stale_reap_key_does_not_refuse_an_unpruned_export(monkeypatch):
    from core.export.worker import _handle_export

    monkeypatch.setattr(reap_mod, "sync_reap_report", lambda *a, **k: None)
    backend, q = _ExportBackend(), queue.Queue()
    backend.current_model = _Model()
    backend._reap_key = (0.25, 512, None)
    backend._reap_model = weakref.ref(_Model())
    _handle_export(backend, {"export_type": "merged", "save_directory": "/tmp/o"}, q)
    assert backend.calls and backend.calls[0][0] == "merged"


def test_hub_file_checks_fall_back_to_the_local_cache(monkeypatch):
    # Offline (or a failed request), a cached repo still answers from the Hub cache, so a cached
    # export never needs the network just to show the card.
    import huggingface_hub
    from huggingface_hub import constants

    cache = {}

    def from_cache(
        repo,
        filename,
        cache_dir = None,
    ):
        return cache.get(filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", from_cache)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    cache["mtp_head.safetensors"] = "/cache/snap/mtp_head.safetensors"
    assert reap_mod.hub_file_exists("org/lora", "mtp_head.safetensors") is True
    # Offline with the snapshot cached but no such file: the loader cannot fetch it either.
    cache.clear()
    cache["config.json"] = "/cache/snap/config.json"
    assert reap_mod.hub_file_exists("org/moe", "hf_quant_config.json") is False
    # Nothing cached: still unknown.
    cache.clear()
    with pytest.raises(reap_mod.MoeProbeError):
        reap_mod.hub_file_exists("org/moe", "hf_quant_config.json")

    # Online but the request fails: only a definite cache answer counts (a snapshot alone does not).
    class Api:
        def __init__(self, token = None):
            pass

        def file_exists(self, repo, filename):
            raise ConnectionError("unreachable")

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    cache["config.json"] = "/cache/snap/config.json"
    with pytest.raises(reap_mod.MoeProbeError):
        reap_mod.hub_file_exists("org/moe", "hf_quant_config.json")
    cache["hf_quant_config.json"] = "/cache/snap/hf_quant_config.json"
    assert reap_mod.hub_file_exists("org/moe", "hf_quant_config.json") is True


@pytest.mark.parametrize(
    "tier, default, offered",
    [
        ("457", "5.5.0", False),
        ("default", "4.57.6", False),  # Python 3.9: the default tier is transformers 4.57.6
        ("default", "5.5.0", True),
        ("530", "4.57.6", True),
        (RuntimeError("probe failed"), "5.5.0", True),  # unknown: Core still refuses clearly
    ],
)
def test_probe_hides_reap_when_the_export_would_run_transformers_4(
    tmp_path, monkeypatch, tier, default, offered
):
    import utils.transformers_version as tv

    folder = tmp_path / "Mixtral-tiny"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"num_local_experts": 8, "num_experts_per_tok": 2})
    )

    def activation_tier(model, token = None):
        if isinstance(tier, Exception):
            raise tier
        return tier

    monkeypatch.setattr(tv, "get_transformers_activation_tier", activation_tier)
    monkeypatch.setattr(tv, "TRANSFORMERS_DEFAULT_VERSION", default)
    assert (reap_mod.moe_export_info(str(folder)) is not None) is offered


def test_moe_info_route_answers_503_when_a_confirmed_adapters_base_cannot_be_read(monkeypatch):
    from fastapi import HTTPException

    from routes import export as export_route
    import routes.models as models_route
    import utils.models.model_config as model_config

    monkeypatch.setattr(models_route, "_require_model_access_or_caller_token", _async_none)
    monkeypatch.setattr(
        reap_mod, "hub_file_exists", lambda repo, name, token = None: name == "adapter_config.json"
    )
    # The adapter_config.json download failed after its retries.
    monkeypatch.setattr(model_config, "get_base_model_from_lora_identifier", lambda m, t = None: None)
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            export_route.get_moe_info(
                model = "org/lora", hf_token = None, current_subject = "t", allow_ambient = False
            )
        )
    assert err.value.status_code == 503


def test_a_stray_mtp_head_file_in_a_full_checkpoint_does_not_hide_reap(tmp_path):
    # The loader attaches a full checkpoint's head only per its training manifest; a leftover
    # mtp_head.safetensors next to full weights is never loaded.
    folder = tmp_path / "full-moe"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"num_experts": 8, "num_experts_per_tok": 2}))
    (folder / "mtp_head.safetensors").write_bytes(b"head")
    assert reap_mod.moe_export_info(str(folder))["num_experts"] == 8
