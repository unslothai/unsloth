# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's export load is authorized with whatever adapter_config.json the folder holds at
check time. One written after the check (no base pinned) named another account's model, which the worker
then loaded and merged. The worker must refuse an adapter config that appeared after the check."""

import ast
import inspect
import json

from core.export import export as export_mod
from core.export import orchestrator, worker


def test_an_adapter_config_written_after_the_check_is_refused(tmp_path):
    folder = tmp_path / "chat-folder"
    folder.mkdir()
    assert export_mod._checkpoint_load_refusal(str(folder), None, adapter_free = True) is None
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/home/owner/outputs/run"})
    )
    assert export_mod._checkpoint_load_refusal(str(folder), None, adapter_free = True)
    # The owner, and a check that already saw the adapter config, are unchanged.
    assert export_mod._checkpoint_load_refusal(str(folder), None) is None


def test_the_check_time_state_reaches_the_worker_load():
    assert (
        "adapter_free"
        in inspect.signature(orchestrator.ExportOrchestrator.load_checkpoint).parameters
    )
    assert "adapter_free" in inspect.signature(export_mod.ExportBackend.load_checkpoint).parameters
    tree = ast.parse(inspect.getsource(worker))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "load_checkpoint"
    ]
    assert any(k.arg == "adapter_free" for c in calls for k in c.keywords)


def test_a_merged_decision_checkpoint_still_loads_when_no_adapter_was_seen(tmp_path, monkeypatch):
    # Merged Clef / Laya runs carry their own weights and load no adapter base, so "no adapter at
    # check time" is true for them and must not refuse them; an adapter-only Clef run still is.
    merged = tmp_path / "clef-merged"
    merged.mkdir()
    for name in ("joint_head.safetensors", "joint_head_config.json", "config.json"):
        (merged / name).write_text("{}")
    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    monkeypatch.setattr(backend, "cleanup_memory", lambda: None, raising = False)
    loaded = []
    monkeypatch.setattr(
        backend,
        "_load_decision_checkpoint",
        lambda path, *a, **k: loaded.append(a) or (True, "ok"),
        raising = False,
    )
    assert backend.load_checkpoint(str(merged), adapter_free = True) == (True, "ok")
    assert loaded == [("clef", False)]

    adapter_only = tmp_path / "clef-adapter"
    adapter_only.mkdir()
    for name in ("joint_head.safetensors", "joint_head_config.json", "adapter_config.json"):
        (adapter_only / name).write_text("{}")
    ok, message = backend.load_checkpoint(str(adapter_only), adapter_free = True)
    assert not ok and "changed after it was checked" in message
