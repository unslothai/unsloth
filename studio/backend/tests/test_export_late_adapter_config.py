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
