# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Load and export are two calls, so another load can land between them; the orchestrator checks
# expected_checkpoint under the lock the export holds.

from __future__ import annotations

import pytest

import models


@pytest.fixture
def orchestrator(monkeypatch, tmp_path):
    from core.export import orchestrator as orchestrator_module

    orch = orchestrator_module.ExportOrchestrator()
    sent = []
    monkeypatch.setattr(orch, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(
        orch,
        "_wait_response",
        lambda kind, timeout = None: {"success": True, "message": "Exported", "output_path": None},
    )
    loaded = tmp_path / "run" / "checkpoint-30"
    loaded.mkdir(parents = True)
    orch.current_checkpoint = str(loaded)
    return orch, sent, loaded, orchestrator_module


def test_an_export_of_a_replaced_checkpoint_sends_nothing(orchestrator, tmp_path):
    orch, sent, _loaded, module = orchestrator
    result = orch.export_gguf(save_directory = "out", expected_checkpoint = str(tmp_path / "other"))
    assert result == (False, module.CHECKPOINT_CHANGED, None)
    assert sent == []
    assert orch.is_export_active() is False


@pytest.mark.parametrize("unloaded", [False, True])
def test_the_loaded_checkpoint_or_none_expected_exports(orchestrator, unloaded):
    orch, sent, loaded, _module = orchestrator
    if unloaded:
        orch.current_checkpoint = None
        assert orch.export_lora_adapter(save_directory = "out")[0] is True
    else:
        assert orch.export_merged_model(save_directory = "out", expected_checkpoint = f"{loaded}/")[0]
    assert [cmd["export_type"] for cmd in sent] == ["lora" if unloaded else "merged"]
    assert "expected_checkpoint" not in sent[0]


@pytest.mark.parametrize("name", ["BaseModel", "GGUF", "LoRAAdapter", "MergedModel"])
def test_every_export_request_takes_expected_checkpoint(name):
    model = getattr(models, f"Export{name}Request")
    assert model(save_directory = "out", expected_checkpoint = "/x").expected_checkpoint == "/x"
    assert model(save_directory = "out").expected_checkpoint is None
