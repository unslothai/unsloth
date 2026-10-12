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


class _Worker:
    pid = 0

    def __init__(self):
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout = None):
        pass


def test_cancel_with_an_op_id_stops_only_that_op(orchestrator):
    orch, _sent, _loaded, _module = orchestrator
    worker = orch._proc = _Worker()
    assert orch._begin_op("theirs")
    # Another caller's op is running: the worker is left alone.
    assert orch.cancel_export(expected_op_id = "mine") is False
    assert worker.terminated is False
    assert orch.cancel_export(expected_op_id = "theirs") is True
    assert worker.terminated is True


def test_an_op_cancelled_before_it_started_is_refused_when_it_starts(orchestrator):
    orch, sent, _loaded, module = orchestrator
    orch._proc = _Worker()
    assert orch.cancel_export(expected_op_id = "queued") is False
    assert orch.export_gguf(save_directory = "out", op_id = "queued") == (
        False,
        module.OP_CANCELLED,
        None,
    )
    assert sent == []
    # Refused once: the id is spent, and an op without an id runs as before.
    assert orch.export_gguf(save_directory = "out", op_id = "queued")[0] is True
    assert orch.export_gguf(save_directory = "out")[0] is True
    assert orch._active_op_id is None


def test_cancel_without_an_op_id_stops_whatever_runs(orchestrator):
    orch, _sent, _loaded, _module = orchestrator
    worker = orch._proc = _Worker()
    orch._begin_op("theirs")
    assert orch.cancel_export() is True
    assert worker.terminated is True


@pytest.mark.parametrize("name", ["BaseModel", "GGUF", "LoRAAdapter", "MergedModel"])
def test_every_load_and_export_request_takes_an_op_id(name):
    model = getattr(models, f"Export{name}Request")
    assert model(save_directory = "out", op_id = "op-1").op_id == "op-1"
    assert models.LoadCheckpointRequest(checkpoint_path = "/x", op_id = "op-2").op_id == "op-2"
    assert models.ExportCancelRequest().expected_op_id is None


def test_a_cancel_landing_while_the_old_worker_is_torn_down_stops_the_new_one(
    orchestrator, monkeypatch
):
    # The matched cancel finds no live worker to kill, so the load must not carry on in a fresh one.
    orch, _sent, _loaded, module = orchestrator
    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "sidecar_swap_in_progress", lambda: False)
    spawned, shut = [], []

    def shutdown(timeout = 10.0):
        shut.append(orch._proc)
        if len(shut) == 1:
            orch._proc = None
            assert orch.cancel_export(expected_op_id = "load-1") is False
        return True

    def spawn(config):
        orch._proc = _Worker()
        spawned.append(orch._proc)

    monkeypatch.setattr(orch, "_shutdown_subprocess", shutdown)
    monkeypatch.setattr(orch, "_spawn_subprocess", spawn)
    monkeypatch.setattr(
        orch,
        "_wait_response",
        lambda *a, **k: pytest.fail("a cancelled load must not wait for the worker"),
    )
    monkeypatch.setattr("utils.hardware.gpu_ids_with_torch_kernels", lambda: [])
    orch._proc = _Worker()
    assert orch.load_checkpoint("/ckpt", op_id = "load-1") == (False, module.OP_CANCELLED)
    assert len(spawned) == 1 and shut[-1] is spawned[0]
    assert orch.current_checkpoint is None
    assert orch.get_last_op()["status"] == "cancelled"
