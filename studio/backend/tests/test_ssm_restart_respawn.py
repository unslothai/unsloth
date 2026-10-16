# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An SSM kernel installed after the worker's transformers imported the model's modeling module is
only bound by a fresh worker: the worker asks for one and the orchestrator respawns once."""

import threading
from types import SimpleNamespace

from core.inference import orchestrator as orch_mod
from core.inference import worker as inference_worker
from core.inference.orchestrator import InferenceOrchestrator


def _orchestrator(monkeypatch, responses):
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator._stop_ledger = None
    orchestrator._pending_teardowns = None
    orchestrator.__dict__.update(
        loading_models = set(),
        active_model_name = None,
        models = {},
        _proc = None,
        _resp_queue = object(),
        load_generation = 0,
        _subprocess_shutdown_lock = threading.Lock(),
    )
    spawned = []
    shutdowns = []
    alive = {"value": False}
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *_a, **_k: ([], "auto"))
    monkeypatch.setattr(orch_mod, "get_device", lambda: SimpleNamespace(value = "cpu"))
    monkeypatch.setattr("utils.transformers_version.needs_transformers_5", lambda _model: False)
    monkeypatch.setattr("utils.transformers_version.sidecar_swap_kind", lambda: None)
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: alive["value"])

    def _spawn(config, *args):
        spawned.append(config)
        alive["value"] = True

    def _shutdown(timeout = 5):
        shutdowns.append(timeout)
        alive["value"] = False
        return True

    monkeypatch.setattr(orchestrator, "_spawn_subprocess", _spawn)
    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", _shutdown)
    monkeypatch.setattr(orchestrator, "_wait_response", lambda *_a, **_k: responses.pop(0))
    return orchestrator, spawned, shutdowns


def _restart():
    return {
        "type": "loaded",
        "success": False,
        "error_kind": inference_worker.SSM_RESTART_REQUIRED,
        "message": "restart",
    }


def test_the_orchestrator_respawns_once_when_the_worker_needs_new_kernels(monkeypatch):
    responses = [
        _restart(),
        {"type": "loaded", "success": True, "model_info": {"identifier": "acme/hybrid"}},
    ]
    orchestrator, spawned, shutdowns = _orchestrator(monkeypatch, responses)
    assert orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid")) is True
    assert len(spawned) == 2
    assert shutdowns
    assert orchestrator.active_model_name == "acme/hybrid"


def test_a_second_restart_request_fails_instead_of_looping(monkeypatch):
    orchestrator, spawned, _ = _orchestrator(monkeypatch, [_restart(), _restart()])
    try:
        orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid"))
    except Exception as exc:
        assert "restart" in str(exc)
    else:
        raise AssertionError("a second restart request must fail the load")
    assert len(spawned) == 2


class _Queue(list):
    def put(self, item):
        self.append(item)


def _fake_provision(installed, model_types):
    from utils.ssm_runtime import SsmKernels
    def _provision(probe, target, hf_token, **kwargs):
        kernels = SsmKernels(True, True, True, probe, frozenset(model_types))
        return kernels, list(installed)

    return _provision


def test_the_worker_asks_for_a_restart_when_the_modeling_module_is_already_imported(monkeypatch):
    from utils import ssm_runtime

    monkeypatch.setattr(
        ssm_runtime, "provision_ssm_kernels", _fake_provision(["mamba_ssm"], {"nemotron_h"})
    )
    monkeypatch.setitem(
        inference_worker.sys.modules,
        "transformers.models.nemotron_h.modeling_nemotron_h",
        SimpleNamespace(),
    )
    queue = _Queue()
    assert inference_worker._ensure_ssm_kernels([("acme/hybrid", "acme/hybrid")], queue) is False
    assert queue[-1]["error_kind"] == inference_worker.SSM_RESTART_REQUIRED


def test_no_restart_when_nothing_new_was_installed_or_nothing_was_imported(monkeypatch):
    from utils import ssm_runtime

    monkeypatch.setitem(
        inference_worker.sys.modules,
        "transformers.models.nemotron_h.modeling_nemotron_h",
        SimpleNamespace(),
    )
    monkeypatch.setattr(ssm_runtime, "provision_ssm_kernels", _fake_provision([], {"nemotron_h"}))
    assert inference_worker._ensure_ssm_kernels([("acme/hybrid", "acme/hybrid")], _Queue())

    monkeypatch.delitem(
        inference_worker.sys.modules, "transformers.models.nemotron_h.modeling_nemotron_h"
    )
    monkeypatch.setattr(
        ssm_runtime, "provision_ssm_kernels", _fake_provision(["mamba_ssm"], {"nemotron_h"})
    )
    assert inference_worker._ensure_ssm_kernels([("acme/hybrid", "acme/hybrid")], _Queue())
