# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An SSM kernel installed after the worker's transformers imported the model's modeling module is
only bound by a fresh worker: the worker asks for one and the orchestrator respawns once."""

import sys
import threading
from types import SimpleNamespace

import pytest

from core.inference import orchestrator as orch_mod
from core.inference import worker as inference_worker
from core.inference.orchestrator import InferenceOrchestrator


def _orchestrator(
    monkeypatch,
    responses,
    shutdown_results = (),
):
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
        spawned.append(dict(config))  # the orchestrator reuses one sub_config
        alive["value"] = True

    results = list(shutdown_results)

    def _shutdown(timeout = 5):
        shutdowns.append(timeout)
        exited = results.pop(0) if results else True
        if exited:
            alive["value"] = False
        return exited

    def _wait(*_a, **_k):
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(orchestrator, "_spawn_subprocess", _spawn)
    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", _shutdown)
    monkeypatch.setattr(orchestrator, "_wait_response", _wait)
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


def _loaded():
    return {"type": "loaded", "success": True, "model_info": {"identifier": "acme/hybrid"}}


def _stall():
    from utils.hf_xet_fallback import DownloadStallError
    return DownloadStallError("stalled")


def test_a_worker_that_survives_shutdown_is_not_respawned_over(monkeypatch):
    # The worker outlived terminate/kill: its handle stays, and no second worker is spawned over
    # one still holding GPU memory (the model-switch guard's rule).
    orchestrator, spawned, _ = _orchestrator(
        monkeypatch, [_restart(), _loaded()], shutdown_results = [False]
    )
    try:
        orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid"))
    except RuntimeError as exc:
        assert "did not exit" in str(exc)
    else:
        raise AssertionError("the load must fail while the old worker survives")
    assert len(spawned) == 1
    assert orchestrator.active_model_name is None


def test_a_survivor_also_blocks_the_xet_retry(monkeypatch):
    orchestrator, spawned, _ = _orchestrator(
        monkeypatch, [_stall(), _loaded()], shutdown_results = [False]
    )
    try:
        orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid"))
    except RuntimeError as exc:
        assert "did not exit" in str(exc)
    else:
        raise AssertionError("the load must fail while the old worker survives")
    assert len(spawned) == 1


def test_an_ssm_restart_keeps_the_xet_retry(monkeypatch):
    orchestrator, spawned, _ = _orchestrator(monkeypatch, [_restart(), _stall(), _loaded()])
    assert orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid")) is True
    assert [config["disable_xet"] for config in spawned] == [False, False, True]


def test_a_xet_retry_keeps_the_ssm_restart(monkeypatch):
    orchestrator, spawned, _ = _orchestrator(monkeypatch, [_stall(), _restart(), _loaded()])
    assert orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid")) is True
    assert [config["disable_xet"] for config in spawned] == [False, True, True]


def test_spawns_are_bounded_at_three_and_the_stall_message_is_accurate(monkeypatch):
    orchestrator, spawned, _ = _orchestrator(monkeypatch, [_restart(), _stall(), _stall()])
    try:
        orchestrator.load_model(SimpleNamespace(identifier = "acme/hybrid"))
    except RuntimeError as exc:
        assert "even with HF_HUB_DISABLE_XET=1" in str(exc)
    else:
        raise AssertionError("a second stall must fail the load")
    assert len(spawned) == 3


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


# Only modeling modules capture kernel functions at import; a config or tokenizer import does not.


def test_a_configuration_import_alone_needs_no_restart(monkeypatch):
    from utils import ssm_runtime

    monkeypatch.setitem(
        sys.modules, "transformers.models.nemotron_h.configuration_nemotron_h", SimpleNamespace()
    )
    monkeypatch.setitem(sys.modules, "transformers.models.nemotron_h", SimpleNamespace())
    monkeypatch.delitem(
        sys.modules, "transformers.models.nemotron_h.modeling_nemotron_h", raising = False
    )
    assert ssm_runtime.modeling_modules_needing_restart({"nemotron_h"}) == []
    queue = _Queue()
    monkeypatch.setattr(
        ssm_runtime, "provision_ssm_kernels", _fake_provision(["mamba_ssm"], {"nemotron_h"})
    )
    assert inference_worker._ensure_ssm_kernels([("acme/hybrid", "acme/hybrid")], queue) is True


@pytest.mark.parametrize(
    "module",
    [
        # transformers 5.x spells "-" and "." in the repo name out.
        "transformers_modules.nvidia.NVIDIA_hyphen_Nemotron_hyphen_Nano_hyphen_9B_hyphen_v2"
        ".0123abcd.modeling_nemotron_h",
        # 4.x keeps them.
        "transformers_modules.nvidia.NVIDIA-Nemotron-Nano-9B-v2.0123abcd.modeling_nemotron_h",
    ],
)
def test_an_imported_remote_code_modeling_module_needs_a_restart(monkeypatch, module):
    from utils import ssm_runtime

    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    # Found by the checkpoint it came from, even with no model type resolved.
    assert ssm_runtime.modeling_modules_needing_restart(set(), (model,)) == [module]
    assert module in ssm_runtime.modeling_modules_needing_restart({"nemotron_h"})
    monkeypatch.setattr(ssm_runtime, "provision_ssm_kernels", _fake_provision(["mamba_ssm"], set()))
    queue = _Queue()
    assert inference_worker._ensure_ssm_kernels([(model, model)], queue) is False
    assert queue[-1]["error_kind"] == inference_worker.SSM_RESTART_REQUIRED


def test_another_remote_models_modules_are_not_a_reason_to_restart(monkeypatch):
    from utils import ssm_runtime

    other = "transformers_modules.acme.Some_hyphen_Chat.0123.modeling_chat"
    monkeypatch.setitem(sys.modules, other, SimpleNamespace())
    config = "transformers_modules.nvidia.Nemotron_hyphen_H.0123.configuration_nemotron_h"
    monkeypatch.setitem(sys.modules, config, SimpleNamespace())
    found = ssm_runtime.modeling_modules_needing_restart({"nemotron_h"}, ("nvidia/Nemotron-H",))
    assert other not in found and config not in found
