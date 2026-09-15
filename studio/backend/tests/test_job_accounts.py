# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account isolation at job admission, observation, and execution boundaries."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import importlib
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from core.training.training import TrainingBackend, TrainingProgress
from utils.account_context import AccountContext, OWNER, arun_as, current_account, run_as
from utils.paths import (
    exports_root,
    outputs_root,
    rag_root,
    studio_db_path,
    tensorboard_root,
    workspace_root,
)

ALICE = AccountContext("alice-job-account", "alice")
BOB = AccountContext("bob-job-account", "bob")


@pytest.fixture(autouse = True)
def multi(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(jobs, "_retired", set())


class FakeProcess:
    pid = None
    exitcode = 0

    def __init__(self):
        self.running = False
        self.terminated = False

    def start(self):
        self.running = True

    def is_alive(self):
        return self.running

    def terminate(self):
        self.terminated = True
        self.running = False

    def join(self, timeout = None):
        pass

    kill = terminate


class FakeThread:
    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout = None):
        pass


@pytest.fixture
def training(monkeypatch):
    from core.training import training as module
    from utils import transformers_version
    from storage import studio_db

    backend = TrainingBackend()
    monkeypatch.setattr(studio_db, "mark_run_cancel_requested", lambda run_id: True)
    process = FakeProcess()
    captures = []
    monkeypatch.setattr(module, "prepare_gpu_selection", lambda *a, **k: ([], {}))
    monkeypatch.setattr(module, "_apply_cache_pins", lambda config: None)
    monkeypatch.setattr(transformers_version, "sidecar_swap_in_progress", lambda: False)
    monkeypatch.setattr(module._CTX, "Queue", queue.Queue)
    monkeypatch.setattr(module._CTX, "Process", lambda **kwargs: captures.append(kwargs) or process)
    monkeypatch.setattr(module, "account_thread", lambda **kwargs: FakeThread())
    monkeypatch.setattr(backend, "_ensure_db_run_created", lambda: None)
    monkeypatch.setattr(backend, "_start_stop_watchdog", lambda *a, **k: None)
    return backend, process, captures


def start_training(backend, account = ALICE):
    return run_as(
        account, backend.start_training, "job-a", model_name = "org/model", hf_token = "account-token"
    )


def test_training_admission_status_metrics_and_cancel_are_private(training, monkeypatch):
    from routes import training as route

    backend, proc, captures = training
    assert start_training(backend)
    assert backend.job_account == ALICE
    assert captures[0]["kwargs"]["account"] == ALICE
    assert captures[0]["kwargs"]["config"]["model_name"] == "org/model"
    backend._progress = TrainingProgress(is_training = True, status_message = "Secret model", loss = 9.5)
    backend.loss_history = [9.5]
    backend.step_history = [1]
    backend._output_dir = "/private/alice"
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    status = asyncio.run(arun_as(BOB, route.get_training_status("bob")))
    assert status.message == "Busy"
    assert status.job_id == "" and status.details is None and status.metric_history is None
    metrics = asyncio.run(arun_as(BOB, route.get_training_metrics(current_subject = "bob")))
    assert metrics.loss_history == [] and metrics.current_loss is None
    assert run_as(BOB, backend.trainer.get_training_progress).loss != 9.5
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, backend.stop_training, expected_job_id = "job-a")
    assert exc.value.status_code == 404
    assert proc.is_alive()
    with pytest.raises(HTTPException) as exc:
        start_training(backend, BOB)
    assert exc.value.status_code == 409
    assert backend.job_account == ALICE
    assert run_as(ALICE, backend.stop_training, save = False, expected_job_id = "job-a")


def test_pending_start_is_owned_and_failed_validation_releases_slot(training):
    backend, _, _ = training
    run_as(ALICE, backend.reserve_start_request, "alice-request", "a")
    assert backend.job_account == ALICE
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, backend.reserve_start_request, "bob-request", "b")
    assert exc.value.status_code == 409
    run_as(
        ALICE, backend.resolve_start_request, "alice-request", state = "rejected", message = "Failed"
    )
    jobs.refresh_job_owner(backend)
    assert backend.job_account is None
    run_as(BOB, backend.reserve_start_request, "bob-request", "b")
    assert backend.job_account == BOB
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, backend.get_start_request, "alice-request")
    assert exc.value.status_code == 404


def test_finished_training_results_remain_private_and_successor_clears_them(training):
    backend, proc, _ = training
    assert start_training(backend)
    proc.running = False
    backend._progress = TrainingProgress(is_completed = True, status_message = "Alice done", loss = 4.0)
    backend.loss_history.append(4.0)
    jobs.refresh_job_owner(backend)
    assert backend.job_account is None
    assert run_as(BOB, backend.get_training_status)[1].loss != 4.0
    run_as(BOB, backend.reserve_start_request, "bob", "b")
    assert backend.loss_history == []
    assert backend._progress.status_message != "Alice done"


@pytest.fixture
def export(monkeypatch):
    from core.export.orchestrator import ExportOrchestrator

    backend = ExportOrchestrator()
    proc = FakeProcess()

    def spawn(config):
        proc.start()
        backend._proc = proc

    monkeypatch.setattr(backend, "_spawn_subprocess", spawn)
    monkeypatch.setattr(backend, "_send_cmd", lambda command: None)
    monkeypatch.setattr(
        backend,
        "_wait_response",
        lambda *a, **k: {
            "success": True,
            "message": "done",
            "is_peft": True,
            "checkpoint": "/arbitrary/checkpoint",
        },
    )
    monkeypatch.setattr(backend, "_shutdown_subprocess", lambda **kwargs: proc.terminate() or True)
    yield backend, proc
    proc.running = False


def test_export_finished_checkpoint_and_logs_are_private(export):
    backend, proc = export
    checkpoint = str(run_as(ALICE, outputs_root) / "checkpoint")
    assert run_as(ALICE, backend.load_checkpoint, checkpoint)[0]
    backend._append_log({"line": "alice secret metric", "ts": 1})
    assert backend.job_account is None
    assert run_as(BOB, backend.get_logs_since, 0) == ([], 0)
    assert run_as(BOB, backend.get_last_op) is None
    for operation in (backend.cancel_export, backend.cleanup_memory):
        with pytest.raises(HTTPException) as exc:
            run_as(BOB, operation)
        assert exc.value.status_code == 404
    with pytest.raises(HTTPException):
        run_as(BOB, backend.export_lora_adapter, str(run_as(BOB, exports_root)))
    assert proc.is_alive()


def test_export_busy_status_hides_checkpoint(export, monkeypatch):
    from routes import export as route

    backend, _ = export
    backend._result_account = ALICE
    backend.current_checkpoint = "alice-private-model"
    backend._export_active = True
    monkeypatch.setattr(route, "get_export_backend", lambda: backend)
    result = asyncio.run(arun_as(BOB, route.get_export_status("bob")))
    assert result.active_op_kind == "busy"
    assert result.current_checkpoint is None and result.is_peft is False


def test_export_owner_single_mode_preserves_operation_and_status_bytes(export, monkeypatch):
    from routes import export as route

    backend, _ = export
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(route, "get_export_backend", lambda: backend)
    assert run_as(OWNER, backend.load_checkpoint, "/arbitrary/checkpoint") == (True, "done")
    result = asyncio.run(route.get_export_status("unsloth"))
    expected = {
        "current_checkpoint": "/arbitrary/checkpoint",
        "is_vision": False,
        "is_peft": True,
        "is_export_active": False,
        "active_op_kind": None,
        "last_op_seq": 1,
        "last_op_kind": "load_checkpoint",
        "last_op_status": "success",
        "last_op_output_path": None,
        "last_op_error": None,
    }
    assert result.model_dump_json() == json.dumps(expected, separators = (",", ":"))
    exported = backend.export_lora_adapter("/arbitrary/export")
    assert json.dumps(exported, separators = (",", ":")) == '[true,"done",null]'
    assert backend.cancel_export() is True


def test_recipe_job_status_dataset_events_and_cancel_are_private(monkeypatch):
    from core.data_recipe.jobs import manager as module

    manager = module.JobManager()
    process = FakeProcess()
    monkeypatch.setattr(module._CTX, "Queue", queue.Queue)
    monkeypatch.setattr(module._CTX, "Process", lambda **kwargs: process)
    monkeypatch.setattr(module, "account_thread", lambda **kwargs: FakeThread())
    job_id = run_as(ALICE, manager.start, recipe = {}, run = {})
    assert manager.job_account == ALICE
    assert run_as(BOB, manager.get_current_status) == {"status": "busy"}
    assert run_as(BOB, manager.get_status, job_id) is None
    assert run_as(BOB, manager.get_dataset, job_id, limit = 10) is None
    assert run_as(BOB, manager.subscribe, job_id) is None
    with pytest.raises(HTTPException):
        run_as(BOB, manager.cancel, job_id)
    assert process.is_alive()
    process.running = False
    jobs.refresh_job_owner(manager)
    assert manager.job_account is None
    assert run_as(BOB, manager.start, recipe = {}, run = {}) != job_id


class _OwnedJobService:
    def __init__(self):
        self._account_job_lock = threading.RLock()
        self._account_inflight = 0
        self._result_account = None
        self.job_account = None
        self._account_clear = None
        self.live = False

    def _account_active(self):
        return self.live

    @jobs.owned_job()
    def start(self):
        self.live = True
        return "started"


def test_owner_start_while_the_last_managed_account_is_deactivated_keeps_its_own_attribution(
    monkeypatch,
):
    mode = {"multi": True}
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: mode["multi"])
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)

    service = _OwnedJobService()
    run_as(ALICE, service.start)
    service.live = False
    jobs.refresh_job_owner(service)
    assert service._result_account == ALICE

    mode["multi"] = False
    run_as(OWNER, service.start)
    assert service._result_account == OWNER and service.job_account == OWNER

    mode["multi"] = True
    assert not run_as(OWNER, jobs.job_is_foreign, service)
    assert run_as(ALICE, jobs.job_is_foreign, service)
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.require_job_owner, service)
    assert exc.value.status_code == 404


def test_deactivating_the_last_managed_account_keeps_its_job_private(training, monkeypatch):
    from routes import training as route

    backend, proc, _ = training
    assert start_training(backend)
    backend._progress = TrainingProgress(is_training = True, status_message = "Secret model", loss = 9.5)
    backend.loss_history = [9.5]
    backend.step_history = [1]
    backend._output_dir = "/private/alice"
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)

    status = asyncio.run(arun_as(OWNER, route.get_training_status("unsloth")))
    assert status.message == "Busy"
    assert status.job_id == "" and status.details is None and status.metric_history is None
    metrics = asyncio.run(arun_as(OWNER, route.get_training_metrics(current_subject = "unsloth")))
    assert metrics.loss_history == [] and metrics.current_loss is None
    with pytest.raises(HTTPException) as exc:
        run_as(OWNER, backend.stop_training, expected_job_id = "job-a")
    assert exc.value.status_code == 404
    assert proc.is_alive()


def test_deactivating_the_last_managed_account_keeps_an_in_flight_start_contained(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)

    assert run_as(ALICE, jobs.managed_account) is True
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.validate_job_paths, {"output_dir": "/owner/private/outputs"})
    assert exc.value.status_code == 403
    assert run_as(ALICE, jobs.account_hf_token, "") is False

    args, kwargs = run_as(
        ALICE,
        jobs.account_process_spec,
        "core.training.worker",
        "run_training_process",
        {},
        {"config": {"model_name": "org/model"}},
    )
    assert args[0:2] == ("core.training.account_jobs", "run_account_child")
    assert kwargs["account"] == ALICE
    assert kwargs["job_module"] == "core.training.worker"
    assert kwargs["job_target"] == "run_training_process"


def test_an_owner_request_keeps_the_legacy_spawn_and_paths(monkeypatch):
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)

    assert jobs.managed_account() is False
    run_as(OWNER, jobs.validate_job_paths, {"output_dir": "/arbitrary/outputs"})
    args, kwargs = run_as(
        OWNER, jobs.account_process_spec, "core.training.worker", "run_training_process", {}, {}
    )
    assert args[0:2] == ("core.training.worker", "run_training_process")
    assert "account" not in kwargs


def test_reactivation_clears_the_retirement_left_by_a_failed_delete(monkeypatch):
    from core import research_runs
    from core.export.orchestrator import ExportOrchestrator
    from core.rag import folder_sync, ingestion
    from hub.services.datasets import downloads

    def fail():
        raise RuntimeError("a worker is still alive")

    broken = ExportOrchestrator()
    broken._result_account = ALICE
    monkeypatch.setattr(broken, "_account_cancel", fail)
    monkeypatch.setattr(jobs, "_services", [broken])
    monkeypatch.setattr(downloads, "retire_account_downloads", lambda: None)
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    with pytest.raises(RuntimeError):
        jobs.retire_account_jobs(ALICE)

    service = _OwnedJobService()
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, service.start)
    assert exc.value.status_code == 403

    jobs.restore_account_jobs(ALICE.account_id)
    assert run_as(ALICE, jobs.account_is_retired) is False
    assert run_as(ALICE, service.start) == "started"


def test_an_install_that_never_had_a_managed_account_does_no_job_bookkeeping(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: False)

    service = _OwnedJobService()
    service._account_job_lock = None  # any reservation would raise here
    assert run_as(OWNER, service.start) == "started"
    assert service._result_account is None and service.job_account is None


def test_diffusion_status_and_stop_are_private():
    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService()
    service._result_account = service.job_account = ALICE
    service._proc = FakeProcess()
    service._proc.start()
    service._state.update(base_model = "secret", loss = 123.0, output_dir = "/alice")
    status = run_as(BOB, service.status)
    assert status["status"] == "busy"
    assert status["base_model"] is None and status["loss"] is None and status["output_dir"] is None
    with pytest.raises(HTTPException):
        run_as(BOB, service.stop)
    assert service._proc.is_alive()


def test_diffusion_history_resolves_per_account():
    from core.training.diffusion_training_service import _runs_dir
    assert run_as(ALICE, _runs_dir) == run_as(ALICE, tensorboard_root) / "diffusion"
    assert run_as(BOB, _runs_dir) != run_as(ALICE, _runs_dir)


@pytest.mark.parametrize(
    "field",
    [
        "model_local_path",
        "checkpoint_path",
        "dataset_local_path",
        "local_datasets",
        "output_dir",
        "save_directory",
        "imatrix_path",
        "tensorboard_dir",
        "resume_from_checkpoint",
    ],
)
def test_managed_request_paths_cannot_cross_account_roots(field):
    private = str(run_as(BOB, workspace_root) / "secret")
    value = [private] if field == "local_datasets" else private
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.validate_job_paths, {field: value})
    assert exc.value.status_code == 403
    run_as(OWNER, jobs.validate_job_paths, {field: value})


def test_managed_relative_export_destinations_resolve_under_the_account_exports_root():
    for name in ("llama-3-merged", "my-run/checkpoint-100", "model-GGUF"):
        run_as(ALICE, jobs.validate_job_paths, {"save_directory": name})
    for bad in (str(run_as(BOB, exports_root) / "stolen"), "../../escape"):
        with pytest.raises(HTTPException) as exc:
            run_as(ALICE, jobs.validate_job_paths, {"save_directory": bad})
        assert exc.value.status_code == 403
    own = str(run_as(ALICE, exports_root) / "mine")
    run_as(ALICE, jobs.validate_job_paths, {"save_directory": own})
    run_as(OWNER, jobs.validate_job_paths, {"save_directory": "llama-3-merged"})


def test_managed_paths_resolve_symlinks_and_accept_private_outputs(tmp_path):
    own = run_as(ALICE, workspace_root)
    own.mkdir(parents = True)
    link = own / "link"
    link.symlink_to(tmp_path, target_is_directory = True)
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.account_path, link / "new-file")
    path = own / "outputs" / "new-run"
    assert run_as(ALICE, jobs.account_path, path) == path
    assert (
        run_as(ALICE, jobs.account_path, "org/public-model", reference = True) == "org/public-model"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"enable_wandb": True},
        {"s3_dataset": {"use_iam_role": True}},
        {"s3_dataset": {"region": "us-east-1"}},
    ],
)
def test_managed_jobs_require_explicit_credentials(config):
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.require_explicit_credentials, config)
    assert exc.value.status_code == 403
    run_as(OWNER, jobs.require_explicit_credentials, config)


def test_hf_ambient_fallback_is_disabled_only_for_managed_accounts(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "owner-secret")
    assert run_as(ALICE, jobs.account_hf_token, None) is False
    assert run_as(ALICE, jobs.account_hf_token, "alice-secret") == "alice-secret"
    assert run_as(OWNER, jobs.account_hf_token, None) is None


@pytest.mark.parametrize(
    "recipe",
    [
        {"seed_config": {"source": {"path": "/outside/private"}}},
        {"model_providers": [{"api_key_env": "OWNER_SECRET"}]},
    ],
)
def test_recipe_cannot_read_foreign_paths_or_environment_secrets(recipe):
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.validate_recipe_access, recipe)
    run_as(OWNER, jobs.validate_recipe_access, recipe)


def _child_probe(result_queue):
    from utils.paths import tmp_root

    path = outputs_root() / "worker-account.txt"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(current_account().account_id)
    result_queue.put(
        {
            "account": current_account().account_id,
            "output": str(path),
            "exports": str(exports_root()),
            "rag": str(rag_root()),
            "tensorboard": str(tensorboard_root()),
            "hf": os.environ.get("HF_TOKEN"),
            "wandb": os.environ.get("WANDB_API_KEY"),
            "aws": os.environ.get("AWS_ACCESS_KEY_ID"),
            "implicit": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"),
            "tmp": str(tmp_root()),
        }
    )


def test_spawned_worker_binds_account_before_import_and_scrubs_ambient_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "owner-hf")
    monkeypatch.setenv("WANDB_API_KEY", "owner-wandb")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "owner-aws")
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target = jobs.run_account_child,
        kwargs = {
            "account": ALICE,
            "job_module": __name__,
            "job_target": "_child_probe",
            "result_queue": result_queue,
        },
    )
    process.start()
    try:
        result = result_queue.get(timeout = 30)
        process.join(timeout = 10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout = 5)
        result_queue.close()
    assert result["account"] == ALICE.account_id
    assert Path(result["output"]).read_text() == ALICE.account_id
    assert Path(result["output"]).parent == run_as(ALICE, outputs_root)
    for name, root in (
        ("exports", exports_root),
        ("rag", rag_root),
        ("tensorboard", tensorboard_root),
    ):
        assert result[name] == str(run_as(ALICE, root))
    assert result["hf"] is result["wandb"] is result["aws"] is None
    assert result["implicit"] == "1"
    assert os.environ["HF_TOKEN"] == "owner-hf"


def test_rag_in_memory_event_queues_are_account_scoped(monkeypatch):
    from core.rag import ingestion

    monkeypatch.setattr(
        ingestion,
        "_jobs",
        {
            run_as(ALICE, jobs.account_key, "same-id"): queue.Queue(),
            run_as(BOB, jobs.account_key, "same-id"): queue.Queue(),
        },
    )
    run_as(ALICE, ingestion._emit, "same-id", {"secret": "alice"})
    assert ingestion._jobs[run_as(ALICE, jobs.account_key, "same-id")].get_nowait() == {
        "secret": "alice"
    }
    assert ingestion._jobs[run_as(BOB, jobs.account_key, "same-id")].empty()


def test_research_cancel_uses_account_keys():
    from core.research_runs import ResearchSupervisor

    supervisor = ResearchSupervisor(SimpleNamespace(state = SimpleNamespace()))
    alice = run_as(ALICE, supervisor._cancel_event, "same-id")
    bob = run_as(BOB, supervisor._cancel_event, "same-id")
    run_as(ALICE, supervisor.cancel, "same-id")
    assert alice.is_set() and not bob.is_set()


def test_retirement_cancels_only_target_accounts_jobs(training, monkeypatch):
    from core.rag import ingestion, folder_sync
    from core import research_runs

    backend, proc, _ = training
    assert start_training(backend)
    cancelled = []
    monkeypatch.setattr(backend, "_account_cancel", lambda: cancelled.append(current_account()))
    monkeypatch.setattr(jobs, "_services", {backend})
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    jobs.retire_account_jobs(BOB)
    assert cancelled == [] and proc.is_alive()
    jobs.retire_account_jobs(ALICE)
    assert cancelled == [ALICE]
    with pytest.raises(HTTPException) as exc:
        start_training(backend)
    assert exc.value.status_code == 403


def test_existing_event_stream_stops_before_successor_account_frame():
    service = SimpleNamespace(_result_account = ALICE)

    async def events():
        yield "alice frame"
        service._result_account = BOB
        yield "bob secret"

    async def consume():
        return [item async for item in jobs.account_event_stream(service, events())]

    result = asyncio.run(arun_as(ALICE, consume()))
    assert result == ["alice frame", 'event: busy\ndata: {"status":"busy"}\n\n']


def test_remote_training_cannot_borrow_the_owner_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "owner-secret")
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.validate_job_paths, {"model_name": "org/model"})
    assert exc.value.status_code == 403
    run_as(ALICE, jobs.validate_job_paths, {"model_name": "org/model", "hf_token": "alice-secret"})


def test_actual_training_s3_config_key_refuses_ambient_role():
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.validate_job_paths, {"s3_config": {"use_iam_role": True}})


def test_dataset_download_registries_do_not_share_jobs(monkeypatch):
    from hub.services.datasets import downloads

    monkeypatch.setattr(downloads, "_account_registries", {})
    alice = run_as(ALICE, downloads._account_registry)
    bob = run_as(BOB, downloads._account_registry)
    alice.claim("org/dataset", "http", repo_type = "dataset", repo_id = "org/dataset")
    assert len(alice.active_job_refs()) == 1
    assert bob.active_job_refs() == []
    assert downloads._registry.active_job_refs("org/dataset") == []
    assert run_as(OWNER, downloads._account_registry) is downloads._registry


def test_managed_account_cannot_delete_shared_dataset_cache():
    from hub.services.datasets.cache_inventory import delete_cached_dataset_response
    with pytest.raises(HTTPException) as exc:
        asyncio.run(arun_as(ALICE, delete_cached_dataset_response("org/dataset")))
    assert exc.value.status_code == 403


def test_research_claim_uses_the_database_of_each_account(monkeypatch):
    from core import research_runs

    seen = []
    monkeypatch.setattr(research_runs, "job_accounts", lambda: [OWNER, ALICE, BOB])

    def claim(worker_id):
        seen.append(current_account())
        return {"id": "alice-run"} if current_account() == ALICE else None

    monkeypatch.setattr(research_runs.db, "claim_next", claim)
    supervisor = research_runs.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace()))
    account, run = supervisor._claim_account_run()
    assert seen == [OWNER, ALICE]
    assert account == ALICE and run == {"id": "alice-run"}
    assert current_account() == OWNER


def test_research_claim_skips_an_account_whose_database_fails(monkeypatch):
    import sqlite3

    from core import research_runs

    monkeypatch.setattr(research_runs, "job_accounts", lambda: [ALICE, BOB])

    def claim(worker_id):
        if current_account() == ALICE:
            raise sqlite3.DatabaseError("file is not a database")
        return {"id": "bob-run"}

    monkeypatch.setattr(research_runs.db, "claim_next", claim)
    supervisor = research_runs.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace()))
    assert supervisor._claim_account_run() == (BOB, {"id": "bob-run"})


def test_research_claim_rotates_across_accounts(monkeypatch):
    """A busy account must not starve the next one; the supervisor runs one run at a time."""
    from core import research_runs

    monkeypatch.setattr(research_runs, "job_accounts", lambda: [ALICE, BOB])
    queued = {ALICE.account_id: ["alice-1", "alice-2", "alice-3"], BOB.account_id: ["bob-1"]}

    def claim(worker_id):
        pending = queued[current_account().account_id]
        return {"id": pending.pop(0)} if pending else None

    monkeypatch.setattr(research_runs.db, "claim_next", claim)
    supervisor = research_runs.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace()))
    claimed = [supervisor._claim_account_run() for _ in range(3)]
    assert [account for account, _ in claimed] == [ALICE, BOB, ALICE]
    assert [run["id"] for _, run in claimed] == ["alice-1", "bob-1", "alice-2"]


def test_folder_sync_claim_carries_account_data(monkeypatch):
    from core.rag import folder_sync

    monkeypatch.setattr(folder_sync, "job_accounts", lambda: [ALICE, BOB])
    monkeypatch.setattr(
        folder_sync,
        "_next_job",
        lambda: ("job-b", "folder-b") if current_account() == BOB else None,
    )
    assert folder_sync._next_account_job() == (BOB, "job-b", "folder-b")


def test_rag_lease_keys_include_the_account(monkeypatch):
    from core.rag import job_leases

    monkeypatch.setattr(job_leases, "_active", set())
    monkeypatch.setattr(job_leases, "_thread", SimpleNamespace(is_alive = lambda: True))
    run_as(ALICE, job_leases.activate, "ingestion", "same-id")
    run_as(BOB, job_leases.activate, "ingestion", "same-id")
    assert job_leases._active == {(ALICE, "ingestion", "same-id"), (BOB, "ingestion", "same-id")}


def test_owner_training_start_status_cancel_bytes_are_unchanged(training, monkeypatch):
    from routes import training as route

    backend, _, captured = training
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    assert start_training(backend, OWNER) is True
    assert captured[0]["args"][0:2] == ("core.training.worker", "run_training_process")
    assert "account" not in captured[0]["kwargs"]
    status = asyncio.run(route.get_training_status("unsloth"))
    expected = {
        "job_id": "job-a",
        "start_request_id": None,
        "start_request_state": None,
        "phase": "configuring",
        "is_training_running": True,
        "eval_enabled": False,
        "message": "Initializing training...",
        "error": None,
        "warnings": [],
        "details": {
            "epoch": 0,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "learning_rate": None,
            "output_dir": None,
        },
        "metric_history": None,
    }
    assert status.model_dump_json() == json.dumps(expected, separators = (",", ":"))
    assert backend.stop_training(save = False, expected_job_id = "job-a") is True


def test_retired_background_jobs_do_not_recreate_storage(monkeypatch):
    from core.rag import account_db
    from storage import research_runs_db

    monkeypatch.setattr(jobs, "_retired", {ALICE.account_id})
    opened = []
    monkeypatch.setattr(
        account_db._db, "get_connection", lambda: opened.append(current_account()) or "rag"
    )
    monkeypatch.setattr(
        research_runs_db,
        "_studio_connection",
        lambda: opened.append(current_account()) or "research",
    )
    for getter in (account_db.get_connection, research_runs_db.get_connection):
        with pytest.raises(RuntimeError, match = "retired"):
            run_as(ALICE, getter)
    assert opened == []
    assert run_as(BOB, account_db.get_connection) == "rag"
    assert run_as(OWNER, research_runs_db.get_connection) == "research"


def test_retirement_stops_the_captured_worker_process(training, monkeypatch):
    from core.rag import folder_sync, ingestion
    from core import research_runs
    from hub.services.datasets import downloads

    backend, process, _ = training
    assert start_training(backend)
    monkeypatch.setattr(jobs, "_services", {backend})
    monkeypatch.setattr(backend, "_account_cancel", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    monkeypatch.setattr(downloads, "retire_account_downloads", lambda: None)
    jobs.retire_account_jobs(ALICE)
    assert process.terminated and not process.is_alive()


def test_renaming_an_account_does_not_change_job_ownership(training):
    backend, _, _ = training
    assert start_training(backend)
    renamed = AccountContext(ALICE.account_id, "alice-renamed")
    assert run_as(renamed, jobs.job_is_foreign, backend) is False
    assert run_as(BOB, jobs.job_is_foreign, backend) is True


def test_rag_embedder_never_reads_the_owner_hf_login(monkeypatch):
    from core.rag import embeddings
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "owner-secret")
    assert run_as(ALICE, embeddings._ambient_hf_token) is False
    assert run_as(OWNER, embeddings._ambient_hf_token) == "owner-secret"


def test_foreign_completed_job_reports_idle(export, monkeypatch):
    from routes import export as route

    backend, _ = export
    backend._result_account = ALICE
    backend.current_checkpoint = "alice-private"
    monkeypatch.setattr(route, "get_export_backend", lambda: backend)
    result = asyncio.run(arun_as(BOB, route.get_export_status("bob")))
    assert result.current_checkpoint is None and result.active_op_kind is None
    assert result.is_export_active is False


def test_status_pump_restart_keeps_the_original_owner_after_multi_mode_begins(
    training, monkeypatch
):
    from core.training import training as module

    backend, process, _ = training
    backend._proc = process
    backend._pump_running = True
    backend._event_queue = queue.Queue()
    process.start()
    spawned = []
    monkeypatch.setattr(
        module, "account_thread", lambda **kwargs: spawned.append(kwargs) or FakeThread()
    )
    run_as(ALICE, backend._ensure_pump_alive)
    assert spawned[0]["account"] == OWNER


def test_new_account_waits_for_previous_jobs_finalizer(training):
    backend, process, _ = training
    assert start_training(backend)
    process.running = False
    backend._progress = TrainingProgress(is_completed = True)
    backend._pump_thread = SimpleNamespace(is_alive = lambda: True)
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, backend.reserve_start_request, "bob", "b")
    assert exc.value.status_code == 409
    assert backend._result_account == ALICE


def test_only_server_resolved_resource_pins_can_use_shared_hub_cache(monkeypatch):
    from utils import hf_cache_settings

    shared = run_as(OWNER, workspace_root) / "shared-model-cache"
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: shared)
    config = {"model_snapshot_path": str(shared / "model/snapshots/revision")}
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.validate_job_paths, config)
    run_as(ALICE, jobs.validate_job_paths, config, cached_resources = True)
    with pytest.raises(HTTPException):
        run_as(
            ALICE,
            jobs.validate_job_paths,
            {"model_snapshot_path": str(run_as(BOB, outputs_root))},
            cached_resources = True,
        )


def test_s3_download_temporary_files_belong_to_the_account(monkeypatch):
    from core.training import s3_dataset
    from utils.paths import tmp_root

    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_list_dataset_keys", lambda *args: ["data.jsonl"])
    client = SimpleNamespace(download_file = lambda bucket, key, path: Path(path).write_text("{}\n"))
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda config: client)
    download = run_as(ALICE, s3_dataset.prepare_s3_dataset_download, {"bucket": "bucket"})
    try:
        assert Path(download.temp_dir).parent == run_as(ALICE, tmp_root)
        assert Path(download.files[0]).read_text() == "{}\n"
    finally:
        download.cleanup()


def test_retirement_keeps_checkpoint_files_for_directory_renaming(monkeypatch):
    from core.training.training import _cleanup_cancelled_checkpoints

    root = run_as(ALICE, outputs_root) / "run" / "checkpoint-1"
    root.mkdir(parents = True)
    checkpoint = root / "adapter.safetensors"
    checkpoint.write_text("preserve")
    monkeypatch.setattr(jobs, "_retired", {ALICE.account_id})
    run_as(ALICE, _cleanup_cancelled_checkpoints, root.parent)
    assert checkpoint.read_text() == "preserve"


def test_retirement_attempts_every_service_when_one_cancel_fails(monkeypatch):
    from core.rag import folder_sync, ingestion
    from core import research_runs
    from hub.services.datasets import downloads
    from core.export.orchestrator import ExportOrchestrator

    broken = ExportOrchestrator()
    healthy = ExportOrchestrator()
    broken._result_account = healthy._result_account = ALICE
    cancelled = []

    def fail():
        raise RuntimeError("cancel failed")

    monkeypatch.setattr(broken, "_account_cancel", fail)
    monkeypatch.setattr(healthy, "_account_cancel", lambda: cancelled.append("healthy"))
    monkeypatch.setattr(jobs, "_services", [broken, healthy])
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: cancelled.append("folders"))
    monkeypatch.setattr(
        ingestion, "retire_account_ingestions", lambda: cancelled.append("ingestion")
    )
    monkeypatch.setattr(
        research_runs, "retire_account_research", lambda account: cancelled.append("research")
    )
    monkeypatch.setattr(
        downloads, "retire_account_downloads", lambda: cancelled.append("downloads")
    )
    with pytest.raises(RuntimeError, match = "keep its directories"):
        jobs.retire_account_jobs(ALICE)
    assert cancelled == ["healthy", "downloads", "ingestion", "folders", "research"]


def test_unstructured_recipe_path_lists_are_contained():
    with pytest.raises(HTTPException):
        run_as(
            ALICE,
            jobs.validate_recipe_access,
            {
                "seed_config": {
                    "source": {
                        "seed_type": "unstructured",
                        "paths": [str(run_as(BOB, workspace_root) / "notes.txt")],
                    }
                }
            },
        )


def test_hugging_face_recipe_seeds_use_explicit_account_credentials():
    source = {
        "seed_type": "hf",
        "path": "datasets/org/dataset/**/*.parquet",
        "token": "alice-token",
    }
    run_as(ALICE, jobs.validate_recipe_access, {"source": source})
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.validate_recipe_access, {"source": {**source, "path": "/owner/private"}})
    with pytest.raises(HTTPException):
        run_as(ALICE, jobs.validate_recipe_access, {"source": {**source, "token": ""}})


@pytest.mark.parametrize("account", [OWNER, ALICE])
def test_recipe_workflow_key_belongs_to_the_starting_account(account, monkeypatch):
    from routes.data_recipe import jobs as route
    from auth import storage

    captured = []
    monkeypatch.setattr(
        route, "_resolve_local_v1_endpoint", lambda request: "http://127.0.0.1:8888/v1"
    )
    monkeypatch.setattr(
        route, "_ensure_selected_local_model_loaded", lambda recipe, local_names: None
    )
    monkeypatch.setattr(
        storage,
        "create_api_key",
        lambda **kwargs: captured.append(kwargs) or ("sk-private", {"id": 42}),
    )
    recipe = {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"provider": "local", "alias": "model"}],
        "columns": [{"column_type": "llm-text", "model_alias": "model"}],
    }
    assert (
        run_as(account, route._inject_local_providers, recipe, SimpleNamespace(), "generation")
        == 42
    )
    assert captured[0]["username"] == account.username
    assert captured[0]["expect_gen"] == "generation"
    assert recipe["model_providers"][0]["api_key"] == "sk-private"


def test_diffusion_status_releases_ownership_after_child_teardown():
    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService()
    service._result_account = service.job_account = ALICE
    service._proc = FakeProcess()
    service._state.update(status = "completed", base_model = "alice-private")
    state = run_as(BOB, service.status)
    assert service.job_account is None
    assert state["status"] == "idle" and state["base_model"] is None


def test_startup_reconciliation_visits_every_account_database(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(jobs, "job_accounts", lambda: [OWNER, ALICE, BOB])
    alice_db = run_as(ALICE, studio_db_path)
    alice_db.parent.mkdir(parents = True, exist_ok = True)
    alice_db.write_bytes(b"")
    # BOB has never used Studio: reconciling would create a database for him.
    assert jobs.startup_reconciliation_accounts() == [OWNER, ALICE]
    assert not run_as(BOB, studio_db_path).exists()


def test_startup_settles_the_runs_of_every_account():
    import ast

    source = (Path(jobs.__file__).resolve().parents[2] / "main.py").read_text(encoding = "utf-8")
    direct = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"cleanup_orphaned_runs", "reconcile_orphaned_runs"}
    ]
    assert direct == []
    assert "startup_reconciliation_accounts" in source


def test_a_managed_download_blocks_the_shared_dataset_delete(monkeypatch):
    from hub.services.datasets import downloads
    from hub.services.datasets.cache_inventory import delete_cached_dataset_response

    from hub.services.datasets import cache_inventory

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(downloads, "_deleting", set(), raising = False)
    deleted = []
    monkeypatch.setattr(
        cache_inventory,
        "_delete_cached_dataset_blocking",
        lambda *args, **kwargs: deleted.append(args) or {"deleted": True},
    )
    alice = run_as(ALICE, downloads._account_registry)
    alice.claim("org/dataset", "http", repo_type = "dataset", repo_id = "org/dataset")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(delete_cached_dataset_response("org/dataset"))
    assert exc.value.status_code == 400
    assert "Cancel the active download" in exc.value.detail
    assert deleted == []


def test_an_account_registry_opened_during_a_delete_cannot_claim(monkeypatch):
    from hub.services.datasets import downloads

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(downloads, "_deleting", set(), raising = False)
    assert downloads.begin_delete("org/dataset")
    bob = run_as(BOB, downloads._account_registry)
    claimed, state = bob.claim("org/dataset", "http", repo_type = "dataset", repo_id = "org/dataset")
    assert not claimed and state == "deleting"
    downloads.end_delete("org/dataset")
    assert run_as(BOB, downloads._account_registry).claim(
        "org/dataset", "http", repo_type = "dataset", repo_id = "org/dataset"
    )[0]


def test_dataset_download_progress_refuses_another_accounts_private_repo(monkeypatch):
    from hub.services.datasets import downloads
    from hub.services.models import account_access

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(account_access, "repo_is_public", lambda *a, **k: False)
    monkeypatch.setattr(
        account_access,
        "model_grants",
        lambda: {"dataset:org/private-set"} if current_account() == ALICE else set(),
    )
    scanned = []

    async def progress(**kwargs):
        scanned.append(kwargs["repo_id"])
        return {"downloaded_bytes": 41, "cache_path": "/shared/hub/datasets--org--private-set"}

    monkeypatch.setattr(downloads.snapshot_progress, "snapshot_progress_response", progress)
    read = downloads.get_dataset_download_progress_response("org/private-set")
    assert asyncio.run(arun_as(ALICE, read))["downloaded_bytes"] == 41
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(BOB, downloads.get_dataset_download_progress_response("org/private-set"))
        )
    assert exc.value.status_code == 404
    assert scanned == ["org/private-set"]


def test_a_second_account_cannot_start_the_same_dataset_download(monkeypatch):
    from hub.services.datasets import downloads

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(downloads, "_deleting", set(), raising = False)
    monkeypatch.setattr(downloads, "resolve_cached_repo_id_case", lambda repo_id, **_k: repo_id)
    monkeypatch.setattr(
        downloads.download_registry, "download_transport_unavailable_reason", lambda _t: None
    )
    monkeypatch.setattr(downloads.download_manifest, "clear_cancel_marker", lambda *a, **k: None)
    monkeypatch.setattr(downloads.account_access, "authorize_download", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "launch_worker",
        lambda registry, key, **kwargs: launched.append(key) or "running",
    )

    def start(account):
        request = SimpleNamespace(repo_id = "Org/Data", use_xet = False, transport_mode = "http")
        return asyncio.run(arun_as(account, downloads.download_dataset_response(request)))

    assert start(ALICE)["accepted"]
    bob = start(BOB)
    assert not bob["accepted"] and not bob["attached"]
    assert launched == ["org/data"]


def test_dataset_download_request_authorizes_before_reporting_a_foreign_job(monkeypatch):
    from hub.services.datasets import downloads
    from hub.services.models import account_access

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(downloads, "_deleting", set(), raising = False)
    monkeypatch.setattr(downloads, "resolve_cached_repo_id_case", lambda repo_id, **_k: repo_id)
    monkeypatch.setattr(
        downloads.download_registry, "download_transport_unavailable_reason", lambda _t: None
    )
    monkeypatch.setattr(downloads.download_manifest, "clear_cancel_marker", lambda *a, **k: None)
    monkeypatch.setattr(
        downloads.download_lifecycle, "launch_worker", lambda registry, key, **kwargs: "running"
    )

    def authorize(repo_id, repo_type, hf_token):
        if current_account() == BOB:
            raise HTTPException(status_code = 404, detail = "Repository not found")

    monkeypatch.setattr(account_access, "authorize_download", authorize)

    def start(account):
        request = SimpleNamespace(repo_id = "org/private", use_xet = False, transport_mode = "http")
        return asyncio.run(arun_as(account, downloads.download_dataset_response(request)))

    assert start(ALICE)["accepted"]
    with pytest.raises(HTTPException) as exc:
        start(BOB)
    assert exc.value.status_code == 404


def test_dataset_transport_status_refuses_another_accounts_private_repo(monkeypatch):
    from hub.services.datasets import downloads
    from hub.services.models import account_access

    monkeypatch.setattr(downloads, "_account_registries", {})
    monkeypatch.setattr(account_access, "repo_is_public", lambda *a, **k: False)
    monkeypatch.setattr(
        account_access,
        "model_grants",
        lambda: {"dataset:org/private-set"} if current_account() == ALICE else set(),
    )
    monkeypatch.setattr(downloads, "has_active_incomplete_blobs", lambda *a: True)
    monkeypatch.setattr(
        downloads.download_registry, "read_active_transport_marker", lambda *a: "http"
    )
    monkeypatch.setattr(downloads.download_registry, "is_resumable_partial", lambda *a: True)
    read = downloads.get_dataset_transport_status_response("org/private-set")
    assert asyncio.run(arun_as(ALICE, read))["has_partial"]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(BOB, downloads.get_dataset_transport_status_response("org/private-set"))
        )
    assert exc.value.status_code == 404


def test_startup_reconciliation_settles_a_deactivated_accounts_interrupted_runs(
    tmp_path, monkeypatch
):
    from auth import storage
    from storage import studio_db

    home = tmp_path / "install"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(storage, "DB_PATH", home / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", home / "auth" / ".bootstrap_password")
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    policy.invalidate_account_cache()
    try:
        storage.create_initial_user("unsloth", "account-password", "unsloth-jwt-secret")
        storage.create_initial_user("alice", "account-password", "alice-jwt-secret")
        alice = storage.get_account("alice")
        run_as(
            alice,
            studio_db.create_run,
            id = "interrupted",
            model_name = "m",
            dataset_name = "d",
            config_json = "{}",
            started_at = "2026-01-01T00:00:00Z",
            total_steps = 10,
        )
        assert run_as(alice, studio_db.get_run, "interrupted")["status"] == "running"

        storage.set_account_active(alice.account_id, False)
        policy.invalidate_account_cache()

        reconciled = jobs.startup_reconciliation_accounts()
        assert alice.account_id in {account.account_id for account in reconciled}
        for account in reconciled:
            run_as(account, studio_db.cleanup_orphaned_runs)
        assert run_as(alice, studio_db.get_run, "interrupted")["status"] == "error"
    finally:
        policy.invalidate_account_cache()


def test_retirement_cancels_model_downloads_and_no_late_grant_recreates_the_workspace(
    tmp_path, monkeypatch
):
    """A model download of a deleted account is killed, and a worker that still completes cannot
    rebuild the workspace retirement just renamed aside."""
    import subprocess
    import sys

    from hub.services import download_lifecycle
    from hub.services.models import account_access, downloads as model_downloads
    from hub.utils import download_registry
    from routes.accounts import retire_account_roots
    from core.rag import folder_sync, ingestion
    from core import research_runs
    from hub.services.datasets import downloads as dataset_downloads
    from utils.paths import storage_roots as roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(jobs, "_services", [])
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    monkeypatch.setattr(dataset_downloads, "retire_account_downloads", lambda: None)

    repo_id = "acme/retired-model"
    registry = model_downloads._registry
    key = model_downloads._download_job_key(repo_id, None)
    claimed, state = registry.claim(
        key, download_registry.TRANSPORT_HTTP, repo_type = "model", repo_id = repo_id
    )
    assert claimed, state
    run_as(ALICE, download_lifecycle.record_download_account, registry, key)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stderr = subprocess.PIPE
    )
    assert registry.register_process(key, proc)
    workspace = run_as(ALICE, roots.workspace_root)
    workspace.mkdir(parents = True, exist_ok = True)

    try:
        retire_account_roots(ALICE)
        assert proc.poll() is not None, "the retired account's model worker is still running"
        assert not workspace.exists()

        # A worker of the same account that still reaches a clean exit must not write the account back.
        registry.drop_process(key, proc)
        finished = subprocess.Popen([sys.executable, "-c", "pass"], stderr = subprocess.PIPE)
        registry.register_process(key, finished)
        run_as(
            ALICE,
            lambda: download_lifecycle.finalize_worker_exit(
                registry,
                key,
                finished,
                hf_token = None,
                label = repo_id,
                log_prefix = "test",
                logger = download_lifecycle.logger,
                repo_type = "model",
                repo_id = repo_id,
            ),
        )
        assert not workspace.exists(), "a late completion recreated a deleted workspace"
        assert run_as(ALICE, account_access.model_grants) == set()
    finally:
        for handle in (proc, finished if "finished" in dir() else None):
            if handle is not None and handle.poll() is None:
                handle.kill()
        registry.set_job(key, "idle")


def test_retirement_is_refused_while_a_start_is_in_flight(training, monkeypatch):
    """Retiring under an admitted start deadlocked on the lifecycle lock or spawned a child afterwards."""
    backend, _, _ = training
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet([backend]))
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    paused, resume = threading.Event(), threading.Event()

    def persist():
        paused.set()
        assert resume.wait(10)

    monkeypatch.setattr(backend, "_ensure_db_run_created", persist)
    run_as(ALICE, backend.reserve_start_request, "request", "job")
    starter = threading.Thread(
        target = lambda: run_as(
            ALICE,
            backend.start_training,
            "job",
            start_request_id = "request",
            model_name = "org/model",
            hf_token = "account-token",
        ),
        daemon = True,
    )
    starter.start()
    assert paused.wait(10)
    outcome = []
    retiring = threading.Thread(
        target = lambda: outcome.append(
            _catch(jobs.AccountRetirementError, jobs.retire_account_jobs, ALICE)
        ),
        daemon = True,
    )
    retiring.start()
    retiring.join(5)
    resume.set()
    starter.join(10)
    assert not retiring.is_alive() and not starter.is_alive()
    assert isinstance(outcome[0], jobs.AccountRetirementError)


def _catch(error, fn, *args):
    try:
        return fn(*args)
    except error as exc:
        return exc


def test_deleting_an_account_cancels_its_video_render_and_keeps_its_roots(monkeypatch, tmp_path):
    from core.inference import video
    from routes.accounts import retire_account_roots
    from utils.paths import storage_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    backend = video.VideoBackend()
    monkeypatch.setattr(video, "_backend", backend)
    backend._state = SimpleNamespace(
        family = SimpleNamespace(
            name = "fam", default_fps = 24, default_num_frames = 49, frame_step = 4, frame_offset = 1
        ),
        h3_task = None,
        engine = "diffusers",
        repo_id = "r",
    )
    backend._resolve_keyframes = lambda *a, **k: (None, None, 512, 512, "t2v")
    backend._resolve_references = lambda *a, **k: None
    backend._resolve_flow_shifts = lambda *a, **k: (None, None)
    monkeypatch.setattr(video, "validate_video_request_shape", lambda *a, **k: None)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancels = []

    def render(**kwargs):
        cancels.append(kwargs["cancel_event"])
        entered.set()
        try:
            assert release.wait(10)
        finally:
            with backend._lock:
                backend._generate_job_active = False
                backend._active_generate_cancel = None
            done.set()

    monkeypatch.setattr(backend, "_run_generate", render)
    root = run_as(ALICE, storage_roots.workspace_root)
    root.mkdir(parents = True)
    run_as(ALICE, backend.begin_generate, prompt = "review", steps = 5)
    try:
        assert entered.wait(10)
        with pytest.raises(jobs.AccountRetirementError):
            retire_account_roots(ALICE)
        assert root.exists() and cancels[0].is_set()
    finally:
        release.set()
        assert done.wait(10)


def test_retirement_reaps_dataset_downloads_and_refuses_when_one_survives(tmp_path, monkeypatch):
    """A dataset worker of a deleted account must be reaped, and a failed kill must block deletion."""
    import subprocess
    import sys

    from hub.services import download_lifecycle
    from hub.services.datasets import downloads as dataset_downloads
    from hub.services.models import downloads as model_downloads
    from hub.utils import download_registry
    from core.rag import folder_sync, ingestion
    from core import research_runs

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_services", [])
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    monkeypatch.setattr(model_downloads, "retire_account_downloads", lambda: None)

    repo_id = "acme/retired-dataset"
    registry = run_as(ALICE, dataset_downloads._account_registry)
    key = dataset_downloads._download_job_key(repo_id)

    def start_worker():
        claimed, state = registry.claim(
            key, download_registry.TRANSPORT_HTTP, repo_type = "dataset", repo_id = repo_id
        )
        assert claimed, state
        run_as(ALICE, download_lifecycle.record_download_account, registry, key)
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], stderr = subprocess.PIPE
        )
        assert registry.register_process(key, worker)
        return worker

    proc = start_worker()
    try:
        jobs.retire_account_jobs(ALICE)
        assert proc.poll() is not None, "the retired account's dataset worker is still running"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        registry.drop_process(key, proc)
        registry.set_job(key, "idle")

    # A worker that cannot be killed must fail retirement instead of leaving the token in flight.
    jobs.restore_account_jobs(ALICE.account_id)
    survivor = start_worker()
    monkeypatch.setattr(
        survivor, "kill", lambda: (_ for _ in ()).throw(PermissionError("kill denied"))
    )
    try:
        with pytest.raises(jobs.AccountRetirementError):
            jobs.retire_account_jobs(ALICE)
        assert survivor.poll() is None
    finally:
        monkeypatch.undo()
        survivor.kill()
        survivor.wait()
        registry.drop_process(key, survivor)
        registry.set_job(key, "idle")


def test_a_late_finalizer_cannot_recreate_a_deleted_accounts_workspace(monkeypatch, tmp_path):
    """cancel_all() only signals, so a finalizer still running after the rename must not mkdir the account back."""
    from core.inference import image_gallery
    from routes.accounts import retire_account_roots
    from state import active_generations
    from storage import studio_db
    from utils.paths import storage_roots
    from utils.paths.storage_roots import RetiredAccountError

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    active_generations.reset_for_tests()
    root = run_as(ALICE, storage_roots.workspace_root)
    root.mkdir(parents = True)

    cancel_event = threading.Event()
    started, gate, finished = threading.Event(), threading.Event(), threading.Event()
    refused = []

    def producer():
        with active_generations.ActiveGeneration(
            cancel_event, run_id = "run-1", thread_id = "t-1", account_id = ALICE.account_id
        ):
            started.set()
            assert cancel_event.wait(10)
            # The finalizers that run after the cancel is observed: db.finish_run, then the gallery write.
            assert gate.wait(10)
            for call in (studio_db.get_connection, image_gallery.gallery_dir):
                with pytest.raises(RetiredAccountError):
                    run_as(ALICE, call)
                refused.append(call)
        finished.set()

    worker = threading.Thread(target = producer, daemon = True)
    worker.start()
    try:
        assert started.wait(10)
        retire_account_roots(ALICE)
        assert not root.exists()
    finally:
        gate.set()
    assert finished.wait(10)
    worker.join(10)
    assert len(refused) == 2
    assert not root.exists()


def test_retirement_reaps_stt_downloads_and_fences_a_parked_start(tmp_path, monkeypatch):
    """A dictation download of a deleted account is cancelled, its engine is freed for other
    accounts, and a start parked on remote validation cannot spawn a worker afterwards."""
    import subprocess
    import sys

    from core import research_runs
    from core.rag import folder_sync, ingestion
    from hub.services.datasets import downloads as dataset_downloads
    from hub.services.models import account_access as access, downloads as model_downloads
    from routes import inference

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_services", [])
    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)
    monkeypatch.setattr(dataset_downloads, "retire_account_downloads", lambda: None)
    monkeypatch.setattr(model_downloads, "retire_account_downloads", lambda: None)
    monkeypatch.setattr(inference, "_stt_download_accounts", {})
    monkeypatch.setattr(inference, "_stt_grant_pending", {})
    monkeypatch.setattr(inference, "_stt_repo_reference", lambda model, engine: model)
    monkeypatch.setattr(access, "authorize_download", lambda *a: None)
    monkeypatch.setattr(access, "account_hf_token", lambda token: "alice-token")

    # A sidecar-shaped module whose transfer is a real child process.
    handles = {"proc": None, "tokens": []}

    def start_model_download(
        model,
        hf_token = None,
        revision = None,
    ):
        handles["tokens"].append(hf_token)
        handles["proc"] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], stderr = subprocess.PIPE
        )

    def cancel_model_download():
        proc = handles["proc"]
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        proc.wait(timeout = 10)
        return True

    module = SimpleNamespace(
        start_model_download = start_model_download,
        download_status = lambda: {
            "downloading": handles["proc"] is not None and handles["proc"].poll() is None
        },
        cancel_model_download = cancel_model_download,
        is_model_downloaded = lambda model: False,
    )
    monkeypatch.setattr(inference, "_stt_download_module", lambda engine: module)
    monkeypatch.setattr(inference, "_resolve_serving_stt_engine", lambda engine: "transformers")

    try:
        run_as(
            ALICE,
            inference._start_account_stt_download,
            module,
            "transformers",
            "org/private",
            "alice-token",
        )
        worker = handles["proc"]
        assert worker.poll() is None
        jobs.retire_account_jobs(ALICE)
        assert worker.poll() is not None, "the retired account's dictation worker is still running"
        assert inference._stt_download_accounts == {}
        # The engine must not stay claimed by an account that no longer exists.
        run_as(
            BOB,
            inference._start_account_stt_download,
            module,
            "transformers",
            "org/other",
            "bob-token",
        )
        handles["proc"].kill()

        # A start parked on remote validation when retirement lands must not reach the worker.
        jobs.restore_account_jobs(ALICE.account_id)
        handles["proc"], handles["tokens"] = None, []
        parked, release = threading.Event(), threading.Event()

        def validate_remote_model(model, token = None):
            parked.set()
            assert release.wait(20)
            return {"revision": "0" * 40}

        from core.inference import stt_sidecar
        from models.inference import SttLoadRequest

        monkeypatch.setattr(stt_sidecar, "validate_remote_model", validate_remote_model)
        outcome = []
        request = SttLoadRequest(model = "org/private", engine = "transformers")
        thread = threading.Thread(
            target = lambda: outcome.append(
                _catch(
                    HTTPException,
                    lambda: asyncio.run(
                        arun_as(
                            ALICE,
                            inference.stt_download(request, current_subject = "alice", hf_token = None),
                        )
                    ),
                )
            ),
            daemon = True,
        )
        thread.start()
        assert parked.wait(20)
        jobs.retire_account_jobs(ALICE)
        release.set()
        thread.join(30)
        assert not thread.is_alive()
        assert isinstance(outcome[0], HTTPException) and outcome[0].status_code == 403
        assert handles["proc"] is None, "a download spawned for an account retired mid-validation"
        assert handles["tokens"] == []
    finally:
        for handle in (handles["proc"],):
            if handle is not None and handle.poll() is None:
                handle.kill()


def test_a_creation_in_flight_cannot_outlive_the_rename_that_retires_the_roots(
    monkeypatch, tmp_path
):
    """The tombstone is only consulted once the roots are gone, so the check and the mkdir must
    share the retirement lock: otherwise a finalizer that passed the check recreates the account."""
    from routes.accounts import retire_account_roots
    from state import active_generations
    from utils.paths import storage_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)
    monkeypatch.setattr("core.inference.mcp_client.close_mcp_sessions", lambda: None)
    monkeypatch.setattr("core.inference.mcp_client.invalidate_tool_cache", lambda: None)
    active_generations.reset_for_tests()
    root = run_as(ALICE, storage_roots.workspace_root)
    root.mkdir(parents = True)

    checked, release = threading.Event(), threading.Event()
    real_ensure_dir = storage_roots._mkdir

    def slow_ensure_dir(path):
        # Past the existence gate of ensure_account_dir; hand the CPU to the deleting thread.
        checked.set()
        assert release.wait(10)
        return real_ensure_dir(path)

    monkeypatch.setattr(storage_roots, "_mkdir", slow_ensure_dir)
    outcome, retirement = {}, {}

    def finalizer():
        try:
            outcome["path"] = run_as(
                ALICE,
                lambda: storage_roots.ensure_account_dir(storage_roots.account_path("images")),
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    def deleter():
        try:
            retire_account_roots(ALICE)
        except BaseException as exc:  # noqa: BLE001
            retirement["error"] = exc

    worker = threading.Thread(target = finalizer, daemon = True)
    worker.start()
    try:
        assert checked.wait(10)
        remover = threading.Thread(target = deleter, daemon = True)
        remover.start()
        remover.join(2)
        assert remover.is_alive(), "the rename ran while a directory creation was mid-flight"
    finally:
        release.set()
    worker.join(10)
    remover.join(10)
    assert not retirement, retirement
    assert not root.exists(), f"the deleted private root was recreated: {outcome}"
    aside = [p for p in (tmp_path / "home" / "accounts").iterdir() if p.name != ALICE.account_id]
    assert aside and (aside[0] / "images").exists(), "the in-flight creation was not renamed aside"


@pytest.mark.parametrize("engine", ["diffusers", "sd_cpp"])
def test_deleting_an_account_cancels_its_image_generation_and_keeps_its_roots(
    monkeypatch, tmp_path, engine
):
    """An image render holds the GPU and the account's uploads, and its gallery write would hit the
    retired root as a 500; retirement must stop it and refuse until it has drained, like video."""
    from core.inference import diffusion, sd_cpp_backend
    from hub.services.models import account_access as access
    from routes.accounts import retire_account_roots
    from utils.account_context import current_account_id
    from utils.paths import storage_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    monkeypatch.setattr(access, "_generation_holders", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)

    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancel = threading.Event()

    if engine == "diffusers":
        backend = diffusion.DiffusionBackend()
        monkeypatch.setattr(diffusion, "_diffusion_backend", backend)

        def denoise():
            # The real slot: it binds the cancel event and the acting account under the lock.
            with backend._generation_slot(cancel):
                entered.set()
                release.wait(10)
            done.set()
    else:
        backend = sd_cpp_backend.SdCppDiffusionBackend()
        monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_backend", backend)

        def denoise():
            with backend._generate_lock, access.media_generation_slot("diffusion"):
                with backend._lock:
                    backend._active_generate_cancel = cancel
                    backend._active_generate_account = current_account_id()
                entered.set()
                release.wait(10)
                with backend._lock:
                    backend._active_generate_cancel = None
                    backend._active_generate_account = None
            done.set()

    root = run_as(ALICE, storage_roots.workspace_root)
    root.mkdir(parents = True)
    thread = threading.Thread(target = lambda: run_as(ALICE, denoise))
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(jobs.AccountRetirementError):
            retire_account_roots(ALICE)
        assert root.exists(), "roots were retired under a live image generation"
        assert cancel.is_set(), "the account's image generation was never cancelled"
    finally:
        release.set()
        cancel.set()
        assert done.wait(10)
        thread.join(10)


def test_retirement_leaves_another_accounts_image_generation_alone(monkeypatch, tmp_path):
    """The cancel is account-scoped: BOB's render survives ALICE's deletion."""
    from core.inference import diffusion
    from hub.services.models import account_access as access
    from routes.accounts import retire_account_roots
    from utils.paths import storage_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_services", weakref.WeakSet())
    monkeypatch.setattr(access, "_generation_holders", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    for module, name in (
        ("hub.services.datasets.downloads", "retire_account_downloads"),
        ("hub.services.models.downloads", "retire_account_downloads"),
        ("core.rag.ingestion", "retire_account_ingestions"),
        ("core.rag.folder_sync", "retire_account_sync"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, lambda: None)
    monkeypatch.setattr("core.research_runs.retire_account_research", lambda account: None)

    backend = diffusion.DiffusionBackend()
    monkeypatch.setattr(diffusion, "_diffusion_backend", backend)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancel = threading.Event()

    def denoise():
        with backend._generation_slot(cancel):
            entered.set()
            release.wait(10)
        done.set()

    run_as(ALICE, storage_roots.workspace_root).mkdir(parents = True)
    thread = threading.Thread(target = lambda: run_as(BOB, denoise))
    thread.start()
    try:
        assert entered.wait(10)
        retire_account_roots(ALICE)
        assert not cancel.is_set(), "another account's image generation was cancelled"
    finally:
        release.set()
        cancel.set()
        assert done.wait(10)
        thread.join(10)
