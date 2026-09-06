# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account isolation at job admission, observation, and execution boundaries."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from core.training.training import TrainingBackend, TrainingProgress
from utils.account_context import AccountContext, OWNER, arun_as, current_account, run_as
from utils.paths import exports_root, outputs_root, rag_root, tensorboard_root, workspace_root

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
