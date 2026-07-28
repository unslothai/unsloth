# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.training.provenance import (
    RESOURCE_PROVENANCE_KEY,
    build_worker_provenance_event,
    exact_dataset_snapshot_path,
    normalize_worker_provenance_event,
    resource_provenance_allows_resume,
)
from hub.utils import dataset_cache, hf_cache_state


def _load_training_route():
    path = Path(__file__).resolve().parents[1] / "routes" / "training.py"
    spec = importlib.util.spec_from_file_location("training_provenance_route", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _cache_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [tmp_path])
    datasets_cache = tmp_path / "datasets-processed"
    datasets_cache.mkdir()
    monkeypatch.setattr(dataset_cache, "hf_datasets_cache_roots", lambda: [datasets_cache])


def _model_snapshot(
    root: Path,
    repo_id: str,
    commit: str,
    *,
    quantized: bool = False,
    weights: bool = True,
) -> Path:
    snapshot = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    config = {"quantization_config": {"load_in_4bit": True}} if quantized else {}
    (snapshot / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def _dataset_snapshot(root: Path, repo_id: str, commit: str) -> Path:
    snapshot = root / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"dataset")
    return snapshot


def _complete_event(tmp_path: Path, *, load_in_4bit: bool = False):
    model = _model_snapshot(
        tmp_path,
        "org/model",
        "model-commit",
        quantized = load_in_4bit,
    )
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": load_in_4bit,
    }
    event = build_worker_provenance_event(
        config,
        object(),
        model_load_target = str(model),
        model_load_in_4bit = load_in_4bit,
        dataset_loaded_from_exact_snapshot = True,
    )
    return config, event, model, dataset


def test_exact_direct_snapshots_produce_complete_resumable_provenance(tmp_path):
    config, event, model, dataset = _complete_event(tmp_path)

    updates = normalize_worker_provenance_event(event, config)
    persisted = {**config, **updates}

    assert updates["actual_model_repo_id"] == "org/model"
    assert updates["model_snapshot_path"] == str(model.resolve())
    assert updates["dataset_snapshot_path"] == str(dataset.resolve())
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert resource_provenance_allows_resume(persisted) is True


def test_loaded_model_metadata_attests_actual_quantized_redirect(tmp_path):
    _model_snapshot(tmp_path, "org/selected", "selected-commit")
    actual = _model_snapshot(
        tmp_path,
        "org/actual-4bit",
        "actual-commit",
        quantized = True,
    )
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        config = SimpleNamespace(
            _name_or_path = "org/actual-4bit",
            _commit_hash = "actual-commit",
        )
    )
    config = {
        "model_name": "org/selected",
        "model_snapshot_path": None,
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": True,
    }

    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/selected",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)

    assert updates["actual_model_repo_id"] == "org/actual-4bit"
    assert updates["model_snapshot_path"] == str(actual.resolve())
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"


def test_resume_load_target_validates_redirect_snapshot_against_actual_repo(tmp_path):
    from core.training.training import resolve_training_model_load_target

    actual = _model_snapshot(
        tmp_path,
        "org/actual-4bit",
        "actual-commit",
        quantized = True,
    )

    assert resolve_training_model_load_target(
        {
            "model_name": "org/selected",
            "actual_model_repo_id": "org/actual-4bit",
            "model_snapshot_path": str(actual),
            "resume_from_checkpoint": "/outputs/run/checkpoint-5",
            "load_in_4bit": True,
        }
    ) == str(actual.resolve())


def test_resume_route_restores_attested_actual_model_repo(tmp_path):
    from models.training import TrainingStartRequest

    route = _load_training_route()

    actual = _model_snapshot(
        tmp_path,
        "org/actual-4bit",
        "actual-commit",
        quantized = True,
    )
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    source_config = {
        "model_name": "org/selected",
        "actual_model_repo_id": "org/actual-4bit",
        "model_snapshot_path": str(actual),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "load_in_4bit": True,
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "complete",
            "model_status": "attested",
            "dataset_status": "attested",
            "reasons": [],
        },
    }
    request = TrainingStartRequest(
        model_name = "org/selected",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
    )

    actual_repo_id = route._apply_resume_resource_provenance(
        request,
        {
            "model_name": "org/selected",
            "config_json": source_config,
        },
    )

    assert actual_repo_id == "org/actual-4bit"
    assert request.model_snapshot_path == str(actual)
    assert request.dataset_snapshot_path == str(dataset)


def test_4bit_attestation_rejects_full_precision_selected_snapshot(tmp_path):
    _model_snapshot(tmp_path, "org/selected", "selected-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        config = SimpleNamespace(
            _name_or_path = "org/selected",
            _commit_hash = "selected-commit",
        )
    )
    config = {
        "model_name": "org/selected",
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": True,
    }

    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/selected",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["status"] == "incomplete"
    assert updates["model_snapshot_path"] is None
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "incomplete"


def test_config_only_snapshot_cannot_attest_model_weights(tmp_path):
    _model_snapshot(tmp_path, "org/model", "config-only", weights = False)
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        config = SimpleNamespace(
            _name_or_path = "org/model",
            _commit_hash = "config-only",
        )
    )
    config = {
        "model_name": "org/model",
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": False,
    }

    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/model",
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = True,
    )

    assert event["model"]["status"] == "incomplete"
    assert normalize_worker_provenance_event(event, config)[RESOURCE_PROVENANCE_KEY][
        "status"
    ] == "incomplete"


def test_processed_dataset_cache_is_not_an_immutable_snapshot(tmp_path):
    processed = tmp_path / "datasets-processed" / "org___dataset"
    processed.mkdir()

    assert exact_dataset_snapshot_path(str(processed), "org/dataset") is None


def test_dataset_snapshot_without_data_cannot_attest_provenance(tmp_path):
    snapshot = tmp_path / "datasets--org--dataset" / "snapshots" / "empty"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("metadata only", encoding = "utf-8")
    (snapshot / "dataset_infos.json").write_text("{}", encoding = "utf-8")

    assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") is None


def test_resume_revalidates_dataset_payload(tmp_path):
    config, event, _, dataset = _complete_event(tmp_path)
    persisted = {**config, **normalize_worker_provenance_event(event, config)}
    (dataset / "train.parquet").unlink()

    assert resource_provenance_allows_resume(persisted) is False


@pytest.mark.parametrize(
    "filename",
    [
        "train.arrow",
        "train.tsv",
        "train.txt",
        "images.zip",
        "sample.png",
        "sample.wav",
    ],
)
def test_supported_dataset_payloads_can_attest_exact_snapshot(tmp_path, filename):
    snapshot = tmp_path / "datasets--org--dataset" / "snapshots" / "payload"
    snapshot.mkdir(parents = True)
    (snapshot / filename).write_bytes(b"payload")

    assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") == str(
        snapshot.resolve()
    )


def test_shared_hf_loader_marks_only_successful_exact_dataset_load(tmp_path):
    from core.training import worker

    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "train_split": "train",
    }
    cached = object()

    with patch.object(worker, "_load_cached_dataset_for_config", return_value = cached):
        loaded, eval_dataset = worker._load_hf_train_and_eval_datasets(
            config,
            None,
            lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
            lambda _message: None,
        )

    assert loaded is cached
    assert eval_dataset is None
    assert config["_dataset_loaded_from_exact_snapshot"] is True


@pytest.mark.parametrize("status", ["pending", "incomplete"])
def test_unattested_current_provenance_preserves_resume_compatibility(status):
    assert resource_provenance_allows_resume(
        {RESOURCE_PROVENANCE_KEY: {"version": 1, "status": status}}
    ) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"version": 99, "status": "complete"},
        {"version": 1, "status": "unknown"},
        "complete",
    ],
)
def test_malformed_provenance_is_rejected_while_legacy_is_unchanged(marker):
    assert resource_provenance_allows_resume({RESOURCE_PROVENANCE_KEY: marker}) is False
    assert resource_provenance_allows_resume({"model_name": "legacy/model"}) is True


def test_resume_eligibility_preserves_pending_and_legacy_compatibility(monkeypatch):
    from core.training import resume

    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)
    base_run = {
        "status": "stopped",
        "final_step": 1,
        "total_steps": 2,
        "output_dir": "/outputs/run",
        "resumed_later": False,
    }

    assert resume.can_resume_run(
        {
            **base_run,
            "config_json": json.dumps(
                {
                    "model_name": "org/model",
                    RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
                }
            ),
        }
    ) is True
    assert resume.can_resume_run(
        {
            **base_run,
            "config_json": json.dumps({"model_name": "legacy/model"}),
        }
    ) is True


def test_parent_persists_sanitized_attested_config_and_updates_respawn_state(tmp_path):
    from core.training.training import TrainingBackend

    config, event, model, dataset = _complete_event(tmp_path)
    backend = TrainingBackend()
    backend.current_job_id = "run-1"
    backend._db_run_created = True
    backend._db_config = {
        **config,
        "model_snapshot_path": None,
        "dataset_snapshot_path": None,
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }
    backend._last_full_config = {
        **backend._db_config,
        "hf_token": "hf_secret",
        "wandb_token": "wandb_secret",
        "subject": "user",
    }
    saved: list[tuple[str, dict]] = []

    def persist(run_id: str, config_json: str) -> bool:
        saved.append((run_id, json.loads(config_json)))
        return True

    with patch("storage.studio_db.update_run_config_json", side_effect = persist):
        backend._handle_resource_provenance_event(event)

    assert backend._db_config["model_snapshot_path"] == str(model.resolve())
    assert backend._db_config["dataset_snapshot_path"] == str(dataset.resolve())
    assert backend._last_full_config["hf_token"] == "hf_secret"
    assert backend._last_full_config["model_snapshot_path"] == str(model.resolve())
    assert saved[0][0] == "run-1"
    assert saved[0][1][RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert "hf_token" not in saved[0][1]
    assert "wandb_token" not in saved[0][1]
    assert "subject" not in saved[0][1]


def test_incomplete_worker_event_clears_stale_parent_pins(tmp_path):
    from core.training.provenance import incomplete_worker_provenance_event
    from core.training.training import TrainingBackend

    config, _, _, _ = _complete_event(tmp_path)
    backend = TrainingBackend()
    backend.current_job_id = "run-2"
    backend._db_run_created = True
    backend._db_config = {
        **config,
        "actual_model_repo_id": "org/model",
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }
    backend._last_full_config = dict(backend._db_config)

    with patch("storage.studio_db.update_run_config_json", return_value = True):
        backend._handle_resource_provenance_event(
            incomplete_worker_provenance_event(
                "model_cache_fallback",
                "dataset_cache_fallback",
            )
        )

    assert backend._db_config["actual_model_repo_id"] is None
    assert backend._db_config["model_snapshot_path"] is None
    assert backend._db_config["dataset_snapshot_path"] is None
    assert backend._db_config[RESOURCE_PROVENANCE_KEY]["status"] == "incomplete"


def test_db_config_update_only_mutates_running_run(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.create_run(
        id = "run-db",
        model_name = "org/model",
        dataset_name = "org/dataset",
        config_json = '{"before":true}',
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )

    assert studio_db.update_run_config_json("run-db", '{"after":true}') is True
    assert json.loads(studio_db.get_run("run-db")["config_json"]) == {"after": True}

    studio_db.finish_run(
        id = "run-db",
        status = "stopped",
        ended_at = "2026-01-01T00:01:00Z",
        final_step = 1,
        final_loss = 1.0,
        duration_seconds = 60.0,
        loss_sparkline = "[]",
    )
    assert studio_db.update_run_config_json("run-db", '{"late":true}') is False
    assert json.loads(studio_db.get_run("run-db")["config_json"]) == {"after": True}
