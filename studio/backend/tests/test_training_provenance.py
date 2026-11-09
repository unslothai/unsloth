# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.training.provenance import (
    ExactResumeResourcesUnavailable,
    RESOURCE_PROVENANCE_KEY,
    _MODEL_WEIGHT_RE,
    _snapshot_has_model_weights,
    attest_loaded_dataset,
    build_worker_provenance_event,
    effective_training_load_in_4bit,
    exact_resume_requires_current_4bit,
    exact_resume_resource_requirements,
    exact_dataset_snapshot_path,
    exact_model_snapshot_path,
    local_model_resume_expectation,
    normalize_worker_provenance_event,
    resource_provenance_allows_resume,
    validate_exact_model_pin,
)
from hub.utils import dataset_cache, hf_cache_state


def _shared_setup_1(tmp_path):
    snapshot = _dataset_snapshot(
        tmp_path,
        "org/dataset",
        "dataset-commit",
    )
    return snapshot


def _shared_setup_2(model_snapshot, tmp_path):
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model_snapshot),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": True,
    }

    event = build_worker_provenance_event(
        config,
        SimpleNamespace(),
        model_load_target = str(model_snapshot),
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    return config, event


def _shared_setup_3(config, model):
    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/selected",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)
    return event, updates


def _shared_setup_4(backend, config):
    backend._db_run_created = True
    backend._db_config = {
        **config,
        "model_snapshot_path": None,
        "dataset_snapshot_path": None,
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }


def _shared_setup_5(tmp_path):
    actual = _model_snapshot(
        tmp_path,
        "org/actual-4bit",
        "actual-commit",
        quantized = True,
    )
    return actual


def _shared_setup_6(dataset):
    config = {
        "model_name": "org/model",
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": True,
    }
    return config


def _shared_setup_7(tmp_path):
    from core.training.training import TrainingBackend

    config, event, _, _ = _complete_event(tmp_path)
    backend = TrainingBackend()
    return backend, config, event


def _load_training_route():
    path = Path(__file__).resolve().parents[1] / "routes" / "training.py"
    spec = importlib.util.spec_from_file_location("training_provenance_route", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _cache_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [tmp_path])
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


def test_a_4bit_run_declares_that_the_latest_sidecar_would_strand_it(tmp_path):
    """The upgrade gate must decide from stored config, since a sidecar install cannot be undone."""
    config, event, model, _dataset = _complete_event(tmp_path, load_in_4bit = True)
    persisted = {**config, **normalize_worker_provenance_event(event, config)}
    assert "require_exact_resume_resources" not in persisted
    assert "require_exact_model_resource" not in persisted

    assert exact_resume_requires_current_4bit(persisted) is True
    with patch("utils.transformers_version.latest_tier_active_for", return_value = True):
        with pytest.raises(ExactResumeResourcesUnavailable):
            effective_training_load_in_4bit(
                {**persisted, "require_exact_model_resource": True}, str(model), None
            )


def test_a_16bit_run_has_no_4bit_load_mode_to_lose(tmp_path):
    config, event, _model, _dataset = _complete_event(tmp_path, load_in_4bit = False)
    persisted = {**config, **normalize_worker_provenance_event(event, config)}
    assert exact_resume_requires_current_4bit(persisted) is False


def test_an_unresumable_provenance_is_not_made_worse_by_the_install(tmp_path):
    config, event, _model, _dataset = _complete_event(tmp_path, load_in_4bit = True)
    persisted = {**config, **normalize_worker_provenance_event(event, config)}
    persisted[RESOURCE_PROVENANCE_KEY] = {
        **persisted[RESOURCE_PROVENANCE_KEY],
        "model_status": "incomplete",
    }
    with pytest.raises(ExactResumeResourcesUnavailable):
        exact_resume_resource_requirements(persisted)
    assert exact_resume_requires_current_4bit(persisted) is False


@pytest.mark.parametrize(
    "repo_id",
    ["_owner/_repo_", f"{'a' * 96}/{'b' * 96}"],
)
def test_hub_valid_repo_ids_keep_exact_resume_provenance(tmp_path, repo_id):
    model = _model_snapshot(tmp_path, repo_id, "model-commit")
    dataset = _dataset_snapshot(tmp_path, repo_id, "dataset-commit")
    config = {
        "model_name": repo_id,
        "model_snapshot_path": str(model),
        "hf_dataset": repo_id,
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": False,
    }
    event = build_worker_provenance_event(
        config,
        SimpleNamespace(
            config = SimpleNamespace(
                _name_or_path = repo_id,
                _commit_hash = "model-commit",
            )
        ),
        model_load_target = str(model),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = True,
    )

    updates = normalize_worker_provenance_event(event, config)
    persisted = {**config, **updates}

    assert updates["actual_model_repo_id"] == repo_id
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert exact_resume_resource_requirements(persisted) == (True, True)


def test_loaded_model_metadata_attests_actual_quantized_redirect(tmp_path):
    _model_snapshot(tmp_path, "org/selected", "selected-commit")
    actual = _shared_setup_5(tmp_path)
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

    event, updates = _shared_setup_3(config, model)

    assert updates["actual_model_repo_id"] == "org/actual-4bit"
    assert updates["model_snapshot_path"] == str(actual.resolve())
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"


def test_resume_load_target_validates_redirect_snapshot_against_actual_repo(tmp_path):
    from core.training.training import resolve_training_model_load_target
    actual = _shared_setup_5(tmp_path)

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

    actual = _shared_setup_5(tmp_path)
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

    actual_repo_id, _, _, _, _, _ = route._prepare_resume_resource_provenance(
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

    event, updates = _shared_setup_3(config, model)

    assert event["model"]["status"] == "incomplete"
    assert updates["model_snapshot_path"] is None
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "incomplete"


def test_4bit_attestation_accepts_verified_runtime_quantization(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/selected", "selected-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        is_loaded_in_4bit = True,
        config = SimpleNamespace(
            _name_or_path = "org/selected",
            _commit_hash = "selected-commit",
        ),
    )
    config = {
        "model_name": "org/selected",
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": True,
    }

    event, updates = _shared_setup_3(config, model)

    assert event["model"]["status"] == "attested"
    assert event["model"]["load_mode"] == "runtime_4bit"
    assert updates["model_snapshot_path"] == str(model_snapshot)
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert updates[RESOURCE_PROVENANCE_KEY]["model_load_mode"] == "runtime_4bit"
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_mlx_native_metadata_attests_unpinned_hub_load(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        _hf_repo = "org/model",
        _unsloth_base_commit_hash = "model-commit",
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
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "unquantized"
    assert updates["model_snapshot_path"] == str(model_snapshot)
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_mlx_runtime_4bit_metadata_attests_unpinned_hub_load(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        _hf_repo = "org/model",
        _unsloth_base_commit_hash = "model-commit",
        _unsloth_quantized_source = "runtime",
        _unsloth_quantization_policy = {"enabled": True, "bits": 4},
    )
    config = _shared_setup_6(dataset)

    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/model",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "runtime_4bit"
    assert updates["model_snapshot_path"] == str(model_snapshot)
    assert resource_provenance_allows_resume({**config, **updates}) is True


class _MappingModel(dict):
    """Stand-in for mlx.nn.Module, which is also a dict; its metadata lives in attributes, not keys."""

    def __init__(self, **attrs):
        super().__init__()
        for key, value in attrs.items():
            setattr(self, key, value)


def _mlx_runtime_4bit_attrs():
    return {
        "_hf_repo": "org/model",
        "_unsloth_base_commit_hash": "model-commit",
        "_unsloth_quantized_source": "runtime",
        "_unsloth_quantization_policy": {"enabled": True, "bits": 4},
    }


def _real_mlx_module(attrs):
    nn = pytest.importorskip("mlx.nn", reason = "MLX is only installed on Apple Silicon")
    module = nn.Linear(4, 4)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(lambda attrs: SimpleNamespace(**attrs), id = "plain-object"),
        pytest.param(lambda attrs: _MappingModel(**attrs), id = "dict-subclass"),
        pytest.param(_real_mlx_module, id = "real-mlx-module"),
    ],
)
def test_mlx_runtime_4bit_attests_however_the_model_stores_metadata(tmp_path, make_model):
    """A model that is also a dict must attest like a plain object, or MLX 4-bit runs go unattested."""
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = _shared_setup_6(dataset)

    event = build_worker_provenance_event(
        config,
        make_model(_mlx_runtime_4bit_attrs()),
        model_load_target = "org/model",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["status"] == "attested"
    assert event["model"]["load_mode"] == "runtime_4bit"
    assert updates["model_snapshot_path"] == str(model_snapshot)
    assert resource_provenance_allows_resume({**config, **updates}) is True


@pytest.mark.parametrize(
    ("source", "policy"),
    [
        ("mlx_config", {"enabled": True, "bits": 4}),
        ("runtime", {"enabled": False, "bits": 4}),
        ("runtime", {"enabled": True, "bits": 8}),
        ("runtime_dense_nf4", {"enabled": True, "bits": 4}),
    ],
)
def test_mlx_runtime_4bit_attestation_requires_exact_runtime_policy(tmp_path, source, policy):
    _model_snapshot(tmp_path, "org/model", "model-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    model = SimpleNamespace(
        _hf_repo = "org/model",
        _unsloth_base_commit_hash = "model-commit",
        _unsloth_quantized_source = source,
        _unsloth_quantization_policy = policy,
    )
    config = _shared_setup_6(dataset)

    event = build_worker_provenance_event(
        config,
        model,
        model_load_target = "org/model",
        model_load_in_4bit = True,
        dataset_loaded_from_exact_snapshot = True,
    )

    assert event["model"]["status"] == "incomplete"


def test_mlx_top_level_quantization_attests_prequantized_snapshot(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 64}}),
        encoding = "utf-8",
    )
    config, event = _shared_setup_2(model_snapshot, tmp_path)
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "prequantized_4bit"
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_mlx_top_level_8bit_quantization_attests_prequantized_8bit_snapshot(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}}),
        encoding = "utf-8",
    )
    config, event = _shared_setup_2(model_snapshot, tmp_path)
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "prequantized_8bit"
    assert updates[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_mlx_mixed_quantization_widths_attest_prequantized_4_8bit_snapshot(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 8,
                    "group_size": 64,
                    "overrides": {"decoder.layers.0": {"bits": 4}},
                }
            }
        ),
        encoding = "utf-8",
    )
    config, event = _shared_setup_2(model_snapshot, tmp_path)
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "prequantized_4_8bit"
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_mlx_top_level_8bit_quantization_attests_regardless_of_the_4bit_toggle(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}}),
        encoding = "utf-8",
    )
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model_snapshot),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "load_in_4bit": False,
    }
    event = build_worker_provenance_event(
        config,
        SimpleNamespace(),
        model_load_target = str(model_snapshot),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = True,
    )
    updates = normalize_worker_provenance_event(event, config)

    assert event["model"]["load_mode"] == "prequantized_8bit"
    assert resource_provenance_allows_resume({**config, **updates}) is True


def test_prequantized_8bit_pin_refuses_resume_once_the_snapshot_declares_4bit(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8}}),
        encoding = "utf-8",
    )
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model_snapshot),
        "load_in_4bit": True,
        "resume_model_load_mode": "prequantized_8bit",
    }
    assert validate_exact_model_pin(config) == str(model_snapshot.resolve())

    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4}}),
        encoding = "utf-8",
    )
    with pytest.raises(ExactResumeResourcesUnavailable):
        validate_exact_model_pin(config)


def test_legacy_unquantized_marker_resumes_even_if_the_snapshot_now_declares_8bit(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8}}),
        encoding = "utf-8",
    )
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model_snapshot),
        "load_in_4bit": False,
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "complete",
            "model_load_mode": "unquantized",
        },
    }
    assert validate_exact_model_pin(config) == str(model_snapshot.resolve())


@pytest.mark.parametrize(
    ("stored_mode", "canonical_mode", "quantization"),
    [
        ("prequantized_08bit", "prequantized_8bit", {"bits": 8}),
        ("prequantized_8_8bit", "prequantized_8bit", {"bits": 8}),
        (
            "prequantized_8_4bit",
            "prequantized_4_8bit",
            {"bits": 8, "overrides": {"decoder.layers.0": {"bits": 4}}},
        ),
        ("prequantized_bit", "prequantized_8bit", {"bits": 8}),
    ],
)
def test_non_canonical_stored_load_mode_refuses_resume(
    tmp_path, stored_mode, canonical_mode, quantization
):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": quantization}),
        encoding = "utf-8",
    )
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model_snapshot),
        "load_in_4bit": True,
    }

    assert validate_exact_model_pin({**config, "resume_model_load_mode": canonical_mode}) == str(
        model_snapshot.resolve()
    )
    with pytest.raises(ExactResumeResourcesUnavailable):
        validate_exact_model_pin({**config, "resume_model_load_mode": stored_mode})


def test_snapshot_declaring_out_of_range_width_does_not_attest(tmp_path):
    model_snapshot = _model_snapshot(tmp_path, "org/model", "model-commit")
    (model_snapshot / "config.json").write_text(
        json.dumps({"quantization": {"bits": 32}}),
        encoding = "utf-8",
    )
    config, event = _shared_setup_2(model_snapshot, tmp_path)

    assert event["model"]["status"] == "incomplete"


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
    assert (
        normalize_worker_provenance_event(event, config)[RESOURCE_PROVENANCE_KEY]["status"]
        == "incomplete"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors-00001-of-00001",
        "model-00000-of-00001.safetensors",
        "model.safetensors",
    ],
)
def test_extension_before_shard_weight_names_count_as_model_weights(tmp_path, filename):
    """Qwen3.5 Hub shards use model.safetensors-00001-of-00001.safetensors (#15405)."""
    assert _MODEL_WEIGHT_RE.fullmatch(filename)
    snapshot = _model_snapshot(tmp_path, "unsloth/Qwen3.5-2B", "qwen35", weights = False)
    (snapshot / filename).write_bytes(b"weights")
    assert _snapshot_has_model_weights(snapshot)
    assert exact_model_snapshot_path(str(snapshot), "unsloth/Qwen3.5-2B") == str(snapshot.resolve())


def test_exact_model_snapshot_accepts_own_blob_symlink(tmp_path):
    snapshot = _model_snapshot(
        tmp_path,
        "org/model",
        "blob-backed",
        weights = False,
    )
    blob = snapshot.parent.parent / "blobs" / "weight-blob"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(os.path.relpath(blob, snapshot))

    assert exact_model_snapshot_path(str(snapshot), "org/model") == str(snapshot.resolve())


def _shared_store_blob_symlink(repo: Path, link: Path, payload: bytes) -> Path:
    sha = "c791637d" * 8
    shared = repo.parent / "blobs" / sha[:2] / sha
    shared.parent.mkdir(parents = True, exist_ok = True)
    shared.write_bytes(payload)
    (shared.parent / f"{sha}.lock").touch()
    (repo / "blobs").mkdir(exist_ok = True)
    repo_blob = repo / "blobs" / ("fe5874" + "0" * 58)
    repo_blob.symlink_to(os.path.relpath(shared, repo_blob.parent))
    link.symlink_to(os.path.relpath(repo_blob, link.parent))
    return shared


def test_exact_model_snapshot_accepts_hub_shared_blob_store(tmp_path):
    snapshot = _model_snapshot(tmp_path, "org/model", "xet-backed", weights = False)
    _shared_store_blob_symlink(snapshot.parent.parent, snapshot / "model.safetensors", b"weights")

    assert exact_model_snapshot_path(str(snapshot), "org/model") == str(snapshot.resolve())


@pytest.mark.parametrize("target", ["outside-cache", "other-repo-blobs"])
def test_exact_model_snapshot_rejects_blob_symlink_escaping_repo(
    tmp_path, tmp_path_factory, target
):
    snapshot = _model_snapshot(tmp_path, "org/model", "escaping", weights = False)
    if target == "outside-cache":
        escaped = tmp_path_factory.mktemp("elsewhere") / "blobs" / "c7" / "weights"
    else:
        escaped = tmp_path / "models--org--other" / "blobs" / "weights"
    escaped.parent.mkdir(parents = True)
    escaped.write_bytes(b"weights")
    repo_blob = snapshot.parent.parent / "blobs" / "weight-blob"
    repo_blob.parent.mkdir()
    repo_blob.symlink_to(escaped)
    (snapshot / "model.safetensors").symlink_to(os.path.relpath(repo_blob, snapshot))

    assert exact_model_snapshot_path(str(snapshot), "org/model") is None


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json"])
def test_exact_model_snapshot_rejects_external_file_symlink(tmp_path, filename):
    snapshot = _model_snapshot(tmp_path, "org/model", "external-link")
    external = tmp_path / "external" / filename
    external.parent.mkdir()
    external.write_bytes(b"external")
    (snapshot / filename).unlink()
    (snapshot / filename).symlink_to(external)

    assert exact_model_snapshot_path(str(snapshot), "org/model") is None


def test_exact_model_snapshot_rejects_cross_snapshot_symlink(tmp_path):
    snapshot = _model_snapshot(tmp_path, "org/model", "selected")
    _model_snapshot(tmp_path, "org/model", "other")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(
        os.path.join(os.pardir, "other", "model.safetensors")
    )

    assert exact_model_snapshot_path(str(snapshot), "org/model") is None


def test_exact_model_snapshot_rejects_symlinked_directory(tmp_path):
    snapshot = _model_snapshot(tmp_path, "org/model", "symlinked-directory")
    external = tmp_path / "external-directory"
    external.mkdir()
    (snapshot / "external-directory").symlink_to(
        external,
        target_is_directory = True,
    )

    assert exact_model_snapshot_path(str(snapshot), "org/model") is None


def test_exact_model_snapshot_rejects_unreadable_file(tmp_path):
    snapshot = _model_snapshot(tmp_path, "org/model", "unreadable")
    unreadable = (snapshot / "model.safetensors").resolve()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == unreadable:
            raise PermissionError("unreadable")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open):
        assert exact_model_snapshot_path(str(snapshot), "org/model") is None


def test_exact_model_snapshot_rejects_walk_error(tmp_path):
    snapshot = _model_snapshot(tmp_path, "org/model", "walk-error")

    with patch(
        "core.training.provenance.os.walk",
        side_effect = PermissionError("unreadable"),
    ):
        assert exact_model_snapshot_path(str(snapshot), "org/model") is None


def test_processed_dataset_cache_is_not_an_immutable_snapshot(tmp_path):
    processed = tmp_path / "datasets-processed" / "org___dataset"
    processed.mkdir()

    assert exact_dataset_snapshot_path(str(processed), "org/dataset") is None


def test_exact_dataset_snapshot_rejects_cross_snapshot_symlink(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    _dataset_snapshot(
        tmp_path,
        "org/dataset",
        "other-commit",
    )
    (snapshot / "train.parquet").unlink()
    (snapshot / "train.parquet").symlink_to(
        os.path.join(os.pardir, "other-commit", "train.parquet")
    )

    assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") is None


def test_exact_dataset_snapshot_rejects_unreadable_file(tmp_path):
    snapshot = _dataset_snapshot(
        tmp_path,
        "org/dataset",
        "unreadable",
    )
    unreadable = (snapshot / "train.parquet").resolve()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == unreadable:
            raise PermissionError("unreadable")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open):
        assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") is None


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

    assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") == str(snapshot.resolve())


def _loaded_dataset(*sources: str, num_bytes: int = 7):
    return SimpleNamespace(
        info = SimpleNamespace(
            download_checksums = {
                source: {"num_bytes": num_bytes, "checksum": None} for source in sources
            }
        )
    )


@pytest.mark.parametrize(
    "source",
    [
        "hf://datasets/org/dataset@dataset-commit/train.parquet",
        ("https://huggingface.co/datasets/org/dataset/resolve/dataset-commit/train.parquet"),
    ],
)
def test_loaded_hub_dataset_attests_consumed_commit_snapshot(tmp_path, source):
    snapshot = _shared_setup_1(tmp_path)
    loaded = _loaded_dataset(source)

    assert attest_loaded_dataset("org/dataset", loaded) == (str(snapshot.resolve()), None)


def test_loaded_hub_dataset_attests_local_snapshot_source(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    loaded = _loaded_dataset(str(snapshot / "train.parquet"))

    assert attest_loaded_dataset("org/dataset", loaded) == (str(snapshot.resolve()), None)


def test_loaded_hub_dataset_rejects_source_size_mismatch(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    loaded = _loaded_dataset(
        str(snapshot / "train.parquet"),
        num_bytes = 6,
    )

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


@pytest.mark.parametrize("num_bytes", [None, True, -1, "7"])
def test_loaded_hub_dataset_rejects_invalid_recorded_size(tmp_path, num_bytes):
    snapshot = _shared_setup_1(tmp_path)
    loaded = SimpleNamespace(
        info = SimpleNamespace(
            download_checksums = {
                str(snapshot / "train.parquet"): {
                    "num_bytes": num_bytes,
                    "checksum": None,
                }
            }
        )
    )

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_revision_unattested")


def test_loaded_hub_dataset_accepts_snapshot_blob_symlink(tmp_path):
    repo = tmp_path / "datasets--org--dataset"
    snapshot = repo / "snapshots" / "dataset-commit"
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (blobs / "payload").write_bytes(b"dataset")
    (snapshot / "train.parquet").symlink_to(os.path.relpath(blobs / "payload", snapshot))
    loaded = _loaded_dataset(str(snapshot / "train.parquet"))

    assert attest_loaded_dataset("org/dataset", loaded) == (str(snapshot.resolve()), None)


def test_loaded_hub_dataset_accepts_hub_shared_blob_store(tmp_path):
    repo = tmp_path / "datasets--org--dataset"
    snapshot = repo / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    _shared_store_blob_symlink(repo, snapshot / "train.parquet", b"dataset")
    loaded = _loaded_dataset(str(snapshot / "train.parquet"))

    assert attest_loaded_dataset("org/dataset", loaded) == (str(snapshot.resolve()), None)
    assert exact_dataset_snapshot_path(str(snapshot), "org/dataset") == str(snapshot.resolve())


def test_loaded_hub_dataset_rejects_local_source_symlink_outside_repo(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    external = tmp_path / "external.parquet"
    external.write_bytes(b"external")
    source = snapshot / "external.parquet"
    source.symlink_to(external)
    loaded = _loaded_dataset(str(source))

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_source_unattested")


def test_loaded_hub_dataset_rejects_cross_snapshot_symlink(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    other_snapshot = _dataset_snapshot(
        tmp_path,
        "org/dataset",
        "other-commit",
    )
    (other_snapshot / "validation.parquet").write_bytes(b"validation")
    (snapshot / "validation.parquet").symlink_to(
        os.path.join(os.pardir, "other-commit", "validation.parquet")
    )
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/validation.parquet")

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


@pytest.mark.parametrize("local", [False, True])
def test_loaded_hub_dataset_validates_each_snapshot_once_per_call(tmp_path, monkeypatch, local):
    files = 60
    snapshot = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    for index in range(files):
        (snapshot / f"train-{index:03d}.parquet").write_bytes(b"dataset")
    sources = [
        str(snapshot / f"train-{index:03d}.parquet")
        if local
        else f"hf://datasets/org/dataset@dataset-commit/train-{index:03d}.parquet"
        for index in range(files)
    ]
    original = dataset_cache.resolved_dataset_snapshot_file
    calls = []

    def counting(snapshot_path, source_path):
        calls.append(source_path)
        return original(snapshot_path, source_path)

    monkeypatch.setattr(dataset_cache, "resolved_dataset_snapshot_file", counting)
    train = _loaded_dataset(*sources[: files // 2])
    evaluation = _loaded_dataset(*sources[files // 2 :])

    for _ in range(2):
        calls.clear()
        assert attest_loaded_dataset("org/dataset", train, evaluation) == (
            str(snapshot.resolve()),
            None,
        )
        per_source = 2 if local else 1
        assert len(calls) == files + per_source * files

    (snapshot / "train-059.parquet").write_bytes(b"datase")
    assert attest_loaded_dataset("org/dataset", train, evaluation) == (
        None,
        "dataset_snapshot_unavailable",
    )


def test_loaded_hub_dataset_rejects_mixed_train_eval_commits(tmp_path):
    _dataset_snapshot(tmp_path, "org/dataset", "train-commit")
    _dataset_snapshot(tmp_path, "org/dataset", "eval-commit")
    train = _loaded_dataset("hf://datasets/org/dataset@train-commit/train.parquet")
    evaluation = _loaded_dataset("hf://datasets/org/dataset@eval-commit/train.parquet")

    assert attest_loaded_dataset("org/dataset", train, evaluation) == (
        None,
        "dataset_metadata_ambiguous",
    )


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/train.parquet",
        "hf://datasets/other/dataset@dataset-commit/train.parquet",
        "/tmp/downloads/train.parquet",
    ],
)
def test_loaded_hub_dataset_rejects_unattested_sources(tmp_path, source):
    _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")

    assert attest_loaded_dataset("org/dataset", _loaded_dataset(source)) == (
        None,
        "dataset_source_unattested",
    )


def test_loaded_hub_dataset_requires_local_commit_payload(tmp_path):
    loaded = _loaded_dataset("hf://datasets/org/dataset@evicted-commit/train.parquet")

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


def test_loaded_hub_dataset_requires_every_consumed_source_file(tmp_path):
    _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    loaded = _loaded_dataset(
        "hf://datasets/org/dataset@dataset-commit/train.parquet",
        "hf://datasets/org/dataset@dataset-commit/validation.parquet",
    )

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


def test_loaded_hub_dataset_rejects_encoded_windows_traversal(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    (snapshot.parent / "outside.parquet").write_bytes(b"outside")
    (snapshot / "..\\outside.parquet").write_bytes(b"outside")
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/%2e%2e%5coutside.parquet")

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


@pytest.mark.parametrize(
    "source_path",
    [
        "..\\outside.parquet",
        "nested\\..\\outside.parquet",
        "C:outside.parquet",
        "C:\\outside.parquet",
        "\\\\server\\share\\outside.parquet",
        "%5c%5cserver%5cshare%5coutside.parquet",
        "%2foutside.parquet",
        "%00outside.parquet",
    ],
)
def test_dataset_snapshot_source_rejects_cross_platform_paths(tmp_path, source_path):
    from core.training import provenance
    snapshot = _shared_setup_1(tmp_path)

    assert provenance._dataset_snapshot_contains(str(snapshot), source_path) is False


def test_loaded_hub_dataset_rejects_external_snapshot_symlink(tmp_path):
    snapshot = _shared_setup_1(tmp_path)
    external = tmp_path / "external.parquet"
    external.write_bytes(b"external")
    (snapshot / "external.parquet").symlink_to(external)
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/external.parquet")

    assert attest_loaded_dataset("org/dataset", loaded) == (None, "dataset_snapshot_unavailable")


def test_loaded_hub_dataset_rejects_missing_local_snapshot_source(tmp_path):
    snapshot = _shared_setup_1(tmp_path)

    assert attest_loaded_dataset(
        "org/dataset",
        _loaded_dataset(str(snapshot / "missing.parquet")),
    ) == (None, "dataset_source_unattested")


def test_loaded_hub_dataset_matches_repo_id_case_insensitively(tmp_path):
    snapshot = _dataset_snapshot(
        tmp_path,
        "Org/Dataset",
        "dataset-commit",
    )
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/train.parquet")

    assert attest_loaded_dataset("Org/Dataset", loaded) == (str(snapshot.resolve()), None)


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


def test_shared_hf_loader_attests_first_remote_dataset_load(tmp_path):
    from core.training import worker

    snapshot = _shared_setup_1(tmp_path)
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/train.parquet")
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
    }

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        lambda *_args, **_kwargs: loaded,
        lambda _message: None,
    )

    assert dataset is loaded
    assert eval_dataset is None
    assert config["dataset_snapshot_path"] == str(snapshot.resolve())
    assert config["_dataset_loaded_from_exact_snapshot"] is True


def test_first_remote_hub_load_produces_resumable_provenance(tmp_path):
    from core.training import worker

    model = _model_snapshot(
        tmp_path,
        "org/model",
        "model-commit",
    )
    dataset = _dataset_snapshot(
        tmp_path,
        "org/dataset",
        "dataset-commit",
    )
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/train.parquet")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model),
        "hf_dataset": "org/dataset",
        "load_in_4bit": False,
    }

    worker._load_hf_train_and_eval_datasets(
        config,
        None,
        lambda *_args, **_kwargs: loaded,
        lambda _message: None,
    )
    event = build_worker_provenance_event(
        config,
        object(),
        model_load_target = str(model),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = config["_dataset_loaded_from_exact_snapshot"],
    )
    persisted = {
        **config,
        **normalize_worker_provenance_event(event, config),
    }

    assert persisted["dataset_snapshot_path"] == str(dataset.resolve())
    assert persisted[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert resource_provenance_allows_resume(persisted) is True


def test_embedding_hf_loader_attests_first_remote_dataset_load(tmp_path):
    from core.training import worker

    snapshot = _shared_setup_1(tmp_path)
    loaded = _loaded_dataset("hf://datasets/org/dataset@dataset-commit/train.parquet")
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
    }

    dataset = worker._load_embedding_hf_dataset(
        config,
        lambda *_args, **_kwargs: loaded,
        lambda _message: None,
    )

    assert dataset is loaded
    assert config["dataset_snapshot_path"] == str(snapshot.resolve())
    assert config["_dataset_loaded_from_exact_snapshot"] is True


@pytest.mark.parametrize("status", ["pending", "incomplete"])
def test_unattested_current_provenance_without_hub_resources_can_resume(status):
    config = {RESOURCE_PROVENANCE_KEY: {"version": 1, "status": status}}

    assert exact_resume_resource_requirements(config) == (False, False)
    assert resource_provenance_allows_resume(config) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"status": "pending"},
        {"version": 2, "status": "pending"},
    ],
    ids = ["missing-version", "wrong-version"],
)
def test_malformed_pending_provenance_is_rejected_consistently(marker):
    config = {RESOURCE_PROVENANCE_KEY: marker}

    with pytest.raises(ExactResumeResourcesUnavailable, match = "provenance is invalid"):
        exact_resume_resource_requirements(config)
    assert resource_provenance_allows_resume(config) is False


def test_unattested_current_hub_dataset_cannot_resume_mutable_revision(tmp_path):
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model),
        "hf_dataset": "org/dataset",
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "incomplete"},
    }

    assert resource_provenance_allows_resume(config) is False


def test_current_provenance_enforces_only_hub_resources(tmp_path):
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model),
        "local_datasets": ["/datasets/train.jsonl"],
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "incomplete",
            "model_status": "attested",
            "dataset_status": "incomplete",
        },
    }

    assert exact_resume_resource_requirements(config) == (True, False)
    assert resource_provenance_allows_resume(config) is True


def test_attested_hub_identity_cannot_be_shadowed_by_relative_local_path(tmp_path, monkeypatch):
    (tmp_path / "org" / "model").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    marker = {
        "version": 1,
        "status": "incomplete",
        "model_status": "incomplete",
        "dataset_status": "incomplete",
    }

    local_config = {
        "model_name": "org/model",
        RESOURCE_PROVENANCE_KEY: marker,
    }
    assert exact_resume_resource_requirements(local_config) == (False, False)
    assert resource_provenance_allows_resume(local_config) is True

    config = {
        "model_name": "selected/model",
        "actual_model_repo_id": "org/model",
        RESOURCE_PROVENANCE_KEY: marker,
    }

    with pytest.raises(
        ExactResumeResourcesUnavailable,
        match = "model revision used by this run was not attested",
    ):
        exact_resume_resource_requirements(config)
    assert resource_provenance_allows_resume(config) is False


def test_pending_current_hub_pins_are_not_treated_as_attested(tmp_path):
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(model),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }

    assert resource_provenance_allows_resume(config) is False


def test_stop_and_save_before_attestation_allows_resume(monkeypatch):
    """Regression for #8150: Resume hidden after Stop and Save on hub models."""
    from core.training import resume

    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)
    config = {
        "model_name": "unsloth/Qwen3.5-0.8B",
        "hf_dataset": "org/dataset",
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }
    run = {
        "status": "stopped",
        "final_step": 10,
        "total_steps": 100,
        "output_dir": "/outputs/run-8150",
        "resumed_later": False,
        "config_json": json.dumps(config),
    }

    assert exact_resume_resource_requirements(config) == (False, False)
    assert resource_provenance_allows_resume(config) is True
    assert resume.can_resume_run(run) is True


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


def test_resume_eligibility_allows_pending_before_attestation_and_preserves_legacy(monkeypatch):
    from core.training import resume

    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)
    base_run = {
        "status": "stopped",
        "final_step": 1,
        "total_steps": 2,
        "output_dir": "/outputs/run",
        "resumed_later": False,
    }

    assert (
        resume.can_resume_run(
            {
                **base_run,
                "config_json": json.dumps(
                    {
                        "model_name": "org/model",
                        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
                    }
                ),
            }
        )
        is True
    )
    assert (
        resume.can_resume_run(
            {
                **base_run,
                "config_json": json.dumps({"model_name": "legacy/model"}),
            }
        )
        is True
    )


def test_parent_persists_sanitized_attested_config_and_updates_respawn_state(tmp_path):
    from core.training.training import TrainingBackend

    config, event, model, dataset = _complete_event(tmp_path)
    backend = TrainingBackend()
    backend.current_job_id = "run-1"
    _shared_setup_4(backend, config)
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
    monkeypatch.setattr(studio_db, "_schema_ready", set())
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
        config_json = '{"final":true}',
    )
    assert studio_db.update_run_config_json("run-db", '{"late":true}') is False
    assert json.loads(studio_db.get_run("run-db")["config_json"]) == {"final": True}


def test_finalization_persists_provenance_after_event_update_failure(tmp_path):
    backend, config, event = _shared_setup_7(tmp_path)
    backend.current_job_id = "run-final"
    backend._db_run_created = True
    backend._db_config = {
        **config,
        "model_snapshot_path": None,
        "dataset_snapshot_path": None,
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
        "hf_token": "hf_secret",
        "wandb_token": "wandb_secret",
        "subject": "user",
    }
    finished: list[dict] = []

    with (
        patch(
            "storage.studio_db.update_run_config_json",
            side_effect = RuntimeError("database is locked"),
        ),
        patch("core.training.training.time.sleep"),
    ):
        backend._handle_resource_provenance_event(event)

    with patch(
        "storage.studio_db.finish_run",
        side_effect = lambda **kwargs: finished.append(kwargs),
    ):
        backend._finalize_run_in_db(status = "stopped")

    persisted = json.loads(finished[0]["config_json"])
    assert persisted[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    assert "hf_token" not in persisted
    assert "wandb_token" not in persisted
    assert "subject" not in persisted


def test_finalization_waits_for_inflight_provenance(tmp_path):
    backend, config, event = _shared_setup_7(tmp_path)
    backend.current_job_id = "run-race"
    _shared_setup_4(backend, config)
    entered = threading.Event()
    release = threading.Event()
    finish_entered = threading.Event()
    finished: list[dict] = []

    def normalize(event_value, config_value):
        entered.set()
        assert release.wait(timeout = 5)
        return normalize_worker_provenance_event(event_value, config_value)

    def finish(**kwargs):
        finish_entered.set()
        finished.append(kwargs)

    with (
        patch(
            "core.training.provenance.normalize_worker_provenance_event",
            side_effect = normalize,
        ),
        patch("storage.studio_db.update_run_config_json", return_value = True),
        patch(
            "storage.studio_db.finish_run",
            side_effect = finish,
        ),
    ):
        provenance_thread = threading.Thread(
            target = backend._handle_resource_provenance_event,
            args = (event,),
        )
        provenance_thread.start()
        assert entered.wait(timeout = 5)
        finalize_thread = threading.Thread(
            target = backend._finalize_run_in_db,
            kwargs = {"status": "stopped"},
        )
        finalize_thread.start()
        assert not finish_entered.wait(timeout = 0.1)
        release.set()
        provenance_thread.join(timeout = 5)
        finalize_thread.join(timeout = 5)

    assert not provenance_thread.is_alive()
    assert not finalize_thread.is_alive()
    persisted = json.loads(finished[0]["config_json"])
    assert persisted[RESOURCE_PROVENANCE_KEY]["status"] == "complete"


def test_provenance_continues_after_failed_finalization(tmp_path):
    backend, config, event = _shared_setup_7(tmp_path)
    backend.current_job_id = "run-finalize-failure"
    _shared_setup_4(backend, config)
    finish_entered = threading.Event()
    release_finish = threading.Event()
    provenance_done = threading.Event()

    def fail_finish(**_kwargs):
        finish_entered.set()
        assert release_finish.wait(timeout = 5)
        raise RuntimeError("database is locked")

    def apply_provenance():
        backend._handle_resource_provenance_event(event)
        provenance_done.set()

    with (
        patch("storage.studio_db.finish_run", side_effect = fail_finish),
        patch("storage.studio_db.update_run_config_json", return_value = True) as update,
        patch("core.training.training.time.sleep"),
    ):
        finalize_thread = threading.Thread(
            target = backend._finalize_run_in_db,
            kwargs = {"status": "stopped"},
        )
        finalize_thread.start()
        assert finish_entered.wait(timeout = 5)
        provenance_thread = threading.Thread(target = apply_provenance)
        provenance_thread.start()
        assert not provenance_done.wait(timeout = 0.1)
        release_finish.set()
        finalize_thread.join(timeout = 5)
        provenance_thread.join(timeout = 5)

    assert not finalize_thread.is_alive()
    assert not provenance_thread.is_alive()
    assert backend._run_finalized is False
    assert backend._db_config[RESOURCE_PROVENANCE_KEY]["status"] == "complete"
    update.assert_called_once()


def _continued_adapter_run(tmp_path):
    adapter = tmp_path / "outputs" / "adapter"
    adapter.mkdir(parents = True)
    base = _model_snapshot(tmp_path, "org/base", "base-commit")
    dataset = _dataset_snapshot(tmp_path, "org/dataset", "dataset-commit")
    source_config = {
        "model_name": str(adapter),
        "actual_model_repo_id": "org/base",
        "model_snapshot_path": str(base),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "load_in_4bit": False,
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "complete",
            "model_status": "attested",
            "model_load_mode": "unquantized",
            "dataset_status": "attested",
            "reasons": [],
        },
    }
    return adapter, base, dataset, source_config


def test_a_continued_adapter_run_resumes_with_its_base_checked_after_load(tmp_path):
    from core.training import worker
    from core.training.training import _apply_cache_pins, _build_training_worker_config
    from models.training import TrainingStartRequest

    route = _load_training_route()
    adapter, base, dataset, source_config = _continued_adapter_run(tmp_path)
    request = TrainingStartRequest(
        model_name = str(adapter),
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
        load_in_4bit = False,
    )

    actual, exact_model, exact_dataset, exact_all, load_mode, overrides = (
        route._prepare_resume_resource_provenance(
            request, {"model_name": str(adapter), "config_json": source_config}
        )
    )

    # The base is not pinned as the load target, or the bare base would load without the adapter.
    assert (actual, exact_model, exact_dataset, exact_all, load_mode) == (
        None,
        False,
        True,
        False,
        None,
    )
    expected = {"repo_id": "org/base", "snapshot_path": str(base), "load_mode": "unquantized"}
    assert overrides == {"resume_expected_model": expected}
    # Kwargs shaped like the route's, through the real builder and pin steps.
    config = _build_training_worker_config(
        {
            **request.model_dump(),
            "actual_model_repo_id": actual,
            "resume_from_checkpoint": str(tmp_path / "outputs" / "run" / "checkpoint-5"),
            "require_exact_resume_resources": exact_all,
            "require_exact_model_resource": exact_model,
            "require_exact_dataset_resource": exact_dataset,
            **overrides,
        }
    )
    _apply_cache_pins(config)
    events: list[dict] = []
    assert worker._verify_config_pins(config, SimpleNamespace(put = events.append)) is True, events
    assert config["model_snapshot_path"] is None
    assert config["dataset_snapshot_path"] == str(dataset.resolve())
    assert config["resume_expected_model"] == expected

    # Refused for a moved base, the resume records its dataset alone, so a retry checks the same base.
    from core.training.provenance import initialize_resource_provenance
    from core.training.training import _sanitize_db_config

    _model_snapshot(tmp_path, "org/base", "newer-commit")
    moved = SimpleNamespace(
        config = SimpleNamespace(_name_or_path = "org/base", _commit_hash = "newer-commit")
    )
    initialize_resource_provenance(config)
    events = []
    assert not worker._emit_resource_provenance(
        SimpleNamespace(put = events.append),
        config,
        moved,
        model_load_target = str(adapter),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = True,
    )
    row = {**_sanitize_db_config(config), **normalize_worker_provenance_event(events[0], config)}
    *_, retry = route._prepare_resume_resource_provenance(
        request, {"model_name": str(adapter), "config_json": row}
    )
    assert retry == overrides


@pytest.mark.parametrize(
    "loaded_commit, quantized_base, expected_error",
    [
        ("base-commit", False, None),
        ("newer-commit", False, "no longer resolves"),
        ("base-commit", True, "trained on a 'unquantized' load"),
    ],
)
def test_a_resumed_adapter_run_stops_unless_it_loaded_the_base_it_trained_on(
    tmp_path, loaded_commit, quantized_base, expected_error
):
    from core.training import worker

    adapter, base, _dataset, source_config = _continued_adapter_run(tmp_path)
    if loaded_commit != "base-commit":
        _model_snapshot(tmp_path, "org/base", loaded_commit)
    if quantized_base:
        (base / "config.json").write_text(
            json.dumps({"quantization_config": {"load_in_4bit": True}}), encoding = "utf-8"
        )
    model = SimpleNamespace(
        config = SimpleNamespace(_name_or_path = "org/base", _commit_hash = loaded_commit)
    )
    config = {
        **source_config,
        "resume_expected_model": local_model_resume_expectation(source_config),
    }
    events: list[dict] = []

    ok = worker._emit_resource_provenance(
        SimpleNamespace(put = events.append),
        config,
        model,
        model_load_target = str(adapter),
        model_load_in_4bit = quantized_base,
        dataset_loaded_from_exact_snapshot = False,
    )

    assert ok is (expected_error is None)
    assert [event["type"] for event in events] == (
        ["resource_provenance"] if expected_error is None else ["resource_provenance", "error"]
    )
    # A refused load's base is not recorded, or it would become the next resume's reference.
    assert (events[0]["model"]["status"] == "attested") is (expected_error is None)
    assert expected_error is None or expected_error in events[1]["error"]


def test_a_continued_adapter_resume_under_the_latest_runtime_is_refused_before_loading(tmp_path):
    expected = {"repo_id": "org/base", "snapshot_path": "/base", "load_mode": "runtime_4bit"}
    # A failed resume's row, whose only check is the expectation it carries.
    stored = {
        "model_name": str(tmp_path),
        "load_in_4bit": True,
        "resume_expected_model": expected,
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
    }
    assert exact_resume_requires_current_4bit(stored) is True

    with patch("utils.transformers_version.latest_tier_active_for", return_value = True):
        with pytest.raises(ExactResumeResourcesUnavailable, match = "original 4-bit"):
            effective_training_load_in_4bit(
                {"load_in_4bit": True, "resume_expected_model": expected}, "/adapter", None
            )


def test_only_a_local_model_run_gets_a_loaded_base_expectation(tmp_path):
    _adapter, _base, _dataset, source_config = _continued_adapter_run(tmp_path)

    assert local_model_resume_expectation({**source_config, "model_name": "org/base"}) is None
    assert local_model_resume_expectation({**source_config, "actual_model_repo_id": None}) is None


_STREAM_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _streaming_config(revision = _STREAM_COMMIT):
    return {
        "model_name": "org/model",
        "hf_dataset": "org/dataset",
        "dataset_streaming": True,
        "dataset_revision": revision,
        "load_in_4bit": False,
    }


@pytest.mark.parametrize(
    "revision, attested",
    [(_STREAM_COMMIT, True), (None, False), ("main", False)],
)
def test_a_stream_is_attested_by_the_commit_it_was_pinned_to(tmp_path, revision, attested):
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    config = {**_streaming_config(revision), "model_snapshot_path": str(model)}

    event = build_worker_provenance_event(
        config,
        object(),
        model_load_target = str(model),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = False,
    )
    updates = normalize_worker_provenance_event(event, config)

    marker = updates[RESOURCE_PROVENANCE_KEY]
    assert marker["dataset_status"] == ("attested" if attested else "incomplete")
    assert marker["status"] == ("complete" if attested else "incomplete")
    assert ("dataset_streaming_unattested" in marker["reasons"]) is not attested


def test_a_stream_attests_only_the_commit_this_process_pinned(tmp_path):
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    config = {**_streaming_config(), "model_snapshot_path": str(model)}
    event = build_worker_provenance_event(
        config,
        object(),
        model_load_target = str(model),
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = False,
    )
    event["dataset"]["revision"] = "f" * 40

    marker = normalize_worker_provenance_event(event, config)[RESOURCE_PROVENANCE_KEY]

    assert marker["dataset_status"] == "incomplete"


@pytest.mark.parametrize("revision", [_STREAM_COMMIT, None])
def test_a_pinned_stream_resumes_at_its_commit(tmp_path, revision):
    from fastapi import HTTPException

    from core.training import worker
    from core.training.training import _apply_cache_pins, _build_training_worker_config
    from models.training import TrainingStartRequest

    route = _load_training_route()
    model = _model_snapshot(tmp_path, "org/model", "model-commit")
    source_config = {
        **_streaming_config(revision),
        "actual_model_repo_id": "org/model",
        "model_snapshot_path": str(model),
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "max_steps": 10,
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "complete",
            "model_status": "attested",
            "model_load_mode": "unquantized",
            "dataset_status": "attested",
            "reasons": [],
        },
    }
    request = TrainingStartRequest(
        model_name = "org/model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
        dataset_streaming = True,
        max_steps = 10,
    )
    if revision is None:
        with pytest.raises(HTTPException, match = "exact snapshots are no longer available"):
            route._prepare_resume_resource_provenance(
                request, {"model_name": "org/model", "config_json": source_config}
            )
        return

    repo, exact_model, exact_dataset, exact_all, mode, overrides = (
        route._prepare_resume_resource_provenance(
            request, {"model_name": "org/model", "config_json": source_config}
        )
    )

    assert overrides == {"dataset_revision": _STREAM_COMMIT}
    config = _build_training_worker_config(
        {
            "model_name": "org/model",
            "hf_dataset": "org/dataset",
            "dataset_streaming": True,
            "model_snapshot_path": request.model_snapshot_path,
            "actual_model_repo_id": repo,
            "resume_model_load_mode": mode,
            "resume_from_checkpoint": "/outputs/run/checkpoint-5",
            "require_exact_resume_resources": exact_all,
            "require_exact_model_resource": exact_model,
            "require_exact_dataset_resource": exact_dataset,
            "load_in_4bit": False,
            **overrides,
        }
    )
    _apply_cache_pins(config)
    events: list[dict] = []
    assert worker._verify_config_pins(config, SimpleNamespace(put = events.append)) is True, events
    assert config["dataset_revision"] == _STREAM_COMMIT


@pytest.mark.parametrize("hub_commit", [_STREAM_COMMIT, None])
def test_a_new_stream_is_pinned_to_the_current_hub_commit(monkeypatch, hub_commit):
    import huggingface_hub
    from core.training.training import _apply_cache_pins

    def _info(self, repo_id, **kw):
        if hub_commit is None:
            raise ConnectionError("hub unreachable")
        return SimpleNamespace(sha = hub_commit)

    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", _info)
    config = {"model_name": "org/model", "hf_dataset": "org/dataset", "dataset_streaming": True}

    _apply_cache_pins(config)

    assert config["dataset_revision"] == hub_commit
    assert config["dataset_snapshot_path"] is None
    assert any("cannot be resumed" in m for m in config["cache_pin_warnings"]) is (
        hub_commit is None
    )


def test_a_resume_that_dies_before_attesting_hands_on_its_checks(tmp_path):
    from core.training.provenance import initialize_resource_provenance
    from core.training.training import (
        _apply_cache_pins,
        _build_training_worker_config,
        _sanitize_db_config,
    )
    from models.training import TrainingStartRequest

    route = _load_training_route()
    adapter, _base, _dataset, source_config = _continued_adapter_run(tmp_path)
    source_config = {**source_config, **_streaming_config(), "model_name": str(adapter)}
    source_config.pop("dataset_snapshot_path")
    request = TrainingStartRequest(
        model_name = str(adapter),
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/dataset",
        dataset_streaming = True,
        load_in_4bit = False,
    )

    def resume_from(stored):
        *_, overrides = route._prepare_resume_resource_provenance(
            request, {"model_name": str(adapter), "config_json": stored}
        )
        return overrides

    first = resume_from(source_config)
    assert set(first) == {"dataset_revision", "resume_expected_model"}
    config = _build_training_worker_config(
        {**request.model_dump(), "resume_from_checkpoint": "/outputs/run/checkpoint-5", **first}
    )
    _apply_cache_pins(config)
    initialize_resource_provenance(config)

    # The row a resume leaves when it is refused or dies before its worker attests.
    assert resume_from(_sanitize_db_config(config)) == first
