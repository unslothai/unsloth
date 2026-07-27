# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from hub.utils import hf_cache_state
from models.training import TrainingStartRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse = True)
def _known_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [tmp_path])


def _load_route_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / "routes" / "training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(**overrides) -> TrainingStartRequest:
    payload = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "hf_dataset": "org/dataset",
    }
    payload.update(overrides)
    return TrainingStartRequest(**payload)


def _refusing_backend() -> SimpleNamespace:
    return SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: pytest.fail("backend should not start"),
    )


def _start(route, request):
    return asyncio.run(route.start_training(request, current_subject = "test-user"))


@pytest.mark.parametrize(
    ("model_format", "expected"),
    [
        ("gguf", "GGUF models are inference-only"),
        ("adapter", "Adapter models are inference-only"),
    ],
)
def test_start_rejects_untrainable_model_formats(model_format, expected):
    route = _load_route_module(f"training_route_reject_{model_format}")
    request = _request(model_format = model_format)

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert expected in exc_info.value.detail


def test_start_rejects_adapter_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_adapter_dir")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert "Adapter-only local models" in exc_info.value.detail


def test_start_rejects_partial_adapter_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_partial_adapter_dir")
    (tmp_path / "adapter_config.json").write_text("{}")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "Adapter-only local models" in exc_info.value.detail


def test_start_rejects_gguf_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_gguf_dir")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert "GGUF-only local models" in exc_info.value.detail


def test_start_rejects_nested_gguf_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_nested_gguf_dir")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "GGUF-only local models" in exc_info.value.detail


def test_untrainable_gate_passes_trainable_local_dir(tmp_path):
    route = _load_route_module("training_route_pass_trainable_dir")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    route._reject_untrainable_model_request(request)


def test_untrainable_gate_does_not_trust_claimed_safetensors(tmp_path):
    route = _load_route_module("training_route_pass_safetensors_format")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "GGUF-only local models" in exc_info.value.detail


def test_untrainable_gate_inspects_verified_snapshot_path(tmp_path):
    route = _load_route_module("training_route_snapshot_format")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "adapter_config.json").write_text("{}")
    (snapshot / "adapter_model.safetensors").write_bytes(b"x")
    request = _request(
        model_format = "safetensors",
        model_snapshot_path = str(snapshot),
        resume_from_checkpoint = "/outputs/run",
    )

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "Adapter-only local models" in exc_info.value.detail


def test_untrainable_gate_inspects_discovered_known_cache(tmp_path):
    route = _load_route_module("training_route_discovered_cache_format")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "adapter_config.json").write_text("{}")
    request = _request(
        model_known_cached = True,
        model_format = "safetensors",
    )

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "Adapter-only local models" in exc_info.value.detail


def test_optimizer_checkpoint_does_not_make_adapter_trainable(tmp_path):
    route = _load_route_module("training_route_adapter_optimizer_artifact")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "optimizer.pt").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert "Adapter-only local models" in exc_info.value.detail


@pytest.mark.parametrize(
    "cache_overrides",
    [
        {"dataset_known_cached": True},
        {"dataset_local_path": "/tmp/hf-cache/datasets--org--dataset"},
    ],
)
def test_streaming_rejects_cached_dataset_hints(cache_overrides):
    route = _load_route_module("training_route_streaming_cached")
    request = _request(dataset_streaming = True, max_steps = 10, **cache_overrides)

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 422
    assert "local" in exc_info.value.detail
    assert "cache" in exc_info.value.detail


def test_route_forwards_cache_reference_fields():
    route = _load_route_module("training_route_forwards_cache_fields")
    request = _request(
        model_known_cached = True,
        model_local_path = "/tmp/hf-cache/models--unsloth--test",
        model_format = "safetensors",
        dataset_known_cached = True,
        dataset_local_path = "/tmp/hf-cache/datasets--org--dataset",
    )

    captured: dict = {}

    def fake_start_training(**kwargs):
        captured.update(kwargs)
        return True

    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = fake_start_training,
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch(
            "core.inference.get_inference_backend",
            return_value = type("InferenceBackend", (), {"active_model_name": None})(),
        ),
        patch(
            "core.export.get_export_backend",
            return_value = type("ExportBackend", (), {"current_checkpoint": None})(),
        ),
    ):
        response = _start(route, request)

    assert response.status == "queued"
    assert captured["model_known_cached"] is True
    assert captured["model_local_path"] == "/tmp/hf-cache/models--unsloth--test"
    assert captured["model_format"] == "safetensors"
    assert captured["dataset_known_cached"] is True
    assert captured["dataset_local_path"] == "/tmp/hf-cache/datasets--org--dataset"


def test_training_backend_forwards_cache_reference_config():
    from core.training.training import TrainingBackend

    backend = TrainingBackend()

    class DummyProcess:
        pid = 12345

        def start(self):
            return None

    class DummyThread:
        def start(self):
            return None

    dummy_queue = object()

    with (
        patch(
            "core.training.training.prepare_gpu_selection",
            return_value = ([0], {"selection_mode": "auto"}),
        ),
        patch(
            "core.training.training._CTX.Queue",
            side_effect = [dummy_queue, dummy_queue],
        ),
        patch("core.training.training._CTX.Process", return_value = DummyProcess()) as mock_process,
        patch("core.training.training.threading.Thread", return_value = DummyThread()),
        patch("core.training.training._resolve_model_snapshot", return_value = None),
        patch("hub.utils.dataset_cache.latest_cached_dataset_path", return_value = None),
    ):
        backend.start_training(
            job_id = "test-cache-refs",
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            model_known_cached = True,
            model_local_path = "/tmp/models--unsloth--test",
            model_format = "safetensors",
            dataset_known_cached = True,
            dataset_local_path = "/tmp/datasets--org--dataset",
        )

    config = mock_process.call_args.kwargs["kwargs"]["config"]
    assert config["model_known_cached"] is True
    assert config["model_local_path"] == "/tmp/models--unsloth--test"
    assert config["model_format"] == "safetensors"
    assert config["dataset_known_cached"] is True
    assert config["dataset_local_path"] == "/tmp/datasets--org--dataset"
    assert config["model_snapshot_path"] is None
    assert config["dataset_snapshot_path"] is None
    assert config["cache_pin_warnings"]
    assert "cache_pin_warnings" not in backend._db_config
    assert backend._db_config["model_snapshot_path"] is None
    assert backend._db_config["dataset_snapshot_path"] is None


def _dataset_repo_with_ref(root: Path, repo_id: str, commit: str = "rev") -> Path:
    repo_root = root / f"datasets--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / commit
    snap.mkdir(parents = True)
    (snap / "train.parquet").write_bytes(b"x")
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(commit)
    return snap


def _model_repo_with_ref(root: Path, repo_id: str, commit: str = "rev") -> Path:
    repo_root = root / f"models--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / commit
    snap.mkdir(parents = True)
    (snap / "config.json").write_text("{}")
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(commit)
    return snap


def test_apply_cache_pins_fresh_start_resolves_snapshots(tmp_path):
    from core.training.training import _apply_cache_pins

    model_snap = _model_repo_with_ref(tmp_path, "unsloth/test")
    dataset_snap = _dataset_repo_with_ref(tmp_path, "org/dataset")

    config = {
        "model_name": "unsloth/test",
        "model_known_cached": True,
        "model_local_path": str(model_snap.parent.parent),
        "hf_dataset": "org/dataset",
        "dataset_known_cached": True,
        "dataset_local_path": str(dataset_snap.parent.parent),
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] == str(model_snap.resolve())
    assert config["dataset_snapshot_path"] == str(dataset_snap.resolve())
    assert config["cache_pin_warnings"] == []


def test_apply_cache_pins_fresh_ignores_client_pins(tmp_path):
    from core.training.training import _apply_cache_pins

    config = {
        "model_name": "unsloth/test",
        "model_snapshot_path": str(tmp_path / "client-supplied"),
        "hf_dataset": "",
        "dataset_snapshot_path": str(tmp_path / "client-dataset"),
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] is None
    assert config["dataset_snapshot_path"] is None


def test_apply_cache_pins_resume_prefers_recorded_pin(tmp_path):
    from core.training.training import _apply_cache_pins

    repo_root = tmp_path / "models--unsloth--test"
    old = repo_root / "snapshots" / "commit-old"
    new = repo_root / "snapshots" / "commit-new"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    (old / "config.json").write_text("{}")
    (new / "config.json").write_text("{}")
    import os

    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("commit-new")

    config = {
        "model_name": "unsloth/test",
        "resume_from_checkpoint": "/outputs/run/checkpoint-5",
        "model_snapshot_path": str(old),
        "hf_dataset": "",
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] == str(old.resolve())
    assert config["cache_pin_warnings"] == []


def test_apply_cache_pins_resume_evicted_pin_warns(tmp_path):
    from core.training.training import _apply_cache_pins

    repo_root = tmp_path / "models--unsloth--test"
    present = repo_root / "snapshots" / "commit-present"
    present.mkdir(parents = True)
    (present / "config.json").write_text("{}")

    config = {
        "model_name": "unsloth/test",
        "resume_from_checkpoint": "/outputs/run/checkpoint-5",
        "model_snapshot_path": str(repo_root / "snapshots" / "commit-gone"),
        "hf_dataset": "",
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] is None
    assert any("no longer on" in w for w in config["cache_pin_warnings"])


def test_apply_cache_pins_resume_pin_rejects_foreign(tmp_path):
    from core.training.training import _apply_cache_pins

    foreign = tmp_path / "somewhere" / "snapshots" / "rev"
    foreign.mkdir(parents = True)
    (foreign / "config.json").write_text("{}")

    config = {
        "model_name": "unsloth/test",
        "resume_from_checkpoint": "/outputs/run/checkpoint-5",
        "model_snapshot_path": str(foreign),
        "hf_dataset": "",
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] is None


def test_worker_rejects_inference_only_model_formats():
    from core.training import worker

    assert "GGUF" in worker._untrainable_model_format_error({"model_format": "gguf"})
    assert "Adapter" in worker._untrainable_model_format_error({"model_format": "adapter"})
    assert worker._untrainable_model_format_error({"model_format": "safetensors"}) is None
    assert worker._untrainable_model_format_error({}) is None


def test_worker_local_files_only_flags():
    from core.training import worker

    assert worker._model_local_files_only({"model_snapshot_path": "/x"}) is True
    assert worker._model_local_files_only({"model_known_cached": True}) is False
    assert worker._model_local_files_only({"model_local_path": "/x"}) is False
    assert worker._model_local_files_only({}) is False
    assert worker._dataset_local_files_only({"dataset_snapshot_path": "/x"}) is True
    assert worker._dataset_local_files_only({"dataset_known_cached": True}) is False
    assert worker._dataset_local_files_only({}) is False


def test_worker_security_scans_exact_model_load_target():
    from core.training import worker

    snapshot = "/cache/models--org--model/snapshots/deadbeef"
    scanned: list[str] = []
    decision = SimpleNamespace(blocked = False)

    with (
        patch(
            "utils.models.model_config.get_base_model_from_lora_identifier",
            return_value = None,
        ),
        patch(
            "utils.security.security_load_subdirs",
            return_value = (),
        ),
        patch(
            "utils.security.evaluate_file_security",
            side_effect = lambda target, **kwargs: scanned.append(target) or decision,
        ),
    ):
        result = worker._model_load_security_error(
            {"model_name": "org/model", "trust_remote_code": False},
            snapshot,
            None,
        )

    assert result is None
    assert scanned == [snapshot]


def test_worker_security_consent_uses_exact_target_and_base():
    from core.training import worker

    snapshot = "/cache/models--org--adapter/snapshots/deadbeef"
    consent_targets: list[str] = []
    file_decision = SimpleNamespace(blocked = False)
    consent_decision = SimpleNamespace(blocked = False)

    with (
        patch(
            "utils.models.model_config.get_base_model_from_lora_identifier",
            return_value = "org/base",
        ),
        patch(
            "utils.security.security_load_subdirs",
            return_value = (),
        ),
        patch(
            "utils.security.evaluate_file_security",
            return_value = file_decision,
        ),
        patch(
            "utils.security.evaluate_remote_code_consent_for_targets",
            side_effect = (
                lambda targets, **kwargs: consent_targets.extend(targets)
                or consent_decision
            ),
        ),
    ):
        result = worker._model_load_security_error(
            {
                "model_name": "org/adapter",
                "trust_remote_code": True,
                "subject": "test-user",
            },
            snapshot,
            "hf_test",
        )

    assert result is None
    assert consent_targets == [snapshot, "org/base"]


def test_worker_resolves_cached_model_snapshot():
    from core.training import worker

    assert (
        worker._resolve_cached_model_load_name(
            {"model_name": "unsloth/test", "model_snapshot_path": "/snap/dir"}
        )
        == "/snap/dir"
    )
    assert (
        worker._resolve_cached_model_load_name({"model_name": "unsloth/test"})
        == "unsloth/test"
    )
    assert (
        worker._resolve_cached_model_load_name(
            {"model_name": "unsloth/test", "model_snapshot_path": None}
        )
        == "unsloth/test"
    )


def test_worker_cached_dataset_load_requires_verified_path():
    from core.training import worker

    assert (
        worker._load_cached_dataset_for_config(
            {"hf_dataset": "org/dataset", "dataset_known_cached": True}, "train"
        )
        is None
    )

    with patch(
        "hub.utils.dataset_cache.load_cached_hf_dataset",
        return_value = {"loaded": True},
    ) as load_cached:
        result = worker._load_cached_dataset_for_config(
            {
                "hf_dataset": "org/dataset",
                "dataset_snapshot_path": "/verified/cache",
                "subset": "english",
            },
            "validation",
            "hf_test",
        )

    assert result == {"loaded": True}
    load_cached.assert_called_once_with(
        "org/dataset",
        "/verified/cache",
        subset = "english",
        split = "validation",
        token = "hf_test",
    )


def test_worker_cached_eval_failure_reloads_remote_pair():
    from core.training import worker

    cached_train = object()
    remote_train = object()
    remote_eval = object()
    cached_calls: list[str] = []
    remote_calls: list[str] = []
    statuses: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/verified/cache",
        "train_split": "train",
        "eval_split": "validation",
        "subset": "english",
    }

    def load_cached(request_config, split, token):
        cached_calls.append(split)
        if split == "validation":
            raise FileNotFoundError(split)
        return cached_train

    def load_remote(repo_id, **kwargs):
        remote_calls.append(kwargs["split"])
        return remote_eval if kwargs["split"] == "validation" else remote_train

    with patch.object(worker, "_load_cached_dataset_for_config", side_effect = load_cached):
        dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
            config,
            "hf_test",
            load_remote,
            statuses.append,
        )

    assert dataset is remote_train
    assert eval_dataset is remote_eval
    assert cached_calls == ["train", "validation"]
    assert remote_calls == ["train", "validation"]
    assert any("reloading train and eval" in status for status in statuses)


def test_worker_model_retry_refreshes_tokenizer_before_dataset():
    from core.training import worker

    events: list[tuple[str, object]] = []
    trainer = SimpleNamespace(
        pre_detect_and_load_tokenizer = (
            lambda **kwargs: events.append(("tokenizer", kwargs))
        )
    )
    config = {
        "max_seq_length": 2048,
        "is_dataset_image": True,
        "is_dataset_audio": False,
        "trust_remote_code": True,
    }
    expected = (object(), object())

    def reload_dataset():
        events.append(("dataset", None))
        return expected

    result = worker._reload_dataset_with_remote_model_tokenizer(
        trainer,
        config,
        "org/model",
        "hf_test",
        reload_dataset,
    )

    assert result is expected
    assert [event[0] for event in events] == ["tokenizer", "dataset"]
    tokenizer_kwargs = events[0][1]
    assert tokenizer_kwargs["model_name"] == "org/model"
    assert tokenizer_kwargs["model_load_name"] == "org/model"
    assert tokenizer_kwargs["local_files_only"] is False
    assert tokenizer_kwargs["hf_token"] == "hf_test"


def test_worker_bootstrap_drops_vanished_pins_and_emits_warnings(tmp_path):
    from core.training import worker

    events: list[dict] = []
    queue = SimpleNamespace(put = events.append)
    config = {
        "model_name": "unsloth/test",
        "hf_dataset": "org/dataset",
        "model_snapshot_path": str(tmp_path / "gone-model"),
        "dataset_snapshot_path": str(tmp_path / "gone-dataset"),
        "cache_pin_warnings": ["cached model missing; downloading"],
    }

    worker._verify_config_pins(config, queue)

    assert config["model_snapshot_path"] is None
    assert config["dataset_snapshot_path"] is None
    assert any(
        event.get("type") == "status"
        and event.get("message") == "cached model missing; downloading"
        for event in events
    )


@pytest.mark.parametrize(
    "bad_path",
    ["/tmp/\x00bad", "/tmp/../etc", "..\\windows\\escape", "x" * 4097],
)
def test_cache_local_paths_reject_unsafe_values(bad_path):
    with pytest.raises(ValidationError):
        _request(model_local_path = bad_path)
    with pytest.raises(ValidationError):
        _request(dataset_local_path = bad_path)
    with pytest.raises(ValidationError):
        _request(model_snapshot_path = bad_path)
    with pytest.raises(ValidationError):
        _request(dataset_snapshot_path = bad_path)


def test_cache_local_paths_accept_windows_drive_paths():
    request = _request(model_local_path = "G:\\hfcache\\models--unsloth--test")
    assert request.model_local_path == "G:\\hfcache\\models--unsloth--test"


def test_cache_local_paths_blank_normalizes_to_none():
    request = _request(model_local_path = "   ", dataset_local_path = "")
    assert request.model_local_path is None
    assert request.dataset_local_path is None
