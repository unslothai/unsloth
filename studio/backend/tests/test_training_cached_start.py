# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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
    module._hub_unreachable = lambda: False
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


@pytest.mark.parametrize("repo_id", ["_team/dataset_", "dataset_"])
def test_training_request_accepts_hugging_face_underscore_boundaries(repo_id):
    assert _request(hf_dataset = repo_id).hf_dataset == repo_id


def _refusing_backend() -> SimpleNamespace:
    return SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: pytest.fail("backend should not start"),
    )


class _FakeHFValidationError(Exception):
    pass


def _accept_repo_id(_repo_id):
    return None


def _fake_hf_modules(api_type, validate_repo_id = _accept_repo_id):
    return {
        "huggingface_hub": SimpleNamespace(HfApi = api_type),
        "huggingface_hub.utils": SimpleNamespace(
            HFValidationError = _FakeHFValidationError,
            validate_repo_id = validate_repo_id,
        ),
    }


@contextmanager
def _patch_model_info(**kwargs):
    model_info = Mock(**kwargs)
    with patch.dict(
        sys.modules,
        {"huggingface_hub": SimpleNamespace(model_info = model_info)},
    ):
        yield model_info


def _start(
    route,
    request,
    *,
    run_dataset_preflight = False,
):
    if run_dataset_preflight:
        return asyncio.run(route.start_training(request, current_subject = "test-user"))
    with (
        patch.object(route, "_preflight_hf_dataset_request", return_value = None),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
    ):
        return asyncio.run(route.start_training(request, current_subject = "test-user"))


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


def test_hf_dataset_preflight_uses_hub_metadata_and_token():
    route = _load_route_module("training_route_hf_dataset_preflight_success")
    request = _request(hf_token = "hf_test")
    calls: list[tuple[str, float]] = []

    class FakeApi:
        def __init__(self, token = None):
            assert token == "hf_test"

        def dataset_info(self, repo_id, *, timeout):
            calls.append((repo_id, timeout))
            return object()

    with (
        patch.object(route, "hf_env_offline", return_value = False),
        patch.dict(sys.modules, _fake_hf_modules(FakeApi)),
    ):
        route._preflight_hf_dataset_request(request)

    assert calls == [("org/dataset", route._REMOTE_DATASET_METADATA_TIMEOUT_SECONDS)]


def test_hf_dataset_preflight_rejects_invalid_repo_id():
    route = _load_route_module("training_route_hf_dataset_preflight_invalid")

    class FakeApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unexpected Hub metadata call")

    def reject_repo_id(_repo_id):
        raise _FakeHFValidationError("invalid repo")

    with (
        patch.dict(sys.modules, _fake_hf_modules(FakeApi, reject_repo_id)),
        pytest.raises(HTTPException) as exc_info,
    ):
        route._preflight_hf_dataset_request(_request(hf_dataset = "org/team/dataset"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "hf_dataset_verification_failed"


def test_hf_dataset_preflight_skips_local_sources(tmp_path):
    route = _load_route_module("training_route_local_dataset_skips_hf_preflight")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"text":"hello"}\n', encoding = "utf-8")
    request = _request(
        hf_dataset = None,
        local_datasets = [str(dataset_path)],
        model_format = "gguf",
    )

    with (
        patch.object(route, "get_training_backend", return_value = _refusing_backend()),
        patch.object(route, "resolve_dataset_path", return_value = dataset_path),
        patch.object(
            route,
            "_preflight_hf_dataset_request",
            side_effect = AssertionError("unexpected Hub call"),
        ),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        pytest.raises(HTTPException) as exc_info,
    ):
        _start(route, request, run_dataset_preflight = True)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_model_gguf_not_trainable"
    assert "GGUF" in exc_info.value.detail["message"]


def test_hf_dataset_preflight_accepts_usable_selected_cache(tmp_path):
    route = _load_route_module("training_route_cached_dataset_skips_hf_preflight")
    request = _request(
        dataset_known_cached = True,
        dataset_local_path = str(tmp_path / "datasets--org--dataset"),
    )

    class UnexpectedApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unexpected Hub call")

    with (
        patch(
            "hub.utils.dataset_cache.training_dataset_cache_pin",
            return_value = (tmp_path / "snapshot", "revision"),
        ),
        patch.dict(sys.modules, _fake_hf_modules(UnexpectedApi)),
    ):
        route._preflight_hf_dataset_request(request)


def test_hf_dataset_preflight_verifies_an_unpinned_repo_with_a_stray_cache(tmp_path):
    # Without a cache pin the start still downloads, so an unrelated cached copy buys no pass.
    route = _load_route_module("training_route_unpinned_dataset_verifies")
    calls: list[str] = []

    class FakeApi:
        def __init__(self, token = None):
            pass

        def dataset_info(self, repo_id, *, timeout):
            calls.append(repo_id)
            return object()

    with (
        patch(
            "hub.utils.dataset_cache.training_dataset_cache_pin",
            return_value = (tmp_path / "snapshot", "revision"),
        ),
        patch.object(route, "hf_env_offline", return_value = False),
        patch.dict(sys.modules, _fake_hf_modules(FakeApi)),
    ):
        route._preflight_hf_dataset_request(_request())

    assert calls == ["org/dataset"]


def test_hf_dataset_preflight_accepts_an_unpinned_cached_repo_offline(tmp_path):
    # Offline there is no Hub to verify against, so a cached copy still starts.
    route = _load_route_module("training_route_unpinned_dataset_offline")

    class UnexpectedApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unexpected Hub call")

    with (
        patch(
            "hub.utils.dataset_cache.training_dataset_cache_pin",
            return_value = (tmp_path / "snapshot", "revision"),
        ),
        patch.object(route, "hf_env_offline", return_value = True),
        patch.dict(sys.modules, _fake_hf_modules(UnexpectedApi)),
    ):
        route._preflight_hf_dataset_request(_request())


def test_streaming_dataset_preflight_bypasses_generic_cache():
    route = _load_route_module("training_route_streaming_dataset_preflight")
    request = _request(dataset_streaming = True, max_steps = 10)
    calls: list[tuple[str, float]] = []

    class FakeApi:
        def __init__(self, token = None):
            assert token is False

        def dataset_info(self, repo_id, *, timeout):
            calls.append((repo_id, timeout))
            return object()

    with (
        patch(
            "hub.utils.dataset_cache.training_dataset_cache_pin",
            side_effect = AssertionError("streaming must not inspect the local cache"),
        ),
        patch.object(route, "hf_env_offline", return_value = False),
        patch.dict(sys.modules, _fake_hf_modules(FakeApi)),
    ):
        route._preflight_hf_dataset_request(request)

    assert calls == [("org/dataset", route._REMOTE_DATASET_METADATA_TIMEOUT_SECONDS)]


def test_streaming_dataset_preflight_rejects_offline_before_cache_lookup():
    route = _load_route_module("training_route_streaming_dataset_offline")
    request = _request(dataset_streaming = True, max_steps = 10)

    class UnexpectedApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unexpected Hub call")

    with (
        patch(
            "hub.utils.dataset_cache.training_dataset_cache_pin",
            side_effect = AssertionError("streaming must not inspect the local cache"),
        ),
        patch.object(route, "hf_env_offline", return_value = True),
        patch.dict(sys.modules, _fake_hf_modules(UnexpectedApi)),
        pytest.raises(HTTPException) as exc_info,
    ):
        route._preflight_hf_dataset_request(request)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "hf_dataset_streaming_offline"
    assert "disable streaming" in exc_info.value.detail["message"].lower()


def test_hf_dataset_preflight_rejects_before_backend_start():
    route = _load_route_module("training_route_dataset_preflight_before_start")
    request = _request()
    backend = _refusing_backend()
    error = HTTPException(
        status_code = 404,
        detail = {
            "code": "hf_dataset_verification_failed",
            "message": "Dataset not found",
        },
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(route, "_preflight_hf_dataset_request", side_effect = error),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        pytest.raises(HTTPException) as exc_info,
    ):
        _start(route, request, run_dataset_preflight = True)

    assert exc_info.value is error


@pytest.mark.parametrize(
    ("model_format", "code", "expected"),
    [
        ("gguf", "training_model_gguf_not_trainable", "GGUF models are inference-only"),
        (
            "adapter",
            "training_model_adapter_not_trainable",
            "Adapter models are inference-only",
        ),
    ],
)
def test_start_rejects_untrainable_model_formats(model_format, code, expected):
    route = _load_route_module(f"training_route_reject_{model_format}")
    request = _request(model_format = model_format)

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == code
    assert expected in exc_info.value.detail["message"]


def test_start_rejects_adapter_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_adapter_dir")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


def test_start_rejects_missing_local_model(tmp_path):
    route = _load_route_module("training_route_reject_missing_local_model")
    request = _request(model_name = str(tmp_path / "missing-model"))

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_unavailable"
    assert "Local model path was not found" in exc_info.value.detail["message"]


def test_start_rejects_local_dir_without_trainable_weights(tmp_path):
    route = _load_route_module("training_route_reject_weightless_local_model")
    (tmp_path / "config.json").write_text("{}")
    request = _request(model_name = str(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_weights_missing"
    assert "does not contain trainable weights" in exc_info.value.detail["message"]


def test_start_rejects_claimed_cache_without_trainable_weights(tmp_path):
    route = _load_route_module("training_route_reject_weightless_cache")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    request = _request(model_known_cached = True, model_format = "unknown")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_weights_missing"
    assert "does not contain trainable weights" in exc_info.value.detail["message"]


@pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_uncached_offline_model_fails_without_remote_probe(monkeypatch, offline_variable):
    route = _load_route_module(f"training_route_uncached_offline_{offline_variable}")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv(offline_variable, "true")

    with patch.object(route, "_remote_untrainable_model_format") as remote_probe:
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "hf_model_not_cached_offline"
    assert "not available in the local cache" in exc_info.value.detail["message"]
    remote_probe.assert_not_called()


def test_start_rejects_partial_adapter_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_partial_adapter_dir")
    (tmp_path / "adapter_config.json").write_text("{}")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


def test_start_rejects_gguf_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_gguf_dir")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    with patch.object(route, "get_training_backend", return_value = _refusing_backend()):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_gguf_only"
    assert "GGUF-only local models" in exc_info.value.detail["message"]


def test_start_rejects_nested_gguf_only_local_dir(tmp_path):
    route = _load_route_module("training_route_reject_nested_gguf_dir")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_gguf_only"
    assert "GGUF-only local models" in exc_info.value.detail["message"]


def test_untrainable_gate_rejects_incomplete_local_probe(tmp_path):
    route = _load_route_module("training_route_incomplete_local_probe")
    for index in range(3):
        (tmp_path / f"artifact-{index}.txt").write_text("x")
    request = _request(model_name = str(tmp_path))

    with patch.object(route, "_LOCAL_MODEL_PROBE_LIMIT", 2):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_scan_incomplete"
    assert "too large or could not be read safely" in exc_info.value.detail["message"]


def test_untrainable_gate_passes_trainable_local_dir(tmp_path):
    route = _load_route_module("training_route_pass_trainable_dir")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    request = _request(model_name = str(tmp_path))

    route._reject_untrainable_model_request(request)


def test_wsl_windows_model_path_is_normalized_for_preflight_and_worker(monkeypatch, tmp_path):
    route = _load_route_module("training_route_wsl_windows_model_path")
    model_path = tmp_path / "models" / "alpha"
    model_path.mkdir(parents = True)
    (model_path / "config.json").write_text("{}")
    (model_path / "model.safetensors").write_bytes(b"x")
    windows_path = r"C:\models\alpha"
    captured: dict = {}
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: captured.update(kwargs) or True,
    )
    normalized: list[str] = []

    def normalize_model_path(value: str) -> str:
        normalized.append(value)
        return str(model_path) if value == windows_path else value

    monkeypatch.setattr(route, "normalize_path", normalize_model_path)

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch("utils.transformers_version.latest_tier_active_for", return_value = False),
        patch(
            "core.inference.get_inference_backend",
            return_value = type("InferenceBackend", (), {"active_model_name": None})(),
        ),
        patch(
            "core.export.get_export_backend",
            return_value = type("ExportBackend", (), {"current_checkpoint": None})(),
        ),
    ):
        response = _start(
            route,
            _request(model_name = windows_path, model_local_path = windows_path),
        )

    assert response.status == "queued"
    assert captured["model_name"] == str(model_path.resolve())
    assert captured["model_local_path"] == str(model_path)
    assert normalized == [windows_path]


def test_wsl_windows_cache_hint_is_normalized_for_preflight(monkeypatch):
    route = _load_route_module("training_route_wsl_windows_cache_hint")
    windows_path = r"C:\cache\models--unsloth--test"
    normalized_path = "/mnt/c/cache/models--unsloth--test"
    probed: list[str | None] = []

    monkeypatch.setattr(
        route,
        "normalize_path",
        lambda value: normalized_path if value == windows_path else value,
    )

    with (
        patch(
            "core.training.training._resolve_model_snapshot",
            side_effect = lambda _model_name, local_path: probed.append(local_path) or None,
        ),
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
    ):
        result = route._reject_untrainable_model_request(
            _request(model_known_cached = True, model_local_path = windows_path)
        )

    assert probed == [normalized_path]
    assert result.model_local_path == normalized_path


def test_untrainable_gate_does_not_trust_claimed_safetensors(tmp_path):
    route = _load_route_module("training_route_pass_safetensors_format")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_gguf_only"
    assert "GGUF-only local models" in exc_info.value.detail["message"]


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
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


def test_untrainable_gate_inspects_selected_cache(tmp_path):
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
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


def test_online_probe_ignores_unadvertised_stale_cache(tmp_path):
    route = _load_route_module("training_route_online_ignores_stale_cache")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "adapter_config.json").write_text("{}")

    with patch.object(route, "_remote_untrainable_model_format", return_value = None):
        route._reject_untrainable_model_request(_request())


def test_unavailable_probe_inspects_unadvertised_cache(tmp_path):
    route = _load_route_module("training_route_outage_inspects_cache")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "adapter_config.json").write_text("{}")

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = HTTPException(status_code = 503, detail = "unavailable"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


def test_unavailable_probe_uses_cached_root_level_model(tmp_path):
    route = _load_route_module("training_route_outage_root_level_cache")
    snapshot = tmp_path / "models--bert-base-uncased" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x")

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = HTTPException(status_code = 503, detail = "unavailable"),
    ):
        result = route._reject_untrainable_model_request(_request(model_name = "bert-base-uncased"))

    assert result.cached_model_pin == ("bert-base-uncased", str(snapshot.resolve()))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429])
def test_client_error_probe_uses_unadvertised_cached_model(tmp_path, status_code):
    route = _load_route_module(f"training_route_client_error_cache_{status_code}")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x")
    metadata_error = HTTPException(
        status_code = status_code,
        detail = f"hub metadata error {status_code}",
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = metadata_error,
    ):
        result = route._reject_untrainable_model_request(_request())

    assert result.cached_model_pin == ("unsloth/test", str(snapshot.resolve()))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429])
def test_client_error_probe_without_cache_preserves_error(status_code):
    route = _load_route_module(f"training_route_client_error_no_cache_{status_code}")
    metadata_error = HTTPException(
        status_code = status_code,
        detail = f"hub metadata error {status_code}",
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = metadata_error,
    ):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value is metadata_error


def test_client_error_probe_with_incomplete_cache_preserves_error(tmp_path):
    route = _load_route_module("training_route_client_error_incomplete_cache")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    metadata_error = HTTPException(
        status_code = 403,
        detail = "hub metadata error 403",
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = metadata_error,
    ):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value is metadata_error


@pytest.mark.parametrize(
    "index_mode",
    ["absent", "references-missing-shard", "omits-missing-shard"],
)
def test_client_error_probe_with_partial_shards_preserves_error(tmp_path, index_mode):
    route = _load_route_module(f"training_route_client_error_partial_shards_{index_mode}")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    (snapshot / first_shard).write_bytes(b"x")
    if index_mode != "absent":
        indexed_shards = (
            [first_shard, second_shard]
            if index_mode == "references-missing-shard"
            else [first_shard]
        )
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"layer.{index}": shard for index, shard in enumerate(indexed_shards)
                    },
                },
            )
        )
    metadata_error = HTTPException(
        status_code = 403,
        detail = "hub metadata error 403",
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = metadata_error,
    ):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value is metadata_error


def test_client_error_probe_uses_complete_sharded_cache(tmp_path):
    route = _load_route_module("training_route_client_error_complete_shards")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    (snapshot / first_shard).write_bytes(b"x")
    (snapshot / second_shard).write_bytes(b"x")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": first_shard,
                    "layer.1": second_shard,
                },
            },
        )
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = HTTPException(status_code = 403, detail = "unavailable"),
    ):
        result = route._reject_untrainable_model_request(_request())

    assert result.cached_model_pin == ("unsloth/test", str(snapshot.resolve()))


def test_incomplete_safetensors_index_is_not_masked_by_pytorch_weights(tmp_path):
    route = _load_route_module("training_route_incomplete_safe_index_with_pytorch")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    (snapshot / first_shard).write_bytes(b"x")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": first_shard,
                    "layer.1": second_shard,
                },
            },
        )
    )
    (snapshot / "pytorch_model.bin").write_bytes(b"x")
    metadata_error = HTTPException(
        status_code = 403,
        detail = "hub metadata error 403",
    )

    with patch.object(
        route,
        "_remote_untrainable_model_format",
        side_effect = metadata_error,
    ):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(_request())

    assert exc_info.value is metadata_error


@pytest.mark.parametrize("offline", [False, True])
def test_unadvertised_cache_pin_reaches_worker(monkeypatch, tmp_path, offline):
    from core.training import worker
    from core.training.training import _apply_cache_pins, _build_training_worker_config

    route = _load_route_module(f"training_route_cache_pin_to_worker_{offline}")
    snapshot = tmp_path / "models--bert-base-uncased" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x")
    captured: dict = {}
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: captured.update(kwargs) or True,
    )
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        probe_error = AssertionError("offline cache start must not query Hub metadata")
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        probe_error = HTTPException(status_code = 503, detail = "unavailable")

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(
            route,
            "_remote_untrainable_model_format",
            side_effect = probe_error,
        ),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch("utils.transformers_version.latest_tier_active_for", return_value = False),
        patch(
            "core.inference.get_inference_backend",
            return_value = type("InferenceBackend", (), {"active_model_name": None})(),
        ),
        patch(
            "core.export.get_export_backend",
            return_value = type("ExportBackend", (), {"current_checkpoint": None})(),
        ),
    ):
        response = _start(route, _request(model_name = "bert-base-uncased"))

    assert response.status == "queued"
    assert captured["actual_model_repo_id"] == "bert-base-uncased"
    assert captured["model_snapshot_path"] == str(snapshot.resolve())
    assert captured["require_validated_model_snapshot"] is True

    config = _build_training_worker_config(captured)
    _apply_cache_pins(config)
    events: list[dict] = []
    assert worker._verify_config_pins(config, SimpleNamespace(put = events.append)) is True
    assert config["actual_model_repo_id"] == "bert-base-uncased"
    assert config["model_snapshot_path"] == str(snapshot.resolve())
    assert config["model_revision"] == "rev"
    assert config["require_validated_model_snapshot"] is True
    assert worker._cache_artifact_fallback_allowed(
        config,
        ValueError("Either model_file or model_proto must be specified."),
        "model",
    ) is (not offline)


def test_selected_cached_model_tokenizer_failure_allows_hub_fallback(tmp_path):
    from core.training import worker
    from core.training.training import _apply_cache_pins

    route = _load_route_module("training_route_selected_cache_tokenizer_fallback")
    snapshot = tmp_path / "models--unsloth--test" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x")

    preflight = route._reject_untrainable_model_request(
        _request(model_known_cached = True, model_format = "safetensors")
    )

    assert preflight.cached_model_pin is None
    config = {
        "model_name": preflight.model_name,
        "model_known_cached": True,
        "model_local_path": preflight.model_local_path,
    }
    _apply_cache_pins(config)

    assert config["model_snapshot_path"] == str(snapshot.resolve())
    assert config["actual_model_repo_id"] == "unsloth/test"
    assert config["model_revision"] == "rev"
    assert config.get("require_validated_model_snapshot", False) is False
    error = ValueError("Either model_file or model_proto must be specified.")
    assert worker._cache_artifact_fallback_allowed(config, error, "model") is True

    with patch(
        "utils.transformers_version.get_transformers_activation_tier",
        side_effect = AssertionError("an exact-revision retry must not probe mutable HEAD"),
    ):
        assert worker._drop_model_pin_for_fallback(config, None) == "unsloth/test"
    assert config["model_snapshot_path"] is None
    assert config["actual_model_repo_id"] == "unsloth/test"
    assert config["model_revision"] == "rev"


@pytest.mark.parametrize(
    "error",
    [
        TypeError("stat: path should be string, bytes, os.PathLike or integer, not NoneType"),
        TypeError("expected str, bytes or os.PathLike object, not NoneType"),
        OSError("Can't load processor for 'org/model'."),
        OSError("Can't load image processor for 'org/model'."),
        OSError("Can't load feature extractor for 'org/model'."),
    ],
)
def test_worker_model_cache_fallback_recognizes_missing_tokenizer_and_processor(monkeypatch, error):
    from core.training import worker

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    assert worker._cache_artifact_fallback_allowed({}, error, "model") is True


@pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_worker_incomplete_model_cache_reports_actionable_offline_error(
    monkeypatch, offline_variable
):
    from core.training import worker

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv(offline_variable, "1")
    config = {"model_revision": "deadbeef"}
    error = TypeError("expected str, bytes or os.PathLike object, not NoneType")

    assert worker._cache_artifact_fallback_allowed(config, error, "model") is False
    fallback_error = worker._model_cache_fallback_error(config, error)
    assert fallback_error is not None
    assert "Offline mode is enabled" in str(fallback_error)
    assert "deadbeef" in str(fallback_error)


def test_worker_exact_resume_rejects_incomplete_model_cache_with_clear_error(monkeypatch):
    from core.training import worker

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    config = {"require_exact_model_resource": True}
    error = OSError("Can't load processor for '/cache/snapshot'.")

    assert worker._cache_artifact_fallback_allowed(config, error, "model") is False
    fallback_error = worker._model_cache_fallback_error(config, error)
    assert fallback_error is not None
    assert "exact cached model snapshot is incomplete" in str(fallback_error)


def test_mlx_pinned_fallback_rejects_cross_repository_bnb_remap():
    from core.training import worker

    error = worker._mlx_revision_fallback_error(
        {
            "model_name": "unsloth/test-bnb-4bit",
            "model_revision": "deadbeef",
        }
    )

    assert error is not None
    assert "different base repository" in str(error)
    assert "deadbeef" in str(error)
    assert (
        worker._mlx_revision_fallback_error(
            {"model_name": "org/mlx-model", "model_revision": "deadbeef"}
        )
        is None
    )


def test_untrainable_gate_rejects_remote_adapter():
    route = _load_route_module("training_route_remote_adapter")
    request = _request()

    with patch.object(route, "_remote_untrainable_model_format", return_value = "adapter"):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_remote_model_adapter_only"
    assert "Adapter models are inference-only" in exc_info.value.detail["message"]


def test_untrainable_gate_rejects_remote_gguf_only_repository():
    route = _load_route_module("training_route_remote_gguf")
    request = _request()

    with patch.object(route, "_remote_untrainable_model_format", return_value = "gguf"):
        with pytest.raises(HTTPException) as exc_info:
            route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_remote_model_gguf_only"
    assert "GGUF-only remote models are inference-only" in exc_info.value.detail["message"]


def test_remote_format_probe_preserves_root_level_repo_id():
    route = _load_route_module("training_route_remote_adapter_metadata")
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "adapter_config.json")])

    with _patch_model_info(return_value = info) as model_info:
        assert route._remote_untrainable_model_format("bert-base-uncased", "hf-token") == "adapter"

    model_info.assert_called_once_with(
        "bert-base-uncased",
        token = "hf-token",
        timeout = route._REMOTE_MODEL_METADATA_TIMEOUT_SECONDS,
    )


def test_remote_format_probe_resolves_the_bicodec_alias():
    # "Spark-TTS-0.5B/LLM" is a registry alias, not a repo: probing it literally 404s, so preflight
    # has to probe what the trainer downloads.
    route = _load_route_module("training_route_remote_bicodec_alias")
    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "LLM/config.json"),
            SimpleNamespace(rfilename = "LLM/model.safetensors"),
        ]
    )

    with _patch_model_info(return_value = info) as model_info:
        assert route._remote_untrainable_model_format("Spark-TTS-0.5B/LLM", None) is None

    assert model_info.call_args_list[0].args[0] == "unsloth/Spark-TTS-0.5B"


def test_remote_format_probe_retries_transient_failure():
    route = _load_route_module("training_route_remote_adapter_metadata_retry")
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "adapter_config.json")])

    with _patch_model_info(
        side_effect = [OSError("timed out"), info],
    ) as model_info:
        assert route._remote_untrainable_model_format("bert-base-uncased", "hf-token") == "adapter"

    assert model_info.call_args_list == [
        call(
            "bert-base-uncased",
            token = "hf-token",
            timeout = route._REMOTE_MODEL_METADATA_TIMEOUT_SECONDS,
        ),
        call(
            "bert-base-uncased",
            token = "hf-token",
            timeout = route._REMOTE_MODEL_METADATA_RETRY_TIMEOUT_SECONDS,
        ),
    ]


@pytest.mark.parametrize("status_code", [408, 429, 502])
def test_remote_format_probe_retries_transient_hub_status(status_code):
    route = _load_route_module(f"training_route_remote_status_retry_{status_code}")
    transient_error = RuntimeError("transient Hub failure")
    transient_error.response = SimpleNamespace(status_code = status_code)
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "adapter_config.json")])

    with _patch_model_info(
        side_effect = [transient_error, info],
    ) as model_info:
        assert route._remote_untrainable_model_format("test", "hf-token") == "adapter"

    assert model_info.call_count == 2
    assert model_info.call_args_list[1].kwargs["timeout"] == (
        route._REMOTE_MODEL_METADATA_RETRY_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize("status_code", [401, 403])
def test_remote_format_probe_maps_hub_access_denial_to_validation_error(status_code):
    route = _load_route_module(f"training_route_remote_access_denied_{status_code}")
    access_error = RuntimeError("access denied")
    access_error.response = SimpleNamespace(status_code = status_code)

    with _patch_model_info(side_effect = access_error) as model_info:
        with pytest.raises(HTTPException) as exc_info:
            route._remote_untrainable_model_format("test", "hf-token")

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "hf_model_access_denied"
    assert "denied access" in exc_info.value.detail["message"].lower()
    assert "token" in exc_info.value.detail["message"].lower()
    model_info.assert_called_once()


def test_remote_format_probe_reports_stable_verification_error_code():
    route = _load_route_module("training_route_remote_verification_failed")
    not_found = RuntimeError("not found")
    not_found.response = SimpleNamespace(status_code = 404)

    with _patch_model_info(side_effect = not_found):
        with pytest.raises(HTTPException) as exc_info:
            route._remote_untrainable_model_format("test", "hf-token")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == "hf_model_verification_failed"
    assert "could not be verified" in exc_info.value.detail["message"].lower()


def test_structured_start_error_preserves_code_for_reconciliation():
    route = _load_route_module("training_route_structured_start_error")
    error = HTTPException(
        status_code = 503,
        detail = {
            "code": "hf_model_metadata_unavailable",
            "message": "Model metadata unavailable",
        },
    )

    assert route._http_exception_error(error) == (
        "Model metadata unavailable",
        "hf_model_metadata_unavailable",
    )


@pytest.mark.parametrize("status_code", [408, 502])
def test_remote_format_probe_exhausted_transient_status_reports_unavailable(status_code):
    route = _load_route_module(f"training_route_remote_status_exhausted_{status_code}")
    transient_error = RuntimeError("transient Hub failure")
    transient_error.response = SimpleNamespace(status_code = status_code)

    with _patch_model_info(side_effect = transient_error) as model_info:
        with pytest.raises(HTTPException) as exc_info:
            route._remote_untrainable_model_format("test", "hf-token")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "hf_model_metadata_unavailable"
    assert "temporarily unavailable" in exc_info.value.detail["message"].lower()
    assert model_info.call_count == 2


def test_remote_format_probe_rejects_gguf_only_repository():
    route = _load_route_module("training_route_remote_gguf_metadata")
    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "README.md"),
            SimpleNamespace(rfilename = "weights/model-Q4_K_M.GGUF"),
        ]
    )

    with _patch_model_info(return_value = info):
        assert route._remote_untrainable_model_format("org/model", None) == "gguf"


def test_remote_format_probe_allows_repository_with_trainable_weights():
    route = _load_route_module("training_route_remote_mixed_metadata")
    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "model.safetensors.index.json"),
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf"),
        ]
    )

    with _patch_model_info(return_value = info):
        assert route._remote_untrainable_model_format("org/model", None) is None


def test_remote_format_probe_reports_rate_limit_without_token_guidance():
    route = _load_route_module("training_route_remote_adapter_rate_limit")
    rate_limit_error = RuntimeError("rate limited")
    rate_limit_error.response = SimpleNamespace(status_code = 429)

    with _patch_model_info(side_effect = rate_limit_error):
        with pytest.raises(HTTPException) as exc_info:
            route._remote_untrainable_model_format("test", "hf-token")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["code"] == "hf_model_verification_rate_limited"
    assert "rate-limited" in exc_info.value.detail["message"].lower()
    assert "access token" not in exc_info.value.detail["message"].lower()


def test_optimizer_checkpoint_does_not_make_adapter_trainable(tmp_path):
    route = _load_route_module("training_route_adapter_optimizer_artifact")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "optimizer.pt").write_bytes(b"x")
    request = _request(model_name = str(tmp_path), model_format = "safetensors")

    with pytest.raises(HTTPException) as exc_info:
        route._reject_untrainable_model_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "training_local_model_adapter_only"
    assert "Adapter-only local models" in exc_info.value.detail["message"]


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


@pytest.mark.parametrize(
    ("request_overrides", "expected"),
    [
        ({"training_type": "Continued Pretraining"}, "Continued Pretraining"),
        ({"is_embedding": True}, "Embedding model training"),
        ({"is_dataset_audio": True}, "Audio dataset training"),
        ({"use_loftq": True}, "LoftQ"),
        ({"use_dora": True}, "DoRA"),
    ],
)
def test_mlx_start_rejects_unsupported_training_config(request_overrides, expected):
    from utils.hardware import hardware

    route = _load_route_module(f"training_route_mlx_reject_{expected}")
    request = _request(**request_overrides)

    with (
        patch.object(route, "get_training_backend", return_value = _refusing_backend()),
        patch.object(hardware, "DEVICE", hardware.DeviceType.MLX),
        pytest.raises(HTTPException) as exc_info,
    ):
        _start(route, request)

    assert exc_info.value.status_code == 400
    assert expected in exc_info.value.detail


def test_mlx_start_detects_hardware_before_platform_validation():
    from utils.hardware import hardware

    route = _load_route_module("training_route_mlx_warm_platform_reject")
    request = _request(training_type = "Continued Pretraining")
    detection_calls = 0

    def detect_mlx():
        nonlocal detection_calls
        detection_calls += 1
        hardware.DEVICE = hardware.DeviceType.MLX
        return hardware.DEVICE

    with (
        patch.object(route, "get_training_backend", return_value = _refusing_backend()),
        patch.object(hardware, "DEVICE", None),
        patch("utils.hardware.ensure_hardware_detected", side_effect = detect_mlx),
        pytest.raises(HTTPException) as exc_info,
    ):
        _start(route, request)

    assert detection_calls == 1
    assert exc_info.value.status_code == 400
    assert "Continued Pretraining" in exc_info.value.detail


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
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch("utils.transformers_version.latest_tier_active_for", return_value = False),
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
    assert captured["require_exact_resume_resources"] is False
    assert captured["require_exact_model_resource"] is False
    assert captured["require_exact_dataset_resource"] is False


def test_training_request_does_not_accept_client_strict_resume_flag():
    request = _request(require_exact_resume_resources = True)

    assert not hasattr(request, "require_exact_resume_resources")


def test_stop_route_passes_expected_job_id_to_backend():
    route = _load_route_module("training_route_scoped_stop")
    calls: list[dict] = []
    backend = SimpleNamespace(
        is_training_active = lambda: True,
        stop_training = lambda **kwargs: calls.append(kwargs) or True,
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
    ):
        response = asyncio.run(
            route.stop_training(
                route.TrainingStopRequest(
                    save = False,
                    expected_job_id = "job_old",
                ),
                current_subject = "test-user",
            )
        )

    assert response.status == "stopped"
    assert calls == [{"save": False, "expected_job_id": "job_old"}]


def test_stop_route_reports_superseded_job_without_mutation():
    route = _load_route_module("training_route_superseded_stop")
    calls: list[dict] = []
    backend = SimpleNamespace(
        current_job_id = "job_new",
        is_training_active = lambda: True,
        stop_training = lambda **kwargs: calls.append(kwargs) or True,
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        pytest.raises(route.HTTPException) as exc_info,
    ):
        asyncio.run(
            route.stop_training(
                route.TrainingStopRequest(save = False, expected_job_id = "job_old"),
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert calls == []


def test_stop_request_requires_job_scope():
    route = _load_route_module("training_route_required_stop_scope")

    with pytest.raises(ValueError):
        route.TrainingStopRequest(save = True)


def test_reset_route_reports_superseded_job_without_mutation():
    route = _load_route_module("training_route_scoped_reset")
    calls: list[str | None] = []
    backend = SimpleNamespace(
        reset_training_state = lambda expected_job_id = None: (
            calls.append(expected_job_id) or "superseded"
        )
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
    ):
        response = asyncio.run(
            route.reset_training(
                route.TrainingResetRequest(expected_job_id = "job_old"),
                current_subject = "test-user",
            )
        )

    assert response == {"status": "superseded"}
    assert calls == ["job_old"]


def test_reset_route_without_body_stays_supported():
    """Pre-rework clients POST /reset with no body; that must keep working when idle."""
    route = _load_route_module("training_route_unscoped_reset")
    calls: list[str | None] = []
    backend = SimpleNamespace(
        reset_training_state = lambda expected_job_id = None: (calls.append(expected_job_id) or "ok")
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
    ):
        response = asyncio.run(route.reset_training(current_subject = "test-user"))

    assert response == {"status": "ok"}
    assert calls == [None]


def test_unscoped_reset_cannot_touch_a_live_run(monkeypatch):
    """...but it may not force-terminate a run it cannot prove it owns."""
    from core.training.training import TrainingBackend

    backend = TrainingBackend.__new__(TrainingBackend)
    TrainingBackend.__init__(backend)
    backend.current_job_id = "job_new"
    backend._cancel_requested = True
    monkeypatch.setattr(backend, "is_training_active", lambda: True)
    monkeypatch.setattr(
        backend, "force_terminate", lambda **_kw: pytest.fail("unscoped reset terminated a run")
    )

    # A stop was already requested, so this is the pre-rework cancel-then-dismiss flow and
    # the client may clear its UI. Still no force_terminate: the stub above would fail.
    assert backend.reset_training_state() == "superseded"
    assert backend.reset_training_state(expected_job_id = "job_old") == "superseded"

    # No stop requested, so a bodyless reset of a live run is stale: 409, not a 200 that
    # would tell an older client a running job had been cleared.
    backend._cancel_requested = False
    assert backend.reset_training_state() == "active"


def test_runtime_4bit_resume_reaches_worker_with_source_resource_pins(tmp_path):
    route = _load_route_module("training_route_resume_resource_provenance")
    model_root = tmp_path / "models--unsloth--test"
    old_model = model_root / "snapshots" / "commit-old"
    new_model = model_root / "snapshots" / "commit-new"
    for snapshot in (old_model, new_model):
        snapshot.mkdir(parents = True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"x")
    dataset_root = tmp_path / "datasets--org--dataset"
    old_dataset = dataset_root / "snapshots" / "commit-old"
    new_dataset = dataset_root / "snapshots" / "commit-new"
    for snapshot in (old_dataset, new_dataset):
        snapshot.mkdir(parents = True)
        (snapshot / "train.parquet").write_bytes(b"x")

    source_config = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "hf_dataset": "org/dataset",
        "format_type": "alpaca",
        "model_known_cached": True,
        "model_local_path": str(model_root),
        "model_format": "safetensors",
        "model_snapshot_path": str(old_model),
        "dataset_known_cached": True,
        "dataset_local_path": str(dataset_root),
        "dataset_snapshot_path": str(old_dataset),
        "load_in_4bit": True,
        "resource_provenance": {
            "version": 1,
            "status": "complete",
            "model_status": "attested",
            "model_load_mode": "runtime_4bit",
            "dataset_status": "attested",
            "reasons": [],
        },
    }
    resume_run = {
        "id": "source-run",
        "model_name": "unsloth/test",
        "config_json": json.dumps(source_config),
    }
    request = _request(
        resume_from_checkpoint = "/outputs/source-run",
        model_snapshot_path = str(new_model),
        dataset_snapshot_path = str(new_dataset),
    )
    captured: dict = {}
    tier_targets: list[str] = []
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: captured.update(kwargs) or True,
    )

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "normalize_resume_output_dir", return_value = "/outputs/source-run"),
        patch.object(route, "get_resumable_run_by_output_dir", return_value = resume_run),
        patch.object(route, "can_resume_run", return_value = True),
        patch.object(
            route,
            "get_resume_checkpoint_path",
            return_value = "/outputs/source-run/checkpoint-5",
        ),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch(
            "utils.transformers_version.latest_tier_active_for",
            side_effect = lambda target, _token: tier_targets.append(target) or False,
        ),
        patch(
            "core.inference.get_inference_backend",
            return_value = type("InferenceBackend", (), {"active_model_name": None})(),
        ),
        patch(
            "core.export.get_export_backend",
            return_value = type(
                "ExportBackend",
                (),
                {
                    "current_checkpoint": None,
                    "is_export_active": lambda self: False,
                },
            )(),
        ),
    ):
        response = _start(route, request)

    assert response.status == "queued"
    assert captured["model_snapshot_path"] == str(old_model)
    assert captured["dataset_snapshot_path"] == str(old_dataset)
    # The route posix-normalizes this field, so compare against the same shape.
    assert captured["model_local_path"] == model_root.as_posix()
    assert captured["dataset_local_path"] == str(dataset_root)
    assert captured["load_in_4bit"] is True
    assert captured["require_exact_resume_resources"] is True
    assert captured["require_exact_model_resource"] is True
    assert captured["require_exact_dataset_resource"] is True
    assert captured["resume_model_load_mode"] == "runtime_4bit"
    assert tier_targets == [str(old_model.resolve())]

    from core.training.training import (
        _apply_cache_pins,
        _build_training_worker_config,
    )

    with patch(
        "core.training.training.get_device",
        return_value = SimpleNamespace(value = "cuda"),
    ):
        worker_config = _build_training_worker_config(captured)
    _apply_cache_pins(worker_config)

    assert worker_config["resume_model_load_mode"] == "runtime_4bit"
    assert worker_config["model_snapshot_path"] == str(old_model.resolve())


def test_resume_resource_provenance_restores_model_structure():
    route = _load_route_module("training_route_resume_model_structure")
    request = _request(
        load_in_4bit = False,
        use_lora = False,
        lora_r = 64,
        lora_alpha = 128,
        lora_dropout = 0.25,
        target_modules = ["client_proj"],
        gradient_checkpointing = "client",
        use_dora = True,
        optim = "sgd",
        lr_scheduler_type = "cosine",
        embedding_learning_rate = 5e-5,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = False,
        finetune_mlp_modules = True,
    )
    source_structure = {
        "load_in_4bit": True,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "gradient_checkpointing": "unsloth",
        "use_rslora": True,
        "use_loftq": False,
        "use_dora": False,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "linear",
        "embedding_learning_rate": 2e-5,
        "finetune_vision_layers": True,
        "finetune_language_layers": False,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": False,
    }
    resume_run = {
        "model_name": "unsloth/test",
        "config_json": {
            "model_name": "unsloth/test",
            "training_type": "LoRA/QLoRA",
            "hf_dataset": "org/dataset",
            **source_structure,
        },
    }

    route._prepare_resume_resource_provenance(request, resume_run)

    for field, expected in source_structure.items():
        assert getattr(request, field) == expected


@pytest.mark.parametrize(
    "invalid_structure",
    [
        {"lora_r": 0},
        {"use_rslora": True, "use_dora": True},
        {"gradient_checkpointing": True},
    ],
)
def test_resume_resource_provenance_rejects_invalid_stored_structure(invalid_structure):
    route = _load_route_module("training_route_resume_invalid_structure")
    request = _request()
    resume_run = {
        "model_name": "unsloth/test",
        "config_json": {
            "model_name": "unsloth/test",
            "training_type": "LoRA/QLoRA",
            "hf_dataset": "org/dataset",
            **invalid_structure,
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        route._prepare_resume_resource_provenance(request, resume_run)

    assert exc_info.value.status_code == 409
    assert "invalid training configuration" in exc_info.value.detail


def test_exact_resume_rejects_latest_tier_load_mode_change(tmp_path):
    route = _load_route_module("training_route_resume_latest_tier")
    model = tmp_path / "models--unsloth--test" / "snapshots" / "model-commit"
    model.mkdir(parents = True)
    (model / "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
    (model / "model.safetensors").write_bytes(b"x")
    dataset = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-commit"
    dataset.mkdir(parents = True)
    (dataset / "train.parquet").write_bytes(b"x")
    resume_run = {
        "id": "source-run",
        "model_name": "unsloth/test",
        "config_json": {
            "model_name": "unsloth/test",
            "training_type": "LoRA/QLoRA",
            "hf_dataset": "org/dataset",
            "format_type": "alpaca",
            "model_snapshot_path": str(model),
            "dataset_snapshot_path": str(dataset),
            "load_in_4bit": True,
            "resource_provenance": {
                "version": 1,
                "status": "complete",
                "model_status": "attested",
                "dataset_status": "attested",
                "reasons": [],
            },
        },
    }
    request = _request(resume_from_checkpoint = "/outputs/source-run")

    with (
        patch.object(route, "get_training_backend", return_value = _refusing_backend()),
        patch.object(route, "normalize_resume_output_dir", return_value = "/outputs/source-run"),
        patch.object(route, "get_resumable_run_by_output_dir", return_value = resume_run),
        patch.object(route, "can_resume_run", return_value = True),
        patch.object(
            route,
            "get_resume_checkpoint_path",
            return_value = "/outputs/source-run/checkpoint-5",
        ),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch(
            "utils.transformers_version.latest_tier_active_for",
            return_value = True,
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            _start(route, request)

    assert exc_info.value.status_code == 409
    assert "original 4-bit model load mode" in exc_info.value.detail


@pytest.mark.parametrize(
    ("request_overrides", "detail"),
    [
        ({"model_name": "other/model"}, "selected model"),
        ({"hf_dataset": "other/dataset"}, "selected dataset"),
        ({"training_type": "Full Finetuning"}, "training type"),
    ],
)
def test_resume_resource_provenance_rejects_identity_changes(request_overrides, detail):
    route = _load_route_module(f"training_route_resume_identity_{detail.replace(' ', '_')}")
    request = _request(**request_overrides)
    resume_run = {
        "model_name": "unsloth/test",
        "config_json": {
            "model_name": "unsloth/test",
            "training_type": "LoRA/QLoRA",
            "hf_dataset": "org/dataset",
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        route._prepare_resume_resource_provenance(request, resume_run)

    assert exc_info.value.status_code == 409
    assert detail in exc_info.value.detail


def test_legacy_resume_config_cannot_inject_cache_pins():
    route = _load_route_module("training_route_resume_legacy_cache_pins")
    request = _request(
        hf_dataset = None,
        model_known_cached = True,
        model_local_path = "/cache/models--unsloth--test",
        model_snapshot_path = "/cache/models--unsloth--test/snapshots/client",
        dataset_known_cached = True,
        dataset_local_path = "/cache/datasets--org--dataset",
        dataset_snapshot_path = "/cache/datasets--org--dataset/snapshots/client",
        local_datasets = ["/client/data.jsonl"],
        local_eval_datasets = ["/client/eval.jsonl"],
        s3_config = {
            "bucket": "client-bucket",
            "access_key_id": "client-key",
            "secret_access_key": "client-secret",
        },
    )
    source_config = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "hf_dataset": "",
        "require_exact_resume_resources": True,
    }
    resume_run = {
        "model_name": "unsloth/test",
        "config_json": source_config,
    }

    route._prepare_resume_resource_provenance(request, resume_run)

    assert request.model_known_cached is False
    assert request.model_local_path is None
    assert request.model_snapshot_path is None
    assert request.dataset_known_cached is False
    assert request.dataset_local_path is None
    assert request.dataset_snapshot_path is None
    assert request.local_datasets == []
    assert request.local_eval_datasets == []
    assert request.s3_config is None

    from core.training.provenance import resource_provenance_is_complete

    assert resource_provenance_is_complete(source_config) is False


@pytest.mark.parametrize("status", ["pending", "incomplete"])
def test_unattested_current_hub_model_resume_is_rejected(status):
    route = _load_route_module(f"training_route_unattested_hub_model_{status}")
    request = _request(hf_dataset = None)
    resume_run = {
        "model_name": "unsloth/test",
        "config_json": {
            "model_name": "unsloth/test",
            "training_type": "LoRA/QLoRA",
            "hf_dataset": "",
            "resource_provenance": {"version": 1, "status": status},
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        route._prepare_resume_resource_provenance(request, resume_run)

    assert exc_info.value.status_code == 409


def test_foreign_absolute_resume_paths_are_rejected_on_native_host():
    from core.training.resume import (
        artifacts_present,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
    )

    foreign = (
        "/var/lib/unsloth/outputs/run"
        if os.name == "nt"
        else r"C:\Users\alice\.unsloth\studio\outputs\run"
    )

    with pytest.raises(ValueError, match = "different operating system"):
        normalize_resume_output_dir(foreign)
    assert get_resume_checkpoint_path(foreign) is None
    assert artifacts_present(foreign) is False


def test_resume_symlink_loop_paths_fail_closed(monkeypatch, tmp_path):
    from core.training.resume import (
        artifacts_present,
        get_resume_checkpoint_path,
        normalize_resume_output_dir,
    )

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    loop = outputs / "loop"
    try:
        loop.symlink_to(loop, target_is_directory = True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable: {error}")

    assert artifacts_present(str(loop)) is False
    assert get_resume_checkpoint_path(str(loop)) is None
    with pytest.raises(ValueError):
        normalize_resume_output_dir(str(loop))


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
    assert backend._db_config["resource_provenance"] == {"version": 1, "status": "pending"}


def _dataset_repo_with_ref(
    root: Path,
    repo_id: str,
    commit: str = "rev",
) -> Path:
    repo_root = root / f"datasets--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / commit
    snap.mkdir(parents = True)
    (snap / "train.parquet").write_bytes(b"x")
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(commit)
    return snap


def _model_repo_with_ref(
    root: Path,
    repo_id: str,
    commit: str = "rev",
) -> Path:
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
    assert config["dataset_revision"] == "rev"
    assert config["cache_pin_warnings"] == []


def test_resolve_model_snapshot_keeps_selected_cache_path_strict(monkeypatch):
    from core.training.training import _resolve_model_snapshot
    from hub.utils import hf_cache_state

    selected_path = "/cache-a/models--unsloth--test"
    fallback_path = Path("/cache-b/models--unsloth--test")
    fallback_snapshot = "/cache-b/models--unsloth--test/snapshots/rev"
    resolved_paths = []

    def resolve_snapshot(local_path, *_args, **_kwargs):
        resolved_paths.append(local_path)
        return fallback_snapshot if local_path == str(fallback_path) else None

    monkeypatch.setattr(hf_cache_state, "latest_snapshot_from_cache_path", resolve_snapshot)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda *_args: iter([fallback_path]),
    )

    # Each path is probed once per metadata set, so assert which paths were consulted, not how many.
    assert _resolve_model_snapshot("unsloth/test", selected_path) is None
    assert set(resolved_paths) == {selected_path}

    resolved_paths.clear()
    assert _resolve_model_snapshot("unsloth/test", None) == fallback_snapshot
    assert set(resolved_paths) == {str(fallback_path)}


@pytest.mark.parametrize(
    "resume_from_checkpoint",
    [None, "/outputs/run/checkpoint-5"],
)
def test_apply_cache_pins_keeps_local_model_out_of_hub_cache_resolution(
    tmp_path, resume_from_checkpoint
):
    from core.training.training import (
        _apply_cache_pins,
        resolve_training_model_load_target,
    )

    model_path = tmp_path / "models" / "custom"
    model_path.mkdir(parents = True)
    config = {
        "model_name": str(model_path),
        "model_known_cached": True,
        "model_local_path": str(model_path),
        "model_snapshot_path": str(model_path),
        "actual_model_repo_id": "stale/repo-id",
        "resume_from_checkpoint": resume_from_checkpoint,
        "hf_dataset": "",
    }

    _apply_cache_pins(config)

    assert config["model_snapshot_path"] is None
    assert config["actual_model_repo_id"] is None
    assert config["cache_pin_warnings"] == []
    assert resolve_training_model_load_target(config) == str(model_path)


def test_legacy_cached_dataset_loads_offline_without_completion_manifest(monkeypatch, tmp_path):
    from core.training import worker
    from core.training.training import _apply_cache_pins

    snapshot = _dataset_repo_with_ref(
        tmp_path,
        "org/dataset",
        "dataset-commit",
    )
    config = {
        "model_name": "unsloth/test",
        "hf_dataset": "org/dataset",
        "dataset_known_cached": True,
        "dataset_local_path": str(snapshot.parent.parent),
        "train_split": "train",
    }
    _apply_cache_pins(config)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    events: list[dict] = []

    assert worker._verify_config_pins(config, SimpleNamespace(put = events.append)) is True
    assert config["dataset_snapshot_path"] == str(snapshot.resolve())
    assert config["dataset_revision"] == "dataset-commit"

    dataset = SimpleNamespace(
        info = SimpleNamespace(
            download_checksums = {
                str(snapshot / "train.parquet"): {
                    "num_bytes": 1,
                    "checksum": None,
                }
            }
        )
    )
    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        return_value = dataset,
    ):
        loaded, evaluation = worker._load_hf_train_and_eval_datasets(
            config,
            None,
            lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
            lambda _message: None,
        )

    assert loaded is dataset
    assert evaluation is None
    assert config["_dataset_loaded_from_exact_snapshot"] is True
    assert events == []


@pytest.mark.parametrize("load_in_4bit", [False, True])
def test_training_model_load_target_uses_verified_inactive_snapshot(tmp_path, load_in_4bit):
    from core.training.training import resolve_training_model_load_target
    model_snap = _model_repo_with_ref(tmp_path, "unsloth/test", "commit-old")

    assert resolve_training_model_load_target(
        {
            "model_name": "unsloth/test",
            "model_known_cached": True,
            "model_local_path": str(model_snap),
            "load_in_4bit": load_in_4bit,
        }
    ) == str(model_snap.resolve())


def test_training_model_load_target_rejects_evicted_strict_resume_pin(tmp_path):
    from core.training.provenance import ExactResumeResourcesUnavailable
    from core.training.training import resolve_training_model_load_target
    with pytest.raises(ExactResumeResourcesUnavailable, match = "model snapshot"):
        resolve_training_model_load_target(
            {
                "model_name": "unsloth/test",
                "model_snapshot_path": str(tmp_path / "evicted"),
                "resume_from_checkpoint": "/outputs/run/checkpoint-5",
                "require_exact_resume_resources": True,
                "load_in_4bit": True,
            }
        )


def test_backend_rechecks_exact_4bit_resume_after_sidecar_reservation(tmp_path):
    from core.training.provenance import ExactResumeResourcesUnavailable
    from core.training.training import TrainingBackend

    model = tmp_path / "models--unsloth--test" / "snapshots" / "model-rev"
    model.mkdir(parents = True)
    (model / "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
    (model / "model.safetensors").write_bytes(b"x")
    dataset = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-rev"
    dataset.mkdir(parents = True)
    (dataset / "train.parquet").write_bytes(b"x")
    before_spawn_called = False

    def before_spawn():
        nonlocal before_spawn_called
        before_spawn_called = True

    backend = TrainingBackend()
    with (
        patch(
            "core.training.resume.get_resume_checkpoint_path",
            return_value = "/outputs/run/checkpoint-5",
        ),
        patch("utils.transformers_version.sidecar_swap_in_progress", return_value = False),
        patch("utils.transformers_version.latest_tier_active_for", return_value = True),
    ):
        with pytest.raises(ExactResumeResourcesUnavailable, match = "original 4-bit"):
            backend.start_training(
                job_id = "exact-resume",
                before_spawn = before_spawn,
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                hf_dataset = "org/dataset",
                model_snapshot_path = str(model),
                dataset_snapshot_path = str(dataset),
                resume_from_checkpoint = "/outputs/run/checkpoint-5",
                require_exact_resume_resources = True,
                load_in_4bit = True,
            )

    assert before_spawn_called is False
    assert backend._spawn_in_progress is False


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


def test_apply_cache_pins_attested_resume_rejects_evicted_dataset(tmp_path):
    from core.training.provenance import ExactResumeResourcesUnavailable
    from core.training.training import _apply_cache_pins

    model = tmp_path / "models--unsloth--test" / "snapshots" / "model-rev"
    model.mkdir(parents = True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"x")
    dataset = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-rev"
    dataset.mkdir(parents = True)

    config = {
        "model_name": "unsloth/test",
        "model_snapshot_path": str(model),
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": str(dataset),
        "resume_from_checkpoint": "/outputs/run/checkpoint-5",
        "load_in_4bit": False,
        "require_exact_resume_resources": True,
    }

    with pytest.raises(ExactResumeResourcesUnavailable, match = "dataset snapshot"):
        _apply_cache_pins(config)


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


@pytest.mark.parametrize("offline", [False, True])
def test_worker_security_scans_exact_model_load_target(offline):
    from core.training import worker

    snapshot = "/cache/models--org--model/snapshots/deadbeef"
    scanned: list[tuple[str, bool]] = []
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
            side_effect = lambda target, **kwargs: scanned.append((target, kwargs["local_only_load"]))
            or decision,
        ),
        patch("utils.utils.hf_env_offline", return_value = offline),
    ):
        result = worker._model_load_security_error(
            {
                "model_name": "org/model",
                "model_snapshot_path": snapshot,
                "trust_remote_code": False,
            },
            snapshot,
            None,
        )

    assert result is None
    assert scanned == [(snapshot, offline)]


def test_worker_remote_retry_security_scan_is_not_local_only():
    from core.training import worker

    snapshot = "/cache/models--org--model/snapshots/deadbeef"
    scanned: list[tuple[str, bool]] = []
    decision = SimpleNamespace(blocked = False)

    with (
        patch(
            "utils.models.model_config.get_base_model_from_lora_identifier",
            return_value = None,
        ),
        patch("utils.security.security_load_subdirs", return_value = ()),
        patch(
            "utils.security.evaluate_file_security",
            side_effect = lambda target, **kwargs: scanned.append((target, kwargs["local_only_load"]))
            or decision,
        ),
        patch("utils.utils.hf_env_offline", return_value = False),
    ):
        result = worker._model_load_security_error(
            {
                "model_name": "org/model",
                "model_snapshot_path": snapshot,
                "trust_remote_code": False,
            },
            "org/model",
            None,
        )

    assert result is None
    assert scanned == [("org/model", False)]


def test_worker_security_scopes_pinned_target_before_registry_fallback():
    from core.training import worker

    snapshot = "/cache/models--org--model/snapshots/deadbeef"
    scanned: list[tuple[str, tuple[str, ...]]] = []
    decision = SimpleNamespace(blocked = False)

    with (
        patch(
            "utils.models.model_config.get_base_model_from_lora_identifier",
            return_value = None,
        ),
        patch(
            "utils.security.security_load_subdirs",
            side_effect = lambda target, _token: (("LLM",) if target == snapshot else ("registry",)),
        ),
        patch(
            "utils.security.evaluate_file_security",
            side_effect = lambda target, **kwargs: scanned.append((target, kwargs["load_subdirs"]))
            or decision,
        ),
        patch("utils.utils.hf_env_offline", return_value = False),
    ):
        result = worker._model_load_security_error(
            {
                "model_name": "org/model",
                "model_snapshot_path": snapshot,
                "trust_remote_code": False,
            },
            snapshot,
            None,
        )

    assert result is None
    assert scanned == [(snapshot, ("LLM", "registry"))]


def test_worker_security_consent_uses_exact_target_and_base():
    from core.training import worker

    snapshot = "/cache/models--org--adapter/snapshots/deadbeef"
    consent_targets: list[str] = []
    consent_kwargs: dict = {}
    file_decision = SimpleNamespace(blocked = False)
    consent_decision = SimpleNamespace(blocked = False)

    def evaluate_consent(targets, **kwargs):
        consent_targets.extend(targets)
        consent_kwargs.update(kwargs)
        return consent_decision

    with (
        patch(
            "utils.models.model_config.get_base_model_from_lora_identifier",
            return_value = "org/base",
        ),
        patch(
            "utils.security.security_load_subdirs",
            side_effect = lambda target, _token: (
                ("LLM",) if target in {snapshot, "org/adapter"} else ()
            ),
        ),
        patch(
            "utils.security.evaluate_file_security",
            return_value = file_decision,
        ),
        patch(
            "utils.security.evaluate_remote_code_consent_for_targets",
            side_effect = evaluate_consent,
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
    assert consent_kwargs["load_subdirs_by_target"] == {
        snapshot: ("LLM",),
        "org/base": (),
    }


def test_worker_resolves_cached_model_snapshot():
    from core.training import worker

    assert (
        worker._resolve_cached_model_load_name(
            {"model_name": "unsloth/test", "model_snapshot_path": "/snap/dir"}
        )
        == "/snap/dir"
    )
    assert worker._resolve_cached_model_load_name({"model_name": "unsloth/test"}) == "unsloth/test"
    assert (
        worker._resolve_cached_model_load_name(
            {"model_name": "unsloth/test", "model_snapshot_path": None}
        )
        == "unsloth/test"
    )


def test_worker_4bit_tier_check_uses_model_load_target():
    from core.training import worker

    snapshot = "/cache/models--org--model/snapshots/deadbeef"
    checked: list[tuple[str, str | None]] = []

    with patch(
        "utils.transformers_version.latest_tier_active_for",
        side_effect = lambda target, token: checked.append((target, token)) or True,
    ):
        enabled = worker._effective_training_load_in_4bit(
            {"load_in_4bit": True},
            snapshot,
            "hf_test",
        )

    assert enabled is False
    assert checked == [(snapshot, "hf_test")]


def test_worker_strict_resume_rejects_4bit_tier_change():
    from core.training import worker
    from core.training.provenance import ExactResumeResourcesUnavailable
    with patch(
        "utils.transformers_version.latest_tier_active_for",
        return_value = True,
    ):
        with pytest.raises(ExactResumeResourcesUnavailable, match = "original 4-bit"):
            worker._effective_training_load_in_4bit(
                {
                    "load_in_4bit": True,
                    "require_exact_resume_resources": True,
                },
                "/cache/models--org--model/snapshots/deadbeef",
                None,
            )


def test_worker_model_cache_fallback_requires_same_transformers_tier():
    from core.training import worker

    config = {
        "model_name": "org/model",
        "model_snapshot_path": "/cache/snapshot",
        "actual_model_repo_id": "org/model",
    }

    with patch(
        "utils.transformers_version.get_transformers_activation_tier",
        side_effect = ["default", "530"],
    ):
        with pytest.raises(RuntimeError, match = "different Transformers runtime"):
            worker._drop_model_pin_for_fallback(config, None)

    assert config["model_snapshot_path"] == "/cache/snapshot"
    assert config["actual_model_repo_id"] == "org/model"


def test_worker_model_cache_fallback_drops_pin_for_matching_transformers_tier():
    from core.training import worker

    config = {
        "model_name": "org/model",
        "model_snapshot_path": "/cache/snapshot",
        "actual_model_repo_id": "org/model",
    }

    with patch(
        "utils.transformers_version.get_transformers_activation_tier",
        return_value = "530",
    ):
        target = worker._drop_model_pin_for_fallback(config, "hf_test")

    assert target == "org/model"
    assert config["model_snapshot_path"] is None
    assert config["actual_model_repo_id"] == "org/model"


def test_worker_model_cache_fallback_preserves_exact_revision_without_head_probe():
    from core.training import worker

    config = {
        "model_name": "org/model",
        "model_snapshot_path": "/cache/models--org--model/snapshots/deadbeef",
        "actual_model_repo_id": "org/model",
        "model_revision": "deadbeef",
    }

    with patch(
        "utils.transformers_version.get_transformers_activation_tier",
        side_effect = AssertionError("the pinned revision already selected the active tier"),
    ):
        target = worker._drop_model_pin_for_fallback(config, "hf_test")

    assert target == "org/model"
    assert config["model_snapshot_path"] is None
    assert config["actual_model_repo_id"] == "org/model"
    assert config["model_revision"] == "deadbeef"


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


def test_worker_cached_dataset_loader_forwards_row_limit():
    from core.training import worker

    with patch(
        "hub.utils.dataset_cache.load_cached_hf_dataset",
        return_value = {"loaded": True},
    ) as load_cached:
        result = worker._load_cached_dataset_for_config(
            {
                "hf_dataset": "org/dataset",
                "dataset_snapshot_path": "/verified/cache",
            },
            "train",
            row_limit = 33,
        )

    assert result == {"loaded": True}
    load_cached.assert_called_once_with(
        "org/dataset",
        "/verified/cache",
        subset = None,
        split = "train",
        token = None,
        row_limit = 33,
    )


def test_shared_hf_loader_bounds_cached_train_slice():
    from core.training import worker

    cached = object()
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/verified/cache",
        "train_split": "train",
        "dataset_slice_start": 8,
        "dataset_slice_end": 32,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        return_value = cached,
    ) as load_cached:
        dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
            config,
            None,
            lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
            lambda _message: None,
        )

    assert dataset is cached
    assert eval_dataset is None
    load_cached.assert_called_once_with(config, "train", None, row_limit = 33)


def test_embedding_hf_loader_bounds_cached_train_slice():
    from core.training import worker

    cached = object()
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/verified/cache",
        "train_split": "train",
        "dataset_slice_start": 8,
        "dataset_slice_end": 32,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        return_value = cached,
    ) as load_cached:
        dataset = worker._load_embedding_hf_dataset(
            config,
            lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
            lambda _message: None,
        )

    assert dataset is cached
    load_cached.assert_called_once_with(config, "train", None, row_limit = 33)


@pytest.mark.parametrize(
    "cached_eval_error",
    [
        FileNotFoundError("validation"),
        ValueError("Unknown split \"validation\". Should be one of ['train']."),
    ],
)
def test_worker_cached_eval_failure_reloads_remote_pair(cached_eval_error):
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
        "eval_steps": 0.1,
        "subset": "english",
    }

    def load_cached(request_config, split, token):
        cached_calls.append(split)
        if split == "validation":
            raise cached_eval_error
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
    assert config["_dataset_loaded_from_exact_snapshot"] is False


def test_worker_explicit_eval_failure_is_fatal():
    from core.training import worker

    calls: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
        "eval_split": "validation",
        "eval_steps": 0.1,
    }

    def load_remote(_repo_id, **kwargs):
        split = kwargs["split"]
        calls.append(split)
        if split == "validation":
            raise ValueError("validation is unavailable")
        return object()

    with pytest.raises(ValueError, match = "validation is unavailable"):
        worker._load_hf_train_and_eval_datasets(
            config,
            None,
            load_remote,
            lambda _message: None,
        )

    assert calls == ["train", "validation"]


def test_worker_does_not_load_explicit_eval_when_evaluation_is_disabled():
    from core.training import worker

    train = object()
    calls: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
        "eval_split": "validation",
        "eval_steps": 0,
    }

    def load_remote(_repo_id, **kwargs):
        calls.append(kwargs["split"])
        return train

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        load_remote,
        lambda _message: None,
    )

    assert dataset is train
    assert eval_dataset is None
    assert calls == ["train"]


def test_worker_same_train_and_eval_split_defers_to_held_out_split():
    from core.training import worker

    train = object()
    calls: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
        "eval_split": "train",
        "eval_steps": 0.1,
    }

    def load_remote(_repo_id, **kwargs):
        calls.append(kwargs["split"])
        return train

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        load_remote,
        lambda _message: None,
    )

    assert dataset is train
    assert eval_dataset is None
    assert calls == ["train"]


def test_worker_auto_eval_load_failure_warns_and_falls_back(monkeypatch):
    from core.training import worker

    train = SimpleNamespace(info = SimpleNamespace(splits = None))
    warnings: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train",
        "eval_split": None,
        "eval_steps": 0.1,
    }
    monkeypatch.setattr(
        "datasets.get_dataset_split_names",
        lambda **_kwargs: ["train", "validation"],
    )

    def load_remote(_repo_id, **kwargs):
        if kwargs["split"] == "validation":
            raise OSError("eval download failed")
        return train

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        load_remote,
        lambda _message: None,
        warnings.append,
    )

    assert dataset is train
    assert eval_dataset is None
    assert len(warnings) == 1
    assert "held-out split" in warnings[0]
    assert "eval download failed" in warnings[0]


def test_worker_auto_eval_excludes_every_split_in_training_instruction(monkeypatch):
    from core.training import worker

    train = SimpleNamespace(info = SimpleNamespace(splits = None))
    held_out = list(range(20))
    calls: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "train_split": "train + validation",
        "eval_split": None,
        "eval_steps": 0.1,
    }
    monkeypatch.setattr(
        "datasets.get_dataset_split_names",
        lambda **_kwargs: ["train", "validation", "test"],
    )

    def load_remote(_repo_id, **kwargs):
        split = kwargs["split"]
        calls.append(split)
        return held_out if split == "test" else train

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        load_remote,
        lambda _message: None,
    )

    assert dataset is train
    assert eval_dataset is held_out
    assert calls == ["train + validation", "test"]


def test_worker_cached_auto_eval_without_split_metadata_stays_offline(monkeypatch):
    from core.training import worker

    train = SimpleNamespace(info = SimpleNamespace(splits = None))
    cache_calls: list[str] = []
    warnings: list[str] = []
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/verified/cache",
        "train_split": "train",
        "eval_split": None,
        "eval_steps": 0.1,
    }

    def load_cached(_config, split, _token, **_kwargs):
        cache_calls.append(split)
        return train

    monkeypatch.setattr(worker, "_load_cached_dataset_for_config", load_cached)
    monkeypatch.setattr(
        "datasets.get_dataset_split_names",
        lambda **_kwargs: pytest.fail("cached auto eval must not probe the Hub"),
    )

    dataset, eval_dataset = worker._load_hf_train_and_eval_datasets(
        config,
        None,
        lambda *_args, **_kwargs: pytest.fail("cached auto eval must not load remotely"),
        lambda _message: None,
        warnings.append,
    )

    assert dataset is train
    assert eval_dataset is None
    assert cache_calls == ["train"]
    assert warnings == []


def test_worker_model_retry_refreshes_tokenizer_before_dataset():
    from core.training import worker

    events: list[tuple[str, object]] = []
    trainer = SimpleNamespace(
        pre_detect_and_load_tokenizer = (lambda **kwargs: events.append(("tokenizer", kwargs)))
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
        "deadbeef",
    )

    assert result is expected
    assert [event[0] for event in events] == ["tokenizer", "dataset"]
    tokenizer_kwargs = events[0][1]
    assert tokenizer_kwargs["model_name"] == "org/model"
    assert tokenizer_kwargs["model_load_name"] == "org/model"
    assert tokenizer_kwargs["local_files_only"] is False
    assert tokenizer_kwargs["hf_token"] == "hf_test"
    assert tokenizer_kwargs["model_revision"] == "deadbeef"


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


def test_worker_bootstrap_rejects_vanished_strict_resume_pins(tmp_path):
    from core.training import worker

    events: list[dict] = []
    queue = SimpleNamespace(put = events.append)
    config = {
        "model_name": "unsloth/test",
        "hf_dataset": "org/dataset",
        "model_snapshot_path": str(tmp_path / "gone-model"),
        "dataset_snapshot_path": str(tmp_path / "gone-dataset"),
        "load_in_4bit": False,
        "require_exact_resume_resources": True,
    }

    assert worker._verify_config_pins(config, queue) is False
    assert events == [
        {
            "type": "error",
            "error": "The exact model snapshot for this run is no longer available.",
            "stack": "",
            "ts": events[0]["ts"],
        }
    ]


def test_worker_enforces_exact_hub_model_with_local_dataset(tmp_path):
    from core.training import worker

    snapshot = tmp_path / "models--org--model" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")
    config = {
        "model_name": "org/model",
        "model_snapshot_path": str(snapshot),
        "local_datasets": ["/datasets/train.jsonl"],
        "require_exact_model_resource": True,
        "require_exact_dataset_resource": False,
    }
    events: list[dict] = []
    queue = SimpleNamespace(put = events.append)

    assert worker._verify_config_pins(config, queue) is True
    assert config["model_snapshot_path"] == str(snapshot.resolve())
    assert events == []

    weights.unlink()
    assert worker._verify_config_pins(config, queue) is False
    assert "exact model snapshot" in events[-1]["error"]


def test_strict_resume_disables_cache_artifact_fallback():
    from core.training import worker

    error = FileNotFoundError("evicted")

    assert worker._cache_artifact_fallback_allowed({}, error, "model") is True
    assert (
        worker._cache_artifact_fallback_allowed(
            {"require_exact_resume_resources": True},
            error,
            "model",
        )
        is False
    )
    assert (
        worker._cache_artifact_fallback_allowed(
            {"require_exact_model_resource": True},
            error,
            "model",
        )
        is False
    )
    assert (
        worker._cache_artifact_fallback_allowed(
            {"require_exact_model_resource": True},
            error,
            "dataset",
        )
        is True
    )


def test_strict_resume_cached_dataset_failure_never_loads_remote():
    from core.training import worker
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/cache/exact",
        "train_split": "train",
        "require_exact_resume_resources": True,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        side_effect = FileNotFoundError("evicted"),
    ):
        with pytest.raises(FileNotFoundError, match = "evicted"):
            worker._load_hf_train_and_eval_datasets(
                config,
                None,
                lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
                lambda _message: None,
            )


def test_pinned_dataset_cache_failure_never_falls_back_offline(monkeypatch):
    from core.training import worker

    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/cache/exact",
        "dataset_revision": "dataset-commit",
        "train_split": "train",
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        side_effect = FileNotFoundError("corrupt cache"),
    ):
        with pytest.raises(FileNotFoundError, match = "corrupt cache"):
            worker._load_hf_train_and_eval_datasets(
                config,
                None,
                lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
                lambda _message: None,
            )


def test_missing_pinned_dataset_fails_preflight_offline(monkeypatch):
    from core.training import worker

    monkeypatch.setenv("HF_HUB_OFFLINE", "true")
    events: list[dict] = []
    config = {
        "hf_dataset": "org/dataset",
        "dataset_revision": "dataset-commit",
        "dataset_snapshot_path": None,
    }

    assert worker._verify_config_pins(config, SimpleNamespace(put = events.append)) is False
    assert len(events) == 1
    assert "cannot be downloaded while offline" in events[0]["error"]


def test_mlx_adapter_accepts_and_preserves_dataset_cache_pins():
    from core.training.training import _MLXTrainerAdapter

    adapter = _MLXTrainerAdapter()

    result = adapter.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        dataset_revision = "dataset-commit",
        require_exact_resume_resources = True,
    )

    assert result is not None
    assert adapter._dataset_config["dataset_snapshot_path"] == "/cache/snapshot"
    assert adapter._dataset_config["dataset_revision"] == "dataset-commit"
    assert adapter._dataset_config["require_exact_dataset_resource"] is True


def test_strict_resume_cached_dataset_none_never_loads_remote():
    from core.training import worker
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/cache/exact",
        "train_split": "train",
        "require_exact_resume_resources": True,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        return_value = None,
    ):
        with pytest.raises(FileNotFoundError, match = "exact cached dataset split 'train'"):
            worker._load_hf_train_and_eval_datasets(
                config,
                None,
                lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
                lambda _message: None,
            )


def test_strict_resume_cached_eval_none_never_loads_remote():
    from core.training import worker
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/cache/exact",
        "train_split": "train",
        "eval_split": "validation",
        "eval_steps": 0.1,
        "require_exact_resume_resources": True,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        side_effect = [object(), None],
    ):
        with pytest.raises(
            FileNotFoundError,
            match = "exact cached dataset split 'validation'",
        ):
            worker._load_hf_train_and_eval_datasets(
                config,
                None,
                lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
                lambda _message: None,
            )


def test_strict_resume_embedding_cached_dataset_none_never_loads_remote():
    from core.training import worker
    config = {
        "hf_dataset": "org/dataset",
        "dataset_snapshot_path": "/cache/exact",
        "train_split": "train",
        "require_exact_resume_resources": True,
    }

    with patch.object(
        worker,
        "_load_cached_dataset_for_config",
        return_value = None,
    ):
        with pytest.raises(FileNotFoundError, match = "exact cached dataset split 'train'"):
            worker._load_embedding_hf_dataset(
                config,
                lambda *_args, **_kwargs: pytest.fail("remote load must not run"),
                lambda _message: None,
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
