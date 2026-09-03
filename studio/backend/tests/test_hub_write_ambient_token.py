# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend's own HF_TOKEN must not be lent to API-key callers on Hub write paths.

Issue #10126: Write endpoints (recipe dataset publishing, model exports pushing to Hub,
and checkpoint loading) must require an explicit HF token from sk-unsloth API key callers
rather than borrowing the server operator's ambient HF_TOKEN.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from routes import export as export_routes
from routes.data_recipe import jobs as data_recipe_jobs_routes


async def _fake_ensure_export_supported():
    pass


def _create_test_app(via_api_key: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(data_recipe_jobs_routes.router, prefix = "/api/data-recipe")
    app.include_router(export_routes.router, prefix = "/api")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[get_current_credential] = lambda: ("alice", "cred-1")
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return app


# ==============================================================================
# Data Recipe Publish Endpoint Tests
# ==============================================================================


def test_recipe_publish_refuses_api_key_without_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {"repo_id": "org/dataset", "description": "Test dataset", "hf_token": None},
    )
    assert response.status_code == 400
    assert (
        "Hugging Face token is required to publish datasets when authenticated via API key"
        in response.json()["detail"]
    )


def test_recipe_publish_allows_api_key_with_explicit_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    seen_token = {}

    def _fake_publish(artifact_path, repo_id, description, hf_token, private):
        seen_token["token"] = hf_token
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)
    monkeypatch.setattr(data_recipe_jobs_routes, "publish_recipe_dataset", _fake_publish)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {
            "repo_id": "org/dataset",
            "description": "Test dataset",
            "hf_token": "hf_custom_caller_token",
        },
    )
    assert response.status_code == 200
    assert seen_token["token"] == "hf_custom_caller_token"


def test_recipe_publish_allows_ui_session_without_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    seen_token = {}

    def _fake_publish(artifact_path, repo_id, description, hf_token, private):
        seen_token["token"] = hf_token
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)
    monkeypatch.setattr(data_recipe_jobs_routes, "publish_recipe_dataset", _fake_publish)

    app = _create_test_app(via_api_key = False)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {"repo_id": "org/dataset", "description": "Test dataset", "hf_token": None},
    )
    assert response.status_code == 200
    assert seen_token["token"] is None


# ==============================================================================
# Export Endpoints Tests
# ==============================================================================


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        (
            "/api/export/merged",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/merged-model",
                "hf_token": None,
            },
        ),
        (
            "/api/export/base",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/base-model",
                "hf_token": None,
            },
        ),
        (
            "/api/export/gguf",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/gguf-model",
                "hf_token": None,
                "quantization_method": "q4_k_m",
            },
        ),
        (
            "/api/export/lora",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/lora-model",
                "hf_token": None,
            },
        ),
    ],
)
def test_export_push_to_hub_refuses_api_key_without_token(monkeypatch, endpoint, payload):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(endpoint, json = payload)
    assert response.status_code == 400
    assert (
        "Hugging Face token is required to push to Hub when authenticated via API key"
        in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "endpoint,method_name,payload",
    [
        (
            "/api/export/merged",
            "export_merged_model",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/merged-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
        (
            "/api/export/base",
            "export_base_model",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/base-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
        (
            "/api/export/gguf",
            "export_gguf",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/gguf-model",
                "hf_token": "hf_explicit_key_123",
                "quantization_method": "q4_k_m",
            },
        ),
        (
            "/api/export/lora",
            "export_lora_adapter",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/lora-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
    ],
)
def test_export_push_to_hub_allows_api_key_with_explicit_token(
    monkeypatch, endpoint, method_name, payload
):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    getattr(mock_backend, method_name).return_value = (True, "Export successful", "/tmp/export")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)
    monkeypatch.setattr(
        export_routes,
        "_export_details",
        lambda *args, **kwargs: {"output_path": "/tmp/export"},
    )

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(endpoint, json = payload)
    assert response.status_code == 200
    call_kwargs = getattr(mock_backend, method_name).call_args.kwargs
    assert call_kwargs["hf_token"] == "hf_explicit_key_123"


def test_load_checkpoint_passes_allow_ambient_false_for_api_key(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": None},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] is None
    assert call_kwargs["allow_ambient"] is False


def test_load_checkpoint_passes_allow_ambient_true_for_ui_session(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = False)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": None},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] is None
    assert call_kwargs["allow_ambient"] is True


def test_load_checkpoint_passes_explicit_token_for_api_key(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": "hf_explicit_123"},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] == "hf_explicit_123"
    assert call_kwargs["allow_ambient"] is False


@pytest.fixture
def worker_in_process(monkeypatch):
    """Let run_export_process run here without it taking over the pytest process.

    _setup_log_capture dup2s pipes over fds 1 and 2 and rebinds sys.stdout/sys.stderr with no
    teardown: it is written for the dedicated subprocess, so in-process it swallows every later
    line pytest prints. The offline probe is a live Hub request, and the code's own env opt-out
    keeps it out of a unit test.
    """
    from core.export import worker

    monkeypatch.setattr(worker, "_setup_log_capture", lambda resp_queue: None)
    monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "0")
    return worker


def test_worker_scrubs_ambient_token_when_allow_ambient_false(monkeypatch, worker_in_process):
    import os

    worker = worker_in_process

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.setenv("HF_HUB_TOKEN", "hf_hub_secret_456")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_secret_789")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen_env = {}

    def _fake_activate(path, token):
        seen_env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen_env["HF_HUB_TOKEN"] = os.environ.get("HF_HUB_TOKEN")
        seen_env["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen_env["passed_token"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    config = {
        "checkpoint_path": "/tmp/model",
        "allow_ambient": False,
        "hf_token": None,
    }

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = config,
        )

    assert seen_env.get("HF_TOKEN") is None
    assert seen_env.get("HF_HUB_TOKEN") is None
    assert seen_env.get("DISABLE_IMPLICIT") == "1"
    # The sentinel, not None: None asks the tier probe to go and find a credential.
    assert seen_env.get("passed_token") is False


def test_worker_env_holds_no_credential_at_all_when_the_caller_sent_its_own_token(
    monkeypatch, worker_in_process
):
    """A caller token does not make the operator's harmless, and does not belong here either.

    get_token() reads HF_TOKEN and HUGGING_FACE_HUB_TOKEN, so leaving the operator's in place
    would let every in-worker Hub call still at token=None authenticate as the operator. Putting
    this caller's token there instead just inverts the leak: the worker outlives the load and
    goes on serving export commands from other callers. It travels as an argument instead.
    """
    import os

    worker = worker_in_process

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.setenv("HF_HUB_TOKEN", "hf_hub_secret_456")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_secret_789")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_legacy_secret_012")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen_env = {}

    def _fake_activate(path, token):
        seen_env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen_env["HF_HUB_TOKEN"] = os.environ.get("HF_HUB_TOKEN")
        seen_env["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        seen_env["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        seen_env["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen_env["passed_token"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    config = {
        "checkpoint_path": "/tmp/model",
        "allow_ambient": False,
        "hf_token": "hf_caller_own_token",
    }

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = config,
        )

    assert seen_env.get("HF_TOKEN") is None
    assert seen_env.get("HF_HUB_TOKEN") is None
    assert seen_env.get("HUGGING_FACE_HUB_TOKEN") is None
    assert seen_env.get("HUGGINGFACEHUB_API_TOKEN") is None
    assert seen_env.get("DISABLE_IMPLICIT") == "1"
    # Not lost, just not ambient: this load still runs under the credential it was given.
    assert seen_env.get("passed_token") == "hf_caller_own_token"


def test_worker_preserves_ambient_token_when_allow_ambient_true(monkeypatch, worker_in_process):
    import os

    worker = worker_in_process

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen_env = {}

    def _fake_activate(path, token):
        seen_env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen_env["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen_env["passed_token"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    config = {
        "checkpoint_path": "/tmp/model",
        "allow_ambient": True,
        "hf_token": None,
    }

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = config,
        )

    assert seen_env.get("HF_TOKEN") == "hf_operator_secret_123"
    assert seen_env.get("DISABLE_IMPLICIT") is None
    assert seen_env.get("passed_token") is None


@pytest.mark.parametrize(
    "allow_ambient,caller_token,expected",
    [
        (False, None, False),
        (False, "hf_caller_own_token", "hf_caller_own_token"),
        (True, None, None),
    ],
)
def test_worker_load_preflight_sees_the_callers_credential(
    monkeypatch, allow_ambient, caller_token, expected
):
    """The flag crosses the process boundary, but the preflight helpers read the token.

    model_config.py's shared-cache guards are gated on is_anonymous(), so a plain None would
    let an API-key caller read the operator's cached private snapshots even with the env
    scrubbed. _handle_load has to hand them the sentinel.
    """
    from core.export import worker
    from utils import security as security_pkg
    from utils.models import model_config

    seen = {}

    class _Decision:
        blocked = False

    def _record(key, value, result):
        seen[key] = value
        return result

    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda path, token: _record("lora_base", token, None),
    )
    monkeypatch.setattr(
        security_pkg,
        "security_load_subdirs",
        lambda target, token: _record("load_subdirs", token, ()),
    )
    monkeypatch.setattr(security_pkg, "load_scan_target", lambda target, subdirs: (target, subdirs))
    monkeypatch.setattr(
        security_pkg,
        "evaluate_file_security",
        lambda target, hf_token, load_subdirs: _record("file_security", hf_token, _Decision()),
    )

    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    backend.is_vision = False
    backend.is_peft = False

    worker._handle_load(
        backend,
        {
            "checkpoint_path": "someone/private-model",
            "load_in_4bit": False,
            "hf_token": caller_token,
            "allow_ambient": allow_ambient,
        },
        MagicMock(),
    )

    assert seen["lora_base"] == expected
    assert seen["load_subdirs"] == expected
    assert seen["file_security"] == expected
    assert backend.load_checkpoint.call_args.kwargs["hf_token"] == expected


@pytest.mark.parametrize(
    "endpoint,method_name,payload",
    [
        (
            "/api/export/gguf",
            "export_gguf",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": False,
                "imatrix": True,
                "quantization_method": "q4_k_m",
                "hf_token": None,
            },
        ),
        (
            "/api/export/lora",
            "export_lora_adapter",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": False,
                "gguf": True,
                "hf_token": None,
            },
        ),
    ],
)
def test_local_export_stays_anonymous_for_an_api_key_caller(
    monkeypatch, endpoint, method_name, payload
):
    """A local export is not a token-free export.

    imatrix resolution and LoRA-GGUF base resolution both read the Hub, in a worker some other
    caller may have loaded, so a plain None here would let this caller borrow that credential.
    """
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    getattr(mock_backend, method_name).return_value = (True, "Export successful", "/tmp/export")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)
    monkeypatch.setattr(
        export_routes, "_export_details", lambda *args, **kwargs: {"output_path": "/tmp/export"}
    )

    app = _create_test_app(via_api_key = True)
    response = TestClient(app).post(endpoint, json = payload)

    assert response.status_code == 200
    assert getattr(mock_backend, method_name).call_args.kwargs["hf_token"] is False


def test_local_export_keeps_the_ambient_fallback_for_a_ui_session(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.export_gguf.return_value = (True, "Export successful", "/tmp/export")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)
    monkeypatch.setattr(
        export_routes, "_export_details", lambda *args, **kwargs: {"output_path": "/tmp/export"}
    )

    app = _create_test_app(via_api_key = False)
    response = TestClient(app).post(
        "/api/export/gguf",
        json = {
            "save_directory": "/tmp/export",
            "push_to_hub": False,
            "imatrix": True,
            "quantization_method": "q4_k_m",
            "hf_token": None,
        },
    )

    assert response.status_code == 200
    assert mock_backend.export_gguf.call_args.kwargs["hf_token"] is None


@pytest.mark.parametrize(
    "hf_token,expected", [(False, False), ("hf_caller_own_token", "hf_caller_own_token")]
)
def test_export_backend_forwards_the_sentinel_into_the_gguf_lora_conversion(
    monkeypatch, tmp_path, hf_token, expected
):
    """save.py only falls back to get_token() when token is None, so False has to reach it."""
    from core.export import export as export_backend_module

    seen = {}

    class _FakeModel:
        peft_config: dict = {}

        @staticmethod
        def save_pretrained_gguf(
            save_directory,
            tokenizer,
            save_method = None,
            quantization_method = None,
            token = None,
        ):
            seen["token"] = token

    # Another test in the suite blanks FastLanguageModel, and the runtime gate reads it.
    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_IS_MLX", False)
    monkeypatch.setattr(export_backend_module, "_apply_wsl_sudo_patch", lambda: None)

    backend = export_backend_module.ExportBackend()
    backend.current_model = _FakeModel()
    backend.current_tokenizer = object()
    backend.is_peft = True

    success, message, _path = backend.export_lora_adapter(
        save_directory = str(tmp_path),
        gguf = True,
        hf_token = hf_token,
    )

    assert success, message
    assert seen["token"] == expected


@pytest.mark.parametrize(
    "checkpoint,hf_token,offline,should_refuse",
    [
        ("someone/private-model", False, True, True),
        ("someone/private-model", False, False, False),
        ("someone/private-model", "hf_caller_own_token", True, False),
        ("someone/private-model", None, True, False),
        ("/tmp/local-checkpoint", False, True, False),
        # The regression the model_id form caused: a locally trained adapter names a Hub base,
        # so testing the resolved id refused every offline export of a local LoRA, which is
        # Studio's main flow and always non-ambient over MCP.
        ("<local-lora-adapter>", False, True, False),
    ],
)
def test_offline_anonymous_load_will_not_read_the_operators_cache(
    monkeypatch, tmp_path, checkpoint, hf_token, offline, should_refuse
):
    """Offline, nothing authenticates, so the shared cache would serve whatever the operator
    downloaded. The scrubbed environment cannot help; only a refusal can. It has to key on what
    the caller named, though, not on what that resolves to."""
    if checkpoint == "<local-lora-adapter>":
        (tmp_path / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "unsloth/llama-3.2-1B-Instruct", "peft_type": "LORA"}'
        )
        (tmp_path / "adapter_model.safetensors").touch()
        checkpoint = str(tmp_path)
    from core.export import export as export_backend_module

    reached = {}

    def _fake_detect(model_id, hf_token, local_files_only):
        reached["probe"] = True
        raise RuntimeError("stop")

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: offline)
    monkeypatch.setattr(export_backend_module, "detect_audio_type", _fake_detect)
    # Isolate the offline branch; the online authorization check has its own tests.
    monkeypatch.setattr(
        export_backend_module, "_anonymous_access_allowed", lambda repo, off: (not off, "refused")
    )

    backend = export_backend_module.ExportBackend()
    success, message = backend.load_checkpoint(checkpoint_path = checkpoint, hf_token = hf_token)

    assert success is False
    if should_refuse:
        assert "refused" in message or "not served to API callers" in message
        assert "probe" not in reached, "refused loads must not touch the cache at all"
    else:
        assert reached.get("probe"), f"expected the load to proceed, got: {message}"


@pytest.mark.parametrize(
    "hf_token,expected",
    [
        # None is not anonymous to unsloth: hf_login(None) calls get_token(), which reads the
        # operator's stored ~/.cache/huggingface/token and logs in with it. Only False is.
        (False, False),
        ("hf_caller_own_token", "hf_caller_own_token"),
        (None, None),
    ],
)
def test_weight_loader_never_gets_none_for_an_anonymous_caller(monkeypatch, hf_token, expected):
    from core.export import export as export_backend_module

    seen = {}

    class _Stop(Exception):
        pass

    class _FakeLoader:
        @staticmethod
        def from_pretrained(**kwargs):
            seen["loader"] = kwargs["token"]
            raise _Stop()

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: False)
    monkeypatch.setattr(
        export_backend_module, "_anonymous_access_allowed", lambda repo, off: (True, "")
    )
    monkeypatch.setattr(
        export_backend_module,
        "detect_audio_type",
        lambda model_id, hf_token, local_files_only: seen.setdefault("audio", hf_token) and None,
    )
    monkeypatch.setattr(
        export_backend_module,
        "is_vision_model",
        lambda model_id, hf_token, local_files_only: bool(seen.setdefault("vision", hf_token))
        and False,
    )
    monkeypatch.setattr(export_backend_module, "FastLanguageModel", _FakeLoader)

    backend = export_backend_module.ExportBackend()
    backend.load_checkpoint(checkpoint_path = "someone/private-model", hf_token = hf_token)

    assert seen["audio"] == expected
    assert seen["vision"] == expected
    assert seen["loader"] == expected


def test_hf_login_treats_none_as_fetch_the_operators_stored_token():
    """The upstream contract the sentinel exists for; if this flips, the threading is moot."""
    import inspect

    from unsloth.models._utils import hf_login

    src = inspect.getsource(hf_login)
    assert "if token is None:" in src and "get_token()" in src
    # False must not be routed into the get_token() branch.
    assert hf_login(False) is False


def test_non_ambient_worker_gets_a_private_hf_token_store(monkeypatch, tmp_path, worker_in_process):
    """Scrubbing the environment is half of it.

    get_token() also reads ~/.cache/huggingface/token, and login() writes there, so a shared
    store leaks the operator's credential in and the caller's credential out.
    """
    import os

    worker = worker_in_process
    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.delenv("HF_TOKEN_PATH", raising = False)
    store = str(tmp_path / "store")
    os.makedirs(store)

    seen = {}

    def _fake_activate(path, token):
        seen["token_path"] = os.environ.get("HF_TOKEN_PATH")
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = {
                "checkpoint_path": "/tmp/m",
                "allow_ambient": False,
                "hf_token": None,
                "hf_token_store": store,
            },
        )

    path = seen["token_path"]
    assert path == os.path.join(store, "token")
    assert not os.path.exists(path), "the private store starts empty, so get_token() finds nothing"


@pytest.mark.parametrize(
    "cmd_token,expect_env_token,expect_disable_implicit",
    [
        (False, None, "1"),
        ("hf_caller_own_token", "hf_caller_own_token", "0"),
        (None, "hf_operator_secret_123", None),
    ],
)
def test_export_command_scopes_the_credential_it_can_reach(
    monkeypatch, cmd_token, expect_env_token, expect_disable_implicit
):
    """The GGUF converters build their child env from os.environ, and anything left at
    token=None calls get_token(), so a command inside a worker some other caller loaded has to
    be scoped both ways."""
    import os

    from core.export import worker

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    with worker._credential_scope(cmd_token):
        assert os.environ.get("HF_TOKEN") == expect_env_token
        assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == expect_disable_implicit
        if cmd_token is False:
            import huggingface_hub.constants as hf_constants
            from huggingface_hub import get_token

            assert "unsloth-export-hf-cmd-" in hf_constants.HF_TOKEN_PATH
            assert get_token() is None, "no stored login is reachable for an anonymous command"

    # Restored, so the next command decides for itself.
    assert os.environ.get("HF_TOKEN") == "hf_operator_secret_123"
    assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") is None


def test_orchestrator_owns_the_token_store_so_a_kill_cannot_orphan_it(monkeypatch, tmp_path):
    """atexit does not run on terminate() or kill(), and the cancel path uses both."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    store = o._new_token_store()
    assert os.path.isdir(store)

    # Whatever the loader persisted there goes with it.
    with open(os.path.join(store, "token"), "w") as fh:
        fh.write("hf_caller_own_token")

    o._discard_token_store()
    assert not os.path.exists(store)

    # A second load replaces the first store rather than accumulating them.
    first = o._new_token_store()
    second = o._new_token_store()
    assert first != second
    assert not os.path.exists(first)
    o._discard_token_store()


def test_token_store_survives_the_shutdown_of_the_worker_it_replaces(monkeypatch, tmp_path):
    """Allocating before the shutdown handed the new worker a directory the shutdown deleted."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    spawned: dict = {}

    class _Alive:
        def is_alive(self):
            return True

    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_shutdown_subprocess", lambda *a, **kw: o._discard_token_store() or True
    )
    monkeypatch.setattr(
        o, "_spawn_subprocess", lambda cfg: spawned.update(cfg) or (_ for _ in ()).throw(_Boom())
    )

    class _Boom(Exception):
        pass

    try:
        o.load_checkpoint(checkpoint_path = "someone/model", allow_ambient = False)
    except Exception:
        pass

    store = spawned.get("hf_token_store")
    assert store, "a non-ambient load must get a private store"
    assert os.path.isdir(store), "the store the worker was given must still exist"
    o._discard_token_store()


def test_token_store_is_removed_when_the_worker_is_already_dead(monkeypatch):
    """The early return skipped the cleanup, so a crashed worker's store outlived it."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    store = o._new_token_store()
    with open(os.path.join(store, "token"), "w") as fh:
        fh.write("hf_caller_own_token")

    o._proc = None
    assert o._shutdown_subprocess() is True
    assert not os.path.exists(store)


def test_cancelling_an_export_discards_the_token_store():
    """cancel_export kills the worker directly and _run_export swallows the error, so neither
    _shutdown_subprocess cleanup path runs."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    store = o._new_token_store()
    with open(os.path.join(store, "token"), "w") as fh:
        fh.write("hf_caller_own_token")

    class _Proc:
        pid = 1234

        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self._alive = False

        def join(self, timeout = None):
            pass

    o._proc = _Proc()
    assert o.cancel_export() is True
    assert not os.path.exists(store)


@pytest.mark.parametrize(
    "gated,private,raises,allowed",
    [
        (False, False, False, True),
        ("manual", False, False, False),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_anonymous_access_check_reads_the_gated_flag(monkeypatch, gated, private, raises, allowed):
    """A gated repo's metadata is public and only its files are held back, so model_info
    succeeding is not access."""
    from core.export import export as export_backend_module

    class _Info:
        pass

    info = _Info()
    info.gated = gated
    info.private = private

    class _Api:
        def model_info(
            self,
            repo_id,
            token = None,
        ):
            assert token is False, "the check must ask anonymously"
            if raises:
                raise RuntimeError("401")
            return info

    monkeypatch.setattr(export_backend_module, "HfApi", _Api)
    ok, _why = export_backend_module._anonymous_access_allowed("owner/repo", offline = False)
    assert ok is allowed


def test_anonymous_access_check_refuses_when_it_cannot_ask(monkeypatch):
    from core.export import export as export_backend_module

    def _boom(*a, **kw):
        pytest.fail("offline must not reach the Hub")

    monkeypatch.setattr(export_backend_module, "HfApi", _boom)
    ok, why = export_backend_module._anonymous_access_allowed("owner/repo", offline = True)
    assert ok is False
    assert "Hub is unreachable" in why


def test_a_remote_adapters_base_is_authorized_too(monkeypatch):
    """A public adapter must not stand in front of a cached private base."""
    from core.export import export as export_backend_module
    from utils.models import model_config

    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda path, token: "owner/private-base",
    )
    assert export_backend_module._remote_load_targets("owner/public-adapter") == [
        "owner/public-adapter",
        "owner/private-base",
    ]

    checked = []

    def _check(repo, offline):
        checked.append(repo)
        return (repo != "owner/private-base", "refused")

    monkeypatch.setattr(export_backend_module, "_anonymous_access_allowed", _check)
    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: False)
    monkeypatch.setattr(
        export_backend_module,
        "detect_audio_type",
        lambda *a, **kw: pytest.fail("must refuse before touching the cache"),
    )

    backend = export_backend_module.ExportBackend()
    ok, message = backend.load_checkpoint(checkpoint_path = "owner/public-adapter", hf_token = False)

    assert ok is False
    assert message == "refused"
    assert checked == ["owner/public-adapter", "owner/private-base"]


def test_a_local_checkpoints_base_is_not_authorized(monkeypatch, tmp_path):
    """The exemption that keeps offline local LoRA exports working stays."""
    from core.export import export as export_backend_module

    monkeypatch.setattr(
        export_backend_module,
        "_anonymous_access_allowed",
        lambda repo, offline: pytest.fail("a local checkpoint is not a remote target"),
    )
    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: False)

    reached = {}

    def _probe(model_id, hf_token, local_files_only):
        reached["probe"] = model_id
        raise RuntimeError("stop")

    monkeypatch.setattr(export_backend_module, "detect_audio_type", _probe)
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "unsloth/llama-3.2-1B-Instruct", "peft_type": "LORA"}'
    )
    (tmp_path / "adapter_model.safetensors").touch()

    backend = export_backend_module.ExportBackend()
    backend.load_checkpoint(checkpoint_path = str(tmp_path), hf_token = False)
    assert reached["probe"] == "unsloth/llama-3.2-1B-Instruct"


def test_a_worker_surviving_cancellation_keeps_its_token_store():
    """Pulling HF_TOKEN_PATH out from under a live worker breaks it and orphans a new one."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    store = o._new_token_store()

    class _Immortal:
        pid = 99

        def is_alive(self):
            return True

        def terminate(self):
            pass

        def kill(self):
            pass

        def join(self, timeout = None):
            pass

    o._proc = _Immortal()
    try:
        assert o.cancel_export() is True
        assert os.path.isdir(store), "a survivor still points at this store"
    finally:
        # Do not leave a live-looking handle for the orchestrator's atexit to shut down.
        o._proc = None
        o._discard_token_store()


def test_an_ambient_command_in_a_non_ambient_worker_gets_its_own_store(monkeypatch):
    """The worker store may hold what the load caller's token persisted; a later UI export
    must not read it back out through get_token()."""
    import os

    from core.export import worker

    monkeypatch.setattr(worker, "_WORKER_IS_NON_AMBIENT", True)
    monkeypatch.setenv("HF_TOKEN_PATH", "/nonexistent/worker-store/token")

    with worker._credential_scope(None):
        scoped = os.environ["HF_TOKEN_PATH"]
        assert "unsloth-export-hf-cmd-" in scoped
        assert not os.path.exists(scoped)

    assert os.environ["HF_TOKEN_PATH"] == "/nonexistent/worker-store/token"


def test_an_ambient_command_in_an_ambient_worker_keeps_the_operator_store(monkeypatch):
    import os

    from core.export import worker

    monkeypatch.setattr(worker, "_WORKER_IS_NON_AMBIENT", False)
    monkeypatch.setenv("HF_TOKEN_PATH", "/home/op/.cache/huggingface/token")

    with worker._credential_scope(None):
        assert os.environ["HF_TOKEN_PATH"] == "/home/op/.cache/huggingface/token"


def test_cancellation_does_not_delete_a_replacement_workers_store():
    """cancel_export holds no lock, so a reload can install a new store while it is joining."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    cancelled_store = o._new_token_store()

    class _Proc:
        pid = 7

        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self._alive = False

        def join(self, timeout = None):
            # The reload lands while this thread is still joining.
            o._token_store = replacement

        def kill(self):
            pass

    replacement = os.path.join(os.path.dirname(cancelled_store), "replacement-store")
    os.makedirs(replacement, exist_ok = True)

    o._proc = _Proc()
    try:
        assert o.cancel_export() is True
        assert os.path.isdir(replacement), "the live worker's store must survive"
        assert not os.path.exists(cancelled_store), "the cancelled worker's store must go"
        assert o._token_store == replacement
    finally:
        o._proc = None
        import shutil

        shutil.rmtree(replacement, ignore_errors = True)
        o._token_store = None


def test_a_cache_snapshot_path_is_authorized_as_its_repository(monkeypatch, tmp_path):
    """An absolute snapshot path is local only in spelling, and /hub/cached-models hands API
    callers exactly these ids."""
    from core.export import export as export_backend_module

    snap = tmp_path / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots" / "abc123"
    snap.mkdir(parents = True)

    assert (
        export_backend_module._cache_snapshot_repo(str(snap)) == "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert export_backend_module._needs_anonymous_authorization(str(snap)) is True
    # A checkpoint the user trained is still exempt.
    plain = tmp_path / "outputs" / "my-run"
    plain.mkdir(parents = True)
    assert export_backend_module._needs_anonymous_authorization(str(plain)) is False
    assert export_backend_module._remote_load_targets(str(snap))[0] == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    checked = []
    monkeypatch.setattr(
        export_backend_module,
        "_anonymous_access_allowed",
        lambda repo, offline: (checked.append(repo), (False, "refused"))[1],
    )
    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: False)
    monkeypatch.setattr(
        export_backend_module,
        "detect_audio_type",
        lambda *a, **kw: pytest.fail("must refuse before touching the snapshot"),
    )

    backend = export_backend_module.ExportBackend()
    ok, _msg = backend.load_checkpoint(checkpoint_path = str(snap), hf_token = False)
    assert ok is False
    assert checked == ["meta-llama/Llama-3.1-8B-Instruct"]


def test_a_worker_that_dies_mid_export_has_its_token_store_reaped():
    """_run_export catches the crash and returns, so no shutdown ever runs."""
    import os

    from core.export import orchestrator as orch

    o = orch.ExportOrchestrator()
    store = o._new_token_store()
    with open(os.path.join(store, "token"), "w") as fh:
        fh.write("hf_caller_own_token")

    class _Dead:
        def is_alive(self):
            return False

    o._proc = _Dead()
    o._read_resp = lambda timeout = None: None

    # Through the real path: _wait_response is where the crash is noticed.
    with pytest.raises(RuntimeError, match = "crashed during wait"):
        o._wait_response("export_gguf_done", timeout = 1.0)

    assert o._proc is None
    assert not os.path.exists(store)
