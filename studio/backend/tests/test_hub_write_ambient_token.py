# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend's own HF_TOKEN must not be lent to an sk-unsloth API key (issue #10126).

Two things have to hold. A write path refuses rather than publishing or pushing as the
operator, and a load started by such a caller does not *run* as the operator: that needs the
anonymous sentinel rather than None, because None is what unsloth reads as "go and find a
credential" (``if token is None: get_token()``).
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


def _app(via_api_key: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(data_recipe_jobs_routes.router, prefix = "/api/data-recipe")
    app.include_router(export_routes.router, prefix = "/api")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[get_current_credential] = lambda: ("alice", "cred-1")
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return app


def _publish_manager(monkeypatch, seen):
    mgr = MagicMock()
    mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/artifacts",
    }
    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mgr)
    monkeypatch.setattr(
        data_recipe_jobs_routes,
        "publish_recipe_dataset",
        lambda artifact_path, repo_id, description, hf_token, private: (
            seen.update(token = hf_token) or f"https://huggingface.co/datasets/{repo_id}"
        ),
    )


# ---------------------------------------------------------------- recipe dataset publish


def test_publish_refuses_an_api_key_without_a_token(monkeypatch):
    """The hole this PR exists for: publish_recipe_dataset passes the token straight to
    HuggingFaceHubClient and card.push_to_hub, so None publishes as the host's login."""
    _publish_manager(monkeypatch, {})
    response = TestClient(_app(via_api_key = True)).post(
        "/api/data-recipe/jobs/job-1/publish",
        json = {"repo_id": "org/dataset", "description": "d", "hf_token": None},
    )
    assert response.status_code == 400
    assert "required to publish datasets" in response.json()["detail"]


@pytest.mark.parametrize(
    "via_api_key,token,expected",
    [(True, "hf_caller", "hf_caller"), (False, None, None)],
)
def test_publish_allows_a_token_bearing_key_and_a_ui_session(
    monkeypatch, via_api_key, token, expected
):
    seen: dict = {}
    _publish_manager(monkeypatch, seen)
    response = TestClient(_app(via_api_key)).post(
        "/api/data-recipe/jobs/job-1/publish",
        json = {"repo_id": "org/dataset", "description": "d", "hf_token": token},
    )
    assert response.status_code == 200
    assert seen["token"] == expected


# ------------------------------------------------------------------------ export routes

_EXPORTS = [
    ("/api/export/merged", "export_merged_model", {}),
    ("/api/export/base", "export_base_model", {}),
    ("/api/export/gguf", "export_gguf", {"quantization_method": "q4_k_m"}),
    ("/api/export/lora", "export_lora_adapter", {}),
]


def _backend(monkeypatch, method):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    backend = MagicMock()
    getattr(backend, method).return_value = (True, "ok", "/tmp/export")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    monkeypatch.setattr(
        export_routes, "_export_details", lambda *a, **kw: {"output_path": "/tmp/export"}
    )
    return backend


@pytest.mark.parametrize("endpoint,method,extra", _EXPORTS)
def test_push_refuses_an_api_key_without_a_token(monkeypatch, endpoint, method, extra):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    response = TestClient(_app(via_api_key = True)).post(
        endpoint,
        json = {"save_directory": "/tmp/x", "push_to_hub": True, "repo_id": "o/m", **extra},
    )
    assert response.status_code == 400
    assert "required to push to Hub" in response.json()["detail"]


@pytest.mark.parametrize("endpoint,method,extra", _EXPORTS)
def test_push_with_a_token_reaches_the_backend_unchanged(monkeypatch, endpoint, method, extra):
    backend = _backend(monkeypatch, method)
    response = TestClient(_app(via_api_key = True)).post(
        endpoint,
        json = {
            "save_directory": "/tmp/x",
            "push_to_hub": True,
            "repo_id": "o/m",
            "hf_token": "hf_caller",
            **extra,
        },
    )
    assert response.status_code == 200
    assert getattr(backend, method).call_args.kwargs["hf_token"] == "hf_caller"


@pytest.mark.parametrize("endpoint,method,extra", _EXPORTS)
@pytest.mark.parametrize("via_api_key,expected", [(True, False), (False, None)])
def test_a_local_export_carries_the_callers_identity(
    monkeypatch, endpoint, method, extra, via_api_key, expected
):
    """No push, so nothing is refused, but the credential still has to be spelled: a local
    GGUF export resolves its base and imatrix from the Hub."""
    backend = _backend(monkeypatch, method)
    response = TestClient(_app(via_api_key)).post(
        endpoint, json = {"save_directory": "/tmp/x", "push_to_hub": False, **extra}
    )
    assert response.status_code == 200
    assert getattr(backend, method).call_args.kwargs["hf_token"] is expected


@pytest.mark.parametrize(
    "via_api_key,token,expected_token,expected_flag",
    [
        (True, None, False, False),
        (True, "hf_caller", "hf_caller", False),
        (False, None, None, True),
    ],
)
def test_load_checkpoint_forwards_the_policy(
    monkeypatch, via_api_key, token, expected_token, expected_flag
):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)

    response = TestClient(_app(via_api_key)).post(
        "/api/load-checkpoint", json = {"checkpoint_path": "o/m", "hf_token": token}
    )
    assert response.status_code == 200
    kwargs = backend.load_checkpoint.call_args.kwargs
    assert kwargs["hf_token"] is expected_token or kwargs["hf_token"] == expected_token
    assert kwargs["allow_ambient"] is expected_flag


# ------------------------------------------------------------------------ export worker


@pytest.fixture
def worker_in_process(monkeypatch):
    """Let run_export_process run here without it taking over the pytest process:
    _setup_log_capture dup2s pipes over fds 1 and 2 with no teardown, and
    UNSLOTH_OFFLINE_PROBE keeps a live Hub request out of a unit test."""
    from core.export import worker

    monkeypatch.setattr(worker, "_setup_log_capture", lambda resp_queue: None)
    monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "0")
    return worker


@pytest.mark.parametrize(
    "allow_ambient,caller_token,env_token,disable_implicit,passed",
    [
        (False, None, None, "1", None),
        (False, "hf_caller", "hf_caller", "0", "hf_caller"),
        (True, None, "hf_operator_secret", None, None),
    ],
)
def test_the_worker_environment_matches_the_callers_policy(
    monkeypatch, worker_in_process, allow_ambient, caller_token, env_token, disable_implicit, passed
):
    import os

    worker = worker_in_process
    for key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.setenv(key, "hf_operator_secret")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen: dict = {}

    def _fake_activate(path, token):
        seen["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen["HF_HUB_TOKEN"] = os.environ.get("HF_HUB_TOKEN")
        seen["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen["passed"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)
    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = {
                "checkpoint_path": "/tmp/model",
                "allow_ambient": allow_ambient,
                "hf_token": caller_token,
            },
        )

    assert seen["HF_TOKEN"] == env_token
    assert seen["DISABLE_IMPLICIT"] == disable_implicit
    if not allow_ambient:
        # The operator's aliases go even when the caller supplied its own credential.
        assert seen["HF_HUB_TOKEN"] is None
    assert seen["passed"] == passed


def test_an_old_orchestrator_config_keeps_the_previous_behaviour(monkeypatch, worker_in_process):
    """A config from before this change has neither key; it must read as ambient."""
    import os

    worker = worker_in_process
    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen: dict = {}

    def _fake_activate(path, token):
        seen["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)
    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = {"checkpoint_path": "/tmp/m", "hf_token": None},
        )

    assert seen["HF_TOKEN"] == "hf_operator_secret"
    assert seen["DISABLE_IMPLICIT"] is None


@pytest.mark.parametrize(
    "allow_ambient,caller_token,expected",
    [(False, None, False), (False, "hf_caller", "hf_caller"), (True, None, None)],
)
def test_the_load_preflight_runs_under_the_callers_credential(
    monkeypatch, allow_ambient, caller_token, expected
):
    """model_config's shared-cache guards read is_anonymous(), so a plain None walks past
    them; every preflight helper but tier detection gets the same canonical value."""
    from core.export import worker
    from utils import security as security_pkg
    from utils import transformers_version
    from utils.models import model_config

    seen: dict = {}

    def _record(key, value, result):
        seen[key] = value
        return result

    class _Decision:
        blocked = False

    monkeypatch.setattr(
        transformers_version,
        "latest_tier_active_for",
        lambda name, token: _record("tier", token, False),
    )
    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda path, token: _record("lora_base", token, None),
    )
    monkeypatch.setattr(
        security_pkg, "security_load_subdirs", lambda t, token: _record("subdirs", token, ())
    )
    monkeypatch.setattr(security_pkg, "load_scan_target", lambda t, subdirs: (t, subdirs))
    monkeypatch.setattr(
        security_pkg,
        "evaluate_file_security",
        lambda target, hf_token, load_subdirs: _record("file_security", hf_token, _Decision()),
    )

    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    backend.is_vision = backend.is_peft = False

    worker._handle_load(
        backend,
        {
            "checkpoint_path": "owner/model",
            "load_in_4bit": True,
            "hf_token": caller_token,
            "allow_ambient": allow_ambient,
        },
        MagicMock(),
    )

    assert seen["lora_base"] == expected
    assert seen["subdirs"] == expected
    assert seen["file_security"] == expected
    assert backend.load_checkpoint.call_args.kwargs["hf_token"] == expected
    # Tier detection reads config.json off the hub cache, which the sentinel is refused.
    assert seen["tier"] == (expected or None)


# ------------------------------------------------------------------------ export backend


@pytest.mark.parametrize(
    "hf_token,expected", [(False, False), ("hf_caller", "hf_caller"), (None, None)]
)
def test_the_weight_loader_never_receives_none_for_an_anonymous_caller(
    monkeypatch, hf_token, expected
):
    """The whole point of the sentinel: hf_login(None) calls get_token(), which reads the
    operator's stored login from disk, and no environment scrub touches that."""
    from core.export import export as export_backend_module

    seen: dict = {}

    class _Stop(Exception):
        pass

    class _Loader:
        @staticmethod
        def from_pretrained(**kwargs):
            seen["loader"] = kwargs["token"]
            raise _Stop()

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: False)
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
    monkeypatch.setattr(export_backend_module, "FastLanguageModel", _Loader)

    export_backend_module.ExportBackend().load_checkpoint(
        checkpoint_path = "owner/model", hf_token = hf_token
    )

    # The probes take the plain token; only the loaders get the sentinel.
    assert seen["audio"] == (expected or None)
    assert seen["vision"] == (expected or None)
    assert seen["loader"] == expected


def test_hf_login_reads_none_as_fetch_the_operators_stored_token():
    """The upstream contract the sentinel exists for. If this flips, the threading is moot."""
    import inspect

    # Not importorskip: that skips only on ModuleNotFoundError, and unsloth raises a plain
    # ImportError ("Unsloth: torch not found") on a torch-less install, which is supported.
    try:
        from unsloth.models._utils import hf_login
    except ImportError as exc:
        pytest.skip(f"unsloth needs torch, which this install does not ship: {exc}")

    src = inspect.getsource(hf_login)
    assert "if token is None:" in src and "get_token()" in src
    assert hf_login(False) is False


@pytest.mark.parametrize("hf_token,expected", [(False, False), ("hf_caller", "hf_caller")])
def test_a_local_gguf_lora_conversion_carries_the_sentinel(
    monkeypatch, tmp_path, hf_token, expected
):
    """save.py substitutes get_token() only when the token is None, so False has to reach it."""
    from core.export import export as export_backend_module

    seen: dict = {}

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

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_IS_MLX", False)
    monkeypatch.setattr(export_backend_module, "_apply_wsl_sudo_patch", lambda: None)

    backend = export_backend_module.ExportBackend()
    backend.current_model = _FakeModel()
    backend.current_tokenizer = object()
    backend.is_peft = True

    success, message, _path = backend.export_lora_adapter(
        save_directory = str(tmp_path), gguf = True, hf_token = hf_token
    )
    assert success, message
    assert seen["token"] == expected


@pytest.mark.parametrize("hf_token,expected", [(False, False), ("hf_caller", "hf_caller")])
def test_a_local_merged_save_carries_the_sentinel(monkeypatch, tmp_path, hf_token, expected):
    """Not a push, but the merge resolves its base repo, and save.py turns a None into
    get_token(), which is the operator's stored login."""
    from core.export import export as export_backend_module

    seen: dict = {}

    class _FakeModel:
        @staticmethod
        def save_pretrained_merged(
            save_directory,
            tokenizer,
            save_method = None,
            token = None,
        ):
            seen["token"] = token

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_IS_MLX", False)

    backend = export_backend_module.ExportBackend()
    backend.current_model = _FakeModel()
    backend.current_tokenizer = object()

    backend.export_merged_model(save_directory = str(tmp_path), hf_token = hf_token)
    assert seen["token"] == expected


def test_offline_type_detection_is_not_degraded_by_the_sentinel(monkeypatch):
    """model_config's cache guards refuse an anonymous cached read, which offline turns a
    cached vision model into a text one. Detection keeps the plain token for that reason."""
    from core.export import export as export_backend_module

    seen: dict = {}

    class _Stop(Exception):
        pass

    class _Loader:
        @staticmethod
        def from_pretrained(**kwargs):
            seen["loader"] = kwargs["token"]
            raise _Stop()

    monkeypatch.setattr(export_backend_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_backend_module, "_hf_offline", lambda: True)
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
    monkeypatch.setattr(export_backend_module, "FastLanguageModel", _Loader)

    export_backend_module.ExportBackend().load_checkpoint(
        checkpoint_path = "owner/vlm", hf_token = False
    )

    assert seen["audio"] is None, "an anonymous probe must not be forced down the guard"
    assert seen["vision"] is None
    assert seen["loader"] is False, "the loader still gets the sentinel"


def test_offline_tier_detection_is_not_degraded_by_the_sentinel(monkeypatch, tmp_path):
    """_load_config_json refuses a hub-cache read for the sentinel, so offline a cached model
    whose tier is only in its config.json drops to the default sidecar."""
    import json
    from types import SimpleNamespace

    from utils import transformers_version as tv

    repo = "acme/private-finetune"
    repo_dir = tmp_path / ("models--" + repo.replace("/", "--"))
    (repo_dir / "snapshots" / "abc123").mkdir(parents = True)
    (repo_dir / "snapshots" / "abc123" / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]})
    )
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("abc123")

    monkeypatch.setattr(tv, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_TIER_PROBE", "1")

    def _tier(hf_token):
        for cache in (
            tv._config_json_cache,
            tv._config_needs_510_cache,
            tv._config_needs_550_cache,
            tv._config_needs_530_cache,
            tv._tokenizer_class_cache,
        ):
            cache.clear()
        return tv.get_transformers_tier(repo, hf_token)

    assert _tier(False) == "default", "the sentinel is what breaks the cached tier read"
    assert _tier(None) == "530", "the plain token reads it, as it did before this change"
