# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_hf_precheck(monkeypatch):
    hf_precheck_path = _BACKEND_DIR / "core" / "export" / "hf_precheck.py"
    spec = importlib.util.spec_from_file_location("test_hf_precheck", hf_precheck_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    hub_paths = types.ModuleType("hub.utils.paths")
    hub_paths.is_valid_repo_id = lambda repo_id: bool(
        repo_id and "/" in repo_id and ".." not in repo_id
    )
    monkeypatch.setitem(sys.modules, "hub.utils.paths", hub_paths)

    class _FakeResponse:
        def __init__(self, status_code = 200):
            self.status_code = status_code

    class _FakeHTTPError(Exception):
        def __init__(self, status_code = 401):
            super().__init__(f"HTTP {status_code}")
            self.response = _FakeResponse(status_code)

    class _RepositoryNotFoundError(Exception):
        pass

    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_utils.HfHubHTTPError = _FakeHTTPError
    hf_utils.RepositoryNotFoundError = _RepositoryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hf_utils)

    class _FakeHfApi:
        create_repo_calls = 0

        def __init__(self, token = None):
            self.token = token

        def whoami(self, token = None):
            if self.token == "bad":
                raise _FakeHTTPError(401)
            return {"name": "alice", "orgs": [{"name": "my-org"}]}

        def repo_info(
            self,
            repo_id,
            repo_type = "model",
            token = None,
        ):
            if repo_id == "alice/existing":
                return {"id": repo_id}
            raise _RepositoryNotFoundError()

        def create_repo(self, **kwargs):
            _FakeHfApi.create_repo_calls += 1
            if kwargs.get("repo_id", "").startswith("blocked/"):
                raise _FakeHTTPError(403)
            return kwargs.get("repo_id")

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.HfApi = _FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    loggers = types.ModuleType("loggers")
    loggers.get_logger = lambda *args, **kwargs: types.SimpleNamespace(
        warning = lambda *a, **k: None,
        warning_once = lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "loggers", loggers)

    monkeypatch.setitem(sys.modules, "test_hf_precheck", module)
    spec.loader.exec_module(module)
    return module


def test_precheck_rejects_invalid_repo_id(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_upload("bad repo id", hf_token = "ok")
    assert result.ok is False
    assert "Invalid repository ID" in result.message


def test_precheck_rejects_wrong_namespace(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_upload("other-user/model", hf_token = "ok")
    assert result.ok is False
    assert "write access" in result.message


def test_precheck_accepts_new_repo_in_user_namespace(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_upload("alice/new-model", hf_token = "ok", private = True)
    assert result.ok is True
    assert "private repository" in result.message
    assert result.details["repo_exists"] is False


def test_precheck_accepts_existing_repo(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_upload("alice/existing", hf_token = "ok")
    assert result.ok is True
    assert "existing repository" in result.message
    assert result.details["repo_exists"] is True


def test_precheck_rejects_bad_token(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_upload("alice/model", hf_token = "bad")
    assert result.ok is False
    assert "Invalid or expired" in result.message


def test_precheck_does_not_create_repo(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    huggingface_hub = sys.modules["huggingface_hub"]
    huggingface_hub.HfApi.create_repo_calls = 0
    result = mod.precheck_hub_upload("alice/new-model", hf_token = "ok", private = True)
    assert result.ok is True
    assert huggingface_hub.HfApi.create_repo_calls == 0


def test_precheck_credentials_only_returns_username(monkeypatch):
    mod = _load_hf_precheck(monkeypatch)
    result = mod.precheck_hub_credentials(hf_token = "ok")
    assert result.ok is True
    assert result.details["username"] == "alice"
    assert "Logged in to Hugging Face" in result.message
