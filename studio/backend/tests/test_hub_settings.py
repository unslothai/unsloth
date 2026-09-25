# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The saved Hugging Face endpoint drives HF_ENDPOINT / HF_DATASETS_SERVER."""

from __future__ import annotations

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import json
import os
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import storage.studio_db as studio_db
import utils.hub_settings as hub_settings

MIRROR = "https://hf-mirror.com"
OPERATOR_DS = "https://ds.example.com"


@pytest.fixture
def store(monkeypatch):
    import huggingface_hub.constants as constants

    values: dict = {}
    monkeypatch.setattr(constants, "HF_URL_HOSTS", constants.HF_URL_HOSTS)
    monkeypatch.setattr(
        studio_db, "get_app_settings", lambda keys: {k: values[k] for k in keys if k in values}
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates, **_: values.update(updates) or values
    )
    monkeypatch.setattr(hub_settings, "_operator_env", None)
    monkeypatch.setenv("HF_ENDPOINT", "hf-mirror.com")
    monkeypatch.setenv("HF_DATASETS_SERVER", OPERATOR_DS)
    monkeypatch.delenv(hub_settings.SOURCE_ENV, raising = False)
    yield values
    monkeypatch.undo()
    hub_settings._refresh_imported_hub_libraries()


def test_operator_environment_stands_until_the_owner_saves(store):
    hub_settings.apply_hub_settings()
    assert os.environ["HF_ENDPOINT"] == MIRROR
    assert os.environ["HF_DATASETS_SERVER"] == OPERATOR_DS
    assert hub_settings.get_hub_settings().hf_endpoint == MIRROR
    assert store == {}


def test_saved_endpoint_reaches_env_and_imported_libraries(store):
    import datasets.config as datasets_config
    import huggingface_hub.constants as constants
    import huggingface_hub.hf_api as hf_api
    from huggingface_hub import HfFileSystem

    before = HfFileSystem()
    saved = hub_settings.set_hub_settings("http://127.0.0.1:9000/", True)
    assert saved.hf_endpoint == "http://127.0.0.1:9000"
    assert os.environ["HF_ENDPOINT"] == "http://127.0.0.1:9000"
    assert os.environ["HF_DATASETS_SERVER"] == "http://127.0.0.1:9000"
    assert constants.ENDPOINT == "http://127.0.0.1:9000"
    assert constants.HUGGINGFACE_CO_URL_TEMPLATE.startswith("http://127.0.0.1:9000/")
    assert hf_api.api.endpoint == "http://127.0.0.1:9000"
    assert datasets_config.HF_ENDPOINT == "http://127.0.0.1:9000"
    assert datasets_config.HUB_DATASETS_URL.startswith("http://127.0.0.1:9000/datasets/")
    assert "127.0.0.1" in constants.HF_URL_HOSTS
    assert before not in HfFileSystem._cache.values()

    hub_settings.set_hub_settings("", True)
    assert "HF_ENDPOINT" not in os.environ
    assert os.environ["HF_DATASETS_SERVER"] == OPERATOR_DS
    assert constants.ENDPOINT == "https://huggingface.co"
    assert hf_api.api.endpoint == "https://huggingface.co"


def test_datasets_server_follows_only_when_asked(store):
    hub_settings.set_hub_settings(MIRROR, False)
    assert os.environ["HF_ENDPOINT"] == MIRROR
    assert os.environ["HF_DATASETS_SERVER"] == OPERATOR_DS
    hub_settings.set_hub_settings(MIRROR, True)
    assert os.environ["HF_DATASETS_SERVER"] == MIRROR


def test_the_startup_read_leaves_studio_db_as_it_found_it(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "studio.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE app_settings (key TEXT PRIMARY KEY, value_json TEXT)")
        conn.execute(
            "INSERT INTO app_settings VALUES (?, ?)", (hub_settings.SOURCE_KEY, '"modelscope"')
        )
    conn.close()
    before = db.read_bytes()
    monkeypatch.delitem(sys.modules, "storage.studio_db")
    monkeypatch.setattr("utils.paths.storage_roots.studio_db_path", lambda: db)
    assert hub_settings.get_hub_settings().source == hub_settings.MODELSCOPE
    assert db.read_bytes() == before
    db.unlink()
    assert hub_settings.get_hub_settings().source == hub_settings.HUGGINGFACE
    assert not db.exists()


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def test_route_saves_and_reports(client, store):
    body = client.put(
        "/hub",
        json = {"hf_endpoint": "HTTPS://hf-mirror.com/", "datasets_server_follows_endpoint": True},
    ).json()
    assert body == {
        "hf_endpoint": MIRROR,
        "datasets_server_follows_endpoint": True,
        "source": "huggingface",
        "active_source": "huggingface",
    }
    assert client.get("/hub").json() == body
    assert os.environ["HF_DATASETS_SERVER"] == MIRROR


def test_only_the_owner_reads_the_endpoint(client, monkeypatch):
    from fastapi import HTTPException

    async def refuse():
        raise HTTPException(status_code = 403)

    monkeypatch.setattr(settings.policy, "require_owner", refuse)
    assert client.get("/hub").status_code == 403


@pytest.mark.parametrize(
    "raw, canonical",
    [("", ""), (" hf-mirror.com/ ", MIRROR), ("http://localhost:8080", "http://localhost:8080")],
)
def test_validate_hub_endpoint_canonicalises(raw, canonical):
    assert hub_settings.validate_hub_endpoint(raw) == canonical


@pytest.mark.parametrize(
    "bad",
    ["http://example.com", "ftp://example.com", "https://u:p@example.com", "https://x.com?a=1"],
)
def test_route_rejects_unusable_endpoints(client, store, bad):
    response = client.put(
        "/hub", json = {"hf_endpoint": bad, "datasets_server_follows_endpoint": False}
    )
    assert response.status_code == 400
    assert store == {}


def test_access_verdicts_do_not_cross_endpoints(store, tmp_path, monkeypatch):
    from hub.services.models import account_access
    from hub.utils import hf_tokens

    monkeypatch.setattr(account_access, "_public_repos", {})
    monkeypatch.setattr(account_access, "_public_verdicts_path", lambda: tmp_path / "proofs.json")
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_repo_access_cache", {})
    hub_settings.set_hub_settings("https://a.example.com", False)

    asked = []

    def yes_then_switch(*_, endpoint):
        asked.append(endpoint)
        hub_settings.set_hub_settings("https://b.example.com", False)
        return True

    monkeypatch.setattr(account_access, "_hub_public_answer", yes_then_switch)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", yes_then_switch)
    assert account_access.repo_is_public("org/repo") is True
    hub_settings.set_hub_settings("https://a.example.com", False)
    assert hf_tokens._explicit_token_reaches_repo("org/repo", None, "model") is True
    assert asked == ["https://a.example.com"] * 2

    monkeypatch.setattr(account_access, "_hub_public_answer", lambda *_, **__: None)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_, **__: None)
    assert account_access.repo_is_public("org/repo") is False
    assert hf_tokens._explicit_token_reaches_repo("org/repo", None, "model") is None
    hub_settings.set_hub_settings("https://a.example.com", False)
    monkeypatch.setattr(account_access, "_public_repos", {})
    assert account_access.repo_is_public("org/repo") is True

    proofs = tmp_path / "proofs.json"
    proofs.write_text(json.dumps({"model:org/old": time.time()}))
    assert account_access.repo_is_public("org/old") is False
    account_access.adopt_unnamed_public_proofs(hub_settings.operator_hf_endpoint())
    assert list(json.loads(proofs.read_text())) == [f"{MIRROR}|model:org/old"]
    assert account_access.repo_is_public("org/old") is False
    hub_settings.set_hub_settings(MIRROR, False)
    assert account_access.repo_is_public("org/old") is True


def test_probes_ask_the_endpoint_they_are_given(store, monkeypatch):
    from hub.services.models import account_access
    from hub.utils import hf_tokens

    asked = []

    class _Api:
        def __init__(self, endpoint = None):
            asked.append(endpoint)

        def repo_info(self, *_a, **_k):
            raise OSError

    class _Session:
        def get(self, url, **_k):
            asked.append(url)
            raise OSError

    monkeypatch.setattr(account_access, "HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    account_access._hub_public_answer("org/repo", "model", endpoint = "https://a.example.com")
    hf_tokens._probe_repo_access("org/repo", None, "model", endpoint = "https://a.example.com")
    assert asked == [
        "https://a.example.com",
        "https://a.example.com/api/models/org/repo/auth-check",
    ]


def test_hub_decisions_are_remembered_per_endpoint(store, monkeypatch):
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    import utils.hf_token_validation as validation
    import asyncio

    import utils.utils as utils_module
    from core.inference import openai_auto_download as auto_download
    from utils.security import trusted_org

    checks, apis = [], MagicMock()
    apis.return_value.model_info.return_value = SimpleNamespace(id = "unsloth/x", author = "unsloth")
    monkeypatch.setattr(validation, "_cache", {})
    monkeypatch.setattr(
        validation,
        "_check_remote",
        lambda token, endpoint: checks.append(endpoint)
        or validation.TokenValidationResult(status = "invalid"),
    )
    monkeypatch.setattr(trusted_org, "_verdict_cache", {})
    monkeypatch.setattr("huggingface_hub.HfApi", apis)
    monkeypatch.setattr(auto_download, "_not_servable", {})
    monkeypatch.setattr(utils_module, "_hf_reachability", None)

    for endpoint in ("https://a.example.com", "https://b.example.com"):
        hub_settings.set_hub_settings(endpoint, False)
        validation.validate_hf_token("hf_x", rate_key = "k")
        assert trusted_org.is_trusted_org_repo("unsloth/x")
        assert not auto_download._is_not_servable("org/repo", None)
        auto_download._mark_not_servable("org/repo", None, endpoint)
    assert checks == ["https://a.example.com", "https://b.example.com"]
    assert [call.kwargs["endpoint"] for call in apis.call_args_list] == [
        "https://a.example.com",
        "https://b.example.com",
    ]

    hub_settings.set_hub_settings("https://a.example.com", False)

    def no_gguf_then_switch(*_a, **_k):
        hub_settings.set_hub_settings("https://c.example.com", False)
        return SimpleNamespace(siblings = [])

    apis.return_value.model_info.side_effect = no_gguf_then_switch
    assert asyncio.run(auto_download._is_downloadable_model("org/other", None)) is False
    assert apis.call_args.kwargs["endpoint"] == "https://a.example.com"
    assert not auto_download._is_not_servable("org/other", None)
    assert auto_download._is_not_servable("org/other", None, "https://a.example.com")

    utils_module._hf_reachability = (time.monotonic(), True)
    hub_settings.set_hub_settings("https://b.example.com", False)
    assert utils_module._hf_reachability is None


def test_modelscope_points_hub_clients_at_the_adapter_and_back(store, monkeypatch):
    import huggingface_hub.constants as constants
    import hub.modelscope.router as modelscope
    from utils.hf_endpoint import browser_hf_endpoint

    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    settings = hub_settings.set_hub_source("modelscope")
    assert (settings.source, hub_settings.active_source()) == ("modelscope", "modelscope")
    assert os.environ["HF_ENDPOINT"] == constants.ENDPOINT == "http://127.0.0.1:1234"
    assert hub_settings.hugging_face_endpoint() == MIRROR
    assert os.environ[hub_settings.SOURCE_ENV] == "modelscope"
    assert browser_hf_endpoint() == "https://huggingface.co"

    hub_settings.set_hub_source("huggingface")
    assert os.environ["HF_ENDPOINT"] == MIRROR == browser_hf_endpoint()

    def no_adapter():
        raise RuntimeError("port exhausted")

    monkeypatch.setattr(modelscope, "internal_endpoint", no_adapter)
    assert hub_settings.set_hub_source("modelscope").source == "modelscope"
    assert hub_settings.active_source() == "huggingface" and os.environ["HF_ENDPOINT"] == MIRROR


def test_modelscope_answers_never_open_the_shared_cache_or_trust_code(store, monkeypatch):
    from fastapi import HTTPException

    from hub.services.models import account_access
    from utils.security import trusted_org

    monkeypatch.setenv(hub_settings.SOURCE_ENV, hub_settings.MODELSCOPE)
    monkeypatch.setattr(account_access, "_public_repos", {})
    monkeypatch.setattr(account_access, "_hub_public_answer", lambda *_, **__: True)
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(trusted_org, "_verdict_cache", {})
    assert account_access.repo_is_public("org/repo") is False
    with pytest.raises(HTTPException) as refused:
        account_access.authorize_download("org/repo", "model", None)
    assert refused.value.status_code == 403
    assert trusted_org.is_trusted_org_repo("unsloth/Qwen3-8B", verify_remote = False) is False

    monkeypatch.setenv(hub_settings.SOURCE_ENV, hub_settings.HUGGINGFACE)
    assert account_access.repo_is_public("org/repo") is True
    assert trusted_org.is_trusted_org_repo("unsloth/Qwen3-8B", verify_remote = False) is True


def test_route_switches_the_source(client, store, monkeypatch):
    import hub.modelscope.router as modelscope

    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    body = client.put("/hub/source", json = {"source": "modelscope"}).json()
    assert (body["source"], body["active_source"]) == ("modelscope", "modelscope")
    assert client.get("/hub").json()["source"] == "modelscope"
    assert client.put("/hub/source", json = {"source": "gitee"}).status_code == 422
    client.put("/hub/source", json = {"source": "huggingface"})
