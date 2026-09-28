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

    def compare_and_set(
        key,
        expected,
        value,
        absent = (),
    ):
        if values.get(key) != expected or any(k in values for k in absent):
            return False
        values[key] = value
        return True

    monkeypatch.setattr(studio_db, "compare_and_set_app_setting", compare_and_set)
    monkeypatch.setattr(hub_settings, "_operator_env", None)
    monkeypatch.setattr(hub_settings, "_operator_endpoints", None, raising = False)
    monkeypatch.setattr(hub_settings, "_saved_only_endpoints", frozenset(), raising = False)
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


@pytest.mark.parametrize(
    "flag, configured, automatic",
    [
        ("1", {}, True),
        ("0", {}, False),
        ("1", {hub_settings.SOURCE_KEY: "huggingface"}, False),
        ("1", {hub_settings.HF_ENDPOINT_KEY: ""}, False),
        ("1", {"HF_ENDPOINT": MIRROR}, False),
    ],
)
def test_mainland_china_defaults_to_modelscope_until_the_hub_is_configured(
    store, monkeypatch, flag, configured, automatic
):
    import hub.modelscope.router as modelscope

    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    monkeypatch.delenv("HF_ENDPOINT")
    monkeypatch.setenv("UNSLOTH_MIRROR_FALLBACK", flag)
    for key, value in configured.items():
        if key == "HF_ENDPOINT":
            monkeypatch.setenv(key, value)
        else:
            store[key] = value
    hub_settings.apply_hub_settings()
    settings = hub_settings.get_hub_settings()
    expected = hub_settings.MODELSCOPE if automatic else hub_settings.HUGGINGFACE
    assert (settings.source, settings.source_automatic) == (expected, automatic)
    assert hub_settings.active_source() == expected
    assert store == {k: v for k, v in configured.items() if k != "HF_ENDPOINT"}


@pytest.mark.parametrize(
    "zone, resolvers, platform, expected",
    [
        ("Asia/Shanghai", "", "linux", True),
        (":/usr/share/zoneinfo/Asia/Urumqi", "", "darwin", True),
        ("Asia/Singapore", "", "linux", False),
        ("Asia/Singapore", "nameserver 192.168.1.1\nnameserver 223.5.5.5\n", "linux", True),
        ("Asia/Singapore", "nameserver 223.5.5.50\n", "linux", False),
    ],
)
def test_mainland_china_follows_the_installer_rules(
    tmp_path, monkeypatch, zone, resolvers, platform, expected
):
    from utils import mainland_china

    conf = tmp_path / "resolv.conf"
    conf.write_text(resolvers)
    monkeypatch.setattr(mainland_china, "_RESOLV_CONFS", (str(tmp_path / "missing"), str(conf)))
    monkeypatch.setattr(mainland_china.sys, "platform", platform)
    monkeypatch.setenv("TZ", zone)
    monkeypatch.delenv("UNSLOTH_MIRROR_FALLBACK")
    mainland_china.in_mainland_china.cache_clear()
    try:
        assert mainland_china.china_mirrors_enabled() is expected
    finally:
        mainland_china.in_mainland_china.cache_clear()


@pytest.mark.parametrize(
    "zone, resolvers, expected",
    [
        ("China Standard Time", [], True),
        ("Singapore Standard Time", [], False),
        ("Singapore Standard Time", ["192.168.1.1", "223.5.5.5"], True),
        ("Singapore Standard Time", ["223.5.5.50", "8.8.8.8"], False),
    ],
)
def test_windows_reads_the_registry_time_zone_and_adapter_resolvers(
    monkeypatch, zone, resolvers, expected
):
    from utils import mainland_china

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    registry = _types.SimpleNamespace(
        HKEY_LOCAL_MACHINE = None,
        OpenKey = lambda *_: _Key(),
        QueryValueEx = lambda _key, name: (zone, 1),
    )
    monkeypatch.setattr(mainland_china, "_windows_resolvers", lambda: resolvers)
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(mainland_china.sys, "platform", "win32")
    monkeypatch.delenv("TZ", raising = False)
    mainland_china.in_mainland_china.cache_clear()
    try:
        assert mainland_china.in_mainland_china() is expected
    finally:
        mainland_china.in_mainland_china.cache_clear()


@pytest.mark.parametrize(
    "database, automatic", [(None, True), ("garbage", False), ("malformed row", False)]
)
def test_only_a_missing_database_reads_as_unconfigured(monkeypatch, database, automatic):
    import sqlite3

    from utils.account_context import OWNER, run_as
    from utils.paths.storage_roots import studio_db_path

    monkeypatch.delitem(sys.modules, "storage.studio_db")
    monkeypatch.delenv("HF_ENDPOINT", raising = False)
    monkeypatch.setenv("UNSLOTH_MIRROR_FALLBACK", "1")
    monkeypatch.setattr(hub_settings, "_operator_env", None)
    path = run_as(OWNER, studio_db_path)
    path.parent.mkdir(parents = True, exist_ok = True)
    if database == "garbage":
        path.write_bytes(b"not a database" * 100)
    elif database == "malformed row":
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE app_settings (key TEXT PRIMARY KEY, value_json TEXT)")
            conn.execute("INSERT INTO app_settings VALUES (?, ?)", (hub_settings.SOURCE_KEY, "{"))
    assert hub_settings.get_hub_settings().source_automatic is automatic


def test_windows_resolvers_come_from_adapters_that_are_up():
    import ctypes

    from utils import mainland_china as mc

    keep = []

    def servers(*addresses):
        head = ctypes.POINTER(mc._Server)()
        for address in reversed(addresses):
            family, octets = (2, address.split(".")) if "." in address else (23, ["0"] * 4)
            raw = (ctypes.c_ubyte * 16)(family, 0, 0, 0, *map(int, octets))
            node = mc._Server(
                next = head, address = mc._Address(ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte)), 16)
            )
            keep.extend([raw, node])
            head = ctypes.pointer(node)
        return head

    head = ctypes.POINTER(mc._Adapter)()
    for status, dns in reversed(
        [
            (1, servers("::1", "192.168.1.1")),
            (2, servers("223.5.5.5")),
            (1, servers("119.29.29.29")),
        ]
    ):
        adapter = mc._Adapter(next = head, dns = dns, oper_status = status)
        keep.append(adapter)
        head = ctypes.pointer(adapter)
    assert mc._up_adapter_resolvers(head) == ["192.168.1.1", "119.29.29.29"]


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: False
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


def test_an_api_key_cannot_move_the_endpoint_or_source(client, store):
    before = client.get("/hub").json()
    client.app.dependency_overrides[settings.authenticated_via_api_key] = lambda: True
    assert (
        client.put(
            "/hub",
            json = {
                "hf_endpoint": "https://collector.example",
                "datasets_server_follows_endpoint": True,
            },
        ).status_code
        == 403
    )
    assert client.put("/hub/source", json = {"source": "modelscope"}).status_code == 403
    assert client.post("/hub/source-notice").status_code == 403
    client.app.dependency_overrides[settings.authenticated_via_api_key] = lambda: False
    assert client.get("/hub").json() == before


def test_the_owner_is_told_once_and_the_automatic_source_is_kept(client, store, monkeypatch):
    import hub.modelscope.router as modelscope

    def no_adapter():
        raise RuntimeError("port exhausted")

    monkeypatch.delenv("HF_ENDPOINT")
    monkeypatch.setenv("UNSLOTH_MIRROR_FALLBACK", "1")
    monkeypatch.setattr(modelscope, "internal_endpoint", no_adapter)
    hub_settings.apply_hub_settings()
    assert client.post("/hub/source-notice").json() == {"granted": False}
    assert store == {}

    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    hub_settings.apply_hub_settings()
    grants = [client.post("/hub/source-notice").json()["granted"] for _ in range(2)]
    assert grants == [True, False]
    assert store == {hub_settings.SOURCE_KEY: "modelscope"}
    store.clear()
    claim = studio_db.compare_and_set_app_setting

    def endpoint_saved_meanwhile(*args, **kwargs):
        # Another tab saves an endpoint between this claim's read and its insert.
        store[hub_settings.HF_ENDPOINT_KEY] = MIRROR
        return claim(*args, **kwargs)

    monkeypatch.setattr(studio_db, "compare_and_set_app_setting", endpoint_saved_meanwhile)
    assert client.post("/hub/source-notice").json() == {"granted": False}
    assert store == {hub_settings.HF_ENDPOINT_KEY: MIRROR}
    store.clear()
    store[hub_settings.SOURCE_KEY] = "modelscope"
    monkeypatch.setenv("UNSLOTH_MIRROR_FALLBACK", "0")
    assert client.get("/hub").json()["source"] == "modelscope"


def test_the_claim_insert_requires_the_endpoint_to_stay_unsaved():
    source, endpoint = hub_settings.SOURCE_KEY, hub_settings.HF_ENDPOINT_KEY
    studio_db.upsert_app_settings({endpoint: MIRROR})
    assert not studio_db.compare_and_set_app_setting(source, None, "modelscope", absent = (endpoint,))
    assert studio_db.get_app_setting(source, None) is None
    assert studio_db.compare_and_set_app_setting(source, None, "modelscope", absent = ("unset",))
    assert studio_db.get_app_setting(source, None) == "modelscope"


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


def test_modelscope_adapter_is_reached_past_a_configured_proxy(store, monkeypatch):
    import http.server
    import socket
    import threading
    import hub.modelscope.router as modelscope
    from huggingface_hub.utils import _http

    class Ok(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_a):
            pass

    adapter = http.server.HTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target = adapter.serve_forever, daemon = True).start()
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_proxy = f"http://127.0.0.1:{closed.getsockname()[1]}"
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, dead_proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "example.com")
    url = f"http://127.0.0.1:{adapter.server_port}"
    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: url)
    _http.close_session()
    _http.get_session()
    try:
        hub_settings.set_hub_source("modelscope")
        assert _http.get_session().get(url + "/").status_code == 200
        assert os.environ["NO_PROXY"] == "example.com,127.0.0.1"
    finally:
        hub_settings.set_hub_source("huggingface")
        _http.close_session()
        adapter.shutdown()
        closed.close()


def test_modelscope_keeps_a_system_proxy(store, monkeypatch):
    # macOS / Windows: getproxies() = environment proxies OR the system (SCDynamicStore / registry) ones.
    import urllib.request
    import hub.modelscope.router as modelscope

    system = {"http": "http://10.0.0.1:7890", "https": "http://10.0.0.1:7890"}
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: urllib.request.getproxies_environment() or dict(system),
    )
    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    try:
        hub_settings.set_hub_source("modelscope")
        hub_settings.set_hub_source("huggingface")
        assert urllib.request.getproxies().get("https") == system["https"]
    finally:
        hub_settings.set_hub_source("huggingface")


def test_live_switch_does_not_abort_a_streaming_download(store, monkeypatch):
    import http.server
    import threading
    import hub.modelscope.router as modelscope
    from huggingface_hub.utils import _http

    started, release = threading.Event(), threading.Event()

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(6 * 65536))
            self.end_headers()
            self.wfile.write(b"x" * 65536)
            self.wfile.flush()
            started.set()
            release.wait(10)
            for _ in range(5):
                time.sleep(0.2)
                self.wfile.write(b"x" * 65536)
                self.wfile.flush()

        def log_message(self, *_a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1")  # the test server itself is reachable
    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://localhost:1234")
    out = {}

    def stream():
        try:
            with _http.get_session().stream("GET", f"http://127.0.0.1:{server.server_port}/") as r:
                out["n"] = sum(len(c) for c in r.iter_bytes())
        except Exception as exc:  # noqa: BLE001
            out["error"] = repr(exc)

    _http.close_session()
    t = threading.Thread(target = stream)
    t.start()
    try:
        assert started.wait(10)
        hub_settings.set_hub_source("modelscope")
        release.set()
        t.join(10)
        assert out == {"n": 6 * 65536}
    finally:
        release.set()
        hub_settings.set_hub_source("huggingface")
        server.shutdown()


def test_modelscope_switch_survives_huggingface_hub_0x(store, monkeypatch):
    # 0.36.x (Python 3.9 installs) has requests sessions and no close_session / _CLIENT_LOCK.
    import types
    import hub.modelscope.router as modelscope

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.utils._http",
        types.SimpleNamespace(reset_sessions = lambda: None),
    )
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    try:
        assert hub_settings.set_hub_source("modelscope").source == "modelscope"
        assert "127.0.0.1" in os.environ["NO_PROXY"].split(",")
    finally:
        hub_settings.set_hub_source("huggingface")


def test_a_saved_endpoint_stays_out_of_the_csp_the_operators_does_not(store):
    from utils.hf_endpoint import csp_asset_sources, csp_connect_sources

    hub_settings.apply_hub_settings()
    assert csp_connect_sources() == (MIRROR, OPERATOR_DS)
    hub_settings.set_hub_settings("https://10.0.0.5:9000", True)
    assert os.environ["HF_ENDPOINT"] == "https://10.0.0.5:9000"
    assert csp_connect_sources() == ()
    hub_settings.set_hub_settings("http://127.0.0.1:9700", False)
    assert csp_connect_sources() == (OPERATOR_DS,) and csp_asset_sources() == ()
    hub_settings.set_hub_settings("", False)
    assert csp_connect_sources() == ("https://huggingface.co", OPERATOR_DS)
