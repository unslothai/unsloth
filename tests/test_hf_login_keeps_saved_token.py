# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import logging
import os
from pathlib import Path
import sys
import threading
import types
from typing import Optional

import huggingface_hub
import pytest
from huggingface_hub import constants, hf_api

UTILS_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load(name = "hf_login", **extra):
    names = {
        "hf_login": ("_saved_hf_token_without_oidc", "hf_login"),
        "resolve_vllm_model_path": ("resolve_vllm_model_path",),
        "maybe_prefetch_hf_snapshot": ("maybe_prefetch_hf_snapshot",),
    }[name]
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "os": os,
        "threading": threading,
        "_HF_TOKEN_LOGIN_LOCK": threading.RLock(),
    }
    namespace.update(extra)
    exec(compile(ast.Module(nodes, []), str(UTILS_PATH), "exec"), namespace)
    return namespace[name]


def _load_hf_login():
    return _load()


@pytest.fixture
def hub(monkeypatch, tmp_path):
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_OIDC_RESOURCE",
        "HF_OIDC_ID_TOKEN",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(tmp_path / "stored_tokens"))
    whoami_calls = []

    def whoami(token = None, **kwargs):
        whoami_calls.append(token)
        return {"name": "user", "auth": {"accessToken": {"displayName": token, "role": "read"}}}

    monkeypatch.setattr(hf_api, "whoami", whoami)
    return whoami_calls


def test_passing_a_token_keeps_the_saved_login(hub):
    huggingface_hub.login(token = "hf_saved_write")
    hub.clear()

    assert _load_hf_login()("hf_passed_read") == "hf_passed_read"

    assert huggingface_hub.get_token() == "hf_saved_write"
    assert hub == []


def test_no_token_uses_the_saved_login(hub):
    huggingface_hub.login(token = "hf_saved_write")
    hub.clear()

    assert _load_hf_login()(None) == "hf_saved_write"
    assert hub == []


def test_a_token_on_a_machine_with_no_login_is_saved(hub):
    hf_login = _load_hf_login()

    assert hf_login(None) is None
    assert hf_login("hf_passed") == "hf_passed"
    assert huggingface_hub.get_token() == "hf_passed"


def test_false_means_no_token(hub, monkeypatch):
    lookups = []
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: lookups.append(1))

    assert _load_hf_login()(False) is False
    assert lookups == []


def test_an_oidc_identity_is_not_treated_as_a_saved_login(hub, monkeypatch):
    logged_in = []
    monkeypatch.setenv("HF_OIDC_RESOURCE", "repo:org/model")
    monkeypatch.setenv("HF_OIDC_ID_TOKEN", "oidc_identity")
    monkeypatch.setattr(
        huggingface_hub,
        "get_token",
        lambda: "hf_oidc" if "HF_OIDC_RESOURCE" in os.environ else None,
    )
    monkeypatch.setattr(huggingface_hub, "login", lambda token: logged_in.append(token))

    assert _load_hf_login()("hf_passed") == "hf_passed"
    assert logged_in == ["hf_passed"]
    assert os.environ["HF_OIDC_RESOURCE"] == "repo:org/model"
    assert os.environ["HF_OIDC_ID_TOKEN"] == "oidc_identity"


def test_an_oidc_identity_does_not_hide_a_saved_login(hub, monkeypatch):
    huggingface_hub.login(token = "hf_saved_write")
    hub.clear()
    monkeypatch.setenv("HF_OIDC_RESOURCE", "repo:org/model")
    monkeypatch.setenv("HF_OIDC_ID_TOKEN", "oidc_identity")
    logins = []
    monkeypatch.setattr(huggingface_hub, "login", lambda token: logins.append(token))

    assert _load_hf_login()("hf_passed_read") == "hf_passed_read"
    assert logins == []
    assert hub == []
    assert os.environ["HF_OIDC_RESOURCE"] == "repo:org/model"
    assert os.environ["HF_OIDC_ID_TOKEN"] == "oidc_identity"


def test_no_token_still_uses_an_oidc_identity(hub, monkeypatch):
    monkeypatch.setenv("HF_OIDC_RESOURCE", "repo:org/model")
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_oidc")

    assert _load_hf_login()(None) == "hf_oidc"


def test_vllm_snapshot_does_not_expose_the_load_token_to_other_threads(hub):
    huggingface_hub.login(token = "hf_saved_write")
    entered = threading.Event()
    release = threading.Event()
    calls = []
    result = []

    def prefetch(model_name, **kwargs):
        calls.append((model_name, kwargs))
        entered.set()
        assert release.wait(5)
        return "/cache/private-snapshot"

    resolve = _load(
        "resolve_vllm_model_path",
        maybe_prefetch_hf_snapshot = prefetch,
    )
    thread = threading.Thread(
        target = lambda: result.append(resolve("org/private", "hf_passed_read"))
    )
    thread.start()
    assert entered.wait(5)
    unrelated_token = huggingface_hub.get_token()
    release.set()
    thread.join(5)

    assert unrelated_token == "hf_saved_write"
    assert result == ["/cache/private-snapshot"]
    assert calls == [("org/private", {"token": "hf_passed_read", "return_path": True})]
    assert "HF_TOKEN" not in os.environ


def test_prefetch_can_return_the_authenticated_snapshot_path(hub, monkeypatch):
    calls = []

    def download(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return "/cache/private-snapshot"

    module = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    module.snapshot_download_with_xet_fallback = download
    module.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", module)
    prefetch = _load(
        "maybe_prefetch_hf_snapshot",
        _prefetch_ignore_patterns = lambda *args, **kwargs: [],
    )

    assert (
        prefetch("org/private", token = "hf_passed_read", return_path = True)
        == "/cache/private-snapshot"
    )
    assert calls[0][0] == "org/private"
    assert calls[0][1]["token"] == "hf_passed_read"


def test_vllm_snapshot_failure_leaves_the_ambient_login_unchanged(hub):
    huggingface_hub.login(token = "hf_saved_write")

    def fail(*args, **kwargs):
        raise RuntimeError("download failed")

    resolve = _load("resolve_vllm_model_path", maybe_prefetch_hf_snapshot = fail)
    with pytest.raises(RuntimeError, match = "download failed"):
        resolve("org/private", "hf_passed_read")
    assert huggingface_hub.get_token() == "hf_saved_write"
    assert "HF_TOKEN" not in os.environ


def test_false_or_missing_token_does_not_preload_vllm_model(hub):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected prefetch")

    resolve = _load("resolve_vllm_model_path", maybe_prefetch_hf_snapshot = fail)
    assert resolve("org/public") == "org/public"
    assert resolve("org/public", False) == "org/public"


@pytest.mark.parametrize("path", ["llama.py", "vision.py"])
def test_vllm_receives_the_authenticated_local_snapshot(path):
    source = (UTILS_PATH.parent / path).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "resolve_vllm_model_path"
        for call in calls
    )
    load_calls = [
        call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "load_vllm"
    ]
    assert load_calls
    assert "model_name = _vllm_model_name" in source


def test_offline_fp8_source_loads_receive_the_token():
    source = (UTILS_PATH.parent / "loader_utils.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_offline_quantize_to_fp8"
    )
    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert len(loads) == 3
    assert all(any(keyword.arg == "token" for keyword in call.keywords) for call in loads)


def test_no_process_global_load_token_scope_remains():
    source = UTILS_PATH.read_text(encoding = "utf-8")
    assert "hf_token_env" not in source
    assert "hf_token_for_load" not in source
