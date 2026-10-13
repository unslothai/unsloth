# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import contextlib
import logging
import os
from pathlib import Path
from typing import Optional

import huggingface_hub
import pytest
from huggingface_hub import constants, hf_api

UTILS_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load(name = "hf_login"):
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "contextlib": contextlib,
        "os": os,
    }
    exec(compile(ast.Module([node], []), str(UTILS_PATH), "exec"), namespace)
    return namespace[name]


def _load_hf_login():
    return _load()


@pytest.fixture
def hub(monkeypatch, tmp_path):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
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


def test_passed_token_is_active_only_inside_hf_token_env(hub):
    huggingface_hub.login(token = "hf_saved_write")
    hf_token_env = _load("hf_token_env")

    with hf_token_env("hf_passed_read"):
        assert huggingface_hub.get_token() == "hf_passed_read"
    assert huggingface_hub.get_token() == "hf_saved_write"
    assert "HF_TOKEN" not in os.environ

    with hf_token_env(False):
        assert huggingface_hub.get_token() == "hf_saved_write"
