# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import contextlib
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Optional

import huggingface_hub
import pytest
from huggingface_hub import constants, hf_api

UTILS_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load(name = "hf_login"):
    names = ("hf_login", "hf_token_env", "hf_token_for_load")
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "contextlib": contextlib,
        "functools": functools,
        "inspect": inspect,
        "os": os,
    }
    exec(compile(ast.Module(nodes, []), str(UTILS_PATH), "exec"), namespace)
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


def _tokenless_load(
    model_name,
    max_seq_length = 2048,
    token = None,
    **kwargs,
):
    return huggingface_hub.get_token()


def test_tokenless_reads_inside_the_load_use_the_passed_token(hub):
    huggingface_hub.login(token = "hf_saved_write")
    load = _load("hf_token_for_load")(_tokenless_load)

    assert load("org/gated", token = "hf_passed_read") == "hf_passed_read"
    assert load("org/gated", 2048, "hf_passed_read") == "hf_passed_read"
    assert load("org/gated") == "hf_saved_write"
    assert load("org/gated", token = False) == "hf_saved_write"
    assert huggingface_hub.get_token() == "hf_saved_write"
    assert "HF_TOKEN" not in os.environ


def test_the_load_scope_still_saves_a_token_on_a_machine_with_no_login(hub):
    load = _load("hf_token_for_load")(_tokenless_load)

    assert load("org/gated", token = "hf_passed") == "hf_passed"
    assert huggingface_hub.get_token() == "hf_passed"


@pytest.mark.parametrize(
    "path, cls",
    [("loader.py", "FastLanguageModel"), ("loader.py", "FastModel"), ("vision.py", "FastBaseModel")],
)
def test_the_token_scope_runs_inside_the_offline_guard(path, cls):
    tree = ast.parse((UTILS_PATH.parent / path).read_text(encoding = "utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    names = [d.id for d in method.decorator_list if isinstance(d, ast.Name)]
    assert names.index("_offline_aware_load") < names.index("hf_token_for_load")
