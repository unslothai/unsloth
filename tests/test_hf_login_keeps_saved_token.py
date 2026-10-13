# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import contextlib
import functools
import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Optional

import huggingface_hub
import pytest
from huggingface_hub import constants, hf_api
from huggingface_hub.utils import build_hf_headers

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
        "threading": threading,
        "_HF_TOKEN_LOAD_LOCK": threading.RLock(),
    }
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


def test_passed_token_is_active_only_inside_hf_token_env(hub):
    huggingface_hub.login(token = "hf_saved_write")
    hf_token_env = _load("hf_token_env")

    with hf_token_env("hf_passed_read"):
        assert huggingface_hub.get_token() == "hf_passed_read"
    assert huggingface_hub.get_token() == "hf_saved_write"
    assert "HF_TOKEN" not in os.environ

    with hf_token_env(False):
        assert huggingface_hub.get_token() == "hf_saved_write"


def test_passed_token_overrides_oidc_and_the_implicit_token_opt_out(hub, monkeypatch):
    monkeypatch.setenv("HF_OIDC_RESOURCE", "repo:org/model")
    monkeypatch.setenv("HF_OIDC_ID_TOKEN", "oidc_identity")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    hf_token_env = _load("hf_token_env")

    with hf_token_env("hf_passed_read"):
        assert os.environ["HF_TOKEN"] == "hf_passed_read"
        assert "HF_OIDC_RESOURCE" not in os.environ
        assert "HF_OIDC_ID_TOKEN" not in os.environ
        assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"
        assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is False
        assert build_hf_headers()["authorization"] == "Bearer hf_passed_read"

    assert os.environ["HF_OIDC_RESOURCE"] == "repo:org/model"
    assert os.environ["HF_OIDC_ID_TOKEN"] == "oidc_identity"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is True


def test_scope_restores_an_existing_env_token_after_an_exception(hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_original")

    def fail(model_name, token = None):
        assert huggingface_hub.get_token() == "hf_passed"
        raise RuntimeError("boom")

    load = _load("hf_token_for_load")(fail)
    with pytest.raises(RuntimeError, match = "boom"):
        load("org/gated", token = "hf_passed")
    assert os.environ["HF_TOKEN"] == "hf_original"


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


def test_overlapping_loads_do_not_mix_or_leak_tokens(hub):
    huggingface_hub.login(token = "hf_saved")
    a_inside = threading.Event()
    b_calling = threading.Event()
    b_inside = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()

    def tokenless_read(model_name, token = None):
        if token == "hf_a":
            a_inside.set()
            assert release_a.wait(5)
        else:
            b_inside.set()
            assert release_b.wait(5)
        return huggingface_hub.get_token()

    load = _load("hf_token_for_load")(tokenless_read)

    def run_a():
        return load("org/a", token = "hf_a")

    def run_b():
        b_calling.set()
        return load("org/b", token = "hf_b")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        future_a = pool.submit(run_a)
        assert a_inside.wait(5)
        future_b = pool.submit(run_b)
        assert b_calling.wait(5)
        b_overlapped_a = b_inside.wait(1)
        release_a.set()
        result_a = future_a.result(timeout = 5)
        assert b_inside.wait(5)
        release_b.set()
        result_b = future_b.result(timeout = 5)

    assert (
        b_overlapped_a,
        result_a,
        result_b,
        huggingface_hub.get_token(),
        os.environ.get("HF_TOKEN"),
    ) == (False, "hf_a", "hf_b", "hf_saved", None)


@pytest.mark.parametrize(
    "path, cls",
    [
        ("loader.py", "FastLanguageModel"),
        ("loader.py", "FastModel"),
        ("vision.py", "FastBaseModel"),
    ],
)
def test_the_token_scope_runs_inside_the_offline_guard(path, cls):
    tree = ast.parse((UTILS_PATH.parent / path).read_text(encoding = "utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )
    names = [d.id for d in method.decorator_list if isinstance(d, ast.Name)]
    assert names.index("_offline_aware_load") < names.index("hf_token_for_load")
