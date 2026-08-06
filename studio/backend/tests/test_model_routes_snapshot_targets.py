# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The model routes must agree with the training resolver about cached snapshots.

Two ways they disagreed, both reachable from resume:

* ``_model_config_inspection_target`` probed only the snapshot root, so a cached
  Spark-TTS/BiCodec copy (everything trainable under ``LLM/``) made ``/api/models/config``
  answer "Selected cached model is no longer available" for a cache the training
  resolver happily accepts.
* the ``model_snapshot_repo_id`` guard used an ``owner/repo``-only regex, so resuming or
  scanning a namespace-less Hub model such as ``gpt2`` returned 400 before the snapshot
  could be inspected, even though the shared validator and the picker both allow the
  one-segment form.
"""

import json

import pytest
from fastapi import HTTPException

from hub.utils.paths import is_valid_repo_id
from routes import models as models_routes


_BICODEC = "unsloth/Spark-TTS-0.5B"
_PLAIN = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [root])
    return root


@pytest.fixture
def bicodec_subdirs(monkeypatch):
    import utils.security as security_pkg
    monkeypatch.setattr(
        security_pkg,
        "security_load_subdirs",
        lambda model_name, hf_token = None, local_files_only = False: ("LLM",)
        if model_name == _BICODEC
        else (),
    )


def _snapshot(
    cache_root,
    repo_id,
    revision = "b" * 40,
):
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(revision, encoding = "utf-8")
    return snapshot


def _write_model(directory):
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (directory / "model.safetensors").write_bytes(b"\x00" * 256)


def test_a_subdir_loaded_cache_is_inspectable(cache_root, bicodec_subdirs):
    snapshot = _snapshot(cache_root, _BICODEC)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    _write_model(snapshot / "LLM")

    resolved = models_routes._model_config_inspection_target(_BICODEC, True, str(snapshot))

    assert resolved == str(snapshot)


def test_an_ordinary_cache_is_still_inspectable(cache_root, bicodec_subdirs):
    snapshot = _snapshot(cache_root, _PLAIN)
    _write_model(snapshot)

    assert models_routes._model_config_inspection_target(_PLAIN, True, str(snapshot)) == str(
        snapshot
    )


def test_a_missing_cache_still_404s(cache_root, bicodec_subdirs):
    snapshot = _snapshot(cache_root, _BICODEC)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")

    with pytest.raises(HTTPException) as excinfo:
        models_routes._model_config_inspection_target(_BICODEC, True, str(snapshot))

    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("repo_id", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_namespace_less_hub_ids_are_valid(repo_id):
    """The guard must match the shared validator the rest of the app uses."""
    assert is_valid_repo_id(repo_id) is True
    assert models_routes._is_valid_repo_id(repo_id) is False, (
        "the owner/repo-only regex is what made the snapshot guard reject these; "
        "if it now accepts them this test is pinning the wrong thing"
    )


@pytest.mark.parametrize(
    "repo_id", ["", "   ", "a/b/c", "../etc", "owner/repo.git", "own--er/repo"]
)
def test_genuinely_invalid_ids_are_still_rejected(repo_id):
    assert is_valid_repo_id(repo_id) is False


def test_the_snapshot_guard_uses_the_shared_validator():
    """Wiring contract: the 400 branch must not be back on the two-segment regex."""
    import inspect

    source = inspect.getsource(models_routes)
    guard = source.split("snapshot_repo_id = model_snapshot_repo_id.strip()", 1)[1]
    guard = guard.split("if local_model:", 1)[0]

    assert "_shared_is_valid_repo_id(snapshot_repo_id)" in guard
    assert "not _is_valid_repo_id(snapshot_repo_id)" not in guard
