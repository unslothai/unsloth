# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resuming a pinned snapshot must use the same load roots as everything else.

``ca7c72e75`` taught the cached-snapshot probes about ``security_load_subdirs`` so a repo
like ``unsloth/Spark-TTS-0.5B`` -- whose snapshot root holds only ``README.md`` and
``config.yaml``, with everything trainable under ``LLM/`` -- is not reported as absent.
The resume branch of ``_reject_untrainable_model_request`` kept its own hardcoded
``("config.json", "adapter_config.json")`` tuple, so the one path that *already has* a
server-verified pin was the one that could not see it: ``latest_snapshot_from_cache_path``
returned None, ``path`` stayed None, and the very next block turned that into
409 ``hf_model_not_cached_offline`` for a cache sitting right there on disk. Online it is
no better -- it falls through to a remote metadata round trip that offline users cannot
make and that the pin exists precisely to avoid.

Resuming is when the pin matters most, so it has to agree with the resolver.
"""

import json

import pytest
from fastapi import HTTPException

from models.training import TrainingStartRequest
from routes import training as training_routes


_BICODEC = "unsloth/Spark-TTS-0.5B"
_PLAIN = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """A tmp dir registered as an HF cache root, as validated_repo_cache_path requires."""
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


@pytest.fixture
def offline(monkeypatch):
    """Offline is where the miss is unrecoverable, so it is the sharpest observable."""
    monkeypatch.setattr(training_routes, "hf_env_offline", lambda: True)


def _snapshot(
    cache_root,
    repo_id,
    revision = "c" * 40,
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
    (directory / "model.safetensors").write_bytes(b"\x00" * 512)


def _request(model_name, snapshot_path):
    return TrainingStartRequest(
        model_name = model_name,
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        resume_from_checkpoint = "/runs/run-1/checkpoint-10",
        model_snapshot_path = str(snapshot_path),
    )


def test_a_pinned_subdir_snapshot_is_accepted_on_resume(cache_root, bicodec_subdirs, offline):
    snapshot = _snapshot(cache_root, _BICODEC)
    # The real Spark-TTS layout: nothing loadable at the snapshot root.
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    _write_model(snapshot / "LLM")

    result = training_routes._reject_untrainable_model_request(_request(_BICODEC, snapshot))

    assert result.model_name == _BICODEC


def test_a_pinned_root_loading_snapshot_is_unaffected(cache_root, bicodec_subdirs, offline):
    snapshot = _snapshot(cache_root, _PLAIN)
    _write_model(snapshot)

    result = training_routes._reject_untrainable_model_request(_request(_PLAIN, snapshot))

    assert result.model_name == _PLAIN


def test_an_empty_pinned_snapshot_is_still_refused(cache_root, bicodec_subdirs, offline):
    """Widening the probe must not turn "nothing usable here" into a false positive."""
    snapshot = _snapshot(cache_root, _BICODEC)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    (snapshot / "LLM").mkdir()

    with pytest.raises(HTTPException) as excinfo:
        training_routes._reject_untrainable_model_request(_request(_BICODEC, snapshot))

    assert excinfo.value.status_code == 409


def test_resume_keeps_the_checkpoints_own_pin(cache_root, bicodec_subdirs, offline):
    """cached_model_pin is for fresh offline starts; resume must not re-pin the run."""
    snapshot = _snapshot(cache_root, _BICODEC)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    _write_model(snapshot / "LLM")

    result = training_routes._reject_untrainable_model_request(_request(_BICODEC, snapshot))

    assert result.cached_model_pin is None


def test_load_subdir_lookup_failure_degrades_to_root_only(cache_root, offline, monkeypatch):
    """Detection can raise offline or for a gated repo; resume must not break with it."""
    import utils.security as security_pkg

    def boom(
        model_name,
        hf_token = None,
        local_files_only = False,
    ):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(security_pkg, "security_load_subdirs", boom)
    snapshot = _snapshot(cache_root, _PLAIN)
    _write_model(snapshot)

    result = training_routes._reject_untrainable_model_request(_request(_PLAIN, snapshot))

    assert result.model_name == _PLAIN
