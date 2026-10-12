# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Documents & RAG > Embedding device: RAG_EMBED_DEVICE first, then the owner's saved choice."""

from pathlib import Path
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import HTTPException

from core.rag import config, embeddings
from core.rag.embed_llama_server import LlamaServerBackend
from routes import settings as settings_routes
from storage import studio_db
from utils import embedding_model_settings as ems
from utils.account_context import AccountContext, bind_account, reset_account
from utils.hardware.hardware import DeviceType

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(config, "EMBED_DEVICE", "auto")
    ems.forget_cached_embedding_device()
    yield
    ems.forget_cached_embedding_device()


@pytest.fixture
def released(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "release_backend", lambda: calls.append(1) or True)
    return calls


def _put(device):
    return settings_routes.update_embedding_device(
        settings_routes.EmbeddingDevicePayload(device = device), current_subject = "owner"
    )


def test_unset_keeps_each_backends_hardware_default():
    assert config.embed_device_preference() == "auto"
    assert config.embed_device_locked() is False


def test_saved_gpu_moves_every_model_to_the_gpu(monkeypatch, released):
    """Both backends read the one preference, so the choice is not tied to a model."""
    monkeypatch.setattr(embeddings, "get_device", lambda: DeviceType.CUDA)
    assert embeddings._device() == "cpu"

    response = _put("gpu")

    assert response.embedding_device == "gpu"
    assert response.embedding_device_locked is False
    assert released == [1]
    assert embeddings._device() == "cuda"
    assert LlamaServerBackend()._use_gpu() is True
    # A Settings choice is a preference: a failed GPU start still falls back to CPU.
    assert config.embed_device_requires_gpu() is False


def test_saved_cpu_keeps_llama_server_off_the_gpu(monkeypatch, released):
    monkeypatch.setattr(LlamaServerBackend, "_gpu_available", staticmethod(lambda: True))
    assert LlamaServerBackend()._use_gpu() is True
    _put("cpu")
    assert LlamaServerBackend()._use_gpu() is False


def test_saving_the_same_device_again_keeps_the_embedder_resident(released):
    _put("gpu")
    _put("gpu")
    assert released == [1]
    _put("auto")
    assert released == [1, 1]
    assert config.embed_device_preference() == "auto"


def test_the_choice_is_installation_wide(released):
    _put("gpu")
    token = bind_account(ALICE)
    try:
        assert config.embed_device_preference() == "gpu"
    finally:
        reset_account(token)


@pytest.mark.parametrize("env, expected", [("gpu", "gpu"), (" cuda ", "gpu"), ("cpu", "cpu")])
def test_a_device_in_the_environment_wins_and_locks_settings(monkeypatch, released, env, expected):
    ems.set_embedding_device("cpu" if expected == "gpu" else "gpu")
    monkeypatch.setattr(config, "EMBED_DEVICE", env)

    assert config.embed_device_preference() == expected
    assert settings_routes._embedding_device_state() == (expected, True)
    with pytest.raises(HTTPException) as exc:
        _put("auto")
    assert exc.value.status_code == 409
    assert released == []


def test_environment_auto_leaves_the_choice_to_settings(monkeypatch, released):
    monkeypatch.setattr(config, "EMBED_DEVICE", "auto")
    _put("gpu")
    assert config.embed_device_preference() == "gpu"


def test_an_unrecognized_stored_value_reads_as_auto():
    studio_db.upsert_app_settings({ems.EMBEDDING_DEVICE_SETTING_KEY: "banana"})
    assert ems.get_stored_embedding_device() == "auto"
    with pytest.raises(ValueError):
        ems.set_embedding_device("banana")
