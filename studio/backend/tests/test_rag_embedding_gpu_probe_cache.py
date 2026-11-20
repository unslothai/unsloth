# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto backend choice with no embedder loaded: a found GPU is not re-probed, a miss always is."""

import pytest

from core.rag import config, embeddings
from core.inference.llama_cpp import LlamaCppBackend


@pytest.fixture
def probe(monkeypatch):
    state = {"gpus": [(0, 40000)], "calls": 0, "binary": None, "finds": 0, "probed_with": []}

    def _gpus(*_a, **kw):
        state["calls"] += 1
        state["probed_with"].append(kw.get("binary"))
        return state["gpus"]

    def _find():
        state["finds"] += 1
        return state["binary"]

    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_gpus))
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(_find))
    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    embeddings._reset_backend()
    monkeypatch.setattr(embeddings, "_gpu_found_key", None, raising = False)
    yield state
    embeddings._reset_backend()


def test_found_gpu_is_not_probed_again(probe):
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 1


def test_no_gpu_answer_is_probed_every_time(probe, tmp_path):
    """A miss is never cached: the GPU may only have been slow to answer."""
    binary = tmp_path / "llama-server"
    binary.write_text("")
    probe["binary"] = str(binary)
    probe["gpus"] = []
    assert embeddings._resolve_auto() == "llama-server"
    assert embeddings._resolve_auto() == "llama-server"
    assert probe["calls"] == 2
    probe["gpus"] = [(0, 40000)]
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 3


def test_miss_looks_up_the_binary_once(probe, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("")
    probe["binary"] = str(binary)
    probe["gpus"] = []
    assert embeddings._resolve_auto() == "llama-server"
    assert probe["finds"] == 1
    assert probe["probed_with"] == [str(binary)]


def test_environment_change_probes_again(probe, monkeypatch):
    assert embeddings._resolve_auto() == "sentence-transformers"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "r36-probe-mask")
    probe["gpus"] = []
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 2


def test_binary_change_probes_again(probe, tmp_path):
    """A Vulkan or SYCL build answers the probe differently."""
    assert embeddings._resolve_auto() == "sentence-transformers"
    binary = tmp_path / "llama-server"
    binary.write_text("cuda")
    probe["binary"] = str(binary)
    probe["gpus"] = []
    assert embeddings._resolve_auto() == "llama-server"
    assert probe["calls"] == 2
    probe["gpus"] = [(0, 40000)]
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 3
    binary.write_text("vulkan build")
    probe["gpus"] = []
    assert embeddings._resolve_auto() == "llama-server"
    assert probe["calls"] == 4


def test_unload_probes_again(probe):
    assert embeddings._resolve_auto() == "sentence-transformers"
    embeddings.release_backend()
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 2


def test_plan_for_unsaved_model_probes_once(probe, monkeypatch):
    import utils.embedding_model_settings as ems

    monkeypatch.setattr(ems, "get_stored_backend", lambda _model: None)
    for _ in range(3):
        assert (
            embeddings.resolved_backend_for_model("unsloth/bge-small-en-v1.5")
            == "sentence-transformers"
        )
    assert probe["calls"] == 1


def test_mask_changed_during_probe_is_not_cached(probe, monkeypatch, tmp_path):
    """A mask lifted while the probe runs must not mark the masked environment as having a GPU."""
    binary = tmp_path / "llama-server"
    binary.write_text("")
    probe["binary"] = str(binary)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def _gpus(*_a, **_kw):
        probe["calls"] += 1
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        return [(0, 40000)]

    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_gpus))
    assert embeddings._resolve_auto() == "sentence-transformers"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert embeddings._resolve_auto() == "sentence-transformers"
    assert probe["calls"] == 2
