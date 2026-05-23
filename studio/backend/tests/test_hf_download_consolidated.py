# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint snapshot downloads must keep ``consolidated.*`` weights when they
are a repo's only weight format, and skip them only when transformers-format
shards make them redundant."""

from types import SimpleNamespace

import huggingface_hub
import pytest

from workers.hf_download import _resolve_snapshot_ignore_patterns
from utils.hf_cache_scan import DownloadRegistry
from utils.hf_snapshot_filters import snapshot_download_size, snapshot_weight_size


def _patch_siblings(monkeypatch, filenames):
    siblings = [SimpleNamespace(rfilename = name) for name in filenames]

    def fake_model_info(repo_id, token = None, timeout = None):
        return SimpleNamespace(siblings = siblings)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)


def test_consolidated_dropped_when_transformers_shards_present(monkeypatch):
    _patch_siblings(monkeypatch, [
        "config.json",
        "consolidated.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ])
    patterns = _resolve_snapshot_ignore_patterns("org/mistral", None)
    assert "consolidated*" in patterns


def test_consolidated_kept_when_only_weight_format(monkeypatch):
    _patch_siblings(monkeypatch, [
        "config.json",
        "tokenizer.json",
        "consolidated.safetensors",
    ])
    patterns = _resolve_snapshot_ignore_patterns("org/consolidated-only", None)
    assert "consolidated*" not in patterns


def test_consolidated_pth_only_is_kept(monkeypatch):
    _patch_siblings(monkeypatch, [
        "params.json",
        "consolidated.00.pth",
    ])
    patterns = _resolve_snapshot_ignore_patterns("org/pth-only", None)
    assert "consolidated*" not in patterns


def test_pytorch_bin_shards_make_consolidated_redundant(monkeypatch):
    _patch_siblings(monkeypatch, [
        "config.json",
        "consolidated.00.pth",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    ])
    patterns = _resolve_snapshot_ignore_patterns("org/bin-shards", None)
    assert "consolidated*" in patterns


def test_metadata_failure_keeps_consolidated(monkeypatch):
    def boom(repo_id, token = None, timeout = None):
        raise RuntimeError("network down")

    monkeypatch.setattr(huggingface_hub, "model_info", boom)
    patterns = _resolve_snapshot_ignore_patterns("org/unknown", None)
    assert "consolidated*" not in patterns


def test_baseline_runtime_ignores_are_always_present(monkeypatch):
    _patch_siblings(monkeypatch, ["config.json", "model.safetensors"])
    patterns = _resolve_snapshot_ignore_patterns("org/standard", None)
    for expected in ("*.gguf", "*.onnx", "onnx/*", "openvino/*", "mlx/*"):
        assert expected in patterns


def test_snapshot_size_matches_worker_ignored_files():
    siblings = [
        SimpleNamespace(rfilename = "config.json", size = 10),
        SimpleNamespace(rfilename = "model.safetensors", size = 100),
        SimpleNamespace(rfilename = "model.gguf", size = 1000),
        SimpleNamespace(rfilename = "onnx/model.onnx", size = 1000),
        SimpleNamespace(rfilename = "openvino/model.xml", size = 1000),
        SimpleNamespace(rfilename = "mlx/model.safetensors", size = 1000),
        SimpleNamespace(rfilename = "consolidated.safetensors", size = 500),
    ]

    assert snapshot_download_size(siblings) == 110
    assert snapshot_weight_size(siblings) == 100


def test_download_registry_rejects_concurrent_same_repo_variants():
    registry = DownloadRegistry()

    assert registry.claim("org/repo::Q4_K_M", "http") == (True, "running")
    assert registry.claim("org/repo::Q8_0", "http") == (False, "running")
    assert not registry.adoptable("org/repo::Q8_0")
