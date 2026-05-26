# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint snapshot downloads must keep ``consolidated.*`` weights when they
are a repo's only weight format, and skip them only when transformers-format
shards make them redundant."""

import signal
from io import BytesIO
from types import SimpleNamespace

import huggingface_hub
import pytest

import workers.hf_download as hf_download
import utils.hf_cache_scan as hf_cache_scan
from workers.hf_download import (
    _gguf_variant_patterns,
    _resolve_snapshot_ignore_patterns,
)
from utils.hf_cache_scan import DownloadRegistry, drain_stderr_excerpt
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


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason = "SIGPIPE is POSIX-only")
def test_sigpipe_exit_classifies_as_cancelled():
    sigpipe = int(signal.SIGPIPE)

    assert hf_cache_scan.classify_exit(-sigpipe) == "cancelled"
    assert hf_cache_scan.classify_exit(128 + sigpipe) == "cancelled"


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


def test_metadata_retry_can_drop_consolidated(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename = "config.json"),
        SimpleNamespace(rfilename = "consolidated.safetensors"),
        SimpleNamespace(rfilename = "model.safetensors"),
    ]
    calls = []

    def flaky(repo_id, token = None, timeout = None):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return SimpleNamespace(siblings = siblings)

    monkeypatch.setattr(huggingface_hub, "model_info", flaky)
    monkeypatch.setattr(hf_download.time, "sleep", lambda _seconds: None)
    patterns = _resolve_snapshot_ignore_patterns("org/retry", None)
    assert "consolidated*" in patterns
    assert calls == [10.0, 30.0]


def test_metadata_failure_keeps_consolidated(monkeypatch, capsys):
    calls = 0

    def boom(repo_id, token = None, timeout = None):
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(huggingface_hub, "model_info", boom)
    monkeypatch.setattr(hf_download.time, "sleep", lambda _seconds: None)
    patterns = _resolve_snapshot_ignore_patterns("org/unknown", None)
    captured = capsys.readouterr()
    assert "consolidated*" not in patterns
    assert calls == 2
    assert (
        "metadata unavailable, downloading full snapshot for org/unknown"
        in captured.err
    )


def test_gguf_metadata_failure_reports_metadata_error(monkeypatch, capsys):
    def boom(repo_id, token = None, timeout = None, files_metadata = False):
        raise TimeoutError("metadata timeout")

    monkeypatch.setattr(huggingface_hub, "model_info", boom)
    monkeypatch.setattr(hf_download.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match = "Metadata unavailable"):
        _gguf_variant_patterns("org/gguf", "Q4_K_M", None)
    captured = capsys.readouterr()
    assert "No GGUF shards" not in captured.err
    assert (
        "metadata unavailable, cannot resolve GGUF variant 'Q4_K_M'"
        in captured.err
    )


def test_gguf_variant_patterns_bundle_preferred_mmproj(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename = "model-Q4_K_M-00001-of-00002.gguf"),
        SimpleNamespace(rfilename = "model-Q4_K_M-00002-of-00002.gguf"),
        SimpleNamespace(rfilename = "model-Q8_0.gguf"),
        SimpleNamespace(rfilename = "mmproj-Q8_0.gguf"),
        SimpleNamespace(rfilename = "mmproj-F16.gguf"),
    ]

    def fake_model_info(repo_id, token = None, timeout = None, files_metadata = False):
        return SimpleNamespace(siblings = siblings)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)

    patterns = _gguf_variant_patterns("org/vision-gguf", "Q4_K_M", None)

    assert patterns == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "mmproj-F16.gguf",
    ]


def test_gguf_variant_patterns_prefer_f16_when_bf16_listed_first(monkeypatch):
    """Regression: ``"f16" in name.lower()`` matched both F16 and BF16 because
    ``bf16`` lexically contains ``f16``. When the API listed mmproj-BF16
    before mmproj-F16 the worker silently bundled the wrong precision, and
    the matching downloaded-state check then failed for any cache that
    actually contained mmproj-F16."""
    siblings = [
        SimpleNamespace(rfilename = "model-Q4_K_M.gguf"),
        SimpleNamespace(rfilename = "mmproj-BF16.gguf"),
        SimpleNamespace(rfilename = "mmproj-F16.gguf"),
        SimpleNamespace(rfilename = "mmproj-F32.gguf"),
    ]

    def fake_model_info(repo_id, token = None, timeout = None, files_metadata = False):
        return SimpleNamespace(siblings = siblings)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)

    patterns = _gguf_variant_patterns("org/vision-gguf", "Q4_K_M", None)

    assert patterns == [
        "model-Q4_K_M.gguf",
        "mmproj-F16.gguf",
    ]


def test_gguf_variant_patterns_logs_non_f16_mmproj_fallback(monkeypatch, capsys):
    siblings = [
        SimpleNamespace(rfilename = "model-Q4_K_M.gguf"),
        SimpleNamespace(rfilename = "mmproj-Q8_0.gguf"),
    ]

    def fake_model_info(repo_id, token = None, timeout = None, files_metadata = False):
        return SimpleNamespace(siblings = siblings)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)

    patterns = _gguf_variant_patterns("org/vision-gguf", "Q4_K_M", None)
    captured = capsys.readouterr()

    assert patterns == ["model-Q4_K_M.gguf", "mmproj-Q8_0.gguf"]
    assert "No F16 mmproj found for org/vision-gguf" in captured.err
    assert "mmproj-Q8_0.gguf" in captured.err


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


def test_download_registry_pending_cancel_registration_rejects_without_process():
    registry = DownloadRegistry()
    proc = SimpleNamespace()

    assert registry.claim("org/repo::", "http") == (True, "running")
    assert registry.mark_pending_cancel("org/repo::")
    assert not registry.register_process("org/repo::", proc)
    assert registry.get_process("org/repo::") is None
    assert not registry.drop_process("org/repo::", proc)
    assert registry.get_job("org/repo::").state == "cancelled"
    assert registry.claim("org/repo::", "http") == (True, "running")


def test_download_registry_generation_rejects_stale_cancel():
    registry = DownloadRegistry()
    key = "org/repo::"
    proc1 = SimpleNamespace()
    proc2 = SimpleNamespace()

    assert registry.claim(key, "http") == (True, "running")
    generation1 = registry.current_generation(key)
    assert registry.register_process(key, proc1)
    registry.set_job(key, "cancelled")

    assert registry.claim(key, "http") == (True, "running")
    generation2 = registry.current_generation(key)
    assert generation2 != generation1
    assert registry.register_process(key, proc2)

    assert not registry.request_cancel(key, proc2, generation1)
    assert registry.get_job(key).state == "running"
    assert registry.request_cancel(key, proc2, generation2)
    assert registry.get_job(key).state == "cancelling"


def test_download_registry_generation_rejects_stale_pending_cancel():
    registry = DownloadRegistry()
    key = "org/repo::"
    proc = SimpleNamespace()

    assert registry.claim(key, "http") == (True, "running")
    generation1 = registry.current_generation(key)
    registry.set_job(key, "cancelled")

    assert registry.claim(key, "http") == (True, "running")
    generation2 = registry.current_generation(key)
    assert generation2 != generation1

    assert not registry.mark_pending_cancel(key, generation1)
    assert registry.register_process(key, proc)
    assert registry.get_process(key) is proc
    assert registry.get_job(key).state == "running"


def test_resumable_partial_is_limited_to_active_cache_root(monkeypatch, tmp_path):
    active_root = tmp_path / "active"
    legacy_root = tmp_path / "legacy"
    active_root.mkdir()
    legacy_root.mkdir()
    repo_dir_name = "models--org--repo"
    legacy_repo = legacy_root / repo_dir_name
    legacy_blobs = legacy_repo / "blobs"
    legacy_blobs.mkdir(parents = True)
    (legacy_repo / ".transport").write_text("http")
    (legacy_blobs / "etag.incomplete").write_bytes(b"partial")

    monkeypatch.setattr(hf_cache_scan, "_hf_cache_root", lambda **_kwargs: active_root)
    monkeypatch.setattr(
        hf_cache_scan,
        "_hf_cache_roots",
        lambda: [active_root, legacy_root],
    )

    assert hf_cache_scan.has_incomplete_blobs("model", "org/repo")
    assert hf_cache_scan.read_transport_marker("model", "org/repo") == "http"
    assert not hf_cache_scan.has_active_incomplete_blobs("model", "org/repo")
    assert hf_cache_scan.read_active_transport_marker("model", "org/repo") is None
    assert not hf_cache_scan.is_resumable_partial("model", "org/repo")


def test_worker_stderr_excerpt_keeps_prefix_and_suffix():
    data = b"repo phase prefix " + (b"x" * 1200) + b" final traceback suffix"
    excerpt = drain_stderr_excerpt(BytesIO(data), edge_bytes = 64).decode()

    assert excerpt.startswith("repo phase prefix")
    assert "...[stderr truncated]..." in excerpt
    assert excerpt.endswith("final traceback suffix")
