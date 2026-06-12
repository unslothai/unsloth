# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Tests for model-update detection and the /update endpoint.

Covers:
  * get_gguf_variants degrades to update_available=False when the remote
    update check fails (offline / rate-limit / gated), instead of 500ing.
  * get_gguf_variants reports update_available correctly on success.
  * update_hf_model's bicodec branch passes a valid snapshot_download kwarg.
"""
import asyncio
import sys
import types
from types import SimpleNamespace

if "structlog" not in sys.modules:
    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger=_DummyLogger, get_logger=lambda *a, **k: _DummyLogger())

import pytest
import routes.models as M
from models.models import UpdateRequest


def _variants():
    return [
        SimpleNamespace(filename="model-Q4_K_M.gguf", quant="Q4_K_M", size_bytes=1000),
        SimpleNamespace(filename="model-Q8_0.gguf", quant="Q8_0", size_bytes=2000),
    ]


def _seed_cache(tmp_path, repo_id, blob_ids, gguf_files):
    repo = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    for name, size in gguf_files.items():
        (snap / name).write_bytes(b"\0" * size)
    blobs = repo / "blobs"
    blobs.mkdir()
    for b in blob_ids:
        (blobs / b).write_bytes(b"x")
    return repo


@pytest.fixture
def patch_hf(monkeypatch):
    """Patch the huggingface_hub bits get_gguf_variants pulls in."""
    def _apply(tmp_path, *, paths_info_impl):
        import huggingface_hub as hf
        monkeypatch.setattr(hf.constants, "HF_HUB_CACHE", str(tmp_path), raising=False)
        monkeypatch.setattr(
            M, "list_gguf_variants",
            lambda r, hf_token=None: (_variants(), False), raising=True)
        monkeypatch.setattr(M, "is_local_path", lambda p: False, raising=False)
        monkeypatch.setattr(hf, "try_to_load_from_cache", lambda **k: None, raising=False)
        monkeypatch.setattr(hf, "get_paths_info", paths_info_impl, raising=False)
    return _apply


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_update_check_failure_does_not_500(tmp_path, patch_hf):
    """A failed update check must not break the variant listing."""
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids=["oldsha"], gguf_files={"model-Q4_K_M.gguf": 1000})

    def boom(**kwargs):
        raise RuntimeError("429 Too Many Requests / offline")

    patch_hf(tmp_path, paths_info_impl=boom)
    resp = _call(M.get_gguf_variants(repo_id=repo, hf_token=None, current_subject="t"))
    assert len(resp.variants) == 2
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.downloaded is True
    assert q4.update_available is False


def test_update_check_detects_update(tmp_path, patch_hf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids=["oldsha"], gguf_files={"model-Q4_K_M.gguf": 1000})

    def ok(repo_id, paths, token=None):
        return [SimpleNamespace(path="model-Q4_K_M.gguf",
                                lfs=SimpleNamespace(sha256="NEWsha"), blob_id=None)]

    patch_hf(tmp_path, paths_info_impl=ok)
    resp = _call(M.get_gguf_variants(repo_id=repo, hf_token=None, current_subject="t"))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is True


def test_update_check_no_update_when_blob_matches(tmp_path, patch_hf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids=["samesha"], gguf_files={"model-Q4_K_M.gguf": 1000})

    def ok(repo_id, paths, token=None):
        return [SimpleNamespace(path="model-Q4_K_M.gguf",
                                lfs=SimpleNamespace(sha256="samesha"), blob_id=None)]

    patch_hf(tmp_path, paths_info_impl=ok)
    resp = _call(M.get_gguf_variants(repo_id=repo, hf_token=None, current_subject="t"))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is False


def test_bicodec_update_uses_valid_snapshot_download_kwarg(monkeypatch):
    """The bicodec branch must call snapshot_download with local_dir."""
    import inspect
    captured = {}

    fake_cfg = SimpleNamespace(
        is_local=False, is_gguf=False, is_audio=True, audio_type="bicodec",
        is_lora=False, base_model=None, path="unsloth/orpheus-3b-0.1-ft",
        is_vision=False, gguf_hf_repo=None, gguf_variant=None)
    monkeypatch.setattr(M.ModelConfig, "from_identifier",
                        classmethod(lambda cls, **k: fake_cfg))

    import routes.inference as I
    monkeypatch.setattr(I, "get_llama_cpp_backend",
                        lambda: SimpleNamespace(_cancel_event=SimpleNamespace(
                            is_set=lambda: False, set=lambda: None, clear=lambda: None)))

    import huggingface_hub as hf
    real_params = set(inspect.signature(hf.snapshot_download).parameters)

    def fake_sd(**kwargs):
        bad = set(kwargs) - real_params
        if bad:
            raise TypeError(
                f"snapshot_download() got an unexpected keyword argument {sorted(bad)[0]!r}")
        captured.update(kwargs)
        return "/fake/path"
    monkeypatch.setattr(hf, "snapshot_download", fake_sd, raising=True)

    req = UpdateRequest(repo_id="unsloth/orpheus-3b-0.1-ft", hf_token=None, gguf_variant=None)
    resp = _call(M.update_hf_model(request=req, current_subject="t"))
    assert resp.model_path == "/fake/path"
    assert "local_dir" in captured and "local_path" not in captured
