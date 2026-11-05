# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/models/local keeps per-file and per-name verdicts across requests: the listing must stay
identical to an uncached scan, notice every add / remove / rewrite, and skip repeated work."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

import routes.models as models_route
from core.inference import diffusion_families, video_families
from hub.utils import download_manifest
from utils import audio_tokens
from utils.scan_memo import scan_memo, scan_memo_active

_CSM_TOKENS = {"1": {"content": "<|AUDIO|>"}, "2": {"content": "<|audio_eos|>"}}
_PLAIN_TOKENS = {"1": {"content": "<|PLAIN|>"}, "2": {"content": "<|plain_eos|>"}}


def _clear_cross_request_caches():
    audio_tokens._TOKENIZER_AUDIO_TYPE_CACHE.clear()
    diffusion_families._detect_family_memo_state = ((), {})
    video_families._detect_video_family_memo_state = ((), (), {})


@pytest.fixture(autouse = True)
def _fresh_caches():
    _clear_cross_request_caches()
    yield
    _clear_cross_request_caches()


def _write_tokenizer(path: Path, tokens: dict) -> None:
    path.write_text(json.dumps({"added_tokens_decoder": tokens}), encoding = "utf-8")


def _make_repo(hub: Path, repo_id: str, files: dict[str, str]) -> Path:
    snapshot = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot.parent.parent / "refs").mkdir()
    (snapshot.parent.parent / "refs" / "main").write_text("rev", encoding = "utf-8")
    for name, body in files.items():
        target = snapshot / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(body, encoding = "utf-8")
    return snapshot


@pytest.fixture
def hub(monkeypatch, tmp_path):
    active = tmp_path / "hub"
    active.mkdir()
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.hermes_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.ollama_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.omlx_model_dirs", lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])
    config = json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    _make_repo(active, "org/plain-chat", {"config.json": config, "model.safetensors": "x"})
    speech = _make_repo(active, "org/speech-csm", {"config.json": config, "model.safetensors": "x"})
    _write_tokenizer(speech / "tokenizer_config.json", _CSM_TOKENS)
    _make_repo(
        active,
        "org/flux.1-dev-copy",
        {"model_index.json": json.dumps({"_class_name": "FluxPipeline"}), "vae/config.json": "{}"},
    )
    _make_repo(active, "org/wan2.1-t2v-1.3b-diffusers", {"model_index.json": "{}"})
    _make_repo(active, "org/tiny-GGUF", {"tiny-Q4_K_M.gguf": "GGUF"})
    return active


def _scan(models_root: Path) -> list[dict]:
    rows = asyncio.run(models_route._shared_compat_local_inventory_scan(models_root))
    return [row.model_dump(mode = "json") for row in rows]


def _cold_scan(models_root: Path) -> list[dict]:
    _clear_cross_request_caches()
    return _scan(models_root)


def _row(rows: list[dict], model_id: str) -> dict:
    return next(row for row in rows if row["model_id"] == model_id)


def _speech_tokenizer(hub: Path) -> Path:
    return hub / "models--org--speech-csm" / "snapshots" / "rev" / "tokenizer_config.json"


def test_repeat_scan_is_identical_to_a_cold_scan(hub, tmp_path):
    first = _scan(tmp_path / "models")
    second = _scan(tmp_path / "models")
    assert first == second == _cold_scan(tmp_path / "models")
    assert _row(first, "org/speech-csm")["audio_type"] == "csm"
    assert _row(first, "org/speech-csm")["task"] == "text-to-speech"
    assert _row(first, "org/plain-chat")["audio_type"] is None


def test_second_scan_reads_no_tokenizer_config(hub, tmp_path, monkeypatch):
    reads: list[str] = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    _scan(tmp_path / "models")
    assert reads.count("tokenizer_config.json") >= 1
    reads.clear()
    _scan(tmp_path / "models")
    assert reads.count("tokenizer_config.json") == 0


def test_rewrite_with_same_size_and_new_mtime_is_reread(hub, tmp_path):
    tokenizer = _speech_tokenizer(hub)
    before = _scan(tmp_path / "models")
    assert _row(before, "org/speech-csm")["audio_type"] == "csm"

    old = tokenizer.stat()
    _write_tokenizer(tokenizer, _PLAIN_TOKENS)
    assert tokenizer.stat().st_size == old.st_size
    os.utime(tokenizer, ns = (old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))

    after = _scan(tmp_path / "models")
    assert _row(after, "org/speech-csm")["audio_type"] is None
    assert after == _cold_scan(tmp_path / "models")


@pytest.mark.skipif(sys.platform == "win32", reason = "ctime is creation time on Windows")
def test_rewrite_that_restores_size_and_mtime_is_still_reread(hub, tmp_path):
    tokenizer = _speech_tokenizer(hub)
    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] == "csm"

    old = tokenizer.stat()
    _write_tokenizer(tokenizer, _PLAIN_TOKENS)
    os.utime(tokenizer, ns = (old.st_atime_ns, old.st_mtime_ns))
    new = tokenizer.stat()
    assert (new.st_size, new.st_mtime_ns) == (old.st_size, old.st_mtime_ns)

    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] is None


def test_failed_tokenizer_read_is_retried_on_the_next_scan(hub, tmp_path, monkeypatch):
    real_read_text = Path.read_text
    failures = [OSError(errno.EMFILE, "Too many open files")]

    def flaky_read_text(self, *args, **kwargs):
        if self.name == "tokenizer_config.json" and failures:
            raise failures.pop()
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] is None
    # The file never changed, so its stat key is the same: the failed read must not be what it answers.
    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] == "csm"


def test_added_and_removed_models_show_up_on_the_next_scan(hub, tmp_path):
    before = _scan(tmp_path / "models")
    assert "org/speech-csm" in {row["model_id"] for row in before}

    shutil.rmtree(hub / "models--org--speech-csm")
    added = _make_repo(
        hub,
        "org/second-speech",
        {"config.json": json.dumps({"model_type": "llama"}), "model.safetensors": "x"},
    )
    _write_tokenizer(added / "tokenizer_config.json", _CSM_TOKENS)

    after = _scan(tmp_path / "models")
    ids = {row["model_id"] for row in after}
    assert "org/speech-csm" not in ids
    assert _row(after, "org/second-speech")["audio_type"] == "csm"
    assert after == _cold_scan(tmp_path / "models")


def test_removed_tokenizer_config_drops_the_audio_type(hub, tmp_path):
    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] == "csm"
    _speech_tokenizer(hub).unlink()
    assert _row(_scan(tmp_path / "models"), "org/speech-csm")["audio_type"] is None


def test_cancel_marker_written_between_scans_marks_the_row_partial(hub, tmp_path):
    assert _row(_scan(tmp_path / "models"), "org/plain-chat")["partial"] is False
    assert download_manifest.write_cancel_marker("model", "org/plain-chat", None, hub_cache = hub)
    after = _scan(tmp_path / "models")
    assert _row(after, "org/plain-chat")["partial"] is True
    assert after == _cold_scan(tmp_path / "models")
    download_manifest.clear_cancel_marker("model", "org/plain-chat", None, hub_cache = hub)
    assert _row(_scan(tmp_path / "models"), "org/plain-chat")["partial"] is False


def test_state_probes_inside_a_scan_skip_absent_files(hub, tmp_path, monkeypatch):
    assert download_manifest.write_cancel_marker("model", "org/plain-chat", None, hub_cache = hub)
    probed: list[Path] = []
    real_is_file = Path.is_file

    def counting_is_file(self, *args, **kwargs):
        probed.append(self)
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    with scan_memo():
        assert scan_memo_active()
        assert download_manifest.has_cancel_marker("model", "org/plain-chat", hub_cache = hub)
        assert not download_manifest.has_cancel_marker("model", "org/speech-csm", hub_cache = hub)
    # Only the marker that exists is stat'ed; every absent candidate was answered by the listing.
    assert [path.name for path in probed] == [
        download_manifest.marker_path("model", "org/plain-chat", None, hub_cache = hub).name
    ]
    assert not scan_memo_active()

    probed.clear()
    assert not download_manifest.has_cancel_marker("model", "org/speech-csm", hub_cache = hub)
    assert len(probed) > 1  # outside a scan every candidate is probed as before


def test_state_listing_falls_back_to_a_probe_on_a_case_variant(tmp_path):
    directory = tmp_path / "scope"
    directory.mkdir()
    (directory / "Models--Org--Repo.json").write_text("{}", encoding = "utf-8")
    with scan_memo():
        assert download_manifest._state_file_may_exist(directory / "models--org--repo.json")
        assert not download_manifest._state_file_may_exist(directory / "models--org--other.json")
        assert not download_manifest._state_file_may_exist(tmp_path / "missing" / "x.json")


@pytest.mark.parametrize(
    "name",
    [
        "black-forest-labs/FLUX.1-dev",
        "Qwen/Qwen-Image-Edit-2511",
        "org/flux.3-klein-pup",
        "/models/models--org--sdxl-base/snapshots/abc/unet",
        "unsloth/Llama-3.2-1B-Instruct",
        "",
    ],
)
def test_family_memo_matches_the_uncached_detector(name):
    assert diffusion_families.detect_family(name) is diffusion_families._detect_family_by_name(name)
    assert diffusion_families.detect_family(name) is diffusion_families._detect_family_by_name(name)
    assert video_families.detect_video_family(name) is video_families._detect_video_family_by_name(
        name
    )
    assert video_families.detect_video_family(name) is video_families._detect_video_family_by_name(
        name
    )


def test_family_memo_follows_a_replaced_table(monkeypatch):
    name = "black-forest-labs/FLUX.1-dev"
    assert diffusion_families.detect_family(name) is not None
    monkeypatch.setattr(diffusion_families, "_FAMILIES", ())
    assert diffusion_families.detect_family(name) is None

    video = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert video_families.detect_video_family(video) is not None
    monkeypatch.setattr(video_families, "_FAMILIES", ())
    assert video_families.detect_video_family(video) is None
