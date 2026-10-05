# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Warming the base repo's text encoder shards in the page cache beside the LTX-2.3 checkpoint read."""

import json
import threading

import pytest

from core.inference import video_ltx2


def _wait_for_prefetch_threads(timeout = 10.0):
    for t in [t for t in threading.enumerate() if t.name.startswith("unsloth-prefetch")]:
        t.join(timeout)
        assert not t.is_alive(), "prefetch threads never finished"


def _fake_cache(
    tmp_path, shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
):
    commit = "a" * 40
    repo = tmp_path / "models--Lightricks--LTX-2"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(commit)
    folder = repo / "snapshots" / commit / "text_encoder"
    folder.mkdir(parents = True)
    weight_map = {f"w{i}": name for i, name in enumerate(shards)}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in shards:
        (folder / name).write_bytes(b"x" * 4096 * 3)
    return folder


def test_prefetch_has_a_kill_switch(monkeypatch):
    monkeypatch.delenv(video_ltx2.PREFETCH_ENV, raising = False)
    assert video_ltx2.prefetch_enabled()
    for off in ("0", "off", "false", "no"):
        monkeypatch.setenv(video_ltx2.PREFETCH_ENV, off)
        assert not video_ltx2.prefetch_enabled()


def test_text_encoder_files_come_from_the_cached_index_only(tmp_path):
    folder = _fake_cache(tmp_path)
    files = video_ltx2._text_encoder_files("Lightricks/LTX-2", str(tmp_path))
    assert files == sorted(str(p) for p in folder.glob("model-*.safetensors"))
    # A shard the index names but the cache lacks is left to the loader; an uncached repo yields nothing.
    (folder / "model-00002-of-00002.safetensors").unlink()
    assert video_ltx2._text_encoder_files("Lightricks/LTX-2", str(tmp_path)) == files[:1]
    assert video_ltx2._text_encoder_files("Lightricks/LTX-2.3", str(tmp_path)) == []


def test_uncached_bytes_reads_zero_once_a_file_was_read(tmp_path):
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"y" * (1 << 20))
    path.read_bytes()
    assert video_ltx2._uncached_bytes(str(path)) == 0
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert video_ltx2._uncached_bytes(str(empty)) == 0


def _pretend_cold(monkeypatch, available_mib = 1 << 30):
    from core.inference import diffusion_memory
    monkeypatch.setattr(video_ltx2, "_uncached_bytes", lambda path: 2 << 30)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: available_mib)


def test_prefetch_reads_every_uncached_file(tmp_path, monkeypatch):
    folder = _fake_cache(tmp_path)
    files = video_ltx2._text_encoder_files("Lightricks/LTX-2", str(tmp_path))
    _pretend_cold(monkeypatch)
    opened = []
    real_open = open

    def _open(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(video_ltx2, "open", _open, raising = False)
    stop = video_ltx2.start_prefetch(files)
    assert stop is not None
    _wait_for_prefetch_threads()
    assert sorted(set(opened)) == files
    assert folder.exists()


def test_prefetch_declines_a_warm_cache_or_a_small_host(tmp_path, monkeypatch):
    _fake_cache(tmp_path)
    files = video_ltx2._text_encoder_files("Lightricks/LTX-2", str(tmp_path))
    monkeypatch.setattr(video_ltx2, "_uncached_bytes", lambda path: 0)
    assert video_ltx2.start_prefetch(files) is None
    # 4 GiB to warm against 6 GiB available: it would push other pages out, so it is left to the loader.
    _pretend_cold(monkeypatch, available_mib = 6 << 10)
    assert video_ltx2.start_prefetch(files) is None
    _pretend_cold(monkeypatch, available_mib = None)
    assert video_ltx2.start_prefetch(files) is None
    assert video_ltx2.start_prefetch([]) is None


@pytest.mark.parametrize("fails", [False, True])
def test_the_pipeline_load_stops_its_prefetch(monkeypatch, fails):
    stop = threading.Event()
    started = []
    monkeypatch.delenv(video_ltx2.PREFETCH_ENV, raising = False)
    monkeypatch.setattr(video_ltx2, "_text_encoder_files", lambda repo, cache: ["a", "b"])
    monkeypatch.setattr(video_ltx2, "_live_cache_dir", lambda: "cache")
    monkeypatch.setattr(video_ltx2, "start_prefetch", lambda paths: started.append(paths) or stop)

    def _assemble(checkpoint_path, **kwargs):
        assert started == [["a", "b"]] and not stop.is_set()
        if fails:
            raise RuntimeError("CUDA out of memory")
        return "pipe"

    monkeypatch.setattr(video_ltx2, "_assemble_ltx23_pipeline", _assemble)
    if fails:
        with pytest.raises(RuntimeError):
            video_ltx2.load_ltx23_pipeline("ltx.safetensors", base_repo = "Lightricks/LTX-2")
    else:
        assert (
            video_ltx2.load_ltx23_pipeline("ltx.safetensors", base_repo = "Lightricks/LTX-2")
            == "pipe"
        )
    assert stop.is_set()


def test_a_supplied_encoder_needs_no_prefetch(monkeypatch):
    calls = []
    monkeypatch.setattr(video_ltx2, "start_prefetch", lambda paths: calls.append(paths))
    monkeypatch.setattr(video_ltx2, "_assemble_ltx23_pipeline", lambda path, **kwargs: kwargs)
    out = video_ltx2.load_ltx23_pipeline(
        "ltx.safetensors", base_repo = "Lightricks/LTX-2", text_encoder = "fp8", is_gguf = False
    )
    assert calls == [] and out["text_encoder"] == "fp8" and out["is_gguf"] is False
