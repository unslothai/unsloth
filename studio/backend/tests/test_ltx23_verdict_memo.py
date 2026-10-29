# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""is_ltx23_checkpoint reads a checkpoint header once per file identity: a load asks twice (download
estimate, then the load), and a GGUF header walk builds a view per tensor."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from core.inference import video_ltx2  # noqa: E402

KEY = "model.diffusion_model.transformer_blocks.0.scale_shift_table"


@pytest.fixture
def reads(monkeypatch):
    monkeypatch.setattr(video_ltx2, "_LTX23_VERDICTS", {}, raising = False)
    calls = []
    headers = {}

    def fake_read(path):
        calls.append(os.fspath(path))
        header = headers.get(Path(path).read_bytes())
        if header is None:
            raise RuntimeError("bad magic")
        return header

    monkeypatch.setattr(video_ltx2, "read_checkpoint_header", fake_read)
    return calls, headers


def test_repeat_probes_read_the_header_once(reads, tmp_path):
    calls, headers = reads
    headers[b"23"] = {"other": (1,), KEY: (4096, 9)}
    path = tmp_path / "ltx.gguf"
    path.write_bytes(b"23")
    assert [video_ltx2.is_ltx23_checkpoint(p) for p in (path, str(path), path)] == [True] * 3
    assert len(calls) == 1


def test_a_rewritten_or_replaced_file_is_read_again(reads, tmp_path):
    calls, headers = reads
    headers[b"23"] = {KEY: (4096, 9)}
    headers[b"20_"] = {KEY: (4096, 6)}
    path = tmp_path / "ltx.gguf"
    path.write_bytes(b"23")
    assert video_ltx2.is_ltx23_checkpoint(path) is True
    path.write_bytes(b"20_")
    assert video_ltx2.is_ltx23_checkpoint(path) is False
    other = tmp_path / "other.gguf"
    other.write_bytes(b"23")
    os.replace(other, path)
    assert video_ltx2.is_ltx23_checkpoint(path) is True
    assert len(calls) == 3


def test_failures_are_not_kept(reads, tmp_path):
    calls, headers = reads
    path = tmp_path / "ltx.gguf"
    path.write_bytes(b"23")
    assert video_ltx2.is_ltx23_checkpoint(path) is False
    headers[b"23"] = {KEY: (9, 4096)}
    assert video_ltx2.is_ltx23_checkpoint(path) is True
    assert len(calls) == 2


def test_a_missing_path_is_not_cached_and_keeps_the_warning(reads, tmp_path):
    calls, _ = reads
    missing = tmp_path / "missing.gguf"
    assert video_ltx2.is_ltx23_checkpoint(missing) is False
    assert video_ltx2.is_ltx23_checkpoint(missing) is False
    assert len(calls) == 2
    assert video_ltx2._LTX23_VERDICTS == {}


def test_a_symlink_shares_its_targets_verdict(reads, tmp_path):
    calls, headers = reads
    headers[b"23"] = {KEY: (9, 4096)}
    blob = tmp_path / "blob"
    blob.write_bytes(b"23")
    link = tmp_path / "ltx.safetensors"
    try:
        link.symlink_to(blob)
    except (OSError, NotImplementedError):
        pytest.skip("os.symlink unavailable here (e.g. Windows without Developer Mode)")
    assert video_ltx2.is_ltx23_checkpoint(link) is True
    assert video_ltx2.is_ltx23_checkpoint(blob) is True
    assert len(calls) == 1


def test_first_matching_table_decides_and_no_table_is_false(reads, tmp_path):
    calls, headers = reads
    headers[b"a"] = {"x." + KEY: (6, 1), "y." + KEY: (9, 1)}
    headers[b"bb"] = {"unrelated": (9,)}
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"a")
    b.write_bytes(b"bb")
    for _ in range(2):
        assert video_ltx2.is_ltx23_checkpoint(a) is False
        assert video_ltx2.is_ltx23_checkpoint(b) is False
    assert len(calls) == 2


def test_the_cache_is_bounded(reads, tmp_path, monkeypatch):
    calls, headers = reads
    monkeypatch.setattr(video_ltx2, "_LTX23_VERDICTS_MAX", 4, raising = False)
    headers[b"23"] = {KEY: (9, 1)}
    for i in range(10):
        p = tmp_path / f"{i}.gguf"
        p.write_bytes(b"23")
        assert video_ltx2.is_ltx23_checkpoint(p) is True
        assert len(video_ltx2._LTX23_VERDICTS) <= 4


def test_real_gguf_and_safetensors_headers(tmp_path, monkeypatch):
    """The real readers, counted: a second probe of either format reads nothing."""
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    monkeypatch.setattr(video_ltx2, "_LTX23_VERDICTS", {}, raising = False)
    tensors = {
        KEY: np.zeros((9, 8), dtype = np.float16),
        "blocks.0.w": np.zeros((4, 4), dtype = np.float16),
    }
    st = tmp_path / "ltx.safetensors"
    safetensors_numpy.save_file(tensors, str(st))
    gg = tmp_path / "ltx.gguf"
    writer = gguf.GGUFWriter(str(gg), "ltxv")
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    real = video_ltx2.read_checkpoint_header
    calls = []
    monkeypatch.setattr(video_ltx2, "read_checkpoint_header", lambda p: calls.append(p) or real(p))
    for _ in range(3):
        assert video_ltx2.is_ltx23_checkpoint(st) is True
        assert video_ltx2.is_ltx23_checkpoint(gg) is True
    assert len(calls) == 2
