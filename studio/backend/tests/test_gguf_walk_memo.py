# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF string array is walked once per process: later walks of the same unchanged file seek
to the remembered end. Every way the file can change must force a fresh walk."""

from __future__ import annotations

import io
import os
import struct
import threading
from types import SimpleNamespace

import pytest

import utils.models.gguf_metadata as gm
from core.inference.llama_cpp import LlamaCppBackend
from utils.models.gguf_metadata import skip_gguf_strings, take_gguf_strings_by_first_byte

_MARKUP = frozenset(b"<[")


def _entry(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def _array(entries: list[bytes]) -> bytes:
    return b"HEAD" + b"".join(_entry(e) for e in entries) + b"TAIL"


def _entries(n: int, marker: bytes = b"<|im_start|>") -> list[bytes]:
    return [marker if i % 97 == 0 else b"tok%d" % i for i in range(n)]


@pytest.fixture(autouse = True)
def _fresh_memo(monkeypatch):
    memo = getattr(gm, "_WALK_MEMO", {})
    memo.clear()
    # Files written by the test are seconds old; let them be remembered unless a test says otherwise.
    monkeypatch.setattr(gm, "_WALK_MEMO_SETTLE_NS", 0, raising = False)
    yield
    memo.clear()


@pytest.fixture
def walks(monkeypatch):
    counter = {"skip": 0, "take": 0}
    skip, take = gm._skip_gguf_strings_walk, gm._take_gguf_strings_walk

    def counted_skip(f, count):
        counter["skip"] += 1
        return skip(f, count)

    def counted_take(f, count, first_bytes):
        counter["take"] += 1
        return take(f, count, first_bytes)

    monkeypatch.setattr(gm, "_skip_gguf_strings_walk", counted_skip)
    monkeypatch.setattr(gm, "_take_gguf_strings_walk", counted_take)
    return counter


def _skip(path, count: int):
    with open(path, "rb") as f:
        f.seek(4)
        done = skip_gguf_strings(f, count)
        return done, f.tell()


def _take(path, count: int):
    with open(path, "rb") as f:
        f.seek(4)
        done, kept = take_gguf_strings_by_first_byte(f, count, _MARKUP)
        return done, kept, f.tell()


def _make_old(path, seconds: int = 60):
    st = os.stat(path)
    os.utime(path, ns = (st.st_atime_ns, st.st_mtime_ns - seconds * 10**9))


def test_second_walk_of_an_unchanged_file_reads_no_entries(tmp_path, monkeypatch):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(5000)))
    first = _skip(path, 5000)
    unpacked = []
    real = gm._U64_AT
    monkeypatch.setattr(gm, "_U64_AT", lambda *a: unpacked.append(1) or real(*a))
    assert _skip(path, 5000) == first
    assert unpacked == []


def test_hits_return_what_a_fresh_walk_returns(tmp_path, walks):
    entries = _entries(20000) + [b"<" + b"z" * (gm._STRING_SCAN_MAX + 1), b"[x]"]
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(entries))
    take1 = _take(path, len(entries))
    skip1 = _skip(path, len(entries))
    take2 = _take(path, len(entries))
    skip2 = _skip(path, len(entries))
    assert take1 == take2 and skip1 == skip2
    # Stopped at the long entry, so a skip result is not derived from the take: two walks, then none.
    assert walks == {"skip": 1, "take": 1}
    assert take1[0] == skip1[0] == 20000
    gm._WALK_MEMO.clear()
    assert _take(path, len(entries)) == take1 and _skip(path, len(entries)) == skip1


def test_a_finished_take_answers_the_skip_of_the_same_array(tmp_path, walks):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    done, kept, end = _take(path, 3000)
    assert (done, end) == _skip(path, 3000)
    assert walks == {"skip": 0, "take": 1}


def test_take_hit_returns_a_new_list(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _, kept1, _ = _take(path, 3000)
    kept1.append(b"<mutated>")
    _, kept2, _ = _take(path, 3000)
    assert b"<mutated>" not in kept2


@pytest.mark.parametrize("count", [0, 1, 2999, 3000])
def test_count_and_offset_are_part_of_the_key(tmp_path, count):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _skip(path, 3000)
    with open(path, "rb") as f:
        f.seek(4)
        assert skip_gguf_strings(f, count) == count
        expected = 4 + sum(len(_entry(e)) for e in _entries(3000)[:count])
        assert f.tell() == expected


def test_same_size_rewrite_with_the_old_mtime_restored_is_walked_again(tmp_path, walks):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _make_old(path)
    old = os.stat(path)
    assert _skip(path, 3000)[0] == 3000
    # Same size, same mtime (cp -p / rsync -t / tar style), different lengths inside.
    size = old.st_size
    rewritten = b"HEAD" + struct.pack("<Q", gm._STRING_SCAN_MAX + 10)
    rewritten += b"\0" * (size - len(rewritten))
    with open(path, "r+b") as f:
        f.write(rewritten)
    os.utime(path, ns = (old.st_atime_ns, old.st_mtime_ns))
    new = os.stat(path)
    assert (new.st_size, new.st_mtime_ns, new.st_ino) == (old.st_size, old.st_mtime_ns, old.st_ino)
    if new.st_ctime_ns == old.st_ctime_ns:
        pytest.skip("filesystem did not move ctime")
    assert _skip(path, 3000)[0] == 0
    assert walks["skip"] == 2


def test_a_recently_changed_file_is_not_remembered(tmp_path, monkeypatch, walks):
    monkeypatch.setattr(gm, "_WALK_MEMO_SETTLE_NS", 3_000_000_000)
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _skip(path, 3000)
    _skip(path, 3000)
    assert walks["skip"] == 2
    assert not gm._WALK_MEMO
    # Older than the settle window (mtime and ctime), it is remembered.
    monkeypatch.setattr(gm.time, "time_ns", lambda: os.stat(path).st_ctime_ns + 3_000_000_001)
    _skip(path, 3000)
    _skip(path, 3000)
    assert walks["skip"] == 3


def test_growing_file_is_walked_again(tmp_path, walks):
    path = tmp_path / "m.gguf"
    data = _array(_entries(3000))
    path.write_bytes(data[: len(data) // 2])
    first = _skip(path, 3000)
    assert first[0] < 3000
    with open(path, "ab") as f:
        f.write(data[len(data) // 2 :])
    second = _skip(path, 3000)
    assert second[0] == 3000
    assert walks["skip"] == 2


def test_zero_inode_is_never_remembered(tmp_path, monkeypatch, walks):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    real_fstat = os.fstat

    def fstat_without_inode(fd):
        st = real_fstat(fd)
        fields = {n: getattr(st, n) for n in dir(st) if n.startswith("st_")}
        fields["st_ino"] = 0
        return SimpleNamespace(**fields)

    monkeypatch.setattr(gm.os, "fstat", fstat_without_inode)
    _skip(path, 3000)
    _skip(path, 3000)
    assert walks["skip"] == 2
    assert not gm._WALK_MEMO


def test_file_changed_during_the_walk_is_not_remembered(tmp_path, monkeypatch, walks):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    real = gm._walk_file_state
    calls = []

    def state_changes_after_first(f):
        st = real(f)
        calls.append(1)
        return st if len(calls) % 2 else st[:2] + (st[2] + 1,) + st[3:]

    monkeypatch.setattr(gm, "_walk_file_state", state_changes_after_first)
    _skip(path, 3000)
    assert not gm._WALK_MEMO


def test_streams_without_a_real_file_are_not_remembered(walks):
    data = _array(_entries(3000))
    for _ in range(2):
        f = io.BytesIO(data)
        f.seek(4)
        assert skip_gguf_strings(f, 3000) == 3000
    assert walks["skip"] == 2
    assert not gm._WALK_MEMO


def test_memo_is_bounded_lru(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "_WALK_MEMO_MAX_ENTRIES", 4)
    paths = []
    for i in range(6):
        p = tmp_path / f"m{i}.gguf"
        p.write_bytes(_array(_entries(100 + i)))
        paths.append(p)
    for i, p in enumerate(paths[:4]):
        _skip(p, 100 + i)
    _skip(paths[0], 100)  # refresh: most recently used
    _skip(paths[4], 104)
    _skip(paths[5], 105)
    assert len(gm._WALK_MEMO) == 4
    inodes = {k[1] for k in gm._WALK_MEMO}
    assert os.stat(paths[0]).st_ino in inodes
    assert os.stat(paths[1]).st_ino not in inodes


def test_large_kept_lists_are_not_remembered(tmp_path, monkeypatch, walks):
    monkeypatch.setattr(gm, "_WALK_MEMO_MAX_KEPT_BYTES", 10)
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    assert _take(path, 3000) == _take(path, 3000)
    assert walks["take"] == 2


def test_concurrent_walks_agree(tmp_path):
    entries = _entries(30000)
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(entries))
    expected_end = len(_array(entries)) - 4
    results, errors = [], []

    def worker():
        try:
            for _ in range(5):
                results.append(_skip(path, 30000))
                results.append(_take(path, 30000)[::2])
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert set(results) == {(30000, expected_end)}


def test_llama_reader_and_metadata_reader_share_the_walk(tmp_path, walks):
    """A cold load reads the header with both modules; the tokens array is walked once."""
    tokens = _entries(5000)
    kv = [
        (b"general.architecture", 8, _entry(b"llama")),
        (
            b"tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, len(tokens)) + b"".join(_entry(t) for t in tokens),
        ),
        (b"general.name", 8, _entry(b"x")),
    ]
    body = struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv))
    for k, vt, v in kv:
        body += _entry(k) + struct.pack("<I", vt) + v
    path = tmp_path / "m.gguf"
    path.write_bytes(body)
    for _ in range(3):
        probe = object.__new__(LlamaCppBackend)
        probe._model_identifier = "t"
        probe._read_gguf_metadata(str(path))
        assert probe._architecture == "llama"
    gm.read_gguf_general_metadata(str(path))
    gm._parse_gguf_has_mrope_sections(str(path))
    assert walks == {"skip": 0, "take": 1}


def test_rename_replacement_with_the_old_mtime_is_walked_again(tmp_path, walks):
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _make_old(path)
    old = os.stat(path)
    assert _skip(path, 3000)[0] == 3000
    # A download or copy lands next to the file and is renamed over it, timestamps preserved.
    staged = tmp_path / "m.gguf.incomplete"
    rewritten = b"HEAD" + struct.pack("<Q", gm._STRING_SCAN_MAX + 10)
    staged.write_bytes(rewritten + b"\0" * (old.st_size - len(rewritten)))
    os.utime(staged, ns = (old.st_atime_ns, old.st_mtime_ns))
    os.replace(staged, path)
    assert _skip(path, 3000)[0] == 0
    assert walks["skip"] == 2


def test_in_place_rewrite_is_walked_again_when_ctime_is_creation_time(tmp_path, monkeypatch, walks):
    """Windows reports the creation time as st_ctime, so only mtime moves on a rewrite."""
    path = tmp_path / "m.gguf"
    path.write_bytes(_array(_entries(3000)))
    _make_old(path)
    created = os.stat(path).st_ctime_ns - 60 * 10**9
    real_fstat = os.fstat

    def fstat_with_creation_ctime(fd):
        st = real_fstat(fd)
        fields = {n: getattr(st, n) for n in dir(st) if n.startswith("st_")}
        fields["st_ctime_ns"] = created
        return SimpleNamespace(**fields)

    monkeypatch.setattr(gm.os, "fstat", fstat_with_creation_ctime)
    assert _skip(path, 3000)[0] == 3000
    size = os.stat(path).st_size
    rewritten = b"HEAD" + struct.pack("<Q", gm._STRING_SCAN_MAX + 10)
    with open(path, "r+b") as f:
        f.write(rewritten + b"\0" * (size - len(rewritten)))
    assert _skip(path, 3000)[0] == 0
    assert walks["skip"] == 2
