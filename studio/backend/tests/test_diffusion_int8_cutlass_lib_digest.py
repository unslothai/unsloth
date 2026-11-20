# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUTLASS race cache key hashes the prebuilt library once per process, not once per weight shape.

The plan asks for the key once per int8 weight shape (39 times in a FLUX.3 klein INT8 load), and each ask read and
hashed the whole library again: 2.9 to 3.3 s of every L4 load. Runs anywhere: the library is a temp file and the
card is faked."""

from __future__ import annotations

import builtins
import os

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


def _new_process():
    memo = getattr(g8, "_sources_digest", None)
    if memo is not None:
        memo.cache_clear()


@pytest.fixture
def lib(tmp_path, monkeypatch):
    path = tmp_path / "wheel" / "libunsloth_i8gemm.so"
    path.parent.mkdir()
    path.write_bytes(b"\x7fELF" + os.urandom(1 << 20))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA L4")
    monkeypatch.setattr(cut, "library_path", lambda: str(path))
    _new_process()
    yield path
    _new_process()


def _reads(monkeypatch, path):
    seen = []
    real = builtins.open

    def counting(file, *a, **k):
        if isinstance(file, (str, os.PathLike)) and os.fspath(file) == os.fspath(path):
            seen.append(1)
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", counting)
    return seen


def test_the_library_is_read_once_per_process_not_per_shape(lib, monkeypatch):
    reads = _reads(monkeypatch, lib)
    keys = {g8._cut_cache_key(0) for _ in range(39)}
    assert len(keys) == 1 and None not in keys
    assert len(reads) == 1


def test_a_rebuilt_library_changes_the_next_process_key(lib):
    before = g8._cut_cache_key(0)
    lib.write_bytes(b"\x7fELF" + os.urandom(1 << 20))
    _new_process()
    after = g8._cut_cache_key(0)
    assert before and after and before != after


def test_the_same_build_reinstalled_keeps_its_verdicts(lib):
    before = g8._cut_cache_key(0)
    data = lib.read_bytes()
    lib.unlink()
    lib.write_bytes(data)
    _new_process()
    assert g8._cut_cache_key(0) == before


def test_no_library_still_keys_on_the_sources(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA L4")
    monkeypatch.setattr(cut, "library_path", lambda: None)
    _new_process()
    assert g8._cut_cache_key(0)


def test_the_key_hashes_the_same_bytes_as_before(lib):
    import hashlib

    h = hashlib.sha256(open(g8.__file__, "rb").read())
    h.update(open(cut.__file__, "rb").read())
    h.update(lib.read_bytes())
    assert g8._cut_cache_key(0).endswith("|" + h.hexdigest()[:20])
