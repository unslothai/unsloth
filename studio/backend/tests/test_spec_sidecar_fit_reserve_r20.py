# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Twentieth-round cells: a warm capability probe does not re-read the server binary.

A cached capability hit re-checks the borrowed-draft marker on every load, which scanned the
whole server binary for the libraries it links each time. The embedded names are now read
once per file version; whether each library exists is still checked on every call.
"""

from __future__ import annotations

import os
import pathlib

from core.inference import llama_cpp as _lc


def _counting_reads(monkeypatch):
    reads = []
    real = pathlib.Path.read_bytes

    def read_bytes(self):
        reads.append(self.name)
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    return reads


def test_the_linked_library_names_are_read_once_per_binary_version(tmp_path, monkeypatch):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\0libllama-common.so.0\0" * 4)
    lib = tmp_path / "libllama-common.so.0"
    lib.write_bytes(b"lib")
    reads = _counting_reads(monkeypatch)
    for _ in range(3):
        assert _lc._linked_llama_libs(server, tmp_path) == [lib]
    assert reads.count("llama-server") == 1
    # Existence stays fresh: a removed library is not reported from the cache.
    lib.unlink()
    assert _lc._linked_llama_libs(server, tmp_path) == []
    assert reads.count("llama-server") == 1
    # A rebuilt binary is read again.
    lib.write_bytes(b"lib")
    server.write_bytes(b"\0libllama-server-impl.so\0")
    st = server.stat()
    os.utime(server, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    (tmp_path / "libllama-server-impl.so").write_bytes(b"impl")
    assert _lc._linked_llama_libs(server, tmp_path) == [tmp_path / "libllama-server-impl.so"]
    assert reads.count("llama-server") == 2
