# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The deferred-write RAM gate credits only reclaimable cgroup page cache, never tmpfs / shared memory."""

from __future__ import annotations

import builtins
import io
import pathlib

from core.inference import diffusion_prequant_cache as C

GiB = 1 << 30


def _fake_cgroup(monkeypatch, tmp_path, *, limit, used, file, shmem):
    node = tmp_path / "sys" / "fs" / "cgroup" / "studio.slice"
    node.mkdir(parents = True)
    (node / "memory.max").write_text(f"{limit}\n")
    (node / "memory.current").write_text(f"{used}\n")
    (node / "memory.stat").write_text(f"anon {used - file}\nfile {file}\nshmem {shmem}\n")
    real_path, real_open = pathlib.Path, builtins.open

    def _path(p, *rest):
        p = str(p)
        if p.startswith("/sys/fs/cgroup"):
            return real_path(str(tmp_path / "sys") + p[len("/sys") :], *rest)
        return real_path(p, *rest)

    def _open(p, *a, **k):
        if str(p) == "/proc/self/cgroup":
            return io.StringIO("0::/studio.slice\n")
        return real_open(p, *a, **k)

    monkeypatch.setattr(C, "Path", _path)
    monkeypatch.setattr(C, "open", _open, raising = False)


def test_shared_memory_is_not_counted_as_available(monkeypatch, tmp_path):
    # 16 GiB cap, 15 GiB charged, 8 GiB of it /dev/shm: only 1 GiB is actually free.
    _fake_cgroup(monkeypatch, tmp_path, limit = 16 * GiB, used = 15 * GiB, file = 8 * GiB, shmem = 8 * GiB)
    limit, avail = C._cgroup_memory_bytes()
    assert limit == 16 * GiB
    assert avail == 1 * GiB


def test_clean_page_cache_is_still_credited(monkeypatch, tmp_path):
    _fake_cgroup(monkeypatch, tmp_path, limit = 16 * GiB, used = 15 * GiB, file = 8 * GiB, shmem = 2 * GiB)
    assert C._cgroup_memory_bytes() == (16 * GiB, 7 * GiB)
