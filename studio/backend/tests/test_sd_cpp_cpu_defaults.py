# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp ``--threads``: ``UNSLOTH_CPU_THREADS`` reaches it, and a cgroup CPU quota caps the default."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend as bk


def _cgroups(tmp_path, quotas, own = "/user.slice/app.scope"):
    """A cgroup v2 tree where ``quotas`` maps a cgroup path to its cpu.max line, plus /proc/self/cgroup."""
    root = tmp_path / "cg"
    for rel, line in quotas.items():
        d = root / rel.lstrip("/")
        d.mkdir(parents = True, exist_ok = True)
        (d / "cpu.max").write_text(line + "\n")
    (root / own.lstrip("/")).mkdir(parents = True, exist_ok = True)
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{own}\n")
    return str(root), str(proc)


@pytest.mark.parametrize(
    "quotas, expected",
    [
        ({}, None),
        ({"/user.slice/app.scope": "max 100000"}, None),
        ({"/user.slice/app.scope": "400000 100000"}, 4),
        ({"/user.slice/app.scope": "250000 100000"}, 3),
        # A parent's quota binds too, and the tightest one wins.
        ({"/user.slice": "200000 100000", "/user.slice/app.scope": "800000 100000"}, 2),
        ({"/": "600000 100000"}, 6),
        ({"/user.slice/app.scope": "garbage"}, None),
    ],
)
def test_the_tightest_quota_over_the_own_cgroup_and_its_parents(tmp_path, quotas, expected):
    root, proc = _cgroups(tmp_path, quotas)
    assert bk._cgroup_cpu_limit(root, proc) == expected


def test_a_container_with_its_own_cgroup_namespace_reads_the_root(tmp_path):
    root, proc = _cgroups(tmp_path, {"/": "400000 100000"}, own = "/")
    assert bk._cgroup_cpu_limit(root, proc) == 4


def test_no_cgroup_v2_entry_reads_the_root_only(tmp_path):
    root, _ = _cgroups(tmp_path, {"/": "300000 100000"})
    missing = str(tmp_path / "missing")
    assert bk._cgroup_cpu_limit(root, missing) == 3


def _host(monkeypatch, *, cpus = 192, limit = None, platform = "linux"):
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(bk.sys, "platform", platform)
    monkeypatch.setattr(bk.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: limit)


def test_without_a_quota_the_default_is_unchanged(monkeypatch):
    _host(monkeypatch)
    assert bk._default_threads() == 96
    _host(monkeypatch, cpus = None)
    assert bk._default_threads() == 8


def test_a_quota_caps_the_default(monkeypatch):
    _host(monkeypatch, limit = 4)
    assert bk._default_threads() == 4
    _host(monkeypatch, cpus = 8, limit = 16)
    assert bk._default_threads() == 4


def test_other_platforms_never_read_cgroups(monkeypatch):
    _host(monkeypatch, cpus = 10, platform = "darwin")
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: pytest.fail("no cgroups off Linux"))
    assert bk._default_threads() == 5


@pytest.mark.parametrize("value, expected", [("12", 12), ("0", 4), ("x", 4), ("", 4), ("-3", 4)])
def test_unsloth_cpu_threads_wins_over_the_default_and_the_quota(monkeypatch, value, expected):
    _host(monkeypatch, limit = 4)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", value)
    assert bk._default_threads() == expected
