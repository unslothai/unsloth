# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_sd_cpp_probe_output`` spawns ``--help`` once per build of a binary; failures and other probes are never cached."""

from __future__ import annotations

import os
import subprocess
import types

import pytest

from core.inference import sd_cpp_backend as bk


@pytest.fixture
def fake_run(monkeypatch):
    """Records each spawn; ``state`` sets what the next spawn returns (``raise`` wins over ``rc``)."""
    state = {"calls": [], "rc": 0, "stdout": "usage: sd-cli --ref-video --max-vram", "raise": None}

    def _run(cmd, **kwargs):
        state["calls"].append(list(cmd))
        if state["raise"] is not None:
            raise state["raise"]
        return types.SimpleNamespace(returncode = state["rc"], stdout = state["stdout"], stderr = "")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(bk, "_HELP_TEXT_CACHE", {}, raising = False)
    return state


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "sd-server"
    path.write_bytes(b"\x7fELF build one")
    return str(path)


def test_repeated_help_spawns_once(fake_run, binary):
    first = bk._sd_cpp_probe_output(binary, "--help")
    fake_run["stdout"] = "a later spawn would see this"
    assert [bk._sd_cpp_probe_output(binary, "--help") for _ in range(4)] == [first] * 4
    assert first == "usage: sd-cli --ref-video --max-vram"
    assert len(fake_run["calls"]) == 1
    assert bk.sd_cpp_supports_minimax_h3(binary)
    assert bk._layer_split_supported(binary) is False
    assert len(fake_run["calls"]) == 1


@pytest.mark.parametrize("failure", ["rc", "timeout", "oserror"])
def test_failed_help_is_not_cached(fake_run, binary, failure):
    if failure == "rc":
        fake_run["rc"] = 1
    elif failure == "timeout":
        fake_run["raise"] = subprocess.TimeoutExpired(binary, 20)
    else:
        fake_run["raise"] = OSError("exec format error")
    assert bk._sd_cpp_probe_output(binary, "--help") is None
    assert bk._sd_cpp_probe_output(binary, "--help") is None
    assert len(fake_run["calls"]) == 2
    fake_run["rc"], fake_run["raise"] = 0, None
    assert bk._sd_cpp_probe_output(binary, "--help") == fake_run["stdout"]
    assert len(fake_run["calls"]) == 3


def test_rewritten_binary_reprobes(fake_run, binary):
    assert bk._sd_cpp_probe_output(binary, "--help") == "usage: sd-cli --ref-video --max-vram"
    # Same size, new mtime.
    with open(binary, "wb") as f:
        f.write(b"\x7fELF build two")
    st = os.stat(binary)
    os.utime(binary, ns = (st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    fake_run["stdout"] = "usage: newer build"
    assert bk._sd_cpp_probe_output(binary, "--help") == "usage: newer build"
    # Different size, mtime pinned back.
    with open(binary, "wb") as f:
        f.write(b"\x7fELF build three, longer")
    os.utime(binary, ns = (st.st_atime_ns, st.st_mtime_ns))
    fake_run["stdout"] = "usage: third build"
    assert bk._sd_cpp_probe_output(binary, "--help") == "usage: third build"
    assert len(fake_run["calls"]) == 3
    assert len(bk._HELP_TEXT_CACHE) == 1


def test_list_devices_is_never_cached(fake_run, binary):
    fake_run["stdout"] = "CUDA0: NVIDIA test card"
    for _ in range(3):
        assert bk._sd_cpp_probe_output(binary, "--list-devices") == "CUDA0: NVIDIA test card"
    assert len(fake_run["calls"]) == 3
    assert bk._HELP_TEXT_CACHE == {}


def test_missing_binary_spawns_every_time(fake_run, tmp_path):
    missing = str(tmp_path / "gone" / "sd-cli")
    fake_run["raise"] = FileNotFoundError(missing)
    assert bk._sd_cpp_probe_output(missing, "--help") is None
    assert bk._sd_cpp_probe_output(missing, "--help") is None
    assert len(fake_run["calls"]) == 2
    # A stat failure with a spawn that answers still answers, uncached.
    fake_run["raise"] = None
    assert bk._sd_cpp_probe_output(missing, "--help") == fake_run["stdout"]
    assert bk._sd_cpp_probe_output(missing, "--help") == fake_run["stdout"]
    assert len(fake_run["calls"]) == 4
    assert bk._HELP_TEXT_CACHE == {}


def test_cache_stays_bounded(fake_run, tmp_path):
    for i in range(bk._HELP_TEXT_CACHE_MAX + 5):
        path = tmp_path / f"sd-cli-{i}"
        path.write_bytes(b"x")
        bk._sd_cpp_probe_output(str(path), "--help")
    assert len(bk._HELP_TEXT_CACHE) == bk._HELP_TEXT_CACHE_MAX
