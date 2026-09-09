# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolving the RPC server binary has to mean the same thing on every platform.

`os.access(path, os.X_OK)` is not an executability test on Windows: it succeeds for any file
that exists. The guard therefore accepted a text file as a server binary. The second half is
worse, and a fix for the first alone leaves it in place: `_RPC_SERVER_NAMES` is POSIX-shaped,
extensionless names first, and the search takes the first hit, so a stray extensionless
`ggml-rpc-server` sitting beside the real `ggml-rpc-server.exe` won. The resolution could report
success while handing back the wrong file.

Windows is exercised through the product's own `_on_windows` seam rather than by patching
`os.name`, which pathlib reads to choose its flavour and which therefore makes `Path()` try to
build a `WindowsPath` on Linux. A separate test pins that seam to `os.name`. POSIX
behaviour is asserted directly and must not change: there the extensionless name is correct and
the exec bit means what it says.
"""

from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cluster():
    spec = importlib.util.spec_from_file_location(
        "spark_cluster_for_rpc_resolution", REPO / "studio" / "spark_cluster.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, *, executable: bool) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"binary")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _bundle(cluster, monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    # rpc_server_binary also searches ~/src/llamacpp-rpc, which exists on a developer machine
    # and would answer for the tree under test.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(cluster.shutil, "which", lambda name: None)
    return root


def test_windows_prefers_the_executable_name_over_a_stray_extensionless_file(
    monkeypatch, tmp_path
) -> None:
    """The one that reported success while returning the wrong file."""
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    decoy = _write(root / "build" / "bin" / "ggml-rpc-server", executable = False)
    real = _write(root / "build" / "bin" / "ggml-rpc-server.exe", executable = False)

    monkeypatch.setattr(cluster, "_on_windows", lambda windows = None: True)
    resolved = cluster.rpc_server_binary()
    assert resolved == str(real), f"resolved {resolved}, decoy was {decoy}"


def test_windows_rejects_a_file_that_is_not_executable_there(monkeypatch, tmp_path) -> None:
    """With only the extensionless file present the answer is None, not that file."""
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)

    monkeypatch.setattr(cluster, "_on_windows", lambda windows = None: True)
    assert cluster.rpc_server_binary() is None


def test_posix_still_resolves_the_extensionless_binary(monkeypatch, tmp_path) -> None:
    """No regression: the extensionless name is the right one here."""
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    real = _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)
    _write(root / "build" / "bin" / "ggml-rpc-server.exe", executable = False)

    monkeypatch.setattr(cluster, "_on_windows", lambda windows = None: False)
    assert cluster.rpc_server_binary() == str(real)


def test_posix_still_rejects_a_file_without_the_exec_bit(monkeypatch, tmp_path) -> None:
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    _write(root / "build" / "bin" / "ggml-rpc-server", executable = False)

    monkeypatch.setattr(cluster, "_on_windows", lambda windows = None: False)
    assert cluster.rpc_server_binary() is None


def test_the_platform_seam_follows_os_name() -> None:
    """What the resolution asks, on whatever platform this runs."""
    cluster = _cluster()
    assert cluster._on_windows() is (os.name == "nt")
    assert cluster._on_windows(True) is True
    assert cluster._on_windows(False) is False


@pytest.mark.parametrize(
    "windows,leading",
    [(True, "ggml-rpc-server.exe"), (False, "ggml-rpc-server")],
)
def test_the_search_order_follows_the_platform(windows: bool, leading: str) -> None:
    cluster = _cluster()
    names = cluster.rpc_server_names(windows = windows)
    assert names[0] == leading
    # Ordering only. Dropping a name would make some installs unresolvable.
    assert set(names) == set(cluster._RPC_SERVER_NAMES)


def test_the_peer_probe_decides_for_itself() -> None:
    """`_BUNDLE_PROBE` runs on the peer, which need not be this platform, so the same ordering
    and the same executability test have to be inside the probe rather than baked in here."""
    source = cluster_source = _cluster()._BUNDLE_PROBE
    assert "os.name" in source, "the probe does not look at the peer's platform at all"
    assert "def order(" in source, "the probe does not order names by platform"
    assert "os.access" in cluster_source, "the probe lost its POSIX exec-bit test"
