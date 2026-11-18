# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolving the RPC server binary has to mean the same thing on every platform."""

from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

import sys

import pytest

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason = "DGX Spark tooling and setup.sh assume a POSIX host"
)

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
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(cluster.shutil, "which", lambda name: None)
    return root


def test_the_toolkit_resolves_through_the_shared_lookup(monkeypatch, tmp_path) -> None:
    """spark_cluster answers exactly what studio/llama_rpc.py answers; the rules live there."""
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    real = _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)
    from studio import llama_rpc

    monkeypatch.setattr(llama_rpc.shutil, "which", lambda name: None)
    assert cluster.rpc_server_binary() == llama_rpc.rpc_server_binary() == str(real)
    assert cluster.llama_bundle_dir() == llama_rpc.llama_bundle_dir() == root
    assert cluster.rpc_server_names() == llama_rpc.rpc_server_names()
    assert cluster._rpc_server_location() == (str(real), root)


@_POSIX_ONLY
def test_the_bundle_identity_names_the_server_the_installer_placed(monkeypatch, tmp_path) -> None:
    cluster = _cluster()
    root = _bundle(cluster, monkeypatch, tmp_path)
    server = _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)
    _write(root / "build" / "bin" / "libggml-rpc.so", executable = False)
    identity = cluster.llama_bundle_identity(root)
    assert identity["rpc_server"] == str(server)
    assert identity["rpc_lib"] == str(root / "build" / "bin" / "libggml-rpc.so")


def test_the_peer_probe_decides_for_itself() -> None:
    """`_BUNDLE_PROBE` runs on the peer, which need not be this platform, so the same ordering and the
    same executability test have to be inside the probe rather than baked in here."""
    source = cluster_source = _cluster()._BUNDLE_PROBE
    assert "os.name" in source, "the probe does not look at the peer's platform at all"
    assert "def order(" in source, "the probe does not order names by platform"
    assert "os.access" in cluster_source, "the probe lost its POSIX exec-bit test"
