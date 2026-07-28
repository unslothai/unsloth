# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioral tests for the --with-llama-cpp-dir 'unmanaged local link' contract.

When the canonical llama.cpp dir is a symlink (POSIX) / junction (Windows) to a
user's own checkout, Unsloth must treat it as externally managed:
  - the in-app updater must not offer or apply a prebuilt over the link
  - orphan cleanup must not kill a llama-server the user launched from that tree

These exercise real link behavior rather than grepping the scripts.
"""

import os
import subprocess
from pathlib import Path

import pytest

from utils import llama_cpp_update as u
from core.inference.llama_cpp import LlamaCppBackend


@pytest.fixture(autouse = True)
def _no_whisper_piggyback(monkeypatch):
    # Keep the whisper piggyback probe off the host: these tests exercise the
    # llama local-link contract only.
    monkeypatch.setattr(u, "_whisper_chain_status", lambda **kwargs: None)


def _make_link(link: Path, target: Path) -> None:
    """Create a directory junction (Windows) / symlink (POSIX); neither needs
    elevation."""
    target.mkdir(parents = True, exist_ok = True)
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check = True,
            capture_output = True,
            text = True,
        )
    else:
        link.symlink_to(target, target_is_directory = True)


def _server_subpath() -> Path:
    return Path(
        "build/bin/Release/llama-server.exe" if os.name == "nt" else "build/bin/llama-server"
    )


class _FakeProc:
    def __init__(self, pid: int, exe: str) -> None:
        self.info = {"pid": pid, "name": "llama-server", "exe": exe}
        self.killed = False

    def kill(self) -> None:
        self.killed = True


def test_is_external_link_detects_link_vs_plain_dir(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert u._is_external_link(plain) is False

    link = tmp_path / "link"
    _make_link(link, tmp_path / "tgt")
    assert u._is_external_link(link) is True


def test_active_install_is_local_link(tmp_path: Path) -> None:
    link = tmp_path / "llama.cpp"
    _make_link(link, tmp_path / "tgt")
    binary = str(link / _server_subpath())
    assert u._active_install_is_local_link(binary) is True

    # A plain (non-link) llama.cpp dir is Unsloth-managed, not a local link.
    plain = tmp_path / "plain" / "llama.cpp"
    plain.mkdir(parents = True)
    assert u._active_install_is_local_link(str(plain / _server_subpath())) is False


def test_get_update_status_reports_local_link(tmp_path: Path, monkeypatch) -> None:
    link = tmp_path / "llama.cpp"
    _make_link(link, tmp_path / "tgt")
    monkeypatch.setattr(u, "_find_binary", lambda: str(link / _server_subpath()))
    st = u.get_update_status()
    assert st["supported"] is False
    assert st["update_available"] is False
    assert st["local_link"] is True


def test_start_update_refuses_local_link(tmp_path: Path, monkeypatch) -> None:
    link = tmp_path / "llama.cpp"
    _make_link(link, tmp_path / "tgt")
    monkeypatch.setattr(u, "_find_binary", lambda: str(link / _server_subpath()))
    res = u.start_update()
    assert res["started"] is False
    assert res["reason"] == "local_link"


def _run_orphan_scan(monkeypatch, studio_root: Path, fake: _FakeProc) -> int:
    # psutil drives the cross-platform process scan; skip (rather than error) if a
    # minimal test env lacks it. CI installs it so these tests actually run.
    psutil = pytest.importorskip("psutil")

    monkeypatch.setattr(
        LlamaCppBackend,
        "_resolved_studio_root_and_is_legacy",
        staticmethod(lambda: (studio_root.resolve(), False)),
    )
    monkeypatch.setattr(LlamaCppBackend, "_reap_recorded_pid", staticmethod(lambda: 0))
    monkeypatch.setattr(psutil, "process_iter", lambda attrs = None: iter([fake]))
    return LlamaCppBackend._kill_orphaned_servers()


def test_orphan_cleanup_spares_local_link_tree(tmp_path: Path, monkeypatch) -> None:
    studio_root = tmp_path / "studio-home"
    studio_root.mkdir()
    external = tmp_path / "external"
    (external / _server_subpath().parent).mkdir(parents = True)
    (external / _server_subpath()).write_text("x")
    _make_link(studio_root / "llama.cpp", external)

    exe_under_link = str((external / _server_subpath()).resolve())
    fake = _FakeProc(os.getpid() + 777, exe_under_link)
    killed = _run_orphan_scan(monkeypatch, studio_root, fake)
    assert killed == 0
    assert fake.killed is False


def test_orphan_cleanup_kills_under_real_root(tmp_path: Path, monkeypatch) -> None:
    # Control: a real (non-link) managed root still gets its orphan reaped, so
    # the spare-the-link test above is meaningful (not a no-op).
    studio_root = tmp_path / "studio-home"
    bin_dir = studio_root / "llama.cpp" / _server_subpath().parent
    bin_dir.mkdir(parents = True)
    exe = studio_root / "llama.cpp" / _server_subpath()
    exe.write_text("x")

    fake = _FakeProc(os.getpid() + 888, str(exe.resolve()))
    killed = _run_orphan_scan(monkeypatch, studio_root, fake)
    assert killed == 1
    assert fake.killed is True
