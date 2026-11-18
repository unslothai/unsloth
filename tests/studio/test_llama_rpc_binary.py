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
    sys.platform == "win32", reason = "the POSIX exec-bit rules do not apply on Windows"
)

REPO = Path(__file__).resolve().parents[2]


def _lookup():
    spec = importlib.util.spec_from_file_location(
        "llama_rpc_for_rpc_resolution", REPO / "studio" / "llama_rpc.py"
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


def _bundle(rpc, monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(rpc.shutil, "which", lambda name: None)
    return root


def test_windows_prefers_the_executable_name_over_a_stray_extensionless_file(
    monkeypatch, tmp_path
) -> None:
    """The one that reported success while returning the wrong file."""
    rpc = _lookup()
    root = _bundle(rpc, monkeypatch, tmp_path)
    decoy = _write(root / "build" / "bin" / "ggml-rpc-server", executable = False)
    real = _write(root / "build" / "bin" / "ggml-rpc-server.exe", executable = False)

    monkeypatch.setattr(rpc, "_on_windows", lambda windows = None: True)
    resolved = rpc.rpc_server_binary()
    assert resolved == str(real), f"resolved {resolved}, decoy was {decoy}"


def test_windows_rejects_a_file_that_is_not_executable_there(monkeypatch, tmp_path) -> None:
    """With only the extensionless file present the answer is None, not that file."""
    rpc = _lookup()
    root = _bundle(rpc, monkeypatch, tmp_path)
    _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)

    monkeypatch.setattr(rpc, "_on_windows", lambda windows = None: True)
    assert rpc.rpc_server_binary() is None


def test_posix_still_resolves_the_extensionless_binary(monkeypatch, tmp_path) -> None:
    """No regression: the extensionless name is the right one here."""
    rpc = _lookup()
    root = _bundle(rpc, monkeypatch, tmp_path)
    real = _write(root / "build" / "bin" / "ggml-rpc-server", executable = True)
    _write(root / "build" / "bin" / "ggml-rpc-server.exe", executable = False)

    monkeypatch.setattr(rpc, "_on_windows", lambda windows = None: False)
    assert rpc.rpc_server_binary() == str(real)


@_POSIX_ONLY
def test_posix_still_rejects_a_file_without_the_exec_bit(monkeypatch, tmp_path) -> None:
    rpc = _lookup()
    root = _bundle(rpc, monkeypatch, tmp_path)
    _write(root / "build" / "bin" / "ggml-rpc-server", executable = False)

    monkeypatch.setattr(rpc, "_on_windows", lambda windows = None: False)
    assert rpc.rpc_server_binary() is None


def test_the_platform_seam_follows_os_name() -> None:
    """What the resolution asks, on whatever platform this runs."""
    rpc = _lookup()
    assert rpc._on_windows() is (os.name == "nt")
    assert rpc._on_windows(True) is True
    assert rpc._on_windows(False) is False


@pytest.mark.parametrize(
    "windows,leading",
    [(True, "ggml-rpc-server.exe"), (False, "ggml-rpc-server")],
)
def test_the_search_order_follows_the_platform(windows: bool, leading: str) -> None:
    rpc = _lookup()
    names = rpc.rpc_server_names(windows = windows)
    assert names[0] == leading
    # Ordering only. Dropping a name would make some installs unresolvable.
    assert set(names) == set(rpc._RPC_SERVER_NAMES)


def _clean_roots(monkeypatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    for var in ("UNSLOTH_LLAMA_CPP_PATH", "UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@_POSIX_ONLY
def test_unsloth_home_holds_the_bundle_setup_installed(monkeypatch, tmp_path) -> None:
    _clean_roots(monkeypatch, tmp_path)
    master = tmp_path / "portable"
    binary = _write(master / "llama.cpp" / "build" / "bin" / "ggml-rpc-server", executable = True)
    monkeypatch.setenv("UNSLOTH_HOME", f"  {master}  ")
    rpc = _lookup()
    monkeypatch.setattr(rpc.shutil, "which", lambda name: None)

    assert rpc.llama_bundle_dir() == (master / "llama.cpp").resolve()
    assert rpc.rpc_server_location() == (str(binary.resolve()), (master / "llama.cpp").resolve())


def test_an_explicit_llama_cpp_path_still_outranks_unsloth_home(monkeypatch, tmp_path) -> None:
    _clean_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "pinned"))

    assert _lookup().llama_bundle_dir() == tmp_path / "pinned"


def test_a_blank_unsloth_home_counts_as_unset(monkeypatch, tmp_path) -> None:
    home = _clean_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_HOME", "   ")

    assert _lookup().llama_bundle_dir() == home / ".unsloth" / "llama.cpp"


def test_the_recorded_master_root_is_honoured_only_around_studio(monkeypatch, tmp_path) -> None:
    _clean_roots(monkeypatch, tmp_path)
    master = tmp_path / "portable"
    studio = master / "studio"
    (studio / "share").mkdir(parents = True)
    (studio / "share" / ".unsloth-master-root").write_text(f"{master}\n", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))

    assert _lookup().llama_bundle_dir() == master.resolve() / "llama.cpp"

    # A note naming a root that does not contain Studio is not trusted.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (studio / "share" / ".unsloth-master-root").write_text(f"{elsewhere}\n", encoding = "utf-8")

    assert _lookup().llama_bundle_dir() == studio / "llama.cpp"
