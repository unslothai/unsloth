# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama.cpp link to a user's checkout is unmanaged; a link into UNSLOTH_STUDIO_APP is not."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils import llama_cpp_path_settings as path_settings
from utils import llama_cpp_update as u
from utils import whisper_cpp_update as w
from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend


@pytest.fixture(autouse = True)
def _no_whisper_piggyback(monkeypatch):
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

    plain = tmp_path / "plain" / "llama.cpp"
    plain.mkdir(parents = True)
    assert u._active_install_is_local_link(str(plain / _server_subpath())) is False


def test_a_link_into_the_studio_app_tree_is_not_a_local_link(tmp_path: Path, monkeypatch) -> None:
    """Image links into UNSLOTH_STUDIO_APP are Unsloth's own install, so the in-app update must stay on."""
    app = tmp_path / "app"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_APP", str(app))

    link = home / "llama.cpp"
    _make_link(link, app / "llama.cpp")
    assert u._active_install_is_local_link(str(link / _server_subpath())) is False

    whisper_link = home / "whisper.cpp"
    _make_link(whisper_link, app / "whisper.cpp")
    whisper_binary = str(whisper_link / "build" / "bin" / "whisper-server")
    assert w._active_install_is_local_link(whisper_binary) is False

    outside = home / "outside.cpp" / "llama.cpp"
    outside.parent.mkdir()
    _make_link(outside, tmp_path / "my-checkout")
    assert u._active_install_is_local_link(str(outside / _server_subpath())) is True


def test_without_the_studio_app_tree_every_link_stays_external(tmp_path: Path, monkeypatch) -> None:
    """Outside the Docker image nothing sets the variable, so the contract is unchanged.
    An empty value counts as unset: only a real tree may exempt a link."""
    app = tmp_path / "app"
    link = tmp_path / "home" / "llama.cpp"
    link.parent.mkdir()
    _make_link(link, app / "llama.cpp")
    binary = str(link / _server_subpath())
    monkeypatch.delenv("UNSLOTH_STUDIO_APP", raising = False)
    assert u._active_install_is_local_link(binary) is True
    monkeypatch.setenv("UNSLOTH_STUDIO_APP", "   ")
    assert u._active_install_is_local_link(binary) is True


def test_a_sibling_of_the_studio_app_tree_is_not_inside_it(tmp_path: Path, monkeypatch) -> None:
    """$UNSLOTH_STUDIO_APP=/opt/unsloth-studio-app must not claim /opt/unsloth-studio-appx."""
    monkeypatch.setenv("UNSLOTH_STUDIO_APP", str(tmp_path / "app"))
    link = tmp_path / "home" / "llama.cpp"
    link.parent.mkdir()
    _make_link(link, tmp_path / "appx" / "llama.cpp")
    assert u._active_install_is_local_link(str(link / _server_subpath())) is True


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


def _fake_procfs(tmp_path: Path, fake: _FakeProc) -> Path:
    """Build a /proc-shaped tree holding a single llama-server process."""
    root = tmp_path / "fake-proc"
    entry = root / str(fake.info["pid"])
    entry.mkdir(parents = True)
    # comm sits between the first '(' and the last ')'; starttime is field 22.
    filler = " ".join(["0"] * 18)  # fields 4..21
    (entry / "stat").write_bytes(
        f"{fake.info['pid']} (llama-server) S {filler} 1000".encode("utf-8")
    )
    (entry / "exe").symlink_to(fake.info["exe"])
    (root / "self").mkdir()
    other = root / str(fake.info["pid"] + 1)
    other.mkdir()
    (other / "stat").write_bytes(f"1 (python3) S {filler} 1000".encode("utf-8"))
    return root


def _run_orphan_scan(
    monkeypatch,
    studio_root: Path,
    fake: _FakeProc,
    scan: str = "psutil",
    tmp_path: Path = None,
) -> int:
    psutil = pytest.importorskip("psutil")

    monkeypatch.setattr(
        LlamaCppBackend,
        "_resolved_studio_root_and_is_legacy",
        staticmethod(lambda: (studio_root.resolve(), False)),
    )
    monkeypatch.setattr(LlamaCppBackend, "_reap_recorded_pid", staticmethod(lambda: 0))

    # Pin parent liveness: the invented PID may be a real process on busy runners.
    monkeypatch.setattr(LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False))

    if scan == "procfs":
        # Fixture pid is not real, so intercept the signal.
        if sys.platform != "linux":
            pytest.skip("the procfs scan only runs on Linux")
        monkeypatch.setattr(llama_cpp_module, "_PROC_ROOT", str(_fake_procfs(tmp_path, fake)))

        def _fake_kill(pid, sig):
            if pid == fake.info["pid"]:
                fake.kill()
                return
            raise ProcessLookupError(pid)

        monkeypatch.setattr(os, "kill", _fake_kill)
    else:
        monkeypatch.setattr(llama_cpp_module, "_PROC_ROOT", str(studio_root / "no-such-proc"))
        monkeypatch.setattr(psutil, "process_iter", lambda attrs = None: iter([fake]))
    return LlamaCppBackend._kill_orphaned_servers()


@pytest.mark.parametrize("scan", ["psutil", "procfs"])
def test_orphan_cleanup_spares_local_link_tree(tmp_path: Path, monkeypatch, scan) -> None:
    studio_root = tmp_path / "studio-home"
    studio_root.mkdir()
    external = tmp_path / "external"
    (external / _server_subpath().parent).mkdir(parents = True)
    (external / _server_subpath()).write_text("x")
    _make_link(studio_root / "llama.cpp", external)

    exe_under_link = str((external / _server_subpath()).resolve())
    fake = _FakeProc(os.getpid() + 777, exe_under_link)
    killed = _run_orphan_scan(monkeypatch, studio_root, fake, scan, tmp_path)
    assert killed == 0
    assert fake.killed is False


@pytest.mark.parametrize("scan", ["psutil", "procfs"])
def test_orphan_cleanup_kills_under_real_root(tmp_path: Path, monkeypatch, scan) -> None:
    studio_root = tmp_path / "studio-home"
    bin_dir = studio_root / "llama.cpp" / _server_subpath().parent
    bin_dir.mkdir(parents = True)
    exe = studio_root / "llama.cpp" / _server_subpath()
    exe.write_text("x")

    fake = _FakeProc(os.getpid() + 888, str(exe.resolve()))
    killed = _run_orphan_scan(monkeypatch, studio_root, fake, scan, tmp_path)
    assert killed == 1
    assert fake.killed is True


@pytest.mark.parametrize("scan", ["psutil", "procfs"])
def test_orphan_cleanup_spares_studio_selected_custom_tree(
    tmp_path: Path, monkeypatch, scan
) -> None:
    studio_root = tmp_path / "studio-home"
    studio_root.mkdir()
    custom_root = tmp_path / "user-owned-llama.cpp"
    binary = custom_root / _server_subpath()
    binary.parent.mkdir(parents = True)
    binary.write_text("x")
    monkeypatch.setattr(
        path_settings,
        "get_stored_custom_llama_cpp_path",
        lambda: custom_root.resolve(),
    )

    fake = _FakeProc(os.getpid() + 999, str(binary.resolve()))
    killed = _run_orphan_scan(monkeypatch, studio_root, fake, scan, tmp_path)

    assert killed == 0
    assert fake.killed is False
