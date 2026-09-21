# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import zipfile

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "install_mxc_prebuilt_under_test", _STUDIO / "install_mxc_prebuilt.py"
)
installer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(installer)


def _archive(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            bundle.writestr(info, payload)
    return output.getvalue()


@pytest.fixture
def official_release(tmp_path, monkeypatch):
    payload = b"official-wxc"
    archive = _archive(
        [
            ("README.txt", b"Microsoft MXC release", None),
            (installer.mxc_runtime.RELEASE_MEMBER, payload, stat.S_IFREG | 0o644),
        ]
    )
    source = tmp_path / "official.zip"
    source.write_bytes(archive)
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime, "WXC_EXEC_SIZE", len(payload))
    monkeypatch.setattr(
        installer.mxc_runtime, "WXC_EXEC_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(installer.mxc_runtime, "RELEASE_ARCHIVE_SIZE", len(archive))
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RELEASE_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )
    monkeypatch.setattr(
        installer, "_download", lambda destination: destination.write_bytes(archive)
    )
    return payload, archive


def test_installs_only_the_official_wxc_member(tmp_path, official_release):
    payload, _archive_bytes = official_release
    install_dir = tmp_path / "installed" / "windows-x86_64"
    assert installer.install_mxc_release(install_dir) is True
    assert sorted(path.name for path in install_dir.iterdir()) == ["wxc-exec.exe"]
    assert (install_dir / "wxc-exec.exe").read_bytes() == payload
    assert installer.mxc_runtime.selected_runtime(package_root = install_dir).path.name == (
        "wxc-exec.exe"
    )


def test_matching_install_is_reused_without_download(tmp_path, monkeypatch, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    assert installer.install_mxc_release(install_dir) is True
    monkeypatch.setattr(
        installer,
        "_download",
        lambda _destination: pytest.fail("verified installation was downloaded again"),
    )
    assert installer.install_mxc_release(install_dir) is False


def test_corrupt_wxc_is_replaced_safely(tmp_path, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    install_dir.mkdir(parents = True)
    (install_dir / "wxc-exec.exe").write_bytes(b"corrupt")
    assert installer.install_mxc_release(install_dir) is True
    installer.mxc_runtime.selected_runtime(package_root = install_dir)


def test_archive_checksum_mismatch_never_becomes_active(tmp_path, monkeypatch, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    monkeypatch.setattr(installer.mxc_runtime, "RELEASE_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(installer.MxcInstallError, match = "checksum mismatch"):
        installer.install_mxc_release(install_dir)
    assert not install_dir.exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("other.exe", b"official-wxc", None)],
        [
            (installer.mxc_runtime.RELEASE_MEMBER, b"official-wxc", None),
            (installer.mxc_runtime.RELEASE_MEMBER, b"official-wxc", None),
        ],
        [("../x64/wxc-exec.exe", b"official-wxc", None)],
        [(installer.mxc_runtime.RELEASE_MEMBER, b"target", stat.S_IFLNK | 0o777)],
    ],
    ids = ["missing", "duplicate", "traversal", "symlink"],
)
def test_unsafe_archive_shapes_are_rejected(tmp_path, monkeypatch, official_release, entries):
    archive = _archive(entries)
    monkeypatch.setattr(installer.mxc_runtime, "RELEASE_ARCHIVE_SIZE", len(archive))
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RELEASE_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )
    monkeypatch.setattr(
        installer, "_download", lambda destination: destination.write_bytes(archive)
    )
    with pytest.raises(installer.MxcInstallError):
        installer.install_mxc_release(tmp_path / "installed" / "windows-x86_64")


def test_windows_setup_always_checks_mxc_without_optional_os_environment_variable():
    setup = (_STUDIO / "setup.ps1").read_text(encoding = "utf-8")
    start = setup.index("# Windows MXC Preview is an optional, pinned prebuilt")
    end = setup.index("# ── Pre-install transformers", start)
    mxc_setup = setup[start:end]

    assert "$env:OS" not in mxc_setup
    assert (
        "[System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT"
    ) in mxc_setup
    assert setup.index("# Windows MXC Preview is an optional, pinned prebuilt") > setup.index(
        'step "python" "dependencies up to date"'
    )


def test_non_windows_host_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(
        installer,
        "_download",
        lambda _destination: pytest.fail("non-Windows installer attempted a download"),
    )
    with pytest.raises(installer.MxcInstallError, match = "Windows-only"):
        installer.install_mxc_release(tmp_path / "installed")
