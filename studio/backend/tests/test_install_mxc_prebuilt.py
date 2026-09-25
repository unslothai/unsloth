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


_HOST_PREP = b"official-host-prep"


def _pin_host_prep(monkeypatch, payload = _HOST_PREP):
    monkeypatch.setattr(installer.mxc_runtime, "WXC_HOST_PREP_SIZE", len(payload))
    monkeypatch.setattr(
        installer.mxc_runtime, "WXC_HOST_PREP_SHA256", hashlib.sha256(payload).hexdigest()
    )


def _serve(monkeypatch, archive):
    monkeypatch.setattr(installer.mxc_runtime, "RELEASE_ARCHIVE_SIZE", len(archive))
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RELEASE_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )
    monkeypatch.setattr(
        installer, "_download", lambda destination: destination.write_bytes(archive)
    )


@pytest.fixture
def official_release(tmp_path, monkeypatch):
    payload = b"official-wxc"
    archive = _archive(
        [
            ("README.txt", b"Microsoft MXC release", None),
            (installer.mxc_runtime.RELEASE_MEMBER, payload, stat.S_IFREG | 0o644),
            (installer.mxc_runtime.RELEASE_HOST_PREP_MEMBER, _HOST_PREP, stat.S_IFREG | 0o644),
            ("arm64/wxc-host-prep.exe", b"arm64", stat.S_IFREG | 0o644),
        ]
    )
    _pin_host_prep(monkeypatch)
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
    assert sorted(path.name for path in install_dir.iterdir()) == [
        "wxc-exec.exe",
        "wxc-host-prep.exe",
    ]
    assert (install_dir / "wxc-exec.exe").read_bytes() == payload
    assert (install_dir / "wxc-host-prep.exe").read_bytes() == _HOST_PREP
    assert installer.mxc_runtime.selected_host_prep(package_root = install_dir).path.name == (
        "wxc-host-prep.exe"
    )
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


def test_install_without_host_prep_is_refreshed(tmp_path, official_release):
    # Installs from before host-prep was pinned must pick it up, not count as current.
    payload, _archive_bytes = official_release
    install_dir = tmp_path / "installed" / "windows-x86_64"
    install_dir.mkdir(parents = True)
    (install_dir / "wxc-exec.exe").write_bytes(payload)
    assert installer.install_mxc_release(install_dir) is True
    installer.mxc_runtime.selected_host_prep(package_root = install_dir)


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
        [(installer.mxc_runtime.RELEASE_MEMBER, b"official-wxc", None)],
        [
            (installer.mxc_runtime.RELEASE_MEMBER, b"official-wxc", None),
            (installer.mxc_runtime.RELEASE_HOST_PREP_MEMBER, b"resized-host-prep", None),
        ],
        [
            (installer.mxc_runtime.RELEASE_MEMBER, b"official-wxc", None),
            (installer.mxc_runtime.RELEASE_HOST_PREP_MEMBER, _HOST_PREP, None),
            (installer.mxc_runtime.RELEASE_HOST_PREP_MEMBER.upper(), _HOST_PREP, None),
        ],
    ],
    ids = [
        "missing",
        "duplicate",
        "traversal",
        "symlink",
        "host_prep_missing",
        "host_prep_size",
        "host_prep_duplicate",
    ],
)
def test_unsafe_archive_shapes_are_rejected(tmp_path, monkeypatch, official_release, entries):
    _serve(monkeypatch, _archive(entries))
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


@pytest.fixture
def prepared_install(tmp_path, monkeypatch, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    installer.install_mxc_release(install_dir)
    ran: list[tuple[str, str]] = []

    def record(executable, step):
        ran.append((str(executable), step))
        return 0

    monkeypatch.setattr(installer, "_run_host_prep", record)
    return install_dir, ran


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (("prepare-null-device",), ["prepare-null-device"]),
        (None, ["prepare-system-drive", "prepare-null-device"]),
        ((), []),
    ],
    ids = ["only_missing", "unknown_runs_both", "prepared"],
)
def test_prepare_host_runs_only_the_steps_mxc_reports_missing(
    monkeypatch, prepared_install, reported, expected
):
    install_dir, ran = prepared_install
    monkeypatch.setattr(installer.mxc_runtime, "probe_host_prep_steps", lambda **_kwargs: reported)
    assert installer.prepare_host(install_dir) == tuple(expected)
    host_prep = str((install_dir / "wxc-host-prep.exe").resolve())
    assert ran == [(host_prep, step) for step in expected]


def test_prepare_host_refuses_a_tampered_binary_before_elevating(monkeypatch, prepared_install):
    install_dir, ran = prepared_install
    (install_dir / "wxc-host-prep.exe").write_bytes(b"x" * len(_HOST_PREP))
    monkeypatch.setattr(installer.mxc_runtime, "probe_host_prep_steps", lambda **_kwargs: None)
    with pytest.raises(installer.mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        installer.prepare_host(install_dir)
    assert ran == []


def test_prepare_host_reports_a_failed_step(monkeypatch, prepared_install):
    install_dir, _ran = prepared_install
    monkeypatch.setattr(installer.mxc_runtime, "probe_host_prep_steps", lambda **_kwargs: None)
    monkeypatch.setattr(installer, "_run_host_prep", lambda _executable, _step: 65)
    with pytest.raises(installer.MxcInstallError, match = "prepare-system-drive failed"):
        installer.prepare_host(install_dir)


@pytest.mark.parametrize("elevated", [True, False])
def test_host_prep_elevates_only_when_needed(monkeypatch, tmp_path, elevated):
    calls: list[tuple] = []
    monkeypatch.setattr(installer, "_is_elevated", lambda: elevated)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(("run", argv)) or subprocess_result(0),
    )
    monkeypatch.setattr(
        installer,
        "_run_elevated",
        lambda executable, arguments, _directory: calls.append(
            ("runas", [str(executable), *arguments])
        )
        or 0,
    )
    executable = tmp_path / "wxc-host-prep.exe"
    assert installer._run_host_prep(executable, "prepare-null-device") == 0
    assert calls == [
        ("run" if elevated else "runas", [str(executable), "prepare-null-device", "--quiet"])
    ]


def test_elevated_host_prep_is_never_timed_out(monkeypatch, tmp_path):
    # prepare-system-drive ran 3 to 7 minutes on a CI runner; a kill leaves the drive half re-ACLed.
    seen: dict = {}
    monkeypatch.setattr(installer, "_is_elevated", lambda: True)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda argv, **kwargs: seen.update(kwargs) or subprocess_result(0),
    )
    assert installer._run_host_prep(tmp_path / "wxc-host-prep.exe", "prepare-system-drive") == 0
    assert "timeout" not in seen


def subprocess_result(code):
    import subprocess
    return subprocess.CompletedProcess(args = [], returncode = code)


def test_prepare_host_cli_defaults_to_the_managed_runtime(monkeypatch, tmp_path):
    managed = tmp_path / "managed"
    seen: list[Path] = []
    monkeypatch.setattr(installer.mxc_runtime, "_installed_package_root", lambda: managed)
    monkeypatch.setattr(installer, "prepare_host", lambda path: seen.append(path) or ())
    assert installer.main(["--prepare-host"]) == 0
    assert seen == [managed]
    with pytest.raises(SystemExit):
        installer.main([])
