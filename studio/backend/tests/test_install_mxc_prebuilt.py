# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "install_mxc_prebuilt_under_test", _STUDIO / "install_mxc_prebuilt.py"
)
installer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(installer)


_HARDLINK = -1


@pytest.fixture(autouse = True)
def _legacy_pin(monkeypatch):
    # Host preparation exists only in the 1.0.1 pin (the AppContainer + DACL tier).
    monkeypatch.setenv(installer.mxc_runtime.PIN_ENV, installer.mxc_runtime.LEGACY_PIN.version)


def _archive(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj = output, mode = "w:gz") as bundle:
        for name, payload, mode in entries:
            info = tarfile.TarInfo(name)
            if mode == _HARDLINK or (mode is not None and stat.S_ISLNK(mode)):
                info.type = tarfile.LNKTYPE if mode == _HARDLINK else tarfile.SYMTYPE
                info.linkname = payload.decode()
                bundle.addfile(info)
                continue
            info.size = len(payload)
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


_HOST_PREP = b"official-host-prep"
_LEGACY = installer.mxc_runtime.LEGACY_PIN
_EXEC_MEMBER = _LEGACY.artifacts[("x86_64", "exec")][0]
_PREP_MEMBER = _LEGACY.artifacts[("x86_64", "prep")][0]
_ARM64_EXEC_MEMBER = _LEGACY.artifacts[("arm64", "exec")][0]
_ARM64_PREP_MEMBER = _LEGACY.artifacts[("arm64", "prep")][0]


def _digest(payload: bytes) -> tuple[int, str]:
    return len(payload), hashlib.sha256(payload).hexdigest()


def _pin(
    archive: bytes,
    payloads: dict,
    *,
    version = "9.9.9",
):
    """A pin shaped like LEGACY_PIN whose approved bytes are this test's payloads."""
    artifacts = {}
    for (architecture, kind), payload in payloads.items():
        member = _LEGACY.artifacts[(architecture, kind)][0]
        artifacts[(architecture, kind)] = (member, *_digest(payload))
    return installer.mxc_runtime.MxcPin(
        version = version,
        revision = "0" * 40,
        archive_size = len(archive),
        archive_sha256 = hashlib.sha256(archive).hexdigest(),
        artifacts = artifacts,
    )


def _serve(
    monkeypatch,
    archive,
    payloads = None,
):
    payloads = payloads or {("x86_64", "exec"): b"official-wxc", ("x86_64", "prep"): _HOST_PREP}
    pin = _pin(archive, payloads)
    monkeypatch.setattr(installer.mxc_runtime, "selected_pin", lambda: pin)
    monkeypatch.setattr(installer.mxc_runtime, "target_pin", lambda: pin)
    monkeypatch.setattr(
        installer, "_download", lambda destination, _pin: destination.write_bytes(archive)
    )
    return pin


@pytest.fixture
def official_release(tmp_path, monkeypatch):
    payload = b"official-wxc"
    archive = _archive(
        [
            ("README.txt", b"Microsoft MXC release", None),
            (_EXEC_MEMBER, payload, stat.S_IFREG | 0o644),
            (_PREP_MEMBER, _HOST_PREP, stat.S_IFREG | 0o644),
            ("package/bin/arm64/wxc-host-prep.exe", b"arm64", stat.S_IFREG | 0o644),
        ]
    )
    monkeypatch.delenv(installer.mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime, "_expected_architecture", lambda: "x86_64")
    # A supported build whatever the test host runs (Windows Server 2022 CI is build 20348).
    monkeypatch.setattr(
        installer.mxc_runtime.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(build = installer.mxc_runtime.MIN_WINDOWS_BUILD, product_type = 1),
        raising = False,
    )
    _serve(monkeypatch, archive)
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


def test_a_pin_without_host_prep_installs_only_wxc_exec(tmp_path, monkeypatch, official_release):
    payload = b"official-wxc-native-only"
    archive = _archive(
        [
            (_EXEC_MEMBER, payload, stat.S_IFREG | 0o644),
            (_PREP_MEMBER, _HOST_PREP, stat.S_IFREG | 0o644),
        ]
    )
    _serve(monkeypatch, archive, {("x86_64", "exec"): payload})
    install_dir = tmp_path / "installed" / "windows-x86_64"
    assert installer.install_mxc_release(install_dir) is True
    assert sorted(path.name for path in install_dir.iterdir()) == ["wxc-exec.exe"]
    assert installer.install_mxc_release(install_dir) is False


def test_arm64_pc_installs_the_arm64_members(tmp_path, monkeypatch):
    x64, arm64, arm64_prep = b"official-wxc", b"official-wxc-arm64", b"official-host-prep-arm64"
    archive = _archive(
        [
            (_EXEC_MEMBER, x64, stat.S_IFREG | 0o644),
            (_PREP_MEMBER, _HOST_PREP, stat.S_IFREG | 0o644),
            (_ARM64_EXEC_MEMBER, arm64, stat.S_IFREG | 0o644),
            (_ARM64_PREP_MEMBER, arm64_prep, stat.S_IFREG | 0o644),
        ]
    )
    monkeypatch.delenv(installer.mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(installer.mxc_runtime, "_expected_architecture", lambda: "arm64")
    _serve(
        monkeypatch,
        archive,
        {
            ("x86_64", "exec"): x64,
            ("x86_64", "prep"): _HOST_PREP,
            ("arm64", "exec"): arm64,
            ("arm64", "prep"): arm64_prep,
        },
    )
    install_dir = tmp_path / "installed" / "windows-arm64"
    assert installer.install_mxc_release(install_dir) is True
    assert (install_dir / "wxc-exec.exe").read_bytes() == arm64
    assert (install_dir / "wxc-host-prep.exe").read_bytes() == arm64_prep
    assert installer.mxc_runtime.selected_runtime(package_root = install_dir).architecture == "arm64"
    assert installer.install_mxc_release(install_dir) is False


def test_matching_install_is_reused_without_download(tmp_path, monkeypatch, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    assert installer.install_mxc_release(install_dir) is True
    monkeypatch.setattr(
        installer,
        "_download",
        lambda _destination, _pin: pytest.fail("verified installation was downloaded again"),
    )
    assert installer.install_mxc_release(install_dir) is False


def test_corrupt_wxc_is_replaced_safely(tmp_path, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    install_dir.mkdir(parents = True)
    (install_dir / "wxc-exec.exe").write_bytes(b"corrupt")
    assert installer.install_mxc_release(install_dir) is True
    installer.mxc_runtime.selected_runtime(package_root = install_dir)


def test_install_without_host_prep_is_refreshed(tmp_path, official_release):
    payload, _archive_bytes = official_release
    install_dir = tmp_path / "installed" / "windows-x86_64"
    install_dir.mkdir(parents = True)
    (install_dir / "wxc-exec.exe").write_bytes(payload)
    assert installer.install_mxc_release(install_dir) is True
    installer.mxc_runtime.selected_host_prep(package_root = install_dir)


def test_archive_checksum_mismatch_never_becomes_active(tmp_path, monkeypatch, official_release):
    install_dir = tmp_path / "installed" / "windows-x86_64"
    pin = installer.mxc_runtime.selected_pin()
    bad = installer.mxc_runtime.MxcPin(
        version = pin.version,
        revision = pin.revision,
        archive_size = pin.archive_size,
        archive_sha256 = "0" * 64,
        artifacts = pin.artifacts,
    )
    monkeypatch.setattr(installer.mxc_runtime, "selected_pin", lambda: bad)
    monkeypatch.setattr(installer.mxc_runtime, "target_pin", lambda: bad)
    with pytest.raises(installer.MxcInstallError, match = "checksum mismatch"):
        installer.install_mxc_release(install_dir)
    assert not install_dir.exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("other.exe", b"official-wxc", None)],
        [
            (_EXEC_MEMBER, b"official-wxc", None),
            (_EXEC_MEMBER, b"official-wxc", None),
        ],
        [("../package/bin/x64/wxc-exec.exe", b"official-wxc", None)],
        [(_EXEC_MEMBER, b"target", stat.S_IFLNK | 0o777)],
        [(_EXEC_MEMBER, b"package/other.exe", _HARDLINK)],
        [(_EXEC_MEMBER, b"official-wxc", None)],
        [
            (_EXEC_MEMBER, b"official-wxc", None),
            (_PREP_MEMBER, b"resized-host-prep", None),
        ],
        [
            (_EXEC_MEMBER, b"official-wxc", None),
            (_PREP_MEMBER, _HOST_PREP, None),
            (_PREP_MEMBER.upper(), _HOST_PREP, None),
        ],
    ],
    ids = [
        "missing",
        "duplicate",
        "traversal",
        "symlink",
        "hardlink",
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
        lambda _destination, _pin: pytest.fail("non-Windows installer attempted a download"),
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


def _windows_split(command_line: str) -> list[str]:
    """CommandLineToArgvW rules for the parameter string ShellExecuteExW and CreateProcess pass on."""
    args, current, quoted, i, pending = [], [], False, 0, False
    while i < len(command_line):
        char = command_line[i]
        if char == "\\":
            run = len(command_line[i:]) - len(command_line[i:].lstrip("\\"))
            if i + run < len(command_line) and command_line[i + run] == '"':
                pending = True
                current.append("\\" * (run // 2))
                if run % 2:
                    current.append('"')
                    i += run + 1
                    continue
                i += run
                continue
            current.append("\\" * run)
            i += run
            pending = True
            continue
        if char == '"':
            if quoted and command_line[i + 1 : i + 2] == '"':
                current.append('"')
                i += 2
                continue
            quoted, pending = not quoted, True
        elif char in " \t" and not quoted:
            if pending:
                args.append("".join(current))
            current, pending = [], False
        else:
            current.append(char)
            pending = True
        i += 1
    if pending:
        args.append("".join(current))
    return args


def _decoded_script(argv):
    received = _windows_split(installer.subprocess.list2cmdline(argv[1:]))
    assert received == argv[1:]
    assert received.index("-Command") == len(received) - 2
    return received[-1]


@pytest.fixture
def windows_dirs(monkeypatch):
    monkeypatch.setattr(installer, "_system_directory", lambda: "C:\\Windows\\System32")
    return os.path.join("C:\\Windows\\System32", "WindowsPowerShell", "v1.0", "powershell.exe")


@pytest.mark.parametrize("elevated", [True, False])
def test_host_prep_elevates_only_when_needed(monkeypatch, tmp_path, elevated, windows_dirs):
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
    assert [kind for kind, _argv in calls] == ["run" if elevated else "runas"]
    argv = calls[0][1]
    assert argv[0] == windows_dirs
    script = _decoded_script(argv)
    assert f"[IO.File]::Copy('{executable}', $exe)" in script
    assert "& $exe 'prepare-null-device' '--quiet'" in script


@pytest.mark.parametrize("elevated", [True, False])
def test_host_prep_script_is_passed_as_plain_text(monkeypatch, tmp_path, elevated, windows_dirs):
    calls: list = []
    monkeypatch.setattr(installer, "_is_elevated", lambda: elevated)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(argv) or subprocess_result(0),
    )
    monkeypatch.setattr(
        installer,
        "_run_elevated",
        lambda executable, arguments, _directory: calls.append([str(executable), *arguments]) or 0,
    )
    install = tmp_path / "O'Brien's dir" / "a b\\" / "wxc-host-prep.exe"
    assert installer._run_host_prep(install, "prepare-system-drive") == 0
    argv = calls[0]
    assert not any(
        arg.lower().startswith(("-enc", "-e", "-ec")) for arg in argv[1:] if arg != "-Command"
    )
    script = _decoded_script(argv)
    assert "\n" not in script and '"' not in script
    assert "[IO.File]::Copy('" + str(install).replace("'", "''") + "', $exe)" in script
    assert script == installer._host_prep_script(install, ["prepare-system-drive"])


def test_host_prep_refuses_a_path_it_cannot_quote(tmp_path):
    with pytest.raises(installer.MxcInstallError):
        installer._host_prep_script(
            tmp_path / 'a"b' / "wxc-host-prep.exe", ["prepare-system-drive"]
        )


def test_host_prep_runs_elevated_only_from_an_admin_only_copy(tmp_path):
    # The install dir is user-writable; host-prep LoadLibrary's dbghelp.dll from its own dir.
    script = installer._host_prep_script(tmp_path / "wxc-host-prep.exe", ["prepare-system-drive"])
    assert "GetFolderPath('CommonApplicationData')" in script
    assert "$env:ProgramData" not in script and "$env:SystemRoot" not in script
    # Windows PowerShell cannot load Get-FileHash when started from pwsh 7 (PSModulePath).
    assert "Get-FileHash" not in script and "Copy-Item" not in script
    assert "/inheritance:r" in script and "*S-1-5-32-544" in script and "*S-1-5-18" in script
    lock, empty, copy, hashed, run = (
        script.index("icacls.exe"),
        script.index("GetFileSystemEntries"),
        script.index("[IO.File]::Copy"),
        script.index(installer.mxc_runtime.artifact("prep")[2].upper()),
        script.index("& $exe"),
    )
    assert lock < empty < copy < hashed < run
    assert str(tmp_path) not in script[run:]


@pytest.mark.parametrize("code", sorted(installer._HOST_PREP_SCRIPT_ERRORS))
def test_host_prep_staging_failures_are_errors(monkeypatch, tmp_path, code, windows_dirs):
    monkeypatch.setattr(installer, "_is_elevated", lambda: True)
    monkeypatch.setattr(
        installer.subprocess, "run", lambda argv, **_kwargs: subprocess_result(code)
    )
    with pytest.raises(installer.MxcInstallError, match = installer._HOST_PREP_SCRIPT_ERRORS[code]):
        installer._run_host_prep(tmp_path / "wxc-host-prep.exe", "prepare-system-drive")


def test_elevated_host_prep_is_never_timed_out(monkeypatch, tmp_path, windows_dirs):
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


@pytest.mark.parametrize("elevated", [True, False])
def test_an_elevated_prepare_never_replays_the_user_journal(monkeypatch, tmp_path, elevated):
    # The journal is user-writable and names the ACLs wxc-exec rewrites on replay.
    seen: dict = {}
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setattr(installer, "_is_elevated", lambda: elevated)
    monkeypatch.setattr(
        installer.mxc_runtime,
        "probe_host_prep_steps",
        lambda **kwargs: seen.update(kwargs) or (),
    )
    assert installer.prepare_host(tmp_path) == ()
    assert seen["replay_journal"] is (not elevated)


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


def _auto_primary(
    monkeypatch,
    tmp_path,
    tier = "base-container",
    probe = None,
):
    """main() on a PC that selected 1.1.0 by itself; returns the pins install_mxc_release was asked for."""
    monkeypatch.delenv(installer.mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(installer.mxc_runtime, "_windows_release", lambda: (1, 29000, 1))
    monkeypatch.setattr(
        installer.mxc_runtime, "target_pin", lambda: installer.mxc_runtime.PRIMARY_PIN
    )
    monkeypatch.setattr(
        installer, "_is_elevated", lambda: False
    )  # GitHub's Windows runners are admin
    installed = []

    def install(
        install_dir,
        pin = None,
        *,
        qualify = None,
    ):
        installed.append(pin.version)
        if qualify is not None:
            qualify(Path(install_dir))
        return True

    monkeypatch.setattr(installer, "install_mxc_release", install)
    monkeypatch.setattr(
        installer.mxc_runtime, "probe_tier", probe or (lambda _root, pin = None: tier)
    )
    return installed


@pytest.mark.parametrize(
    ("tier", "falls_back"),
    [("base-container", False), ("appcontainer-dacl", True), (None, True)],
)
def test_primary_install_falls_back_to_legacy_when_the_probe_finds_no_built_in_container(
    tmp_path, monkeypatch, tier, falls_back
):
    installed = _auto_primary(monkeypatch, tmp_path, tier)
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 0
    legacy = installer.mxc_runtime.LEGACY_PIN.version
    primary = installer.mxc_runtime.PRIMARY_PIN.version
    assert installed == ([primary, legacy] if falls_back else [primary])
    assert installer.mxc_runtime.demoted_by_probe((1, 29000, 1)) is falls_back


def test_a_probe_that_cannot_run_never_demotes(tmp_path, monkeypatch):
    def broken(_root, pin = None):
        raise installer.mxc_runtime.MxcRuntimeUnavailable(
            "wxc-exec --probe exited 1 without a report"
        )

    installed = _auto_primary(monkeypatch, tmp_path, probe = broken)
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 0
    assert installed == [installer.mxc_runtime.PRIMARY_PIN.version]
    assert not installer.mxc_runtime.demoted_by_probe((1, 29000, 1))


def test_a_failed_fallback_records_no_demotion(tmp_path, monkeypatch, capsys):
    _auto_primary(monkeypatch, tmp_path, tier = None)

    def install(
        _dir,
        pin = None,
        *,
        qualify = None,
    ):
        if qualify is not None:
            qualify(Path(_dir))
        raise installer.MxcInstallError(f"MXC {pin.version} download failed")

    monkeypatch.setattr(installer, "install_mxc_release", install)
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 1
    # Nothing was replaced, so the next run asks 1.1.0 again instead of trusting a half-done fallback.
    assert not installer.mxc_runtime.demoted_by_probe((1, 29000, 1))
    assert "download failed" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "1", "yes", "0.8.0"])
def test_an_unknown_forced_pin_still_lets_the_probe_decide(tmp_path, monkeypatch, value):
    installed = _auto_primary(monkeypatch, tmp_path, tier = None)
    monkeypatch.setenv(installer.mxc_runtime.PIN_ENV, value)
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 0
    assert installed[-1] == installer.mxc_runtime.LEGACY_PIN.version


def test_a_forced_pin_is_never_second_guessed(tmp_path, monkeypatch):
    installed = []
    monkeypatch.setenv(installer.mxc_runtime.PIN_ENV, installer.mxc_runtime.PRIMARY_PIN.version)

    def install(
        _dir,
        pin = None,
        *,
        qualify = None,
    ):
        assert qualify is None
        installed.append(pin)
        return True

    monkeypatch.setattr(installer, "install_mxc_release", install)
    monkeypatch.setattr(
        installer.mxc_runtime, "probe_tier", lambda *_a, **_k: pytest.fail("probed a forced pin")
    )
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 0
    assert installed == [installer.mxc_runtime.PRIMARY_PIN]


def test_a_candidate_that_fails_qualification_never_replaces_the_working_runtime(
    tmp_path, monkeypatch, official_release
):
    install_dir = tmp_path / "rt"
    assert installer.install_mxc_release(install_dir)
    before = {path.name: path.read_bytes() for path in install_dir.iterdir()}
    newer = b"newer-wxc"
    archive = _archive([(_EXEC_MEMBER, newer, stat.S_IFREG | 0o644)])
    candidate = _pin(archive, {("x86_64", "exec"): newer}, version = "9.9.10")
    monkeypatch.setattr(
        installer, "_download", lambda destination, _pin: destination.write_bytes(archive)
    )
    seen = []

    def qualify(root):
        seen.append((root != install_dir, (root / "wxc-exec.exe").read_bytes()))
        raise installer.NoBuiltinContainer(None)

    with pytest.raises(installer.NoBuiltinContainer):
        installer.install_mxc_release(install_dir, candidate, qualify = qualify)
    # The candidate was probed in its staging folder, then dropped; the working runtime is untouched.
    assert seen == [(True, newer)]
    assert {path.name: path.read_bytes() for path in install_dir.iterdir()} == before
    assert not [
        path for path in tmp_path.iterdir() if path.name.startswith(".rt-")
    ]  # stage removed


def test_a_matching_install_is_still_qualified(tmp_path, official_release):
    install_dir = tmp_path / "rt"
    assert installer.install_mxc_release(install_dir)
    seen = []
    assert not installer.install_mxc_release(install_dir, qualify = seen.append)
    assert seen == [install_dir.resolve()]


@pytest.mark.parametrize("known", [True, False])
def test_the_runtime_being_replaced_replays_its_dacl_journal_first(tmp_path, monkeypatch, known):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    journal = installer.mxc_runtime.dacl_state_dir()
    (journal / "pending.json").write_text("{}", encoding = "utf-8")
    old = tmp_path / "rt"
    old.mkdir()
    payload = b"retiring-wxc"
    (old / "wxc-exec.exe").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        installer.mxc_runtime, "RETIRED_DACL_EXEC_SHA256", {digest} if known else set()
    )
    ran = []
    monkeypatch.setattr(installer, "_is_elevated", lambda: False)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda argv, **kwargs: ran.append((argv, kwargs["env"]["MXC_DACL_STATE_DIR"]))
        or subprocess_result(0),
    )
    installer._replay_retiring_journal(old)
    if known:
        assert ran == [([str(old / "wxc-exec.exe"), "--probe"], str(journal))]
    else:
        assert ran == []


def test_an_elevated_installer_never_replays_the_user_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    journal = installer.mxc_runtime.dacl_state_dir()
    (journal / "pending.json").write_text("{}", encoding = "utf-8")
    old = tmp_path / "rt"
    old.mkdir()
    (old / "wxc-exec.exe").write_bytes(b"retiring-wxc")
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RETIRED_DACL_EXEC_SHA256",
        {hashlib.sha256(b"retiring-wxc").hexdigest()},
    )
    monkeypatch.setattr(installer, "_is_elevated", lambda: True)
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *_a, **_k: pytest.fail("replayed as administrator")
    )
    installer._replay_retiring_journal(old)


def test_no_journal_means_no_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    installer.mxc_runtime.dacl_state_dir()
    old = tmp_path / "rt"
    old.mkdir()
    (old / "wxc-exec.exe").write_bytes(b"x")
    monkeypatch.setattr(installer, "_is_elevated", lambda: False)
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *_a, **_k: pytest.fail("replayed an empty journal")
    )
    installer._replay_retiring_journal(old)


def _retiring_runtime(
    tmp_path,
    monkeypatch,
    *,
    elevated = False,
    stderr = "",
    code = 0,
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    journal = installer.mxc_runtime.dacl_state_dir()
    (journal / "pending.json").write_text("{}", encoding = "utf-8")
    old = tmp_path / "rt"
    old.mkdir()
    (old / "wxc-exec.exe").write_bytes(b"retiring-wxc")
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RETIRED_DACL_EXEC_SHA256",
        {hashlib.sha256(b"retiring-wxc").hexdigest()},
    )
    monkeypatch.setattr(installer, "_is_elevated", lambda: elevated)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode = code, stdout = "{}", stderr = stderr),
    )
    return old


@pytest.mark.parametrize(
    ("elevated", "stderr", "code"),
    [
        (True, "", 0),
        (False, "DACL recovery: restored 3 ACE(s), 1 error(s)", 0),
        (False, "DACL recovery failed: access denied", 0),
        (False, "", 1),
    ],
    ids = ["elevated", "restore-error", "recovery-failed", "exit-code"],
)
def test_an_unfinished_replay_keeps_the_runtime_only_a_dacl_less_pin_would_replace(
    tmp_path, monkeypatch, capsys, elevated, stderr, code
):
    old = _retiring_runtime(tmp_path, monkeypatch, elevated = elevated, stderr = stderr, code = code)
    # 1.1.0 can never replay that journal: the swap must wait.
    with pytest.raises(installer.MxcInstallError, match = "kept the current MXC runtime"):
        installer._replay_retiring_journal(old, installer.mxc_runtime.PRIMARY_PIN)
    # 1.0.1 replays the same journal itself, so it may go ahead after a warning.
    installer._replay_retiring_journal(old, installer.mxc_runtime.LEGACY_PIN)
    assert "[mxc-prebuilt]" in capsys.readouterr().out


@pytest.mark.parametrize("drained", [True, False])
def test_a_clean_replay_lets_a_dacl_less_pin_replace_the_runtime_once_the_journal_is_empty(
    tmp_path, monkeypatch, drained
):
    old = _retiring_runtime(
        tmp_path, monkeypatch, stderr = "DACL recovery: restored 2 ACE(s), 0 error(s)"
    )
    journal = installer.mxc_runtime.dacl_state_path()

    def replay(*_a, **_k):
        if drained:
            for entry in journal.glob("*.json"):
                entry.unlink()
        return SimpleNamespace(
            returncode = 0, stdout = "{}", stderr = "DACL recovery: restored 2 ACE(s), 0 error(s)"
        )

    monkeypatch.setattr(installer.subprocess, "run", replay)
    if drained:
        installer._replay_retiring_journal(old, installer.mxc_runtime.PRIMARY_PIN)
    else:
        # A live run's entry is skipped without an error: 1.1.0 could never replay it later.
        with pytest.raises(installer.MxcInstallError, match = "still lists entries"):
            installer._replay_retiring_journal(old, installer.mxc_runtime.PRIMARY_PIN)


def test_a_failed_replay_leaves_the_retiring_runtime_installed(
    tmp_path, monkeypatch, official_release
):
    install_dir = tmp_path / "rt"
    assert installer.install_mxc_release(install_dir)
    before = {path.name: path.read_bytes() for path in install_dir.iterdir()}
    monkeypatch.setattr(
        installer.mxc_runtime,
        "RETIRED_DACL_EXEC_SHA256",
        {hashlib.sha256(before["wxc-exec.exe"]).hexdigest()},
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    (installer.mxc_runtime.dacl_state_dir() / "pending.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(installer, "_is_elevated", lambda: False)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode = 0, stdout = "{}", stderr = "DACL recovery failed"),
    )
    monkeypatch.setattr(
        installer.mxc_runtime,
        "_open_artifact_guard",
        lambda _path: SimpleNamespace(close = lambda: None),
    )
    newer = b"newer-wxc"
    archive = _archive([(_EXEC_MEMBER, newer, stat.S_IFREG | 0o644)])
    candidate = _pin(archive, {("x86_64", "exec"): newer}, version = "9.9.10")
    monkeypatch.setattr(
        installer, "_download", lambda destination, _pin: destination.write_bytes(archive)
    )
    with pytest.raises(installer.MxcInstallError):
        installer.install_mxc_release(install_dir, candidate)
    assert {path.name: path.read_bytes() for path in install_dir.iterdir()} == before


def _no_verdict(_root, pin = None):
    raise installer.mxc_runtime.MxcRuntimeUnavailable("wxc-exec --probe exited 1 without a report")


@pytest.mark.parametrize("legacy_installed", [True, False])
def test_no_verdict_keeps_a_working_legacy_runtime(tmp_path, monkeypatch, legacy_installed):
    install_dir = tmp_path / "rt"
    monkeypatch.setattr(installer, "_is_elevated", lambda: False)
    monkeypatch.setattr(installer.mxc_runtime, "probe_tier", _no_verdict)
    monkeypatch.setattr(
        installer,
        "_already_installed",
        lambda root, pin: legacy_installed
        and root == install_dir
        and pin is installer.mxc_runtime.LEGACY_PIN,
    )
    stage = tmp_path / ".rt-stage"
    if legacy_installed:
        # An unproven 1.1.0 never replaces a 1.0.1 that works; the next install asks again.
        with pytest.raises(installer.KeptCurrentRuntime):
            installer._require_builtin_container(stage, install_dir = install_dir)
    else:
        installer._require_builtin_container(stage, install_dir = install_dir)
    # An installed 1.1.0 that cannot be probed stays: there is nothing better to keep.
    installer._require_builtin_container(install_dir, install_dir = install_dir)


def test_main_keeping_the_current_runtime_is_not_an_error(tmp_path, monkeypatch, capsys):
    _auto_primary(monkeypatch, tmp_path)

    def install(
        _dir,
        pin = None,
        *,
        qualify = None,
    ):
        raise installer.KeptCurrentRuntime(
            "MXC 1.1.0 could not be probed; kept the installed MXC 1.0.1"
        )

    monkeypatch.setattr(installer, "install_mxc_release", install)
    assert installer.main(["--install-dir", str(tmp_path / "rt")]) == 0
    assert "kept the installed MXC 1.0.1" in capsys.readouterr().out
    assert not installer.mxc_runtime.demoted_by_probe((1, 29000, 1))


@pytest.mark.parametrize("unreadable", ["journal", "runtime"])
def test_an_unreadable_journal_or_runtime_blocks_only_a_dacl_less_swap(
    tmp_path, monkeypatch, capsys, unreadable
):
    old = _retiring_runtime(tmp_path, monkeypatch)
    if unreadable == "journal":
        real_iterdir = Path.iterdir

        def iterdir(self):
            if self == installer.mxc_runtime.dacl_state_path():
                raise PermissionError("access denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", iterdir)
    else:

        def unreadable_hash(_path):
            raise PermissionError("access denied")

        monkeypatch.setattr(installer.mxc_runtime, "_sha256_file", unreadable_hash)
    with pytest.raises(installer.MxcInstallError, match = "could not be read"):
        installer._replay_retiring_journal(old, installer.mxc_runtime.PRIMARY_PIN)
    installer._replay_retiring_journal(old, installer.mxc_runtime.LEGACY_PIN)
    assert "could not be read" in capsys.readouterr().out


def test_the_marker_follows_the_install_dir_studio_home(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    install_dir = tmp_path / "custom" / "studio" / "mxc-runtime" / "windows-x86_64"
    installer._align_studio_home(install_dir)
    # setup.ps1 resolves UNSLOTH_HOME for --install-dir; the marker and journal now land beside it.
    assert installer.mxc_runtime._studio_root() == tmp_path / "custom" / "studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "explicit"))
    installer._align_studio_home(install_dir)
    assert installer.mxc_runtime._studio_root() == tmp_path / "explicit"


def test_prepare_host_reads_the_pin_beside_a_custom_install_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    install_dir = tmp_path / "custom" / "studio" / "mxc-runtime" / "windows-x86_64"
    seen = []
    monkeypatch.setattr(
        installer,
        "prepare_host",
        lambda root: seen.append(installer.mxc_runtime._studio_root()) or (),
    )
    assert installer.main(["--prepare-host", "--install-dir", str(install_dir)]) == 0
    # The demotion marker that picked 1.0.1 there is read from that Studio home, not the default one.
    assert seen == [tmp_path / "custom" / "studio"]
