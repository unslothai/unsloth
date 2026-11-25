# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Select and verify Studio's pinned Microsoft WXC executable."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

MXC_SCHEMA_VERSION = "1.0.0"
PROFILE_ID = "unsloth-mxc-windows-processcontainer-v3"
RELEASE_REPOSITORY = "microsoft/mxc"
_NPM_TARBALL = "https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{version}.tgz"


@dataclass(frozen = True, eq = False)
class MxcPin:
    """One pinned @microsoft/mxc-sdk npm release: the archive and each approved (architecture, kind) member."""

    version: str
    revision: str
    archive_size: int
    archive_sha256: str
    # (architecture, "exec" | "prep") -> (archive member, size, sha256)
    artifacts: dict

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def asset(self) -> str:
        return f"@microsoft/mxc-sdk@{self.version}"

    @property
    def url(self) -> str:
        return _NPM_TARBALL.format(version = self.version)

    @property
    def has_host_prep(self) -> bool:
        return any(kind == "prep" for _architecture, kind in self.artifacts)


# Native BaseContainer only (1.1.0 removed the AppContainer + DACL tier, microsoft/mxc#1558).
PRIMARY_PIN = MxcPin(
    version = "1.1.0",
    revision = "c66201839d63cae2239db5e92f1f1d6076cf5181",
    archive_size = 40_567_504,
    archive_sha256 = "22806bcc1ff61768c47c8463905129cd479a96bcfcd54c9acf93f316e4b82e19",
    artifacts = {
        ("x86_64", "exec"): (
            "package/bin/x64/wxc-exec.exe",
            12_700_520,
            "6d466c6b0109bab4bcace6738e11f886d801ea977346db49707488fcbd3afc8a",
        ),
        ("arm64", "exec"): (
            "package/bin/arm64/wxc-exec.exe",
            7_278_392,
            "afbffebc7ba0ef06c6cd24eff0a3792db3e25faca388e4aaca092745ea328fd6",
        ),
    },
)
# Newest release with the AppContainer + DACL tier and wxc-host-prep.
LEGACY_PIN = MxcPin(
    version = "1.0.1",
    revision = "fda29ac63bc306b7f97f3b86b4b63d44f7f40cf2",
    archive_size = 38_256_855,
    archive_sha256 = "416d9607c9d4652395d95c3d9c5f5605f1f2f845da1a63228bee41b7392cef49",
    artifacts = {
        ("x86_64", "exec"): (
            "package/bin/x64/wxc-exec.exe",
            12_152_632,
            "090f5d1f7bb93104ab08dcd4e8034ef954aeefcdaf6658dde5f7dff4cfc12f3a",
        ),
        ("x86_64", "prep"): (
            "package/bin/x64/wxc-host-prep.exe",
            1_275_200,
            "c18783ad3dd62c0a54530b25192b7c4015706b3f518cd9619259ae9750d4ab6e",
        ),
        ("arm64", "exec"): (
            "package/bin/arm64/wxc-exec.exe",
            5_945_656,
            "91aa4171e8809f33f19622c381116188eab7d8c8cffad231230a31b58146ee77",
        ),
        ("arm64", "prep"): (
            "package/bin/arm64/wxc-host-prep.exe",
            804_704,
            "74ccae215a9c5e81215bc7124aa31cdd7a0b2215a79b5cf8865865bbdaaf93e2",
        ),
    },
)
# Earlier pins, recognised only to replay their DACL journal before an upgrade replaces them.
RETIRED_DACL_EXEC_SHA256 = {
    "6049c64723af1173c3739dc6cd6b2f33f6c021bb2832c4216233cba7f71aee9a",  # 0.8.0 x64
    "dde1c592270e9a659b01dccad70362da7b99fec114885fa4d625507aa775a503",  # 0.8.0 arm64
}
PINS = (PRIMARY_PIN, LEGACY_PIN)
PIN_ENV = "UNSLOTH_MXC_PIN"
# First UBR with BaseContainer per release (MXC docs/backends/process-container/os-version-support.md).
BUILTIN_CONTAINER_MIN_UBR = {26100: 9278, 26200: 9278, 26300: 9550, 28000: 2804}
_DEMOTED_MARKER = "pin-demoted.json"
_ARCHITECTURES = ("x86_64", "arm64")
HOST_PREP_STEPS = ("prepare-system-drive", "prepare-null-device")
HOST_PREP_PROBE_SECONDS = 10.0
DACL_RECOVERY_SECONDS = 60.0
# mxc_probe never uses MXC below this build, and some of those (Server 2019) cannot load wxc-exec: the
# loader shows a modal "System Error" dialog instead of returning an exit code (#16169).
MIN_WINDOWS_BUILD = 26100

_WXC_EXEC_NAME = "wxc-exec.exe"
_WXC_HOST_PREP_NAME = "wxc-host-prep.exe"
_ARTIFACT_NAMES = {"exec": _WXC_EXEC_NAME, "prep": _WXC_HOST_PREP_NAME}
logger = logging.getLogger(__name__)
_lock = threading.RLock()


class MxcRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen = True)
class RuntimeInfo:
    path: Path
    sha256: str
    architecture: str = "x86_64"
    pin: MxcPin = PRIMARY_PIN

    @property
    def identity(self) -> str:
        material = {
            "architecture": self.architecture,
            "mxcRevision": self.pin.revision,
            "releaseTag": self.pin.tag,
            "sha256": self.sha256,
            "size": _artifact(self.architecture, "exec", self.pin)[1],
        }
        encoded = json.dumps(material, sort_keys = True, separators = (",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class RuntimeLease:
    info: RuntimeInfo
    _guard: object
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        close = getattr(self._guard, "close", None)
        if close is not None:
            close()
        self._guard = None

    def __enter__(self) -> RuntimeLease:
        return self

    def __exit__(self, *_args) -> None:
        self.release()


def _studio_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    return Path(override).expanduser() if override else Path.home() / ".unsloth" / "studio"


def _installed_package_root() -> Path:
    try:
        architecture = _expected_architecture()
    except MxcRuntimeUnavailable:
        architecture = "x86_64"
    return _studio_root() / "mxc-runtime" / f"windows-{architecture}"


def dacl_state_path() -> Path:
    """WXC's DACL restore journal: one fixed place, so every start (probe included) reaps the same orphans."""
    # Absolute: wxc-exec runs with the runtime dir as its cwd, so a relative home would split the journal.
    return Path(os.path.abspath(_studio_root() / "mxc-runtime" / "dacl-restore"))


def dacl_state_dir() -> Path:
    path = dacl_state_path()
    path.mkdir(parents = True, exist_ok = True)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MxcRuntimeUnavailable("the managed wxc-exec.exe could not be read") from exc
    return digest.hexdigest()


def _require_plain_directory(path: Path) -> Path:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise MxcRuntimeUnavailable("the managed MXC runtime is not installed") from exc
    if not path.is_dir() or path.is_symlink() or attributes & 0x400:
        raise MxcRuntimeUnavailable(
            "the managed MXC runtime must be a non-reparse directory",
        )
    return path.resolve()


_machine_cache: tuple = (None, None)


def _expected_architecture() -> str:
    """The OS architecture, not the interpreter's: an x64 Python on an ARM64 PC still needs the ARM64 build."""
    global _machine_cache
    from . import wsl_host

    # Once per process (it loads kernel32 through ctypes); keyed on the function so a replacement is asked.
    if _machine_cache[0] is not wsl_host.native_machine:
        _machine_cache = (wsl_host.native_machine, wsl_host.native_machine())
    machine = _machine_cache[1]
    if machine not in _ARCHITECTURES:
        raise MxcRuntimeUnavailable(
            f"Microsoft MXC is supported only on Windows x64 and ARM64, not {machine or 'unknown'}",
        )
    return machine


@functools.lru_cache(maxsize = 1)  # a Windows update applies at reboot, so once per process
def _windows_release() -> tuple[int, int, int] | None:
    """(product type, build, UBR) of this Windows; UBR is -1 when only it is unreadable, None when nothing is."""
    try:
        version = sys.getwindowsversion()
        product_type, build = int(version.product_type), int(version.build)
    except (ValueError, TypeError, AttributeError):
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:
            ubr = int(winreg.QueryValueEx(key, "UBR")[0])
    except (OSError, ValueError, TypeError, ImportError):
        ubr = -1
    return product_type, build, ubr


def builtin_container_supported(release: tuple[int, int, int] | None) -> bool | None:
    """Whether this Windows has the built-in container MXC 1.1.0 needs; None when it cannot tell."""
    if release is None:
        return None
    product_type, build, ubr = release
    if product_type != 1:  # not VER_NT_WORKSTATION: Windows Server has no BaseContainer
        return False
    if build in BUILTIN_CONTAINER_MIN_UBR:
        if ubr < 0:
            return None  # the update revision decides, and it is unreadable: the installer's probe does
        return ubr >= BUILTIN_CONTAINER_MIN_UBR[build]
    if build < min(BUILTIN_CONTAINER_MIN_UBR):
        return False
    # Newer than the table, or an Insider build between rows: the installer's --probe decides.
    return True if build > max(BUILTIN_CONTAINER_MIN_UBR) else None


def _demoted_marker_path() -> Path:
    return _studio_root() / "mxc-runtime" / _DEMOTED_MARKER


def _release_label(release: tuple[int, int, int] | None) -> str:
    # Unreadable is a release of its own: a fallback recorded there must keep applying there.
    return "unknown" if release is None else f"{release[0]}:{release[1]}.{release[2]}"


def demoted_by_probe(release: tuple[int, int, int] | None = None) -> bool:
    """The installer saw MXC 1.1.0 report no built-in container on this exact Windows release."""
    label = _release_label(_windows_release() if release is None else release)
    try:
        recorded = json.loads(_demoted_marker_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return False
    # Keyed on the release and the demoted pin: a Windows update or a newer primary pin is tried again.
    return (
        isinstance(recorded, dict)
        and recorded.get("windows") == label
        and recorded.get("from") == PRIMARY_PIN.version
    )


def record_probe_demotion(reason: str) -> None:
    label = _release_label(_windows_release())
    path = _demoted_marker_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"windows": label, "from": PRIMARY_PIN.version, "reason": reason[:500]}),
        encoding = "utf-8",
    )
    os.replace(temporary, path)  # a reader never sees half a marker


def forced_pin() -> MxcPin | None:
    """The pin UNSLOTH_MXC_PIN names, if it names one of PINS; anything else is ignored."""
    override = (os.environ.get(PIN_ENV) or "").strip()
    return next((pin for pin in PINS if override == pin.version), None)


def target_pin() -> MxcPin:
    """The pin the installer puts on this PC: MXC 1.1.0 where Windows has the built-in container, else 1.0.1."""
    forced = forced_pin()
    if forced is not None:
        return forced
    if sys.platform != "win32":
        return PRIMARY_PIN
    release = _windows_release()
    if builtin_container_supported(release) is False or demoted_by_probe(release):
        return LEGACY_PIN
    return PRIMARY_PIN


def _legacy_still_installed() -> bool:
    """MXC 1.0.1's files, by size alone, in the managed folder; every launch still verifies the bytes."""
    try:
        architecture = _expected_architecture()
        root = _installed_package_root()
        sizes = {kind: (root / name).lstat().st_size for kind, name in _ARTIFACT_NAMES.items()}
    except (OSError, MxcRuntimeUnavailable):
        return False
    return all(sizes[kind] == LEGACY_PIN.artifacts[(architecture, kind)][1] for kind in sizes)


def selected_pin() -> MxcPin:
    """target_pin(), but an installed 1.0.1 (which also uses the built-in container) runs until replaced."""
    target = target_pin()
    if target is PRIMARY_PIN and forced_pin() is None and sys.platform == "win32":
        if _legacy_still_installed():
            return LEGACY_PIN
    return target


def _artifact(
    architecture: str,
    kind: str,
    pin: MxcPin | None = None,
) -> tuple[str, int, str]:
    pin = pin or selected_pin()
    try:
        return pin.artifacts[(architecture, kind)]
    except KeyError:
        raise MxcRuntimeUnavailable(
            f"Microsoft MXC {pin.version} has no {'wxc-host-prep' if kind == 'prep' else 'wxc-exec'} "
            f"for {architecture}"
        ) from None


def artifact(kind: str, pin: MxcPin | None = None) -> tuple[str, int, str]:
    """(package member, size, sha256) of ``exec`` or ``prep`` for this PC's architecture."""
    return _artifact(_expected_architecture(), kind, pin)


def _validate_artifact(
    package_root: Path,
    name: str,
    kind: str,
    pin: MxcPin | None = None,
) -> RuntimeInfo:
    pin = pin or selected_pin()
    architecture = _expected_architecture()
    _member, size, sha256 = _artifact(architecture, kind, pin)
    root = _require_plain_directory(package_root)
    executable = root / name
    if not executable.is_file() or executable.is_symlink():
        raise MxcRuntimeUnavailable(f"the managed {name} is missing or is not a regular file")
    try:
        metadata = executable.stat()
    except OSError as exc:
        raise MxcRuntimeUnavailable(f"the managed {name} is unreadable") from exc
    if getattr(metadata, "st_nlink", 1) != 1:
        raise MxcRuntimeUnavailable(
            f"the managed {name} has an unapproved hard link",
        )
    if metadata.st_size != size:
        raise MxcRuntimeUnavailable(f"the managed {name} size is not approved")
    digest = _sha256_file(executable)
    if digest != sha256:
        raise MxcRuntimeUnavailable(f"the managed {name} digest is not approved")
    return RuntimeInfo(path = executable.resolve(), sha256 = digest, architecture = architecture, pin = pin)


def _validate_runtime(package_root: Path, pin: MxcPin | None = None) -> RuntimeInfo:
    return _validate_artifact(package_root, _WXC_EXEC_NAME, "exec", pin)


def _validate_host_prep(package_root: Path, pin: MxcPin | None = None) -> RuntimeInfo:
    return _validate_artifact(package_root, _WXC_HOST_PREP_NAME, "prep", pin)


def selected_runtime(*, package_root: Path | None = None, pin: MxcPin | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    _expected_architecture()
    return _validate_runtime(package_root or _installed_package_root(), pin)


def selected_host_prep(*, package_root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("MXC host preparation is Windows-only")
    _expected_architecture()
    return _validate_host_prep(package_root or _installed_package_root())


def wxc_path() -> Path:
    return selected_runtime().path


_identity_cache: dict[tuple, tuple[tuple, RuntimeInfo]] = {}


def _file_stamp(path: Path) -> tuple:
    metadata = path.lstat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        getattr(metadata, "st_birthtime_ns", None),
        metadata.st_ctime_ns,  # creation time on Windows before Python 3.12
        getattr(metadata, "st_nlink", 1),
    )


def verified_installation_identity() -> str:
    """A full verification: what status, setup and repair decide on."""
    try:
        return selected_runtime().identity
    except Exception:
        forget_verified_runtime()
        raise


def installation_identity() -> str:
    """For the per-call fingerprints: rehashed only when the file's stamp changes. Never authorizes a
    launch (acquire_runtime verifies the bytes) nor reports the runtime healthy (verified_installation_identity)."""
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    architecture = _expected_architecture()
    pin = selected_pin()
    _member, size, sha256 = _artifact(architecture, "exec", pin)
    root = _require_plain_directory(_installed_package_root())
    path = root / _WXC_EXEC_NAME
    try:
        before = _file_stamp(path)
    except OSError as exc:
        raise MxcRuntimeUnavailable(
            f"the managed {_WXC_EXEC_NAME} is missing or unreadable"
        ) from exc
    key = (str(path), architecture, pin.version, size, sha256)
    with _lock:
        hit = _identity_cache.get(key)
        if hit is not None and hit[0] == before:
            return hit[1].identity
        _identity_cache.pop(key, None)
    info = _validate_runtime(root, pin)
    try:
        after = _file_stamp(path)
    except OSError:
        return info.identity
    if after == before:
        with _lock:
            _identity_cache[key] = (before, info)
    return info.identity


def forget_verified_runtime() -> None:
    """Drop cached identities, e.g. after the installer swaps the runtime folder."""
    with _lock:
        _identity_cache.clear()


class _WindowsHandleGuard:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.windll.kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            close_handle(self.handle)
            self.handle = 0


def _open_artifact_guard(path: Path) -> object:
    if os.name != "nt":
        return path.open("rb")
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(path), 0x80000000, 0x1, None, 3, 0x80, None)
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise MxcRuntimeUnavailable(f"the managed {path.name} could not be locked for launch")
    return _WindowsHandleGuard(int(handle))


def _require_supported_windows() -> None:
    windows_version = getattr(sys, "getwindowsversion", None)
    if windows_version is not None and windows_version().build < MIN_WINDOWS_BUILD:
        raise MxcRuntimeUnavailable(
            f"Microsoft MXC needs Windows 11 build {MIN_WINDOWS_BUILD} or newer"
        )


def _acquire(select, package_root: Path | None, name: str) -> RuntimeLease:
    _require_supported_windows()
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    _expected_architecture()
    with _lock:
        # One digest, under the guard: it denies write, rename and delete for the whole lease.
        root = _require_plain_directory(package_root or _installed_package_root())
        candidate = root / name
        try:
            reparse = getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400
        except OSError:
            reparse = True
        if reparse or not candidate.is_file() or candidate.is_symlink():
            raise MxcRuntimeUnavailable(f"the managed {name} is missing or is not a regular file")
        # Before the guard opens, so a path redirected meanwhile no longer matches.
        expected = candidate.resolve()
        guard = _open_artifact_guard(candidate)
        try:
            info = select(package_root = root)
            if info.path != expected:
                raise MxcRuntimeUnavailable(f"the managed {name} changed during acquisition")
        except Exception:
            guard.close()
            raise
        return RuntimeLease(info = info, _guard = guard)


def acquire_runtime(*, package_root: Path | None = None, pin: MxcPin | None = None) -> RuntimeLease:
    def select(*, package_root: Path | None) -> RuntimeInfo:
        return selected_runtime(package_root = package_root, pin = pin)

    return _acquire(select, package_root, _WXC_EXEC_NAME)


def acquire_host_prep(*, package_root: Path | None = None) -> RuntimeLease:
    """Pinned wxc-host-prep held deny-write, so the elevated launch runs the verified bytes."""
    return _acquire(selected_host_prep, package_root, _WXC_HOST_PREP_NAME)


def _run_wxc_probe(
    package_root: Path | None,
    env: dict[str, str] | None,
    *,
    replay_journal: bool = True,
    timeout: float = HOST_PREP_PROBE_SECONDS,
    pin: MxcPin | None = None,
):
    # --probe reaps orphaned ACEs first: point it at Studio's journal, not %LOCALAPPDATA%'s.
    env = dict(os.environ if env is None else env)
    with tempfile.TemporaryDirectory(prefix = "unsloth-mxc-empty-journal-") as empty:
        # An elevated probe must not replay a user-writable journal: it names the ACLs to rewrite.
        env["MXC_DACL_STATE_DIR"] = str(dacl_state_dir()) if replay_journal else empty
        with acquire_runtime(package_root = package_root, pin = pin) as lease:
            return subprocess.run(
                [str(lease.info.path), "--probe"],
                stdin = subprocess.DEVNULL,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                cwd = str(lease.info.path.parent),
                env = env,
                timeout = timeout,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check = False,
            )


def dacl_recovery_clean(completed) -> bool:
    """Whether a `wxc-exec --probe` run replayed the DACL journal without a restore error."""
    stderr = completed.stderr or ""
    # main.rs prints "DACL recovery: ... N error(s)" only when there was work, or "DACL recovery failed".
    report = re.search(r"DACL recovery: .*?(\d+) error\(s\)", stderr)
    # Deleting a journal that is already gone is not a restore error: a concurrent run finished it.
    already_gone = len(re.findall(r"^\s*remove .*\(os error 2\)\s*$", stderr, re.MULTILINE))
    return (
        completed.returncode == 0
        and "DACL recovery failed" not in stderr
        and (report is None or int(report.group(1)) <= already_gone)
    )


def recover_dacl_state(env: dict[str, str] | None = None) -> bool:
    """Replay the DACL journal now; True only when wxc-exec reports no recovery error."""
    try:
        # Longer than the advice probe: it runs right after a kill, often beside other DACL launches.
        completed = _run_wxc_probe(None, env, timeout = DACL_RECOVERY_SECONDS)
    except Exception as exc:  # noqa: BLE001 - an unknown outcome is not a clean one
        logger.warning("MXC DACL recovery did not run: %s", exc)
        return False
    clean = dacl_recovery_clean(completed)
    if not clean:
        logger.warning(
            "MXC DACL recovery reported a problem: %s", (completed.stderr or "").strip()[-500:]
        )
    return clean


def probe_host_prep_steps(
    *,
    package_root: Path | None = None,
    env: dict[str, str] | None = None,
    replay_journal: bool = True,
) -> tuple[str, ...] | None:
    """Host preparation `wxc-exec --probe` reports missing; None when it cannot tell."""
    if not selected_pin().has_host_prep:
        return ()  # MXC 1.1.0 has no AppContainer + DACL tier, so nothing to prepare
    try:
        completed = _run_wxc_probe(package_root, env, replay_journal = replay_journal)
        warnings = json.loads(completed.stdout).get("warnings")
    except Exception:  # noqa: BLE001 - advice only, never a capability verdict
        return None
    if completed.returncode != 0 or not isinstance(warnings, list):
        return None
    # MXC names the verb in each Tier 3 warning (fallback_detector.rs push_host_prep_warnings).
    return tuple(
        step
        for step in HOST_PREP_STEPS
        if any(isinstance(item, str) and f"wxc-host-prep {step}" in item for item in warnings)
    )


def probe_tier(
    package_root: Path,
    env: dict[str, str] | None = None,
    *,
    pin: MxcPin | None = None,
) -> str | None:
    """The tier `wxc-exec --probe` names, or None; raises MxcRuntimeUnavailable when it prints no report."""
    # The installer may run elevated, and the tier needs no journal: never replay Studio's here.
    completed = _run_wxc_probe(
        package_root, env, replay_journal = False, timeout = DACL_RECOVERY_SECONDS, pin = pin
    )
    try:
        report = json.loads(completed.stdout)
    except ValueError:
        report = None
    if completed.returncode != 0 or not isinstance(report, dict):
        raise MxcRuntimeUnavailable(
            f"wxc-exec --probe exited {completed.returncode} without a report: "
            f"{(completed.stderr or completed.stdout or '').strip()[-300:]}"
        )
    tier = report.get("tier")
    return tier if isinstance(tier, str) and tier else None
