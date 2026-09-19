# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Manifest-owned immutable runtime generations for the Windows MXC supervisor."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import threading

MXC_REVISION = "ca7ea12ac6bd9f5420d6adecb37e32a8158da476"
MXC_SCHEMA_VERSION = "0.8.0-alpha"
MXC_PATCH_SHA256 = "4741ee9db1f389e6c43f8badcbf2c53076e3d7e5e546d6abd474459c30fff77c"
MXC_PATCHED_TREE = "2c5f5245a4676173f5e4c9e03576eb68d02c47d2"
PATCHED_CARGO_LOCK_SHA256 = "1313685ae6b926cde46a96d1c481e326e059b3b649f96205bb78674075a8c594"
RUNNER_PROTOCOL_VERSION = 1
PROFILE_ID = "unsloth-mxc-windows-basecontainer-v1"
PROFILE_VERSION = 1
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_VERSION = "unsloth-mxc-preview-1"
EXPECTED_TARGET = "x86_64-pc-windows-msvc"
EXPECTED_FEATURES = ("mxc-no-dacl-api",)
_MANIFEST_NAME = "runtime-manifest.json"
_RUNNER_NAME = "unsloth-mxc-runner.exe"
_TRUST_NAME = "runtime-trust.json"
_CURRENT_NAME = "current.json"
_STATE_NAME = "runtime-state.json"
_DISABLED_NAME = "runtime-disabled.json"
_PACKAGE_TRUST_NAME = "runtime-package.json"
_MANAGEMENT_LOCK_NAME = ".runtime-management.lock"
_GENERATION_RE = re.compile(r"^mxc-[0-9a-f]{12}-[0-9a-f]{16}$")
_MAX_MANIFEST = 256 * 1024


class MxcRuntimeUnavailable(RuntimeError):
    def __init__(self, message: str, *, code: str = "runtime_unavailable") -> None:
        super().__init__(message)
        self.code = code


class RuntimeState(str, Enum):
    READY = "ready"
    NOT_INSTALLED = "not_installed"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"
    UNTRUSTED = "untrusted"
    REPAIR_REQUIRED = "repair_required"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNSUPPORTED_WINDOWS_BUILD = "unsupported_windows_build"


@dataclass(frozen=True)
class RuntimeStatus:
    state: RuntimeState
    reason: str
    generation: str | None = None
    runner_sha256: str | None = None
    previous_generation: str | None = None
    last_operation: str | None = None
    repair_available: bool = False
    repair_requires_user_action: bool = False


@dataclass(frozen=True)
class RuntimeInfo:
    generation: str
    path: Path
    manifest_path: Path
    runner_sha256: str
    manifest_sha256: str
    runner_source_identity: str
    development: bool
    production_ready: bool

    @property
    def identity(self) -> str:
        material = {
            "generation": self.generation,
            "manifestSha256": self.manifest_sha256,
            "mxcRevision": MXC_REVISION,
            "protocolVersion": RUNNER_PROTOCOL_VERSION,
            "runnerSha256": self.runner_sha256,
            "runnerSourceIdentity": self.runner_source_identity,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        guard = self._guard
        self._guard = None
        close = getattr(guard, "close", None)
        if close is not None:
            close()
        _release_owner(self.info.generation)

    def __enter__(self) -> RuntimeLease:
        return self

    def __exit__(self, *_args) -> None:
        self.release()


_lock = threading.RLock()
_management_locks_lock = threading.Lock()
_management_locks: dict[str, threading.Lock] = {}
_owners: dict[str, int] = {}
_retire_pending: dict[str, Path] = {}


def _native_root() -> Path:
    return Path(__file__).resolve().parents[3] / "native" / "mxc-runner"


def _packaged_root() -> Path:
    return _native_root() / "bin"


def _approved_package_root() -> Path:
    return _native_root() / "package" / "windows-x86_64"


def _invalidate_probe_cache() -> None:
    try:
        from . import mxc_probe

        mxc_probe.invalidate_cache()
    except (ImportError, AttributeError):
        pass


def _resolved_runtime_root(root: Path | None) -> Path:
    candidate = root or _packaged_root()
    if candidate.exists():
        _require_plain_directory(candidate, "runtime root")
    return candidate.resolve()


def _development_runtime() -> Path:
    return _native_root() / "target" / "reproducible-runtime"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MxcRuntimeUnavailable("the MXC runner artifact could not be read") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MxcRuntimeUnavailable(f"the MXC {label} is missing or unreadable") from exc
    if len(data) > _MAX_MANIFEST:
        raise MxcRuntimeUnavailable(f"the MXC {label} exceeds its size bound")
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise MxcRuntimeUnavailable(f"the MXC {label} is malformed") from exc
    if not isinstance(value, dict):
        raise MxcRuntimeUnavailable(f"the MXC {label} must be an object")
    return value, hashlib.sha256(data).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise MxcRuntimeUnavailable(f"the MXC runtime path is unreadable: {path}") from exc
    return bool(attributes & 0x400) or path.is_symlink()


def _require_plain_directory(path: Path, label: str) -> None:
    if not path.is_dir() or _is_reparse_point(path):
        raise MxcRuntimeUnavailable(
            f"the MXC {label} must be a non-reparse directory",
            code="runtime_untrusted",
        )


def _management_thread_lock(root: Path) -> threading.Lock:
    key = os.path.normcase(os.fspath(root.resolve()))
    with _management_locks_lock:
        return _management_locks.setdefault(key, threading.Lock())


@contextmanager
def _management_guard(root: Path):
    """Serialize runtime metadata mutation without holding execution leases."""
    root.mkdir(parents=True, exist_ok=True)
    lock = _management_thread_lock(root)
    with lock:
        lock_path = root / _MANAGEMENT_LOCK_NAME
        with lock_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _expected_architecture() -> str:
    machine = platform.machine().casefold()
    if machine not in {"amd64", "x86_64"}:
        raise MxcRuntimeUnavailable(
            f"the packaged MXC runtime does not support this architecture: {machine or 'unknown'}",
            code="runtime_incompatible",
        )
    return "x86_64"


def _validate_identity(runner: Path, manifest: dict) -> None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [str(runner), "--identity"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
            close_fds=True,
            creationflags=creationflags,
            env={
                key: value
                for key, value in os.environ.items()
                if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MxcRuntimeUnavailable("the MXC runner identity probe failed") from exc
    if result.returncode:
        raise MxcRuntimeUnavailable("the MXC runner identity probe was refused")
    try:
        identity = json.loads(result.stdout)
    except (UnicodeError, ValueError) as exc:
        raise MxcRuntimeUnavailable("the MXC runner returned malformed identity metadata") from exc
    expected = {
        "protocolVersion": RUNNER_PROTOCOL_VERSION,
        "profileId": PROFILE_ID,
        "profileVersion": PROFILE_VERSION,
        "schemaVersion": MXC_SCHEMA_VERSION,
        "mxcRevision": MXC_REVISION,
        "mxcPatchSha256": MXC_PATCH_SHA256,
        "mxcPatchedTree": MXC_PATCHED_TREE,
        "runnerSourceIdentity": manifest["runnerSourceIdentity"],
        "admissionApi": True,
        "architecture": _expected_architecture(),
    }
    if identity != expected:
        if isinstance(identity, dict) and identity.get("admissionApi") is False:
            raise MxcRuntimeUnavailable(
                "MXC runtime is present but does not expose the required Studio admission API"
            )
        raise MxcRuntimeUnavailable("the MXC runner source or protocol identity is not approved")


def _validate_runtime_directory(
    directory: Path,
    *,
    trust: dict | None,
    development: bool,
) -> RuntimeInfo:
    _require_plain_directory(directory, "runtime generation")
    directory = directory.resolve()
    manifest_path = directory / _MANIFEST_NAME
    manifest, manifest_digest = _read_json(manifest_path, "runtime manifest")
    required = {
        "manifestVersion",
        "runtimeVersion",
        "generation",
        "architecture",
        "target",
        "protocolVersion",
        "profileId",
        "schemaVersion",
        "mxcRepository",
        "mxcRevision",
        "mxcPatchSha256",
        "mxcPatchedTree",
        "cargoLockSha256",
        "runnerSourceIdentity",
        "features",
        "rustc",
        "artifacts",
    }
    if set(manifest) != required:
        raise MxcRuntimeUnavailable(
            "the MXC runtime manifest has an unsupported shape", code="runtime_incompatible"
        )
    expected_values = {
        "manifestVersion": RUNTIME_MANIFEST_VERSION,
        "runtimeVersion": RUNTIME_VERSION,
        "architecture": _expected_architecture(),
        "target": EXPECTED_TARGET,
        "protocolVersion": RUNNER_PROTOCOL_VERSION,
        "profileId": PROFILE_ID,
        "schemaVersion": MXC_SCHEMA_VERSION,
        "mxcRevision": MXC_REVISION,
        "mxcPatchSha256": MXC_PATCH_SHA256,
        "mxcPatchedTree": MXC_PATCHED_TREE,
        "cargoLockSha256": PATCHED_CARGO_LOCK_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected_values.items()):
        raise MxcRuntimeUnavailable(
            "the MXC runtime manifest does not match Studio's pinned profile",
            code="runtime_incompatible",
        )
    if manifest.get("features") != list(EXPECTED_FEATURES):
        raise MxcRuntimeUnavailable(
            "the MXC runtime was built with unapproved features", code="runtime_incompatible"
        )
    generation = manifest.get("generation")
    if not isinstance(generation, str) or not _GENERATION_RE.fullmatch(generation):
        raise MxcRuntimeUnavailable("the MXC runtime generation identifier is invalid")
    source_identity = manifest.get("runnerSourceIdentity")
    if not isinstance(source_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", source_identity):
        raise MxcRuntimeUnavailable("the MXC runner source identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"runner"}:
        raise MxcRuntimeUnavailable("the MXC runtime artifact manifest is invalid")
    artifact = artifacts["runner"]
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "size"}:
        raise MxcRuntimeUnavailable("the MXC runner artifact entry is invalid")
    if artifact.get("path") != _RUNNER_NAME:
        raise MxcRuntimeUnavailable("the MXC runner artifact path is not approved")
    runner = directory / _RUNNER_NAME
    if not runner.is_file() or runner.is_symlink():
        raise MxcRuntimeUnavailable("the MXC runner artifact is missing or is not a regular file")
    try:
        runner_stat = runner.stat()
        size = runner_stat.st_size
    except OSError as exc:
        raise MxcRuntimeUnavailable("the MXC runner artifact metadata is unavailable") from exc
    if getattr(runner_stat, "st_nlink", 1) != 1:
        raise MxcRuntimeUnavailable(
            "the MXC runner artifact has an unapproved hard link",
            code="runtime_untrusted",
        )
    if not isinstance(artifact.get("size"), int) or artifact["size"] != size:
        raise MxcRuntimeUnavailable("the MXC runner artifact size does not match its manifest")
    digest = _sha256_file(runner)
    if artifact.get("sha256") != digest:
        raise MxcRuntimeUnavailable("the MXC runner artifact digest does not match its manifest")
    if generation != f"mxc-{MXC_REVISION[:12]}-{digest[:16]}":
        raise MxcRuntimeUnavailable("the MXC runtime generation is not bound to its runner digest")
    if not development:
        if trust is None:
            raise MxcRuntimeUnavailable(
                "the packaged MXC runtime has no trusted digest allowlist", code="runtime_untrusted"
            )
        if trust != {"manifestSha256": manifest_digest, "runnerSha256": digest}:
            raise MxcRuntimeUnavailable(
                "the MXC runtime does not match Studio's trusted allowlist",
                code="runtime_untrusted",
            )
    _validate_identity(runner, manifest)
    return RuntimeInfo(
        generation=generation,
        path=runner,
        manifest_path=manifest_path,
        runner_sha256=digest,
        manifest_sha256=manifest_digest,
        runner_source_identity=source_identity,
        development=development,
        production_ready=not development and trust is not None,
    )


def _read_packaged_selection(root: Path) -> RuntimeInfo:
    _require_plain_directory(root, "runtime root")
    if (root / _DISABLED_NAME).exists():
        raise MxcRuntimeUnavailable(
            "the Studio-owned MXC runtime is disabled",
            code="runtime_not_installed",
        )
    current, _ = _read_json(root / _CURRENT_NAME, "runtime selection")
    if (
        set(current) != {"manifestVersion", "generation"}
        or current.get("manifestVersion") != RUNTIME_MANIFEST_VERSION
    ):
        raise MxcRuntimeUnavailable("the MXC runtime selection is malformed")
    generation = current.get("generation")
    if not isinstance(generation, str) or not _GENERATION_RE.fullmatch(generation):
        raise MxcRuntimeUnavailable("the MXC runtime selection contains an invalid generation")
    trust_manifest, _ = _read_json(root / _TRUST_NAME, "runtime trust manifest")
    if (
        set(trust_manifest) != {"manifestVersion", "generations"}
        or trust_manifest.get("manifestVersion") != RUNTIME_MANIFEST_VERSION
    ):
        raise MxcRuntimeUnavailable("the MXC runtime trust manifest is malformed")
    generations = trust_manifest.get("generations")
    if not isinstance(generations, dict) or generation not in generations:
        raise MxcRuntimeUnavailable("the selected MXC runtime generation is not allowlisted")
    generation_candidate = root / "generations" / generation
    _require_plain_directory(generation_candidate, "runtime generation")
    generation_root = generation_candidate.resolve()
    expected_parent = (root / "generations").resolve()
    if generation_root.parent != expected_parent:
        raise MxcRuntimeUnavailable("the selected MXC runtime escaped the generation root")
    return _validate_runtime_directory(
        generation_root, trust=generations[generation], development=False
    )


def selected_runtime(*, root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only", code="unsupported_platform")
    packaged = _resolved_runtime_root(root)
    if packaged.exists() and (packaged / _DISABLED_NAME).exists():
        raise MxcRuntimeUnavailable(
            "the Studio-owned MXC runtime is disabled",
            code="runtime_not_installed",
        )
    if (packaged / _CURRENT_NAME).is_file():
        return _read_packaged_selection(packaged)
    repo_root = Path(__file__).resolve().parents[4]
    development = _development_runtime()
    if root is None and (repo_root / ".git").exists() and development.is_dir():
        return _validate_runtime_directory(development, trust=None, development=True)
    raise MxcRuntimeUnavailable(
        "the pinned, manifest-verified Unsloth MXC supervisor is not installed; rerun Windows Studio setup",
        code="runtime_not_installed",
    )


def runner_path() -> Path:
    return selected_runtime().path


def installation_identity() -> str:
    return selected_runtime().identity


class _WindowsHandleGuard:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = 0


def _open_runner_guard(path: Path) -> object:
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
        raise MxcRuntimeUnavailable("the MXC runner could not be locked for launch")
    return _WindowsHandleGuard(int(handle))


def acquire_runtime(*, root: Path | None = None) -> RuntimeLease:
    with _lock:
        info = selected_runtime(root=root)
        guard = _open_runner_guard(info.path)
        try:
            rechecked = selected_runtime(root=root)
            if rechecked != info:
                raise MxcRuntimeUnavailable("the selected MXC runtime changed during acquisition")
        except Exception:
            guard.close()
            raise
        _owners[info.generation] = _owners.get(info.generation, 0) + 1
        return RuntimeLease(info=info, _guard=guard)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_trust_manifest(root: Path, *, allow_missing: bool = False) -> dict:
    path = root / _TRUST_NAME
    if allow_missing and not path.is_file():
        return {"manifestVersion": RUNTIME_MANIFEST_VERSION, "generations": {}}
    trust, _ = _read_json(path, "runtime trust manifest")
    if (
        set(trust) != {"manifestVersion", "generations"}
        or trust.get("manifestVersion") != RUNTIME_MANIFEST_VERSION
        or not isinstance(trust.get("generations"), dict)
    ):
        raise MxcRuntimeUnavailable("the MXC runtime trust manifest is malformed")
    return trust


def _read_state(root: Path) -> dict:
    path = root / _STATE_NAME
    if not path.is_file():
        return {}
    state, _ = _read_json(path, "runtime state")
    if (
        set(state)
        != {
            "manifestVersion",
            "currentGeneration",
            "previousGeneration",
            "lastOperation",
        }
        or state.get("manifestVersion") != RUNTIME_MANIFEST_VERSION
    ):
        raise MxcRuntimeUnavailable("the MXC runtime state is malformed")
    for field in ("currentGeneration", "previousGeneration"):
        value = state.get(field)
        if value is not None and (
            not isinstance(value, str) or not _GENERATION_RE.fullmatch(value)
        ):
            raise MxcRuntimeUnavailable("the MXC runtime state contains an invalid generation")
    if not isinstance(state.get("lastOperation"), str):
        raise MxcRuntimeUnavailable("the MXC runtime state operation is malformed")
    return state


def _selected_generation_or_none(root: Path) -> str | None:
    path = root / _CURRENT_NAME
    if not path.is_file():
        return None
    current, _ = _read_json(path, "runtime selection")
    generation = current.get("generation")
    if (
        set(current) != {"manifestVersion", "generation"}
        or current.get("manifestVersion") != RUNTIME_MANIFEST_VERSION
        or not isinstance(generation, str)
        or not _GENERATION_RE.fullmatch(generation)
    ):
        raise MxcRuntimeUnavailable("the MXC runtime selection is malformed")
    return generation


def _publish_selection(root: Path, generation: str, *, operation: str) -> None:
    try:
        previous = _selected_generation_or_none(root)
    except MxcRuntimeUnavailable:
        if operation != "repair":
            raise
        previous = None
        (root / _CURRENT_NAME).unlink(missing_ok=True)
    _atomic_json(
        root / _CURRENT_NAME,
        {"manifestVersion": RUNTIME_MANIFEST_VERSION, "generation": generation},
    )
    try:
        _atomic_json(
            root / _STATE_NAME,
            {
                "manifestVersion": RUNTIME_MANIFEST_VERSION,
                "currentGeneration": generation,
                "previousGeneration": previous if previous != generation else None,
                "lastOperation": operation,
            },
        )
    except OSError:
        # current.json is the authority. A missing diagnostic state must not undo
        # an already atomic and fully validated selection.
        pass


def _package_approval(package_root: Path) -> tuple[Path, str, str]:
    _require_plain_directory(package_root, "runtime package")
    package, _ = _read_json(package_root / _PACKAGE_TRUST_NAME, "runtime package trust manifest")
    required = {
        "manifestVersion",
        "architecture",
        "generation",
        "manifestSha256",
        "runnerSha256",
    }
    if set(package) != required or package.get("manifestVersion") != RUNTIME_MANIFEST_VERSION:
        raise MxcRuntimeUnavailable(
            "the packaged MXC runtime trust manifest is malformed", code="runtime_untrusted"
        )
    if package.get("architecture") != _expected_architecture():
        raise MxcRuntimeUnavailable(
            "the packaged MXC runtime architecture is incompatible", code="runtime_incompatible"
        )
    generation = package.get("generation")
    manifest_digest = package.get("manifestSha256")
    runner_digest = package.get("runnerSha256")
    if (
        not isinstance(generation, str)
        or not _GENERATION_RE.fullmatch(generation)
        or not isinstance(manifest_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
        or not isinstance(runner_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", runner_digest)
    ):
        raise MxcRuntimeUnavailable(
            "the packaged MXC runtime trust identity is invalid", code="runtime_untrusted"
        )
    validated = _validate_runtime_directory(
        package_root,
        trust={"manifestSha256": manifest_digest, "runnerSha256": runner_digest},
        development=False,
    )
    if validated.generation != generation:
        raise MxcRuntimeUnavailable(
            "the packaged MXC runtime generation is not approved", code="runtime_untrusted"
        )
    return package_root, manifest_digest, runner_digest


def install_generation(
    artifact_directory: Path,
    *,
    approved_manifest_sha256: str,
    approved_runner_sha256: str,
    root: Path | None = None,
    operation: str = "install",
    replace_corrupt: bool = False,
    failure_hook=None,
) -> RuntimeInfo:
    """Publish a fully verified immutable generation, then atomically select it."""
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("MXC runtime installation is Windows-only")
    root = _resolved_runtime_root(root)
    source = _validate_runtime_directory(
        artifact_directory,
        trust={
            "manifestSha256": approved_manifest_sha256,
            "runnerSha256": approved_runner_sha256,
        },
        development=False,
    )
    with _management_guard(root):
        generations = root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        destination = generations / source.generation
        stage = generations / f".{source.generation}.{secrets.token_hex(8)}.tmp"
        displaced: Path | None = None
        created_destination = False
        selection_published = False
        trust_before: dict | None = None
        try:
            stage.mkdir()
            shutil.copy2(source.path, stage / _RUNNER_NAME)
            shutil.copy2(source.manifest_path, stage / _MANIFEST_NAME)
            staged = _validate_runtime_directory(
                stage,
                trust={
                    "manifestSha256": approved_manifest_sha256,
                    "runnerSha256": approved_runner_sha256,
                },
                development=False,
            )
            if failure_hook is not None:
                failure_hook("after_stage_validation")
            if destination.exists():
                try:
                    existing = _validate_runtime_directory(
                        destination,
                        trust={
                            "manifestSha256": approved_manifest_sha256,
                            "runnerSha256": approved_runner_sha256,
                        },
                        development=False,
                    )
                except MxcRuntimeUnavailable:
                    if not replace_corrupt:
                        raise
                    with _lock:
                        if _owners.get(source.generation, 0):
                            raise MxcRuntimeUnavailable(
                                "the corrupt MXC generation is still in use",
                                code="runtime_busy",
                            )
                    displaced = generations / f".{source.generation}.{secrets.token_hex(8)}.corrupt"
                    os.replace(destination, displaced)
                    os.replace(stage, destination)
                else:
                    if existing.runner_sha256 != staged.runner_sha256:
                        raise MxcRuntimeUnavailable("an immutable MXC generation already differs")
                    shutil.rmtree(stage)
            else:
                os.replace(stage, destination)
                created_destination = True

            try:
                trust_manifest = _read_trust_manifest(root, allow_missing=True)
            except MxcRuntimeUnavailable:
                if not replace_corrupt:
                    raise
                trust_manifest = {
                    "manifestVersion": RUNTIME_MANIFEST_VERSION,
                    "generations": {},
                }
            trust_before = json.loads(json.dumps(trust_manifest))
            generations_value = trust_manifest["generations"]
            generations_value[source.generation] = {
                "manifestSha256": approved_manifest_sha256,
                "runnerSha256": approved_runner_sha256,
            }
            _atomic_json(root / _TRUST_NAME, trust_manifest)
            if failure_hook is not None:
                failure_hook("before_pointer_publication")
            with _lock:
                _publish_selection(root, source.generation, operation=operation)
                (root / _DISABLED_NAME).unlink(missing_ok=True)
                selection_published = True
            if failure_hook is not None:
                failure_hook("after_pointer_publication")
            if displaced is not None:
                shutil.rmtree(displaced, ignore_errors=True)
            selected = _read_packaged_selection(root)
            _invalidate_probe_cache()
            return selected
        except Exception:
            if operation == "update" and created_destination and not selection_published:
                shutil.rmtree(destination, ignore_errors=True)
                if trust_before is not None:
                    _atomic_json(root / _TRUST_NAME, trust_before)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)


def install_approved_runtime(
    *,
    package_root: Path | None = None,
    root: Path | None = None,
    operation: str = "install",
    replace_corrupt: bool = False,
    failure_hook=None,
) -> RuntimeInfo:
    package, manifest_digest, runner_digest = _package_approval(
        package_root or _approved_package_root()
    )
    return install_generation(
        package,
        approved_manifest_sha256=manifest_digest,
        approved_runner_sha256=runner_digest,
        root=root,
        operation=operation,
        replace_corrupt=replace_corrupt,
        failure_hook=failure_hook,
    )


def repair_runtime(*, package_root: Path | None = None, root: Path | None = None) -> RuntimeInfo:
    return install_approved_runtime(
        package_root=package_root,
        root=root,
        operation="repair",
        replace_corrupt=True,
    )


def update_runtime(*, package_root: Path | None = None, root: Path | None = None) -> RuntimeInfo:
    return install_approved_runtime(
        package_root=package_root,
        root=root,
        operation="update",
    )


def rollback_runtime(*, root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("MXC rollback is Windows-only", code="unsupported_platform")
    root = _resolved_runtime_root(root)
    with _management_guard(root):
        with _lock:
            current = _selected_generation_or_none(root)
            state = _read_state(root)
            target = state.get("previousGeneration")
            if current is None or not isinstance(target, str) or target == current:
                raise MxcRuntimeUnavailable(
                    "there is no trusted previous MXC generation to roll back to",
                    code="runtime_incompatible",
                )
            trust = _read_trust_manifest(root)
            approval = trust["generations"].get(target)
            if not isinstance(approval, dict):
                raise MxcRuntimeUnavailable(
                    "the rollback MXC generation is not allowlisted", code="runtime_untrusted"
                )
            _validate_runtime_directory(
                root / "generations" / target,
                trust=approval,
                development=False,
            )
            _publish_selection(root, target, operation="rollback")
            selected = _read_packaged_selection(root)
        _invalidate_probe_cache()
        return selected


def runtime_status(*, package_root: Path | None = None, root: Path | None = None) -> RuntimeStatus:
    if sys.platform != "win32":
        return RuntimeStatus(
            RuntimeState.UNSUPPORTED_PLATFORM,
            "MXC runtime lifecycle is Windows-only",
        )
    try:
        packaged = _resolved_runtime_root(root)
    except MxcRuntimeUnavailable as exc:
        return RuntimeStatus(RuntimeState.UNTRUSTED, str(exc), repair_requires_user_action=True)
    repair_available = False
    try:
        _package_approval(package_root or _approved_package_root())
        repair_available = True
    except MxcRuntimeUnavailable:
        pass
    if not (packaged / _CURRENT_NAME).is_file():
        return RuntimeStatus(
            RuntimeState.NOT_INSTALLED,
            "the trusted MXC runtime is not installed",
            repair_available=repair_available,
            repair_requires_user_action=repair_available,
        )
    try:
        info = _read_packaged_selection(packaged)
        state = _read_state(packaged)
    except MxcRuntimeUnavailable as exc:
        mapped = {
            "runtime_incompatible": RuntimeState.INCOMPATIBLE,
            "runtime_untrusted": RuntimeState.UNTRUSTED,
            "unsupported_platform": RuntimeState.UNSUPPORTED_PLATFORM,
        }.get(exc.code, RuntimeState.CORRUPT)
        return RuntimeStatus(
            mapped,
            str(exc),
            repair_available=repair_available,
            repair_requires_user_action=repair_available,
        )
    return RuntimeStatus(
        RuntimeState.READY,
        "the trusted MXC runtime generation is ready",
        generation=info.generation,
        runner_sha256=info.runner_sha256,
        previous_generation=state.get("previousGeneration"),
        last_operation=state.get("lastOperation"),
        repair_available=repair_available,
    )


def _release_owner(generation: str) -> None:
    pending: Path | None = None
    with _lock:
        owners = _owners.get(generation, 0)
        if owners <= 1:
            _owners.pop(generation, None)
            pending = _retire_pending.pop(generation, None)
        else:
            _owners[generation] = owners - 1
    if pending is not None:
        with _management_guard(pending):
            with _lock:
                if not _owners.get(generation, 0):
                    _retire_now(generation, pending)


def _retire_now(generation: str, root: Path) -> None:
    generations = (root / "generations").resolve()
    target = (generations / generation).resolve()
    if target.parent != generations or not _GENERATION_RE.fullmatch(generation):
        raise MxcRuntimeUnavailable("invalid MXC runtime retirement target")
    if not target.exists():
        return
    retired = generations / f".{generation}.{secrets.token_hex(8)}.retired"
    os.replace(target, retired)
    shutil.rmtree(retired)
    trust = _read_trust_manifest(root, allow_missing=True)
    if generation in trust["generations"]:
        del trust["generations"][generation]
        _atomic_json(root / _TRUST_NAME, trust)
    try:
        state = _read_state(root)
    except MxcRuntimeUnavailable:
        state = {}
    if state.get("previousGeneration") == generation:
        state["previousGeneration"] = None
        _atomic_json(root / _STATE_NAME, state)


def retire_generation(generation: str, *, root: Path | None = None) -> bool:
    """Retire a non-current generation now, or after its final lease releases."""
    root = _resolved_runtime_root(root)
    with _management_guard(root):
        with _lock:
            current = _selected_generation_or_none(root)
            if current == generation:
                raise MxcRuntimeUnavailable("the active MXC runtime generation cannot be retired")
            if _owners.get(generation, 0):
                _retire_pending[generation] = root
                return False
            _retire_now(generation, root)
            return True


def garbage_collect(*, root: Path | None = None, keep_previous: int = 1) -> list[str]:
    """Retire trusted generations outside the rollback window and active leases."""
    if keep_previous not in {0, 1}:
        raise ValueError("keep_previous must be zero or one")
    root = _resolved_runtime_root(root)
    retired: list[str] = []
    with _management_guard(root):
        with _lock:
            current = _selected_generation_or_none(root)
            state = _read_state(root)
            keep = {current}
            previous = state.get("previousGeneration")
            if keep_previous and isinstance(previous, str):
                keep.add(previous)
            trust = _read_trust_manifest(root)
            for generation in sorted(tuple(trust["generations"])):
                if generation in keep or _owners.get(generation, 0):
                    continue
                _retire_now(generation, root)
                retired.append(generation)
    return retired


def uninstall_runtime(*, root: Path | None = None) -> bool:
    """Disable new launches, then remove only Studio-owned runtime generations."""
    root = _resolved_runtime_root(root)
    if not root.exists():
        return True
    complete = False
    with _management_guard(root):
        _atomic_json(
            root / _DISABLED_NAME,
            {"manifestVersion": RUNTIME_MANIFEST_VERSION, "disabled": True},
        )
        (root / _CURRENT_NAME).unlink(missing_ok=True)
        _invalidate_probe_cache()
        with _lock:
            if any(count > 0 for count in _owners.values()):
                return False
        generations = root / "generations"
        if generations.exists():
            shutil.rmtree(generations)
        for name in (_TRUST_NAME, _STATE_NAME, _DISABLED_NAME):
            (root / name).unlink(missing_ok=True)
        complete = True
    if complete:
        (root / _MANAGEMENT_LOCK_NAME).unlink(missing_ok=True)
        try:
            root.rmdir()
        except OSError:
            pass
    return complete
