# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Manifest-owned immutable runtime generations for the Windows MXC supervisor."""

from __future__ import annotations

from dataclasses import dataclass
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
_GENERATION_RE = re.compile(r"^mxc-[0-9a-f]{12}-[0-9a-f]{16}$")
_MAX_MANIFEST = 256 * 1024


class MxcRuntimeUnavailable(RuntimeError):
    pass


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
_owners: dict[str, int] = {}
_retire_pending: dict[str, Path] = {}


def _native_root() -> Path:
    return Path(__file__).resolve().parents[3] / "native" / "mxc-runner"


def _packaged_root() -> Path:
    return _native_root() / "bin"


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


def _expected_architecture() -> str:
    machine = platform.machine().casefold()
    if machine not in {"amd64", "x86_64"}:
        raise MxcRuntimeUnavailable(
            f"the packaged MXC runtime does not support this architecture: {machine or 'unknown'}"
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
        raise MxcRuntimeUnavailable("the MXC runtime manifest has an unsupported shape")
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
            "the MXC runtime manifest does not match Studio's pinned profile"
        )
    if manifest.get("features") != list(EXPECTED_FEATURES):
        raise MxcRuntimeUnavailable("the MXC runtime was built with unapproved features")
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
        size = runner.stat().st_size
    except OSError as exc:
        raise MxcRuntimeUnavailable("the MXC runner artifact metadata is unavailable") from exc
    if not isinstance(artifact.get("size"), int) or artifact["size"] != size:
        raise MxcRuntimeUnavailable("the MXC runner artifact size does not match its manifest")
    digest = _sha256_file(runner)
    if artifact.get("sha256") != digest:
        raise MxcRuntimeUnavailable("the MXC runner artifact digest does not match its manifest")
    if generation != f"mxc-{MXC_REVISION[:12]}-{digest[:16]}":
        raise MxcRuntimeUnavailable("the MXC runtime generation is not bound to its runner digest")
    if not development:
        if trust is None:
            raise MxcRuntimeUnavailable("the packaged MXC runtime has no trusted digest allowlist")
        if trust != {"manifestSha256": manifest_digest, "runnerSha256": digest}:
            raise MxcRuntimeUnavailable("the MXC runtime does not match Studio's trusted allowlist")
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
    generation_root = (root / "generations" / generation).resolve()
    expected_parent = (root / "generations").resolve()
    if generation_root.parent != expected_parent:
        raise MxcRuntimeUnavailable("the selected MXC runtime escaped the generation root")
    return _validate_runtime_directory(
        generation_root, trust=generations[generation], development=False
    )


def selected_runtime(*, root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    packaged = (root or _packaged_root()).resolve()
    if (packaged / _CURRENT_NAME).is_file():
        return _read_packaged_selection(packaged)
    repo_root = Path(__file__).resolve().parents[4]
    development = _development_runtime()
    if root is None and (repo_root / ".git").exists() and development.is_dir():
        return _validate_runtime_directory(development, trust=None, development=True)
    raise MxcRuntimeUnavailable(
        "the pinned, manifest-verified Unsloth MXC supervisor is not installed; rerun Windows Studio setup"
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


def install_generation(
    artifact_directory: Path,
    *,
    approved_manifest_sha256: str,
    approved_runner_sha256: str,
    root: Path | None = None,
) -> RuntimeInfo:
    """Publish a fully verified immutable generation, then atomically select it."""
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("MXC runtime installation is Windows-only")
    root = (root or _packaged_root()).resolve()
    source = _validate_runtime_directory(
        artifact_directory,
        trust={
            "manifestSha256": approved_manifest_sha256,
            "runnerSha256": approved_runner_sha256,
        },
        development=False,
    )
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / source.generation
    stage = generations / f".{source.generation}.{secrets.token_hex(8)}.tmp"
    with _lock:
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
            if destination.exists():
                existing = _validate_runtime_directory(
                    destination,
                    trust={
                        "manifestSha256": approved_manifest_sha256,
                        "runnerSha256": approved_runner_sha256,
                    },
                    development=False,
                )
                if existing.runner_sha256 != staged.runner_sha256:
                    raise MxcRuntimeUnavailable("an immutable MXC generation already differs")
                shutil.rmtree(stage)
            else:
                os.replace(stage, destination)

            trust_path = root / _TRUST_NAME
            if trust_path.is_file():
                trust_manifest, _ = _read_json(trust_path, "runtime trust manifest")
                generations_value = trust_manifest.get("generations")
                if not isinstance(generations_value, dict):
                    raise MxcRuntimeUnavailable("the MXC runtime trust manifest is malformed")
            else:
                trust_manifest = {
                    "manifestVersion": RUNTIME_MANIFEST_VERSION,
                    "generations": {},
                }
                generations_value = trust_manifest["generations"]
            generations_value[source.generation] = {
                "manifestSha256": approved_manifest_sha256,
                "runnerSha256": approved_runner_sha256,
            }
            _atomic_json(trust_path, trust_manifest)
            _atomic_json(
                root / _CURRENT_NAME,
                {
                    "manifestVersion": RUNTIME_MANIFEST_VERSION,
                    "generation": source.generation,
                },
            )
            return _read_packaged_selection(root)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)


def _release_owner(generation: str) -> None:
    with _lock:
        owners = _owners.get(generation, 0)
        if owners <= 1:
            _owners.pop(generation, None)
            pending = _retire_pending.pop(generation, None)
            if pending is not None:
                _retire_now(generation, pending)
        else:
            _owners[generation] = owners - 1


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


def retire_generation(generation: str, *, root: Path | None = None) -> bool:
    """Retire a non-current generation now, or after its final lease releases."""
    root = (root or _packaged_root()).resolve()
    with _lock:
        current, _ = _read_json(root / _CURRENT_NAME, "runtime selection")
        if current.get("generation") == generation:
            raise MxcRuntimeUnavailable("the active MXC runtime generation cannot be retired")
        if _owners.get(generation, 0):
            _retire_pending[generation] = root
            return False
        _retire_now(generation, root)
        return True
