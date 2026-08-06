# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import ModuleType

from filelock import FileLock, Timeout

from utils.native_path_leases import child_env_without_native_path_secret
from utils.paths.storage_roots import cache_root
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)


@dataclass(frozen = True)
class PinnedSource:
    name: str
    package: str
    repository: str
    revision: str
    required_files: tuple[str, ...]
    omitted_files: tuple[str, ...] = ()
    generated_files: tuple[tuple[str, str], ...] = ()


SPARK_TTS_SOURCE = PinnedSource(
    name = "Spark-TTS",
    package = "sparktts",
    repository = "https://github.com/SparkAudio/Spark-TTS",
    revision = "2f1ea9082400547242641f5271b6f941c9f439d1",
    required_files = (
        "sparktts/models/audio_tokenizer.py",
        "sparktts/utils/audio.py",
    ),
    generated_files = (("sparktts/__init__.py", ""),),
)

OUTETTS_SOURCE = PinnedSource(
    name = "OuteTTS",
    package = "outetts",
    repository = "https://github.com/edwko/OuteTTS",
    revision = "f5eac6e70d792844c6a6959d900a47af2c061a5b",
    required_files = (
        "outetts/models/config.py",
        "outetts/utils/preprocessing.py",
        "outetts/version/v3/audio_processor.py",
        "outetts/version/v3/prompt_processor.py",
    ),
    omitted_files = (
        "outetts/interface.py",
        "outetts/models/gguf_model.py",
    ),
    generated_files = (("outetts/__init__.py", ""),),
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_IMPORT_LOCK = threading.RLock()


def _git(arguments: list[str], *, source_name: str) -> subprocess.CompletedProcess:
    env = child_env_without_native_path_secret()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", *arguments],
            check = True,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 300,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to install the pinned {source_name} source") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Timed out while installing the pinned {source_name} source") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        message = f"Could not install the pinned {source_name} source"
        raise RuntimeError(f"{message}: {detail}" if detail else message) from error


def _git_bytes(
    arguments: list[str],
    *,
    source_name: str,
    input_data: bytes,
) -> subprocess.CompletedProcess:
    env = child_env_without_native_path_secret()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", *arguments],
            check = True,
            capture_output = True,
            input = input_data,
            timeout = 300,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to install the pinned {source_name} source") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Timed out while installing the pinned {source_name} source") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors = "replace").strip()
        message = f"Could not install the pinned {source_name} source"
        raise RuntimeError(f"{message}: {detail}" if detail else message) from error


def _generated_cache_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return "/__pycache__/" in f"/{normalized}" and normalized.endswith((".pyc", ".pyo"))


def _package_path_parts(relative: str, spec: PinnedSource, *, kind: str) -> tuple[str, ...]:
    normalized = relative.replace("\\", "/")
    parts = tuple(normalized.split("/"))
    if (
        normalized != relative
        or not normalized
        or normalized.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or any(PureWindowsPath(part).drive for part in parts)
        or parts[0] != spec.package
    ):
        raise ValueError(f"Invalid {kind} path for {spec.name}: {relative}")
    return parts


def _configured_package_paths(
    relatives: tuple[str, ...],
    spec: PinnedSource,
    *,
    kind: str,
) -> tuple[str, ...]:
    validated = []
    seen = set()
    for relative in relatives:
        _package_path_parts(relative, spec, kind = kind)
        if relative in seen:
            raise ValueError(f"Invalid {kind} path for {spec.name}: {relative}")
        seen.add(relative)
        validated.append(relative)
    return tuple(validated)


def _generated_file_contents(spec: PinnedSource) -> dict[str, bytes]:
    generated = {}
    for relative, content in spec.generated_files:
        _package_path_parts(relative, spec, kind = "generated")
        if relative in generated:
            raise ValueError(f"Invalid generated path for {spec.name}: {relative}")
        generated[relative] = content.encode("utf-8")
    return generated


def _tracked_package_blobs(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    output = _git(
        [
            "-C",
            str(checkout),
            "ls-tree",
            "-r",
            "-z",
            spec.revision,
            "--",
            spec.package,
        ],
        source_name = spec.name,
    ).stdout
    blobs = {}
    for record in (record for record in output.split("\0") if record):
        metadata, separator, relative = record.partition("\t")
        fields = metadata.split(" ")
        if separator != "\t" or len(fields) != 3:
            raise ValueError(f"Invalid tracked tree entry for {spec.name}")
        mode, object_type, object_id = fields
        _package_path_parts(relative, spec, kind = "tracked")
        if (
            mode not in ("100644", "100755")
            or object_type != "blob"
            or _REVISION_PATTERN.fullmatch(object_id) is None
            or relative in blobs
        ):
            raise ValueError(f"Invalid tracked tree entry for {spec.name}: {relative}")
        blobs[relative] = object_id
    return blobs


def _pinned_blob_digests(
    checkout: Path,
    object_ids: tuple[str, ...],
    spec: PinnedSource,
) -> dict[str, str]:
    unique_object_ids = tuple(dict.fromkeys(object_ids))
    if not unique_object_ids:
        return {}
    result = _git_bytes(
        ["-C", str(checkout), "cat-file", "--batch"],
        source_name = spec.name,
        input_data = "".join(f"{object_id}\n" for object_id in unique_object_ids).encode("ascii"),
    ).stdout
    digests = {}
    offset = 0
    for expected_object_id in unique_object_ids:
        header_end = result.find(b"\n", offset)
        if header_end < 0:
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        fields = result[offset:header_end].split(b" ")
        if len(fields) != 3:
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        object_id, object_type, size_value = fields
        try:
            size = int(size_value)
        except ValueError as error:
            raise ValueError(f"Invalid pinned blob data for {spec.name}") from error
        content_start = header_end + 1
        content_end = content_start + size
        if (
            object_id.decode("ascii", errors = "replace") != expected_object_id
            or object_type != b"blob"
            or size < 0
            or content_end >= len(result)
            or result[content_end : content_end + 1] != b"\n"
        ):
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        digests[expected_object_id] = hashlib.sha256(
            result[content_start:content_end]
        ).hexdigest()
        offset = content_end + 1
    if offset != len(result):
        raise ValueError(f"Invalid pinned blob data for {spec.name}")
    return digests


def _package_file(root: Path, relative: str, spec: PinnedSource) -> Path:
    parts = _package_path_parts(relative, spec, kind = "tracked")
    path = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed in {spec.name} source")
    if not path.is_file():
        raise ValueError(f"Missing tracked file in {spec.name} source: {relative}")
    return path


def _checkout_manifest(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    package_root = checkout / spec.package
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"Missing {spec.package} package")
    omitted = _configured_package_paths(spec.omitted_files, spec, kind = "omitted")
    excluded = set(omitted) | set(_generated_file_contents(spec))
    tracked_blobs = _tracked_package_blobs(checkout, spec)
    pinned_digests = _pinned_blob_digests(
        checkout,
        tuple(tracked_blobs.values()),
        spec,
    )
    manifest = {}
    for relative, object_id in tracked_blobs.items():
        path = _package_file(checkout, relative, spec)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pinned_digests[object_id]:
            raise ValueError(
                f"Tracked file does not match the pinned {spec.name} blob: {relative}"
            )
        if relative not in excluded:
            manifest[relative] = digest
    return manifest


def _runtime_manifest(runtime: Path, spec: PinnedSource) -> dict[str, str]:
    package_root = runtime / spec.package
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"Missing {spec.package} package")
    manifest = {}
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(runtime).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in {spec.name} runtime source")
        if path.is_dir() or _generated_cache_path(relative):
            continue
        if not path.is_file():
            raise ValueError(f"Special files are not allowed in {spec.name} runtime source")
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _expected_runtime_manifest(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    manifest = _checkout_manifest(checkout, spec)
    for relative, content in _generated_file_contents(spec).items():
        manifest[relative] = hashlib.sha256(content).hexdigest()
    return manifest


def _valid_checkout(path: Path, spec: PinnedSource) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        required_files = _configured_package_paths(
            spec.required_files,
            spec,
            kind = "required",
        )
        for relative in required_files:
            required = path.joinpath(*_package_path_parts(relative, spec, kind = "required"))
            if required.is_symlink() or not required.is_file():
                return False
        head = _git(
            ["-C", str(path), "rev-parse", "HEAD"],
            source_name = spec.name,
        ).stdout.strip().lower()
        branch = _git(
            ["-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            source_name = spec.name,
        ).stdout.strip()
        origin = _git(
            ["-C", str(path), "remote", "get-url", "origin"],
            source_name = spec.name,
        ).stdout.strip()
        status = _git(
            ["-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            source_name = spec.name,
        ).stdout
        ignored = _git(
            ["-C", str(path), "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            source_name = spec.name,
        ).stdout
        _checkout_manifest(path, spec)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        head == spec.revision
        and branch == "HEAD"
        and origin.rstrip("/").removesuffix(".git")
        == spec.repository.rstrip("/").removesuffix(".git")
        and not status
        and not ignored
    )


def _remove_owned_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok = True)
    elif path.is_dir():
        shutil.rmtree(path)


def _replace_owned_directory(staging: Path, destination: Path) -> None:
    displaced = None
    if destination.exists() or destination.is_symlink():
        displaced = destination.with_name(f".{destination.name}.invalid-{uuid.uuid4().hex}")
        os.replace(destination, displaced)
    try:
        os.replace(staging, destination)
    except Exception:
        if displaced is not None and not destination.exists():
            os.replace(displaced, destination)
            displaced = None
        raise
    finally:
        if displaced is not None:
            _remove_owned_path(displaced)


def _install_checkout(destination: Path, spec: PinnedSource) -> None:
    workspace = Path(tempfile.mkdtemp(prefix = ".source-", dir = destination.parent))
    checkout = workspace / "checkout"
    hooks = workspace / "hooks"
    hooks.mkdir()
    hook_config = f"core.hooksPath={hooks}"
    try:
        _git(["init", "--quiet", str(checkout)], source_name = spec.name)
        _git(
            ["-C", str(checkout), "config", "core.autocrlf", "false"],
            source_name = spec.name,
        )
        _git(
            ["-C", str(checkout), "remote", "add", "origin", spec.repository],
            source_name = spec.name,
        )
        _git(
            [
                "-c",
                hook_config,
                "-C",
                str(checkout),
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                "origin",
                spec.revision,
            ],
            source_name = spec.name,
        )
        fetched = _git(
            ["-C", str(checkout), "rev-parse", "FETCH_HEAD^{commit}"],
            source_name = spec.name,
        ).stdout.strip().lower()
        if fetched != spec.revision:
            raise RuntimeError(f"{spec.name} returned a different revision than the pinned source")
        _git(
            [
                "-c",
                hook_config,
                "-C",
                str(checkout),
                "checkout",
                "--quiet",
                "--detach",
                spec.revision,
            ],
            source_name = spec.name,
        )
        if not _valid_checkout(checkout, spec):
            raise RuntimeError(f"The downloaded {spec.name} source failed integrity validation")
        _replace_owned_directory(checkout, destination)
    finally:
        _remove_owned_path(workspace)


def _valid_runtime(runtime: Path, checkout: Path, spec: PinnedSource) -> bool:
    if runtime.is_symlink() or not runtime.is_dir():
        return False
    try:
        required_files = _configured_package_paths(
            spec.required_files,
            spec,
            kind = "required",
        )
        omitted_files = _configured_package_paths(
            spec.omitted_files,
            spec,
            kind = "omitted",
        )
        top_level = {
            path.name
            for path in runtime.iterdir()
            if path.name != "__pycache__"
        }
        if top_level != {spec.package}:
            return False
        for relative in required_files:
            required = runtime.joinpath(*_package_path_parts(relative, spec, kind = "required"))
            if required.is_symlink() or not required.is_file():
                return False
        for relative in omitted_files:
            omitted = runtime.joinpath(*_package_path_parts(relative, spec, kind = "omitted"))
            if omitted.exists() or omitted.is_symlink():
                return False
        for relative, content in _generated_file_contents(spec).items():
            generated = runtime / relative
            if generated.is_symlink() or not generated.is_file():
                return False
            if generated.read_bytes() != content:
                return False
        return _runtime_manifest(runtime, spec) == _expected_runtime_manifest(checkout, spec)
    except (OSError, RuntimeError, ValueError):
        return False


def _install_runtime(runtime: Path, checkout: Path, spec: PinnedSource) -> None:
    workspace = Path(tempfile.mkdtemp(prefix = ".runtime-", dir = runtime.parent))
    staging = workspace / "runtime"
    staging.mkdir()
    try:
        for relative, expected_digest in _checkout_manifest(checkout, spec).items():
            source_file = checkout / relative
            destination_file = staging / relative
            destination_file.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(source_file, destination_file)
            if hashlib.sha256(destination_file.read_bytes()).hexdigest() != expected_digest:
                raise RuntimeError(f"{spec.name} source changed while preparing its runtime")
        for relative, content in _generated_file_contents(spec).items():
            destination_file = staging / relative
            destination_file.parent.mkdir(parents = True, exist_ok = True)
            destination_file.write_bytes(content)
        if not _valid_runtime(staging, checkout, spec):
            raise RuntimeError(f"The prepared {spec.name} runtime failed integrity validation")
        _replace_owned_directory(staging, runtime)
    finally:
        _remove_owned_path(workspace)


def ensure_pinned_source(spec: PinnedSource) -> Path:
    revision = spec.revision.lower()
    if _REVISION_PATTERN.fullmatch(revision) is None or revision != spec.revision:
        raise RuntimeError(f"{spec.name} source revision must be a lowercase full Git commit")

    parent = cache_root() / "third-party-sources" / spec.name
    version_root = parent / revision
    checkout = version_root / "source"
    runtime = version_root / "runtime-v1"
    if _valid_checkout(checkout, spec) and _valid_runtime(runtime, checkout, spec):
        return runtime.resolve()

    version_root.mkdir(parents = True, exist_ok = True)
    try:
        with FileLock(str(parent / ".install.lock"), timeout = 300):
            checkout_valid = _valid_checkout(checkout, spec)
            if not checkout_valid:
                from utils.utils import hf_env_offline

                if hf_env_offline():
                    raise RuntimeError(
                        f"The pinned {spec.name} source is not cached and Studio is offline"
                    )
                _install_checkout(checkout, spec)
            if not _valid_runtime(runtime, checkout, spec):
                _install_runtime(runtime, checkout, spec)
    except Timeout as error:
        raise RuntimeError(f"Timed out waiting for another {spec.name} installation") from error

    if not _valid_checkout(checkout, spec) or not _valid_runtime(runtime, checkout, spec):
        raise RuntimeError(f"The installed {spec.name} source failed integrity validation")
    return runtime.resolve()


def ensure_spark_tts_source() -> Path:
    return ensure_pinned_source(SPARK_TTS_SOURCE)


def ensure_outetts_source() -> Path:
    return ensure_pinned_source(OUTETTS_SOURCE)


def _module_is_inside(module: ModuleType, package_root: Path) -> bool:
    origins = []
    origin = getattr(module, "__file__", None)
    if origin:
        origins.append(origin)
    origins.extend(getattr(module, "__path__", ()) or ())
    if not origins:
        return False
    for value in origins:
        try:
            if not Path(value).resolve().is_relative_to(package_root):
                return False
        except (OSError, ValueError):
            return False
    return True


def _purge_package_bytecode(package_root: Path) -> None:
    for directory, child_directories, files in os.walk(package_root, topdown = True):
        directory_path = Path(directory)
        for name in tuple(child_directories):
            path = directory_path / name
            if path.is_symlink():
                child_directories.remove(name)
                if name == "__pycache__":
                    path.unlink()
            elif name == "__pycache__":
                child_directories.remove(name)
                shutil.rmtree(path)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                (directory_path / name).unlink(missing_ok = True)


def _remove_package_modules(package: str) -> None:
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)


def import_pinned_module(
    module_name: str,
    *,
    package: str,
    source: Path | str,
) -> ModuleType:
    if module_name != package and not module_name.startswith(f"{package}."):
        raise ValueError(f"Only {package} modules can be imported from this pinned source")
    source_root = Path(source).resolve()
    unresolved_package_root = source_root / package
    if unresolved_package_root.is_symlink() or not unresolved_package_root.is_dir():
        raise RuntimeError(f"The pinned {package} package is missing")
    package_root = unresolved_package_root.resolve()
    package_init = package_root / "__init__.py"
    if package_init.is_symlink() or not package_init.is_file():
        raise RuntimeError(f"The pinned {package} package is not sealed")

    with _IMPORT_LOCK:
        for name, loaded_module in list(sys.modules.items()):
            if name != package and not name.startswith(f"{package}."):
                continue
            if not _module_is_inside(loaded_module, package_root):
                sys.modules.pop(name, None)

        source_value = str(source_root)
        while source_value in sys.path:
            sys.path.remove(source_value)
        sys.path.insert(0, source_value)
        _purge_package_bytecode(package_root)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
            invalid_modules = sorted(
                name
                for name, loaded_module in sys.modules.items()
                if (name == package or name.startswith(f"{package}."))
                and not _module_is_inside(loaded_module, package_root)
            )
            if invalid_modules:
                names = ", ".join(invalid_modules)
                raise RuntimeError(
                    f"{package} loaded package modules from outside the pinned source: {names}"
                )
            return module
        except BaseException:
            while source_value in sys.path:
                sys.path.remove(source_value)
            _remove_package_modules(package)
            raise


def deactivate_pinned_package(package: str, source: Path | str | None) -> None:
    with _IMPORT_LOCK:
        if source is not None:
            source_value = str(Path(source).resolve())
            while source_value in sys.path:
                sys.path.remove(source_value)
        _remove_package_modules(package)


def import_sparktts_module(module_name: str, source: Path | str) -> ModuleType:
    return import_pinned_module(module_name, package = "sparktts", source = source)


def import_outetts_module(module_name: str, source: Path | str) -> ModuleType:
    return import_pinned_module(module_name, package = "outetts", source = source)
