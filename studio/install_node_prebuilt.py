#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Node.js prebuilt installer for Unsloth Studio.

Downloads an official Node.js archive from nodejs.org into an isolated
``<UNSLOTH_HOME>/node`` and never touches the system Node/npm. Pinning Node 24+
LTS clears the Unsloth frontend build floor (Vite 8: Node ^20.19 || >=22.12,
npm >= 11) with the npm it bundles.

Archives are verified against sha256 digests pinned in ``node_prebuilt_pins.json``
(committed in-tree), not a checksum re-fetched from the same origin as the archive.

Mirrors ``install_llama_prebuilt.py`` so the setup scripts drive it the same way.
Exit codes: 0 success, 1 error, 2 fallback, 3 busy. A re-run that already matches
logs "already matches" and returns 0 without downloading (the scripts grep it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_FALLBACK = 2
EXIT_BUSY = 3

# Node 24 LTS bundles npm 11, clearing Vite 8's floor (Node ^20.19 || >=22.12, npm >= 11).
NODE_MIN_LTS_MAJOR = 24
NPM_MIN_MAJOR = 11

NODE_DIST_BASE = "https://nodejs.org/dist"
NODE_DIST_INDEX = f"{NODE_DIST_BASE}/index.json"

RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HTTP_FETCH_ATTEMPTS = 4
HTTP_FETCH_BASE_DELAY_SECONDS = 0.75
INSTALL_LOCK_TIMEOUT_SECONDS = 300
INSTALL_STAGING_ROOT_NAME = ".staging"
METADATA_FILENAME = "UNSLOTH_NODE_PREBUILT_INFO.json"
METADATA_SCHEMA_VERSION = 1

# Trust anchor: verify archives against sha256 pins committed in
# node_prebuilt_pins.json (in-tree, code-reviewed), not a same-origin checksum.
PINS_FILENAME = "node_prebuilt_pins.json"
PINS_SCHEMA_VERSION = 1
DEFAULT_NODE_CHANNEL = "pinned"
# Opt-in to install an unpinned version, trusting the same-origin SHASUMS256.txt.
ALLOW_UNVERIFIED_ENV = "UNSLOTH_NODE_ALLOW_UNVERIFIED"

# PowerShell renders stderr as NativeCommandError noise; main() flips logs to stdout.
_LOG_TO_STDOUT = False


class PrebuiltFallback(RuntimeError):
    """Recoverable failure -- caller should fall back (exit code 2)."""


class UnpinnedNodeRefused(PrebuiltFallback):
    """Unpinned Node requested without opt-in. Distinct type so the orchestrator
    re-raises it instead of letting the keep-existing path swallow the refusal."""


class BusyInstallConflict(RuntimeError):
    """Another process holds the install lock (exit code 3)."""


def log(message: str) -> None:
    print(
        f"[node-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr
    )


# ── Host detection ──
@dataclass(frozen = True)
class HostInfo:
    system: str  # platform.system()
    machine: str  # lowered platform.machine()
    node_os: str  # nodejs.org token: linux | darwin | win
    node_arch: str  # nodejs.org token: x64 | arm64 | armv7l
    archive_ext: str  # .tar.gz | .zip
    is_windows: bool


def detect_host() -> HostInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"

    if system == "Linux":
        node_os = "linux"
    elif system == "Darwin":
        node_os = "darwin"
    elif is_windows:
        node_os = "win"
    else:
        raise PrebuiltFallback(
            f"unsupported operating system for Node prebuilt: {system}"
        )

    if machine in {"x86_64", "amd64", "x64"}:
        node_arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        node_arch = "arm64"
    else:
        # 32-bit ARM (armv7l) is intentionally unsupported: Node 24 LTS ships no
        # linux-armv7l build, so there is nothing at/above the floor to install.
        raise PrebuiltFallback(
            f"unsupported CPU architecture for Node prebuilt: {machine}"
        )

    # .tar.gz (not .tar.xz) on Unix so the extractor needs no xz; .zip on Windows.
    archive_ext = ".zip" if is_windows else ".tar.gz"
    return HostInfo(
        system = system,
        machine = machine,
        node_os = node_os,
        node_arch = node_arch,
        archive_ext = archive_ext,
        is_windows = is_windows,
    )


# ── URL / asset construction (pure, unit tested) ──
def node_asset_stem(version: str, host: HostInfo) -> str:
    """e.g. node-v24.4.1-linux-x64 (no extension)."""
    return f"node-v{version}-{host.node_os}-{host.node_arch}"


def node_asset_name(version: str, host: HostInfo) -> str:
    return f"{node_asset_stem(version, host)}{host.archive_ext}"


def node_download_url(version: str, asset_name: str) -> str:
    return f"{NODE_DIST_BASE}/v{version}/{asset_name}"


def node_shasums_url(version: str) -> str:
    return f"{NODE_DIST_BASE}/v{version}/SHASUMS256.txt"


def expected_sha256_for(shasums_text: str, asset_name: str) -> str | None:
    """Parse a nodejs.org SHASUMS256.txt ('<hex>  <filename>' per line)."""
    for line in shasums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == asset_name:
            digest = parts[0].lower()
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return digest
    return None


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in value.lstrip("v").split("."))
    except ValueError:
        return ()


def _meets_node_floor(version: str) -> bool:
    """True iff version clears the setup floor (^20.19 || >=22.12 || >=23)."""
    parts = _version_tuple(version)
    if not parts:
        return False
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else 0
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23


def select_node_version(index: list[dict], *, channel: str, min_major: int) -> str:
    """Pick a concrete Node version from nodejs.org index.json.

    channel='lts'    -> newest LTS release line whose major >= min_major.
    channel='latest' -> newest release overall whose major >= min_major.
    Otherwise the channel is treated as an explicit version string.
    """
    if channel not in {"lts", "latest"}:
        return channel.lstrip("v")

    best: tuple[int, ...] | None = None
    best_version: str | None = None
    for entry in index:
        version = str(entry.get("version", "")).lstrip("v")
        parsed = _version_tuple(version)
        if not parsed or parsed[0] < min_major:
            continue
        if channel == "lts" and not entry.get("lts"):
            continue
        if best is None or parsed > best:
            best = parsed
            best_version = version
    if best_version is None:
        raise PrebuiltFallback(
            f"no Node '{channel}' release found at or above major {min_major} in {NODE_DIST_INDEX}"
        )
    return best_version


# ── HTTP (retry/backoff) ──
def _auth_headers() -> dict[str, str]:
    # A User-Agent keeps some proxies/CDNs happy; nodejs.org needs no auth.
    return {"User-Agent": "unsloth-studio-node-prebuilt"}


def is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    return False


def sleep_backoff(attempt: int) -> None:
    delay = HTTP_FETCH_BASE_DELAY_SECONDS * (2 ** max(attempt - 1, 0))
    delay += random.uniform(0.0, 0.2)
    time.sleep(delay)


def download_bytes(url: str, *, timeout: int = 60) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers = _auth_headers())
            with urllib.request.urlopen(request, timeout = timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(
                f"fetch failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying"
            )
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str) -> object:
    return json.loads(download_bytes(url, timeout = 30).decode("utf-8"))


def atomic_replace_from_tempfile(tmp_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    os.replace(tmp_path, destination)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        tmp_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers = _auth_headers())
            with tempfile.NamedTemporaryFile(
                prefix = destination.name + ".tmp-",
                dir = destination.parent,
                delete = False,
            ) as handle:
                tmp_path = Path(handle.name)
                with urllib.request.urlopen(request, timeout = 120) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file from {url}")
            atomic_replace_from_tempfile(tmp_path, destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok = True)
                except Exception:  # noqa: BLE001
                    pass
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(
                f"download failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying"
            )
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file_verified(
    url: str, destination: Path, *, expected_sha256: str, label: str
) -> None:
    for attempt in range(1, 3):
        download_file(url, destination)
        actual = sha256_file(destination)
        if actual == expected_sha256:
            log(f"verified {label} sha256={actual}")
            return
        log(
            f"{label} checksum mismatch {attempt}/2: expected={expected_sha256} actual={actual}"
        )
        destination.unlink(missing_ok = True)
        if attempt == 2:
            raise PrebuiltFallback(f"{label} checksum mismatch after retry")


# ── Pinned digest manifest (trust anchor) ──
def pins_path() -> Path:
    return Path(__file__).resolve().parent / PINS_FILENAME


def load_pins() -> dict:
    path = pins_path()
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except FileNotFoundError as exc:
        raise PrebuiltFallback(f"pinned Node manifest missing: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise PrebuiltFallback(
            f"pinned Node manifest unreadable ({path}): {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("schema_version") != PINS_SCHEMA_VERSION:
        raise PrebuiltFallback(f"pinned Node manifest has an unexpected schema: {path}")
    return data


def pinned_default_version(pins: dict) -> str:
    version = str(pins.get("default_version", "")).lstrip("v")
    if not _version_tuple(version):
        raise PrebuiltFallback(
            "pinned Node manifest is missing a valid 'default_version'"
        )
    return version


def pinned_sha256(pins: dict, version: str, asset_name: str) -> str | None:
    versions = pins.get("versions")
    if not isinstance(versions, dict):
        return None
    entry = versions.get(version.lstrip("v"))
    if not isinstance(entry, dict):
        return None
    digest = entry.get(asset_name)
    if not isinstance(digest, str):
        return None
    digest = digest.strip().lower()
    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
        return digest
    return None


def allow_unverified_node() -> bool:
    return os.environ.get(ALLOW_UNVERIFIED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_expected_sha256(
    pins: dict, version: str, asset: str, *, allow_unverified: bool
) -> str:
    """Sha256 to verify the archive against: the committed pin, or (only with explicit
    opt-in) the same-origin SHASUMS256.txt. Unpinned without opt-in is refused."""
    pinned = pinned_sha256(pins, version, asset)
    if pinned is not None:
        log(f"verifying {asset} against pinned sha256 from {PINS_FILENAME}")
        return pinned

    if not allow_unverified:
        raise UnpinnedNodeRefused(
            f"refusing to install Node v{version}: {asset} is not in the pinned manifest "
            f"({PINS_FILENAME}); its only checksum would arrive over the same channel as the "
            f"archive. Use the pinned default version (unset UNSLOTH_NODE_VERSION), add a pin "
            f"for {asset}, or set {ALLOW_UNVERIFIED_ENV}=1 to trust the upstream SHASUMS256.txt "
            f"at your own risk."
        )

    log(
        f"WARNING: {asset} is not pinned; trusting upstream SHASUMS256.txt because "
        f"{ALLOW_UNVERIFIED_ENV} is set. This checksum shares the archive's origin and is "
        f"not an independent integrity guarantee."
    )
    # A non-UTF8 body just yields no hex match below -> clean PrebuiltFallback.
    shasums = download_bytes(node_shasums_url(version), timeout = 30).decode(
        "utf-8", "replace"
    )
    expected = expected_sha256_for(shasums, asset)
    if not expected:
        raise PrebuiltFallback(f"no sha256 for {asset} in SHASUMS256.txt (v{version})")
    return expected


# ── Safe archive extraction (zip + tar.gz, traversal/symlink guarded) ──
def _safe_extract_path(base: Path, member_name: str) -> Path:
    member_path = Path(member_name.replace("\\", "/"))
    if member_path.is_absolute():
        raise PrebuiltFallback(f"archive member used an absolute path: {member_name}")
    target = (base / member_path).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise PrebuiltFallback(
            f"archive member escaped destination: {member_name}"
        ) from exc
    return target


def _extract_zip_safely(source: Path, base: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = _safe_extract_path(base, member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise PrebuiltFallback(
                    f"zip archive contained a symlink entry: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _extract_tar_safely(source: Path, base: Path) -> None:
    # Node Unix tarballs ship bin/npm, bin/npx, bin/corepack as relative
    # symlinks into lib/node_modules; defer links and resolve after files.
    pending_links: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            target = _safe_extract_path(base, member.name)
            if member.isdir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            if member.islnk() or member.issym():
                pending_links.append((member, target))
                continue
            if not member.isfile():
                raise PrebuiltFallback(
                    f"tar archive contained an unsupported entry: {member.name}"
                )
            target.parent.mkdir(parents = True, exist_ok = True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise PrebuiltFallback(
                    f"tar archive entry could not be read: {member.name}"
                )
            with extracted, target.open("wb") as dst:
                shutil.copyfileobj(extracted, dst)
            if member.mode & 0o111:
                os.chmod(target, target.stat().st_mode | 0o111)

    for member, target in pending_links:
        link_name = member.linkname.replace("\\", "/")
        link_path = Path(link_name)
        if link_path.is_absolute() or not link_name:
            raise PrebuiltFallback(
                f"archive link used an unsafe target: {member.name} -> {link_name}"
            )
        # tar symlink names are link-parent relative; hard-link names are archive-root relative.
        resolved = (
            target.parent / link_path if member.issym() else base / link_path
        ).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise PrebuiltFallback(
                f"archive link escaped destination: {member.name} -> {link_name}"
            ) from exc
        target.parent.mkdir(parents = True, exist_ok = True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if member.issym():
            target.symlink_to(link_name)
        else:  # hard link
            shutil.copy2(resolved, target)


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    if archive_path.name.endswith(".zip"):
        _extract_zip_safely(archive_path, destination)
    elif archive_path.name.endswith(".tar.gz"):
        _extract_tar_safely(archive_path, destination)
    else:
        raise PrebuiltFallback(f"unsupported archive format: {archive_path.name}")


# ── Install lock (concurrent setup runs share one UNSLOTH_HOME) ──
def install_lock_path(install_dir: Path) -> Path:
    return install_dir.parent / f".{install_dir.name}.install.lock"


def _pid_is_alive(pid: int) -> bool:
    """Best-effort process liveness check that never signals the process on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output = True,
                text = True,
                timeout = 5,
                **_windows_hidden_kwargs(),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            # Be conservative if tasklist itself is unavailable.
            return True
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError:
        return False
    return True


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents = True, exist_ok = True)
    if FileLock is None:
        fd: int | None = None
        deadline = time.monotonic() + INSTALL_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.fsync(fd)
                break
            except FileExistsError:
                try:
                    raw = lock_path.read_text().strip()
                except FileNotFoundError:
                    continue
                stale = False
                if raw:
                    try:
                        stale = not _pid_is_alive(int(raw))
                    except ValueError:
                        stale = True
                if stale:
                    # Atomically rename before unlinking so only one racer removes
                    # the stale lock; a process recreating it loses the rename and waits.
                    try:
                        stale_path = lock_path.with_name(
                            f"{lock_path.name}.stale.{os.getpid()}"
                        )
                        os.replace(str(lock_path), str(stale_path))
                        stale_path.unlink(missing_ok = True)
                    except (OSError, ValueError):
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise BusyInstallConflict(
                        f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
                    )
                time.sleep(0.5)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            lock_path.unlink(missing_ok = True)
        return

    try:
        with FileLock(str(lock_path), timeout = INSTALL_LOCK_TIMEOUT_SECONDS):
            yield
    except FileLockTimeout as exc:
        raise BusyInstallConflict(
            f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
        ) from exc


# ── Install layout / metadata / health ──
def node_binary_path(install_dir: Path, host: HostInfo) -> Path:
    return install_dir / "node.exe" if host.is_windows else install_dir / "bin" / "node"


def npm_cli_path(install_dir: Path, host: HostInfo) -> Path:
    # Windows ships npm at <root>\node_modules\npm; Unix at <root>/lib/node_modules/npm.
    if host.is_windows:
        return install_dir / "node_modules" / "npm" / "bin" / "npm-cli.js"
    return install_dir / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"


def _windows_hidden_kwargs() -> dict[str, object]:
    if sys.platform != "win32":
        return {}
    kwargs: dict[str, object] = {}
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if flag:
        kwargs["creationflags"] = flag
    return kwargs


def _run_node(
    install_dir: Path,
    host: HostInfo,
    args: list[str],
    *,
    timeout: int = 120,
) -> str:
    node_bin = node_binary_path(install_dir, host)
    env = os.environ.copy()
    # Keep any `npm -g` writes inside the isolated prefix; Windows npm otherwise
    # defaults its global prefix to %APPDATA%\npm and touches the system install.
    env["NPM_CONFIG_PREFIX"] = str(install_dir)
    env["npm_config_prefix"] = str(install_dir)
    env.pop("NODE_PATH", None)
    result = subprocess.run(
        [str(node_bin), *args],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = env,
        **_windows_hidden_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"node {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def installed_node_version(install_dir: Path, host: HostInfo) -> str | None:
    node_bin = node_binary_path(install_dir, host)
    if not node_bin.exists():
        return None
    try:
        return _run_node(install_dir, host, ["-v"], timeout = 30).lstrip("v")
    except Exception:  # noqa: BLE001
        return None


def installed_npm_major(install_dir: Path, host: HostInfo) -> int | None:
    cli = npm_cli_path(install_dir, host)
    if not cli.exists():
        return None
    try:
        out = _run_node(install_dir, host, [str(cli), "--version"], timeout = 60)
        return _version_tuple(out)[0]
    except Exception:  # noqa: BLE001
        return None


def metadata_path(install_dir: Path) -> Path:
    return install_dir / METADATA_FILENAME


def write_metadata(install_dir: Path, *, version: str, asset: str, sha256: str) -> None:
    payload = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "kind": "node",
        "version": version,
        "asset": asset,
        "sha256": sha256,
    }
    metadata_path(install_dir).write_text(json.dumps(payload, indent = 2) + "\n")


def load_metadata(install_dir: Path) -> dict | None:
    path = metadata_path(install_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def existing_install_matches(
    install_dir: Path,
    host: HostInfo,
    *,
    version: str,
    expected_sha: str | None = None,
) -> bool:
    """True iff the on-disk install is exactly this version, runs, and (when
    expected_sha is given) was recorded with that digest, so a non-pinned or
    tampered artifact is not kept just because its version string matches."""
    meta = load_metadata(install_dir)
    if not meta or meta.get("version") != version:
        return False
    if expected_sha is not None and meta.get("sha256") != expected_sha:
        return False
    if installed_node_version(install_dir, host) != version:
        return False
    npm_major = installed_npm_major(install_dir, host)
    return npm_major is not None and npm_major >= NPM_MIN_MAJOR


def existing_install_usable(install_dir: Path, host: HostInfo) -> bool:
    """True iff the on-disk install runs and clears the npm floor, ignoring version."""
    if not load_metadata(install_dir):
        return False
    if installed_node_version(install_dir, host) is None:
        return False
    npm_major = installed_npm_major(install_dir, host)
    return npm_major is not None and npm_major >= NPM_MIN_MAJOR


def _swap_into_place(extracted_root: Path, install_dir: Path) -> None:
    """Atomically replace install_dir with extracted_root (same filesystem)."""
    install_dir.parent.mkdir(parents = True, exist_ok = True)
    backup: Path | None = None
    if install_dir.exists():
        backup = install_dir.parent / f".{install_dir.name}.old-{os.getpid()}"
        os.replace(install_dir, backup)
    try:
        os.replace(extracted_root, install_dir)
    except OSError:
        if backup is not None and not install_dir.exists():
            os.replace(backup, install_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors = True)


def _ensure_npm_floor(install_dir: Path, host: HostInfo) -> None:
    """Self-upgrade npm inside the isolated prefix if a pinned build ships npm < 11 (no-op on Node 24+)."""
    npm_major = installed_npm_major(install_dir, host)
    if npm_major is not None and npm_major >= NPM_MIN_MAJOR:
        return
    log(
        f"bundled npm {npm_major} below {NPM_MIN_MAJOR}; upgrading npm inside the isolated prefix"
    )
    cli = npm_cli_path(install_dir, host)
    _run_node(
        install_dir,
        host,
        [str(cli), "install", "-g", f"npm@^{NPM_MIN_MAJOR}"],
        timeout = 300,
    )


# ── Orchestration ──
def install_prebuilt(
    install_dir: Path, *, channel: str, min_major: int, force: bool
) -> int:
    host = detect_host()
    pins = load_pins()

    if channel in {"", "pinned", "default"}:
        # Default path: the committed pin, no index.json round-trip.
        version = pinned_default_version(pins)
    elif channel in {"lts", "latest"}:
        try:
            index = fetch_json(NODE_DIST_INDEX)
        except Exception as exc:  # noqa: BLE001
            # nodejs.org unreachable: keep a working isolated Node instead of aborting.
            if not force and existing_install_usable(install_dir, host):
                log(
                    f"Node dist index unreachable ({exc}); keeping existing isolated Node"
                )
                return EXIT_SUCCESS
            raise
        if not isinstance(index, list):
            raise PrebuiltFallback(
                f"unexpected index.json payload from {NODE_DIST_INDEX}"
            )
        version = select_node_version(index, channel = channel, min_major = min_major)
    else:
        version = channel.lstrip("v")
        # Explicit version bypasses min_major; reject anything Vite/OXC cannot use.
        if not _meets_node_floor(version):
            raise PrebuiltFallback(
                f"requested Node v{version} is below the floor (^20.19 || >=22.12 || >=23)"
            )

    asset = node_asset_name(version, host)
    log(f"target Node v{version} ({asset})")

    # Only keep an existing install if it matches the committed pin (or the caller
    # opted out of pinning); an unpinned target without opt-in falls through to the
    # refusal in resolve_expected_sha256 rather than short-circuiting on it.
    pin = pinned_sha256(pins, version, asset)
    allow_unverified = allow_unverified_node()
    may_keep = pin is not None or allow_unverified

    if (
        not force
        and may_keep
        and existing_install_matches(
            install_dir, host, version = version, expected_sha = pin
        )
    ):
        log(f"existing Node install already matches v{version}; nothing to do")
        return EXIT_SUCCESS

    with install_lock(install_lock_path(install_dir)):
        # Re-check under the lock: a concurrent run may have just finished.
        if (
            not force
            and may_keep
            and existing_install_matches(
                install_dir, host, version = version, expected_sha = pin
            )
        ):
            log(f"existing Node install already matches v{version}; nothing to do")
            return EXIT_SUCCESS

        try:
            expected_sha = resolve_expected_sha256(
                pins, version, asset, allow_unverified = allow_unverified
            )

            staging_root = install_dir.parent / INSTALL_STAGING_ROOT_NAME
            staging_root.mkdir(parents = True, exist_ok = True)
            staging = Path(
                tempfile.mkdtemp(
                    prefix = f"{install_dir.name}.staging-", dir = staging_root
                )
            )
            try:
                archive_path = staging / asset
                log(f"downloading {node_download_url(version, asset)}")
                download_file_verified(
                    node_download_url(version, asset),
                    archive_path,
                    expected_sha256 = expected_sha,
                    label = asset,
                )
                extract_dir = staging / "extracted"
                extract_archive(archive_path, extract_dir)

                roots = [p for p in extract_dir.iterdir() if p.is_dir()]
                if len(roots) != 1:
                    raise PrebuiltFallback(
                        f"unexpected archive layout: {[p.name for p in roots]}"
                    )
                extracted_root = roots[0]

                _ensure_npm_floor(extracted_root, host)
                write_metadata(
                    extracted_root, version = version, asset = asset, sha256 = expected_sha
                )
                _swap_into_place(extracted_root, install_dir)
            finally:
                shutil.rmtree(staging, ignore_errors = True)
                try:
                    staging_root.rmdir()
                except OSError:
                    pass
        except UnpinnedNodeRefused:
            # A policy refusal, not a transient failure: fail closed, never keep-existing.
            raise
        except Exception as exc:  # noqa: BLE001
            # Transient download/verify failure: keep an existing usable Node, but
            # never keep a same-version install whose recorded digest is not the pin
            # (the artifact the short-circuit above just rejected). A different usable
            # version is still kept for offline resilience.
            meta = load_metadata(install_dir)
            pin_mismatch = (
                pin is not None
                and bool(meta)
                and meta.get("version") == version
                and meta.get("sha256") != pin
            )
            if (
                not force
                and not pin_mismatch
                and existing_install_usable(install_dir, host)
            ):
                log(f"Node download failed ({exc}); keeping existing isolated Node")
                return EXIT_SUCCESS
            raise

    final_version = installed_node_version(install_dir, host)
    npm_major = installed_npm_major(install_dir, host)
    if final_version != version or npm_major is None or npm_major < NPM_MIN_MAJOR:
        raise PrebuiltFallback(
            f"post-install verification failed: node={final_version} npm_major={npm_major}"
        )
    log(
        f"installed isolated Node v{final_version} (npm {npm_major}.x) at {install_dir}"
    )
    return EXIT_SUCCESS


def main(argv: list[str] | None = None) -> int:
    global _LOG_TO_STDOUT
    _LOG_TO_STDOUT = True

    parser = argparse.ArgumentParser(
        description = "Install an isolated Node.js for Unsloth Studio"
    )
    parser.add_argument(
        "--install-dir",
        required = True,
        help = "isolated Node directory, e.g. <UNSLOTH_HOME>/node",
    )
    parser.add_argument(
        "--node-version",
        default = os.environ.get("UNSLOTH_NODE_VERSION", DEFAULT_NODE_CHANNEL),
        help = (
            f"'pinned' (default; installs the digest-pinned version from {PINS_FILENAME}), "
            f"'lts', 'latest', or an explicit version like 24.4.1. Non-pinned versions require "
            f"{ALLOW_UNVERIFIED_ENV}=1."
        ),
    )
    parser.add_argument("--min-major", type = int, default = NODE_MIN_LTS_MAJOR)
    parser.add_argument(
        "--force", action = "store_true", help = "reinstall even if the version matches"
    )
    args = parser.parse_args(argv)

    install_dir = Path(args.install_dir).expanduser().resolve()
    try:
        return install_prebuilt(
            install_dir,
            channel = args.node_version,
            min_major = args.min_major,
            force = args.force,
        )
    except BusyInstallConflict as exc:
        log(str(exc))
        return EXIT_BUSY
    except UnpinnedNodeRefused as exc:
        # Catch before PrebuiltFallback so the refusal logs its own message.
        log(str(exc))
        return EXIT_FALLBACK
    except PrebuiltFallback as exc:
        log(f"prebuilt unavailable: {exc}")
        return EXIT_FALLBACK
    except Exception as exc:  # noqa: BLE001
        log(f"unexpected error: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
