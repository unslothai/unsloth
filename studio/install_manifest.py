# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install-completeness manifest for Unsloth Studio.

install_python_stack.py drops the manifest before the dependency pass and writes
it back only after the last step, so its presence means "the install finished".
Read by `unsloth studio verify-install`, `desktop-capabilities` (and through it
the Tauri preflight) and setup.sh/setup.ps1's fast path.

Without it an installer killed part-way leaves a venv with `unsloth` but not
studio.txt's dependencies, which still answers `-h` and so looked ready right up
until the backend died on `import structlog`.

Must import inside that half-installed venv: stdlib only, `packaging` optional.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MANIFEST_NAME = "unsloth_install_manifest.json"
MANIFEST_SCHEMA = 1

# Canonical truthy set for UNSLOTH_NO_TORCH, matching install.ps1 / install.sh.
NO_TORCH_TRUTHY: Tuple[str, ...] = ("1", "true", "yes", "on")

# Companion to the no_torch manifest key, next to setup.ps1's .unsloth-studio-owned.
# The manifest is deliberately dropped before every dependency pass, so it cannot
# answer for a run killed mid-pass; this marker is written before that pass and
# outlives it. Without it an interrupted GGUF-only install reads as a stale venv on
# the next update, which then tries to delete the venv it is running out of.
NO_TORCH_MARKER = ".unsloth-no-torch"

# Fingerprinted into the manifest, relative to studio/backend/requirements/.
# Editing one (a --local install) invalidates it and forces a dependency pass.
TRACKED_REQUIREMENT_FILES: Tuple[str, ...] = (
    "studio.txt",
    "base.txt",
    "extras.txt",
    "extras-no-deps.txt",
    "no-torch-runtime.txt",
    "single-env/data-designer-deps.txt",
    "single-env/data-designer.txt",
)

# The import chain studio/backend/run.py walks on startup.
BOOT_REQUIREMENT_FILE = "studio.txt"


def venv_root() -> Path:
    """Directory holding pyvenv.cfg for the interpreter running this code."""
    return Path(sys.prefix)


def manifest_path(root: Optional[Path] = None) -> Path:
    return (root or venv_root()) / MANIFEST_NAME


def requirements_root(script_dir: Optional[Path] = None) -> Path:
    """studio/backend/requirements/ next to this module (or a given studio/ dir)."""
    return (script_dir or Path(__file__).resolve().parent) / "backend" / "requirements"


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def requirement_digests(req_root: Optional[Path] = None) -> Dict[str, str]:
    """sha256 of every tracked requirement file that exists."""
    root = req_root or requirements_root()
    digests: Dict[str, str] = {}
    for name in TRACKED_REQUIREMENT_FILES:
        digest = _sha256(root / name)
        if digest is not None:
            digests[name] = digest
    return digests


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so PyJWT / pyjwt / py_jwt compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _metadata_scan_paths() -> List[str]:
    """This interpreter's site-packages roots, excluding inherited sys.path entries.

    Deduplicated by real path, not by string. purelib hardcodes `lib` while
    platlib follows sys.platlibdir, so on a lib64 build (Fedora, SuSE) the two
    are different strings for one directory -- venv creates lib64 as a symlink
    to lib. Scanning both would report every installed package twice and turn a
    healthy environment into a metadata conflict.
    """
    import sysconfig

    paths: List[str] = []
    seen: set = set()
    try:
        configured = sysconfig.get_paths()
    except Exception:
        return paths
    for key in ("purelib", "platlib"):
        path = configured.get(key)
        if not path or not os.path.isdir(path):
            continue
        try:
            key_path = os.path.realpath(path)
        except OSError:
            key_path = path
        if key_path in seen:
            continue
        seen.add(key_path)
        paths.append(path)
    return paths


def _installed_metadata_records(dist_name: str) -> List[Tuple[str, Optional[Path]]]:
    """Every matching metadata version and its directory, when available."""
    from importlib.metadata import distributions

    wanted = _canonical(dist_name)
    paths = _metadata_scan_paths()
    kwargs = {"path": paths} if paths else {}
    found: List[Tuple[str, Optional[Path]]] = []
    for dist in distributions(**kwargs):
        path = getattr(dist, "_path", None)
        try:
            record_path = Path(os.fspath(path)) if path is not None else None
        except (TypeError, ValueError):
            record_path = None
        try:
            name = dist.metadata.get("Name")
            if name:
                if _canonical(name) == wanted:
                    found.append((dist.version or "", record_path))
                continue
        except Exception:
            pass
        # A nameless or unreadable matching record is itself a conflict. Wheel
        # metadata directory names escape name separators as underscores, so
        # splitting off the final version is unambiguous.
        stem = record_path.name if record_path is not None else ""
        path_name, separator, _version = stem.removesuffix(".dist-info").rpartition("-")
        if stem.endswith(".dist-info") and separator and _canonical(path_name) == wanted:
            found.append(("", record_path))
    return sorted(found, key = lambda record: (record[0], os.fspath(record[1] or "")))


def installed_versions(dist_name: str) -> List[str]:
    """Every metadata version for one canonical distribution name.

    More than one answer is an inconsistent environment, not a choice between
    equivalent records. importlib.metadata.version() returns whichever record
    the filesystem finder happens to yield first, which can be a superseded
    dist-info left behind by an interrupted uninstall.
    """
    return [version for version, _path in _installed_metadata_records(dist_name)]


def invalid_metadata_paths(dist_name: str) -> List[Path]:
    """Matching metadata directories that pip cannot safely identify."""
    return [
        path
        for version, path in _installed_metadata_records(dist_name)
        if not version and path is not None
    ]


def metadata_conflict(versions: Sequence[str]) -> bool:
    """Whether matching metadata records are duplicated or unreadable."""
    return len(versions) > 1 or any(not version for version in versions)


def installed_version_probe(
    dist_name: str, companion_names: Sequence[str] = ()
) -> Tuple[str, bool]:
    """One unambiguous version and whether any requested metadata conflicts."""
    versions = installed_versions(dist_name)
    conflict = metadata_conflict(versions) or any(
        metadata_conflict(installed_versions(name)) for name in companion_names
    )
    version = versions[0] if len(versions) == 1 and versions[0] else ""
    return version, conflict


def _installed_version(dist_name: str, installed: Optional[Dict[str, str]] = None) -> Optional[str]:
    if installed is not None:
        return installed.get(_canonical(dist_name))
    return installed_version_probe(dist_name)[0] or None


def remove_manifest(root: Optional[Path] = None) -> bool:
    """Called before the dependency pass so an aborted run cannot leave a valid one.

    True when no manifest remains. A surviving marker (Windows raises on a
    read-only or locked file) still names this version and these digests, so a
    pass killed afterwards would verify as complete.
    """
    try:
        manifest_path(root).unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def write_manifest(
    root: Optional[Path] = None,
    req_root: Optional[Path] = None,
    steps_total: int = 0,
    package_name: str = "unsloth",
    no_torch: Optional[bool] = None,
) -> Optional[Path]:
    """Record a completed install. Never raises: no manifest reads as incomplete,
    which is the safe answer."""
    payload = {
        "schema": MANIFEST_SCHEMA,
        "completed_at_ms": int(time.time() * 1000),
        "package": package_name,
        "package_version": _installed_version(package_name),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "prefix": str(venv_root()),
        "steps_total": steps_total,
        "requirement_files": requirement_digests(req_root),
    }
    # Additive, so MANIFEST_SCHEMA does not move and every existing manifest stays
    # valid. Absent means "unknown", which is NOT False: only a manifest written by
    # a build that knew about the key can answer, and callers fall back to their own
    # detection otherwise. Recorded because install.ps1 / install.sh export
    # UNSLOTH_NO_TORCH for their own run only -- a later `unsloth studio update`
    # exports nothing and would otherwise reinstall torch into a GGUF-only venv.
    if no_torch is not None:
        payload["no_torch"] = bool(no_torch)
    path = manifest_path(root)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent = 2, sort_keys = True), encoding = "utf-8")
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def read_manifest(root: Optional[Path] = None) -> Optional[dict]:
    try:
        raw = manifest_path(root).read_text(encoding = "utf-8")
    # UnicodeDecodeError is a ValueError, not an OSError: a manifest re-saved as
    # ANSI by an editor (the payload embeds the user profile path, so non-ASCII
    # names show up there) or truncated mid-write must read as "no manifest", not
    # raise. install_python_stack.py resolves no-torch mode through here at import,
    # so anything escaping aborts the whole install.
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def no_torch_marker_path(root: Optional[Path] = None) -> Path:
    return (root or venv_root()) / NO_TORCH_MARKER


def set_no_torch_marker(no_torch: bool, root: Optional[Path] = None) -> None:
    """Record the mode outside the completion manifest. Never raises.

    Written before the dependency pass so an interrupted install still knows what
    it was building. Removed when torch is wanted, so migrating out of no-torch
    does not leave a stale marker behind.
    """
    path = no_torch_marker_path(root)
    try:
        if no_torch:
            path.write_text("", encoding = "utf-8")
        else:
            path.unlink(missing_ok = True)
    except OSError:
        pass


def recorded_no_torch(root: Optional[Path] = None) -> Optional[bool]:
    """The mode this venv was installed with, or None when unknown.

    None means nothing recorded it: no manifest key and no marker. Callers must
    fall back to their own detection on None and never to False, so an install
    made before either existed is not silently switched out of no-torch mode.
    """
    manifest = read_manifest(root)
    if manifest is not None:
        value = manifest.get("no_torch")
        if isinstance(value, bool):
            return value
        # Tolerate a hand-edited manifest that used a string.
        if isinstance(value, str):
            return value.strip().lower() in NO_TORCH_TRUTHY
    # No manifest (dropped before the dependency pass, or the install was killed
    # during it) or one predating the key: the marker is the durable answer.
    try:
        if no_torch_marker_path(root).exists():
            return True
    except OSError:
        pass
    return None


def _parse_requirement_line(line: str) -> Optional[Tuple[str, str, str]]:
    """(distribution name, marker, specifier) for a requirement, or None.

    Covers what studio.txt uses: names, specifiers, inline comments, markers.
    pip flags are skipped.
    """
    text = line.split("#", 1)[0].strip()
    if not text or text.startswith("-"):
        return None
    try:
        from packaging.requirements import Requirement
        requirement = Requirement(text)
        return (
            requirement.name,
            str(requirement.marker or ""),
            str(requirement.specifier),
        )
    except Exception:
        pass
    marker = ""
    if ";" in text:
        text, marker = text.split(";", 1)
        marker = marker.strip()
    name = text.strip()
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<", "[", " "):
        idx = name.find(sep)
        if idx > 0:
            name = name[:idx]
    name = name.strip()
    return (name, marker, "") if name else None


def _marker_applies(marker: str) -> bool:
    """True when the environment marker matches (or cannot be evaluated)."""
    if not marker:
        return True
    try:
        from packaging.markers import Marker
    except Exception:
        # No packaging: assume it applies. Over-reporting costs one extra pass.
        return True
    try:
        return bool(Marker(marker).evaluate())
    except Exception:
        return True


def _version_satisfies(version: str, specifier: str) -> bool:
    if not specifier:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        return SpecifierSet(specifier).contains(version)
    except Exception:
        return False


def missing_requirements(
    req_file: Optional[Path] = None, installed: Optional[Dict[str, str]] = None
) -> List[str]:
    """Distribution names that are missing or outside their required versions.

    Checked via importlib.metadata, not import names, because studio.txt lists
    PyJWT / python-docx / pymupdf whose import names (jwt, docx, fitz) differ.

    `installed` (canonical distribution name -> version) checks a venv other
    than the one running this code, which importlib.metadata cannot see.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    path = req_file or (requirements_root() / BOOT_REQUIREMENT_FILE)
    try:
        lines = path.read_text(encoding = "utf-8").splitlines()
    except OSError:
        return []

    missing: List[str] = []
    for line in lines:
        parsed = _parse_requirement_line(line)
        if parsed is None:
            continue
        name, marker, specifier = parsed
        if not _marker_applies(marker):
            continue
        if installed is not None:
            version = installed.get(_canonical(name))
            if version is None or not _version_satisfies(version, specifier):
                missing.append(name)
            continue
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            missing.append(name)
        except Exception:
            missing.append(name)
        else:
            if not _version_satisfies(dist.version, specifier):
                missing.append(name)
    return missing


def verify_install(
    root: Optional[Path] = None,
    req_root: Optional[Path] = None,
    package_name: str = "unsloth",
    installed: Optional[Dict[str, str]] = None,
    installed_conflicts: Optional[Sequence[str]] = None,
) -> dict:
    """Report whether the managed install finished and can still boot.

    Reason strings are surfaced verbatim by the desktop preflight as its
    staleness reason, so keep them stable.

    Pass `installed`, `installed_conflicts`, and the matching `root` / `req_root`
    to describe a venv other than this interpreter's; without them the version
    and dependency checks would answer for the venv the caller happens to be
    running in.
    """
    reqs = req_root or requirements_root()
    missing = missing_requirements(reqs / BOOT_REQUIREMENT_FILE, installed = installed)
    deps_ok = not missing

    manifest = read_manifest(root)
    manifest_ok = False
    reason: Optional[str] = None

    if manifest is None:
        reason = "studio_install_incomplete"
    elif manifest.get("schema") != MANIFEST_SCHEMA:
        reason = "studio_install_manifest_schema"
    else:
        # `update --package X` records X, so comparing against unsloth would
        # report a permanent version change.
        manifest_package = manifest.get("package") or package_name
        if installed is None:
            companions = () if _canonical(manifest_package) == "unsloth-zoo" else ("unsloth-zoo",)
            current, local_conflict = installed_version_probe(manifest_package, companions)
        else:
            current = _installed_version(manifest_package, installed)
            local_conflict = False
        foreign_conflicts = {_canonical(name) for name in (installed_conflicts or ())}
        core_conflict = _canonical(manifest_package) in foreign_conflicts or (
            _canonical(manifest_package) != "unsloth-zoo" and "unsloth-zoo" in foreign_conflicts
        )
        recorded = manifest.get("package_version")
        if core_conflict or local_conflict:
            reason = "studio_install_metadata_conflict"
        elif current and recorded and current != recorded:
            reason = "studio_install_version_changed"
        elif manifest.get("requirement_files") != requirement_digests(reqs):
            reason = "studio_install_requirements_changed"
        else:
            manifest_ok = True

    if manifest_ok and not deps_ok:
        # Install finished but the boot deps are gone: venv edited afterwards.
        reason = "studio_deps_missing"

    return {
        "ok": manifest_ok and deps_ok,
        "manifest_ok": manifest_ok,
        "deps_ok": deps_ok,
        "missing": missing,
        "reason": None if (manifest_ok and deps_ok) else (reason or "studio_deps_missing"),
    }
