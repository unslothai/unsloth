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
from typing import Dict, List, Optional, Tuple

MANIFEST_NAME = "unsloth_install_manifest.json"
MANIFEST_SCHEMA = 1

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


def _installed_version(dist_name: str, installed: Optional[Dict[str, str]] = None) -> Optional[str]:
    if installed is not None:
        return installed.get(_canonical(dist_name))
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


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
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


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
) -> dict:
    """Report whether the managed install finished and can still boot.

    Reason strings are surfaced verbatim by the desktop preflight as its
    staleness reason, so keep them stable.

    Pass `installed` (and the matching `root` / `req_root`) to describe a venv
    other than this interpreter's; without it the version and dependency checks
    would answer for the venv the caller happens to be running in.
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
        current = _installed_version(manifest.get("package") or package_name, installed)
        recorded = manifest.get("package_version")
        if current and recorded and current != recorded:
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
