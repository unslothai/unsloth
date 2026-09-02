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

# The manifest is dropped before every dependency pass, so it cannot answer for a run killed mid-pass; this marker
# outlives it, or an interrupted GGUF-only install reads as a stale venv.
# Companion to the no_torch manifest key, next to setup.ps1's .unsloth-studio-owned; the next update then tries to
# delete the venv it is running out of.
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


def installed_requirements_root(root: Optional[Path] = None) -> Optional[Path]:
    """The requirements the venv's *installed* package ships, if it has them.

    The digests must describe the files `verify_install` will later read, and
    that is always the installed package's copy: at verify time this module is
    imported out of the venv, so `requirements_root()` resolves there, and
    unsloth_cli/_studio_deps.py looks in the same place for a foreign venv.

    The installer is a different tree. A desktop bundle carries its own
    `studio/install_python_stack.py`, and its requirements are whatever they were
    when that bundle was cut -- so recording the installer's digests makes every
    install stale the moment a tracked requirement file changes upstream. That is
    not hypothetical: v0.1.800-beta (2026-08-14) installed unsloth 2026.8.18,
    #9148 had pinned openai in extras.txt in between, and every fresh Linux and
    macOS desktop install came up `studio_install_requirements_changed` and paid
    an immediate repair pass before it would run.
    """
    prefix = root or venv_root()
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for site in sorted(prefix.glob(pattern)):
            reqs = site / "studio" / "backend" / "requirements"
            if reqs.is_dir():
                return reqs
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

    Deduplicated by real path, not by string: purelib hardcodes `lib` while
    platlib follows sys.platlibdir, so a lib64 build (Fedora, SuSE) names one
    directory twice through venv's lib64 -> lib symlink. Scanning both would
    report every package twice and turn a healthy venv into a conflict.
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
    equivalent records: importlib.metadata.version() returns whichever record
    the finder yields first, which can be a dist-info left by a failed uninstall.
    """
    return [version for version, _path in _installed_metadata_records(dist_name)]


def invalid_metadata_paths(dist_name: str) -> List[Path]:
    """Matching metadata directories that pip cannot safely identify."""
    return [
        path
        for version, path in _installed_metadata_records(dist_name)
        if not version and path is not None
    ]


def pip_backup_metadata_paths(dist_name: str) -> List[Path]:
    """Matching records left behind by an interrupted pip upgrade.

    pip renames the outgoing distribution to a `~` prefixed sibling while it
    installs the replacement, so a kill mid-operation keeps both. The METADATA
    still names the real project, so it counts as a duplicate here, but pip
    calls the directory invalid: `pip uninstall <name>` can never consume it.
    """
    return [
        path
        for _version, path in _installed_metadata_records(dist_name)
        if path is not None and path.name.startswith("~")
    ]


def metadata_conflict(versions: Sequence[str]) -> bool:
    """Whether matching metadata records are duplicated or unreadable."""
    return len(versions) > 1 or any(not version for version in versions)


def _metadata_is_inconsistent(dist_name: str, versions: Optional[List[str]] = None) -> bool:
    """Duplicated, unreadable, or standing on a record pip will not honour.

    A sole `~` backup is the case a version count cannot see: one readable
    version, so nothing looks wrong, while pip refuses the directory and the
    package tree is usually renamed away with it. Left unflagged, the fast path
    calls the package up to date and skips the pass that would reinstall it.
    """
    if versions is None:
        versions = installed_versions(dist_name)
    return bool(metadata_conflict(versions) or pip_backup_metadata_paths(dist_name))


def installed_version_probe(
    dist_name: str, companion_names: Sequence[str] = ()
) -> Tuple[str, bool]:
    """One unambiguous version and whether any requested metadata conflicts."""
    versions = installed_versions(dist_name)
    conflict = _metadata_is_inconsistent(dist_name, versions) or any(
        _metadata_is_inconsistent(name) for name in companion_names
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
    expected_torch_tag: Optional[str] = None,
    expected_torch_tag_pinned: Optional[bool] = None,
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
        # The venv's own copy wins over the caller's: verify_install reads the installed package's requirements, so
        # recording the installer's would compare two trees and call a finished install stale.
        "requirement_files": requirement_digests(installed_requirements_root(root) or req_root),
    }
    # Additive, so MANIFEST_SCHEMA does not move and existing manifests stay valid. Absent means
    # "unknown", NOT False: only a manifest written by a build that knew the key can answer. Recorded
    # because install.ps1 / install.sh export UNSLOTH_NO_TORCH for their own run only, so a later
    # `unsloth studio update` would otherwise reinstall torch into a GGUF-only venv.
    if no_torch is not None:
        payload["no_torch"] = bool(no_torch)
    # The FLAVOR, never the index URL it came from: a pinned index can carry a token
    if expected_torch_tag:
        payload["expected_torch_tag"] = str(expected_torch_tag).strip().lower()
    # Whether that flavor was NAMED by whoever ran the install, or merely what the selection
    # landed on: setup.ps1 picks /cpu automatically on a GPU-less host and publishes it exactly
    # as it publishes a pinned one, and reading the automatic case as deliberate leaves a later
    # eGPU with no repair offered. Absent means unknown, as with every other additive key.
    if expected_torch_tag_pinned is not None:
        payload["expected_torch_tag_pinned"] = bool(expected_torch_tag_pinned)
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
    # UnicodeDecodeError is a ValueError.
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


def recorded_torch_flavor(root: Optional[Path] = None) -> Optional[str]:
    """The torch flavor this venv was installed with, or None when unknown.

    None means nothing recorded it: no manifest, or one written before the key
    existed. Callers must treat None as "unknown" and fall back to their own
    detection, never as "cpu" -- claiming a flavor nobody selected would let a
    repair reinstall over a deliberate build.

    There is no marker companion here (unlike no_torch): the manifest is dropped
    before every dependency pass, so this answers only for the PREVIOUS install,
    which is exactly the question a repair asks. A run whose own setup script
    exported the flavor never reaches this.
    """
    manifest = read_manifest(root)
    if manifest is None:
        return None
    value = manifest.get("expected_torch_tag")
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value or None


def recorded_torch_flavor_was_pinned(root: Optional[Path] = None) -> bool:
    """Whether the recorded flavor was NAMED rather than automatically selected.

    False when nothing recorded it, including a manifest written before the key
    existed. That is the safe direction here and the opposite of the usual "unknown
    falls back to the old behaviour": treating an unproven CPU record as deliberate is
    what leaves a host that has since gained a GPU with no repair offered at all, which
    is the failure this whole field exists to distinguish. A repair is something the
    user can decline; a silently CPU-only GPU box is not.
    """
    manifest = read_manifest(root)
    if manifest is None:
        return False
    # An ACTUAL boolean. bool("false") is True, so a migrated or hand-edited manifest
    # carrying the string would read as a deliberate pin and suppress the repair on a
    # host that never chose one. Anything that is not a bool is unknown provenance, and
    # the safe answer for unknown is the same False an absent key gets.
    return manifest.get("expected_torch_tag_pinned") is True


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


# Shared between wheels, so one uninstall deletes another's recorded files.
# Mirrors _SHARED_NON_RUNTIME_ROOTS in unsloth_cli/_studio_deps.py.
_SHARED_NON_RUNTIME_ROOTS = frozenset(
    (
        "test",
        "tests",
        "doc",
        "docs",
        "example",
        "examples",
        "benchmark",
        "benchmarks",
        "sample",
        "samples",
        "scripts",
    )
)

# `_move_launcher_aside` renames this before setup, so setup.ps1's deep check
# sees it missing on every healthy Windows update. Nothing else is staged, so
# nothing else is excused, or a stray sibling could hide any quarantine.
_STAGED_LAUNCHER_NAME = "unsloth.exe"
_STAGED_LAUNCHER_SUFFIXES = (".update-stale", ".update-backup", ".deleteme")

# Rewritten in place by our own setup: the size claim is waived, absence is not.
_INSTALLER_REWRITTEN_NAMES = frozenset(("package-lock.json",))

# `npm run build` in the installed tree rehashes every asset, so RECORD names
# files our own setup deleted. Skipped whole: they are gone, not shorter.
_INSTALLER_REGENERATED_TREES = (("studio", "frontend", "dist"),)


def _staged_beside(target) -> bool:
    """Whether an absent launcher is one an update moved aside a moment ago.

    Usable, not merely present: `_recover_missing_launcher` reads these through
    `_is_valid_pe`, so a copy it would reject is no excuse. Same two-byte test.
    """
    try:
        path = Path(target)
        if path.name != _STAGED_LAUNCHER_NAME:
            return False
        for suffix in _STAGED_LAUNCHER_SUFFIXES:
            staged = path.with_name(path.name + suffix)
            try:
                if staged.stat().st_size < 2:
                    continue
                with staged.open("rb") as handle:
                    if handle.read(2) == b"MZ":
                        return True
            except OSError:
                continue
        return False
    except (OSError, ValueError):
        return False


def _within(target: Path, anchor: Path) -> bool:
    """Whether a parent-relative row lands inside the environment.

    One outside it belongs to something else, which reinstalling ours cannot fix.
    """
    try:
        resolved = target.resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(anchor)
    except ValueError:
        return False
    return True


def _venv_anchor(site_packages: Path) -> Optional[Path]:
    """The venv a site-packages belongs to, or None and the caller skips them."""
    try:
        current = site_packages.resolve()
    except OSError:
        return None
    # site-packages is 2 (Windows) or 3 (posix) below the prefix.
    for _ in range(4):
        if (current / "pyvenv.cfg").is_file():
            return current
        if current == current.parent:
            break
        current = current.parent
    return None


# Neither installer wraps the scan in a timeout, so a stalled mount would wedge
# setup. Warm cost of the largest real case is ~65ms.
PAYLOAD_SCAN_BUDGET_SECONDS = 5.0


def damaged_payload_files(
    package_name: str = "unsloth",
    limit: int = 3,
    budget_seconds: float = PAYLOAD_SCAN_BUDGET_SECONDS,
    companion_names: Sequence[str] = (),
    scan_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    """Recorded files of the managed distribution that are gone or truncated.

    Every check above reads metadata, which a quarantine of the payload leaves
    intact. Only the named package and companions, unlike `damaged_installed_files`:
    this runs on the fast path to decide whether to repair ours. `scan_paths`
    aims it at another venv. Never raises, and an environment it cannot read or
    finish reading is reported undamaged, since guessing the other way would
    repair a healthy venv on every run.

    The walk's own deadline bounds many slow stats but not one that never
    returns, which a wedged mount produces and no installer wraps in a timeout.
    So it runs on a daemon thread and is abandoned; the interpreter does not
    wait for one at exit (1.04s measured, thread parked in a syscall).
    `budget_seconds = 0` is unbounded, for the installer already committed to a
    full pass.
    """
    if budget_seconds <= 0:
        return _scan_payload_files(package_name, limit, 0.0, companion_names, scan_paths)

    import threading

    done: List[List[str]] = []

    def scan() -> None:
        done.append(
            _scan_payload_files(package_name, limit, budget_seconds, companion_names, scan_paths)
        )

    worker = threading.Thread(target = scan, daemon = True)
    worker.start()
    # The walk's deadline is the ordinary way out and reports what it found;
    # this margin only bounds the wait for a call that is not coming back.
    worker.join(budget_seconds + 1.0)
    return done[0] if done else []


def _scan_payload_files(
    package_name: str,
    limit: int,
    budget_seconds: float,
    companion_names: Sequence[str],
    scan_paths: Optional[Sequence[str]],
) -> List[str]:
    """The walk itself. Bounded between calls only; see `damaged_payload_files`."""
    import csv
    import io
    import stat
    from importlib.metadata import distributions

    found: List[str] = []
    deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None
    try:
        wanted = {_canonical(name) for name in (package_name, *companion_names) if name}
        paths = list(scan_paths) if scan_paths is not None else _metadata_scan_paths()
        if not paths:
            return found
        seen: set = set()
        for dist in distributions(path = paths):
            try:
                name = _canonical(dist.metadata["Name"] or "")
                if name not in wanted or name in seen:
                    continue
                seen.add(name)
                record = dist.read_text("RECORD")
            except Exception:
                continue
            # RECORD is optional per the spec, and unreadable says nothing.
            if not record:
                continue
            try:
                anchor = _venv_anchor(Path(dist.locate_file("")))
            except Exception:
                anchor = None
            # csv, not splitlines: a quoted field may hold a newline
            for row in csv.reader(io.StringIO(record, newline = "")):
                # Every row: batching this let one slow mount overrun 5s by a minute.
                if deadline is not None and time.monotonic() > deadline:
                    return found
                rel = row[0] if row else ""
                if not rel or rel.endswith("/"):
                    continue
                norm = rel.replace("\\", "/")
                if ".dist-info/" in norm or ".egg-info/" in norm or norm.endswith(".pyc"):
                    continue
                parts = tuple(p for p in norm.split("/") if p and p != ".")
                if not parts or norm.startswith("/") or ":" in parts[0]:
                    continue
                if len(parts) > 1 and parts[0] in _SHARED_NON_RUNTIME_ROOTS:
                    continue
                if any(parts[: len(tree)] == tree for tree in _INSTALLER_REGENERATED_TREES):
                    continue
                try:
                    target = dist.locate_file(rel)
                    # `..` is ordinary for console scripts and data files.
                    # Bounded rather than skipped: a quarantined `bin/unsloth`
                    # leaves the tree intact and the command gone.
                    if ".." in parts and (anchor is None or not _within(Path(target), anchor)):
                        continue
                    info = target.stat()
                except FileNotFoundError:
                    if not _staged_beside(target):
                        found.append(f"{rel} is missing")
                except NotADirectoryError:
                    # Not a FileNotFoundError: a parent replaced by a file.
                    found.append(f"{rel} is not reachable")
                except OSError:
                    # Unreadable is not missing, and a reinstall cannot fix it.
                    continue
                else:
                    if not stat.S_ISREG(info.st_mode):
                        found.append(f"{rel} is not a regular file")
                    elif (
                        len(row) >= 3
                        and row[2]
                        and row[2].isdigit()
                        and parts[-1] not in _INSTALLER_REWRITTEN_NAMES
                        and info.st_size < int(row[2])
                    ):
                        found.append(f"{rel} is {info.st_size} bytes, expected {row[2]}")
                if len(found) >= limit:
                    return found
    except Exception:
        return found[:limit]
    return found


def verify_install(
    root: Optional[Path] = None,
    req_root: Optional[Path] = None,
    package_name: str = "unsloth",
    installed: Optional[Dict[str, str]] = None,
    installed_conflicts: Optional[Sequence[str]] = None,
    deep: bool = False,
    scan_paths: Optional[Sequence[str]] = None,
) -> dict:
    """Report whether the managed install finished and can still boot.

    Reason strings are surfaced verbatim by the desktop preflight as its
    staleness reason, so keep them stable.

    Pass `installed`, `installed_conflicts`, and the matching `root` / `req_root`
    to describe a venv other than this interpreter's; without them the version
    and dependency checks would answer for the venv the caller happens to be
    running in.

    `deep` adds the payload scan, off by default because an external CLI loads
    this module out of the venv it drives: opt-out would spend the desktop
    preflight's 10 second budget with no way for an old caller to decline.
    `scan_paths` names that venv's site-packages, without which RECORD rows
    resolve against the wrong tree.
    """
    reqs = req_root or requirements_root()
    missing = missing_requirements(reqs / BOOT_REQUIREMENT_FILE, installed = installed)
    deps_ok = not missing

    manifest = read_manifest(root)
    manifest_ok = False
    reason: Optional[str] = None
    vanished = False

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
        # Every check below compares against `current`, which an absent
        # distribution passes -- as does a manifest written with no version at
        # all, which is what write_manifest records for one already gone.
        vanished = not current
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

    # Last: the only check that touches the filesystem.
    if manifest_ok and deps_ok and deep and (installed is None or scan_paths):
        # Reused, not re-read: a manifest rewritten mid-run would make the scan
        # disagree with the checks that already passed.
        scan_package = (manifest or {}).get("package") or package_name
        # unsloth-zoo only for the default install: `--package X` installs X
        # alone, so its neighbours are not ours to repair.
        companions = ("unsloth-zoo",) if _canonical(scan_package) == "unsloth" else ()
        # No dist-info leaves the scan nothing to walk, and no check above ever
        # looked at the companion's version.
        if not vanished:
            for companion in companions:
                present = (
                    _installed_version(companion, installed)
                    if installed is not None
                    else installed_version_probe(companion)[0]
                )
                if not present:
                    vanished = True
                    break
        if vanished or damaged_payload_files(
            scan_package, companion_names = companions, scan_paths = scan_paths
        ):
            manifest_ok = False
            reason = "studio_install_damaged"

    return {
        "ok": manifest_ok and deps_ok,
        "manifest_ok": manifest_ok,
        "deps_ok": deps_ok,
        "missing": missing,
        "reason": None if (manifest_ok and deps_ok) else (reason or "studio_deps_missing"),
    }
