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
# Where remove_manifest parks the completion manifest. Only the dependency pass reads
# it, and only as evidence of what the LAST completed pass did; verify_install, the
# setup fast path and the desktop preflight never look at it, so a venv whose live
# manifest is gone still reads as half-built everywhere it matters.
PREVIOUS_MANIFEST_NAME = "unsloth_install_manifest.previous.json"
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

# Every file the dependency pass reads to decide what to install, recorded under the
# additive `pass_inputs` key so a step can prove its inputs have not moved.
#
# NOT folded into TRACKED_REQUIREMENT_FILES, even though it is a superset of it:
# verify_install compares the whole `requirement_files` dict, so one new name there
# would report every install in the field as `studio_install_requirements_changed`
# and buy each of them an immediate repair pass. New evidence goes under new keys.
PASS_INPUT_FILES: Tuple[str, ...] = TRACKED_REQUIREMENT_FILES + (
    "diffusers-pin.txt",
    "triton-kernels.txt",
    "overrides.txt",
    "single-env/constraints.txt",
    "single-env/overrides-darwin-arm64.txt",
)

# setup.sh / setup.ps1 write this beside a sidecar directory they created; the runtime
# self-heal in studio/backend/utils/transformers_version.py puts one back after its own
# rebuild. Absent means the directory is not ours to delete, which is also the answer
# `sidecar_is_current` has to give: a rebuild is the only repair, and it starts with rm.
SIDECAR_OWNED_MARKER = ".unsloth-studio-owned"


def venv_root() -> Path:
    """Directory holding pyvenv.cfg for the interpreter running this code."""
    return Path(sys.prefix)


def manifest_path(root: Optional[Path] = None) -> Path:
    return (root or venv_root()) / MANIFEST_NAME


def previous_manifest_path(root: Optional[Path] = None) -> Path:
    return (root or venv_root()) / PREVIOUS_MANIFEST_NAME


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


def digest_file(path) -> Optional[str]:
    """sha256 of one file, or None when it is absent or unreadable.

    None is never equal to a recorded digest, so an input that cannot be read
    forces the step that consumes it to run. That is the safe direction.
    """
    try:
        return _sha256(Path(path))
    except (TypeError, ValueError):
        return None


def pass_input_digests(req_root: Optional[Path] = None) -> Dict[str, str]:
    """sha256 of every PASS_INPUT_FILES entry that exists, relpath -> digest."""
    root = Path(req_root) if req_root is not None else requirements_root()
    digests: Dict[str, str] = {}
    for name in PASS_INPUT_FILES:
        digest = digest_file(root / name)
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

    The file is parked under PREVIOUS_MANIFEST_NAME rather than deleted: setup.ps1
    calls this before pip, torch and triton are replaced, which is before
    install_python_stack.py gets to read what the last pass recorded, and without the
    parked copy every Windows update ran the whole dependency pass again. The parked
    copy is evidence only (see read_previous_manifest) and is dropped by the next
    write_manifest. Deleting it is still the fallback when the rename is refused.
    """
    path = manifest_path(root)
    try:
        os.replace(path, previous_manifest_path(root))
    except FileNotFoundError:
        return True
    except OSError:
        try:
            path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
    return True


def read_previous_manifest(root: Optional[Path] = None) -> Optional[dict]:
    """The manifest remove_manifest parked, or None.

    Evidence of the last COMPLETED pass, for the dependency pass alone: every skip it
    permits is still re-verified on disk, and verify_install never reads it, so this
    can make an update faster but never make a half-built venv look finished.
    """
    try:
        raw = previous_manifest_path(root).read_text(encoding = "utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# The keys write_manifest owns, and the only ones verify_install, the setup fast path
# and desktop-capabilities ever decide on. Additive evidence -- `extra` here, keyword
# arguments to update_manifest -- may never shadow one: a caller that could rewrite
# `package_version` or `requirement_files` would falsely validate or silently invalidate
# an install, and nothing downstream re-derives them. The optional three are in for the
# same reason as the rest: absent means "unknown", so evidence must not be able to
# invent an answer the installing build never gave. ONE constant, because two copies of
# this list is exactly how the two writers would come to disagree.
PROTECTED_MANIFEST_KEYS: Tuple[str, ...] = (
    "schema",
    "completed_at_ms",
    "package",
    "package_version",
    "python",
    "platform",
    "prefix",
    "steps_total",
    "requirement_files",
    "no_torch",
    "expected_torch_tag",
    "expected_torch_tag_pinned",
)


def write_manifest(
    root: Optional[Path] = None,
    req_root: Optional[Path] = None,
    steps_total: int = 0,
    package_name: str = "unsloth",
    no_torch: Optional[bool] = None,
    expected_torch_tag: Optional[str] = None,
    expected_torch_tag_pinned: Optional[bool] = None,
    extra: Optional[Dict[str, object]] = None,
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
    # The FLAVOR, never the index URL it came from: a pinned index can carry a token in its userinfo, query or fragment,
    # and this file lives in the venv and is read back by verify-install, desktop-capabilities and the setup fast path.
    if expected_torch_tag:
        payload["expected_torch_tag"] = str(expected_torch_tag).strip().lower()
    # Whether that flavor was NAMED by whoever ran the install, or merely what the selection
    # landed on: setup.ps1 picks /cpu automatically on a GPU-less host and publishes it exactly
    # as it publishes a pinned one, and reading the automatic case as deliberate leaves a later
    # eGPU with no repair offered. Absent means unknown, as with every other additive key.
    if expected_torch_tag_pinned is not None:
        payload["expected_torch_tag_pinned"] = bool(expected_torch_tag_pinned)
    # The dependency pass's own evidence: `pass_inputs`, `step_results`, `pip_check_ok`,
    # `mlx_health`, `uv_version`, `installer_python_tag`. Additive and never authoritative
    # on its own -- every consumer re-verifies on disk before skipping anything. None is
    # dropped rather than written, so "absent means unknown" holds for these keys too, and
    # nothing here may shadow a field above: those are what verify_install reads.
    for key, value in (extra or {}).items():
        if value is None or key in payload or key in PROTECTED_MANIFEST_KEYS:
            continue
        payload[key] = value
    path = manifest_path(root)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent = 2, sort_keys = True), encoding = "utf-8")
        os.replace(tmp, path)
    except OSError:
        return None
    # The parked copy described the pass before this one; the live file now does.
    try:
        previous_manifest_path(root).unlink()
    except OSError:
        pass
    return path


def update_manifest(root: Optional[Path] = None, **extra: object) -> bool:
    """Merge additive keys into an existing manifest. Never raises.

    For evidence that is only available AFTER the manifest is written -- the MLX
    import probe runs there so a kill during its 180 s timeout cannot lose a
    finished install. False when there is nothing to update, which the callers
    treat as "record nothing", never as a failed install.

    PROTECTED_MANIFEST_KEYS are dropped, exactly as write_manifest drops them from
    `extra`: this merges into a manifest that already means "the install finished",
    so a caller able to rewrite the fields that claim describes could leave the file
    valid-looking and wrong with nothing on disk contradicting it.
    """
    values = {
        key: value
        for key, value in extra.items()
        if value is not None and key not in PROTECTED_MANIFEST_KEYS
    }
    if not values:
        return False
    data = read_manifest(root)
    if data is None:
        return False
    data.update(values)
    path = manifest_path(root)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent = 2, sort_keys = True), encoding = "utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        return False
    return True


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


# A closure walk that never finishes is a full dependency pass on every update, and a
# site-packages on a wedged network mount can produce one. Generous, because the walk is
# in-memory after the index is built and 400 distributions cost well under a second.
CLOSURE_SCAN_BUDGET_SECONDS = 10.0
# Belt to the deadline's braces: a metadata set that somehow cycles without repeating a
# (name, extras) key still stops.
_CLOSURE_MAX_VISITS = 20000


def installed_dependency_index() -> Optional[Dict[str, Tuple[str, List[str]]]]:
    """canonical name -> (version, raw Requires-Dist lines) for this interpreter.

    One pass over the metadata, because every gated step asks the same question about
    the same site-packages and `distribution(name)` re-reads a dist-info per lookup.
    None when the metadata cannot be enumerated at all, which the caller reads as
    "cannot audit" rather than "satisfied".
    """
    from importlib.metadata import distributions
    try:
        index: Dict[str, Tuple[str, List[str]]] = {}
        for dist in distributions(path = _metadata_scan_paths()):
            try:
                name = dist.metadata["Name"]
                version = dist.version
            except Exception:
                # A dist-info with unreadable metadata is damage the caller's other
                # checks report; it must not take the whole index with it.
                continue
            if not name or not version:
                continue
            key = _canonical(str(name))
            # First wins, matching sys.path precedence. A duplicate is a conflict
            # metadata_conflict() already reports, and picking the other copy here
            # would make two checks disagree about the same venv.
            if key in index:
                continue
            try:
                requires = list(dist.requires or [])
            except Exception:
                requires = []
            index[key] = (str(version), requires)
        return index
    except Exception:
        return None


def unsatisfied_closure_requirement(
    req_file: Path,
    index: Optional[Dict[str, Tuple[str, List[str]]]] = None,
    budget_seconds: float = CLOSURE_SCAN_BUDGET_SECONDS,
) -> Optional[str]:
    """The first requirement in *req_file*'s INSTALLED closure that is not met, or None.

    missing_requirements() reads the file's own lines, which stay true after a
    transitive dependency is uninstalled: `mammoth>=1.8.0` is satisfied by a mammoth
    whose `cobble` is gone, and importing it raises. The step that would have repaired
    that is exactly the one being considered for a skip, so the audit has to follow
    Requires-Dist down from each line.

    Only for steps installed WITH dependencies. A `--no-deps` step deliberately leaves
    its requirements' own dependencies unresolved, so auditing one would report a
    conflict the installer created on purpose and force that step to run forever.

    Fails CLOSED, unlike missing_requirements() and violated_constraints(): every
    return path that is not a proven-complete closure names a reason, and the caller
    reads any string as "install it". `packaging` absent, a direct URL requirement whose
    provenance a version cannot answer, an unreadable file, a budget overrun -- none of
    those are evidence that the closure holds, and the cost of being wrong is one
    dependency pass rather than a broken import nobody repairs.
    """
    try:
        from packaging.requirements import Requirement
    except Exception:
        return "<packaging unavailable>"
    if index is None:
        index = installed_dependency_index()
    if index is None:
        return "<metadata unreadable>"
    try:
        lines = Path(req_file).read_text(encoding = "utf-8-sig").splitlines()
    except (OSError, ValueError):
        return "<requirements unreadable>"

    deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None
    # (raw requirement, the extras whose markers are in scope for it). The top level has
    # no extra, so only markers that do not mention one apply.
    pending: List[Tuple[str, Tuple[str, ...]]] = []
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if text.startswith("-"):
            # A pip flag. None of the audited files carry one today, and an `-r` include
            # would hide requirements from this walk entirely.
            return f"<flag line: {text}>"
        pending.append((text, ("",)))

    seen: set = set()
    visits = 0
    while pending:
        visits += 1
        if visits > _CLOSURE_MAX_VISITS:
            return "<closure too large>"
        if deadline is not None and time.monotonic() > deadline:
            return "<closure audit timed out>"
        raw, contexts = pending.pop()
        try:
            requirement = Requirement(raw)
        except Exception:
            return f"<unparseable: {raw}>"
        marker = requirement.marker
        if marker is not None:
            try:
                applies = any(marker.evaluate({"extra": extra}) for extra in contexts)
            except Exception:
                return f"<unevaluable marker: {raw}>"
            if not applies:
                continue
        if requirement.url:
            # A direct reference is satisfied by whatever landed, and the version says
            # nothing about which. _direct_reference_is_installed answers that for the
            # one step that has one, and that step is --no-deps and never gets here.
            return f"{requirement.name} (direct reference)"
        key = _canonical(requirement.name)
        record = index.get(key)
        if record is None:
            return requirement.name
        version, requires = record
        if requirement.specifier and not requirement.specifier.contains(version, prereleases = True):
            return f"{requirement.name} {version}"
        extras = tuple(sorted(_canonical(extra) for extra in requirement.extras))
        visit_key = (key, extras)
        if visit_key in seen:
            continue
        seen.add(visit_key)
        # An extra's own dependencies are declared with `extra == "<name>"` markers, so
        # a requirement asking for foo[bar] puts "bar" in scope for foo's Requires-Dist
        # alongside the unconditional ones. Both spellings, because PEP 685 normalisation
        # of the name inside the marker only arrived with newer build backends: a wheel
        # built before it still says extra == "all_files", and matching only the
        # canonical "all-files" would silently drop that extra's whole subtree.
        child_contexts = ("", *extras, *requirement.extras)
        pending.extend((child, child_contexts) for child in requires)
    return None


def violated_constraints(
    req_file: Optional[Path] = None, installed: Optional[Dict[str, str]] = None
) -> List[str]:
    """Constrained distributions whose INSTALLED version sits outside the pin.

    A constraints file never asks for an install, so an absent distribution is
    not a violation -- only a resident one outside its window is. That is exactly
    what a skipped step has to rule out: `-c constraints.txt` is passed to every
    constrained step, so a constraint that moved under an unchanged requirements
    file is the one input digest equality cannot see.

    Empty on an unreadable file, matching missing_requirements: the caller's other
    evidence still has to pass, and reporting a violation nobody can name would
    force a full pass on every run.
    """
    path = req_file or (requirements_root() / "single-env" / "constraints.txt")
    try:
        lines = path.read_text(encoding = "utf-8-sig").splitlines()
    except (OSError, ValueError):
        return []

    violated: List[str] = []
    for line in lines:
        parsed = _parse_requirement_line(line)
        if parsed is None:
            continue
        name, marker, specifier = parsed
        if not specifier or not _marker_applies(marker):
            continue
        version = _installed_version(name, installed)
        # Absent is not a violation; unparseable metadata is, since the step that
        # would fix it is the one being considered for a skip.
        if version and not _version_satisfies(version, specifier):
            violated.append(name)
    return violated


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
    manifest: Optional[dict] = None,
) -> dict:
    """Report whether the managed install finished and can still boot.

    `manifest` verifies the tree against a manifest the caller already holds -- the
    dependency pass checking a parked one -- instead of the live file; everything
    else about the verdict is unchanged.

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

    if manifest is None:
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


# -- Sidecar directories ------------------------------------------------------
#
# The transformers 5.x sidecars are flat `pip --target` trees, prepended to sys.path by
# the training worker; no interpreter owns them, so nothing here may import from one.
#
# Deliberately MIRRORED from `_venv_dir_is_valid` + `_sidecar_scan_impl` in
# studio/backend/utils/transformers_version.py rather than imported: that module lives
# under studio/backend, imports the backend's logger and settings, and is not reachable
# from the installer's stdlib-only world. Keep the two predicates in sync -- the runtime
# self-heal there is what pays for a disagreement, by rebuilding on every request a
# setup run just declared current.

# Version-tagged extension suffixes (.cpython-313-darwin.so, .cp313-win_amd64.pyd,
# free-threaded .cpython-314t-*). Untagged binaries carry no version and are skipped, as
# are pypy/graalpy/debug spellings: an unrecognised name reports nothing rather than guessing.
_EXT_VERSION_TAG_RE = re.compile(r"\.(?:cpython-|cp)(\d{2,}t?)\b")
# Stable-ABI binaries. A GIL build imports one produced by any older CPython, so they are
# skipped there; a free-threaded build takes a SIGSEGV instead of an ImportError.
_ABI3_EXT_RE = re.compile(r"\.abi3\.(?:so|pyd)$")

SIDECAR_SCAN_BUDGET_SECONDS = 5.0


def _current_ext_tag() -> str:
    import sysconfig
    return "{}{}{}".format(
        sys.version_info.major,
        sys.version_info.minor,
        "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "",
    )


def _sidecar_payload_present(root: Path, dist) -> bool:
    """Whether anything *dist* records as installed is on disk under *root*.

    The fallback for the distributions a directory name cannot reach: one that ships
    top-level MODULES (`six.py`, `typing_extensions.py`) has no directory to find, and
    an import name that matches neither spelling of the project (pillow -> PIL,
    protobuf -> google) has one under a name this cannot guess. Reported stale, they
    make `sidecar_is_current` rebuild a healthy several-hundred-MB tree on every update.

    Deliberately weaker than _sidecar_damaged_files, which is the check that reads
    every RECORD row: this only has to answer "did the payload arrive at all", the
    question the directory probe was asking. Anything unreadable answers no, so the
    existing failure messages still cover the cases they always covered.
    """
    try:
        recorded = list(dist.files or [])
    except Exception:
        return False
    for entry in recorded:
        parts = tuple(part for part in str(entry).replace("\\", "/").split("/") if part)
        # `..` escapes the tree, console scripts are recorded outside it (pip writes
        # ../../bin/hf), and metadata is not payload: none of them says the files landed.
        if (
            not parts
            or ".." in parts
            or parts[0] in ("bin", "Scripts")
            or parts[0].endswith((".dist-info", ".egg-info"))
        ):
            continue
        try:
            if (root / parts[0]).exists():
                return True
        except OSError:
            continue
    return False


def _sidecar_pin_ok(root: Path, spec: str) -> Optional[str]:
    """None when the pin is satisfied in *root*, else why it is not."""
    from importlib.metadata import distributions

    name, _, wanted = spec.partition("==")
    name = name.strip()
    wanted = wanted.strip()
    if not name:
        return None
    canonical = _canonical(name)
    module = canonical.replace("-", "_")
    # The package tree itself, as _venv_dir_is_valid checks it: a dist-info whose
    # payload was removed still answers every metadata question. Two stats, and for
    # every pin that has a directory that is the whole payload question -- the RECORD
    # fallback below is only reached when neither spelling of the name is one.
    directory_present = any((root / candidate).is_dir() for candidate in (module, canonical))
    found: List[str] = []
    payload_present = False
    try:
        for dist in distributions(path = [str(root)]):
            try:
                dist_name = dist.metadata.get("Name") or ""
            except Exception:
                continue
            if _canonical(dist_name) != canonical:
                continue
            found.append(dist.version or "")
            # Only when the directory probe came up empty, and only until one answers:
            # this reads a RECORD, and paying that for every pin on every update is
            # what the stats above exist to avoid.
            if not directory_present and not payload_present:
                payload_present = _sidecar_payload_present(root, dist)
    except Exception:
        return f"{name} metadata unreadable"
    if not found:
        return f"{name} not installed"
    if not directory_present and not payload_present:
        return f"{name} directory missing"
    if len(found) > 1:
        return f"{name} has {len(found)} metadata records"
    if wanted and found[0] != wanted:
        return f"{name}=={found[0] or 'unknown'}, want {wanted}"
    return None


def _sidecar_damaged_files(
    root: Path,
    limit: int = 3,
    budget_seconds: float = SIDECAR_SCAN_BUDGET_SECONDS,
) -> List[str]:
    """RECORD rows under a sidecar that are gone, truncated, or built for another CPython.

    Fails open everywhere: the caller's answer to a finding is to delete several hundred
    MB and refetch, so anything unreadable reports nothing rather than guessing. Bounded,
    because no installer wraps this in a timeout and a stalled mount would wedge setup.
    """
    import csv
    import io
    import stat

    deadline = time.monotonic() + budget_seconds if budget_seconds > 0 else None
    ext_tag = _current_ext_tag()
    entries: List[Tuple[str, str, Optional[int], Path, str]] = []
    owners: Dict[str, int] = {}
    try:
        dist_infos = sorted(root.glob("*.dist-info"))
    except OSError:
        return []
    for dist_info in dist_infos:
        name = dist_info.name.split("-")[0]
        try:
            record = (dist_info / "RECORD").read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            # Absent or unreadable RECORD says nothing about damage.
            continue
        try:
            rows = list(csv.reader(io.StringIO(record)))
        except csv.Error:
            continue
        for row in rows:
            rel = row[0] if row else ""
            if not rel or rel.endswith("/"):
                continue
            if ".dist-info/" in rel or ".egg-info/" in rel or rel.endswith(".pyc"):
                continue
            parts = tuple(part for part in rel.replace("\\", "/").split("/") if part)
            # Console scripts are not checkable in a flat --target tree, and believing
            # them fails CLOSED on a healthy sidecar (pip records ../../bin/hf, uv bin/hf).
            # Nothing here is ever put on PATH.
            if (
                rel.startswith("/")
                or (len(rel) > 1 and rel[1] == ":")
                or ".." in parts
                or (parts and parts[0] in ("bin", "Scripts"))
            ):
                continue
            target = root / rel
            key = os.path.normcase(str(target))
            # Before the filter: a dropped row still owns the path it claims.
            owners[key] = owners.get(key, 0) + 1
            if len(parts) > 1 and parts[0] in _SHARED_NON_RUNTIME_ROOTS:
                continue
            recorded: Optional[int] = None
            if len(row) >= 3 and row[2] and parts[-1] not in _INSTALLER_REWRITTEN_NAMES:
                try:
                    recorded = int(row[2])
                except ValueError:
                    recorded = None
            entries.append((name, rel, recorded, target, key))

    found: List[str] = []
    for name, rel, recorded, target, key in entries:
        # Every row: batching a deadline let one slow mount overrun it by a minute.
        if deadline is not None and time.monotonic() > deadline:
            return found
        try:
            info = target.stat()
        except (FileNotFoundError, NotADirectoryError):
            found.append(f"{name}: {rel} is missing")
        except OSError:
            # Unreadable is not gone, and the answer costs a several-hundred-MB refetch.
            continue
        else:
            if not stat.S_ISREG(info.st_mode):
                found.append(f"{name}: {rel} is not a regular file")
            # A path two distributions claim makes the recorded SIZES ambiguous;
            # a larger file is a packaging collision, not damage.
            elif owners[key] == 1 and recorded is not None and info.st_size < recorded:
                found.append(f"{name}: {rel} is {info.st_size} bytes, expected {recorded}")
            elif rel.endswith((".so", ".pyd")):
                # The BASENAME alone decides: a directory carrying a wheel-style tag
                # (pkg.cp312.libs/) says nothing about the untagged binary inside it.
                base = rel.replace("\\", "/").rsplit("/", 1)[-1]
                match = _EXT_VERSION_TAG_RE.search(base)
                if match and match.group(1) != ext_tag:
                    found.append(
                        f"{name}: {rel} targets cp{match.group(1)}, interpreter is cp{ext_tag}"
                    )
                elif match is None and ext_tag.endswith("t") and _ABI3_EXT_RE.search(base):
                    found.append(
                        f"{name}: {rel} is a stable-ABI build, which free-threaded "
                        f"cp{ext_tag} cannot load"
                    )
        if len(found) >= limit:
            return found
    return found


def sidecar_is_current(
    venv_dir,
    pins: Sequence[str],
    budget_seconds: float = SIDECAR_SCAN_BUDGET_SECONDS,
) -> Tuple[bool, str]:
    """Whether a transformers sidecar directory already holds exactly *pins*, intact.

    `(True, "")` or `(False, reason)`. The reason is logged by the setup scripts, so
    a rebuild always says what it is repairing.

    *pins* are `name==version` or a bare `name` (present at any version). Every check
    is on-disk evidence: nothing here trusts a previous run's record, because the
    directory is what the next `import transformers` will read.
    """
    root = Path(venv_dir)
    try:
        if not root.is_dir():
            return False, "missing"
        if not os.listdir(root):
            return False, "empty"
    except OSError as exc:
        return False, f"unreadable ({exc.__class__.__name__})"
    # Rebuilding means `rm -rf`, so an unowned directory must not be called current:
    # the caller would either delete someone else's tree or abort mid-update.
    if not (root / SIDECAR_OWNED_MARKER).is_file():
        return False, f"no {SIDECAR_OWNED_MARKER} marker"
    for spec in pins:
        problem = _sidecar_pin_ok(root, spec)
        if problem is not None:
            return False, problem
    damaged = _sidecar_damaged_files(root, budget_seconds = budget_seconds)
    if damaged:
        return False, "; ".join(damaged)
    return True, ""


# -- CLI shim -----------------------------------------------------------------
#
# setup.sh and setup.ps1 both need `sidecar_is_current`, and a shell reimplementation of
# it is what let the two drift the last time. Exit 0 current, 1 not current, 2 anything
# this module does not implement.
#
# Both shells also require the marker line below before believing exit 0: an
# install_manifest.py predating this shim has no `__main__` block at all, so running it
# exits 0 with no output, and a bare exit code would read that as "current".
_SIDECAR_CLI_MARKER = "sidecar:"


def _sidecar_cli(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print("usage: install_manifest.py sidecar <dir> <pin>...", file = sys.stderr)
        return 2
    current, reason = sidecar_is_current(argv[0], tuple(argv[1:]))
    print(f"{_SIDECAR_CLI_MARKER} {'current' if current else reason}")
    return 0 if current else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["sidecar"]:
        sys.exit(_sidecar_cli(sys.argv[2:]))
    print(f"usage: {os.path.basename(__file__)} sidecar <dir> <pin>...", file = sys.stderr)
    sys.exit(2)
