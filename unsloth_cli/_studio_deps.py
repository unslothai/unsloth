# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio dependency checks shared by the CLI commands.

The wheel ships studio/ and studio.backend*, so train / export / chat /
inference / studio all work after a plain `pip install unsloth` right up to the
point they import the backend. studio_backend_imports() turns the resulting
traceback into one sentence and the two commands that fix it.

Also loads studio/install_manifest.py for `unsloth studio verify-install`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import typer

# One parent up is the package root: site-packages, or the repo root if editable.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_MANIFEST_MODULE = None
_MANIFEST_LOADED = False


def _manifest_candidates(extra_roots: Sequence[Path] = ()) -> Iterable[Path]:
    yield _PACKAGE_ROOT / "studio" / "install_manifest.py"
    roots: List[Path] = [Path(sys.prefix), *extra_roots]
    for root in roots:
        for pattern in (
            "lib/python*/site-packages/studio/install_manifest.py",
            "Lib/site-packages/studio/install_manifest.py",
        ):
            yield from root.glob(pattern)


def load_install_manifest_module(extra_roots: Sequence[Path] = ()):
    """Load studio/install_manifest.py by file path, or None if unavailable.

    By path for the same reason as studio.backend.run: a partial
    site-packages/studio/ tree can shadow an editable install, which is exactly
    what this check exists to detect.
    """
    global _MANIFEST_MODULE, _MANIFEST_LOADED
    if _MANIFEST_LOADED:
        return _MANIFEST_MODULE

    _MANIFEST_LOADED = True
    for path in _manifest_candidates(extra_roots):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("studio.install_manifest", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        _MANIFEST_MODULE = module
        return _MANIFEST_MODULE
    return None


def _venv_root_for_module(module) -> Optional[Path]:
    """Prefix owning a manifest module, which may be a venv other than ours.

    A prefix only owns the module when the module actually lives in that
    prefix's site-packages. An editable checkout (`./install.sh --local`) keeps
    studio/install_manifest.py in the repo, so the first pyvenv.cfg above it is
    whatever venv the clone happens to sit inside -- frequently an unrelated one
    when the repo is cloned into a directory that is itself a virtualenv. Owning
    it there would verify that venv's packages instead of the managed install and
    report every managed dependency as missing.
    """
    path = _resolved(Path(getattr(module, "__file__", "") or ""))
    for parent in path.parents:
        if (parent / "pyvenv.cfg").is_file():
            for site_packages in _venv_site_packages(parent):
                try:
                    path.relative_to(_resolved(site_packages))
                except ValueError:
                    continue
                return parent
            return None
    return None


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so PyJWT / pyjwt / py_jwt compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _venv_executables(root: Path) -> tuple[Path, ...]:
    return (
        (root / "Scripts" / "python.exe",)
        if os.name == "nt"
        else (root / "bin" / "python", root / "bin" / "python3")
    )


def _venv_site_packages(root: Path) -> List[Path]:
    """The site-packages roots used by this venv's active interpreter.

    A venv upgraded in place can retain lib/pythonX.Y directories from older
    interpreters. Globbing all of them makes ordinary packages look duplicated,
    even though only the active interpreter's directory is importable.

    Deduplicated by real path: purelib hardcodes `lib` while platlib follows
    sys.platlibdir, so a lib64 build (Fedora, SuSE) names one directory twice
    through venv's lib64 -> lib symlink.
    """

    def existing_inside_root(values) -> List[Path]:
        resolved_root = _resolved(root)
        out: List[Path] = []
        seen: set = set()
        for value in values:
            if not value:
                continue
            path = Path(value)
            if not path.is_dir():
                continue
            resolved = _resolved(path)
            if not resolved.is_relative_to(resolved_root) or resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
        return out

    if _resolved(root) == _resolved(Path(sys.prefix)):
        import sysconfig

        try:
            configured = sysconfig.get_paths()
        except Exception:
            configured = {}
        current = existing_inside_root(configured.get(key) for key in ("purelib", "platlib"))
        if current:
            return current

    probe = (
        "import json, sysconfig; "
        "p = sysconfig.get_paths(); "
        "print(json.dumps([p.get('purelib'), p.get('platlib')]))"
    )
    for executable in _venv_executables(root):
        if not executable.is_file():
            continue
        try:
            output = subprocess.check_output(
                [str(executable), "-I", "-c", probe],
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            values = json.loads(output)
        except (OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
            continue
        active = existing_inside_root(values if isinstance(values, list) else [])
        if active:
            return active

    windows_site = root / "Lib" / "site-packages"
    if windows_site.is_dir():
        return [windows_site]

    try:
        config = (root / "pyvenv.cfg").read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        config = ""
    match = re.search(r"(?im)^\s*version\s*=\s*(\d+\.\d+)", config)
    if match:
        versioned = root / "lib" / f"python{match.group(1)}" / "site-packages"
        if versioned.is_dir():
            return [versioned]

    # Test fixtures and incomplete venvs may not have a runnable interpreter or
    # version entry yet. One directory is unambiguous; several are not.
    candidates = sorted(root.glob("lib/python*/site-packages"))
    return candidates if len(candidates) == 1 else []


def _managed_root(extra_roots: Sequence[Path]) -> Optional[Path]:
    """A requested venv that is not the one this CLI runs in.

    The wheel ships studio/, so a CLI installed outside the managed venv always
    finds its own copy of the helper first; without this it would then verify
    its own prefix instead of the venv it was asked about.
    """
    running = _resolved(Path(sys.prefix))
    for root in extra_roots:
        if (root / "pyvenv.cfg").is_file() and _resolved(root) != running:
            return root
    return None


def _distributions_in(root: Path) -> Optional[tuple[Dict[str, str], set[str]]]:
    """Installed versions and metadata conflicts inside another venv.

    importlib.metadata reports the running interpreter only, so a foreign
    site-packages has to be handed to the finder explicitly. Keep multiplicity
    beside the single-version map: collapsing it here would let a foreign venv
    with ambiguous core metadata pass manifest verification.
    """
    paths = [str(path) for path in _venv_site_packages(root)]
    if not paths:
        return None
    from importlib.metadata import Distribution, DistributionFinder

    found: Dict[str, str] = {}
    conflicts: set[str] = set()
    for dist in Distribution.discover(context = DistributionFinder.Context(path = paths)):
        try:
            name = dist.metadata.get("Name")
            if name:
                canonical = _canonical(name)
                version = dist.version or ""
                if not version or canonical in found:
                    conflicts.add(canonical)
                if canonical not in found:
                    found[canonical] = version
                continue
        except Exception:
            pass
        # A nameless or unreadable record must not hide a foreign conflict.
        path = getattr(dist, "_path", None)
        stem = os.path.basename(os.fspath(path)) if path is not None else ""
        path_name, separator, _version = stem.removesuffix(".dist-info").rpartition("-")
        if stem.endswith(".dist-info") and separator:
            conflicts.add(_canonical(path_name))
    return found, conflicts


def _requirements_root_in(root: Path) -> Optional[Path]:
    for path in _venv_site_packages(root):
        reqs = path / "studio" / "backend" / "requirements"
        if reqs.is_dir():
            return reqs
    # An editable install can keep studio/ only in its source checkout. Ask the
    # foreign interpreter to follow its .pth/finder instead of treating the
    # absent site-packages copy as an incomplete environment.
    probe = (
        "import json, pathlib, studio; "
        "print(json.dumps(str(pathlib.Path(studio.__file__).resolve().parent / "
        "'backend' / 'requirements')))"
    )
    for executable in _venv_executables(root):
        if not executable.is_file():
            continue
        try:
            reqs = Path(
                json.loads(
                    subprocess.check_output(
                        [str(executable), "-I", "-c", probe],
                        stderr = subprocess.DEVNULL,
                        text = True,
                        timeout = 5,
                    )
                )
            )
        except (OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if reqs.is_dir():
            return reqs
    return None


def _verify_install_supports(module, parameter: str) -> bool:
    try:
        return parameter in inspect.signature(module.verify_install).parameters
    except (TypeError, ValueError):
        return False


def install_state(extra_roots: Sequence[Path] = ()) -> dict:
    """verify_install() result, or incomplete when the helper cannot be loaded.

    studio/install_manifest.py ships in the same wheel as this file, so a tree
    that has one without the other is a torn install, not an old one: a CLI
    predating both never reaches this code, and the desktop already calls it
    stale on desktop_manageability_version. Answering yes here would launch a
    backend whose own files may be just as absent.
    """
    module = load_install_manifest_module(extra_roots)
    if module is None:
        return {
            "ok": False,
            "manifest_ok": False,
            "deps_ok": False,
            "missing": [],
            "reason": "studio_install_manifest_missing",
        }
    # The requested managed venv is the subject, even though the helper above
    # came from this CLI's own tree.
    root = _managed_root(extra_roots) or _venv_root_for_module(module)
    foreign = root is not None and _resolved(root) != _resolved(Path(sys.prefix))
    foreign_distributions = _distributions_in(root) if foreign else None
    installed = foreign_distributions[0] if foreign_distributions is not None else None
    installed_conflicts = foreign_distributions[1] if foreign_distributions is not None else set()
    req_root = _requirements_root_in(root) if foreign else None
    if foreign and (foreign_distributions is None or req_root is None):
        # Never answer for the caller when the requested environment cannot be
        # inspected. That can turn a torn or ambiguous managed venv into a
        # healthy result merely because the external CLI has matching packages.
        return {
            "ok": False,
            "manifest_ok": False,
            "deps_ok": False,
            "missing": [],
            "reason": "studio_install_incomplete",
        }
    try:
        if (
            installed is not None
            and req_root is not None
            and _verify_install_supports(module, "installed")
        ):
            # That venv's own metadata: unreadable through this interpreter.
            kwargs = {"root": root, "req_root": req_root, "installed": installed}
            if _verify_install_supports(module, "installed_conflicts"):
                kwargs["installed_conflicts"] = installed_conflicts
            return module.verify_install(**kwargs)
        state = module.verify_install(root = root)
        if foreign and not state["deps_ok"]:
            # The manifest came from another venv but the dependency walk ran
            # here, so it says nothing about that venv.
            state = dict(state, deps_ok = True, missing = [])
            state["ok"] = state["manifest_ok"]
            state["reason"] = None if state["ok"] else state["reason"]
        return state
    except Exception as exc:
        return {
            "ok": False,
            "manifest_ok": False,
            "deps_ok": False,
            "missing": [],
            "reason": f"studio_install_check_failed:{type(exc).__name__}",
        }


def _scan_paths() -> Dict[str, list]:
    """`path=` kwarg limiting a distribution scan to this interpreter's own tree.

    Empty when the interpreter's site-packages cannot be resolved, which leaves
    the scan at its default of the whole sys.path: over-scanning is the safe
    direction here, since the alternative is looking at nothing.

    Deduplicated by real path, because purelib hardcodes `lib` while platlib
    follows sys.platlibdir. On a lib64 build (Fedora, SuSE) those are two names
    for one directory, and scanning both would report every installed package
    as having duplicate metadata.
    """
    import sysconfig

    paths = []
    seen: set = set()
    for key in ("purelib", "platlib"):
        try:
            entry = sysconfig.get_paths().get(key)
        except Exception:
            continue
        if not entry or not os.path.isdir(entry):
            continue
        try:
            resolved = os.path.realpath(entry)
        except OSError:
            resolved = entry
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(entry)
    return {"path": paths} if paths else {}


# Top-level dirs several wheels write into, so one uninstall deletes another's
# files and the survivor's RECORD describes a file nothing recreates. einx and
# torchao both ship test/conftest.py, and install_python_stack.py
# force-reinstalls torchao every update; unsloth_zoo <= 2026.8.5 shipped
# tests/ and scripts/ into the same squatted namespace.
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

# Rewritten in place by our own setup: setup.ps1/setup.sh run `npm install`
# inside the installed tree, and npm dedupes hoisted entries under
# legacy-peer-deps, shrinking the lockfile below its recorded size.
_INSTALLER_REWRITTEN_NAMES = frozenset(("package-lock.json",))


def _shared_non_runtime(rel: str) -> bool:
    """A row under a top-level dir several wheels write into.

    Ownership of these is unreliable: whichever wheel installed last wins, and
    any of them uninstalling takes the others' files with it. That is a property
    of the path, not of the distribution claiming it, so it holds for what we
    ship too: unsloth_zoo <= 2026.8.5 packaged a top-level tests/ and scripts/,
    nothing imports either at runtime, and a clobbered copy wedged every update
    with `unsloth_zoo: tests/conftest.py is 8107 bytes, expected 11429`. Our
    runtime trees are unaffected, since none of them is named for a shared root.
    Applied while reading RECORD, not when reporting, so the row also stays out
    of the `limit` budget. Its claim is still counted: dropping a colliding row
    before the tally is what made a retained row look singly owned, so another
    distribution's file got measured against unsloth_zoo's RECORD.
    """
    parts = tuple(p for p in rel.replace("\\", "/").split("/") if p and p != ".")
    return len(parts) > 1 and parts[0] in _SHARED_NON_RUNTIME_ROOTS


def _installer_rewritten(rel: str) -> bool:
    """A file our own setup rewrites in place, so its recorded size drifts.

    Only the size is unreliable. The file disappearing is still damage, so the
    row is kept and only its size dropped.
    """
    return rel.replace("\\", "/").rsplit("/", 1)[-1] in _INSTALLER_REWRITTEN_NAMES


def _installed_distribution_groups():
    """Canonical name -> installed metadata records in this interpreter's tree.

    Each record is (distribution, display name, readable). A nameless or
    unreadable METADATA still belongs to the distribution its directory is
    named after, and still makes that distribution ambiguous -- which is what
    install_manifest.installed_versions() reports for the same directory. It
    is never file-checkable, so it is marked unreadable rather than trusted.

    Readable means a name AND a version: install_manifest.metadata_conflict()
    counts an empty version as inconsistent, so trusting such a record here
    would leave the two checks disagreeing about the same directory.
    """
    from importlib.metadata import distributions

    groups: Dict[str, list] = {}
    for dist in distributions(**_scan_paths()):
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if name:
            try:
                version = dist.version
            except Exception:
                version = None
            groups.setdefault(_canonical(name), []).append((dist, name, bool(version)))
            continue
        # Wheel metadata directory names escape name separators as underscores,
        # so splitting off the final version is unambiguous.
        path = getattr(dist, "_path", None)
        stem = os.path.basename(os.fspath(path)) if path is not None else ""
        path_name, separator, _version = stem.removesuffix(".dist-info").rpartition("-")
        if stem.endswith(".dist-info") and separator and path_name:
            groups.setdefault(_canonical(path_name), []).append((dist, path_name, False))
    return groups


def installed_metadata_conflicts(
    limit: int = 8,
    *,
    names: Optional[Sequence[str]] = None,
    exclude_names: Sequence[str] = (),
) -> List[str]:
    """Distributions with multiple metadata records for one canonical name.

    A duplicate is an ambiguous install, even when both records name the same
    version. importlib.metadata.version() chooses the first finder result rather
    than identifying which RECORD owns the package tree, so none of the records
    can safely drive the file-damage check until the package is reinstalled.

    A single unreadable record counts too, matching
    install_manifest.metadata_conflict(): pip cannot parse it either, so it must
    be reported rather than silently skipped.
    """
    included = None if names is None else {_canonical(name) for name in names}
    excluded = {_canonical(name) for name in exclude_names}
    found: List[str] = []
    groups = _installed_distribution_groups()
    for canonical in sorted(groups):
        if (included is not None and canonical not in included) or canonical in excluded:
            continue
        records = groups[canonical]
        if len(records) < 2 and all(readable for _dist, _name, readable in records):
            continue
        details: List[str] = []
        for dist, _name, _readable in records:
            try:
                version = dist.version or "unknown version"
            except Exception:
                version = "unknown version"
            metadata_path = getattr(dist, "_path", None)
            location = Path(str(metadata_path)).name if metadata_path else "unknown metadata path"
            details.append(f"{version} at {location}")
        detail = ", ".join(sorted(details))
        problem = "multiple metadata records" if len(records) > 1 else "unreadable metadata"
        found.append(f"{canonical}: {problem} ({detail})")
        if len(found) >= limit:
            break
    return found


def damaged_installed_files(limit: int = 8) -> List[str]:
    """Installed files that are gone, or shorter than pip recorded.

    pip treats a distribution with intact metadata as already satisfied, so an
    update reinstalls nothing when a package's FILES are damaged: it reports
    success and the backend then dies on import at boot. A missing-package check
    cannot see this, because the damaged module still imports; the observed
    failure was `cannot import name 'Depends' from 'fastapi'`, not a missing
    fastapi. Comparing each RECORD entry against the filesystem does see it, and
    since nothing is imported it costs well under a second even with torch
    installed.

    Only shrinkage and disappearance count, and only for paths a single
    distribution claims. When two claim one path (descript-audio-codec ships a
    top-level tests/__init__.py that another package overwrites) whichever copy
    landed is the one on disk, so its size says nothing about either RECORD --
    in EITHER direction. Sizes are therefore compared after the whole scan, once
    multiply-owned paths are known, rather than during it.

    Rows that cannot be import-time damage are dropped up front, and a file our
    own setup rewrites keeps its existence check but loses its size. The answer
    to a finding is "reinstall over the top", so a file no reinstall would
    change must not produce one. See _shared_non_runtime, _installer_rewritten.

    Multiple metadata records for one canonical distribution name are excluded
    entirely. None is authoritative: an older RECORD can legitimately name files
    a newer wheel removed, while choosing the first or highest version also fails
    after an interrupted upgrade or a deliberate downgrade. Callers surface that
    separate condition through installed_metadata_conflicts().

    Scanned over the interpreter's own site-packages rather than all of
    sys.path. distributions() searches every sys.path entry, so a damaged
    distribution reachable only through an inherited PYTHONPATH would otherwise
    fail every update while sitting outside the installation, where neither
    printed repair command can reach it.

    RECORD is parsed here rather than read through Distribution.files, which
    drops entries whose file no longer exists and so can never report a deletion.

    Describes this interpreter's environment. Callers that may be running
    outside the managed venv should check first; see install_state().
    """
    import csv
    import io

    entries: List[tuple] = []
    owners: Dict[str, int] = {}
    for records in _installed_distribution_groups().values():
        # An unreadable record cannot be trusted to describe the package tree
        # any more than a duplicated one can.
        ambiguous = len(records) > 1 or not all(readable for _dist, _name, readable in records)
        for dist, name, _readable in records:
            try:
                record = dist.read_text("RECORD")
            except Exception:
                # An unreadable or absent RECORD says nothing about damage: editable
                # installs and system packages legitimately have none.
                continue
            if not record:
                continue
            for row in csv.reader(io.StringIO(record)):
                rel = row[0] if row else ""
                # A trailing slash is a directory entry, which has nothing to check.
                if not rel or rel.endswith("/"):
                    continue
                # Installer-owned metadata is rewritten in place and drifts from the
                # size recorded inside itself; .pyc is regenerated from source.
                if ".dist-info/" in rel or ".egg-info/" in rel or rel.endswith(".pyc"):
                    continue
                try:
                    target = dist.locate_file(rel)
                except Exception:
                    continue
                key = os.path.normcase(str(target))
                # Before either filter: a row we cannot verify still owns the path
                # it claims, so another distribution's size is ambiguous too.
                owners[key] = owners.get(key, 0) + 1
                if ambiguous or _shared_non_runtime(rel):
                    continue
                # The size field is optional and real wheels do leave it blank. Keep
                # the row anyway with an unknown size: existence is still checkable,
                # and dropping the row meant a deletion went unreported.
                recorded: Optional[int] = None
                if len(row) >= 3 and row[2] and not _installer_rewritten(rel):
                    try:
                        recorded = int(row[2])
                    except ValueError:
                        recorded = None
                entries.append((name, rel, recorded, target, key))

    found: List[str] = []
    for name, rel, recorded, target, key in entries:
        try:
            info = target.stat()
        except OSError:
            # Multiple ownership makes the recorded SIZES ambiguous; it cannot
            # explain the file being gone, so this branch runs for shared paths
            # too.
            found.append(f"{name}: {rel} is missing")
        else:
            if not stat.S_ISREG(info.st_mode):
                # A directory standing in for a recorded module still imports as
                # something else, and on POSIX its st_size (commonly 4096) can
                # sail past the shrinkage test.
                found.append(f"{name}: {rel} is not a regular file")
            elif owners[key] == 1 and recorded is not None and info.st_size < recorded:
                found.append(f"{name}: {rel} is {info.st_size} bytes, expected {recorded}")
        if len(found) >= limit:
            return found
    return found


def running_outside_managed_venv(extra_roots: Sequence[Path] = ()) -> bool:
    """True when this interpreter is not the managed venv the caller means.

    A pip-installed CLI can drive an update into a venv it does not live in, and
    anything answered from this interpreter would then describe the wrong tree.
    """
    if not (Path(sys.prefix) / "pyvenv.cfg").is_file():
        # Not a venv at all. On Colab setup.sh deliberately installs the backend
        # into the system Python, whose distro-packaged RECORDs legitimately list
        # files the distro never installed (PEP 627), so a file check here
        # describes the distro rather than Studio.
        return True
    return _managed_root(extra_roots) is not None


def _missing_studio_packages() -> List[str]:
    """Studio packages studio.txt asks for and the venv does not have."""
    module = load_install_manifest_module()
    if module is None:
        return []
    try:
        return list(module.missing_requirements())
    except Exception:
        return []


# studio.txt names distributions, ModuleNotFoundError names the import. Only
# pairs differing by more than PEP 503 normalisation need an entry, and each
# import name below is itself a real but unrelated PyPI project.
_IMPORT_TO_DISTRIBUTION = {
    "jwt": "pyjwt",
    "docx": "python-docx",
    "fitz": "pymupdf",
}


@contextlib.contextmanager
def studio_backend_imports(feature: str = "This command", *, studio_only: bool = False):
    """Report a missing dependency as a message instead of a traceback.

    Only ModuleNotFoundError is intercepted; any other ImportError from the
    backend is a real bug and keeps its traceback.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        studio_missing = _missing_studio_packages()
        # The failed import may not be a studio dependency at all: `train`
        # reaches torch through the same wrapper, so only offer the extra when
        # it helps.
        trigger = exc.name or ""
        # Match on the owning distribution, never the import: `pip install jwt`
        # (or fastmcp.server) installs the wrong thing or nothing at all.
        top = trigger.split(".", 1)[0]
        needed = _IMPORT_TO_DISTRIBUTION.get(top, top)
        wanted = _canonical(needed)
        from_studio = not trigger or any(_canonical(name) == wanted for name in studio_missing)
        if studio_only and not from_studio:
            raise
        typer.echo(
            f"Error: {feature} needs {needed or 'a dependency'}, which is not installed.",
            err = True,
        )
        others = [name for name in studio_missing if _canonical(name) != wanted]
        if others:
            typer.echo(f"  also missing: {', '.join(others)}", err = True)
        typer.echo("", err = True)
        if not from_studio:
            typer.echo(f"  Install it:      pip install {needed}", err = True)
        if from_studio or others:
            typer.echo("  Studio install:  unsloth studio update", err = True)
            typer.echo('  Plain pip:       pip install "unsloth[studio]"', err = True)
        raise typer.Exit(code = 1) from None
