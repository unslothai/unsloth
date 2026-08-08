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
import os
import re
import stat
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
    """Prefix owning a manifest module, which may be a venv other than ours."""
    path = Path(getattr(module, "__file__", "") or "")
    for parent in path.parents:
        if (parent / "pyvenv.cfg").is_file():
            return parent
    return None


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so PyJWT / pyjwt / py_jwt compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _venv_site_packages(root: Path) -> List[Path]:
    out: List[Path] = []
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        out.extend(sorted(root.glob(pattern)))
    return out


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


def _distributions_in(root: Path) -> Optional[Dict[str, str]]:
    """Canonical distribution name -> version inside another venv.

    importlib.metadata reports the running interpreter only, so a foreign
    site-packages has to be handed to the finder explicitly.
    """
    paths = [str(path) for path in _venv_site_packages(root)]
    if not paths:
        return None
    from importlib.metadata import Distribution, DistributionFinder

    found: Dict[str, str] = {}
    try:
        for dist in Distribution.discover(context = DistributionFinder.Context(path = paths)):
            name = getattr(dist, "name", None) or dist.metadata["Name"]
            if name:
                found.setdefault(_canonical(name), dist.version or "")
    except Exception:
        return None
    return found


def _requirements_root_in(root: Path) -> Optional[Path]:
    for path in _venv_site_packages(root):
        reqs = path / "studio" / "backend" / "requirements"
        if reqs.is_dir():
            return reqs
    return None


def _supports_foreign_root(module) -> bool:
    """A manifest helper predating the installed= parameter cannot describe another venv."""
    try:
        return "installed" in inspect.signature(module.verify_install).parameters
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
    installed = _distributions_in(root) if foreign else None
    req_root = _requirements_root_in(root) if foreign else None
    try:
        if installed is not None and req_root is not None and _supports_foreign_root(module):
            # That venv's own metadata: unreadable through this interpreter.
            return module.verify_install(root = root, req_root = req_root, installed = installed)
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
    """
    import sysconfig

    paths = []
    for key in ("purelib", "platlib"):
        try:
            entry = sysconfig.get_paths().get(key)
        except Exception:
            continue
        if entry and entry not in paths and os.path.isdir(entry):
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
    of the `limit` budget. Its ownership claim is still counted: dropping a
    colliding row before the tally is what made a retained row look singly
    owned, which is how the size of somebody else's file got compared against
    unsloth_zoo's RECORD.
    """
    parts = tuple(p for p in rel.replace("\\", "/").split("/") if p and p != ".")
    return len(parts) > 1 and parts[0] in _SHARED_NON_RUNTIME_ROOTS


def _installer_rewritten(rel: str) -> bool:
    """A file our own setup rewrites in place, so its recorded size drifts.

    Only the size is unreliable. The file disappearing is still damage, so the
    row is kept and only its size dropped.
    """
    return rel.replace("\\", "/").rsplit("/", 1)[-1] in _INSTALLER_REWRITTEN_NAMES


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
    from importlib.metadata import distributions

    entries: List[tuple] = []
    owners: Dict[str, int] = {}
    for dist in distributions(**_scan_paths()):
        try:
            name = dist.metadata["Name"] or "?"
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
            # Every real claim counts, including one about to be filtered out.
            # Filtering first is what let a dropped row's file be measured
            # against a retained row's RECORD.
            owners[key] = owners.get(key, 0) + 1
            if _shared_non_runtime(rel):
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
