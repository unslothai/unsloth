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
import re
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
