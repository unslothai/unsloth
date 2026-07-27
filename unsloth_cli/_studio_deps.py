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
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import typer

# unsloth_cli/_studio_deps.py -> one parent up is the package root (site-packages
# after pip install, or the repo root for an editable install).
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


def install_state(extra_roots: Sequence[Path] = ()) -> dict:
    """verify_install() result, or a permissive answer when the helper is absent.

    A tree predating install_manifest.py must not be called broken by a newer
    CLI; the preflight already treats a CLI that cannot answer as stale via
    desktop_manageability_version.
    """
    module = load_install_manifest_module(extra_roots)
    if module is None:
        return {
            "ok": True,
            "manifest_ok": True,
            "deps_ok": True,
            "missing": [],
            "reason": None,
        }
    try:
        return module.verify_install()
    except Exception as exc:
        return {
            "ok": False,
            "manifest_ok": False,
            "deps_ok": False,
            "missing": [],
            "reason": f"studio_install_check_failed:{type(exc).__name__}",
        }


def _missing_packages(first: Optional[str]) -> str:
    """Best-effort list of every missing studio package, not just the first."""
    module = load_install_manifest_module()
    names: List[str] = []
    if module is not None:
        try:
            names = list(module.missing_requirements())
        except Exception:
            names = []
    if not names and first:
        names = [first]
    return ", ".join(names)


@contextlib.contextmanager
def studio_backend_imports(feature: str = "This command"):
    """Report a missing Studio dependency as a message instead of a traceback.

    Only ModuleNotFoundError is intercepted; any other ImportError from the
    backend is a real bug and keeps its traceback.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        missing = _missing_packages(exc.name)
        typer.echo(
            f"Error: {feature} needs the Unsloth Studio dependencies, " f"which are not installed.",
            err = True,
        )
        if missing:
            typer.echo(f"  missing: {missing}", err = True)
        typer.echo("", err = True)
        typer.echo("  Studio install:  unsloth studio update", err = True)
        typer.echo('  Plain pip:       pip install "unsloth[studio]"', err = True)
        raise typer.Exit(code = 1) from None
