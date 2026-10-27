# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep in-process installer runs from writing the venv root (sys.prefix) shared by all xdist workers."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _import_install_manifest():
    """Imports install_manifest if absent; a sys.modules-only check made containment order-dependent."""
    manifest = sys.modules.get("install_manifest")
    if manifest is not None:
        return manifest
    for up in pathlib.Path(__file__).resolve().parents:
        candidate = up / "studio" / "install_manifest.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("install_manifest", candidate)
            module = importlib.util.module_from_spec(spec)
            # Registered before exec so the installer's import binds to this patched module.
            sys.modules["install_manifest"] = module
            spec.loader.exec_module(module)
            return module
    return None


def contain_installer_venv_root(monkeypatch, tmp_path_factory) -> None:
    """Point install_manifest.venv_root at a fresh per-test directory so installer writes stay contained."""
    manifest = _import_install_manifest()
    if manifest is None:
        return
    resolved: list = []

    def _contained_root():
        if not resolved:
            resolved.append(tmp_path_factory.mktemp("installer_venv_root"))
        return resolved[0]

    monkeypatch.setattr(manifest, "venv_root", _contained_root)
