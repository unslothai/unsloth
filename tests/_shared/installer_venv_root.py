# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep an in-process installer run from writing the venv root it is running in.

``install_manifest.venv_root()`` is ``Path(sys.prefix)``: ONE directory, shared by every
xdist worker and every subprocess they spawn. ``install_python_stack()`` removes the
manifest there, writes ``.unsloth-no-torch`` there and writes the manifest back; several
suites drive it in process with only ``subprocess.run`` mocked, so all three writes land in
the real venv and outlive the test that made them.

That is why ``test_amd_fastpath_probe.py`` failed its ``[2.9.0+cpu-None-0]`` case on 17
"Repo tests (CPU)" runs across 7 branches in one day while passing in isolation, on
branches touching no install file: the CLI resolves ``NO_TORCH`` at import through
``install_manifest.recorded_no_torch()``, which reads exactly those two paths, so a leaked
marker made every ``install_python_stack.py`` subprocess return False from
``_amd_torch_needs_dependency_pass()`` at its first line and exit 1 with both streams empty.

Redirecting the resolver beats patching each harness: the hazard is structural, any new
harness calling ``install_python_stack()`` re-introduces it, and no pass count can show it
because the damage is the difference between the tree before the run and the tree after.
Every test that cares about a particular root already passes ``root=`` explicitly.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _import_install_manifest():
    """Register ``studio/install_manifest.py`` under the name the installer imports it by.

    Imports it rather than skipping when it is absent, which is the whole of the containment
    on one of the two roots. Keying on ``sys.modules`` alone made the fixture a no-op whenever
    nothing had imported the installer yet, and the backend suites import it lazily inside the
    test body (``test_torchao_select.py`` pops ``install_python_stack`` and re-imports it from
    a ``_load_module`` helper). Measured with that early return in place: the one leaking
    backend test run alone still rewrote the real manifest, while the whole file did not.
    Order-dependent containment is the failure this fixture exists to remove.

    ``install_manifest`` imports nothing heavier than ``json`` and ``pathlib``, so a suite that
    never touches the installer pays one small module, not a torch import.
    """
    manifest = sys.modules.get("install_manifest")
    if manifest is not None:
        return manifest
    for up in pathlib.Path(__file__).resolve().parents:
        candidate = up / "studio" / "install_manifest.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("install_manifest", candidate)
            module = importlib.util.module_from_spec(spec)
            # Registered BEFORE exec so the installer's own `import install_manifest` binds to
            # this object rather than loading a second, unpatched copy.
            sys.modules["install_manifest"] = module
            spec.loader.exec_module(module)
            return module
    return None


def contain_installer_venv_root(monkeypatch, tmp_path_factory) -> None:
    """Point ``install_manifest.venv_root`` at a per-test directory.

    Created lazily and one per test, so a run that removes the manifest and writes it back
    sees its own writes.
    """
    manifest = _import_install_manifest()
    if manifest is None:
        return
    resolved: list = []

    def _contained_root():
        if not resolved:
            resolved.append(tmp_path_factory.mktemp("installer_venv_root"))
        return resolved[0]

    monkeypatch.setattr(manifest, "venv_root", _contained_root)
