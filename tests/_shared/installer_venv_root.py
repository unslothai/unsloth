# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep an in-process installer run from writing the venv root it is running in.

``install_manifest.venv_root()`` is ``Path(sys.prefix)``: ONE directory, shared by every
xdist worker in a run and by every subprocess any of them spawns. ``install_python_stack()``
removes the manifest there, writes ``.unsloth-no-torch`` there, and writes the manifest
back. Several suites drive ``install_python_stack()`` in process with only
``subprocess.run`` mocked, so all three writes land in the real venv and outlive the test
that made them.

That is the mechanism behind ``test_amd_fastpath_probe.py`` failing its
``[2.9.0+cpu-None-0]`` case on 17 "Repo tests (CPU)" runs across 7 branches in a single
day while passing in isolation, on branches that had touched no install file. The CLI
resolves ``NO_TORCH`` at import through ``install_manifest.recorded_no_torch()``, which
reads exactly those two paths, so for as long as a leaked marker said no-torch every
``install_python_stack.py`` subprocess in the run returned False from
``_amd_torch_needs_dependency_pass()`` at its first line and exited 1 with stdout and
stderr both empty.

Redirecting the resolver rather than patching each harness is deliberate. The hazard is
structural: any new harness that calls ``install_python_stack()`` re-introduces it, and
nothing in a pass count can show it, because the damage is in the difference between the
tree before the run and the tree after. Every test that cares about a particular root
already passes ``root=`` explicitly, so nothing under test wants the real one.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _import_install_manifest():
    """Register ``studio/install_manifest.py`` under the name the installer imports it by.

    IMPORTING IT RATHER THAN SKIPPING WHEN IT IS ABSENT, and that distinction is the whole
    of the containment on one of the two roots. Keying on ``sys.modules`` alone made this
    fixture a no-op whenever nothing had imported the installer YET, and the backend suites
    import it lazily inside the test body (``test_torchao_select.py`` pops
    ``install_python_stack`` and re-imports it from a ``_load_module`` helper), so at fixture
    setup the module is absent, the fixture returned having done nothing, and the import then
    happened inside the test.

    Measured: with that early return in place, running the one leaking backend test on its own
    still rewrote the real manifest, while running the whole file did not -- containment that
    depended on some earlier test having imported the module first. Order-dependent containment
    is the failure this fixture exists to remove, so the module is imported here instead.

    ``install_manifest`` imports nothing heavier than ``json`` and ``pathlib``, so this costs a
    suite that never touches the installer one small module rather than a torch import.
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

    The directory is not created unless something actually resolves the root, and it is one
    directory per test, so a run that removes the manifest and writes it back sees its own
    writes.
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
