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

import sys


def contain_installer_venv_root(monkeypatch, tmp_path_factory) -> None:
    """Point ``install_manifest.venv_root`` at a per-test directory.

    A no-op unless the module is already imported, so suites that never touch the
    installer pay one dict lookup. The directory is not created unless something
    actually resolves the root, and it is one directory per test, so a run that removes
    the manifest and writes it back sees its own writes.
    """
    manifest = sys.modules.get("install_manifest")
    if manifest is None:
        return
    resolved: list = []

    def _contained_root():
        if not resolved:
            resolved.append(tmp_path_factory.mktemp("installer_venv_root"))
        return resolved[0]

    monkeypatch.setattr(manifest, "venv_root", _contained_root)
