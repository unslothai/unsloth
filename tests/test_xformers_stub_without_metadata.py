# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Studio's Windows ROCm run.py puts an xformers stub in sys.modules without installing xformers;
a spawned worker re-runs run.py, so `import unsloth` meets that stub."""

import importlib.machinery
import importlib.metadata
import sys
import types

from unsloth import import_fixes


def test_xformers_stub_without_metadata_is_skipped(monkeypatch):
    stub = types.ModuleType("xformers")
    stub.__spec__ = importlib.machinery.ModuleSpec("xformers", loader = None, is_package = True)
    monkeypatch.setitem(sys.modules, "xformers", stub)

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(import_fixes, "importlib_version", missing)
    assert import_fixes.fix_xformers_performance_issue() is None
