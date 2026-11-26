# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load one `unsloth/models/<name>.py` file by path, without running `unsloth/__init__.py`.

The gloo tests drive the parallel modules (tp_native, ep_native, ...) directly in spawned CPU ranks,
where importing the package would pull in the GPU checks. The loader has no dependency on any
particular model family, so a test for one family can share it without inheriting another family's
skip (a file that importorskips Inkling 2 at module level must not be where this lives)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_models_module(name: str):
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module
