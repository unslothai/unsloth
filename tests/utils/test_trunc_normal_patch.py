# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for trunc_normal low-precision patch compatibility."""

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


_MISSING = object()


def _load_import_fixes_module():
    repo_root = Path(__file__).resolve().parents[2]
    import_fixes_path = repo_root / "unsloth" / "import_fixes.py"
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_local", import_fixes_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _getattr_or_missing(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else _MISSING


def _restore_attr(obj, name, value):
    if value is _MISSING:
        if hasattr(obj, name):
            delattr(obj, name)
        return
    setattr(obj, name, value)


def test_trunc_normal_patch_accepts_positional_generator():
    import_fixes = _load_import_fixes_module()
    patch_fn = import_fixes.patch_trunc_normal_precision_issue

    init_mod = torch.nn.init
    old_fn = init_mod.trunc_normal_
    old_patched = _getattr_or_missing(init_mod, "_unsloth_trunc_normal_patched")
    old_original = _getattr_or_missing(init_mod, "_unsloth_trunc_normal_original")
    try:
        # Normalize to an unpatched baseline before applying the patch.
        if old_original is not _MISSING:
            init_mod.trunc_normal_ = old_original
        if hasattr(init_mod, "_unsloth_trunc_normal_patched"):
            delattr(init_mod, "_unsloth_trunc_normal_patched")
        if hasattr(init_mod, "_unsloth_trunc_normal_original"):
            delattr(init_mod, "_unsloth_trunc_normal_original")

        patch_fn()
        sig = inspect.signature(init_mod.trunc_normal_)
        assert "generator" in sig.parameters
        assert sig.parameters["generator"].kind is not inspect.Parameter.KEYWORD_ONLY

        tensor = torch.empty(1024, dtype = torch.float32)
        gen = torch.Generator()
        gen.manual_seed(3407)

        init_mod.trunc_normal_(tensor, 0.0, 1.0, -2.0, 2.0, gen)
        init_mod.trunc_normal_(tensor, mean = 0.0, std = 1.0, a = -2.0, b = 2.0, generator = gen)
    finally:
        init_mod.trunc_normal_ = old_fn
        _restore_attr(init_mod, "_unsloth_trunc_normal_patched", old_patched)
        _restore_attr(init_mod, "_unsloth_trunc_normal_original", old_original)


def test_trunc_normal_patch_rejects_invalid_generator():
    import_fixes = _load_import_fixes_module()
    patch_fn = import_fixes.patch_trunc_normal_precision_issue

    init_mod = torch.nn.init
    old_fn = init_mod.trunc_normal_
    old_patched = _getattr_or_missing(init_mod, "_unsloth_trunc_normal_patched")
    old_original = _getattr_or_missing(init_mod, "_unsloth_trunc_normal_original")
    try:
        if old_original is not _MISSING:
            init_mod.trunc_normal_ = old_original
        if hasattr(init_mod, "_unsloth_trunc_normal_patched"):
            delattr(init_mod, "_unsloth_trunc_normal_patched")
        if hasattr(init_mod, "_unsloth_trunc_normal_original"):
            delattr(init_mod, "_unsloth_trunc_normal_original")

        patch_fn()
        sig = inspect.signature(init_mod.trunc_normal_)
        if "generator" not in sig.parameters:
            pytest.skip("torch.nn.init.trunc_normal_ lacks a generator parameter")

        tensor = torch.empty(16, dtype = torch.float32)
        with pytest.raises(TypeError):
            init_mod.trunc_normal_(tensor, generator = 123)
    finally:
        init_mod.trunc_normal_ = old_fn
        _restore_attr(init_mod, "_unsloth_trunc_normal_patched", old_patched)
        _restore_attr(init_mod, "_unsloth_trunc_normal_original", old_original)
