# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for UNSLOTH_PYTORCH_MIRROR env var in install_python_stack.py."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# install_python_stack.py lives at repo_root/studio/install_python_stack.py
_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"

OFFICIAL_URL = "https://download.pytorch.org/whl"


def _reload_whl_base(monkeypatch, mirror_value = None):
    """(Re-)import install_python_stack with a controlled env and return _PYTORCH_WHL_BASE."""
    # Remove cached module so the module-level assignment re-executes
    sys.modules.pop("install_python_stack", None)

    if mirror_value is None:
        monkeypatch.delenv("UNSLOTH_PYTORCH_MIRROR", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", mirror_value)

    # Temporarily add the script's directory to sys.path for import
    script_dir = str(_INSTALL_SCRIPT.parent)
    monkeypatch.syspath_prepend(script_dir)

    import install_python_stack

    return install_python_stack._PYTORCH_WHL_BASE


class TestPyTorchMirrorEnvVar:
    """UNSLOTH_PYTORCH_MIRROR controls _PYTORCH_WHL_BASE in install_python_stack."""

    def test_unset_uses_official_url(self, monkeypatch):
        assert _reload_whl_base(monkeypatch) == OFFICIAL_URL

    def test_empty_string_falls_back_to_official(self, monkeypatch):
        assert _reload_whl_base(monkeypatch, "") == OFFICIAL_URL

    def test_custom_mirror_is_used(self, monkeypatch):
        mirror = "https://mirrors.nju.edu.cn/pytorch/whl"
        assert _reload_whl_base(monkeypatch, mirror) == mirror

    def test_trailing_slash_stripped(self, monkeypatch):
        result = _reload_whl_base(monkeypatch, "https://example.com/whl/")
        assert result == "https://example.com/whl"
