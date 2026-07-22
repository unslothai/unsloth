# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the runtime managed-Node resolver (studio/backend/utils/node_runtime.py).

The Unsloth frontend installer may provision an isolated Node under
``<UNSLOTH_HOME>/node`` that is never added to the user's PATH. The backend OXC
validator must still find a usable Node at runtime: a version-adequate system
Node, else the managed isolated one. These tests pin that resolution and the
version floor (kept in sync with the setup scripts' Node decision).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# node_runtime imports sibling backend packages by top-level name, so put
# studio/backend on sys.path before importing it.
_BACKEND = Path(__file__).resolve().parents[3] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

nr = importlib.import_module("utils.node_runtime")


@pytest.fixture(autouse = True)
def _clear_resolver_cache():
    nr._reset_resolved_node()
    yield
    nr._reset_resolved_node()


@pytest.mark.parametrize(
    "version,expected",
    [
        ("v20.19.0", True),
        ("v20.18.9", False),
        ("v21.7.0", False),  # Node 21 (odd, non-LTS) is below the bar
        ("v22.12.0", True),
        ("v22.11.0", False),
        ("v23.0.0", True),
        ("v24.17.0", True),
        ("v18.20.0", False),
        ("not-a-version", False),
        ("", False),
    ],
)
def test_version_floor_matches_setup_bar(version, expected):
    assert nr._version_meets_floor(version) is expected


def test_managed_binary_layout_is_host_aware(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    binary = nr.managed_node_binary()
    if os.name == "nt":
        assert binary == tmp_path / "node" / "node.exe"
    else:
        assert binary == tmp_path / "node" / "bin" / "node"


def test_managed_dir_uses_legacy_sibling_by_default(monkeypatch):
    # No env override -> ~/.unsloth/node (sibling of ~/.unsloth/studio).
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert nr.managed_node_dir() == Path.home() / ".unsloth" / "node"


def _raise_oserror():
    raise OSError("simulated degraded import environment")


def test_managed_dir_fallback_honors_override(monkeypatch, tmp_path):
    # If utils.paths cannot be loaded / studio_root() fails, the resolver must
    # still honor an explicit STUDIO_HOME override (not silently use legacy).
    import utils.paths.storage_roots as sr

    monkeypatch.setattr(sr, "studio_root", _raise_oserror)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    assert nr.managed_node_dir() == tmp_path / "node"


def test_managed_dir_fallback_legacy_without_override(monkeypatch):
    import utils.paths.storage_roots as sr

    monkeypatch.setattr(sr, "studio_root", _raise_oserror)
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert nr.managed_node_dir() == Path.home() / ".unsloth" / "node"


def test_managed_dir_honors_studio_home_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path))
    assert nr.managed_node_dir() == tmp_path / "node"


def test_managed_dir_unsloth_studio_home_wins_over_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path / "other"))
    assert nr.managed_node_dir() == tmp_path / "node"


def test_managed_dir_legacy_valued_override_uses_sibling(monkeypatch):
    # An override set explicitly to the legacy default maps to the sibling
    # ~/.unsloth/node (matching setup.sh / setup.ps1), not ~/.unsloth/studio/node.
    legacy = Path.home() / ".unsloth" / "studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(legacy))
    assert nr.managed_node_dir() == Path.home() / ".unsloth" / "node"


def test_resolve_prefers_adequate_system_node(monkeypatch):
    monkeypatch.setattr(
        nr.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None
    )
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: exe == "/usr/bin/node")
    assert nr.resolve_node_executable() == "/usr/bin/node"


def test_resolve_falls_back_to_managed_when_no_system(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    managed = nr.managed_node_binary()
    managed.parent.mkdir(parents = True, exist_ok = True)
    managed.write_text("#!/bin/sh\necho v24.17.0\n")
    monkeypatch.setattr(nr.shutil, "which", lambda name: None)
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: str(exe) == str(managed))
    assert nr.resolve_node_executable() == str(managed)


def test_resolve_prefers_managed_over_unsuitable_system(monkeypatch, tmp_path):
    # System node present but too old; managed isolated Node is adequate.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    managed = nr.managed_node_binary()
    managed.parent.mkdir(parents = True, exist_ok = True)
    managed.write_text("fake")
    monkeypatch.setattr(nr.shutil, "which", lambda name: "/old/node")
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: str(exe) == str(managed))
    assert nr.resolve_node_executable() == str(managed)


def test_resolve_returns_old_system_as_last_resort(monkeypatch, tmp_path):
    # Old system node, no managed install -> preserve pre-isolation behaviour.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(nr.shutil, "which", lambda name: "/old/node")
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: False)
    assert nr.resolve_node_executable() == "/old/node"


def test_resolve_returns_none_when_nothing_available(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))  # managed dir is empty
    monkeypatch.setattr(nr.shutil, "which", lambda name: None)
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: False)
    assert nr.resolve_node_executable() is None


def test_negative_result_is_not_cached(monkeypatch, tmp_path):
    # A Node that appears after the first (empty) probe must be picked up without
    # a restart, so None must not be memoized.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(nr.shutil, "which", lambda name: None)
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: False)
    assert nr.resolve_node_executable() is None

    managed = nr.managed_node_binary()
    managed.parent.mkdir(parents = True, exist_ok = True)
    managed.write_text("now-installed")
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: str(exe) == str(managed))
    assert nr.resolve_node_executable() == str(managed)


def test_positive_result_is_cached(monkeypatch):
    monkeypatch.setattr(nr.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(nr, "_node_version_ok", lambda exe: True)
    assert nr.resolve_node_executable() == "/usr/bin/node"

    # A cached positive result must not re-probe (shutil.which would now raise).
    def _boom(name):
        raise AssertionError("resolver re-probed despite a cached positive result")

    monkeypatch.setattr(nr.shutil, "which", _boom)
    assert nr.resolve_node_executable() == "/usr/bin/node"
