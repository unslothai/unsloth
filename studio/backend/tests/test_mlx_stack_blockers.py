# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX gate has to say what it is unhappy about.

`mlx_unavailable` is a single verdict covering three packages and four runtime
imports, and the greyed-out Train row could only answer it with "run `unsloth
studio update`". That is a dead end for the usual cause: an update that ran, and
a resolver backtrack that left one package missing or too old for the pinned
transformers. These cover the blocker list that message is built from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


def _fake_versions(monkeypatch, installed: dict[str, str]):
    """Report `installed` as the distributions present, and nothing else."""
    from importlib.metadata import PackageNotFoundError

    def version(name: str) -> str:
        if name in installed:
            return installed[name]
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", version)


def test_a_healthy_stack_reports_no_blockers(monkeypatch):
    _fake_versions(monkeypatch, {"mlx": "0.30.0", "mlx-lm": "0.30.0", "mlx-vlm": "0.5.0"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    assert mr.mlx_stack_blockers() == []
    assert mr.mlx_stack_available() is True


def test_a_missing_package_is_named_with_the_version_it_needs(monkeypatch):
    _fake_versions(monkeypatch, {"mlx": "0.30.0", "mlx-lm": "0.30.0"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert any("mlx-vlm is not installed" in blocker for blocker in blockers)
    assert any("0.4.4" in blocker for blocker in blockers)
    assert mr.mlx_stack_available() is False


def test_a_backtracked_package_names_the_version_it_found(monkeypatch):
    # The reported shape: present, importable, and too old for VLM Train/Export.
    _fake_versions(monkeypatch, {"mlx": "0.30.0", "mlx-lm": "0.30.0", "mlx-vlm": "0.1.0"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert blockers == ["mlx-vlm 0.1.0 is older than 0.4.4"]


def test_every_bad_package_is_listed_not_just_the_first(monkeypatch):
    _fake_versions(monkeypatch, {"mlx": "0.1.0"})
    monkeypatch.setattr(mr, "_mlx_runtime_import_blocker", lambda: None)
    blockers = mr.mlx_stack_blockers()
    assert len(blockers) == 3, blockers


def test_an_import_that_raises_is_reported_with_its_error(monkeypatch):
    # Versions satisfied but the module will not load, which is what a mlx-vlm
    # built against a different transformers looks like from here.
    _fake_versions(monkeypatch, {"mlx": "0.30.0", "mlx-lm": "0.30.0", "mlx-vlm": "0.5.0"})

    def explode(module: str):
        raise ImportError("cannot import name 'AutoProcessor' from 'transformers'")

    monkeypatch.setattr(mr.importlib, "import_module", explode)
    blockers = mr.mlx_stack_blockers()
    assert len(blockers) == 1
    assert "does not import" in blockers[0]
    assert "AutoProcessor" in blockers[0]
    assert mr.mlx_stack_available() is False


def test_versions_are_checked_before_imports(monkeypatch):
    """A too-old package must be named without loading it into this process."""
    _fake_versions(monkeypatch, {"mlx": "0.30.0", "mlx-lm": "0.30.0", "mlx-vlm": "0.1.0"})

    def never(module: str):
        raise AssertionError("imported a package the version check already rejected")

    monkeypatch.setattr(mr.importlib, "import_module", never)
    assert mr.mlx_stack_blockers() == ["mlx-vlm 0.1.0 is older than 0.4.4"]


def test_the_detail_line_never_raises_and_stays_short(monkeypatch):
    from utils.hardware import hardware as hw

    def explode() -> list[str]:
        raise RuntimeError("no")

    monkeypatch.setattr(mr, "mlx_stack_blockers", explode)
    assert hw._mlx_stack_detail() is None

    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: [])
    assert hw._mlx_stack_detail() is None

    monkeypatch.setattr(mr, "mlx_stack_blockers", lambda: ["a", "b", "c", "d"])
    detail = hw._mlx_stack_detail()
    assert detail == "a; b; c"


@pytest.mark.parametrize("reason", ["intel_mac", "no_gpu", "detection_failed", None])
def test_only_the_mlx_verdict_carries_a_detail(monkeypatch, reason):
    """Nothing else has anything specific to add, so nothing else may claim to."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    assert hw.CHAT_ONLY_DETAIL is None
