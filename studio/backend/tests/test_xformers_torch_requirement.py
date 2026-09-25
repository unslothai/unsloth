# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check xFormers torch requirements, version tags, and unreadable metadata (#11545)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

from utils.wheel_utils import xformers_torch_requirement_unmet


def _installed(monkeypatch, *, xformers, torch, requires):
    versions = {"xformers": xformers, "torch": torch}

    def _version(name):
        if versions.get(name) is None:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr("importlib.metadata.version", _version)
    monkeypatch.setattr("importlib.metadata.requires", lambda name: requires)


@pytest.mark.parametrize(
    "xformers, torch, requires, expected",
    [
        ("0.0.35", "2.6.0+cu124", ["torch>=2.10"], ("0.0.35", ">=2.10", "2.6.0+cu124")),
        # 0.0.35 is built against 2.10 on the stable ABI, so a later torch keeps it.
        ("0.0.35", "2.11.0+cu130", ["torch>=2.10"], None),
        ("0.0.29.post3", "2.6.0+cu124", ["torch==2.6.0", "numpy"], None),
        (
            "0.0.29.post3",
            "2.7.1+cu126",
            ["torch==2.6.0"],
            ("0.0.29.post3", "==2.6.0", "2.7.1+cu126"),
        ),
        # AMD's Windows torch is a prerelease; it still satisfies a floor below it.
        ("0.0.32.post2", "2.8.0a0+gitfc14c65", ["torch>=2.7"], None),
        ("0.0.35", "2.6.0+cu124", ["torch>=2.10; extra == 'dev'"], None),
    ],
)
def test_the_declared_torch_requirement_decides(monkeypatch, xformers, torch, requires, expected):
    _installed(monkeypatch, xformers = xformers, torch = torch, requires = requires)
    assert xformers_torch_requirement_unmet() == expected


@pytest.mark.parametrize(
    "xformers, torch, requires",
    [
        (None, "2.6.0+cu124", ["torch>=2.10"]),
        ("0.0.35", None, ["torch>=2.10"]),
        ("0.0.35", "2.6.0+cu124", None),
        ("0.0.35", "not a version", ["torch>=2.10"]),
        ("0.0.35", "2.6.0+cu124", ["torch >= >= 2.10"]),
    ],
    ids = ["no-xformers", "no-torch", "no-requirements", "bad-torch-version", "bad-requirement"],
)
def test_anything_unreadable_is_not_a_verdict(monkeypatch, xformers, torch, requires):
    _installed(monkeypatch, xformers = xformers, torch = torch, requires = requires)
    assert xformers_torch_requirement_unmet() is None
