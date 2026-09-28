# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The released torch range must admit what Studio installs and every torch an extra pins.

`unsloth studio update` re-resolves unsloth under this range: an installed torch outside it
is swapped for PyPI's default on the next update. Studio installs torch 2.13 on the Linux
cu130 Python 3.13 route once this range admits it.
"""

import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")

_PYPROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding = "utf-8")
)


def _base_torch() -> Requirement:
    reqs = [Requirement(r) for r in _PYPROJECT["project"]["dependencies"]]
    torch = [r for r in reqs if r.name == "torch"]
    assert len(torch) == 1, torch
    return torch[0]


@pytest.mark.parametrize("release", ["2.4.0", "2.11.0", "2.12.1", "2.13.0", "2.14.0"])
def test_the_released_range_admits_every_supported_torch(release):
    assert Version(release) in _base_torch().specifier


def test_the_released_range_stops_before_an_unvalidated_minor():
    assert Version("2.15.0") not in _base_torch().specifier


def test_every_torch_an_extra_pins_is_inside_the_released_range():
    spec = _base_torch().specifier
    pinned = set()
    for extra in _PYPROJECT["project"].get("optional-dependencies", {}).values():
        for raw in extra:
            req = Requirement(raw)
            if req.name != "torch" or req.url:
                continue
            for s in req.specifier:
                if s.operator == "==":
                    pinned.add(Version(s.version.split("+", 1)[0]))
    outside = sorted(str(v) for v in pinned if v not in spec)
    assert not outside, f"extras pin torch outside the released range: {outside}"
