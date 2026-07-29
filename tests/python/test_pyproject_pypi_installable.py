# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""This branch is what gets uploaded to PyPI, so its metadata has to satisfy PyPI's rules.

PyPI rejects any PEP 508 direct reference in ``Requires-Dist`` with
``Invalid value for requires_dist. Error: Can't have direct dependency``
(pypi/warehouse#7136), and ``twine check`` does not catch it beforehand
(pypa/twine#726) -- the upload just 400s. ``main`` carries hundreds of direct URL
requirements for the CUDA/XPU/ROCm wheel indexes, so every merge from ``main`` is a
chance to import one here and break publishing. Nothing else guards that.

Offline by design: structural checks only, no network.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_EXTRA_REF = re.compile(r"unsloth\[([\w\-\.,\s]+)\]")


def _load() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))


def _all_requirements() -> list[tuple[str, str]]:
    """(extra name, requirement string) for the base deps and every extra."""
    project = _load()["project"]
    items = [("<base>", r) for r in project.get("dependencies", [])]
    for extra, deps in project.get("optional-dependencies", {}).items():
        items += [(extra, r) for r in deps]
    return items


class TestNoDirectUrlRequirements:
    def test_no_requirement_uses_a_direct_url(self):
        offenders = [
            (extra, req)
            for extra, req in _all_requirements()
            if "@ http://" in req or "@ https://" in req or "@ git+" in req
        ]
        assert offenders == [], (
            "PyPI rejects direct references in Requires-Dist, so these would fail the "
            f"upload (only found at publish time): {offenders[:5]}"
        )

    def test_every_requirement_parses(self):
        packaging_requirements = pytest.importorskip("packaging.requirements")
        bad = []
        for extra, req in _all_requirements():
            try:
                packaging_requirements.Requirement(req)
            except Exception as exc:  # noqa: BLE001 - report every malformed spec at once
                bad.append((extra, req, str(exc)))
        assert bad == [], f"unparseable requirements: {bad}"


class TestExtraReferencesResolve:
    def test_every_unsloth_extra_reference_exists(self):
        """A `unsloth[foo]` pointing at an extra this branch does not define installs
        nothing and fails silently, which is how a partially ported extras block breaks."""
        extras = _load()["project"].get("optional-dependencies", {})
        dangling = []
        for name, deps in extras.items():
            for dep in deps:
                for match in _EXTRA_REF.finditer(dep):
                    for ref in match.group(1).split(","):
                        ref = ref.strip()
                        if ref and ref not in extras:
                            dangling.append((name, ref))
        assert dangling == [], f"extras referencing undefined extras: {dangling}"


class TestAmdExtraIsInstallableFromPyPI:
    """`pip install unsloth[amd]` is the supported AMD entry point, so the extra has to
    exist here and stay a version floor -- a URL pin would be unpublishable."""

    def test_amd_extra_exists_and_floors_bitsandbytes(self):
        extras = _load()["project"].get("optional-dependencies", {})
        assert "amd" in extras, "pyproject.toml must define an `amd` extra"
        specs = [d for d in extras["amd"] if d.lower().startswith("bitsandbytes")]
        assert specs, "the amd extra must constrain bitsandbytes"
        for spec in specs:
            assert spec.startswith("bitsandbytes>=0.50.0"), (
                f"bitsandbytes <= 0.49.2 NaNs at 4-bit decode on ROCm; got {spec!r}"
            )

    def test_amd_extra_pulls_the_torch_free_runtime(self):
        extras = _load()["project"].get("optional-dependencies", {})
        assert any("huggingfacenotorch" in d for d in extras["amd"])
        assert "huggingfacenotorch" in extras
