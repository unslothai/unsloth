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
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


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
        """Ask packaging, not the spelling.

        PEP 508 allows any whitespace around the `@`, and the scheme is case
        insensitive, so `flash-attn@https://...` and `flash-attn @ HTTPS://...` are both
        valid direct references that a substring match on " @ https://" would wave
        through - the exact upload failure this guards. `Requirement.url` is set for
        every form of them.
        """
        packaging_requirements = pytest.importorskip("packaging.requirements")
        offenders = []
        for extra, req in _all_requirements():
            try:
                parsed = packaging_requirements.Requirement(req)
            except Exception:  # noqa: BLE001 - test_every_requirement_parses reports it
                continue
            if parsed.url:
                offenders.append((extra, req))
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
        nothing and fails silently, which is how a partially ported extras block breaks.

        Parsed and canonicalized rather than pattern-matched: project names and extra
        names are both case and separator insensitive (PEP 503, PEP 685), so pip honours
        `Unsloth[Rocm72_Torch2100]` while a lowercase regex would never look at it.
        """
        packaging_requirements = pytest.importorskip("packaging.requirements")
        packaging_utils = pytest.importorskip("packaging.utils")
        canonicalize = packaging_utils.canonicalize_name

        extras = _load()["project"].get("optional-dependencies", {})
        defined = {canonicalize(name) for name in extras}
        dangling = []
        for name, deps in extras.items():
            for dep in deps:
                try:
                    parsed = packaging_requirements.Requirement(dep)
                except Exception:  # noqa: BLE001 - test_every_requirement_parses reports it
                    continue
                if canonicalize(parsed.name) != "unsloth":
                    continue
                for ref in parsed.extras:
                    if canonicalize(ref) not in defined:
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
            assert spec.startswith(
                "bitsandbytes>=0.50.0"
            ), f"bitsandbytes <= 0.49.2 NaNs at 4-bit decode on ROCm; got {spec!r}"

    def test_amd_extra_pulls_the_torch_free_runtime(self):
        extras = _load()["project"].get("optional-dependencies", {})
        assert any("huggingfacenotorch" in d for d in extras["amd"])
        assert "huggingfacenotorch" in extras


class TestRuntimeImportsAreDeclared:
    """This branch's base install has to satisfy every module-scope import on the CLI
    entry path. typer supplied click until 0.26.0 dropped it (#7504); it still supplies
    rich, so rich is satisfied only by chance unless we declare it ourselves."""

    ENTRY_PATH_IMPORTS = ("click", "rich", "structlog", "typer")

    def test_cli_entry_path_imports_are_base_dependencies(self):
        packaging_requirements = pytest.importorskip("packaging.requirements")
        base = _load()["project"].get("dependencies", [])
        declared = {
            packaging_requirements.Requirement(r).name.lower().replace("_", "-") for r in base
        }
        missing = [p for p in self.ENTRY_PATH_IMPORTS if p not in declared]
        assert missing == [], (
            "imported at module scope by unsloth_cli/__init__.py's import chain but not "
            f"declared in base dependencies: {missing}"
        )


class TestAcceleratorExtrasCarryTheirCompanions:
    """Each accelerator extra composes a stack: `-ampere-` variants add flash-attn and
    torch 2.10 variants add the torchcodec audio path. A variant that silently drops one
    installs a quietly weaker environment than its siblings."""

    def test_torch2100_extras_pull_the_audio_path(self):
        extras = _load()["project"].get("optional-dependencies", {})
        targets = [
            n for n in extras if n.endswith("torch2100") and "only" not in n and n.startswith("cu")
        ]
        assert targets, "expected cu*-torch2100 extras to exist"
        missing = [n for n in targets if not any("audio-torch" in d for d in extras[n])]
        assert missing == [], f"torch 2.10 extras missing the audio extra: {missing}"

    def test_ampere_torch280_extras_pull_flash_attention(self):
        extras = _load()["project"].get("optional-dependencies", {})
        targets = [n for n in extras if n.endswith("-ampere-torch280")]
        assert targets, "expected *-ampere-torch280 extras to exist"
        missing = [
            n
            for n in targets
            if not any("flashattention" in d or "flash-attn" in d for d in extras[n])
        ]
        assert missing == [], f"ampere torch 2.8 extras missing flash-attn: {missing}"
