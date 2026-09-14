# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contract tests for the amd / huggingfacenotorch extras.

security-audit.yml builds its hf-stack scan set by indexing
[huggingfacenotorch] straight out of pyproject.toml. When the pip release
branch shipped without that extra, four security jobs died on a bare
KeyError for three weeks and nobody noticed, because the failure looked
like a generic Python crash rather than a missing extra.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SECURITY_AUDIT = REPO_ROOT / ".github" / "workflows" / "security-audit.yml"

# 4-bit decode is unreliable on ROCm before 0.50.0, the first PyPI release
# carrying the full path (bnb #1887, #1979, #2012).
BNB_MIN = Version("0.50.0")


def _extras() -> dict[str, list[str]]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))
    return data["project"]["optional-dependencies"]


def _extras_referenced_by_the_audit_workflow() -> set[str]:
    """Every extra name security-audit.yml reaches into pyproject.toml for.

    Three shapes, and all three have to be read or the check is vacuous: the
    literal index `optional-dependencies"]["name"]`, the guarded helper call
    `extra("name")`, and the shell list the per-extra loop iterates, whose names
    only ever appear in the `for extra in ...` line.
    """
    source = SECURITY_AUDIT.read_text(encoding = "utf-8")
    names = set(re.findall(r'optional-dependencies"\]\["([^"]+)"\]', source))
    names |= set(re.findall(r'\bextra\("([^"]+)"\)', source))
    for listed in re.findall(r"^\s*for extra in ([^;]+); do\s*$", source, re.MULTILINE):
        names |= {word for word in listed.split() if word}
    return names


def _project_name(spec: str) -> str:
    """Leading distribution name of a PEP 508 requirement, lowercased."""
    return re.split(r"[<>=!~;\[\s@]", spec.strip(), maxsplit = 1)[0].strip().lower()


class TestExtrasExist:
    """Both extras must be present, on every branch."""

    @pytest.mark.parametrize("name", ["huggingfacenotorch", "amd"])
    def test_extra_present(self, name: str):
        assert name in _extras(), f"pyproject.toml is missing the [{name}] extra"

    def test_amd_pulls_the_no_torch_stack(self):
        assert any(
            s.replace(" ", "") == "unsloth[huggingfacenotorch]" for s in _extras()["amd"]
        ), "the amd extra must pull unsloth[huggingfacenotorch]"


class TestHuggingfaceNoTorchIsTorchFree:
    """The whole point of the extra is that it names no torch distribution."""

    @pytest.mark.parametrize("banned", ["torch", "torchvision"])
    def test_no_torch_distribution(self, banned: str):
        named = [s for s in _extras()["huggingfacenotorch"] if _project_name(s) == banned]
        assert not named, f"[huggingfacenotorch] must not name {banned}: {named}"


class TestAmdBitsandbytesFloor:
    """Keeps the pre-0.50.0 ROCm range out of the AMD install path."""

    def test_every_marker_line_excludes_the_broken_range(self):
        specs = [s for s in _extras()["amd"] if _project_name(s) == "bitsandbytes"]
        assert specs, "the amd extra must pin bitsandbytes"
        for spec in specs:
            requirement = spec.split(";", 1)[0].strip()
            allowed = SpecifierSet(requirement[len("bitsandbytes") :].strip())
            assert not allowed.contains(
                Version("0.49.2")
            ), f"{requirement} still admits bnb 0.49.2, which predates the ROCm 4-bit fixes"
            assert allowed.contains(BNB_MIN), f"{requirement} excludes the fixed release {BNB_MIN}"


class TestSecurityAuditWorkflowStaysInSync:
    """Every extra the audit workflow indexes has to actually exist.

    This is the generic form of the failure: the workflow reaches into
    pyproject.toml by name, so any rename or omission takes out the scan
    jobs rather than the branch that caused it.
    """

    def test_indexed_extras_exist(self):
        referenced = _extras_referenced_by_the_audit_workflow()
        assert referenced, "expected security-audit.yml to index at least one extra"
        missing = sorted(referenced - set(_extras()))
        assert (
            not missing
        ), f"security-audit.yml indexes extras that pyproject.toml lacks: {missing}"

    @pytest.mark.parametrize("known", ["huggingfacenotorch", "audio-torch211"])
    def test_the_extras_the_scan_set_is_built_from_are_still_named(self, known: str):
        """The scan set loses coverage silently if one of these stops being read.

        `test_indexed_extras_exist` only checks that what IS named exists; it
        passes just as well when the workflow names nothing at all, which is how
        a refactor can quietly drop an extra out of the audit input.
        """
        assert known in _extras_referenced_by_the_audit_workflow()

    def test_every_lookup_is_guarded(self):
        """A bare index is the failure mode this file exists for.

        An `optional-dependencies` lookup outside the `extra()` helper or the
        `try` that wraps the per-extra loop raises a bare KeyError, which reads
        as a generic crash rather than as a missing extra. Allowlist the two
        guarded call sites by their exact line so a third one has to be looked at.
        """
        guarded = {
            'return d["project"]["optional-dependencies"][name]',
            'specs = d["project"]["optional-dependencies"][extra]',
        }
        bare = [
            line.strip()
            for line in SECURITY_AUDIT.read_text(encoding = "utf-8").splitlines()
            if 'optional-dependencies"][' in line
            and not line.strip().startswith("#")
            and line.strip() not in guarded
        ]
        assert not bare, f"unguarded optional-dependencies lookups in security-audit.yml: {bare}"
