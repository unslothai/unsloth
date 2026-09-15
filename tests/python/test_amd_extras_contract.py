# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contract tests for the amd / huggingfacenotorch extras.

security-audit.yml indexes [huggingfacenotorch] straight out of pyproject.toml. A release
branch shipping without it killed four security jobs for three weeks: a bare KeyError reads
as a generic crash, not as a missing extra.
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SECURITY_AUDIT = REPO_ROOT / ".github" / "workflows" / "security-audit.yml"

# 4-bit decode is unreliable on ROCm before bnb 0.50.0 (bnb #1887, #1979, #2012).
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

    All three shapes must be read or the check is vacuous: the literal index, the guarded
    helper call, and the shell list the per-extra loop iterates over.
    """
    source = SECURITY_AUDIT.read_text(encoding = "utf-8")
    names = set()
    for block in _inline_python_blocks():
        for node in ast.walk(ast.parse(block)):
            if _is_optional_dependencies_lookup(node) and isinstance(node.slice, ast.Constant):
                names.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "extra" and node.args and isinstance(node.args[0], ast.Constant)):
                names.add(node.args[0].value)
    for listed in re.findall(r"^\s*for extra in ([^;]+); do\s*$", source, re.MULTILINE):
        names |= {word for word in listed.split() if word}
    return names


def _is_optional_dependencies_lookup(node: ast.AST) -> bool:
    """`<expr>["optional-dependencies"][<key>]`, whatever the key's quote style."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.slice, ast.Constant)
        and node.value.slice.value == "optional-dependencies"
    )


def _contains_node(root: ast.AST, target: ast.AST) -> bool:
    return any(n is target for n in ast.walk(root))


def _catches_key_error(handler: ast.ExceptHandler) -> bool:
    """A bare `except`, or one naming KeyError or a superclass of it."""
    names = {"KeyError", "LookupError", "Exception", "BaseException"}
    t = handler.type
    if t is None:
        return True
    parts = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(p, ast.Name) and p.id in names for p in parts)


def _inline_python_blocks() -> list[str]:
    """Every `python ... <<PY` or `<<'PY'` heredoc in security-audit.yml, dedented."""
    source = SECURITY_AUDIT.read_text(encoding = "utf-8")
    blocks = re.findall(
        r"^[ \t]*python[^\n]*<<'?PY'?[^\n]*\n(.*?)^[ \t]*PY[ \t]*$", source, re.MULTILINE | re.DOTALL
    )
    assert blocks, "expected security-audit.yml to embed python heredocs"
    return [textwrap.dedent(b) for b in blocks]


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
            # The whole broken range, not one release: `>=0.49.3` or `!=0.49.2` must fail too.
            floors = [Version(sp.version) for sp in allowed if sp.operator in (">=", ">", "==", "~=")]
            assert floors and min(floors) >= BNB_MIN, f"{requirement} has no lower bound at or above {BNB_MIN}"
            for old in ("0.45.0", "0.49.2", "0.49.3", "0.49.99"):
                assert not allowed.contains(Version(old)), f"{requirement} still admits bnb {old}"
            assert allowed.contains(BNB_MIN), f"{requirement} excludes the fixed release {BNB_MIN}"


class TestSecurityAuditWorkflowStaysInSync:
    """Every extra the audit workflow indexes has to actually exist.

    The workflow reaches into pyproject.toml by name, so a rename or omission takes out the
    scan jobs rather than the branch that caused it.
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

        `test_indexed_extras_exist` passes just as well when the workflow names nothing.
        """
        assert known in _extras_referenced_by_the_audit_workflow()

    def test_every_lookup_is_guarded(self):
        """A bare index is the failure mode this file exists for.

        Parse every inline Python block and require each optional-dependencies subscript
        to sit inside a try/except, so deleting a guard fails here rather than in CI.
        """
        bare = []
        for block in _inline_python_blocks():
            tree = ast.parse(block)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node  # type: ignore[attr-defined]
            for node in ast.walk(tree):
                if not _is_optional_dependencies_lookup(node):
                    continue
                # Guarded means inside the try BODY (not else/finally) of a try whose handler catches KeyError.
                guarded, child, parent = False, node, getattr(node, "parent", None)
                while parent is not None:
                    if isinstance(parent, ast.Try) and any(_contains_node(stmt, child) for stmt in parent.body):
                        if any(_catches_key_error(h) for h in parent.handlers):
                            guarded = True
                        break
                    child, parent = parent, getattr(parent, "parent", None)
                if not guarded:
                    bare.append(ast.get_source_segment(block, node))
        assert not bare, f"unguarded optional-dependencies lookups in security-audit.yml: {bare}"
