# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""rag_upload_ci.py runs every studio/backend/tests/test_rag_*.py in a venv built only from
rag_upload_requirements.txt. A RAG test that imports a module unconditionally under CI (treating a
missing wheel as a packaging failure) needs that module pinned there, at studio.txt's version."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_REQS = ROOT / "tests/studio/rag_upload_requirements.txt"
STUDIO_REQS = ROOT / "studio/backend/requirements/studio.txt"
PYPROJECT = ROOT / "pyproject.toml"


def _pins(path):
    pins = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def _hnsw_extra_pins():
    text = PYPROJECT.read_text(encoding = "utf-8")
    block = re.search(r"^hnsw = \[(.*?)^\]", text, re.M | re.S).group(1)
    return {
        re.sub(r"[-_.]+", "-", m.group(1)).lower(): m.group(2)
        for m in re.finditer(r'"([A-Za-z0-9_.\-]+)==([^\s;"]+)', block)
    }


def _is_ci_check(node):
    return any(isinstance(n, ast.Constant) and n.value == "CI" for n in ast.walk(node))


def _ci_imports(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.If) and _is_ci_check(node.test):
            for stmt in node.body:
                if isinstance(stmt, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in stmt.names)
    return found


def _ci_imports_by_test():
    tests = sorted((ROOT / "studio/backend/tests").glob("test_rag_*.py"))
    return {t.name: _ci_imports(t) for t in tests}


def test_the_scan_finds_the_hnsw_tier_imports():
    # Control: the reader sees the CI-only import the ANN tests make today.
    found = _ci_imports_by_test()
    assert "usearch" in found.get("test_rag_vec_ann.py", set())


def test_every_ci_only_rag_import_is_pinned_for_the_upload_harness():
    upload, studio = _pins(UPLOAD_REQS), {**_hnsw_extra_pins(), **_pins(STUDIO_REQS)}
    missing, drifted = [], []
    for test, modules in _ci_imports_by_test().items():
        for module in sorted(modules):
            name = re.sub(r"[-_.]+", "-", module).lower()
            if name not in upload:
                missing.append(f"{test}: {module}")
            elif name in studio and studio[name] != upload[name]:
                drifted.append(f"{module}: upload {upload[name]} vs studio {studio[name]}")
    assert not missing, f"not in {UPLOAD_REQS.name}: {missing}"
    assert not drifted, f"pin differs from studio.txt / the hnsw extra: {drifted}"


def test_the_hnsw_extra_pins_are_read():
    assert _hnsw_extra_pins() == {"usearch": "2.26.6", "numkong": "7.8.6"}
