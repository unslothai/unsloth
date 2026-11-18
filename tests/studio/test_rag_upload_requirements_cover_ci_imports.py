# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""rag_upload_ci.py runs every studio/backend/tests/test_rag_*.py in a venv built only from
rag_upload_requirements.txt. A RAG test that imports a module unconditionally under CI (treating a
missing wheel as a packaging failure) needs that module pinned there, at studio.txt's version."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_REQS = ROOT / "tests/studio/rag_upload_requirements.txt"
STUDIO_REQS = ROOT / "studio/backend/requirements/studio.txt"


def _pins(path):
    pins = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


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
    upload, studio = _pins(UPLOAD_REQS), _pins(STUDIO_REQS)
    missing, drifted = [], []
    for test, modules in _ci_imports_by_test().items():
        for module in sorted(modules):
            name = re.sub(r"[-_.]+", "-", module).lower()
            if name not in upload:
                missing.append(f"{test}: {module}")
            elif name in studio and studio[name] != upload[name]:
                drifted.append(f"{module}: upload {upload[name]} vs studio.txt {studio[name]}")
    assert not missing, f"not in {UPLOAD_REQS.name}: {missing}"
    assert not drifted, f"pin differs from studio.txt: {drifted}"


# Import names whose wheel is published under another name.
_DISTRIBUTIONS = {
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "docx": "python-docx",
    "fitz": "pymupdf",
    "yaml": "pyyaml",
}
_LOCAL_ROOTS = (ROOT / "studio/backend", ROOT / "studio/backend/tests", ROOT / "tests/_shared")


def _harness_tests():
    # What rag_upload_ci.py hands its pytest run.
    tests = sorted((ROOT / "studio/backend/tests").glob("test_rag_*.py"))
    return [*tests, ROOT / "studio/backend/tests/test_conversation_archive.py"]


def _is_local(module):
    return any((r / module).is_dir() or (r / f"{module}.py").is_file() for r in _LOCAL_ROOTS)


def _optional_import_nodes(tree):
    """Imports inside `try: ... except ImportError` (or ModuleNotFoundError) are allowed to be absent."""
    optional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError")
            for h in node.handlers
        ):
            for stmt in node.body:
                optional.update(id(n) for n in ast.walk(stmt))
    return optional


def _third_party_imports(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    optional = _optional_import_nodes(tree)
    found = set()
    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top not in sys.stdlib_module_names and not _is_local(top):
                found.add(top)
    return found


def test_the_import_scan_reads_function_local_imports():
    # Control: test_rag_extract_text.py builds its upload fixtures with PIL inside a helper.
    assert "PIL" in _third_party_imports(ROOT / "studio/backend/tests/test_rag_extract_text.py")


def test_every_rag_harness_import_is_installed_by_the_upload_requirements():
    """Any import a harness test makes, at module level or inside a function, needs its wheel in
    the venv: #18293 built PIL images at collection time and the whole file failed to collect."""
    upload = _pins(UPLOAD_REQS)
    missing = sorted(
        f"{test.name}: {module}"
        for test in _harness_tests()
        for module in _third_party_imports(test)
        if re.sub(r"[-_.]+", "-", _DISTRIBUTIONS.get(module, module)).lower() not in upload
    )
    assert not missing, f"not in {UPLOAD_REQS.name}: {missing}"
