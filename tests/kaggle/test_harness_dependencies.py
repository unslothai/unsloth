# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The CI step that runs these tests must be able to import what they import.

The failure this exists against is quiet by construction. `Test the harness`
runs `pytest tests/kaggle -q` against a hand-written pip line, and a module
missing from that line does not stop the step: pytest reports the collection or
call error, the step goes red with everything else green, and the two tests that
could not import are indistinguishable in a summary from tests that never
existed. Measured on run 32925226213, which ran **803 of 805** because
`datasets` was not installed -- and the two it lost are the pair proving
`with_transform` keeps PIL images while `Dataset.from_list` corrupts them, which
is the finding the vision leg was built on.

`pillow` and `pyyaml` were in the same position and passing only because
`transformers` and friends happened to pull them. A transitive dependency is not
a dependency; it is a coincidence that holds until an upstream drops it.

So the rule is an agreement between two things that are edited months apart: the
modules `tests/kaggle/test_*.py` import without a `pytest.importorskip` guard,
and the distributions the workflow installs.

Scope is deliberately the TEST modules, not the payloads under `t4_smoke/` and
`studio_gpu/`. Those import trl, unsloth, unsloth_zoo and vllm, and they do it
lazily inside functions because they run on a Kaggle GPU session rather than on
this runner. Demanding them here would install a CUDA stack on an
`ubuntu-latest` box to run a pure-AST test suite.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests" / "kaggle"
WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"

# Import name -> the distribution that provides it, for the ones that differ.
# Anything not listed is assumed to install under its own name.
DISTRIBUTION = {
    "PIL": "pillow",
    "yaml": "pyyaml",
}

# Directories whose module names are importable because the suite puts them on
# sys.path, so they are not third-party however they are spelled.
LOCAL_DIRS = (
    TESTS,
    TESTS / "t4_smoke",
    TESTS / "studio_gpu",
    ROOT / ".github" / "scripts",
    ROOT / ".github" / "scripts" / "kaggle_t4_ci",
    ROOT / ".github" / "scripts" / "kaggle_studio_ci",
)


def _local_names() -> set[str]:
    names = set()
    for directory in LOCAL_DIRS:
        names.add(directory.name)
        for path in directory.glob("*.py"):
            names.add(path.stem)
    return names


def _imported_third_party() -> dict[str, set[str]]:
    """Every third-party module the test modules import, and where.

    Any depth, not just module level: a `from datasets import Dataset` inside a
    test function fails when that test RUNS, which is exactly the case run
    32925226213 hit, and a module-level-only scan would have missed it.
    """
    local = _local_names()
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        # `pytest.importorskip("x")` is the suite's own way of saying a module
        # is optional, so it is not a claim on the install line.
        guarded = {
            node.args[0].value.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "importorskip"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in sys.stdlib_module_names or name in local or name in guarded:
                    continue
                found.setdefault(name, set()).add(path.name)
    return found


def _install_line() -> str:
    text = WORKFLOW.read_text(encoding = "utf-8")
    step = text.index("- name: Test the harness")
    body = text[step : text.index("\n      - name:", step + 1)]
    matches = re.findall(r"pip install[^\n]*", body)
    assert matches, "the Test the harness step no longer installs anything"
    return " ".join(matches)


def test_every_unguarded_import_is_installed_by_the_workflow():
    """The whole point. A module missing here costs tests silently."""
    line = _install_line()
    missing = {
        module: sorted(files)
        for module, files in _imported_third_party().items()
        if DISTRIBUTION.get(module, module) not in line
    }
    assert not missing, (
        "these modules are imported by tests/kaggle/test_*.py without a "
        "pytest.importorskip guard and are not installed by the Test the "
        f"harness step, so those tests will error rather than run: {missing}"
    )


def test_the_import_scan_finds_something_at_all():
    """A scan that silently matched nothing would satisfy the rule above
    forever. Two modules the suite genuinely imports are named here, so a
    refactor that breaks the walk fails rather than passes."""
    found = _imported_third_party()
    assert "datasets" in found, "the walk no longer sees the vision dataset imports"
    assert "torch" in found
    assert found["datasets"], "no file recorded for datasets"


def test_the_scan_looks_inside_functions_and_not_only_at_module_level():
    """The measured failure was a function-level import. `test_vision_run.py`
    imports `datasets` inside two test bodies and nowhere else, so a
    module-level-only scan reports a clean tree and the step still loses two
    tests."""
    source = (TESTS / "test_vision_run.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    module_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "datasets" not in module_level, (
        "datasets is now imported at module level in test_vision_run.py, which "
        "makes this test tautological; point it at another function-level "
        "import instead of deleting it"
    )
    assert "datasets" in _imported_third_party()


def test_a_guarded_import_is_not_treated_as_a_requirement():
    """`pytest.importorskip` is how this suite says a module is optional, and
    demanding those on the runner would put a CUDA stack on an ubuntu box."""
    line = _install_line()
    assert "vllm" not in line
    assert "unsloth" not in line
