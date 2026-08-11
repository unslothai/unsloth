# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs the refactor guard in CI.

``tests/tools/refactor_guard.py`` pins the tool-call parsing / stripping stack three
ways: the AST and runtime surface of each module, a digest of every guarded function's
output over a 1,833-input corpus, and the string patch targets the test suite relies on.
See that module's docstring for what each check is for.

Re-baseline with ``python tests/tools/refactor_guard.py snapshot`` and review the diff
when a change to these modules is intended.
"""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "tests" / "tools"))

import refactor_guard  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return refactor_guard.build_corpus()


def test_ast_inventory_matches_the_baseline():
    """A dropped or re-signatured top-level name is an import break for some caller."""
    problems = refactor_guard._diff(
        "ast", refactor_guard._read("ast_inventory.json"), refactor_guard.ast_inventory()
    )

    assert not problems, "\n".join(problems[:40])


def test_runtime_surface_matches_the_baseline():
    """Catches re-export aliasing and decorator changes the AST cannot see."""
    problems = refactor_guard._diff(
        "runtime",
        refactor_guard._read("runtime_inventory.json"),
        refactor_guard.runtime_inventory(),
    )

    assert not problems, "\n".join(problems[:40])


def test_guarded_functions_produce_the_same_bytes(corpus):
    problems = refactor_guard._diff(
        "golden",
        refactor_guard._read("golden_outputs.json"),
        refactor_guard.golden_outputs(corpus),
    )

    assert not problems, "\n".join(problems[:40])


def test_no_new_non_idempotent_strip(corpus):
    """``strip(strip(x)) == strip(x)``, or display text depends on stream chunking.

    Two functions already fail this and are recorded in the baseline; the check is that
    the set does not grow.
    """
    baseline = {
        (entry["module"], entry["function"])
        for entry in refactor_guard._read("idempotence_baseline.json")
    }
    new = [
        f"{entry['module']}.{entry['function']} for {entry['input']!r}"
        for entry in refactor_guard.idempotence_failures(corpus)
        if (entry["module"], entry["function"]) not in baseline
    ]

    assert not new, "\n".join(new)


def test_every_string_patch_target_still_resolves():
    """A moved symbol leaves ``patch("mod.NAME")`` pointing at a namespace nobody reads.

    The test then passes while exercising unpatched code, which is the quiet failure this
    check exists to make loud.
    """
    broken = [
        entry
        for entry in refactor_guard.unresolvable_patch_targets()
        if not entry.get("environment")
    ]

    assert not broken, "\n".join(f"{e['target']}: {e['reason']} ({e['tests'][0]})" for e in broken)
