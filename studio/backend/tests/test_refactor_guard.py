# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs the refactor guard in CI.

``tests/tools/refactor_guard.py`` pins the tool-call parsing / stripping stack three ways:
module AST and runtime surface, a digest of every guarded function's output over a
1,833-input corpus, and the test suite's string patch targets. Re-baseline with
``python tests/tools/refactor_guard.py snapshot`` and review the diff.
"""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "tests" / "tools"))

import refactor_guard  # noqa: E402


@pytest.fixture(scope = "module")
def corpus():
    return refactor_guard.build_corpus()


def test_ast_inventory_matches_the_baseline():
    """A dropped or re-signatured top-level name is an import break for some caller."""
    # Same helper ``verify`` uses, so CI and the CLI agree.
    problems = refactor_guard._ast_problems()

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

    Known failures are recorded in the baseline, one entry per boolean variant; the check
    is that the set does not grow. Keyed by variant, so a ``final = True`` failure is no
    licence for the ``final = False`` streaming path to start.
    """
    baseline = {
        (entry["module"], entry["function"], entry.get("variant", ""))
        for entry in refactor_guard._read("idempotence_baseline.json")
    }
    new = [
        f"{entry['module']}.{entry['function']}[{entry['variant']}] for {entry['input']!r}"
        for entry in refactor_guard.idempotence_failures(corpus)
        if (entry["module"], entry["function"], entry["variant"]) not in baseline
    ]

    assert not new, "\n".join(new)


def test_every_string_patch_target_still_resolves():
    """A moved symbol leaves ``patch("mod.NAME")`` pointing at a namespace nobody reads,
    so the test passes while exercising unpatched code."""
    live = refactor_guard.patch_targets()
    broken = [
        entry
        for entry in refactor_guard.unresolvable_patch_targets(live)
        if not entry.get("environment")
    ]

    assert not broken, "\n".join(f"{e['target']}: {e['reason']} ({e['tests'][0]})" for e in broken)


def test_the_recorded_patch_target_inventory_still_matches():
    """Resolving is not enough: the recorded routing has to be the live routing.

    A patch repointed at another resolvable namespace, or dropped, resolves fine. CI runs
    pytest and never the ``verify`` CLI, so the comparison has to live here.
    """
    recorded = {
        target: sorted(tests)
        for target, tests in refactor_guard._read("patch_targets.json").items()
    }
    live = {target: sorted(tests) for target, tests in refactor_guard.patch_targets().items()}

    problems = refactor_guard._diff("patch-targets", recorded, live, additions_matter = False)

    assert not problems, "\n".join(problems[:20])


def test_a_deleted_patch_target_is_not_written_off_as_environmental():
    """The ``environment`` escape hatch must not swallow a genuinely dead target.

    ``importlib.import_module("mod.attr")`` raises ``ModuleNotFoundError`` for a deleted
    attribute just as it does for a missing optional dependency, and the check above
    filters environmental entries out, so conflating the two would make it vacuous.
    """
    broken = refactor_guard.unresolvable_patch_targets(
        {
            "core.tool_healing.deleted_name": ["fake_test.py"],
            "core.inference.deleted_name": ["fake_test.py"],
        }
    )

    assert {entry["target"] for entry in broken} == {
        "core.tool_healing.deleted_name",
        "core.inference.deleted_name",
    }
    assert not any(entry.get("environment") for entry in broken)


def test_no_guarded_function_is_driven_by_a_sentinel():
    """Every guarded function has to be actually called, or its golden digest pins
    nothing and an arbitrary rewrite of it passes."""
    # The whole corpus, not a slice: several functions are constant over any small prefix
    # and only vary once the rarer serializations appear, so a slice reports false gaps.
    corpus = refactor_guard.build_corpus()
    undrivable = sorted(
        name
        for module in refactor_guard.BEHAVIOUR_MODULES
        for name, func in refactor_guard._guarded_functions(module)
        if refactor_guard._drive(func, corpus[0]) == "<undrivable>"
    )

    assert not undrivable, f"no argument fixture for: {undrivable}"

    constant = sorted(
        name
        for module in refactor_guard.BEHAVIOUR_MODULES
        for name, func in refactor_guard._guarded_functions(module)
        if len({repr(refactor_guard._drive(func, text)) for text in corpus}) == 1
    )

    assert len(constant) <= 4, f"too many functions pin a single value: {constant}"


def test_a_dropped_lazy_export_is_not_written_off_as_environmental():
    """``core.inference`` resolves through a PEP 562 ``__getattr__``: a missing optional
    dependency surfaces as ImportError, a dropped export as AttributeError, and only the
    first is an environment gap."""
    broken = refactor_guard.unresolvable_patch_targets(
        {"core.inference.no_such_lazy_export": ["fake_test.py"]}
    )

    assert [entry["target"] for entry in broken] == ["core.inference.no_such_lazy_export"]
    assert not any(entry.get("environment") for entry in broken)


def test_the_scan_order_inside_strip_segment_is_pinned():
    """The arm order in ``strip_segment`` is what this branch unified.

    Swapping the function-XML and GLM arms passed the guard: the corpus only concatenated
    whole calls, and the order shows up only when an arm can eat a later arm's opener.
    """
    from core.inference.tool_call_parser import strip_segment

    text = "<function=x><tool_call> txt <arg_key>c</arg_key></function><parameter=p>"

    assert strip_segment(text, seg_final = True, enabled_tool_names = {"x"}) == "<parameter=p>"


def test_strip_tool_markup_is_pinned_at_both_final_values():
    """``final = False`` is the streaming path, and it was driven at one value only."""
    from core.inference.tool_call_parser import strip_tool_markup

    text = (
        '<function=get_weather><parameter=q><tool_call>{"a":1}</tool_call>'
        "</parameter></function> tail"
    )
    names = {"get_weather"}

    assert strip_tool_markup(text, final = False, enabled_tool_names = names) == " tail"
    assert strip_tool_markup(text, final = True, enabled_tool_names = names) == "tail"
    # trim = False keeps the boundary the removed span exposed, which the nudge replay
    # merges onto a resumed partial.
    untrimmed = strip_tool_markup(text, final = True, trim = False, enabled_tool_names = names)
    assert untrimmed == " tail"


def test_an_unrelated_addition_elsewhere_does_not_fail_the_guard():
    """Additions are not regressions, and a guard that cries on them gets re-snapshotted
    blind. A dropped symbol still has to fail."""
    import copy

    base = refactor_guard.ast_inventory()
    wide = "core.inference.llama_cpp"

    added = copy.deepcopy(base[wide])
    added["symbols"]["_SOMETHING_A_LATER_PR_ADDS"] = {"kind": "assign"}
    assert not refactor_guard._diff("ast", base[wide], added, additions_matter = False)

    dropped = copy.deepcopy(base[wide])
    dropped["symbols"].pop(next(iter(dropped["symbols"])))
    assert refactor_guard._diff("ast", base[wide], dropped, additions_matter = False)
