# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two halves of the external-probe channel, pinned so neither can rot on its own.

A probe needs a way IN (`SBENCH_EXTRA_INIT_SCRIPT`, concatenated after the scene scripts) and a way
OUT (`SBENCH_PAGE_CONSOLE`, a prefix filter on the page's console). Neither is exercised by a
normal run, because both are off unless the environment asks for them, which is exactly the
property that makes them safe and exactly the property that lets them break unnoticed: delete the
console filter and the probe still installs, still samples, and reports nothing, which reads as
"the arm did not fire".

The off-by-default half is the one worth stating. A hook that changed the page even slightly when
unset would make every scored run a probe run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STUDIOBENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDIOBENCH.parent))

PROBE = STUDIOBENCH / "arms" / "content_visibility_probe.js"
MAIN = STUDIOBENCH / "__main__.py"


@pytest.fixture(scope = "module")
def main_src() -> str:
    return MAIN.read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def build_src() -> str:
    return (STUDIOBENCH / "report" / "build.py").read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def probe_src() -> str:
    return PROBE.read_text(encoding = "utf-8")


def test_both_hooks_are_read_from_the_environment(main_src: str):
    for var in ("SBENCH_EXTRA_INIT_SCRIPT", "SBENCH_PAGE_CONSOLE"):
        assert f'os.environ.get("{var}")' in main_src, (
            f"{var} is no longer read by __main__; a probe that relies on it will install and "
            "report nothing, which is indistinguishable from an arm that did not fire"
        )


def test_the_hooks_are_off_unless_asked_for(main_src: str, monkeypatch):
    """Unset means NOTHING is appended and no listener is attached."""

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    monkeypatch.delenv("SBENCH_PAGE_CONSOLE", raising = False)
    import os

    assert os.environ.get("SBENCH_EXTRA_INIT_SCRIPT") is None
    assert os.environ.get("SBENCH_PAGE_CONSOLE") is None
    # Both call sites are guarded by a plain truthiness check on the variable, so an unset
    # variable cannot reach `add_init_script` or `page.on`. Pinned as source, because the failure
    # this guards is a refactor that hoists either call out of its `if`.
    for guarded in ("if extra_init:", "if console_prefix:"):
        assert guarded in main_src, f"{guarded!r} is gone; the hook may no longer be opt-in"


def test_the_probe_exists_and_is_not_a_stub(probe_src: str):
    assert PROBE.is_file()
    assert len(probe_src) > 4_000


def test_the_probe_prefix_is_the_one_the_console_filter_expects(probe_src: str):
    """The filter is an exact prefix match, so a probe with a different prefix is silent."""

    assert 'var PREFIX = "CVPOT ";' in probe_src
    assert "CVPOT " in (STUDIOBENCH / "CONTRIBUTING-perf.md").read_text(encoding = "utf-8"), (
        "the documented invocation and the probe's own prefix have to agree, or the recipe in "
        "CONTRIBUTING-perf.md produces an empty log and a false NOT RUN"
    )


def test_a_probe_run_records_the_gate_that_makes_it_unscorable(main_src: str):
    """The guarantee has to be a gate, not a sentence in a doc.

    A probe run otherwise looks entirely ordinary: same cells, same A/B table. `floor_table`
    refuses it on this field, and it can only do that if `__main__` writes it.
    """

    assert '"probe_init_script": extra_init or None' in main_src
    assert 'rec.gate(\n        "probe_free",' in main_src


def test_roots_are_adopted_at_insertion_not_on_the_sample_tick(probe_src: str):
    """The listener must exist before the element's FIRST transition.

    A root inserted off screen becomes skipped once and then never changes again. Attach on a
    two-second tick and every root mounted inside that tick emits its only event into a void,
    which is exactly the false NOT RUN this probe exists to prevent.
    """

    assert "MutationObserver" in probe_src
    assert "adoptAdded" in probe_src
    # The DOCUMENT, not documentElement. An init script can run before the parser has created
    # the root element, and `observe(null, ...)` throws into the catch that hides it.
    assert "observe(doc, {" in probe_src
    # Added nodes only. A document-wide re-scan per mutation would make the probe the load.
    assert "addedNodes" in probe_src
    assert "adoptAll();" in probe_src


def test_the_fallback_and_padding_buckets_cannot_both_count_one_root(probe_src: str):
    """They mean opposite things, and on this app they are close enough to collide.

    The user root's declared fallback is 60px and its padding is 40px. A single `height <= 64`
    bucket charged a root sitting exactly on its fallback to the zero-remembered-size trap as
    well. Fallback is tested first and wins ties.
    """

    assert "ROLE_PX" in probe_src
    assert "out.fallbackBite += 1;" in probe_src
    assert 'targetHeight(px, "padding")' in probe_src
    # getBoundingClientRect() is the border box, so BOTH targets carry the root's own padding.
    # Comparing against the bare declared length pinned fallbackBite at zero whatever happened.
    assert 'targetHeight(px, "fallback")' in probe_src
    assert 'return (which === "fallback" ? px.fallback : 0) + px.padding;' in probe_src


def test_only_a_skipped_root_can_land_in_a_size_bucket(probe_src: str):
    """`content-visibility: auto` computes to `auto` whether or not it is currently skipping.

    An armed root that is on screen has its ordinary rendered height, and size containment is not
    applying to it. Without the gate, a natural height near a role target reads as the trap.
    """

    assert "skippedState(el) === true" in probe_src


def test_detached_roots_stop_counting_as_skipped(probe_src: str):
    """`thread_reopen` rebuilds the thread, so watched roots leave the document.

    A detached root receives no further transitions, so one whose last event said `skipped` would
    keep inflating `skippedNow` for the rest of the session.
    """

    assert "droppedDetached" in probe_src
    assert "doc.contains(wel)" in probe_src


def test_the_page_scripts_are_installed_as_one_ordered_script(main_src: str):
    """Playwright does not define the order of separate init scripts.

    "The order of evaluation of multiple scripts installed via browserContext.addInitScript() and
    page.addInitScript() is not defined." surfaces.js reads what dom.js and parity.js put on
    `window.__sb`, and a probe may wrap either, so the sequence has to be a property of the string
    rather than of the scheduler.
    """

    assert "page_scripts = [" in main_src
    assert ".join(page_scripts))" in main_src
    # The probe is appended to that list, never registered on its own.
    assert "init_scripts.append(Path(extra_init)" not in main_src


def test_a_refused_run_does_not_leave_a_stale_ab_table(main_src: str):
    """`--resume` reuses the output directory, so `ab.md` may already exist from a clean run."""

    assert 'stale = paths.out / "ab.md"' in main_src
    assert "stale.write_text(" in main_src


def test_a_refused_report_does_not_leave_a_stale_summary(main_src: str):
    """`--report` writes `summary.md` beside the payload, and `--resume` reuses the directory.

    The refusal is a `SystemExit`, which is not an `Exception`, so it would leave the process
    before the write and an earlier clean summary would survive next to a probed payload.
    """

    assert "except SystemExit as exc:" in main_src
    assert 'out = path.parent / "summary.md"' in main_src
    assert "# No summary" in main_src


def test_the_report_refuses_before_it_assembles(build_src: str):
    """A refusal any other failure can pre-empt is not a refusal.

    `assemble_rows` validates the payload schema on the way past, so a probed payload that also
    tripped an unrelated schema complaint reported that instead and was never refused.
    """

    before = build_src.index("refuse_if_probed(_records(path)")
    assert before < build_src.index("payload = assemble_rows(path)")


def test_the_event_counter_is_the_one_potency_rests_on(probe_src: str):
    """`ev_skip` is the only potency signal here that no author CSS can fake."""

    assert "contentvisibilityautostatechange" in probe_src
    assert "ev_skip" in probe_src
    # The geometry route is a documented false negative and must stay documented rather than
    # quietly removed: someone will reimplement it otherwise.
    assert "KNOWN FALSE NEGATIVE" in probe_src
