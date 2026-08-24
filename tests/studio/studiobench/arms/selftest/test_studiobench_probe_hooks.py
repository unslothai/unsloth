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

import re
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


def test_the_probe_path_is_validated_before_anything_is_started(main_src: str):
    """A path typo must not leave a detached Studio holding a port.

    The source is not needed until the browser launches, but reading it there raises after Studio
    and the pacer are up and before the cleanup `finally` around the cell loop is entered, so
    nothing stops them. Reading it in the first second of the run fails while there is nothing to
    clean up.
    """

    read_at = main_src.index("extra_init_source = Path(extra_init).read_text")
    assert read_at < main_src.index("install_studio(ref, home)")
    assert read_at < main_src.index("_probe_init_scripts(extra_init, extra_init_source)")
    assert "except (OSError, UnicodeDecodeError) as exc:" in main_src
    # Read once, used later. A second read at launch time would reintroduce the window.
    assert main_src.count("Path(extra_init).read_text") == 1
    # AND ahead of the archive, which is the other thing a refusal must not arrive after. Reusing
    # an --out without --resume moves the payload aside, so reading the probe after that call let a
    # path typo take payload.jsonl off its standard path while running no benchmark at all. The
    # behaviour is pinned in runtime/selftest/test_studiobench_run_acquisition.py; the order is
    # pinned here because that is what makes it true.
    assert read_at < main_src.index("prepare_payload(")


def test_the_probe_is_installed_without_eval(main_src: str):
    """CSP, not style. Studio serves `script-src 'self'` with no `'unsafe-eval'`, and the DEFAULT
    engine on Linux and macOS is webkit, which enforces that against an init script.

    The probe was briefly handed to indirect eval as a string so that a malformed file could not
    stop the scene scripts. Measured against a page carrying Studio's own header, with the real
    `content_visibility_probe.js`: chromium and firefox installed the probe either way, and webkit
    refused the eval with `EvalError` and installed NOTHING. The isolation was not real there
    either -- Playwright gives webkit its init scripts as one bootstrap unit, so a parse error
    kills them all however the probe is written -- and on the two engines where it is real,
    separate `add_init_script` calls already provide it. So the source goes in as source.
    """

    assert "def _probe_init_scripts(" in main_src
    assert "init_scripts.extend(_probe_init_scripts(extra_init, extra_init_source))" in main_src
    assert "(0, eval)(" not in main_src, (
        "the probe is back on eval; Studio's script-src 'self' blocks it and the probe will not "
        "install at all on the default engine"
    )
    assert "eval(" not in main_src, "any string evaluation is refused by Studio's CSP"


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
    # Matched without pinning the indent. `run()` puts side acquisition under a cleanup guard, so
    # everything from the acquisition loop to the end of setup sits one level deeper than it did,
    # and the gate moving with it is not the thing this test is here to notice.
    assert re.search(r"rec\.gate\(\s*\n\s*\"probe_free\",", main_src)


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


def test_each_scene_script_keeps_its_own_failure_domain(main_src: str):
    """Separate `add_init_script` calls, which is what this always did.

    They were briefly joined into one script to pin the evaluation order, which Playwright does
    not define. That was a regression: the browser evaluates a joined script as one unit, so a
    throw in any one of the three stops the other two, and on the CI fixture it cost
    `message_menu` its More button. Fault isolation beats an ordering guarantee for a sequence
    that has been correct in practice for the life of the file.
    """

    for name in ("scene/dom.js", "scene/parity.js", "scene/surfaces.js"):
        assert f'init_scripts.append(resources.read_text("{name}"))' in main_src
    assert "page_scripts" not in main_src


def test_the_probe_is_its_own_script_and_says_when_it_did_not_install(main_src: str):
    """Order is undefined, so a probe has to be self-contained; that is the documented rule.

    A probe that installed nothing must not read as an arm that did not fire, so both failure
    modes have a channel. A probe that did not PARSE leaves `window.__sbExtraInitScript` unset and
    the second script names it on the console; a probe that parsed and then THREW arrives as a
    `pageerror`. Both listeners are attached whenever a probe is asked for.
    """

    assert "init_scripts.extend(_probe_init_scripts(extra_init, extra_init_source))" in main_src
    assert "window.__sbExtraInitScript" in main_src
    assert "never installed: it did not " in main_src
    assert 'bundle.page.on("pageerror"' in main_src


def test_the_probe_source_is_the_first_thing_in_its_script():
    """REGRESSION. A directive prologue is only a prologue while nothing precedes it.

    ECMA-262 defines a Directive Prologue as the run of expression statements a Script or
    FunctionBody OPENS with, so a probe beginning `"use strict"` stops being strict the moment a
    statement is put in front of it: the directive degrades into a string expression that does
    nothing, and an undeclared assignment inside the probe creates a global instead of throwing.
    The installation stamp used to be that statement. It goes last instead, which changes nothing
    about the probe and still leaves the stamp unset when the file did not run.

    Playwright wraps each init script in `(() => { ... })();` (`playwright-core`, `class
    InitScript`), so the file's own prologue is a FunctionBody prologue -- real, and equally
    destroyed by a prepend.
    """

    import studiobench.__main__ as sb

    source = '"use strict";\nvar ticks = 0;\n'
    scripts = sb._probe_init_scripts("probes/p.js", source)

    assert scripts[0].startswith(source), (
        "something is being prepended to the probe source; a leading directive such as "
        '"use strict" is no longer in the directive prologue and the probe runs with different '
        "semantics than the file it was read from"
    )
    assert "window.__sbExtraInitScript" in scripts[0]
    # Appended after an explicit statement boundary, so neither ASI nor an unterminated
    # expression can join it to the probe's last line.
    assert (
        scripts[0][len(source) :]
        .lstrip("\n")
        .lstrip(";")
        .lstrip("\n")
        .startswith("window.__sbExtraInitScript")
    )


def _seed(tmp_path: Path, text: str = "old report") -> Path:
    for name in ("summary.md", "ab.md"):
        (tmp_path / name).write_text(text, encoding = "utf-8")
    return tmp_path


def test_a_probe_run_invalidates_the_reports_it_inherited(tmp_path: Path):
    """The clean-then-probed direction: the payload beside these is no longer scorable."""
    from studiobench.__main__ import invalidate_stale_reports

    out = _seed(tmp_path)
    rewritten = invalidate_stale_reports(
        out, archived = None, extra_init = "/probes/x.js", log = lambda *_: None
    )

    assert {p.name for p in rewritten} == {"summary.md", "ab.md"}
    for name in ("summary.md", "ab.md"):
        body = (out / name).read_text(encoding = "utf-8")
        assert "not scorable" in body
        assert "/probes/x.js" in body


def test_a_clean_run_invalidates_the_probe_refusal_it_inherited(tmp_path: Path):
    """The probed-then-clean direction, which keying only on the init script left open.

    A probe run replaces both reports with text saying the payload beside them is not scorable.
    The next clean run into the same directory archives that payload and records a scorable one,
    and writes neither report of its own -- `summary.md` only under `--report`, `ab.md` only under
    `--ab` -- so the refusal survives and is read as a finding about a run it never saw.
    """
    from studiobench.__main__ import invalidate_stale_reports

    out = _seed(tmp_path, "NO SUMMARY: ... is not scorable ...")
    archived = out / "payload-20260101-000000.jsonl"
    rewritten = invalidate_stale_reports(
        out, archived = archived, extra_init = None, log = lambda *_: None
    )

    assert {p.name for p in rewritten} == {"summary.md", "ab.md"}
    for name in ("summary.md", "ab.md"):
        body = (out / name).read_text(encoding = "utf-8")
        assert "payload-20260101-000000.jsonl" in body
        # The stale claim is GONE, not merely accompanied by a correction.
        assert "is not scorable" not in body


def test_a_resume_that_changed_nothing_leaves_the_reports_alone(tmp_path: Path):
    """The control. Nothing was archived and nothing is probed, so nothing is stale."""
    from studiobench.__main__ import invalidate_stale_reports

    out = _seed(tmp_path)
    assert invalidate_stale_reports(out, archived = None, extra_init = None) == []
    assert (out / "summary.md").read_text(encoding = "utf-8") == "old report"
    assert (out / "ab.md").read_text(encoding = "utf-8") == "old report"


def test_only_the_reports_that_exist_are_written(tmp_path: Path):
    """A missing report stays missing: this replaces stale files, it does not manufacture them."""
    from studiobench.__main__ import invalidate_stale_reports

    (tmp_path / "summary.md").write_text("old report", encoding = "utf-8")
    rewritten = invalidate_stale_reports(
        tmp_path, archived = tmp_path / "p.jsonl", extra_init = None, log = lambda *_: None
    )

    assert [p.name for p in rewritten] == ["summary.md"]
    assert not (tmp_path / "ab.md").exists()


def test_the_run_passes_the_archive_result_to_the_invalidation(main_src: str):
    """The wiring the behavioural tests above cannot see: `run` must capture what it archived."""

    assert "archived = prepare_payload(" in main_src
    assert (
        "invalidate_stale_reports(paths.out, archived = archived, extra_init = extra_init)"
        in main_src
    )


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


def _node_parses(source: str):
    """Parse `source` the way Playwright ships it: wrapped in its `class InitScript` arrow IIFE.

    Returns None when it parses, or the engine's message when it does not. Skips with no node.
    """

    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on PATH; this assertion needs a real JS parser, not a regex")
    with tempfile.TemporaryDirectory() as tmp:
        # `--check` reads a FILE, so the IIFE is written out rather than passed with `-e`: an
        # argument would have to survive the shell, and the point of this helper is that nothing
        # between the producer and the parser is allowed to edit the bytes.
        script = Path(tmp) / "init_script.js"
        script.write_text("(() => {\n" + source + "\n})();", encoding = "utf-8")
        done = subprocess.run([node, "--check", str(script)], capture_output = True, text = True)
    return None if done.returncode == 0 else done.stderr.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "truncated",
    [
        "var result =",
        "let out =",
        "const seen =",
        "window.__cvpot = {\n  count: 0,\n};\nvar next =",
        '"use strict";\nvar result =',
    ],
)
def test_a_truncated_probe_cannot_stamp_itself_installed(truncated: str):
    """REGRESSION. The stamp must not be able to complete the source it is attesting to.

    A file cut off after an assignment operator is a SyntaxError on its own, and concatenating the
    stamp onto it USED TO MAKE IT VALID: `window.__sbExtraInitScript = <path>` became the dangling
    initializer, so the one variable the probe never got to compute was assigned the stamp and the
    deferred check returned early. Nothing threw, so `pageerror` said nothing either. Measured on
    webkit and chromium: stamp set, console silent, probe never ran -- the harness reporting a
    clean probe run for a probe whose source does not parse.

    This drives the PRODUCER and a real parser, because the defect is JavaScript grammar: any
    assertion made of string matching would have passed against the broken version too.
    """

    import studiobench.__main__ as sb

    assert _node_parses(truncated) is not None, "fixture is not actually malformed"
    combined = sb._probe_init_scripts("probes/p.js", truncated)[0]
    assert _node_parses(combined) is not None, (
        "a truncated probe became a VALID script once the stamp was appended, so "
        "window.__sbExtraInitScript is set by source the probe never executed and the deferred "
        f"check stays silent about a probe that never ran: {combined!r}"
    )


@pytest.mark.parametrize(
    "healthy",
    [
        '"use strict";\nvar ticks = 0;',
        "window.__cvpot = 1;",
        "// nothing but a comment",
        "function f() { return 1; }\nf();",
        "(function () { window.q = 1; })()",
    ],
)
def test_a_healthy_probe_still_parses_and_still_stamps(healthy: str):
    """The boundary must not cost a well-formed probe its stamp."""

    import studiobench.__main__ as sb

    combined = sb._probe_init_scripts("probes/p.js", healthy)[0]
    assert _node_parses(combined) is None, f"the boundary broke a valid probe: {combined!r}"
    assert combined.startswith(healthy), "the probe source is no longer the first thing in it"
