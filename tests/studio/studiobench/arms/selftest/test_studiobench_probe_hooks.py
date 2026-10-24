# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probes are off unless SBENCH_EXTRA_INIT_SCRIPT is set, so a scored run is never a probe run."""

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
    # Pinned as source: the failure is a refactor hoisting either call out of its `if`.
    for guarded in ("if extra_init:", "if console_prefix:"):
        assert guarded in main_src, f"{guarded!r} is gone; the hook may no longer be opt-in"


def test_the_probe_path_is_validated_before_anything_is_started(main_src: str):
    """Read the probe before Unsloth starts, so a bad path cannot leave Unsloth holding a port."""

    read_at = main_src.index("extra_init_source = Path(extra_init).read_text")
    assert read_at < main_src.index("install_studio(ref, home)")
    assert read_at < main_src.index("_probe_init_scripts(extra_init, extra_init_source)")
    assert "except (OSError, UnicodeDecodeError) as exc:" in main_src
    assert main_src.count("Path(extra_init).read_text") == 1
    # Also ahead of the archive, which moves the payload aside; behaviour is pinned in
    # runtime/selftest/test_studiobench_run_acquisition.py.
    assert read_at < main_src.index("prepare_payload(")


def test_the_probe_is_installed_without_eval(main_src: str):
    """Installed as source, not via eval: Unsloth's CSP forbids unsafe-eval, and webkit enforces that."""

    assert "def _probe_init_scripts(" in main_src
    assert "init_scripts.extend(_probe_init_scripts(extra_init, extra_init_source))" in main_src
    assert "(0, eval)(" not in main_src, (
        "the probe is back on eval; Unsloth's script-src 'self' blocks it and the probe will not "
        "install at all on the default engine"
    )
    assert "eval(" not in main_src, "any string evaluation is refused by Unsloth's CSP"


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
    """A probe run must record its gate; otherwise floor_table cannot tell it from a clean run."""

    assert '"probe_init_script": extra_init or None' in main_src
    # Indent not pinned: setup sits under a cleanup guard.
    assert re.search(r"rec\.gate\(\s*\n\s*\"probe_free\",", main_src)


def test_roots_are_adopted_at_insertion_not_on_the_sample_tick(probe_src: str):
    """Adopt roots at insertion: an off-screen root has one transition that a late listener would miss."""

    assert "MutationObserver" in probe_src
    assert "adoptAdded" in probe_src
    # The document, not documentElement: the root element may not exist yet.
    assert "observe(doc, {" in probe_src
    assert "addedNodes" in probe_src
    assert "adoptAll();" in probe_src


def test_the_fallback_and_padding_buckets_cannot_both_count_one_root(probe_src: str):
    """Fallback is tested before padding: one height <= 64 bucket would count a root under both traps."""

    assert "ROLE_PX" in probe_src
    assert "out.fallbackBite += 1;" in probe_src
    assert 'targetHeight(px, "padding")' in probe_src
    assert 'targetHeight(px, "fallback")' in probe_src
    assert 'return (which === "fallback" ? px.fallback : 0) + px.padding;' in probe_src


def test_only_a_skipped_root_can_land_in_a_size_bucket(probe_src: str):
    """content-visibility computes to auto either way, so only a skipped root can land in a size bucket."""

    assert "skippedState(el) === true" in probe_src


def test_detached_roots_stop_counting_as_skipped(probe_src: str):
    """Detached roots get no further transitions, so they must be dropped or skippedNow inflates."""

    assert "droppedDetached" in probe_src
    assert "doc.contains(wel)" in probe_src


def test_each_scene_script_keeps_its_own_failure_domain(main_src: str):
    """Each scene script stays its own add_init_script, so one throw cannot stop the others."""

    for name in ("scene/dom.js", "scene/parity.js", "scene/surfaces.js"):
        assert f'init_scripts.append(resources.read_text("{name}"))' in main_src
    assert "page_scripts" not in main_src


def test_the_probe_is_its_own_script_and_says_when_it_did_not_install(main_src: str):
    """Init script order is undefined, so the probe must be self-contained and report failures itself."""

    assert "init_scripts.extend(_probe_init_scripts(extra_init, extra_init_source))" in main_src
    assert "window.__sbExtraInitScript" in main_src
    assert "never installed: it did not " in main_src
    assert 'bundle.page.on("pageerror"' in main_src


def test_the_probe_source_is_the_first_thing_in_its_script():
    """Probe source must come first: a use strict directive stops being a prologue after any statement."""

    import studiobench.__main__ as sb

    source = '"use strict";\nvar ticks = 0;\n'
    scripts = sb._probe_init_scripts("probes/p.js", source)

    assert scripts[0].startswith(source), (
        "something is being prepended to the probe source; a leading directive such as "
        '"use strict" is no longer in the directive prologue and the probe runs with different '
        "semantics than the file it was read from"
    )
    assert "window.__sbExtraInitScript" in scripts[0]
    # After an explicit statement boundary so ASI cannot join it to the probe's last line.
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
    """A clean run must clear an inherited probe refusal, or it would read as a finding about this run."""
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
    """SystemExit is not an Exception, so a refused report must explicitly overwrite any old summary.md."""

    assert "except SystemExit as exc:" in main_src
    assert 'out = path.parent / "summary.md"' in main_src
    assert "# No summary" in main_src


def test_the_report_refuses_before_it_assembles(build_src: str):
    """Refuse before assembling rows, so an unrelated schema error cannot pre-empt the probe refusal."""

    before = build_src.index("refuse_if_probed(_records(path)")
    assert before < build_src.index("payload = assemble_rows(path)")


def test_the_event_counter_is_the_one_potency_rests_on(probe_src: str):
    """`ev_skip` is the only potency signal here that no author CSS can fake."""

    assert "contentvisibilityautostatechange" in probe_src
    assert "ev_skip" in probe_src
    # Documented so nobody reimplements the geometry route.
    assert "KNOWN FALSE NEGATIVE" in probe_src


def _node_parses(source: str):
    """Parses the source as Playwright wraps it in an arrow IIFE: returns None, or the engine's message."""

    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on PATH; this assertion needs a real JS parser, not a regex")
    with tempfile.TemporaryDirectory() as tmp:
        # `--check` reads a file, so nothing between producer and parser can edit the bytes.
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
    """The install stamp must not complete a truncated probe, or it reports installed when it never ran."""

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
