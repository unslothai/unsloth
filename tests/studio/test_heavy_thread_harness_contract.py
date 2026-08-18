# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The heavy-thread harness must read every guard it records, and must stay portable.

`tests/studio/playwright_heavy_thread.py` measures in a browser and then decides pass/fail in
`main()`. A metric that is recorded but never compared is how a harness goes false-green, which is
the rule already pinned for the #8483 harnesses in test_autoscroll_harness_contract.py.

This file adds the constraint that is specific to this harness: it is meant to run on WebKit and
Firefox as well as Chromium, because Unsloth Desktop is a Tauri webview and not Chromium. Every
CDP counter and the Long Tasks API are Chromium-only, and the failure mode is silent -- a
`longtask` PerformanceObserver on JavaScriptCore never fires, which reads as "no jank" rather than
as "no measurement". So no growth axis and no pass/fail decision may rest on one.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"
FRONTEND = ROOT / "studio" / "frontend"

HARNESS = "playwright_heavy_thread.py"
PROBE = "scroll_predecessor_probe.py"
# Recorded by the harness, produced only by Chromium. None of these may decide anything.
CHROMIUM_ONLY = (
    "layout_count",
    "layout_ms",
    "recalc_style_count",
    "recalc_style_ms",
    "task_ms",
    "long_tasks",
    "long_task_ms",
    "worst_long_task_ms",
)
# The portable four the module docstring promises as the primary numbers.
PORTABLE_PRIMARY = ("longest_stall_ms", "worst_frame_ms", "frames_over_33", "wall_ms")
ACTIONS = ("keystroke", "scroll", "jump", "menu", "delete", "reopen")


def source(name: str) -> str:
    return (STUDIO_TESTS / name).read_text(encoding = "utf-8")


def section(text: str, start: str, end: str) -> str:
    head = text.index(start)
    return text[head : text.index(end, head)]


def growth_axes() -> str:
    return section(source(HARNESS), "GROWTH_AXES = tuple(", "DISCRIMINATION_RATIO")


def verdict() -> str:
    """Everything from `def harness_failures` on: the only place a metric turns into an exit
    code."""
    text = source(HARNESS)
    return text[text.index("def harness_failures") :]


def test_every_measured_action_has_a_growth_axis() -> None:
    # An action that is driven but never checked for growth is an action whose column could be
    # constant at every size without anything failing. The axes are generated from ACTIONS, so
    # what has to hold is that ACTIONS is the generator and that it still lists all six.
    text = source(HARNESS)
    declared = section(text, "ACTIONS = (", ")")
    for action in ACTIONS:
        assert f'"{action}"' in declared, action
    axes = growth_axes()
    for metric in PORTABLE_PRIMARY:
        assert f"for a in ACTIONS" in axes and f'"{metric}"' in axes, metric


def test_the_portable_primaries_are_the_growth_axes() -> None:
    axes = growth_axes()
    for metric in PORTABLE_PRIMARY:
        assert f'"{metric}"' in axes, metric


def test_no_growth_axis_is_chromium_only() -> None:
    # The whole point of the portable metrics: a curve built on CDP counters is a curve that does
    # not exist on the engine Unsloth Desktop actually ships on macOS and Linux.
    axes = growth_axes()
    for metric in CHROMIUM_ONLY:
        assert f'"{metric}"' not in axes, metric


def test_the_verdict_never_rests_on_a_chromium_only_metric() -> None:
    decision = verdict()
    for metric in CHROMIUM_ONLY:
        assert f'["{metric}"]' not in decision, metric


def test_chromium_only_rows_say_so_in_their_own_label() -> None:
    # Off Chromium these print `-`, and a `-` that means "not supported here" must not be read as
    # a zero. The label is the only thing carrying that.
    text = source(HARNESS)
    table = section(text, "TABLE_ROWS = (", "def print_table")
    for metric in CHROMIUM_ONLY:
        for line in table.splitlines():
            if f'"{metric}"' in line:
                assert "chromium only" in line, line


def test_the_longtask_api_is_recorded_as_supported_or_not() -> None:
    # Without this flag an engine with no Long Tasks API reports zero long tasks in exactly the
    # same shape as an engine that had none.
    text = source(HARNESS)
    assert "__longTaskSupported" in text
    assert '("longtask api supported", lambda r: r["long_task_supported"])' in text


def test_the_stall_detector_is_a_timer_and_not_a_message_channel() -> None:
    # Measured, not preference: the MessageChannel ping-pong halves Firefox's frame rate before
    # any application code runs, so it changes the thing it is there to measure.
    text = source(HARNESS)
    assert "new MessageChannel(" not in text, "the recorder must not spin a port"
    assert "setTimeout(stall, 1)" in text


def test_the_verdict_asserts_the_fixture_and_not_just_its_size() -> None:
    # 300K characters of prose would produce a rising curve too, and would be measuring something
    # nobody reported.
    decision = verdict()
    assert 'plan["expectedPerCycle"]' in decision
    assert 'counts.get("highlightedTokens", 0)' in decision


def test_the_verdict_asserts_the_keystroke_reached_the_runtime() -> None:
    # The DOM value is what the harness itself wrote. A keystroke that reached nothing still
    # reports the ~33ms paint floor, which reads as a plausible timing.
    #
    # Checked per repetition, not on the collapsed last one. `summarise` keeps only the LAST
    # repetition's copy of each non-numeric proof, while the median above it is taken over every
    # repetition, so a keystroke that reached a dead composer on repetition 1 still contributed
    # its timing to the reported number and a verdict reading the last repetition alone passes it.
    decision = verdict()
    assert "domText_per_repetition" in decision
    assert "runtimeText_per_repetition" in decision
    assert "for index, (dom, runtime) in enumerate(zip(dom_texts, runtime_texts)):" in decision


def test_every_non_numeric_proof_is_kept_for_every_repetition() -> None:
    # The four non-numeric fields are the only evidence each interaction happened at all, and
    # collapsing them to the last repetition is what makes the check above possible to defeat.
    # Same shape as `dropped_repetitions`, which stops a headline timeout being reported as a
    # median of three.
    text = source(HARNESS)
    summarise_body = section(text, "def summarise(", "class SeededPage")
    assert 'merged[f"{key}_per_repetition"] = values' in summarise_body
    # And the two verdicts that read them must read the per-repetition form.
    decision = verdict()
    assert "bodyPointerEventsAfterClose_per_repetition" in decision
    assert "bodyPointerEvents_per_repetition" in decision


def test_the_paint_floor_is_measured_and_subtracted() -> None:
    # Two rAFs resolve no sooner than two vsync intervals, so an action that never happened still
    # reports ~33ms. Left in a ratio, that floor compresses every axis towards 1 and lets a real
    # regression sit under the discrimination threshold.
    text = source(HARNESS)
    assert "PAINT_FLOOR_JS" in text
    # Was `row["paint_floor_ms"]`, the merged MEDIAN. In the isolated table each action runs on
    # its own page with its own measured floor, so the median belongs to none of them and
    # subtracting it from an action measured on a higher or lower page moved the corrected
    # endpoints and the discrimination ratio with them. The stronger property is asserted here:
    # the subtraction goes through action_floor(), which prefers the floor of the page that
    # produced the timing and falls back to the median only when there is no per-action floor.
    assert "value -= count * action_floor(row, action)" in section(
        text, "def growth(", "def report_growth"
    )
    floor_fn = section(text, "def action_floor(", "def growth(")
    assert '"paint_floor_ms_by_action"' in floor_fn, (
        "action_floor no longer consults the per-action floors, so every action is back on the "
        "merged median"
    )
    assert (
        'row.get("paint_floor_ms", 0)' in floor_fn
    ), "action_floor has no median fallback, so axes that are not per-action lose their floor"


def test_the_verdict_asserts_the_reopen_really_unmounted() -> None:
    # Without this, "re-open" is timing a thread that never left, which is free.
    decision = verdict()
    assert 'reopened["closedMs"] is None' in decision


def test_the_verdict_asserts_discrimination() -> None:
    # A harness where the largest thread costs what the smallest does is not reporting a flat
    # curve, it is reporting that it never drove the page.
    decision = verdict()
    assert 'row["discriminated"]' in decision
    assert "DISCRIMINATION_RATIO" in decision


def test_the_smoke_page_exposes_every_count_the_fixture_gate_needs() -> None:
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    expected = section(page, "const EXPECTED_PER_CYCLE", "};")
    counts = section(page, "counts(): Record<string, number>", "viewportMetrics()")
    for line in expected.splitlines():
        key = line.strip().split(":")[0]
        if key.isidentifier():
            assert f"{key}:" in counts, key


def test_the_recorder_retires_the_callbacks_of_a_finished_run() -> None:
    # `end()` clears `running`, which does NOT unschedule the frame callback and the 1ms timer
    # that are already queued. Actions run back to back, so the next `begin()` can set `running`
    # to true again before those callbacks fire; they then push into the arrays the new run just
    # emptied and re-schedule themselves, and the new action is recorded by two loops at once.
    # Measured on Chromium with this recorder alone on a blank page, five back-to-back 300ms
    # runs: stall_ticks 78, 152, 226, 300, 374 without the token and 77, 78, 77, 78, 78 with it.
    # `longest_stall_ms` and `frames_over_33` are growth axes, so this lands in the verdict.
    recorder = section(source(HARNESS), "const recorder = {", "window.__hv = recorder;")
    assert "generation: 0," in recorder
    # #9016's recorder writes the guard as `generation !== this.generation`, and drops the
    # `!this.running` half because retiring the generation in end() already stops the loop.
    # Same property, checked against the form that is actually in the tree.
    guard = "if (generation !== this.generation) return;"
    assert recorder.count(guard) == 2, "both the frame loop and the stall loop need the guard"
    # The property, asserted as an ORDER rather than as the absence of a string.
    #
    # The defect is a callback belonging to a retired generation writing its sample anyway. What
    # prevents it is the guard running BEFORE the push, which is what this checks. The earlier
    # version of this assertion instead required `if (this.running) nativeRaf(frame);` to be
    # absent, and #9016's recorder keeps that at the RESCHEDULE as a second cheap check while
    # guarding the top of the callback properly. So the old assertion failed against correct
    # code, which is worse than not checking at all.
    for loop, sample in (("frame", "this.frames.push("), ("stall", "this.stalls.push(")):
        body = recorder[recorder.index(f"const {loop} = () => {{") :]
        body = body[: body.index("};")]
        assert guard in body, f"the {loop} loop has no generation guard: {body}"
        assert body.index(guard) < body.index(sample), (
            f"the {loop} loop writes its sample before checking the generation, so a callback "
            f"from a retired run still lands in the next run's array: {body}"
        )


def pointer_precondition() -> str:
    return section(
        source(HARNESS), "def place_pointer_over_message(", "def reveal_last_action_bar("
    )


def test_the_pointer_precondition_moves_the_real_mouse() -> None:
    # A helper that only reads elementFromPoint verifies where the cursor ALREADY is, which on a
    # fresh isolated page is (0, 0). Only page.mouse.move puts it on the conversation, and that is
    # the whole difference between this row and the probe's `gutter_only` arm.
    assert "page.mouse.move(" in pointer_precondition()


def test_the_pointer_precondition_verifies_it_landed_on_a_message() -> None:
    # Moving to a coordinate proves nothing: the scroller's own gutter is inside the scroller in
    # this fixture, so an unverified point can hit the viewport element and make this a second
    # copy of the cheap arm under the expensive arm's label.
    assert "POINTER_TARGET_JS" in pointer_precondition()
    assert "el.closest('[data-role=\"assistant\"]')" in section(
        source(HARNESS), "POINTER_TARGET_JS = ", "def place_pointer_over_message("
    )


def test_the_isolated_scroll_arm_positions_the_pointer() -> None:
    # Each isolated page is fresh, so its mouse has never moved and sits in the gutter. Measured
    # on this tree at 300K: gutter 7.4ms longest stall and 17.9ms worst frame, pointer on content
    # 37.3ms and 29.3ms, both of which are headline portable primaries.
    isolated = section(source(HARNESS), "def isolated_repetitions(", "def sequenced_repetition(")
    assert "drive_scroll(page, cdp)" in isolated


def test_the_sequenced_scroll_arm_positions_the_pointer() -> None:
    # And the carry-over runner too, or the two tables measure different gestures under one name:
    # there the pointer is at the origin on repetition 1 and over content from repetition 2, left
    # by the previous repetition's delete hover, which is two mechanisms inside one median.
    sequenced = section(source(HARNESS), "def sequenced_repetition(", "# Portable headline")
    assert "drive_scroll(page, cdp)" in sequenced


def test_the_verdict_rejects_a_scroll_whose_pointer_was_off_content() -> None:
    # Per repetition, not on the collapsed last one: a repetition that scrolled the gutter still
    # contributed its timing to the reported median.
    decision = verdict()
    assert "pointer_on_message_per_repetition" in decision
    assert "with the pointer off message content on " in decision


def test_the_probe_enables_the_cdp_performance_domain() -> None:
    # `Performance.getMetrics` on a session that never had `Performance.enable` sent returns an
    # EMPTY metric list rather than an error (measured on this tree: 0 metrics without, 36 with).
    # `cdp_counters` then reports None for every counter and the probe's table prints 0.0 in the
    # task, layout and style columns, which reads as an arm that did no main-thread work.
    assert 'cdp.send("Performance.enable")' in source(PROBE)


def probe_tree() -> ast.Module:
    return ast.parse(source(PROBE))


def test_every_predecessor_the_probe_defines_is_registered() -> None:
    # An unregistered `before_*` helper is a predecessor the file claims to test and never runs,
    # and `PROBE_ARMS=<its name>` then filtered the sweep to zero arms.
    tree = probe_tree()
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("before_")
    }
    registered = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "ARMS" for t in node.targets):
            continue
        for arm in node.value.elts:
            for element in arm.elts:
                if isinstance(element, ast.Name):
                    registered.add(element.id)
    assert registered, "ARMS did not parse"
    assert defined == registered, f"never runs: {sorted(defined - registered)}"


def test_the_probe_fails_when_an_arm_fails() -> None:
    # Every arm can throw and be recorded as `failed` while the process still exits 0, so a run
    # that produced no comparison at all is indistinguishable from a successful experiment.
    tree = probe_tree()
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    codes = {
        node.value.value
        for node in ast.walk(main)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
    }
    assert 1 in codes, "main() never returns a failure code"
    assert 'info(f"FAILED arms: ' in source(PROBE)


def test_the_probe_rejects_an_arm_name_it_does_not_have() -> None:
    # PROBE_ARMS=<typo> used to trim the sweep to nothing and print an empty table under a zero
    # exit code.
    assert "PROBE_ARMS names no such arm" in source(PROBE)


def test_the_probe_verifies_the_pointer_before_publishing_a_gutter_arm() -> None:
    # The scroller fills the viewport in this fixture, so "the pointer is outside the scroller"
    # is unsatisfiable and the predicate that demanded it rejected every candidate, left the
    # mouse on an unverified corner and published the two control arms anyway. The scroller
    # element ITSELF is the stable hit-test target the arms are named for.
    probe = source(PROBE)
    park = section(probe, "def park_pointer_in_gutter(", "def before_hover_then_gutter(")
    assert "el === v || !v.contains(el)" in park
    assert '"pointer_stable": True' in park
    # And no verified point at all must fail the arm rather than time it.
    assert "raise RuntimeError(" in park


def test_the_smoke_page_is_served_and_owns_its_dev_server() -> None:
    text = source(HARNESS)
    assert (FRONTEND / "smoke-heavy-thread.html").exists()
    assert (FRONTEND / "smoke-heavy-thread-main.tsx").exists()
    assert "start_vite(PORT)" in text
    assert "stop_process(vite)" in text


def test_fixture_cleanup_runs_after_the_counters_are_read() -> None:
    # run_action brackets the whole page.evaluate with the CDP counters and reads the long-task
    # observer after it returns, so anything the action script does before returning lands in the
    # Chromium-only rows even when the portable recorder window has already closed. The jump's
    # scroll back to the bottom cost 33.2ms against the jump's own 33.5ms at 300K chars, so those
    # rows described two jumps. Cleanup belongs in ACTION_RESETS, applied after every snapshot.
    body = source("playwright_heavy_thread.py")
    run_action = body[body.index("def run_action(") : body.index("ACTION_SCRIPTS = {")]
    assert "ACTION_RESETS" in run_action, "run_action never applies the post-snapshot resets"
    reset_at = run_action.index("ACTION_RESETS")
    for snapshot in ("cdp_counters(before, after)", "long_task_summary(page)"):
        assert snapshot in run_action, f"run_action no longer collects {snapshot}"
        assert (
            run_action.index(snapshot) < reset_at
        ), f"the fixture reset runs before {snapshot}, so its cost lands in that snapshot"


def test_the_jump_does_not_scroll_back_inside_its_own_measurement() -> None:
    # Specifically the regression above: the reset was inline at the end of JUMP_JS.
    body = source("playwright_heavy_thread.py")
    jump = body[body.index("JUMP_JS = ") : body.index("MENU_JS = ")]
    # The scroll to the bottom BEFORE __hv.begin() is setup and is meant to be there. What must
    # not come back is a scroll after __hv.end(), which the recorder no longer sees but the CDP
    # counters and the long-task observer still do.
    after_end = jump[jump.index("__hv.end()") :]
    # `scrollTo(` with the paren: a bare "scrollTo" also matches `viewport.scrollTop`, which is
    # a read and is fine here.
    assert (
        "scrollTo(" not in after_end
    ), "JUMP_JS scrolls again after __hv.end(), inside the evaluate the counters bracket"


def test_the_smoke_page_can_put_a_deleted_message_back() -> None:
    # `delete` removes a message from the runtime's REPOSITORY, not from the view, so neither
    # closeThread/openThread nor expandTools undoes it. Without a restore the isolated delete arm
    # measures a thread that is one message shorter on every repetition.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert "restore(): number {" in page, "the smoke page exposes no way to rebuild the fixture"
    # And it has to re-IMPORT the seeded messages. A restore that only re-renders what the
    # repository still holds puts nothing back.
    restore = section(page, "restore(): number {", "expandTools()")
    assert "aui.thread().import(" in restore, restore


def test_the_isolated_runner_restores_the_fixture_after_a_mutating_action() -> None:
    # REPEATS repetitions of `delete` on ONE page, each removing the last assistant message. The
    # fixture is whole cycles of one message per kind, so the three timings behind the median
    # delete three different subtree types on a shrinking thread, which is the 25K vs 300K growth
    # comparison this file exists for.
    isolated = section(source(HARNESS), "def isolated_repetitions(", "def sequenced_repetition(")
    assert "if name in MUTATING_ACTIONS:" in isolated
    assert "restore_fixture(page)" in isolated
    mutating = section(source(HARNESS), "MUTATING_ACTIONS = (", ")")
    assert '"delete"' in mutating, mutating
    # `__heavyThread.restore()` with the parens, not a bare "restore": `restore_fixture` and the
    # comment above it both contain the word.
    helper = section(source(HARNESS), "def restore_fixture(", "def isolated_repetitions(")
    assert "window.__heavyThread.restore()" in helper


def test_the_isolated_runner_records_the_thread_each_repetition_started_from() -> None:
    # The restore above is an assertion rather than an intention only if the run reads it back.
    isolated = section(source(HARNESS), "def isolated_repetitions(", "def sequenced_repetition(")
    assert "window.__heavyThread.messageCount()" in isolated
    assert 'row["fixture_messages"] = fixture_messages' in isolated
    decision = verdict()
    assert "fixture_messages_per_repetition" in decision
    assert "the fixture was not restored between them" in decision


def test_the_probe_restores_the_fixture_between_mutating_repetitions() -> None:
    # PROBE_REPS repetitions per arm on one seeded page. `delete` and `delete_reopen_keystroke`
    # each remove another assistant message, and the arms are compared against a `nothing` control
    # that still holds the whole fixture.
    probe = source(PROBE)
    assert "window.__heavyThread.restore()" in probe
    mutating = section(probe, "MUTATING_ARMS = (", ")")
    for arm in ('"delete"', '"delete_reopen_keystroke"'):
        assert arm in mutating, mutating
    # And the drift has to be read back, or the restore is only an intention.
    assert "def fixture_drift(" in probe
    assert "raise RuntimeError(drift)" in probe


def test_the_probe_checks_that_each_predecessor_completed() -> None:
    # Every action script encodes a timeout as a null FIELD and a missing element as a null
    # RETURN, neither of which raises, and the probe's arm loop records an arm as failed only on
    # an exception. So a menu that never opened published a scroll with no predecessor in front of
    # it, in the row labelled `menu`, under a zero exit code.
    tree = probe_tree()
    checked = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "checked" in checked, "no predecessor is validated"
    # Every `before_*` that evaluates one of the harness's action scripts has to route it through
    # `checked`, not just some of them.
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("before_"):
            continue
        drives = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "evaluate"
            and any(isinstance(a, ast.Attribute) and a.attr.endswith("_JS") for a in call.args)
        ]
        validated = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "checked"
        ]
        assert len(validated) >= len(drives), f"{node.name} publishes an unchecked predecessor"


def test_the_verdict_reads_every_numeric_proof_per_repetition() -> None:
    # median() returns None the moment one repetition is None, which covers every TIMING because
    # a timed-out action reports null. It does nothing for a proof that is a NUMBER in every
    # repetition and merely the wrong number in one: a jump that landed at [0, bottom, 0] medians
    # to 0, which is exactly what an arrived jump looks like.
    text = source(HARNESS)
    summarise_body = section(text, "def summarise(", "class SeededPage")
    assert "for key in NUMERIC_PROOFS:" in summarise_body
    proofs = section(text, "NUMERIC_PROOFS = (", ")")
    decision = verdict()
    for key in ("scrolledPx", "landedAt", "travelledPx", "itemsWhileOpen", "before", "after"):
        assert f'"{key}"' in proofs, key
        assert f'"{key}_per_repetition"' in decision, key


def test_the_fork_count_stub_answers_with_a_real_zero() -> None:
    # `getForkCount` returns `data.count`, and the badge's guard is `count <= 0`. `undefined <= 0`
    # is false, so a `{}` body renders a badge reading "undefined forks from this message" on every
    # assistant message. Measured at 25000 chars: 10 badges and 4031 DOM nodes with `{}`, 0 badges
    # and 3981 with `{"count":0}`. That is DOM in proportion to thread size, on the axis this
    # harness measures.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    # Pin the fork-count entry to its own body rather than scanning the whole file: other
    # endpoints in the allowlist legitimately answer "{}", so a bare file-wide check for it
    # would fail on them and tell us nothing about this one.
    forks = next(
        (line for line in page.splitlines() if "forks$/" in line),
        "",
    )
    assert forks, "the fork-count endpoint is no longer in the stub allowlist"
    assert '{"count":0}' in forks, (
        "the fork-count stub must answer with a numeric count; an empty object makes the parsed "
        f"count undefined and renders a badge on every assistant message. Got: {forks.strip()!r}"
    )


def test_the_fetch_stub_only_intercepts_fork_counts() -> None:
    # A blanket `/api/` match resolves any other request a measured interaction makes before
    # Playwright emits it, so `measure_cell`'s listener never increments `stray_api_requests` and
    # the API fan-out this harness claims to detect cannot reach it.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert 'url.includes("/api/")' not in page, (
        "the fetch stub is matching every /api/ request again, which hides stray requests from "
        "the harness's own stray_api_requests counter"
    )
    assert (
        "forks$/" in page or "/forks" in page
    ), "the fetch stub must match the fork-count endpoint specifically"


def test_the_api_stub_is_an_allowlist_not_a_blanket_match() -> None:
    # A blanket `/api/` match answers every request the measured interactions make before Playwright
    # emits it, so `stray_api_requests` stays at zero and the fan-out this harness exists to detect
    # is invisible to it. Narrowing it is what revealed the project-list and knowledge-base GETs on
    # reopen, and the delete's own three-request sync.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert (
        'url.includes("/api/")' not in page
    ), "the fetch stub is matching every /api/ request again"
    assert "STUBBED_API" in page, "the fetch stub must answer from an explicit allowlist"


def test_every_stubbed_endpoint_is_reported() -> None:
    # Answering a request inside the page removes its round trip from the timings, which is the
    # point, but it must not remove the request from the record. An endpoint that is answered and
    # not counted is one nobody can see the cost of later.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert "__stubbedApi" in page, "stubbed requests must be recorded on the page"
    harness = source("playwright_heavy_thread.py")
    assert "stubbed_api_requests" in harness, "the harness must read the stubbed-request record"
    # Both halves must reach the table. The seed snapshot alone is the number the harness used to
    # publish, and it cannot contain a single request any measured action made.
    assert (
        '"seed stubbed api requests"' in harness
    ), "the seed-time stubbed-request count must reach the table"
    assert (
        '"action stubbed api requests"' in harness
    ), "the action-attributable stubbed-request count must reach the table"
