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

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"
FRONTEND = ROOT / "studio" / "frontend"

HARNESS = "playwright_heavy_thread.py"
# Recorded by the harness, produced only by Chromium.
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
    # An action that is driven but never checked for growth is an action whose column could be constant at every size
    # without anything failing.
    # The axes are generated from ACTIONS, so what has to hold is that ACTIONS is the generator and that it still lists
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
    # The whole point of the portable metrics:
    axes = growth_axes()
    for metric in CHROMIUM_ONLY:
        assert f'"{metric}"' not in axes, metric


def test_the_verdict_never_rests_on_a_chromium_only_metric() -> None:
    decision = verdict()
    for metric in CHROMIUM_ONLY:
        assert f'["{metric}"]' not in decision, metric


def test_chromium_only_rows_say_so_in_their_own_label() -> None:
    text = source(HARNESS)
    table = section(text, "TABLE_ROWS = (", "def print_table")
    for metric in CHROMIUM_ONLY:
        for line in table.splitlines():
            if f'"{metric}"' in line:
                assert "chromium only" in line, line


def test_the_longtask_api_is_recorded_as_supported_or_not() -> None:
    # Off Chromium these print `-`, and a `-` that means "not supported here" must not be read as
    text = source(HARNESS)
    assert "__longTaskSupported" in text
    assert '("longtask api supported", lambda r: r["long_task_supported"])' in text


def test_the_stall_detector_is_a_timer_and_not_a_message_channel() -> None:
    # Without this flag an engine with no Long Tasks API reports zero long tasks in exactly the same shape as an engine
    # Measured, not preference:
    text = source(HARNESS)
    assert "new MessageChannel(" not in text, "the recorder must not spin a port"
    assert "setTimeout(stall, 1)" in text


def test_the_verdict_asserts_the_fixture_and_not_just_its_size() -> None:
    decision = verdict()
    assert 'plan["expectedPerCycle"]' in decision
    assert 'counts.get("highlightedTokens", 0)' in decision


def test_the_fixture_assertion_survives_deferred_fence_highlighting() -> None:
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    head = page.index("const EXPECTED_PER_CYCLE")
    expected = page[head : page.index("};", head)]
    assert "codeChars: 12000" in expected, "the floor has to be on something deferral cannot move"
    assert "highlightedTokens:" not in expected, "the token floor was the thing deferral broke"


def test_a_fence_may_be_deferred_or_highlighted_but_not_neither() -> None:
    # A floor on the TOKEN count partly measures where the viewport is: the same unchanged fixture dropped from 3,216
    # tokens per cycle to 1,322.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert "unhighlightedMountedFences" in page
    assert 'counts.get("unhighlightedMountedFences", 0)' in verdict()


def test_the_verdict_asserts_the_keystroke_reached_the_runtime() -> None:
    # 300K characters of prose would produce a rising curve too, and would be measuring something nobody reported.
    decision = verdict()
    assert 'keystroke["runtimeText"] != keystroke["domText"]' in decision


def test_the_paint_floor_is_measured_and_subtracted() -> None:
    text = source(HARNESS)
    assert "PAINT_FLOOR_JS" in text
    # Once per double-rAF wait the metric is clocked across, not once per metric:
    assert 'value -= count * row["paint_floor_ms"]' in section(
        text, "def growth(", "def report_growth"
    )


def test_the_verdict_asserts_the_reopen_really_unmounted() -> None:
    decision = verdict()
    assert 'reopened["closedMs"] is None' in decision


def test_the_verdict_asserts_discrimination() -> None:
    # The DOM value is what the harness itself wrote.
    # A keystroke that reached nothing still reports the ~33ms paint floor, which reads as a plausible timing.
    decision = verdict()
    assert 'row["discriminated"]' in decision
    assert "DISCRIMINATION_RATIO" in decision


def test_the_smoke_page_exposes_every_count_the_fixture_gate_needs() -> None:
    # The SETTLEMENT half of the old token floor, asked per block.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    expected = section(page, "const EXPECTED_PER_CYCLE", "};")
    counts = section(page, "counts(): Record<string, number>", "viewportMetrics()")
    for line in expected.splitlines():
        key = line.strip().split(":")[0]
        if key.isidentifier():
            assert f"{key}:" in counts, key


def test_the_smoke_page_is_served_and_owns_its_dev_server() -> None:
    # Two rAFs resolve no sooner than two vsync intervals, so an action that never happened still reports ~33ms.
    text = source(HARNESS)
    assert (FRONTEND / "smoke-heavy-thread.html").exists()
    assert (FRONTEND / "smoke-heavy-thread-main.tsx").exists()
    assert "start_vite(PORT)" in text
    assert "stop_process(vite)" in text


def test_the_fork_count_stub_answers_the_shape_the_endpoint_returns() -> None:
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    # Pin the fork-count entry to its own body rather than scanning the whole file:
    forks = next(
        (line for line in page.splitlines() if "forks$/" in line),
        "",
    )
    assert forks, "the fork-count endpoint is no longer in the stub allowlist"
    assert '{"counts":{}}' in forks, (
        "the fork-count stub must answer the counts map the endpoint returns; another shape "
        f"leaves the parsed map empty only by accident. Got: {forks.strip()!r}"
    )


def _stub_patterns(page: str) -> list[str]:
    """The regex literals in STUBBED_API, as Python patterns.

    They are deliberately plain -- literal path segments, `[^/]+`, `(\\?|$)`, `$` -- so the JS
    source and the Python equivalent differ only in the escaped forward slashes.
    """
    block = page[page.index("const STUBBED_API") : page.index("const stubbedApiCalls")]
    return [literal.replace("\\/", "/") for literal in re.findall(r"\[/(.+?)/,", block)]


def test_the_stub_matches_the_fork_count_url_the_app_actually_requests() -> None:
    # The drift this file exists to catch, checked against the app rather than against a string someone remembered to
    # in #8992 and this allowlist was not moved with it;
    api = (FRONTEND / "src" / "features" / "chat" / "api" / "chat-api.ts").read_text(
        encoding = "utf-8"
    )
    fork_paths = re.findall(r"`(/api/chat/threads/\$\{[^`]*?\}/forks)`", api)
    assert fork_paths, "chat-api.ts no longer builds a fork-count URL this test can read"
    patterns = _stub_patterns(
        (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    )
    for path in fork_paths:
        # A stand-in shaped like the synthetic remoteId the local runtime hands the smoke page.
        url = re.sub(r"\$\{[^}]*\}", "__LOCALID_abc123", path)
        assert any(re.search(pattern, url) for pattern in patterns), (
            f"the smoke page's STUBBED_API allowlist answers none of {url!r}, which the chat "
            "client requests; it would reach the network inside a measured action"
        )


def test_the_fetch_stub_only_intercepts_fork_counts() -> None:
    # `getThreadForkCounts` reads `data.counts` and builds a Map from it, and the badge renders nothing for a message
    # the Map has no entry for.
    # Before that, `{}` against the per-message endpoint left `data.count` undefined, `undefined <= 0` false, and a
    # badge reading "undefined forks from this message" on every assistant message: measured at 25000 chars, 10 badges
    # and 4031 DOM nodes rather than 0 and 3981.
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert 'url.includes("/api/")' not in page, (
        "the fetch stub is matching every /api/ request again, which hides stray requests from "
        "the harness's own stray_api_requests counter"
    )
    assert (
        "forks$/" in page or "/forks" in page
    ), "the fetch stub must match the fork-count endpoint specifically"


def test_the_api_stub_is_an_allowlist_not_a_blanket_match() -> None:
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert (
        'url.includes("/api/")' not in page
    ), "the fetch stub is matching every /api/ request again"
    assert "STUBBED_API" in page, "the fetch stub must answer from an explicit allowlist"


def test_every_stubbed_endpoint_is_reported() -> None:
    # Playwright emits it, so `measure_cell`'s listener never increments `stray_api_requests` and
    # the API fan-out this harness claims to detect cannot reach it.
    # point, but it must not remove the request from the record. An endpoint that is answered and
    # A blanket `/api/` match resolves any other request a measured interaction makes before Playwright emits it, so
    # A blanket `/api/` match answers every request the measured interactions make before Playwright emits it, so
    # Answering a request inside the page removes its round trip from the timings, which is the point, but it must not
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    assert "__stubbedApi" in page, "stubbed requests must be recorded on the page"
    harness = source("playwright_heavy_thread.py")
    assert "stubbed_api_requests" in harness, "the harness must read the stubbed-request record"
    assert '"stubbed api requests"' in harness, "the stubbed-request count must reach the table"
