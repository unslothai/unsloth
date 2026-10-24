# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A waiting queue with nothing streaming must not read as running, or settled pairs get refused."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import parity as P  # noqa: E402

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_PARITY_JS = _STUDIO_TESTS / "studiobench" / "scene" / "parity.js"
_THREAD_TSX = (
    _STUDIO_TESTS.parents[1]
    / "studio"
    / "frontend"
    / "src"
    / "components"
    / "assistant-ui"
    / "thread.tsx"
)

_QUEUE_BUTTON_RUNNING = """
<div class="ml-1.5 flex items-center">
  <button class="aui-composer-send size-9 rounded-full" aria-label="Queue message">
    <span class="aui-sr-only">Queue message</span>
  </button>
</div>
"""

_QUEUE_BUTTON_IDLE = """
<button class="aui-composer-send ml-1.5 size-9 rounded-full" aria-label="Queue message">
  <span class="aui-sr-only">Queue message</span>
</button>
"""

_STOP_BUTTON = """
<button class="aui-composer-cancel size-9 rounded-full" aria-label="Stop generating">
  <span class="aui-sr-only">Stop generating</span>
</button>
"""

#: Dispatched queued branch: neither stopButton() nor queueButton() matches it, and the queue
#: surface may be gone, so it must be recognised from its own control.
_STOP_QUEUED_BUTTON = """
<button class="aui-composer-cancel ml-1.5 size-9 rounded-full" aria-label="Stop queued message">
  <span class="aui-sr-only">Stop queued message</span>
</button>
"""

_QUEUE_STACK = """
<div aria-label="Prompt queue, 1 of 2">
  <div>a prompt that has not been dispatched</div>
</div>
"""


def _page(
    *,
    control: str,
    queue_stack: bool,
    statuses: list[str | None],
    tail: str,
    overlay: str = "",
) -> str:
    """The status hook lives on assistant rows only, so a rename is modelled there, not on user rows."""
    messages = []
    for i, status in enumerate(statuses):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user":
            messages.append('<div data-role="user"><div>the prompt</div></div>')
            continue
        body = f"reply {i} {tail if i == len(statuses) - 1 else ''}"
        attr = 'data-state="running"' if status is None else f'data-status="{status}"'
        messages.append(f'<div data-role="{role}"><div {attr}>{body}</div></div>')
    return f"""<!doctype html><meta charset="utf-8">
<div class="aui-thread-root">
  <div class="aui-thread-viewport">{"".join(messages)}</div>
  <div class="aui-composer-root">
    {_QUEUE_STACK if queue_stack else ""}
    <textarea aria-label="Message input">typed while it ran</textarea>
    {control}
  </div>
</div>{overlay}"""


STREAMING = dict(
    control = _QUEUE_BUTTON_RUNNING,
    queue_stack = False,
    statuses = ["complete", "complete", "complete", "running"],
)
QUEUED_IDLE = dict(
    control = _QUEUE_BUTTON_IDLE,
    queue_stack = True,
    statuses = ["complete", "complete", "complete", "complete"],
)
BLIND = dict(
    control = _STOP_BUTTON,
    queue_stack = False,
    statuses = [None, None, None, None],
)
#: Pinned cost: a waiting queue, a streaming reply and composer text read as not armed.
QUEUED_AND_STREAMING_BLIND = dict(
    control = _QUEUE_BUTTON_RUNNING,
    queue_stack = True,
    statuses = [None, None, None, None],
)


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    pg = browser.new_page(viewport = {"width": 900, "height": 700})
    yield pg
    pg.close()


def _capture(
    page,
    state: dict,
    tail: str = "settled",
) -> dict:
    page.set_content(_page(tail = tail, **state))
    # After the content: `set_content` does not reliably run init scripts.
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture()")
    assert got.get("parity_attempted") is True, got
    return got


def _reading(cap: dict) -> dict:
    return {k: cap.get(k) for k in ("streaming", "in_flight", "in_flight_unplaced", "queued_idle")}


def test_a_waiting_queue_is_not_read_as_a_blind_probe(page):
    """THE REGRESSION. Same button, same empty in-flight list, opposite meanings."""
    idle = _capture(page, QUEUED_IDLE)
    blind = _capture(page, BLIND)
    # Both refuse a fresh send, so `isRunning()` cannot tell them apart.
    assert idle["streaming"] is True and blind["streaming"] is True
    assert idle["in_flight"] == [] and blind["in_flight"] == []
    assert _reading(idle) != _reading(blind)
    assert idle["in_flight_unplaced"] is False, idle
    assert idle["queued_idle"] is True, idle
    assert blind["in_flight_unplaced"] is True, blind
    assert blind["queued_idle"] is False, blind


def test_a_settled_queued_idle_pair_is_scored_rather_than_refused(page):
    """Reading the queue button as running would refuse a settled pair before its digests are compared."""
    base = _capture(page, QUEUED_IDLE, tail = "alpha")
    treat = _capture(page, QUEUED_IDLE, tail = "omega")
    assert base["digest"] != treat["digest"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg3(") for m in got["moved"]), got["moved"]
    refused = P.compare(base, _capture(page, BLIND))
    assert refused["verdict"] == P.NOT_COMPARABLE
    assert "could not be identified" in refused["reason"]


def test_a_real_stream_is_still_placed_and_still_scored(page):
    """The coverage this must not cost: the running branch renders the SAME button."""
    cap = _capture(page, STREAMING)
    assert cap["streaming"] is True
    assert cap["in_flight"] == [3], cap
    assert cap["in_flight_unplaced"] is False and cap["queued_idle"] is False


def test_what_reading_the_queue_surface_gives_up(page):
    """A live stream with a prompt queued reads as queued-idle, so the control is not armed."""
    cap = _capture(page, QUEUED_AND_STREAMING_BLIND)
    assert cap["in_flight"] == []
    assert cap["in_flight_unplaced"] is False, cap
    assert cap["queued_idle"] is True, cap


#: The probes read the DOM in English, so the button labels live in this catalog.
_EN_LOCALE = _THREAD_TSX.parents[2] / "i18n" / "locales" / "en.ts"


def _en_string(key: str) -> str:
    """Reads promptQueue.<key> from en.ts by regex; a missing key fails rather than returning empty."""
    src = _EN_LOCALE.read_text(encoding = "utf-8")
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*\"((?:[^\"\\]|\\.)*)\",?\s*$", src)
    assert match, f"the en catalog no longer defines promptQueue.{key} ({_EN_LOCALE})"
    return match.group(1)


def test_the_shipped_composer_still_renders_the_two_queue_buttons():
    """Checks the composer still picks between two queue labels via the catalog, not English text."""
    if not _THREAD_TSX.exists():
        pytest.skip(f"the shipped composer is not in this checkout: {_THREAD_TSX}")
    src = _THREAD_TSX.read_text(encoding = "utf-8")
    assert src.count("aria-label={followUpLabel}") == 2, (
        "ComposerRightControls no longer renders the Queue button in exactly two places; "
        "re-read which of them can appear on an idle thread"
    )
    assert re.search(
        r'followUpBehavior === "queue"\s*\?\s*"promptQueue\.queueButton"\s*:\s*"promptQueue\.steerButton"',
        src,
    ), (
        "the composer no longer picks its label from followUpBehavior between the queueButton "
        "and steerButton keys; re-read what the two Queue buttons are now called"
    )
    queue_label, steer_label = _en_string("queueButton"), _en_string("steerButton")
    assert queue_label != steer_label, (
        f"the en catalog gives queue and steer the same label {queue_label!r}, so the two "
        "composer states are indistinguishable to every probe that reads the accessible name"
    )
    assert queue_label == "Queue message", queue_label
    assert steer_label == "Steer response", steer_label

    queue_src = (_THREAD_TSX.parent / "prompt-queue-list.tsx").read_text(encoding = "utf-8")
    assert 'aria-label={t("promptQueue.regionLabel"' in queue_src, (
        "PromptQueueStack no longer names itself, so dom.promptQueue() matches nothing and the "
        "queued-idle interval is indistinguishable again"
    )
    # dom.js matches this name by a hard-coded English prefix, so catalog and selector must agree.
    region_label = _en_string("regionLabel")
    prefix = re.search(r'\[aria-label\^="([^"]*)"\]', _DOM_JS.read_text(encoding = "utf-8"))
    assert prefix, "dom.js no longer selects the queue surface by an aria-label prefix"
    assert region_label.startswith(prefix.group(1)), (
        f"dom.js matches the queue surface on {prefix.group(1)!r} but the en catalog names it "
        f"{region_label!r}, so dom.promptQueue() finds nothing and generating() silently "
        "reads a queued-idle thread as streaming"
    )
    assert 'aria-label="Stop queued message"' in src


#: Overlays are walked from `document`, outside `.aui-thread-root`, so they skip the stream.
_MENU = '<div role="menu"><div class="item">Rename</div></div>'
_MENU_CHANGED = '<div role="menu"><div class="item">Rename thread</div></div>'
_SEND_BUTTON = (
    '<button class="aui-composer-send" aria-label="Send message">'
    '<span class="aui-sr-only">Send message</span></button>'
)


def _capture_html(page, html: str) -> dict:
    """`_capture`, for a page built outside the `STATE` dictionaries."""
    page.set_content(html)
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture()")
    assert got.get("parity_attempted") is True, got
    return got


def test_an_overlay_difference_survives_the_blind_probe_refusal(page):
    """Overlay differences survive a blind-probe refusal, since the exit code never reads refused pairs."""
    base = _capture_html(page, _page(tail = "same", overlay = _MENU, **BLIND))
    treat = _capture_html(page, _page(tail = "same", overlay = _MENU_CHANGED, **BLIND))
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    assert len(base["overlays"]) == len(treat["overlays"]) == 1
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert len(got["moved"]) == 1 and got["moved"][0].startswith('overlay0[[role="menu"]]'), got
    assert "walked outside the thread root" in got["reason"]


def test_a_settled_user_row_survives_the_blind_probe_refusal(page):
    """A settled user row is still compared when the probe refuses, since only assistant rows stream."""
    base = _capture_html(page, _page(tail = "same", **BLIND))
    treat = _capture_html(
        page,
        _page(tail = "same", **BLIND).replace(
            "<div>the prompt</div>",
            "<div>the prompt, rendered differently</div>",
            1,
        ),
    )
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg0(user)") for m in got["moved"]), got["moved"]


def test_matching_overlays_still_leave_the_blind_pair_refused(page):
    """The narrowing is not a hole: with nothing independent to say, the refusal stands."""
    base = _capture_html(page, _page(tail = "alpha", overlay = _MENU, **BLIND))
    treat = _capture_html(page, _page(tail = "omega", overlay = _MENU, **BLIND))
    assert base["digest"] != treat["digest"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []


def test_the_scaffold_is_not_an_independent_surface_and_here_is_why(page):
    """The scaffold is not compared, since the composer inside it changes whenever a reply runs."""
    settled = dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)
    generating = dict(settled, control = _STOP_BUTTON)
    a = _capture_html(page, _page(tail = "same", **settled))
    b = _capture_html(page, _page(tail = "same", **generating))
    assert [m["digest"] for m in a["messages"]] == [m["digest"] for m in b["messages"]]
    assert a["digest_scaffold"] != b["digest_scaffold"], (
        "if this ever holds, the composer has left the thread root and the scaffold may be "
        "consulted beside the overlays"
    )


# The composer dock is inside `.aui-thread-root`, so a finished arm shows Send and a writing one Stop.

_SETTLED_STATUSES = ["complete", "complete", "complete", "complete"]
_STREAMING_STATUSES = ["complete", "complete", "complete", "running"]


def _finished(**kw):
    return dict(control = _SEND_BUTTON, queue_stack = False, statuses = _SETTLED_STATUSES, **kw)


def _writing(**kw):
    return dict(control = _STOP_BUTTON, queue_stack = False, statuses = _STREAMING_STATUSES, **kw)


def test_a_scaffold_only_difference_across_a_finished_and_a_running_arm_is_refused(page):
    """A scaffold-only difference between finished and running arms is refused, not a rendering change."""
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat = _capture_html(page, _page(tail = "arr", **_writing()))
    assert base["streaming"] is False and treat["streaming"] is True
    assert treat["in_flight"] == [3] and treat["in_flight_unplaced"] is False
    assert [m["digest"] for m in base["messages"][:3]] == [
        m["digest"] for m in treat["messages"][:3]
    ]
    assert base["digest_scaffold"] != treat["digest_scaffold"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"]
    assert got["moved"] == []


def test_the_same_two_stream_positions_with_both_arms_running_are_unchanged(page):
    """WHY THE NULL COULD NOT SEE IT. One build against itself at two points in one stream: both
    arms render Stop, the scaffolds match, and the bias cancels inside the control."""
    a = _capture_html(page, _page(tail = "arrived at last", **_writing()))
    b = _capture_html(page, _page(tail = "arr", **_writing()))
    assert a["digest_scaffold"] == b["digest_scaffold"]
    assert P.compare(a, b)["verdict"] == P.NOT_COMPARABLE


def test_a_real_message_difference_is_still_reported_across_a_generation_disagreement(page):
    """The withholding is not a blanket. It applies only when the scaffold is the ONLY thing that
    moved; a settled message that differs is reported exactly as before."""
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat_statuses = list(_STREAMING_STATUSES)
    treat = _capture_html(
        page,
        _page(
            tail = "arr",
            control = _STOP_BUTTON,
            queue_stack = False,
            statuses = treat_statuses,
            overlay = "",
        ),
    )
    treat2 = _capture_html(
        page,
        _page(
            tail = "arr",
            control = _STOP_BUTTON,
            queue_stack = False,
            statuses = treat_statuses,
        ).replace("reply 1 ", "reply 1 rewritten "),
    )
    assert treat["messages"][1]["digest"] != treat2["messages"][1]["digest"]
    got = P.compare(base, treat2)
    assert got["verdict"] == P.DIFFER, got
    assert any(m.startswith("msg1(") for m in got["moved"]), got["moved"]


def test_a_scaffold_difference_with_both_arms_agreeing_is_still_a_difference(page):
    """And the coverage this must not cost: when the arms agree about generation, the composer is
    comparable again and a scaffolding change is reported."""
    base = _capture_html(page, _page(tail = "same", **_finished()))
    treat = _capture_html(
        page,
        _page(tail = "same", **_finished()).replace(
            ">typed while it ran<",
            ">a different draft left in the box<",
        ),
    )
    assert base["streaming"] == treat["streaming"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == [
        "thread scaffolding outside any message (%d->%dc)"
        % (base["chars_scaffold"], treat["chars_scaffold"])
    ], got["moved"]


def test_the_blind_branch_reads_the_scaffold_when_the_arms_agree_about_generation(page):
    """Both arms generating, so a scaffolding change is a real finding even with the stream unplaced."""
    base = _capture_html(page, _page(tail = "same", **BLIND))
    treat = _capture_html(
        page,
        _page(tail = "same", **BLIND).replace(
            '<textarea aria-label="Message input">typed while it ran</textarea>',
            '<textarea aria-label="Message input">typed while it ran, differently</textarea>',
        ),
    )
    assert base["in_flight_unplaced"] is True and treat["in_flight_unplaced"] is True
    assert base["streaming"] == treat["streaming"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("thread scaffolding" in m for m in got["moved"]), got["moved"]


#: Treatment dropped its Send button: `runStateControl` returns "" and only the composer differs.
_NO_CONTROL = '<div class="ml-1.5 flex items-center"></div>'


def test_a_composer_regression_between_two_settled_arms_is_reported(page):
    """Generation state must not be read off the composer, or a composer regression excuses itself."""
    base = _capture_html(page, _page(tail = "same", **_finished()))
    treat = _capture_html(
        page,
        _page(tail = "same", **dict(_finished(), control = _NO_CONTROL)),
    )
    assert base["streaming"] is False and treat["streaming"] is False
    assert bool(base["queued_idle"]) is False and bool(treat["queued_idle"]) is False
    assert base["composer_control"] == "Send message" and treat["composer_control"] == ""
    assert P.generation_disagrees(base, treat) is True
    assert base["digest_scaffold"] != treat["digest_scaffold"]
    assert [m["digest"] for m in base["messages"]] == [m["digest"] for m in treat["messages"]]

    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("scaffolding" in m for m in got["moved"]), got["moved"]


def test_a_finished_against_a_running_arm_is_still_refused_after_that(page):
    """And the suppression this must not cost: `streaming` disagrees, so the run state itself says
    the two arms were at different points in one turn and the composer is not evidence of a
    change."""
    base = _capture_html(page, _page(tail = "arrived at last", **_finished()))
    treat = _capture_html(page, _page(tail = "arr", **_writing()))
    assert base["streaming"] is False and treat["streaming"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"]


def test_a_dispatched_queue_wait_is_a_run_state_not_a_rendering_difference(page):
    """A dispatched queue wait looks idle, not running, so it must be refused as run-state timing."""
    dispatched = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _STOP_QUEUED_BUTTON, queue_stack = False)),
    )
    settled = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)),
    )
    assert dispatched["composer_control"] != settled["composer_control"]
    assert bool(dispatched["queued_idle"]) is True, dispatched
    assert bool(settled["queued_idle"]) is False, settled
    assert P._run_state_disagrees(dispatched, settled) is True

    assert P.compare(dispatched, settled)["verdict"] == P.NOT_COMPARABLE
    assert P.compare(settled, dispatched)["verdict"] == P.NOT_COMPARABLE


def test_the_queued_idle_arm_against_a_settled_one_is_still_refused(page):
    """The other legitimate suppression: `isRunning()` cannot separate queued-idle from settled, so
    `queued_idle` is what carries this pair. Without it the Queue button would read as a
    regression."""
    base = _capture_html(page, _page(tail = "same", **QUEUED_IDLE))
    treat = _capture_html(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = _SEND_BUTTON, queue_stack = False)),
    )
    assert base["composer_control"] != treat["composer_control"]
    assert bool(base["queued_idle"]) != bool(treat["queued_idle"]), (
        base["queued_idle"],
        treat["queued_idle"],
    )
    assert P.compare(base, treat)["verdict"] == P.NOT_COMPARABLE


# The style probe walks Send and Stop as separate selectors, so it also sees the composer swap.


def _capture_html_raw(page, html: str) -> dict:
    """`_capture_html`, keeping `styles.sig` so a test can say WHY the digest moved."""
    page.set_content(html)
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    got = page.evaluate("() => window.__sb.parity.capture({ raw: true })")
    assert got.get("parity_attempted") is True, got
    return got


def _style_values(cap: dict) -> list[str]:
    """The probe's readings with the selector NAMES dropped: just the three properties, in order."""
    return [entry.split(":", 1)[1] for entry in cap["styles"]["sig"].split(";") if ":" in entry]


def test_the_style_probe_does_not_report_a_control_swap_as_a_css_regression(page):
    """A control swap is not a CSS change: the style digest includes which selector matched."""
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    writing = _capture_html_raw(page, _page(tail = "same", **_writing()))
    assert settled["styles"]["elements"] == writing["styles"]["elements"]
    assert _style_values(settled) == _style_values(writing), (
        _style_values(settled),
        _style_values(writing),
    )
    assert settled["styles"]["digest"] != writing["styles"]["digest"]
    assert settled["streaming"] is False and writing["streaming"] is True
    verdict, reason = P.compare_styles(settled, writing)
    assert verdict == P.NOT_COMPARABLE, (verdict, reason)
    assert "run-state controls" in reason, reason
    assert P.compare(settled, writing)["style_verdict"] == P.NOT_COMPARABLE


@pytest.mark.parametrize("control", [_QUEUE_BUTTON_IDLE, _STOP_QUEUED_BUTTON])
def test_a_queue_control_missing_from_the_selector_list_is_not_a_css_regression(page, control):
    """A queue control missing from STYLE_SELECTORS shifts the element count, not the CSS."""
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    queued = _capture_html_raw(
        page,
        _page(tail = "same", **dict(QUEUED_IDLE, control = control, queue_stack = False)),
    )
    assert settled["styles"]["elements"] != queued["styles"]["elements"]
    assert bool(settled["queued_idle"]) != bool(queued["queued_idle"]) or (
        settled["streaming"] != queued["streaming"]
    )
    assert P.compare_styles(settled, queued)[0] == P.NOT_COMPARABLE
    assert P.compare(settled, queued)["style_verdict"] == P.NOT_COMPARABLE


def test_a_real_style_regression_between_two_arms_in_one_run_state_is_still_reported(page):
    """THE COVERAGE THIS MUST NOT COST. Same control on both arms, and the treatment hides the
    viewport from CSS alone -- no structural trace whatever, which is the only thing this probe
    exists to see."""
    base = _capture_html_raw(page, _page(tail = "same", **_finished()))
    treat = _capture_html_raw(
        page,
        "<style>.aui-thread-viewport { visibility: hidden }</style>"
        + _page(tail = "same", **_finished()),
    )
    assert base["digest"] == treat["digest"], "the difference must be CSS only"
    assert _style_values(base) != _style_values(treat)
    verdict, reason = P.compare_styles(base, treat)
    assert verdict == P.DIFFER, (verdict, reason)


def test_a_composer_that_lost_its_control_is_still_a_style_finding(page):
    """AND THE SUPPRESSION IS NOT KEYED ON THE COMPOSER TOKEN ALONE. The treatment simply has no
    Send button: the token differs, the probe matches one element fewer, and the run state agrees
    on both independent readings -- so this is a rendering regression and it is reported."""
    base = _capture_html_raw(page, _page(tail = "same", **_finished()))
    treat = _capture_html_raw(
        page,
        _page(tail = "same", **dict(_finished(), control = _NO_CONTROL)),
    )
    assert base["composer_control"] == "Send message" and treat["composer_control"] == ""
    assert P._run_state_disagrees(base, treat) is False
    verdict, reason = P.compare_styles(base, treat)
    assert verdict == P.DIFFER, (verdict, reason)
    assert "different number of elements" in reason


def test_what_the_style_elision_gives_up(page):
    """One aggregate style digest cannot separate a control swap from a CSS regression beside it."""
    settled = _capture_html_raw(page, _page(tail = "same", **_finished()))
    writing_and_broken = _capture_html_raw(
        page,
        "<style>.aui-thread-viewport { visibility: hidden }</style>"
        + _page(tail = "same", **_writing()),
    )
    assert P.compare_styles(settled, writing_and_broken)[0] == P.NOT_COMPARABLE
