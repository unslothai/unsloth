# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When is a thread READY to be measured?

THE OLD GATE COUNTED DOM NODES, and that is the whole problem.
`_wait_for_thread` waited for `[data-role]` to reach the number of messages the seeder had
written, and raised `TimeoutError: the thread mounted 9 of 18 messages in 180s` otherwise. For a
thread that mounts everything, that is a fine proxy: the last thing the app does is mount the last
message, so the count reaching N means the app has finished. For a thread that mounts a WINDOW by
design, the count never reaches N, so the arm that addresses the actual root cause -- the standing
DOM -- could not be scored at all. Raising the timeout would not have helped; the count is not
going to arrive.

WHAT THE GATE IS ACTUALLY FOR. It is not for counting. It is for refusing to start the film while
the app is still building, because a window opened mid-build charges the build to the first action
and reports a flattering number for everything after it. The property worth asserting is therefore
"the app has finished, and what it finished with is the real thread", not "N elements exist".

THE SIGNAL. Four facts, all of which a fully-mounted thread and a correctly virtualised one can
both satisfy, and none of which a half-built one can:

  SETTLED     two samples at least `STABLE_GAP_MS` apart agree on the mounted message count, the
              total element count and the viewport's scrollHeight. A thread still mounting is
              still growing on all three; this is the condition that catches the exact failure the
              gate exists for.
  END PRESENT the LAST message of the seeded thread is mounted, identified by the marker string
              the seeder itself wrote into the last user turn. Mounting runs front to back, so the
              last message is the last thing to appear: a run that has mounted 9 of 18 has mounted
              the first 9 and cannot satisfy this. It is also what a virtualised thread anchored
              at the end of the conversation has by construction.
  TOTAL       every mounted message agrees on `aria-setsize`, and it equals the number of messages
              the seeder wrote. This is the virtualizer's own claim about how long the thread is,
              read from the place the ARIA spec already requires a windowed list to publish it.
              Required in `windowed` mode, recorded but not gated in `full` (where the mounted
              count is the total).
  ANCHORED    the viewport is at the bottom, read from the app's OWN state rather than from
              arithmetic: Studio disables assistant-ui's autoscroll and runs
              `use-intent-aware-autoscroll`, which pins to the bottom on `thread.initialize` and
              hides `.aui-thread-scroll-to-bottom` with `invisible` exactly when it considers
              itself at the bottom. The scrollTop arithmetic is kept as a corroborating reading,
              never as the verdict on its own. Gated in `windowed` mode only, where it is what
              makes the mounted set reproducible between the two arms; in `full` mode every
              message is mounted whatever the scroll position, so gating on it would add a way to
              fail that has nothing to do with readiness.

A NOTE ON WHAT THIS ASKS OF THE ARM. Studio ships no virtualization and no ordinal attributes:
there is no `aria-setsize`, no `aria-posinset` and no `data-message-index` anywhere in the chat
thread today, and `@tanstack/react-virtual` is used only by the hub's model catalog. So TOTAL is
not a signal that exists and is being read; it is a CONTRACT THE VIRTUALIZATION ARM MUST MEET.
That is a requirement, not a favour: WAI-ARIA already requires a list that does not have all of
its items in the DOM to publish `aria-setsize` and `aria-posinset`, so a virtualised thread that
omits them is broken for assistive technology whatever it does to the frame rate. Refusing to
score it is the right answer rather than an inconvenience. `full` mode requires none of this and
runs unchanged against the shipped build.

WHY IT CANNOT PASS EARLY ON A BROKEN ARM. Each of the three interesting breakages is refused by a
different one of those, so no single mistake in an arm can produce a pass:

  still mounting              SETTLED fails (counts still climbing) and END PRESENT fails (the
                              last message has not been reached yet)
  mounts a window but has
  only loaded part of the
  thread into its store       TOTAL fails, because `aria-setsize` is the store's length and it
                              will not equal what the seeder wrote
  mounts a window, claims
  the right total, and has
  really dropped the head     `probe_thread_completeness` fails: it scrolls to the top and
                              requires the FIRST message, by the seeder's own marker, to mount

The residual, stated rather than hidden: an arm that publishes a truthful `aria-setsize`, keeps the
whole thread in its store and materialises rows on demand passes all of it. That is not a hole,
that is a correctly virtualised thread, which is the thing we set out to be able to score.

WHY NOT THE OTHER CANDIDATES.
  the app's own store         there is no supported handle on it from outside. Reaching into a
                              React fibre or a bundler-internal module to read one would make the
                              gate depend on the build's internals, which is exactly what a
                              harness that has to run two different builds in one session cannot
                              afford. `aria-setsize` is the same number, published on purpose.
  scroll-to-end settled       necessary, not sufficient: an empty thread is settled at the bottom.
                              It is one of the four here, not the whole signal.
  the last message's content   used, but via the seeder's marker rather than the corpus text. The
                              corpus goes through a markdown renderer, so its rendered text is not
                              its source text and a tail match on it would be a guess. The marker
                              is plain text this harness wrote itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: The two gate modes. `full` is the historical behaviour plus the settle and end-present
#: conditions; it is STRICTLY STRONGER than what shipped and is what every normal arm runs.
#: `windowed` is the one an arm that deliberately mounts fewer nodes may ask for.
MODE_FULL = "full"
MODE_WINDOWED = "windowed"
MODES = (MODE_FULL, MODE_WINDOWED)

#: How far apart the two agreeing samples must be. 600ms rather than the old 500ms poll, because
#: the thing being ruled out is a mount loop that pauses for a frame, and a gap shorter than a
#: couple of React commits can be spanned by one.
STABLE_GAP_MS = 600
#: How many consecutive agreeing samples are needed. Two, i.e. one confirmed repeat.
STABLE_SAMPLES = 2
#: Bottom tolerance, in CSS pixels. Not zero: a virtualised list settles on an estimated total
#: height and corrects it as real rows are measured, so it lands a few pixels off the exact bottom
#: and stays there. 24px is under one line of text, so it cannot hide a missing message.
BOTTOM_TOLERANCE_PX = 24

DEFAULT_TIMEOUT_S = 180


class ThreadNotReady(TimeoutError):
    """The gate refused. A TimeoutError subclass so existing `except TimeoutError` still catches it.

    Carries the last probe reading on `.detail`, because "the thread mounted 9 of 18 messages"
    told you the count and nothing about which of the four conditions was the one that failed.
    """

    def __init__(self, message: str, detail: dict):
        super().__init__(message)
        self.detail = detail


#: One reading of the page. Cheap enough to run every poll: the expensive part is
#: `getElementsByTagName('*').length`, which is a live HTMLCollection length and not a walk.
PROBE_JS = """
(args) => {
  const [marker, tailChars] = args;
  const D = (window.__sb && window.__sb.dom) || null;
  if (!D) return { probe_attempted: false, reason: "window.__sb.dom is not installed" };
  const vp = D.viewport();
  const roles = Array.from(document.querySelectorAll("[data-role]"));
  const setsizes = [];
  let maxPos = null;
  let withPos = 0;
  // ON THE MESSAGE OR ON THE ROW THAT HOLDS IT. A virtualizer positions each row in a wrapper of
  // its own -- that is how absolute positioning against a measured total height works -- and the
  // ordinal belongs on the row, which is the element that is actually a member of the set. The
  // `[data-role]` message sits inside it. Looking only at `[data-role]` would refuse a correctly
  // implemented arm for putting the attribute in the right place.
  const ordinal = (el, name) => {
    const owner = el.closest("[" + name + "]");
    if (!owner) return null;
    const raw = owner.getAttribute(name);
    if (raw === null || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  for (const el of roles) {
    const ss = ordinal(el, "aria-setsize");
    if (ss !== null) setsizes.push(ss);
    const pi = ordinal(el, "aria-posinset");
    if (pi !== null) {
      withPos += 1;
      if (maxPos === null || pi > maxPos) maxPos = pi;
    }
  }
  const distinct = Array.from(new Set(setsizes));
  // The app's own opinion of "at the bottom". thread.tsx renders the scroll-to-bottom control
  // permanently and hides it with `invisible` when use-intent-aware-autoscroll reports at-bottom,
  // so this is the state the app is acting on rather than a number the driver computed.
  const jump = document.querySelector(".aui-thread-scroll-to-bottom");
  // Set on the viewport WHILE the autoscroll is pinning and removed when it settles. A page that
  // still carries it is mid-pin, whatever its scrollTop currently reads.
  const stabilizer = vp
    ? (vp.style.getPropertyValue("--aui-scroll-stabilizer") || "").trim()
    : "";
  const last = roles.length ? roles[roles.length - 1] : null;
  // The marker is looked for among the MOUNTED user messages only. Searching document.body would
  // find it in a sidebar preview or a title and call the thread ready on the strength of a
  // tooltip.
  let markerFound = false;
  let markerIndex = null;
  if (marker) {
    for (let i = 0; i < roles.length; i += 1) {
      const el = roles[i];
      if (el.getAttribute("data-role") !== "user") continue;
      if ((el.textContent || "").includes(marker)) { markerFound = true; markerIndex = i; }
    }
  }
  return {
    probe_attempted: true,
    mounted: roles.length,
    elements: document.getElementsByTagName("*").length,
    composer: Boolean(D.composer()),
    running: D.isRunning(),
    setsize: distinct.length === 1 ? distinct[0] : null,
    setsize_values: distinct,
    posinset_count: withPos,
    max_posinset: maxPos,
    marker_found: markerFound,
    // Where the marker sits in the mounted run. The last message of the thread is the assistant
    // reply to the last user turn, so a marker found anywhere but the final couple of rows means
    // the mounted window is not at the end even if the viewport says it is.
    marker_from_end: markerIndex === null ? null : roles.length - 1 - markerIndex,
    last_role: last ? last.getAttribute("data-role") : null,
    last_tail: last ? (last.textContent || "").replace(/\\s+/g, " ").trim().slice(-tailChars) : null,
    scroll_height: vp ? vp.scrollHeight : null,
    client_height: vp ? vp.clientHeight : null,
    scroll_top: vp ? vp.scrollTop : null,
    from_bottom: vp ? Math.round(vp.scrollHeight - vp.clientHeight - vp.scrollTop) : null,
    viewport_present: Boolean(vp),
    jump_button_present: Boolean(jump),
    app_says_at_bottom: jump ? jump.classList.contains("invisible") : null,
    pinning: stabilizer !== "",
  };
}
"""

#: The keys that must AGREE between two samples for the page to count as settled. Deliberately
#: three quantities that move for three different reasons: the message list growing, any subtree
#: growing (a highlighter still colouring a fence moves `elements` while `mounted` sits still), and
#: the laid-out height changing (a virtualised list correcting its estimated row heights).
SETTLE_KEYS = ("mounted", "elements", "scroll_height")


@dataclass
class Readiness:
    """The gate's verdict, recorded whether it passed or failed."""

    ready: bool
    mode: str
    expected_messages: int
    waited_ms: float
    conditions: dict = field(default_factory = dict)
    probe: dict = field(default_factory = dict)
    samples: int = 0
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "mode": self.mode,
            "expected_messages": self.expected_messages,
            "waited_ms": round(self.waited_ms, 1),
            "samples": self.samples,
            "conditions": self.conditions,
            "probe": self.probe,
            "reason": self.reason,
        }

    @property
    def failed(self) -> list[str]:
        return sorted(k for k, v in self.conditions.items() if v is False)


def evaluate(probe: dict, previous: Optional[dict], expected_messages: int, mode: str) -> dict:
    """The whole decision, as a pure function of two probe readings. Tested without a browser.

    Returns `{condition: True | False | None}`. `None` means NOT APPLICABLE IN THIS MODE and is
    never treated as a pass or a fail -- the same distinction the parity layer draws between a
    match and a surface that was never measured.
    """
    if not probe.get("probe_attempted"):
        return {"probe": False}

    settled: Any = False
    if previous is not None and previous.get("probe_attempted"):
        settled = all(probe.get(k) == previous.get(k) for k in SETTLE_KEYS)

    out: dict[str, Any] = {
        "composer_present": bool(probe.get("composer")),
        "any_message_mounted": (probe.get("mounted") or 0) > 0,
        "settled": bool(settled),
        # The END of the thread is on screen. Both halves matter: the marker for the last user turn
        # is mounted, AND it is near the end of the mounted run rather than in the middle of it.
        # `2` because the thread ends user, assistant -- the marker's own row plus the reply.
        "end_present": bool(probe.get("marker_found"))
        and (probe.get("marker_from_end") is not None and probe["marker_from_end"] <= 2),
    }

    if mode == MODE_WINDOWED:
        # THE VIRTUALIZER'S OWN CLAIM about the thread's length, and it has to be the truth the
        # seeder wrote. An arm that mounts a window without publishing aria-setsize has not given
        # anyone -- this harness or a screen reader -- any way to know how long the thread is, and
        # it is refused here rather than scored on the assumption that it is fine.
        #
        # UNLESS THE WHOLE THREAD IS MOUNTED, in which case there is nothing left to declare: the
        # DOM is the total. This matters at the small rungs, where the thread is shorter than the
        # window and a virtualised build mounts every message exactly like the shipped one. The
        # requirement has to be true of a fully-mounted thread as well as of a correctly windowed
        # one, or `windowed` becomes a mode that only some rungs of the same arm can pass.
        #
        # It cannot be used to slip past the gate: this branch needs `mounted >= expected`, which
        # is the full-mount condition itself, so a thread that has mounted 9 of 18 gets none of it.
        fully_mounted = (probe.get("mounted") or 0) >= expected_messages
        out["total_declared"] = fully_mounted or probe.get("setsize") is not None
        out["total_matches_seeded"] = fully_mounted or probe.get("setsize") == expected_messages
        out["posinset_on_every_row"] = fully_mounted or (
            probe.get("posinset_count") == probe.get("mounted") and (probe.get("mounted") or 0) > 0
        )
        # THE APP'S ANSWER FIRST, the arithmetic only when the app has not given one. A virtualised
        # list settles on an estimated total height and corrects it as real rows are measured, so
        # its scrollTop arithmetic can read tens of pixels off the bottom while the app -- which
        # knows it is pinned -- is perfectly happy. Trusting the arithmetic over the app would fail
        # a correct arm for a rounding error in a height estimate.
        app_bottom = probe.get("app_says_at_bottom")
        near_bottom = (
            probe.get("from_bottom") is not None and probe["from_bottom"] <= BOTTOM_TOLERANCE_PX
        )
        out["anchored_at_end"] = bool(app_bottom) if app_bottom is not None else near_bottom
        # And the pin must have FINISHED. `--aui-scroll-stabilizer` is on the viewport while
        # use-intent-aware-autoscroll is still pinning and comes off when it settles, so a page
        # carrying it is mid-scroll no matter what its scrollTop says.
        out["pin_settled"] = not probe.get("pinning")
        # Not a gate, a sanity check with teeth: a "windowed" arm that mounted MORE rows than the
        # thread has is not windowed, it is broken, and the reading would be nonsense.
        out["not_over_mounted"] = (probe.get("mounted") or 0) <= expected_messages
    else:
        out["all_messages_mounted"] = (probe.get("mounted") or 0) >= expected_messages
        # Recorded, never gated, in full mode. See the module docstring.
        out["total_declared"] = None
        out["total_matches_seeded"] = None
        out["posinset_on_every_row"] = None
        out["anchored_at_end"] = None
        out["pin_settled"] = None
        out["not_over_mounted"] = None
    return out


def _describe(conditions: dict, probe: dict, expected: int, mode: str) -> str:
    failed = sorted(k for k, v in conditions.items() if v is False)
    bits = [
        f"mounted {probe.get('mounted')} of {expected}",
        f"aria-setsize {probe.get('setsize')}",
        f"last row {probe.get('last_role')!r}",
        f"{probe.get('from_bottom')}px from the bottom",
    ]
    return (
        f"the thread was not ready in {mode} mode: {', '.join(failed) or 'no condition passed'} "
        f"({'; '.join(bits)})"
    )


def wait_for_thread_ready(
    page,
    expected_messages: int,
    *,
    marker: Optional[str] = None,
    mode: str = MODE_FULL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log: Callable[[str], None] = print,
    tail_chars: int = 80,
) -> Readiness:
    """Block until the thread is ready, or raise `ThreadNotReady` saying which condition failed."""
    if mode not in MODES:
        raise ValueError(f"unknown readiness mode {mode!r}; known modes are {list(MODES)}")
    if expected_messages <= 0:
        page.wait_for_selector('textarea[aria-label="Message input"]', timeout = 60_000)
        return Readiness(
            ready = True,
            mode = mode,
            expected_messages = expected_messages,
            waited_ms = 0.0,
            conditions = {"empty_thread": True},
            reason = "the thread has no seeded messages, so only the composer is required",
        )

    started = time.monotonic()
    deadline = started + timeout_s
    previous: Optional[dict] = None
    probe: dict = {}
    conditions: dict = {}
    agreeing = 0
    samples = 0
    last_log: Optional[tuple] = None
    while time.monotonic() < deadline:
        probe = page.evaluate(PROBE_JS, [marker or "", tail_chars]) or {}
        samples += 1
        conditions = evaluate(probe, previous, expected_messages, mode)
        agreeing = agreeing + 1 if conditions.get("settled") else 0
        if all(v is not False for v in conditions.values()) and agreeing >= STABLE_SAMPLES - 1:
            waited = (time.monotonic() - started) * 1000
            log(
                f"  thread ready ({mode}): {probe.get('mounted')} of {expected_messages} messages "
                f"mounted, aria-setsize {probe.get('setsize')}, "
                f"{probe.get('elements'):,} elements, settled after {waited / 1000:.1f}s"
            )
            return Readiness(
                ready = True,
                mode = mode,
                expected_messages = expected_messages,
                waited_ms = waited,
                conditions = conditions,
                probe = probe,
                samples = samples,
            )
        key = (probe.get("mounted"), probe.get("setsize"), tuple(sorted(conditions.items())))
        if key != last_log:
            last_log = key
            failed = sorted(k for k, v in conditions.items() if v is False)
            log(
                f"  waiting: {probe.get('mounted')}/{expected_messages} mounted, "
                f"outstanding {failed}"
            )
        previous = probe
        page.wait_for_timeout(STABLE_GAP_MS)

    waited = (time.monotonic() - started) * 1000
    detail = Readiness(
        ready = False,
        mode = mode,
        expected_messages = expected_messages,
        waited_ms = waited,
        conditions = conditions,
        probe = probe,
        samples = samples,
        reason = _describe(conditions, probe, expected_messages, mode),
    )
    raise ThreadNotReady(detail.reason or "the thread was not ready", detail.as_dict())


#: Scroll to the very top, then back. Used by `probe_thread_completeness`.
TRAVERSE_JS = """
async (toTop) => {
  const D = window.__sb.dom;
  const vp = D.viewport();
  if (!vp) return { ran: false, reason: "no thread viewport" };
  vp.scrollTo({ top: toTop ? 0 : vp.scrollHeight, behavior: "instant" });
  await window.__sbNextPaint();
  return { ran: true, scroll_top: vp.scrollTop, scroll_height: vp.scrollHeight };
}
"""


def probe_thread_completeness(
    page,
    *,
    first_marker: str,
    expected_messages: int,
    timeout_s: float = 60.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Does the thread really CONTAIN the whole conversation, not just show the end of it?

    THE ONE QUESTION THE READINESS GATE CANNOT ANSWER FROM THE END OF THE THREAD. A windowed arm
    anchored at the bottom looks identical whether its store holds all N messages or only the last
    handful, and the difference is data loss. So this drives the only probe a user has: scroll to
    the top and see whether the first message of the conversation arrives.

    Run BEFORE the measured window, never inside it -- it scrolls the viewport the whole length of
    the thread and mounts whatever the arm materialises on the way, which is real work that has no
    business inside anybody's frame rate. The caller re-establishes the resting state afterwards.

    Reported, not raised. A failure here does not mean the readiness gate was wrong; it means the
    arm is losing messages, which is a finding to record against the arm rather than a reason to
    lose the cell.
    """
    out: dict = {"probe_attempted": True, "expected_messages": expected_messages}
    top = page.evaluate(TRAVERSE_JS, True)
    if not isinstance(top, dict) or not top.get("ran"):
        return {
            "probe_attempted": False,
            "reason": (top or {}).get("reason", "the viewport could not be scrolled"),
        }
    deadline = time.monotonic() + timeout_s
    found = False
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = page.evaluate(PROBE_JS, [first_marker or "", 80]) or {}
        if seen.get("marker_found"):
            found = True
            break
        page.wait_for_timeout(250)
    out.update(
        {
            "head_reached": found,
            "min_posinset_seen": 1 if seen.get("marker_found") else None,
            "mounted_at_top": seen.get("mounted"),
            "setsize_at_top": seen.get("setsize"),
            "scroll_height_at_top": top.get("scroll_height"),
        }
    )
    # And back to the end, so the cell resumes from the state the readiness gate described.
    page.evaluate(TRAVERSE_JS, False)
    if not found:
        out["reason"] = (
            "scrolling to the top of the thread never mounted the first message, so the arm is "
            "not holding the whole conversation"
        )
        log(f"  COMPLETENESS FAILED: {out['reason']}")
    else:
        log(
            f"  completeness: the head of the thread mounted on scroll-to-top "
            f"({seen.get('mounted')} rows mounted there, aria-setsize {seen.get('setsize')})"
        )
    return out
