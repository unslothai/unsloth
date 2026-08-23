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

THE SIGNAL. Five facts, all of which a fully-mounted thread and a correctly virtualised one can
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
  ORDINALS    every mounted row publishes an `aria-posinset` that is ACTUALLY A POSITION: at least
              1, unique across the mounted rows, no larger than the set size those same rows
              declare, and -- for a window anchored at the bottom -- reaching the seeded total.
              Presence alone is worth nothing: rows that all publish `0`, all publish the same
              ordinal, or number a bottom-anchored window from 1 satisfy "the attribute is there"
              while telling this gate and a screen reader nothing about where in the thread the
              window sits. Gated in `windowed` mode only, and waived for a thread that is fully
              mounted and publishes no ordinals at all, which is what the shipped build does.
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

WHY IT CANNOT PASS EARLY ON A BROKEN ARM. Each of the interesting breakages is refused by a
different one of those, so no single mistake in an arm can produce a pass:

  still mounting              SETTLED fails (counts still climbing) and END PRESENT fails (the
                              last message has not been reached yet)
  mounts a window but has
  only loaded part of the
  thread into its store       TOTAL fails, because `aria-setsize` is the store's length and it
                              will not equal what the seeder wrote
  publishes ordinals that
  are not positions -- all
  zero, all identical, or
  a bottom window numbered
  from 1                      ORDINALS fails. Each of those is a real virtualizer bug (publishing
                              the index WITHIN the window rather than the position in the thread
                              is the common one) and each one makes the window unlocatable
  mounts a window, claims
  the right total, and has
  really dropped the head     `probe_thread_completeness` fails: it scrolls to the top and
                              requires the FIRST message, by the seeder's own marker, to mount
  keeps the first and last
  pages and loses messages
  from the MIDDLE             `probe_thread_completeness` fails on COVERAGE: the head marker
                              arrives, so the marker check alone is satisfied, and the ordinals
                              recorded on the way up have a hole in them that no scroll position
                              fills

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
  let minPos = null;
  let withPos = 0;
  // THE DISTINCT ordinals, not just how many were published. A window whose rows all carry the
  // same number publishes one on every row, and counting rows cannot tell that apart from a
  // correctly numbered window. The set can.
  const posSeen = new Set();
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
      posSeen.add(pi);
      if (maxPos === null || pi > maxPos) maxPos = pi;
      if (minPos === null || pi < minPos) minPos = pi;
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
    posinset_distinct: posSeen.size,
    max_posinset: maxPos,
    min_posinset: minPos,
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
        published = probe.get("posinset_count") or 0
        out["posinset_on_every_row"] = fully_mounted or (
            probe.get("posinset_count") == probe.get("mounted") and (probe.get("mounted") or 0) > 0
        )
        # ORDINALS THAT ARE ACTUALLY POSITIONS. The condition above only asks whether a number was
        # published on every row, and three malformed shapes satisfy it while carrying no
        # information at all: every row publishing `0`, every row publishing the same ordinal, and
        # a window numbered from 1 rather than from where it sits in the thread. Each is a real
        # virtualizer bug, each leaves the mounted set unlocatable, and each used to pass here.
        #
        # So the numbers have to behave like positions in a set: at least 1, no two rows claiming
        # the same one, and none pointing past the end those same rows declare in `aria-setsize`.
        # `setsize` is the arm's own claim and is already required to equal the seeded total by
        # `total_matches_seeded`; the seeded total stands in when no setsize was published, which
        # is the fully-mounted case below.
        #
        # THE WAIVER, and it is the same one `total_declared` carries: a thread short enough to be
        # mounted whole publishes no ordinals -- the shipped build publishes none anywhere -- and
        # there is nothing to validate. It needs `mounted >= expected`, the full-mount condition
        # itself, AND no ordinals at all, so an arm that publishes malformed ones cannot buy its
        # way out of the check by also mounting everything.
        declared = probe.get("setsize")
        if declared is None:
            declared = expected_messages
        out["posinset_ordinals_valid"] = (fully_mounted and published == 0) or (
            published > 0
            and probe.get("min_posinset") is not None
            and probe["min_posinset"] >= 1
            and probe.get("posinset_distinct") == published
            and probe.get("max_posinset") is not None
            and probe["max_posinset"] <= declared
        )
        # AND THE WINDOW IS AT THE END OF THE THREAD BY ITS OWN NUMBERING. `end_present` proves the
        # last message's TEXT is mounted; this proves the ordinals agree with it. Without it a
        # bottom-anchored window numbered 1..6 out of 18 -- a virtualizer publishing the index
        # within the window instead of the position in the thread -- passes every other condition.
        out["posinset_reaches_end"] = (fully_mounted and published == 0) or (
            probe.get("max_posinset") == expected_messages
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
        # THE VIEWPORT ITSELF, ASSERTED RATHER THAN INFERRED. Every windowed condition above is
        # about what is inside the scroller, and each one degrades to a pass when the scroller is
        # not there: `from_bottom` is null so the arithmetic is skipped, and the app's own answer
        # is read off `.aui-thread-scroll-to-bottom`, which is a DESCENDANT of the viewport but is
        # looked up at DOCUMENT scope. So renaming the viewport class while leaving that control
        # mounted leaves `app_says_at_bottom` true, `anchored_at_end` true, and the cell admitted
        # with no viewport at all.
        #
        # What follows a cell admitted that way is silent in every direction: the completeness
        # probe returns `probe_attempted: false`, the scroll actions return `not_run` and a not-run
        # action blanks only its own timings, and the census viewport fields go null. Nothing
        # refuses, so the film is scored without the surface it was measuring.
        #
        # The signal is already probed and costs nothing to read. It is asserted only here because
        # a fully mounted arm that scrolls the window instead of a div is a shape this harness
        # supports, and `MODE_FULL` must not start requiring a scroller it never needed.
        out["viewport_present"] = bool(probe.get("viewport_present"))
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
        out["posinset_ordinals_valid"] = None
        out["posinset_reaches_end"] = None
        out["anchored_at_end"] = None
        out["pin_settled"] = None
        out["not_over_mounted"] = None
    return out


def _describe(conditions: dict, probe: dict, expected: int, mode: str) -> str:
    failed = sorted(k for k, v in conditions.items() if v is False)
    bits = [
        f"mounted {probe.get('mounted')} of {expected}",
        f"aria-setsize {probe.get('setsize')}",
        # The ordinals THEMSELVES, not just how many rows carried one: "posinset 0..0 on 6 rows,
        # 1 distinct" is the whole diagnosis of a malformed contract, and reading it off the raw
        # probe afterwards is what this message exists to save.
        f"aria-posinset {probe.get('min_posinset')}..{probe.get('max_posinset')} on "
        f"{probe.get('posinset_count')} of {probe.get('mounted')} rows, "
        f"{probe.get('posinset_distinct')} distinct",
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
async ([toTop, steps, stepPx]) => {
  const D = window.__sb.dom;
  const vp = D.viewport();
  if (!vp) return { ran: false, reason: "no thread viewport" };
  // STEPPED, AND WITH A WHEEL EVENT ON EVERY STEP. A single `scrollTo({top: 0})` from the bottom
  // does not work on this app and the reason is already documented in scene/actions.py: Studio
  // replaces assistant-ui's autoscroll with an intent-aware one that reads a jump as programmatic
  // and snaps it straight back to the bottom. The first version of this probe did exactly that,
  // and reported "scrolling to the top never mounted the first message" -- which reads as the arm
  // losing its history and was the probe never leaving the bottom of the thread.
  //
  // The wheel event is what the app's own listeners key off, so it registers as user intent; the
  // scrollTo is what actually moves the viewport in a headless run with no compositor input.
  // Both, in steps, is the same gesture scene/actions.py SCROLL_JS uses for the same reason.
  //
  // A virtualizer also has to be given time to materialise rows as they come into range: each
  // step awaits a paint, so a windowed list gets a frame per step to mount what the step exposed.
  const target = toTop ? 0 : vp.scrollHeight;
  const direction = toTop ? -1 : 1;
  // WHICH MESSAGES THE TRAVERSAL ACTUALLY SAW, ordinal by ordinal.
  //
  // The marker check at the end of the walk asks one question -- did the FIRST message arrive --
  // and a store that kept the first page and the last one and lost everything between them
  // answers it correctly. So every stop records the `aria-posinset` of the rows mounted there,
  // read the same way PROBE_JS reads them (off the row that owns the attribute, which is the
  // positioned wrapper on a real virtualizer), and the union is what the coverage verdict is
  // computed from.
  //
  // `holes` is the stronger of the two readings: a virtualizer mounts a CONTIGUOUS run of the
  // thread, so an ordinal missing from between the smallest and largest mounted at ONE stop was
  // not skipped by this gesture, it is a message the store no longer has. `ranges` records the
  // span each stop mounted, so the caller can tell a continuous sweep -- where every consecutive
  // stop overlapped the last, and an ordinal never seen really was never mounted -- from a
  // coarse one, where the ordinals between two stops were never in view and nothing is known
  // about them.
  const seen = new Set();
  const holes = new Set();
  const ranges = [];
  const record = () => {
    const vals = [];
    for (const el of document.querySelectorAll("[data-role]")) {
      const owner = el.closest("[aria-posinset]");
      if (!owner) continue;
      const n = Number(owner.getAttribute("aria-posinset"));
      if (Number.isFinite(n)) vals.push(n);
    }
    if (!vals.length) return;
    let lo = vals[0];
    let hi = vals[0];
    const here = new Set();
    for (const n of vals) {
      seen.add(n);
      here.add(n);
      if (n < lo) lo = n;
      if (n > hi) hi = n;
    }
    // Bounded, so a thread whose ordinals are nonsense cannot put a million entries in a payload.
    if (holes.size < 1000) {
      for (let k = lo; k <= hi; k += 1) if (!here.has(k)) holes.add(k);
    }
    ranges.push([lo, hi]);
  };
  record();
  for (let i = 0; i < steps; i += 1) {
    const next = toTop
      ? Math.max(0, vp.scrollTop - stepPx)
      : Math.min(vp.scrollHeight, vp.scrollTop + stepPx);
    vp.dispatchEvent(
      new WheelEvent("wheel", { deltaY: direction * stepPx, bubbles: true, cancelable: true }),
    );
    vp.scrollTo({ top: next, behavior: "instant" });
    await window.__sbNextPaint();
    record();
    if (toTop && vp.scrollTop <= 0) break;
    if (!toTop && vp.scrollTop >= vp.scrollHeight - vp.clientHeight - 1) break;
  }
  let continuous = ranges.length > 0;
  for (let i = 1; i < ranges.length; i += 1) {
    const a = ranges[i - 1];
    const b = ranges[i];
    // Touching counts as continuous: rows 1-6 followed by rows 7-12 leaves nothing between them.
    if (b[0] > a[1] + 1 || a[0] > b[1] + 1) { continuous = false; break; }
  }
  const sorted = Array.from(seen).sort((a, b) => a - b);
  return {
    ran: true,
    scroll_top: vp.scrollTop,
    scroll_height: vp.scrollHeight,
    reached_target: toTop ? vp.scrollTop <= 2 : true,
    target,
    ordinals_seen: sorted,
    ordinals_in_window_holes: Array.from(holes).sort((a, b) => a - b),
    sweep_continuous: continuous,
    traversal_stops: ranges.length,
  };
}
"""

#: How the traversal is stepped. 400 steps of 2,000px covers 800,000px of thread, which clears the
#: tallest rung this benchmark runs, and a step that lands at either end breaks out early.
TRAVERSE_STEPS = 400
TRAVERSE_STEP_PX = 2000

#: The ordinals mounted right now, read exactly the way PROBE_JS reads them. Run once at the top of
#: the thread, after the wait for the head marker: the last stop of the gesture reads its rows the
#: instant the paint lands, and an arm that materialises a row a beat later would otherwise have
#: that row counted as one the traversal never saw.
COLLECT_ORDINALS_JS = """
() => {
  const out = [];
  for (const el of document.querySelectorAll("[data-role]")) {
    const owner = el.closest("[aria-posinset]");
    if (!owner) continue;
    const n = Number(owner.getAttribute("aria-posinset"));
    if (Number.isFinite(n)) out.push(n);
  }
  return out;
}
"""

#: How many missing ordinals a verdict names. The COUNT is always exact; the list is capped so a
#: thread that lost half of itself does not write a thousand integers into every payload.
MISSING_ORDINALS_LISTED = 40

#: WHY `ordinal_coverage` GAVE THE ANSWER IT GAVE, which the three-valued verdict on its own cannot
#: say and which the `thread_complete` gate has to know.
#:
#: `not_applicable` and `unmeasured` are BOTH `None` and they are opposites. The first is a question
#: that does not arise on this arm: a fully mounted thread publishes no `aria-posinset` anywhere --
#: the shipped build publishes none -- so there are no ordinals to cover and none missing. The
#: second is a question that does arise and was not answered: the gesture stopped short of the top,
#: or its consecutive stops did not overlap, so rows nobody scrolled past are rows nobody looked at.
#:
#: A gate that passes both reads "we could not tell" as "it was fine", which is the store that kept
#: only its first and last page staying scoreable. A gate that fails both fails the shipped build on
#: every cell for publishing no ordinals. So the two are recorded apart, and only the second is
#: withheld from a pass.
COVERAGE_COMPLETE = "complete"
COVERAGE_INCOMPLETE = "incomplete"
COVERAGE_NOT_APPLICABLE = "not_applicable"
COVERAGE_UNMEASURED = "unmeasured"

#: The coverage states a cell may still be scored on. `unmeasured` is deliberately absent, and
#: `incomplete` is excluded by the verdict itself rather than by this tuple.
COVERAGE_STATES_SCOREABLE = (COVERAGE_COMPLETE, COVERAGE_NOT_APPLICABLE)


def ordinal_coverage(
    traverse: dict,
    expected_messages: int,
    extra_seen: Any = (),
) -> dict:
    """Which of the seeded ordinals the traversal actually SAW, and what that does and does not
    prove. A pure function of the traversal's own record, so it is tested without a browser.

    THREE-VALUED, for the same reason `head_reached` is. `True` is coverage, `False` is a message
    the arm no longer has, and `None` is this probe not having looked -- which is the one answer
    the old code could not give and the reason a coarse gesture must never be allowed to report a
    row it never scrolled past as a row the app lost.

    AND `ordinal_coverage_state` SAYS WHICH KIND OF `None` IT IS, because the two are opposites and
    a caller that cannot tell them apart has to treat them the same. `not_applicable` is a question
    that does not arise -- no row published an ordinal, which is what a fully mounted arm does by
    design -- and `unmeasured` is a question that arises and was not answered. The `thread_complete`
    gate passes the first and refuses the second; collapsing them made a store that kept only its
    first and last page scoreable whenever the sweep happened to be coarse.

    The three ways `None` is the honest answer:

      no row published an ordinal           NOT APPLICABLE. A fully mounted arm publishes none at
                                            all, by design, and MODE_FULL never asked it to.
                                            Checked FIRST: a windowed arm records the ordinals of
                                            its mounted rows before the gesture takes its first
                                            step, so an empty union really does mean an arm that
                                            publishes nothing rather than a gesture that failed
      the gesture never reached the top     UNMEASURED. Nothing was traversed, so nothing was
                                            covered
      consecutive stops did not overlap     UNMEASURED. The gesture jumps `TRAVERSE_STEP_PX` at a
                                            time and a virtualizer mounts what is near the
                                            viewport, so rows between two stops were never in view.
                                            The remedy is a finer step, not a softer verdict

    And the two that are a verdict about the arm:

      a hole inside one mounted window      a virtualizer mounts a CONTIGUOUS run, so an ordinal
                                            missing from between the smallest and the largest
                                            mounted at a single stop -- and mounted at no other
                                            stop either -- is a message the store does not have
      a continuous sweep with a gap         every stop overlapped the last, so the union really is
                                            everything the thread could show, and what is not in
                                            it is not in the thread
    """
    seen = {int(n) for n in (traverse.get("ordinals_seen") or [])}
    seen.update(int(n) for n in (extra_seen or ()))
    expected = set(range(1, expected_messages + 1)) if expected_messages > 0 else set()
    missing = sorted(expected - seen)
    # Intersected with "never seen anywhere", so a row that was briefly absent while its window
    # was still materialising and mounted at the next stop is not reported as a lost message.
    holes = sorted({int(n) for n in (traverse.get("ordinals_in_window_holes") or [])} - seen)
    out: dict[str, Any] = {
        "ordinals_seen_count": len(seen),
        "min_posinset_seen": min(seen) if seen else None,
        "max_posinset_seen": max(seen) if seen else None,
        "ordinals_missing_count": len(missing),
        "ordinals_missing": missing[:MISSING_ORDINALS_LISTED],
        "ordinals_missing_truncated": len(missing) > MISSING_ORDINALS_LISTED,
        "ordinals_in_window_holes": holes[:MISSING_ORDINALS_LISTED],
        "sweep_continuous": traverse.get("sweep_continuous"),
        "traversal_stops": traverse.get("traversal_stops"),
        "coverage_reason": None,
    }
    # NOT APPLICABLE BEFORE UNMEASURED. An arm publishing no ordinals anywhere has nothing to cover
    # whatever the gesture did, and asking about the gesture first would report the shipped build's
    # own shape as a measurement this probe failed to take -- which the gate now refuses to score.
    if not seen:
        out["ordinal_coverage_complete"] = None
        out["ordinal_coverage_state"] = COVERAGE_NOT_APPLICABLE
        out["coverage_reason"] = (
            "no mounted row published aria-posinset during the traversal, so there was nothing to "
            "count. A fully mounted arm publishes none by design, and a windowed one is refused "
            "by the readiness gate long before it reaches this probe"
        )
        return out
    if not traverse.get("reached_target"):
        out["ordinal_coverage_complete"] = None
        out["ordinal_coverage_state"] = COVERAGE_UNMEASURED
        out["coverage_reason"] = (
            "the stepped gesture never reached the top, so an ordinal it did not see is an "
            "ordinal it never looked for"
        )
        return out
    if holes:
        out["ordinal_coverage_complete"] = False
        out["ordinal_coverage_state"] = COVERAGE_INCOMPLETE
        out["coverage_reason"] = (
            f"{len(holes)} ordinal(s) were absent from a mounted window that spanned them and "
            f"appeared at no other scroll position ({out['ordinals_in_window_holes']}), so the "
            "arm is missing messages from the MIDDLE of the thread"
        )
        return out
    if not missing:
        out["ordinal_coverage_complete"] = True
        out["ordinal_coverage_state"] = COVERAGE_COMPLETE
        return out
    if traverse.get("sweep_continuous"):
        out["ordinal_coverage_complete"] = False
        out["ordinal_coverage_state"] = COVERAGE_INCOMPLETE
        out["coverage_reason"] = (
            f"{len(missing)} of {expected_messages} ordinals never mounted ({out['ordinals_missing']}"
            f"{', truncated' if out['ordinals_missing_truncated'] else ''}), and every stop of the "
            "traversal overlapped the one before it, so the sweep had no gap for them to hide in"
        )
        return out
    out["ordinal_coverage_complete"] = None
    out["ordinal_coverage_state"] = COVERAGE_UNMEASURED
    out["coverage_reason"] = (
        f"{len(missing)} of {expected_messages} ordinals never mounted, but consecutive stops of "
        f"the gesture did not overlap, so rows between two stops were never in view. NOT a claim "
        "that the arm lost them, and NOT a coverage result either: run the traversal at a step "
        "small enough for consecutive stops to overlap"
    )
    return out


def probe_thread_completeness(
    page,
    *,
    first_marker: str,
    expected_messages: int,
    timeout_s: float = 60.0,
    log: Callable[[str], None] = print,
    steps: int = TRAVERSE_STEPS,
    step_px: int = TRAVERSE_STEP_PX,
) -> dict:
    """Does the thread really CONTAIN the whole conversation, not just show the end of it?

    THE ONE QUESTION THE READINESS GATE CANNOT ANSWER FROM THE END OF THE THREAD. A windowed arm
    anchored at the bottom looks identical whether its store holds all N messages or only the last
    handful, and the difference is data loss. So this drives the only probe a user has: scroll to
    the top and see whether the first message of the conversation arrives.

    AND THE MARKER AT THE TOP IS NOT ENOUGH ON ITS OWN. A store that kept the first page and the
    last one and lost everything between them mounts the head when you scroll to it, passes every
    reading taken at the bottom, and is missing most of the conversation. So the traversal also
    records the `aria-posinset` of every row it passes and `ordinal_coverage` says what that
    covers -- including, when the gesture's stops did not overlap, that it covers nothing and the
    answer is NOT MEASURED.

    Run BEFORE the measured window, never inside it -- it scrolls the viewport the whole length of
    the thread and mounts whatever the arm materialises on the way, which is real work that has no
    business inside anybody's frame rate. The caller re-establishes the resting state afterwards.

    Reported, not raised. A failure here does not mean the readiness gate was wrong; it means the
    arm is losing messages, which is a finding to record against the arm rather than a reason to
    lose the cell.
    """
    out: dict = {"probe_attempted": True, "expected_messages": expected_messages}
    top = page.evaluate(TRAVERSE_JS, [True, steps, step_px])
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
    at_top = page.evaluate(COLLECT_ORDINALS_JS) or []
    coverage = ordinal_coverage(top, expected_messages, extra_seen = at_top)
    out.update(
        {
            "head_reached": found,
            "mounted_at_top": seen.get("mounted"),
            "setsize_at_top": seen.get("setsize"),
            "scroll_height_at_top": top.get("scroll_height"),
            # DID THE GESTURE ACTUALLY REACH THE TOP? Without this, "the head never mounted" and
            # "the viewport never left the bottom" are the same reading, and the second one is a
            # defect in this probe being reported as data loss in the app.
            "reached_top": top.get("reached_target"),
            "scroll_top_after_gesture": top.get("scroll_top"),
            "traverse_step_px": step_px,
        }
    )
    out.update(coverage)
    # `min_posinset_seen` before there were any ordinals to see: the head marker mounting means
    # position 1 was on the page, whether or not the arm numbers its rows. Kept so an arm that
    # publishes nothing still fills the field the way it always did.
    if out.get("min_posinset_seen") is None and found:
        out["min_posinset_seen"] = 1
    # And back to the end, so the cell resumes from the state the readiness gate described.
    page.evaluate(TRAVERSE_JS, [False, steps, step_px])
    if not found and not top.get("reached_target"):
        # NOT A VERDICT ABOUT THE ARM. The gesture did not get there, so nothing was learned about
        # what the arm holds, and saying otherwise would blame the app for the probe.
        out["head_reached"] = None
        out["reason"] = (
            f"the scroll gesture never reached the top of the thread (stopped at "
            f"{top.get('scroll_top')}px), so this says nothing about what the arm holds"
        )
        log(f"  completeness NOT MEASURED: {out['reason']}")
        return out
    if not found:
        out["reason"] = (
            "scrolling to the top of the thread never mounted the first message, so the arm is "
            "not holding the whole conversation"
        )
        log(f"  COMPLETENESS FAILED: {out['reason']}")
    elif out.get("ordinal_coverage_complete") is False:
        # THE HEAD ARRIVED AND THE THREAD IS STILL INCOMPLETE. This is the case the marker check
        # alone reported as a pass: first page kept, last page kept, middle gone.
        out["reason"] = f"the head of the thread mounted, but {out['coverage_reason']}"
        log(f"  COMPLETENESS FAILED: {out['reason']}")
    elif out.get("ordinal_coverage_state") == COVERAGE_UNMEASURED:
        # THE HEAD ARRIVED AND THE MIDDLE WAS NEVER INSPECTED. Not a finding about the arm, and
        # not a pass either: this is the same first-page-and-last-page store re-entering through
        # the unknown state, so the cell carries a reason and the gate declines to score it.
        out["reason"] = (
            f"the head of the thread mounted, but coverage of the middle was NOT ESTABLISHED: "
            f"{out['coverage_reason']}"
        )
        log(f"  COMPLETENESS NOT ESTABLISHED: {out['reason']}")
    else:
        log(
            f"  completeness: the head of the thread mounted on scroll-to-top "
            f"({seen.get('mounted')} rows mounted there, aria-setsize {seen.get('setsize')}), "
            f"ordinal coverage {out.get('ordinal_coverage_complete')} "
            f"({out.get('ordinals_seen_count')} of {expected_messages} ordinals seen)"
        )
        if out.get("ordinal_coverage_complete") is None:
            # Only the not-applicable kind reaches here; the unmeasured kind is caught above and
            # carries a reason of its own.
            log(f"  coverage DOES NOT APPLY: {out.get('coverage_reason')}")
    return out
