# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Readiness means the app has finished building the real thread, not that N DOM nodes exist."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# `full` is strictly stronger than the old gate; `windowed` is for arms that mount fewer nodes.
MODE_FULL = "full"
MODE_WINDOWED = "windowed"
MODES = (MODE_FULL, MODE_WINDOWED)

# 600ms so a mount loop that pauses for one frame cannot span the gap.
STABLE_GAP_MS = 600
STABLE_SAMPLES = 2
# Not zero: virtualised lists land a few px off; 24px is under one text line.
BOTTOM_TOLERANCE_PX = 24

DEFAULT_TIMEOUT_S = 180


class ThreadNotReady(TimeoutError):
    """TimeoutError subclass so existing except clauses still catch it; `.detail` has the last reading."""

    def __init__(self, message: str, detail: dict):
        super().__init__(message)
        self.detail = detail


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
    """Returns a verdict per condition; None means not applicable in this mode, never a pass or a fail."""
    if not probe.get("probe_attempted"):
        return {"probe": False}

    settled: Any = False
    if previous is not None and previous.get("probe_attempted"):
        settled = all(probe.get(k) == previous.get(k) for k in SETTLE_KEYS)

    out: dict[str, Any] = {
        "composer_present": bool(probe.get("composer")),
        "any_message_mounted": (probe.get("mounted") or 0) > 0,
        "settled": bool(settled),
        # Last user marker mounted within 2 of the end (the thread ends user, assistant).
        "end_present": bool(probe.get("marker_found"))
        and (probe.get("marker_from_end") is not None and probe["marker_from_end"] <= 2),
    }

    if mode == MODE_WINDOWED:
        # The virtualizer's aria-setsize must equal the seeded total, unless the whole thread is mounted.
        fully_mounted = (probe.get("mounted") or 0) >= expected_messages
        out["total_declared"] = fully_mounted or probe.get("setsize") is not None
        out["total_matches_seeded"] = fully_mounted or probe.get("setsize") == expected_messages
        published = probe.get("posinset_count") or 0
        out["posinset_on_every_row"] = fully_mounted or (
            probe.get("posinset_count") == probe.get("mounted") and (probe.get("mounted") or 0) > 0
        )
        # Ordinals must be real positions: >= 1, distinct, within setsize; waived only for a full mount.
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
        # Without this, a bottom window numbered 1..6 of 18 would pass every other condition.
        out["posinset_reaches_end"] = (fully_mounted and published == 0) or (
            probe.get("max_posinset") == expected_messages
        )
        # Prefer the app's at-bottom answer; scrollTop arithmetic on virtualised lists is off by pixels.
        app_bottom = probe.get("app_says_at_bottom")
        near_bottom = (
            probe.get("from_bottom") is not None and probe["from_bottom"] <= BOTTOM_TOLERANCE_PX
        )
        out["anchored_at_end"] = bool(app_bottom) if app_bottom is not None else near_bottom
        # Assert the viewport exists: without it every windowed condition above silently passes.
        out["viewport_present"] = bool(probe.get("viewport_present"))
        # `--aui-scroll-stabilizer` is present while autoscroll is still pinning.
        out["pin_settled"] = not probe.get("pinning")
        out["not_over_mounted"] = (probe.get("mounted") or 0) <= expected_messages
    else:
        out["all_messages_mounted"] = (probe.get("mounted") or 0) >= expected_messages
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


TRAVERSE_JS = """
async ([toTop, steps, stepPx]) => {
  const D = window.__sb.dom;
  const vp = D.viewport();
  if (!vp) return { ran: false, reason: "no thread viewport" };
  // STEPPED, AND WITH A WHEEL EVENT ON EVERY STEP. A single `scrollTo({top: 0})` from the bottom
  // does not work on this app and the reason is already documented in scene/actions.py: Unsloth
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

# 400 steps of 2,000px cover the tallest rung; reaching either end breaks early.
TRAVERSE_STEPS = 400
TRAVERSE_STEP_PX = 2000

# Run once at the top after the head-marker wait, so late-materialising rows are seen.
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

MISSING_ORDINALS_LISTED = 40

# not_applicable and unmeasured are both None but opposite; only unmeasured is withheld from a pass.
COVERAGE_COMPLETE = "complete"
COVERAGE_INCOMPLETE = "incomplete"
COVERAGE_NOT_APPLICABLE = "not_applicable"
COVERAGE_UNMEASURED = "unmeasured"

COVERAGE_STATES_SCOREABLE = (COVERAGE_COMPLETE, COVERAGE_NOT_APPLICABLE)


def ordinal_coverage(
    traverse: dict,
    expected_messages: int,
    extra_seen: Any = (),
) -> dict:
    """None is not_applicable (no ordinals published) or unmeasured; ordinal_coverage_state says which."""
    seen = {int(n) for n in (traverse.get("ordinals_seen") or [])}
    seen.update(int(n) for n in (extra_seen or ()))
    expected = set(range(1, expected_messages + 1)) if expected_messages > 0 else set()
    missing = sorted(expected - seen)
    # Only rows never seen anywhere count, so rows briefly absent during materialisation are not lost.
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
    # Check not-applicable first so an arm with no ordinals is not reported as a failed measurement.
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
    """Scrolls the whole thread to check the first message and ordinal coverage; reported, never raised."""
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
            # Distinguishes a missing head from a gesture that never left the bottom.
            "reached_top": top.get("reached_target"),
            "scroll_top_after_gesture": top.get("scroll_top"),
            "traverse_step_px": step_px,
        }
    )
    out.update(coverage)
    if out.get("min_posinset_seen") is None and found:
        out["min_posinset_seen"] = 1
    # Return to the end so the cell resumes from the state the readiness gate described.
    page.evaluate(TRAVERSE_JS, [False, steps, step_px])
    if not found and not top.get("reached_target"):
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
        out["reason"] = f"the head of the thread mounted, but {out['coverage_reason']}"
        log(f"  COMPLETENESS FAILED: {out['reason']}")
    elif out.get("ordinal_coverage_state") == COVERAGE_UNMEASURED:
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
            log(f"  coverage DOES NOT APPLY: {out.get('coverage_reason')}")
    return out
