// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHETHER A HIGHLIGHTED FENCE IS EVER GIVEN BACK, decided in one pure module.
 *
 * Kept out of any `.tsx` and free of `import.meta`, as `code-fence-mode.ts` and `math-block-mode.ts`
 * are, because the frontend's tests run under `node --experimental-strip-types` and can neither
 * load JSX nor evaluate `import.meta.env`. Every row below is RUN by
 * `tests/code-fence-evict.test.ts` rather than checked by regexes over the source.
 *
 *   "off"     SHIP DEFAULT. A fence latches highlighted the first time it comes near the viewport
 *             and stays that way for the rest of its mount.
 *   "evict"   A fence FAR outside the viewport gives its highlighting back and returns to the
 *             plain shell, so the standing span count stays bounded over a session, not at mount.
 *
 * WHY THE STANDING COUNT IS THE THING. Shiki emits one `<span>` per token. On the reasoning-pane
 * window, 63 mutations over 5 s at 16,958 elements gave 18.9 fps while 49 mutations at 22,789
 * elements gave 14.6 fps, fps against mutation count at r = -0.88: the WRONG sign for "creating
 * spans is the cost" and the right one for "having spans is the cost". So this file is about
 * giving them back rather than about scheduling.
 *
 * TWO STATES, NOT THREE, following `math-block-mode.ts`: the default is already off, so an unset
 * flag and a mistyped one land in the same place.
 */
export type FenceEvictMode = "off" | "evict";

/** Moving this line is the whole of "give highlighting back by default". */
export const SHIP_DEFAULT: FenceEvictMode = "off";

/**
 * @param runtime  `__UNSLOTH_EVICT_FENCE_HIGHLIGHT__`: string, boolean or absent. The boolean is
 *                 the devtools-console form and works BOTH ways, so a session flips without a build.
 * @param build    `VITE_UNSLOTH_EVICT_FENCE_HIGHLIGHT`, `""` when never set.
 */
export const resolveFenceEvictMode = (
  runtime: unknown,
  build: string,
): FenceEvictMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "evict"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "evict"
    ? "evict"
    : raw === ""
      ? SHIP_DEFAULT
      : "off";
};

/*
 * THE TWO BANDS, AND WHY THEY ARE NOT THE SAME BAND.
 *
 * `code-fence-defer.tsx` latches a fence within ONE root height of the scroller's box, which is its
 * `REACH_MARGIN = "100% 0px"`. Evicting on the complement of that predicate would make the gate
 * bidirectional at a single boundary, and a reader resting with a fence at the edge would scrub it
 * in and out once per wheel notch. The previous bidirectional attempt was predicted to save 55%
 * and MEASURED SLOWER, and this is the shape of why.
 *
 * So eviction uses a strictly wider band: given back only outside THREE root heights, which is at
 * least two outside the band that would latch it again. `evictionCannotImmediatelyRelatch` below is
 * that invariant, run as a test rather than asserted in a comment.
 */
export const REACH_BAND = 1;
export const HOLD_BAND = 3;

/**
 * How long a fence must have been highlighted before it is eligible to be given back.
 *
 * A fast scroll crosses both bands within a few hundred milliseconds. Without a dwell that gesture
 * unmounts spans it created two frames earlier: DOM work added to frames that are already dropping,
 * and no reduction in the standing count that outlives the gesture.
 */
export const DWELL_MS = 1500;

/**
 * The most fences one pass may give back.
 *
 * An unbudgeted pass over a thread just scrolled end to end would unmount every block in one task.
 * The count is bounded rather than the time, because the cost bounded here is a commit, not a loop.
 */
export const EVICT_BUDGET = 8;

/** The shortest gap between two passes, so a flick that fires many scroll events runs one. */
export const PASS_INTERVAL_MS = 400;

/** The parts of a `DOMRect` this file needs. Passed in so nothing here touches layout. */
export interface FenceRect {
  top: number;
  bottom: number;
}

/**
 * The scroller's box: `top` and `height` from its `getBoundingClientRect()`, or the window's
 * `{ top: 0, height: innerHeight }` when a fence has no scrolling ancestor. Exactly what `inBand`
 * in `code-fence-defer.tsx` reads.
 */
export interface ScrollerBand {
  top: number;
  height: number;
}

/**
 * Is `rect` inside `band` grown by `multiple` of the band's own height each way?
 *
 * `multiple === REACH_BAND` reproduces `inBand` in `code-fence-defer.tsx` exactly, which is what
 * makes the two predicates below comparable at all. `tests/code-fence-evict.test.ts` runs that
 * equivalence rather than trusting this sentence.
 */
export const withinBand = (
  rect: FenceRect,
  band: ScrollerBand,
  multiple: number,
): boolean =>
  rect.bottom > band.top - multiple * band.height
  && rect.top < band.top + band.height + multiple * band.height;

/**
 * How far outside the band's own box `rect` is, in root heights; zero while any part is inside.
 * Used only to order candidates, so the furthest are given back first.
 */
export const bandDistance = (rect: FenceRect, band: ScrollerBand): number => {
  if (band.height <= 0) return 0;
  const above = band.top - rect.bottom;
  const below = rect.top - (band.top + band.height);
  return Math.max(0, above, below) / band.height;
};

export interface EvictCandidate {
  /** Whatever the caller uses to identify a fence. Returned as-is. */
  readonly id: number;
  readonly rect: FenceRect;
  readonly band: ScrollerBand;
  /** `performance.now()` when this fence latched. */
  readonly latchedAt: number;
  /** A streaming fence is the one the reader is watching and is never given back. */
  readonly streaming: boolean;
}

/**
 * Is this one fence eligible, on its own, ignoring the budget?
 *
 * Cheap tests first: streaming and dwell reject without touching the geometry.
 */
export const fenceIsEvictable = (
  candidate: EvictCandidate,
  now: number,
): boolean =>
  !candidate.streaming
  && now - candidate.latchedAt >= DWELL_MS
  && !withinBand(candidate.rect, candidate.band, HOLD_BAND);

/**
 * THE WHOLE PASS, as one pure function over the state the caller has already read.
 *
 * Returns the ids to give back, furthest first, at most `EVICT_BUDGET`. Empty is the common case.
 *
 * @param selectionLive  a non-collapsed selection exists somewhere in the document.
 * @param printing       the print stylesheet is in force.
 */
export const planEviction = (
  candidates: readonly EvictCandidate[],
  now: number,
  { selectionLive, printing }: { selectionLive: boolean; printing: boolean },
): number[] => {
  /*
   * TWO REFUSALS, BOTH ABOUT NOT CHANGING WHAT THE READER ALREADY HAS.
   *
   * A SELECTION. Giving a fence back unmounts its subtree and a range anchored inside it dies too,
   * so a reader who selects a long thread and copies gets a document that changed under them. The
   * shell holds the same characters, so nothing is lost by waiting.
   *
   * A PRINT. `code-fence-defer.tsx` latches the WHOLE document on `beforeprint`, because a printed
   * page is every fence at once. A pass between that and the snapshot would take pages back.
   */
  if (selectionLive || printing) return [];
  const evictable = candidates.filter((candidate) => fenceIsEvictable(candidate, now));
  evictable.sort(
    (a, b) => bandDistance(b.rect, b.band) - bandDistance(a.rect, a.band),
  );
  return evictable.slice(0, EVICT_BUDGET).map((candidate) => candidate.id);
};

/**
 * WHEN THE NEXT PASS SHOULD RUN, or `null` for "nothing left to do until something moves".
 *
 * A pass is scheduled BY a scroll, so without this eviction would only progress while the reader is
 * scrolling. Two cases need a pass nobody is going to ask for:
 *
 *   - The budget capped this pass, leaving most of a scrolled-through thread standing until the
 *     next gesture. So a pass that evicted anything asks for another; that terminates because each
 *     one removes at least one fence from a finite set.
 *   - A fence is far outside the hold band but still inside its dwell, so the wait comes from ITS
 *     clock, floored at `PASS_INTERVAL_MS` so a fence that just matured cannot tighten the loop.
 *
 * `null` when neither applies, which is the resting state: no timer, no listener work, nothing.
 */
export const nextPassDelayMs = (
  candidates: readonly EvictCandidate[],
  now: number,
  evicted: number,
): number | null => {
  if (evicted > 0) return PASS_INTERVAL_MS;
  let soonest: number | null = null;
  for (const candidate of candidates) {
    if (candidate.streaming) continue;
    if (withinBand(candidate.rect, candidate.band, HOLD_BAND)) continue;
    const wait = DWELL_MS - (now - candidate.latchedAt);
    // `wait <= 0` means eligible but not taken, so the pass was refused outright; rescheduling
    // for that would spin against a live selection.
    if (wait <= 0) continue;
    if (soonest === null || wait < soonest) soonest = wait;
  }
  return soonest === null ? null : Math.max(PASS_INTERVAL_MS, soonest);
};

/**
 * Should a pass run at all, given when the last one ran? `last === null` means none has, and the
 * first scroll after a fence latches must not wait out an interval that never started.
 */
export const passIsDue = (now: number, last: number | null): boolean =>
  last === null || now - last >= PASS_INTERVAL_MS;

/**
 * THE ANTI-CHURN INVARIANT, exported so a test can RUN it over generated geometry rather than trust
 * the two constants to stay ordered: a fence this file gives back must be outside the band that
 * `code-fence-defer.tsx` would use to take it straight back. False means the gate has collapsed to
 * a single boundary and a reader resting at the edge scrubs spans in and out.
 *
 * `hold` and `reach` are parameters ONLY so a test can drive the false case and show the true one
 * is not vacuous. Nothing in the app passes them.
 */
export const evictionCannotImmediatelyRelatch = (
  rect: FenceRect,
  band: ScrollerBand,
  hold: number = HOLD_BAND,
  reach: number = REACH_BAND,
): boolean => withinBand(rect, band, hold) || !withinBand(rect, band, reach);

/**
 * HOW FAR THE READER MUST SCROLL BACK, in pixels, before a fence just given back is taken again.
 * With the two bands EQUAL the invariant above is still true and still worth nothing: the
 * boundaries coincide and one pixel of movement flips the fence. A positive gap removes the churn.
 * Same parameters, same reason: the control row passes equal bands and must read 0.
 */
export const relatchGapPx = (
  band: ScrollerBand,
  hold: number = HOLD_BAND,
  reach: number = REACH_BAND,
): number => Math.max(0, hold - reach) * Math.max(0, band.height);
