// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which code blocks are allowed to carry Shiki tokens, and which render plain.
//
// WHY THERE IS A DECISION HERE AT ALL. At the 100K rung the thread is 63,180 elements and 42,077
// of them are Shiki highlight spans: 67% of the whole document, from 57 code blocks. Interactions
// against that standing DOM collapse to 2.7 fps (reasoning_toggle) and 5.6 fps
// (select_all_copy), while streaming stays at ~60 fps. The cost tracks the standing PRESENCE of
// those nodes, not the work of creating them -- its correlation with mutation count is r = -0.88,
// the wrong sign for creation being the cost. So the lever is how many highlight spans are
// STANDING in the document, and the blocks a user cannot see are the ones that can give them up.
//
// This module is only the decision. It holds no DOM, no React and no timers: geometry is pushed
// in, `now` is passed in, and every transition is observable. That is deliberate -- the rule below
// is the part that decides whether a user ever sees uncoloured code, so it is the part that has to
// be testable without a browser.

/**
 * How far outside the viewport a code block still renders with full Shiki tokens.
 *
 * This is THE knob: it buys the time a block needs to be re-highlighted between entering the
 * buffer and reaching the user's eye. Sized from the measured re-highlight latency rather than
 * chosen round.
 *
 * Re-highlighting one fence of the studiobench corpus from a cold fence cache, with grammars
 * already loaded (the state a scrolling thread is actually in), measured over all 57 content
 * fences at the 100K rung: median 12.7 ms, p90 28.5 ms, max 39.4 ms, at a median 1,783 characters
 * and 25 lines per fence.
 *
 * A fence that size occupies roughly 570 px once rendered (25 lines at ~20 px, plus 32 px of
 * padding, borders and the wrapper's `my-4`), so 2,000 px of buffer holds about 3.5 blocks, and
 * refilling all of them costs about 100 ms at p90. Crossing 2,000 px takes 400 ms at 5,000 px/s,
 * which is a hard trackpad flick. That is a 4x margin. It narrows to parity only around
 * 20,000 px/s, which is a scrollbar drag -- a jump, where a repaint is expected anyway.
 *
 * Raising this hides uncoloured code under faster scrolling and stands more spans in the
 * document; lowering it does the reverse. Nothing else in the mechanism trades those two off.
 */
export const CODE_HIGHLIGHT_BUFFER_PX = 2000;

/**
 * How long after a block last grew it still counts as streaming.
 *
 * A streaming fence re-enters `highlight()` every frame with the whole block, and the plugin
 * throttles its re-tokenization to one refresh per `REFRESH_MS` (250 ms). Four refresh windows is
 * long enough that an ordinary gap between chunks never reads as "finished", and short enough
 * that a finished block becomes eligible for gating well within a second.
 */
export const CODE_HIGHLIGHT_STREAMING_GRACE_MS = 1000;

/** Full Shiki tokens, or one token per line with no colours. */
export type HighlightMode = "highlighted" | "plain";

/** A block's position in the scroll container, in the same coordinates as the viewport height. */
export type BlockGeometry = {
  /** Distance from the top of the viewport to the top of the block. Negative when scrolled past. */
  top: number;
  /** Distance from the top of the viewport to the bottom of the block. */
  bottom: number;
};

export type HighlightDecision = {
  /** `null` means nothing has located this block yet. */
  geometry: BlockGeometry | null;
  viewportHeight: number;
  bufferPx: number;
  /** True while the block is still being streamed into. */
  streaming: boolean;
};

/**
 * The whole rule, in one pure function.
 *
 * Both early returns FAIL OPEN, towards colour. Getting this backwards is the only way this
 * mechanism can produce a visibly wrong thread rather than a merely slow one, so neither is an
 * optimisation to be tightened later.
 */
export const decideHighlightMode = ({
  geometry,
  viewportHeight,
  bufferPx,
  streaming,
}: HighlightDecision): HighlightMode => {
  // The block being streamed is in view by definition -- it is the reply the user is waiting on.
  // It also owns the plugin's incremental caches, which a downgrade mid-stream would be fighting.
  if (streaming) return "highlighted";
  // A block nothing has measured yet is not a block known to be far away. On first render the
  // element does not exist, so there is no geometry to have; serving plain here would decolour
  // every block for the frame before it is located, including the ones on screen.
  if (geometry === null) return "highlighted";
  return geometry.bottom >= -bufferPx &&
    geometry.top <= viewportHeight + bufferPx
    ? "highlighted"
    : "plain";
};

export type CodeHighlightGate = {
  /** The mode a block should render in right now. Unknown ids read as `highlighted`. */
  mode: (id: string, now?: number) => HighlightMode;
  /** Record where a block is, or `null` to forget its position without forgetting the block. */
  place: (id: string, geometry: BlockGeometry | null) => void;
  /** Record that a block grew at `now`, which holds it highlighted for the grace window. */
  markStreaming: (id: string, now?: number) => void;
  setViewportHeight: (height: number) => void;
  /** Called with each block id whose mode may have changed. */
  subscribe: (listener: (id: string) => void) => () => void;
  /** Called when the plugin first produces a result for a block, so a binder can go find it. */
  onAnnounce: (listener: (id: string) => void) => () => void;
  announce: (id: string) => void;
  forget: (id: string) => void;
  readonly bufferPx: number;
};

type Entry = {
  geometry: BlockGeometry | null;
  lastGrowthAt: number;
  mode: HighlightMode;
};

export type GateOptions = {
  bufferPx?: number;
  streamingGraceMs?: number;
  viewportHeight?: number;
  now?: () => number;
};

export const createCodeHighlightGate = ({
  bufferPx = CODE_HIGHLIGHT_BUFFER_PX,
  streamingGraceMs = CODE_HIGHLIGHT_STREAMING_GRACE_MS,
  viewportHeight = 0,
  now = () =>
    typeof performance !== "undefined" && typeof performance.now === "function"
      ? performance.now()
      : Date.now(),
}: GateOptions = {}): CodeHighlightGate => {
  const entries = new Map<string, Entry>();
  const listeners = new Set<(id: string) => void>();
  const announcements = new Set<(id: string) => void>();
  let viewport = viewportHeight;

  const decide = (entry: Entry, at: number): HighlightMode =>
    decideHighlightMode({
      geometry: entry.geometry,
      viewportHeight: viewport,
      bufferPx,
      streaming: at - entry.lastGrowthAt < streamingGraceMs,
    });

  // Notify on the TRANSITION only. The plugin turns each notification into a re-render of that
  // block, so a gate that republished the current mode on every scroll event would re-render
  // every mounted code block on every frame -- the cost this exists to remove.
  const settle = (id: string, entry: Entry, at: number): void => {
    const next = decide(entry, at);
    if (next === entry.mode) return;
    entry.mode = next;
    for (const listener of listeners) listener(id);
  };

  const entryFor = (id: string): Entry => {
    let entry = entries.get(id);
    if (entry === undefined) {
      // Negative infinity, not 0: a block registered by `place` on a page that has been open for
      // a while must not read as having grown at time zero and so be inside the grace window.
      entry = {
        geometry: null,
        lastGrowthAt: Number.NEGATIVE_INFINITY,
        mode: "highlighted",
      };
      entries.set(id, entry);
    }
    return entry;
  };

  return {
    bufferPx,
    mode: (id, at = now()) => {
      const entry = entries.get(id);
      if (entry === undefined) return "highlighted";
      // Read through `decide` rather than the cached mode: the grace window expires on the clock,
      // with no event to settle it.
      entry.mode = decide(entry, at);
      return entry.mode;
    },
    place: (id, geometry) => {
      const entry = entryFor(id);
      entry.geometry = geometry;
      settle(id, entry, now());
    },
    markStreaming: (id, at = now()) => {
      const entry = entryFor(id);
      entry.lastGrowthAt = at;
      settle(id, entry, at);
    },
    setViewportHeight: (height) => {
      if (height === viewport) return;
      viewport = height;
      const at = now();
      for (const [id, entry] of entries) settle(id, entry, at);
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    onAnnounce: (listener) => {
      announcements.add(listener);
      return () => announcements.delete(listener);
    },
    announce: (id) => {
      entryFor(id);
      for (const listener of announcements) listener(id);
    },
    forget: (id) => {
      entries.delete(id);
    },
  };
};
