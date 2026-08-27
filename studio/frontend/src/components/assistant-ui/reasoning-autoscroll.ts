// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the streaming reasoning pane stays pinned to the bottom.
//
// Extracted from reasoning.tsx as a pure reducer because the defect it fixes was
// invisible inline: the pane is a .tsx file, node's type stripping cannot compile
// JSX, so the tests next to it pin SOURCE TEXT rather than behaviour, and no
// amount of text pinning would have caught a wrong comparison. Here it is a plain
// module the tests can drive through the actual sequence.

export const AUTO_SCROLL_THRESHOLD_PX = 24;

export type PinState = {
  /** True once the user has moved away from the bottom on purpose. */
  detached: boolean;
  lastScrollTop: number;
  /** scrollHeight - clientHeight at the previous sample. */
  lastMaxScrollTop: number;
};

export function createPinState(scrollTop: number, maxScrollTop: number): PinState {
  return { detached: false, lastScrollTop: scrollTop, lastMaxScrollTop: maxScrollTop };
}

/**
 * Fold one scroll observation into the pin state.
 *
 * The subtlety, and the whole reason this exists: a `scroll` event fires for EVERY
 * scrollTop change, and the listener cannot ask who caused it. Three things lower
 * scrollTop and only the first is user intent:
 *
 *   1. the user scrolling up;
 *   2. the engine CLAMPING when content shrinks, because scrollTop is always
 *      clamped into [0, scrollHeight - clientHeight] and the old value is gone;
 *   3. our own write when we re-pin to a bottom that has moved up.
 *
 * Treating 2 or 3 as 1 latches `detached`, which stops the pin. The next mutation
 * then lands inside AUTO_SCROLL_THRESHOLD_PX of the bottom, re-attaches, and snaps
 * back. Repeated every time a subtree changes size mid-stream, that oscillation is
 * the flicker: the pane jumps to the top of its content and returns.
 *
 * Geometry cannot separate them, and this is not fixable by better arithmetic.
 * `scroll` is dispatched as a TASK while streaming mutations arrive as microtasks,
 * so by the time the listener runs the content has usually grown back: it sees a low
 * scrollTop against a NEW, LARGER maximum, which is indistinguishable from a user
 * parked mid-document. A first attempt at this fix compared against the previous
 * maximum to recognise the shrink; modelled against that ordering (shrink to 1200,
 * clamp to 944, grow to 4200, and only THEN the event) it detached exactly like the
 * code it replaced. The shrink is already over by the time anyone can look.
 *
 * So this function NEVER detaches. Intent comes only from input events, via
 * `detachByUser`. A scroll observation is pure geometry: it re-attaches when the
 * viewport is back at the bottom, and otherwise just keeps the bookkeeping current.
 */
export function observeScroll(
  prev: PinState,
  scrollTop: number,
  maxScrollTop: number,
): PinState {
  const detached =
    prev.detached && maxScrollTop - scrollTop > AUTO_SCROLL_THRESHOLD_PX;
  return { detached, lastScrollTop: scrollTop, lastMaxScrollTop: maxScrollTop };
}

/**
 * The user moved away from the bottom on purpose: wheel up, an upward key, a touch
 * drag, or a scrollbar drag. Every caller is an input event, which is the only
 * evidence of intent that exists.
 */
export function detachByUser(prev: PinState): PinState {
  return { ...prev, detached: true };
}

/** Record where our own pin write landed, so its scroll event reads as no change. */
export function notePinnedTo(prev: PinState, maxScrollTop: number): PinState {
  return { detached: prev.detached, lastScrollTop: maxScrollTop, lastMaxScrollTop: maxScrollTop };
}

export function shouldAutoScroll(state: PinState): boolean {
  return !state.detached;
}
