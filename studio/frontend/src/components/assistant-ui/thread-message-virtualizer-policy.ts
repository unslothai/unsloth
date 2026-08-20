// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Sizing and keying policy for the windowed message list.
 *
 * Split out of the component because there is no DOM in the test runner: a virtualizer needs a real
 * scroll element and real layout, so the parts that can be asserted are the pure ones. Everything
 * here is a value or a total function over the message array.
 */

/** The only thing the window needs from a message: something stable to key on. */
export type VirtualizedMessage = {
  id: string;
};

/**
 * Estimated height of one message, before it has been measured.
 *
 * Measured, not guessed. Three sizes of the heavy-thread fixture (20/80/220 messages) at 1440x900
 * give 574.2, 566.2 and 564.7 px per message, byte-identical across Chromium, Firefox and WebKit.
 * The same fixture at 1280 wide gives 536.4, and the reduced 144-row fixture 463.1. The first-paint
 * window (16 rows, 7,379 px) gives 461.2, because a thread opens on its short early messages.
 *
 * So the real range is 461 to 574 depending on viewport width and where in a thread you look. This
 * takes 460, the bottom of that range, deliberately: `estimateSize` too LOW makes the virtualizer
 * render more items than it needs, which costs a few extra mounts; too HIGH makes it render fewer
 * than fill the viewport, which paints a blank gap the user sees. The failure modes are not
 * symmetric, so the estimate is biased to the cheap one.
 *
 * Not derived from the studiobench film: its rungs are a handful of 300K-character documents and
 * measure 2,438 to 14,319 px per "message", which is not the shape of a chat message.
 */
export const THREAD_MESSAGE_ESTIMATE_SIZE_PX = 460;

/**
 * Messages rendered beyond each edge of the viewport.
 *
 * Counted in items, not pixels, so it has to be read against the item size: at ~460 px a message
 * and a ~900 px viewport only about two messages are on screen, so 8 either side is roughly 3,700 px
 * of buffer above and below, and about 20 messages mounted in total. Generous on purpose. The whole
 * point of this change is that a thread stops paying for thousands of standing nodes, so twenty is
 * not a cost worth shaving, and a short overscan is what produces blank space during a fast scroll.
 */
export const THREAD_MESSAGE_OVERSCAN = 8;

/**
 * How close to the bottom still counts as "at the end", for `followOnAppend`.
 *
 * Matched to RE_ATTACH_THRESHOLD_PX in use-intent-aware-autoscroll.tsx, which is the distance at
 * which that hook decides a user who scrolled up has come back and re-attaches. The virtualizer
 * follows an appended message exactly when `isAtEnd(scrollEndThreshold)` holds, so giving the two
 * the same threshold is what stops them disagreeing about whether the user is following the stream.
 * The hook's stricter 2 px AT_BOTTOM_THRESHOLD_PX is a different question (is the viewport pinned
 * right now), not this one.
 */
export const THREAD_MESSAGE_SCROLL_END_THRESHOLD_PX = 24;

/**
 * End-anchored virtualizer options.
 *
 * Requires @tanstack/virtual-core >= 3.16.1: `anchorTo`/`followOnAppend`/`scrollEndThreshold` do not
 * exist before 3.16.0, and 3.16.1 is the first release carrying the eager scrollOffset adjustment on
 * prepend. The alternative is hand-rolled scrollTop bookkeeping, which is what this replaces.
 */
export const THREAD_MESSAGE_ANCHORING = {
  // A chat is read from its end. Anchoring to the start makes every measurement correction above the
  // viewport push the content the user is reading.
  anchorTo: "end",
  // Follow a newly appended message, but only from the end: the library checks `isAtEnd` before it
  // follows, so a user who scrolled up is not yanked down.
  followOnAppend: true,
  scrollEndThreshold: THREAD_MESSAGE_SCROLL_END_THRESHOLD_PX,
} as const;

/**
 * The virtualizer key for the message at `index`.
 *
 * The message id, never the index. After a prepend (history paging, a branch switch) every existing
 * message moves to a new index, so an index key renames every item: the virtualizer throws away its
 * measurement cache and React unmounts and remounts every message below the insertion point. That is
 * the same positional-keying failure that parked the block-level windowing work, where Streamdown
 * keyed blocks as `useId()-index` and compared `e.index !== t.index`.
 *
 * The fallback is only reachable if `count` and the message array disagree for a render, which the
 * component avoids by deriving both from the same array. It is deliberately not the bare index, so a
 * transient gap cannot collide with a real message's key.
 */
export function messageKeyAt(
  messages: readonly VirtualizedMessage[],
  index: number,
): string {
  return messages[index]?.id ?? `aui-unresolved-message-${index}`;
}

/**
 * Distance from the top of the scroll element to the top of the list container.
 *
 * The virtualizer works in scroll-element coordinates, but the list is not the first thing in the
 * viewport: the header padding and the welcome block sit above it. Without this offset every item is
 * positioned that many pixels too high.
 *
 * Clamped at 0: during a thread switch the container can be measured while detached or mid-layout,
 * and a negative margin would push the first message off the top of the list.
 */
export function scrollMarginFor(
  containerTop: number,
  scrollElementTop: number,
  scrollTop: number,
): number {
  return Math.max(0, containerTop - scrollElementTop + scrollTop);
}
