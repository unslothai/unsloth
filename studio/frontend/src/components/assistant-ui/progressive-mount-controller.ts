// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Widen-only mount window for the chat thread.
//
// Re-opening a 300K-character thread costs 3175ms on Chromium (#9016), of which 2817ms is script,
// and the whole of that is mounting messages from nothing. Measured against the same fixture, a
// tree that mounts only the last 20 messages instead of all 220 re-opens in 438ms and is FLAT in
// thread length: 441ms at 25K, 470ms at 100K, 438ms at 300K, medians of 8 paired rounds. Re-open
// cost is a function of mounted nodes, not of thread length.
//
// The obvious way to collect that is to virtualize, and this is deliberately not that. A
// virtualizer unmounts what scrolls away, which is what breaks find-in-page, selection across
// messages, copy-the-whole-conversation, screen readers, print and export. Open WebUI shipped
// exactly that in 02690339 and reverted it ten days later in d56d74b3; the comment they left in
// its place calls it "catastrophic mount/destroy thrashing", and the scroll-jump report that
// arrived in between is open-webui#23990.
//
// So this window only ever GROWS. The first commit mounts the tail, each following frame adds a
// chunk, and once the window covers the thread the restriction is dropped and never returns.
// Nothing is ever unmounted, so the document converges to exactly the tree that shipped before
// this change, a few frames later. Every DOM consumer therefore keeps working unchanged, and the
// ones that cannot wait those frames have `completeProgressiveMounts` (see use-progressive-mount).
//
// The thread is always bottom-anchored -- ThreadPrimitive.Viewport is mounted with autoScroll and
// every scrollToBottom* flag off, and useIntentAwareAutoScroll puts a re-opened thread at the
// bottom -- so the window is a single `start` index and the end is always the last message. That
// is the one simplification over the shape this is modelled on, LibreChat#14901, which also has to
// support a top anchor.

/**
 * Indices `[start, count)` are allowed to mount. `null` means no restriction at all, which is the
 * state a settled thread rests in and the only state a short thread is ever in.
 */
export type MountWindow = { start: number } | null;

/**
 * Below this many messages the whole thread mounts in one commit, exactly as it did before this
 * change. Chunking a thread that was never slow only adds frames to it: at 20 messages the
 * re-open is 446ms against a 438ms floor, so there is nothing here to win.
 */
export const MIN_PROGRESSIVE_MESSAGES = 40;

/**
 * Messages in the first commit. One viewport of the #9016 fixture is under two messages, so this
 * is roughly eight viewports of overscan. It is deliberately generous: the floor measurement used
 * 20 messages and read 438ms, so the first commit is bounded by that either way, and a window
 * that undershoots what the user can actually see would paint a gap.
 */
export const INITIAL_MESSAGES = 16;

/**
 * Messages added per widening frame until the window covers the thread.
 *
 * Measured rather than inherited. One build driven at three chunk sizes through #9016's fixture at
 * 300K characters, three rounds of three repetitions each:
 *
 * | chunk | first row | converged | worst frame | frames over 33ms |
 * |---|---|---|---|---|
 * | 16 | 226ms | 3199ms | 240ms | 15 |
 * | 32 | 236ms | 3186ms | 437ms | 9 |
 * | 64 | 222ms | 3170ms | 703ms | 6 |
 *
 * The headline number does not move: the chunk size cannot affect the first commit, which is
 * INITIAL_MESSAGES wide whatever this is, and it does not change the total work either. All it
 * does is distribute the remaining work, trading the height of the build-in against its length.
 * 32 is the midpoint, and it is what LibreChat#14901 uses. 16 and 64 are both defensible; anyone
 * changing it should know they are choosing a shape, not buying a speedup.
 */
export const CHUNK_MESSAGES = 32;

/**
 * The window a thread opens with.
 *
 * A thread that is already running opens unrestricted. Widening and streaming both write to the
 * same scroll position, and a run that starts while the window is still short would commit its
 * reply into a tree that has not reached it; suppressing the window for the whole run is the
 * cheap, obviously correct answer, and a thread that is mid-run was mounted a moment ago anyway.
 */
export function initialWindow(count: number, isRunning: boolean): MountWindow {
  if (isRunning || count < MIN_PROGRESSIVE_MESSAGES) return null;
  return { start: Math.max(0, count - INITIAL_MESSAGES) };
}

/** True once the window admits every message, which is when the restriction can be dropped. */
export function isCovered(current: MountWindow): boolean {
  return current == null || current.start <= 0;
}

/**
 * The next window. Returns `null` at the point the window would cover the thread, so the caller
 * drops the restriction in the same commit that mounts the last chunk rather than a frame later.
 *
 * `count` is re-read on every step because a message can arrive between frames. It only ever
 * moves `start` down, so a thread that grows underneath a live window cannot un-mount anything.
 */
export function widen(current: MountWindow, count: number): MountWindow {
  if (current == null) return null;
  const start = Math.max(0, Math.min(current.start, count) - CHUNK_MESSAGES);
  if (start <= 0) return null;
  return { start };
}

/**
 * Whether index `index` may mount under `current`.
 *
 * Out-of-window indices are not rendered yet rather than rendered-and-hidden: the cost being
 * avoided is mounting them, so hiding them would collect nothing.
 */
export function admits(current: MountWindow, index: number): boolean {
  return current == null || index >= current.start;
}

/** One end of a widening, as the correction below needs to see it. */
export type AnchorSample = {
  /** The anchor row's offset from the top of the scroll container. */
  viewportOffset: number;
  /** The scroll container's scrollTop at the same instant. */
  scrollTop: number;
  /** useIntentAwareAutoScroll's user-gesture counter at the same instant. */
  gestureSeq: number;
};

/**
 * How far to move scrollTop so a widening commit leaves the reader looking at the same content,
 * or null for "do nothing". Pure, so it can be tested as arithmetic rather than as a substring.
 *
 * `viewportCompensates` is the whole reason this has two branches, and it is a property of the
 * ENGINE, not of this code.
 *
 * When it is true the browser implements CSS scroll anchoring and has ALREADY moved scrollTop by
 * the inserted height before this runs, so all that is left is the residual it did not absorb.
 * Viewport space is what measures a residual: it is zero precisely when the anchor is still where
 * the reader left it. Measured on Chromium 151, Firefox 153 and WebKit 26.5 at 150K characters,
 * that residual is 3 to 5 pixels per widening. Because it is that small, a frame the reader
 * scrolled through can simply be dropped: their gesture is indistinguishable from a layout shift
 * in viewport space, and skipping one frame costs single-digit pixels.
 *
 * When it is false NOTHING has compensated, the "residual" is the entire inserted height, and
 * dropping a frame is not affordable. Measured on the same fixture with `overflow-anchor: none`
 * forced on the viewport, which is exactly what every shipping Safari and every iOS browser is
 * today (WebKit implemented scroll anchoring in 307475@main, Feb 2026, shipping in Safari 27):
 * one dropped frame walked a parked reader 19,259px, and dropping every frame walked them
 * 45,873px. So this branch measures in DOCUMENT space, which subtracts the reader's own scroll
 * arithmetically instead of dodging it, and never skips. That is the shape LibreChat#14901 uses,
 * and it is correct here for the same reason it is correct there: with no native compensation,
 * scrollTop moves only when the reader or this code moves it.
 */
export function anchorCorrection(
  captured: AnchorSample,
  now: AnchorSample,
  viewportCompensates: boolean,
): number | null {
  const shift = viewportCompensates
    ? now.viewportOffset - captured.viewportOffset
    : now.viewportOffset +
      now.scrollTop -
      (captured.viewportOffset + captured.scrollTop);
  if (viewportCompensates && now.gestureSeq !== captured.gestureSeq)
    return null;
  // Rounding down to whole pixels keeps a subpixel reflow from issuing a scroll write every frame.
  return Math.abs(shift) >= 1 ? shift : null;
}

/**
 * How many widening frames a thread of `count` messages takes to converge, which is what the
 * tests assert against so the constants above cannot drift into a thread that never finishes.
 */
export function stepsToCover(count: number): number {
  let steps = 0;
  let current = initialWindow(count, false);
  while (current != null) {
    current = widen(current, count);
    steps += 1;
    // The loop is bounded by construction; the guard exists so a future edit to widen that
    // stops making progress fails a test instead of hanging the browser.
    if (steps > count + 2) throw new Error("widen made no progress");
  }
  return steps;
}
