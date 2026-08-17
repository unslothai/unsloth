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
  // A window whose start is at or past the end of the thread cannot withhold anything, and the
  // caller has already rendered every row because of that. Clamping to `count` and then
  // subtracting a chunk would move `start` BACKWARDS relative to what is on screen: {start: 204}
  // against a thread that shrank to 100 becomes {start: 68}, and rows 0 to 67, which are mounted,
  // are unmounted again. Nothing in this design is allowed to unmount a row, so the window is
  // dropped instead.
  if (current.start >= count) return null;
  const start = Math.max(0, current.start - CHUNK_MESSAGES);
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
  /**
   * The anchor row's offset from the top of the SCROLL CONTAINER, not from the top of the window.
   *
   * The difference is not academic in this app. `getBoundingClientRect().top` is measured against
   * the window, so it also moves when the scroll container itself moves, and this container moves
   * for reasons that have nothing to do with the thread: the composer grows a line, the mobile
   * browser chrome slides away, a parent relayouts, the window is resized. Any of those landing
   * between the capture and the widening commit would be read as content inserted above and
   * corrected away, moving a detached reader for no reason. Subtracting the container's own top at
   * both ends removes it.
   */
  viewportOffset: number;
  /** The scroll container's scrollTop at the same instant. */
  scrollTop: number;
  /**
   * The largest scrollTop the container could have had at the same instant, i.e.
   * `scrollHeight - clientHeight`.
   *
   * Only used for the one case where the browser moves scrollTop even with anchoring off: content
   * above a reader near the bottom SHRINKING forces a clamp, because the old offset no longer
   * exists. The document-space delta reports the whole shrink regardless, so applying it on top of
   * the clamp moves the viewport twice by the clamped part. See anchorCorrection.
   */
  maxScrollTop: number;
};

/**
 * How far to move scrollTop so a widening commit leaves the reader looking at the same content,
 * or null for "do nothing". Pure, so it can be tested as arithmetic rather than as a substring.
 *
 * DOCUMENT space, which is only correct because the viewport turns CSS scroll anchoring OFF for
 * as long as the window is open (see progressive-messages). With native anchoring live the
 * browser moves scrollTop by the inserted height itself and a document-space delta reports that
 * insertion anyway, so applying it doubles the browser's own correction: that was the first
 * version of this code and it walked a parked reader's scrollTop 22,897 to 117,104 and dumped
 * them at the bottom of a 300K thread.
 *
 * The second version kept anchoring on, measured the leftover residual in viewport space, and
 * dropped any frame the reader had scrolled through, because in viewport space a gesture and a
 * layout shift are the same number. That is correct exactly while the browser compensates, and
 * it is not in charge of whether it does. Measured at 150K characters:
 *
 * - Every shipping Safari and every iOS browser has no scroll anchoring at all today; WebKit
 *   implemented it in February 2026 and Playwright's WebKit 26.5 has it, which is why this was
 *   invisible on this machine. With `overflow-anchor: none` standing in for those builds, one
 *   dropped frame walked a parked reader 19,259px and dropping every frame walked them 45,873px,
 *   on all three engines.
 * - Anchoring is also suppressed PER FRAME on engines that do have it, including after a
 *   programmatic scroll, which is what a scrollbar drag, PageUp and middle-click autoscroll all
 *   become. Measured on Chromium 151 with anchoring available and a reader scrolling that way
 *   through the build-in: 45,161px of drift, because every one of those frames was both
 *   uncompensated and skipped.
 *
 * So the browser is taken out of the loop rather than trusted to be in it. With anchoring off,
 * scrollTop moves only when the reader or this code moves it, and document space subtracts the
 * reader's own movement arithmetically instead of dodging it, so no frame is ever skipped and
 * there is nothing to feature-probe. Same fixture, same reader: 12px.
 */
export function anchorCorrection(
  captured: AnchorSample,
  now: AnchorSample,
): number | null {
  // What the browser was forced to absorb. With anchoring off it moves scrollTop for exactly one
  // reason: the offset the reader was at stopped existing because the document got shorter above
  // them. That part of the correction has already happened, so only the remainder is applied.
  //
  // This estimates the clamp from the captured scrollTop, so it is exact only for a reader who did
  // not scroll during the frame, and the bound is worth stating because it cannot be closed here.
  // A reader who scrolls in the same frame the shrink lands changes how much the browser actually
  // had to absorb, and the two orderings -- clamp first then the gesture, gesture first then the
  // clamp -- produce the SAME pair of samples while wanting different answers. Nothing measurable
  // inside the frame separates them, so gating on "scrollTop did not end at maxScrollTop" only
  // moves the error from one ordering to the other. This picks clamp-first and is wrong by at most
  // `captured.scrollTop - now.maxScrollTop`, the height of the shrink the reader was hanging over,
  // and never by the size of their gesture. It needs a DETACHED reader parked inside that overhang
  // of the bottom, content above them shrinking, and a scroll in that one frame, while a window is
  // open. Everything else, including any insertion, has this term at zero.
  const clamped = Math.max(0, captured.scrollTop - now.maxScrollTop);
  const shift =
    now.viewportOffset +
    now.scrollTop -
    (captured.viewportOffset + captured.scrollTop) +
    clamped;
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
