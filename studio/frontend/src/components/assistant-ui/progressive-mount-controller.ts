// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Widen-only mount window for the chat thread.
//
// Re-opening a 300K thread costs 3175ms on Chromium (#9016), nearly all of it mounting messages
// from nothing. Mounting only the last 20 of 220 re-opens in 438ms and is FLAT in thread length
// (441ms at 25K, 470ms at 100K, 438ms at 300K, medians of 8 paired rounds): cost follows mounted
// nodes, not thread length.
//
// Not a virtualizer: unmounting what scrolls away breaks find-in-page, cross-message selection,
// copy-all, screen readers, print and export. Open WebUI shipped that in 02690339 and reverted it
// ten days later in d56d74b3 as "catastrophic mount/destroy thrashing" (open-webui#23990).
//
// So the window only ever GROWS: first commit mounts the tail, each frame adds a chunk, and once it
// covers the thread the restriction is dropped for good. Nothing is ever unmounted, so the document
// converges to exactly the pre-change tree a few frames later; consumers that cannot wait have
// `completeProgressiveMounts` (see use-progressive-mount).
//
// The thread is always bottom-anchored, so the window is a single `start` index. That is the one
// simplification over LibreChat#14901, which also has to support a top anchor.

/**
 * Indices `[start, count)` may mount. `null` means no restriction: where a settled thread rests,
 * and the only state a short thread is ever in.
 */
export type MountWindow = { start: number } | null;

/**
 * Below this many messages the whole thread mounts in one commit, as before. Chunking a thread that
 * was never slow only adds frames: at 20 messages, 446ms against a 438ms floor.
 */
export const MIN_PROGRESSIVE_MESSAGES = 40;

/**
 * Messages in the first commit.
 *
 * A window that undershoots what the user can see paints a gap, so what matters is how tall 16 of
 * the SHORTEST rows this app can produce are. One-word rows alternate 126px (user) and 80px
 * (assistant) because the roles carry different padding; sixteen of them measure 1643px, and with
 * the viewport's 48px top inset and 165px bottom spacer that fills every clientHeight up to 1890px.
 * Quote the pair, not their 103px mean: the mean only reproduces the total for an EVEN row count,
 * drifting up to 23px otherwise, which is why INITIAL_MESSAGES is even.
 *
 * Measured, not derived, on a 144-message thread whose last 24 messages are one-word replies, three
 * rounds a point, Chromium 151 / Firefox / WebKit: at 900, 1080, 1440, 1800, 1840 and 1860px of
 * clientHeight the empty band below the last row on the first painted commit is exactly the band a
 * settled thread has, so 0px of it is this window's. First row on screen at 133 to 315ms by engine.
 *
 * Above 1890px it undershoots, left unfixed: 10px of band at 1900px, 110px at 2000px, 270px at
 * 2160px, closed by the first chunk and nothing unreachable meanwhile. Deriving the window from
 * clientHeight would have to guess how many rows fill it before mounting any, and the merge base it
 * is compared against paints NOTHING for 1318ms at 2160px and then the whole thread at once. 270px
 * of white for a third of a second, at a height needing a 4K panel in portrait, beats 1.3s of empty
 * screen at every height. 16 is also what LibreChat#14901 ships for the same job.
 *
 * Probe: tests/studio/probe_compact_tail_gap.py.
 */
export const INITIAL_MESSAGES = 16;

/**
 * Messages added per widening frame until the window covers the thread.
 *
 * Measured, not inherited. One build at three chunk sizes through #9016's 300K fixture, three
 * rounds of three repetitions each:
 *
 * | chunk | first row | converged | worst frame | frames over 33ms |
 * |---|---|---|---|---|
 * | 16 | 226ms | 3199ms | 240ms | 15 |
 * | 32 | 236ms | 3186ms | 437ms | 9 |
 * | 64 | 222ms | 3170ms | 703ms | 6 |
 *
 * Chunk size cannot affect the first commit (always INITIAL_MESSAGES wide) nor the total work; it
 * only trades build-in height against length. 32 is the midpoint and what LibreChat#14901 uses.
 * Changing it chooses a shape, not a speedup.
 */
export const CHUNK_MESSAGES = 32;

/**
 * The most messages the first commit may span, however few of them render.
 *
 * Sizing the tail on renderable rows means walking back across every non-rendering message in
 * between, and that walk is unbounded on its own: sixteen visible messages followed by two hundred
 * system entries walks to zero and the first commit rebuilds every provider in the thread, which is
 * the bound this file exists to create.
 *
 * Four tails. A thread has to be three quarters non-rendering across its whole tail before this
 * binds at all, which no thread the importers produce comes close to; below that it costs nothing
 * and the window is exactly the renderable-row one. When it does bind the first commit shows fewer
 * than INITIAL_MESSAGES rows, possibly none, and widening reaches the rest within a few frames --
 * bounded work either way, against the 1318ms of empty screen the merge base pays for the same
 * thread.
 */
export const MAX_INITIAL_SPAN = INITIAL_MESSAGES * 4;

/**
 * The window a thread opens with. A running thread opens unrestricted: widening and streaming write
 * the same scroll position, and a run starting under a short window would commit its reply into a
 * tree that has not reached it. A mid-run thread was mounted a moment ago anyway.
 */
export function initialWindow(
  count: number,
  isRunning: boolean,
  isRenderable?: (index: number) => boolean,
): MountWindow {
  if (isRunning || count < MIN_PROGRESSIVE_MESSAGES) return null;
  // Sized on rows that RENDER, not on messages. A message whose role the thread supplies no
  // component for paints nothing at all -- `threadMessageKind` returns "none" and `ThreadMessage`
  // returns null -- so it has no height. Counting those into the tail opens an imported
  // conversation ending in system messages on sixteen zero-height rows and no conversation, until
  // a later widening frame rescues it: the exact stall this window exists to remove. Both
  // importers preserve "system" (chat-import.ts, openwebui-import.ts).
  //
  // Optional so the arithmetic stays testable on its own and `stepsToCover` keeps a pure form; a
  // caller that cannot say what renders gets the old count-based window.
  if (!isRenderable) return { start: Math.max(0, count - INITIAL_MESSAGES) };
  // Walk back from the end until the window holds a full tail of renderable rows, but never past
  // MAX_INITIAL_SPAN messages: the walk crosses non-rendering messages too, and without the cap a
  // tail dense with them reaches index 0 and mounts the whole thread. Reaching further back only
  // ever mounts MORE, so this cannot shrink the window or unmount anything.
  const floor = Math.max(0, count - MAX_INITIAL_SPAN);
  let renderable = 0;
  for (let index = count - 1; index >= floor; index -= 1) {
    if (!isRenderable(index)) continue;
    renderable += 1;
    if (renderable >= INITIAL_MESSAGES) return { start: index };
  }
  // The cap bound before a full tail was found. Take the cap: the first commit shows whatever
  // renders inside it, which may be nothing, and widening reaches the rest within a few frames.
  // Bounded and converging beats both alternatives -- mounting the whole thread, and withholding
  // on a thread this short.
  if (floor > 0) return { start: floor };
  // The whole thread fits inside the cap and still holds fewer renderable rows than one tail:
  // there is nothing to withhold, and withholding anyway would hide part of the little there is.
  return null;
}

/** True once the window admits every message, which is when the restriction can be dropped. */
export function isCovered(current: MountWindow): boolean {
  return current == null || current.start <= 0;
}

/**
 * The next window. Returns `null` once it would cover the thread, so the restriction drops in the
 * same commit that mounts the last chunk rather than a frame later.
 *
 * `count` is re-read every step because messages arrive between frames. `start` only ever moves
 * down, so a thread growing under a live window cannot un-mount anything.
 */
export function widen(current: MountWindow, count: number): MountWindow {
  if (current == null) return null;
  // A start at or past the end withholds nothing. Clamping to `count` then subtracting a chunk
  // would move `start` BACKWARDS relative to the screen ({start: 204} on a thread that shrank to
  // 100 becomes {start: 68}, unmounting rows 0 to 67). Drop the window instead.
  if (current.start >= count) return null;
  const start = Math.max(0, current.start - CHUNK_MESSAGES);
  if (start <= 0) return null;
  return { start };
}

/**
 * Whether `index` may mount. Out-of-window indices are not rendered at all rather than hidden: the
 * cost avoided is mounting, so hiding would collect nothing.
 */
export function admits(current: MountWindow, index: number): boolean {
  return current == null || index >= current.start;
}

/** One end of a widening, as the correction below needs to see it. */
export type AnchorSample = {
  /**
   * The anchor row's offset from the top of the SCROLL CONTAINER, not of the window.
   *
   * `getBoundingClientRect().top` is window-relative, so it also moves when the container moves
   * for reasons unrelated to the thread: the composer grows a line, mobile chrome slides away, a
   * parent relayouts, the window resizes. Any of those landing between capture and the widening
   * commit would read as content inserted above and be corrected away, moving a detached reader
   * for nothing. Subtracting the container's own top at both ends removes it.
   */
  viewportOffset: number;
  /** The scroll container's scrollTop at the same instant. */
  scrollTop: number;
  /**
   * The largest scrollTop possible at the same instant (`scrollHeight - clientHeight`).
   *
   * Only for the one case where the browser moves scrollTop with anchoring off: content above a
   * reader near the bottom SHRINKING forces a clamp, since the old offset no longer exists. The
   * document-space delta reports the whole shrink anyway, so applying it on top of the clamp would
   * move the viewport twice by the clamped part. See anchorCorrection.
   */
  maxScrollTop: number;
};

/**
 * How far to move scrollTop so a widening commit leaves the reader looking at the same content,
 * or null for "do nothing". Pure, so it can be tested as arithmetic rather than as a substring.
 *
 * DOCUMENT space, correct only because the viewport turns CSS scroll anchoring OFF while the window
 * is open (see progressive-messages). With native anchoring live the browser moves scrollTop by the
 * inserted height and a document-space delta reports that insertion too, so applying it doubles the
 * browser's correction: that was v1, and it walked a parked reader's scrollTop 22,897 to 117,104.
 *
 * v2 kept anchoring on, measured the residual in viewport space, and dropped any frame the reader
 * scrolled through, since in viewport space a gesture and a layout shift are the same number. That
 * holds only while the browser compensates, and it is not in charge of whether it does. Measured at
 * 150K characters:
 *
 * - Every shipping Safari and iOS browser has no scroll anchoring at all today; WebKit implemented
 *   it in February 2026 and Playwright's WebKit 26.5 has it, which hid this on this machine. With
 *   `overflow-anchor: none` standing in for those builds, one dropped frame walked a parked reader
 *   19,259px and dropping every frame walked them 45,873px, on all three engines.
 * - Anchoring is also suppressed PER FRAME on engines that have it, including after a programmatic
 *   scroll, which is what scrollbar drag, PageUp and middle-click autoscroll all become. Chromium
 *   151 with a reader scrolling that way through the build-in: 45,161px of drift.
 *
 * So the browser is taken out of the loop rather than trusted in it. With anchoring off scrollTop
 * moves only when the reader or this code moves it, and document space subtracts the reader's own
 * movement arithmetically: no frame skipped, nothing to feature-probe. Same fixture, same reader:
 * 12px.
 */
export function anchorCorrection(
  captured: AnchorSample,
  now: AnchorSample,
): number | null {
  // What the browser was forced to absorb. With anchoring off it moves scrollTop for exactly one
  // reason: the reader's offset stopped existing because the document got shorter above them. That
  // part already happened, so only the remainder is applied.
  //
  // The clamp is estimated from the captured scrollTop, so it is exact only for a reader who did
  // not scroll during the frame, and that bound cannot be closed here: the two orderings (clamp
  // then gesture, gesture then clamp) produce the SAME pair of samples while wanting different
  // answers, so gating on "scrollTop did not end at maxScrollTop" just moves the error to the other
  // ordering. This picks clamp-first and is wrong by at most `captured.scrollTop -
  // now.maxScrollTop`, the height of the shrink the reader was hanging over, never by their
  // gesture. It needs a DETACHED reader parked inside that overhang, content above them shrinking,
  // and a scroll in that one frame, with a window open. Everything else has this term at zero.
  const clamped = Math.max(0, captured.scrollTop - now.maxScrollTop);
  const shift =
    now.viewportOffset +
    now.scrollTop -
    (captured.viewportOffset + captured.scrollTop) +
    clamped;
  // Whole pixels only, so a subpixel reflow does not issue a scroll write every frame.
  return Math.abs(shift) >= 1 ? shift : null;
}

/**
 * Widening frames a thread of `count` messages takes to converge. Tests assert on this so the
 * constants above cannot drift into a thread that never finishes.
 */
export function stepsToCover(count: number): number {
  let steps = 0;
  let current = initialWindow(count, false);
  while (current != null) {
    current = widen(current, count);
    steps += 1;
    // Bounded by construction; fails a test rather than hanging the browser if that ever changes.
    if (steps > count + 2) throw new Error("widen made no progress");
  }
  return steps;
}
