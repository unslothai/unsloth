// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Widen-only mount window. Not a virtualizer: unmounting breaks find-in-page, selection, copy,
// screen readers, print and export. The window only grows; nothing is ever unmounted.

/** Indices `[start, count)` may mount; `null` means no restriction (settled or short thread). */
export type MountWindow = { start: number } | null;

/** Below this many messages the whole thread mounts in one commit. */
export const MIN_PROGRESSIVE_MESSAGES = 40;

/** Even, and sized so 16 of the shortest rows fill a clientHeight up to about 1890px. */
export const INITIAL_MESSAGES = 16;

/** Messages added per widening frame; trades build-in height against length, not total work. */
export const CHUNK_MESSAGES = 32;

/** Leave native scrolling and its momentum alone before changing the scroll range. */
export function watchProgressiveScroll(
  viewport: EventTarget,
  now: () => number = () => performance.now(),
) {
  // Let the initial tail paint before starting background work.
  let resumeAt = now() + 300;
  const pause = () => { resumeAt = now() + 300; };
  const onScroll = () => { if (now() < resumeAt) pause(); };
  const onKey = (event: Event) => {
    const key = event as KeyboardEvent;
    if (key.defaultPrevented || !["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(key.key)) return;
    const target = key.target as HTMLElement | null;
    if (target?.isContentEditable || target?.closest?.("input, textarea, select")) return;
    pause();
  };
  const listeners = { wheel: pause, touchmove: pause, keydown: onKey, scroll: onScroll };
  const inputTarget = (viewport as HTMLElement).ownerDocument ?? viewport;
  inputTarget.addEventListener("pointerdown", pause, { capture: true, passive: true });
  inputTarget.addEventListener("beforeinput", pause, { capture: true, passive: true });
  inputTarget.addEventListener("compositionupdate", pause, { capture: true, passive: true });
  for (const [type, listener] of Object.entries(listeners)) {
    viewport.addEventListener(type, listener, { capture: true, passive: true });
  }
  return {
    blocked(pointerHeld: boolean): boolean {
      if (pointerHeld) pause();
      return now() < resumeAt;
    },
    dispose() {
      inputTarget.removeEventListener("pointerdown", pause, { capture: true });
      inputTarget.removeEventListener("beforeinput", pause, { capture: true });
      inputTarget.removeEventListener("compositionupdate", pause, { capture: true });
      for (const [type, listener] of Object.entries(listeners)) {
        viewport.removeEventListener(type, listener, { capture: true });
      }
    },
  };
}

/** Caps the tail walk so a tail of non-rendering messages cannot mount the whole thread. */
export const MAX_INITIAL_SPAN = INITIAL_MESSAGES * 4;

/** A running thread opens unrestricted: widening and streaming write the same scroll position. */
export function initialWindow(
  count: number,
  isRunning: boolean,
  isRenderable?: (index: number) => boolean,
): MountWindow {
  if (isRunning || count < MIN_PROGRESSIVE_MESSAGES) return null;
  // Sized on rows that render (system messages paint nothing); without isRenderable, count-based.
  if (!isRenderable) return { start: Math.max(0, count - INITIAL_MESSAGES) };
  // Never walk past MAX_INITIAL_SPAN, or a tail dense with non-rendering rows mounts everything.
  const floor = Math.max(0, count - MAX_INITIAL_SPAN);
  let renderable = 0;
  for (let index = count - 1; index >= floor; index -= 1) {
    if (!isRenderable(index)) continue;
    renderable += 1;
    if (renderable >= INITIAL_MESSAGES) return { start: index };
  }
  if (floor > 0) return { start: floor };
  return null;
}

export function isCovered(current: MountWindow): boolean {
  return current == null || current.start <= 0;
}

/** Returns null once it would cover the thread, so the restriction drops in the same commit. */
export function widen(current: MountWindow, count: number, chunk = CHUNK_MESSAGES): MountWindow {
  if (current == null) return null;
  // A start past the end withholds nothing; clamping then subtracting would unmount rows.
  if (current.start >= count) return null;
  const start = Math.max(0, current.start - chunk);
  if (start <= 0) return null;
  return { start };
}

/**
 * Largest WebKitGTK step for parked history. 32 restored the pre-windowing load time but made single
 * frames up to ~1.4 s on CPU-painted WebKit; 16 keeps the worst frame at the 2-per-step level.
 */
export const FAR_MOUNT_CHUNK = 16;

/**
 * WebKitGTK step size. Rows prepended within `reach` px above the viewport top stay unparked and can
 * move what the reader sees, so they mount `near` at a time; past that they are parked as they land,
 * so the step grows fourfold per frame up to `max` (each commit has a fixed cost there). `previous`
 * is 0 before the first step, which stays `near` so it does not delay the tail's first paint.
 */
export function nextMountChunk(
  previous: number,
  frontierAbovePx: number | null,
  near: number,
  reach: number,
  max = FAR_MOUNT_CHUNK,
): number {
  if (frontierAbovePx == null || !(frontierAbovePx > reach)) return near;
  return Math.max(near, Math.min(max, previous * 4));
}

/** Out-of-window indices are not rendered at all: hiding would not avoid the mount cost. */
export function admits(current: MountWindow, index: number): boolean {
  return current == null || index >= current.start;
}

export type AnchorSample = {
  /** Offset from the scroll container top, not the window, so container moves are not corrected. */
  viewportOffset: number;
  scrollTop: number;
  /** For the clamp when content above a reader near the bottom shrinks; see anchorCorrection. */
  maxScrollTop: number;
};

/**
 * scrollTop delta keeping the reader on the same content, or null. Document space: correct only
 * because native scroll anchoring is off while the window is open (absent or per-frame
 * suppressed on some engines, so it cannot be trusted).
 */
export function anchorCorrection(
  captured: AnchorSample,
  now: AnchorSample,
  minShift = 1,
): number | null {
  // Subtract what the browser already absorbed via the clamp when content above shrank. Exact only
  // for a reader who did not scroll this frame.
  const clamped = Math.max(0, captured.scrollTop - now.maxScrollTop);
  const shift =
    now.viewportOffset +
    now.scrollTop -
    (captured.viewportOffset + captured.scrollTop) +
    clamped;
  return Math.abs(shift) >= minShift ? shift : null;
}

/** Keep the scroll correction integral; a subpixel leading inset absorbs the remainder. */
export function alignAnchorCorrection(shift: number, inset: number) {
  const scroll = Math.ceil(shift - inset) || 0;
  return { scroll, inset: inset + scroll - shift };
}

/** Tests assert on this so the constants cannot drift into a thread that never converges. */
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
