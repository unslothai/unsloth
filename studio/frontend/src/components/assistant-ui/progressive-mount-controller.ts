// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pacing and anchor arithmetic for the thread's mount window (progressive-messages.tsx). Which
// rows are mounted is decided in thread-window.ts.

/** Messages added per frame while every row is being mounted for a DOM reader (find, print). */
export const CHUNK_MESSAGES = 32;

/** Leave native scrolling and its momentum alone before changing the scroll range. */
/** How long scrolling may run on after the reader's last input (touch momentum lasts seconds). */
const MOMENTUM_MS = 3000;

export function watchProgressiveScroll(
  viewport: EventTarget,
  now: () => number = () => performance.now(),
) {
  // Let the initial tail paint before starting background work.
  let resumeAt = now() + 300;
  let inputAt = Number.NEGATIVE_INFINITY;
  const pause = () => { resumeAt = now() + 300; };
  const onInput = () => { inputAt = now(); pause(); };
  // A scroll extends a pause only near the reader's input (momentum runs on after a fling);
  // the autoscroll following a stream would otherwise hold a full mount off until it ends.
  const onScroll = () => { if (now() < resumeAt && now() - inputAt < MOMENTUM_MS) pause(); };
  const onKey = (event: Event) => {
    const key = event as KeyboardEvent;
    if (key.defaultPrevented || !["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(key.key)) return;
    const target = key.target as HTMLElement | null;
    if (target?.isContentEditable || target?.closest?.("input, textarea, select")) return;
    onInput();
  };
  const listeners = { wheel: onInput, touchmove: onInput, keydown: onKey, scroll: onScroll };
  const inputTarget = (viewport as HTMLElement).ownerDocument ?? viewport;
  inputTarget.addEventListener("pointerdown", onInput, { capture: true, passive: true });
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
      inputTarget.removeEventListener("pointerdown", onInput, { capture: true });
      inputTarget.removeEventListener("beforeinput", pause, { capture: true });
      inputTarget.removeEventListener("compositionupdate", pause, { capture: true });
      for (const [type, listener] of Object.entries(listeners)) {
        viewport.removeEventListener(type, listener, { capture: true });
      }
    },
  };
}

/**
 * Largest WebKitGTK step for parked history. 32 restored the pre-windowing load time but made single
 * frames up to ~1.4 s on CPU-painted WebKit; 16 keeps the worst frame at the 2-per-step level.
 */
export const FAR_MOUNT_CHUNK = 16;

/**
 * WebKitGTK step size for a full mount. Rows prepended within `reach` px above the viewport top
 * stay unparked and can move what the reader sees, so they mount `near` at a time; past that they
 * are parked as they land, so the step grows fourfold per frame up to `max` (each commit has a
 * fixed cost there). `previous` is 0 before the first step, which stays `near`.
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
