// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Widen-only mount window. Not a virtualizer: unmounting breaks find-in-page, selection, copy,
// screen readers, print and export. The window only grows; nothing is ever unmounted.

/**
 * Indices `[start, count)` may mount; `null` means no restriction (settled or short thread).
 * `fillTo` is set only on a first commit that mounted less than INITIAL_MESSAGES: before the first
 * paint the glue may widen it toward `fillTo` until the rows fill the viewport (see fillWindow).
 */
export type MountWindow = { start: number; fillTo?: number } | null;

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
  let inputSeen = false;
  const pause = () => { inputSeen = true; resumeAt = now() + 300; };
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
    /**
     * `atTop`: the reader is stopped at the top of the mounted rows. Wheel, key and touch input
     * then has no momentum left to interrupt, and waiting only holds them against the frontier, so
     * it does not block. Not before the first input (the initial tail still paints first), and
     * never while a pointer is held: a scrollbar drag maps the pointer to scrollTop.
     */
    blocked(pointerHeld: boolean, atTop = false): boolean {
      if (pointerHeld) pause();
      if (atTop && !pointerHeld && inputSeen) return false;
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

/**
 * Renderable rows the first commit mounts before measuring. Usually enough to fill the viewport;
 * when not, fillWindow widens toward INITIAL_MESSAGES in the same task, before the first paint.
 */
export const PROBE_MESSAGES = 4;

/**
 * Px of mounted rows required above the top of the bottom-pinned viewport, in viewport heights,
 * before the first paint. 0 would already paint the same rows; one viewport keeps the runway a
 * reader scrolling up meets before the first widening (input pauses it, see watchProgressiveScroll)
 * and absorbs rows that shrink after the first paint.
 */
export const FILL_RUNWAY_VIEWPORTS = 1;

/**
 * A running thread opens unrestricted: widening and streaming write the same scroll position.
 * With `probe`, the window starts at the last `probe` renderable rows and carries the
 * INITIAL_MESSAGES start as `fillTo`; without it, it is the INITIAL_MESSAGES window itself.
 */
export function initialWindow(
  count: number,
  isRunning: boolean,
  isRenderable?: (index: number) => boolean,
  probe?: number,
): MountWindow {
  if (isRunning || count < MIN_PROGRESSIVE_MESSAGES) return null;
  const probeRows = probe != null && probe > 0 && probe < INITIAL_MESSAGES ? probe : null;
  const withProbe = (start: number, probeStart: number | null): MountWindow =>
    probeStart != null && probeStart > start ? { start: probeStart, fillTo: start } : { start };
  // Sized on rows that render (system messages paint nothing); without isRenderable, count-based.
  if (!isRenderable) {
    return withProbe(
      Math.max(0, count - INITIAL_MESSAGES),
      probeRows == null ? null : count - probeRows,
    );
  }
  // Never walk past MAX_INITIAL_SPAN, or a tail dense with non-rendering rows mounts everything.
  const floor = Math.max(0, count - MAX_INITIAL_SPAN);
  let renderable = 0;
  let probeStart: number | null = null;
  for (let index = count - 1; index >= floor; index -= 1) {
    if (!isRenderable(index)) continue;
    renderable += 1;
    if (renderable === probeRows) probeStart = index;
    if (renderable >= INITIAL_MESSAGES) return withProbe(index, probeStart);
  }
  if (floor > 0) return withProbe(floor, probeStart);
  return null;
}

/** Geometry read in the commit that mounted `current`, before its first paint. */
export type FillSample = {
  /** Px from the first mounted row's top down to the top of the bottom-pinned viewport. */
  runwayPx: number;
  clientHeight: number;
  /** Height of the mounted rows, first row top to last row bottom. */
  rowsPx: number;
};

/**
 * Pre-paint widening of a probe window. Returns `current` itself once the mounted rows reach
 * FILL_RUNWAY_VIEWPORTS above the viewport (nothing to commit), otherwise a wider window: an
 * estimate from the mounted rows' mean height, never past `fillTo`, which it drops on arrival.
 * Unknown geometry (no rows, hidden viewport) goes straight to `fillTo`: the window it replaces.
 */
export function fillWindow(
  current: MountWindow,
  count: number,
  sample: FillSample | null,
): MountWindow {
  if (current?.fillTo == null) return current;
  if (current.start <= current.fillTo || current.start >= count) return { start: current.fillTo };
  const valid =
    sample != null &&
    sample.clientHeight > 0 &&
    Number.isFinite(sample.runwayPx) &&
    Number.isFinite(sample.rowsPx) &&
    sample.rowsPx > 0;
  if (!valid) return { start: current.fillTo };
  const missing = sample.clientHeight * FILL_RUNWAY_VIEWPORTS - sample.runwayPx;
  if (!(missing > 0)) return current;
  const perMessage = sample.rowsPx / (count - current.start);
  // Even and at least a pair, so a user turn and its reply arrive together.
  const add = Math.max(2, 2 * Math.ceil(missing / perMessage / 2));
  const start = current.start - add;
  return start <= current.fillTo ? { start: current.fillTo } : { start, fillTo: current.fillTo };
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
