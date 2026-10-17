// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AgentHeightIndex } from "./agent-history-scroll-window";

/**
 * Native scrollbar gesture snapshot. Physical scrollHeight is a *moving*
 * estimate under virtualization. Capture the thumb/track geometry once;
 * convert the release into a logical document destination exactly once.
 *
 * This module is pure: it never owns scrollTop, mounts rows or subscribes
 * to browser events. The viewport's intent-aware scroll provider remains
 * the sole writer of scrollTop.
 */
export type AgentNativeThumbDrag = Readonly<{
  pointerId: number;
  startY: number;
  startScrollTop: number;
  trackTop: number;
  travelPx: number;
  grabOffsetPx: number;
}>;

export type AgentNativeThumbDestination = Readonly<{
  kind: "top" | "bottom" | "position";
  index: number;
  topPx: number;
}>;

type BeginParams = Readonly<{
  pointerId: number;
  pointerY: number;
  viewportTop: number;
  viewportHeight: number;
  scrollTop: number;
  scrollHeight: number;
}>;

export function beginAgentNativeThumbDrag({
  pointerId, pointerY, viewportTop, viewportHeight, scrollTop, scrollHeight,
}: BeginParams): AgentNativeThumbDrag | null {
  if (![pointerId, pointerY, viewportTop, viewportHeight, scrollTop, scrollHeight]
    .every(Number.isFinite) || viewportHeight <= 0 || scrollHeight <= viewportHeight) return null;
  const maxScroll = scrollHeight - viewportHeight;
  // Chromium/WebView2 enforce a minimum native thumb length. This geometry
  // serves only to determine the grab offset; the resulting logical position
  // is independent of subsequent content-height measurements.
  const thumbPx = Math.min(viewportHeight,
    Math.max(18, viewportHeight * viewportHeight / scrollHeight));
  const travelPx = viewportHeight - thumbPx;
  if (travelPx <= 0) return null;
  const thumbTop = viewportTop +
    Math.min(1, Math.max(0, scrollTop / maxScroll)) * travelPx;
  const grabOffsetPx = pointerY - thumbTop;
  // Track clicks are handled natively and are NOT treated as thumb drags.
  // Allow 7px for browser-native thumb geometry and rounding differences.
  if (grabOffsetPx < -7 || grabOffsetPx > thumbPx + 7) return null;
  return {
    pointerId, startY: pointerY, startScrollTop: scrollTop,
    trackTop: viewportTop, travelPx, grabOffsetPx,
  };
}

type EndParams = Readonly<{
  pointerId: number;
  pointerY: number;
  scrollTop: number;
  viewportHeight: number;
  heights: Pick<AgentHeightIndex, "total" | "length" | "indexAt">;
}>;

export function finishAgentNativeThumbDrag(
  drag: AgentNativeThumbDrag,
  { pointerId, pointerY, scrollTop, viewportHeight, heights }: EndParams,
): AgentNativeThumbDestination | null {
  if (pointerId !== drag.pointerId ||
      ![pointerY, scrollTop, viewportHeight].every(Number.isFinite) ||
      viewportHeight <= 0 || heights.length <= 0) return null;
  // A plain click is not an instruction to replace native track behavior.
  if (Math.abs(pointerY - drag.startY) < 3 &&
      Math.abs(scrollTop - drag.startScrollTop) < 1) return null;
  const fraction = Math.min(1, Math.max(0,
    (pointerY - drag.trackTop - drag.grabOffsetPx) / drag.travelPx));
  const maxPx = Math.max(0, heights.total - viewportHeight);
  // Physical endpoints must be exact. Interior coordinates are continuous
  // logical offsets, not recycled scrollTop coordinates from a shrinking DOM.
  if (fraction <= .01) return { kind: "top", index: 0, topPx: 0 };
  if (fraction >= .99) {
    return { kind: "bottom", index: heights.length - 1, topPx: maxPx };
  }
  const topPx = fraction * maxPx;
  return {
    kind: "position",
    index: heights.indexAt(topPx + viewportHeight / 3),
    topPx,
  };
}

/**
 * During a captured native thumb gesture, maintain the scrollbar's virtual
 * document length even when previously unknown row heights are measured.
 *
 * The top prefix and mounted rows retain their live measured geometry. Only
 * the unmounted suffix absorbs height-ledger revisions. This function never
 * changes scrollTop, and cannot make a negative spacer if one unusually tall
 * mounted row already exceeds the initial height estimate.
 */
export function agentNativeThumbBottomSpacerPx(
  virtualTotalAtDragStart: number,
  mountedEndOffset: number,
): number {
  if (!Number.isFinite(virtualTotalAtDragStart) ||
      !Number.isFinite(mountedEndOffset)) return 0;
  return Math.max(0, virtualTotalAtDragStart - mountedEndOffset);
}

/**
 * A captured Chromium scrollbar drag has TWO positions: the native viewport
 * position (which the user watched during the gesture), and the pointer
 * coordinate. They can diverge in WebView2 even with a fixed scrollHeight.
 *
 * For an interior release, the browser's actual painted viewport wins.
 * Never resnap it to the pointer coordinate merely because the two differ.
 * An explicit track endpoint remains an instruction to reach the first/last
 * message, preserving R3's one-gesture boundary guarantee.
 *
 * Pure decision: this function never writes scrollTop or mounts DOM rows.
 */
export type AgentNativeThumbReleaseAction =
  | Readonly<{ kind: "top" | "bottom"; index: number }>
  | Readonly<{ kind: "keep" | "recover"; index: number }>;

export function resolveAgentNativeThumbRelease(
  pointerDestination: AgentNativeThumbDestination,
  state: Readonly<{
    nativeScrollTop: number;
    nativeScrollHeight: number;
    viewportHeight: number;
    nativeIndex: number;
    rowVisibleAtNativePosition: boolean;
    count: number;
  }>,
): AgentNativeThumbReleaseAction {
  const { nativeScrollTop, nativeScrollHeight, viewportHeight, nativeIndex,
    rowVisibleAtNativePosition, count } = state;
  const last = Math.max(0, count - 1);
  const index = Number.isSafeInteger(nativeIndex)
    ? Math.min(last, Math.max(0, nativeIndex)) : 0;
  if (pointerDestination.kind === "top") return { kind: "top", index: 0 };
  if (pointerDestination.kind === "bottom") return { kind: "bottom", index: last };
  // Chromium has physically reached an endpoint even if native pointer
  // capture reports the finger a few pixels short of the track boundary.
  if (Number.isFinite(nativeScrollTop) && nativeScrollTop <= 1 && index === 0) {
    return { kind: "top", index: 0 };
  }
  if (Number.isFinite(nativeScrollHeight) && Number.isFinite(nativeScrollTop) &&
      Number.isFinite(viewportHeight) &&
      nativeScrollHeight - nativeScrollTop - viewportHeight <= 2 &&
      index === last) return { kind: "bottom", index: last };
  return rowVisibleAtNativePosition
    ? { kind: "keep", index }
    : { kind: "recover", index };
}

/**
 * While the native thumb is still held, mount the logical row currently
 * under Chromium's real scroll position. This authorization is scoped to
 * the captured thumb gesture: normal wheel/anchor scroll events continue to
 * use one-shot reader intent epochs.
 *
 * A progressive DOM-budget window may still be filling in. Retarget only
 * if the thumb moved OUTSIDE its eventual destination; otherwise allow
 * the existing 1-2-row-per-frame staging process to finish.
 */
export function agentNativeThumbLiveTarget(
  current: Readonly<{ start: number; end: number }>,
  stagedDestination: Readonly<{ start: number; end: number }> | null,
  index: number,
  count: number,
): number | null {
  if (!Number.isSafeInteger(index) || !Number.isSafeInteger(count) ||
      count <= 0 || index < 0 || index >= count) return null;
  const desired = stagedDestination ?? current;
  if (index >= desired.start && index < desired.end) return null;
  return index;
}
