// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A single navigation intent owns passive scroll sampling during a search reveal.
 * Epochs make a superseded DOM/layout callback incapable of ending a newer intent.
 */
export type AgentHistoryNavigation =
  | Readonly<{ kind: "reader"; epoch: number }>
  | Readonly<{ kind: "search"; epoch: number; row: number; positioned: boolean }>;

export function initialAgentHistoryNavigation(): AgentHistoryNavigation {
  return { kind: "reader", epoch: 0 };
}

export function requestAgentHistoryReveal(
  current: AgentHistoryNavigation,
  row: number,
  count: number,
): AgentHistoryNavigation {
  if (!Number.isSafeInteger(row) || row < 0 || row >= count) return current;
  return { kind: "search", epoch: current.epoch + 1, row, positioned: false };
}

export function markAgentHistoryRevealPositioned(
  current: AgentHistoryNavigation,
  epoch: number,
): AgentHistoryNavigation {
  if (current.kind !== "search" || current.epoch !== epoch) return current;
  return { ...current, positioned: true };
}

export function completeAgentHistoryReveal(
  current: AgentHistoryNavigation,
  epoch: number,
  visible: boolean,
): AgentHistoryNavigation {
  if (current.kind !== "search" || current.epoch !== epoch || !current.positioned || !visible) return current;
  return { kind: "reader", epoch };
}

export function interruptAgentHistoryReveal(current: AgentHistoryNavigation): AgentHistoryNavigation {
  return current.kind === "search" ? { kind: "reader", epoch: current.epoch + 1 } : current;
}

export function allowsPassiveAgentScroll(current: AgentHistoryNavigation): boolean {
  return current.kind === "reader";
}

/**
 * Only a fresh reader gesture may authorize a new passive window transition.
 * Browser scroll events alone are not proof of reader intent: height measurements,
 * native scroll anchoring and our own viewport correction all produce them.
 *
 * The gesture is consumed when a window transition is accepted, not for every
 * scroll event. This lets one drag traverse a stable window while ensuring that
 * later anchor/layout echoes cannot immediately reverse its chosen destination.
 */
export type AgentReaderScrollIntent = Readonly<{
  epoch: number;
  sampledEpoch: number;
}>;

export function initialAgentReaderScrollIntent(): AgentReaderScrollIntent {
  return { epoch: 0, sampledEpoch: 0 };
}

export function requestAgentReaderScrollIntent(
  current: AgentReaderScrollIntent,
): AgentReaderScrollIntent {
  return { ...current, epoch: current.epoch + 1 };
}

export function allowsAgentReaderScrollIntent(
  current: AgentReaderScrollIntent,
  epoch: number,
): boolean {
  return epoch === current.epoch && epoch > current.sampledEpoch;
}

export function consumeAgentReaderScrollIntent(
  current: AgentReaderScrollIntent,
  epoch: number,
): AgentReaderScrollIntent {
  return allowsAgentReaderScrollIntent(current, epoch)
    ? { ...current, sampledEpoch: epoch } : current;
}

/** A consumed wheel/pointer epoch may finish at the physical edge after React
 * committed an intermediate virtual window. Permit only a short-lived edge
 * reconciliation while that same real gesture is recent, never passive layout
 * or anchor scroll events elsewhere in the transcript.
 */
export function allowsRecentAgentBoundaryReconcile(
  intent: AgentReaderScrollIntent,
  lastGestureAtMs: number,
  nowMs: number,
  rangeStart: number,
  rangeEnd: number,
  total: number,
  atTop: boolean,
  atBottom: boolean,
): boolean {
  return intent.epoch > 0 && intent.sampledEpoch === intent.epoch &&
    nowMs >= lastGestureAtMs && nowMs - lastGestureAtMs <= 750 &&
    ((atTop && rangeStart > 0) || (atBottom && rangeEnd < total));
}
