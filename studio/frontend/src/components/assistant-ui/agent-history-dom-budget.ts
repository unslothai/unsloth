// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Budget targets apply to message descendants, not the entire application DOM. */
import type { AgentScrollWindow } from "./agent-history-scroll-window";

export {
  AGENT_BUFFER_MODES, isAgentBufferMode, agentBufferRowLimit, agentBufferDomLevel,
} from "./agent-history-config.ts";
export type { AgentBufferMode, AgentDomLevel } from "./agent-history-config.ts";
export const AGENT_DOM_TARGETS = { low: 5_000, medium: 8_000, high: 12_000 } as const;
/** Adaptive Fixed grows above the legacy 5-row minimum only when the viewport
 * needs more coverage. These are hard *policy* bounds: a single mandatory
 * extremely large tool row can still exceed the descendant budget. */
export const AGENT_FIXED_VIEWPORT_MAX_ROWS = 32;
export const AGENT_FIXED_VIEWPORT_DOM_TARGET = AGENT_DOM_TARGETS.medium;
/** Unknown rows are charged conservatively until observed, so long chats start small. */
export const AGENT_UNKNOWN_ROW_COST = 1_000;
export const AGENT_DOM_MAX_ROWS = 32;

export class AgentDomCostIndex {
  private costs = new Map<number, number>();
  private size: number;
  constructor(size: number) { this.size = size; }
  resize(size: number) {
    this.size = size;
    for (const key of this.costs.keys()) if (key >= size) this.costs.delete(key);
  }
  measure(row: number, descendants: number): boolean {
    if (!Number.isSafeInteger(row) || row < 0 || row >= this.size ||
      !Number.isFinite(descendants) || descendants < 0) return false;
    const cost = Math.max(1, Math.round(descendants));
    if (this.costs.get(row) === cost) return false;
    this.costs.set(row, cost);
    return true;
  }
  measured(row: number): boolean { return this.costs.has(row); }
  cost(row: number): number { return this.costs.get(row) ?? AGENT_UNKNOWN_ROW_COST; }
}

/**
 * Always include the target. A single giant message can exceed the target;
 * excluding it would break navigation. Expand contiguously up to the budget,
 * preferring rows following the target so the reader can continue downwards.
 */
export function agentBudgetWindow(
  count: number,
  target: number,
  costs: AgentDomCostIndex,
  budget: number,
  maxRows = AGENT_DOM_MAX_ROWS,
): Readonly<{ start: number; end: number }> {
  if (count <= 0) return { start: 0, end: 0 };
  const at = Number.isFinite(target)
    ? Math.max(0, Math.min(count - 1, Math.floor(target)))
    : count - 1;
  let start = at, end = at + 1, used = costs.cost(at);
  // Unknown rows can contain arbitrarily large tool outputs. Admit only one
  // unmeasured row at a time, then rebalance after measuring its real DOM cost.
  let unknown = costs.measured(at) ? 0 : 1;
  const eligible = (row: number, cost: number) =>
    used + cost <= budget && (costs.measured(row) || unknown === 0);
  const add = (row: number, cost: number) => {
    used += cost;
    if (!costs.measured(row)) unknown++;
  };
  while (end - start < maxRows) {
    const previous = start > 0 ? costs.cost(start - 1) : Infinity;
    const next = end < count ? costs.cost(end) : Infinity;
    if (previous === Infinity && next === Infinity) break;
    // Favor earlier rows at the tail, forward progress elsewhere.
    const preferPrevious = end === count || (start > 0 && previous < next);
    if (preferPrevious && eligible(start - 1, previous)) {
      add(start - 1, previous); start--;
    } else if (eligible(end, next)) {
      add(end, next); end++;
    } else if (eligible(start - 1, previous)) {
      add(start - 1, previous); start--;
    } else break;
  }
  return { start, end };
}

/**
 * Budget a single genuinely adjacent physical-viewport recovery row while
 * retaining already mounted rows, instead of collapsing the whole window to
 * an unknown one-row seek. Admit only one previously unknown row at a time.
 *
 * The mandatory new row can exceed the estimated DOM budget once measured,
 * exactly like agentBudgetWindow; otherwise drop rows from the OPPOSITE edge
 * before committing. No native scroll position mutations.
 */
export function agentPhysicalNeighborWindow(
  current: AgentScrollWindow,
  target: number,
  count: number,
  costs: AgentDomCostIndex,
  budget: number,
  maxRows: number,
): AgentScrollWindow | null {
  if (!Number.isSafeInteger(target) || !Number.isSafeInteger(count) ||
      target < 0 || target >= count || !Number.isFinite(budget) || budget <= 0 ||
      !Number.isSafeInteger(maxRows) || maxRows < 1 ||
      current.start < 0 || current.end > count || current.start >= current.end) return null;
  const prepend = target === current.start - 1;
  const append = target === current.end;
  if (!prepend && !append) return null;
  // Never stage a second unmeasured/heavy tool row concurrently.
  if (!costs.measured(target)) {
    for (let row = current.start; row < current.end; row++) {
      if (!costs.measured(row)) return null;
    }
  }
  let start = prepend ? target : current.start;
  let end = append ? target + 1 : current.end;
  let used = 0;
  for (let row = start; row < end; row++) used += costs.cost(row);
  const limit = Math.min(AGENT_DOM_MAX_ROWS, maxRows);
  while ((end - start > limit || used > budget) && end - start > 1) {
    if (prepend) { end--; used -= costs.cost(end); }
    else { used -= costs.cost(start); start++; }
  }
  return { start, end };
}

/**
 * When measurements merely admit additional neighboring messages, keep the
 * reader's already-filled viewport stable rather than moving text every frame.
 * Contracting an over-budget window or revealing an absent target is mandatory.
 */
export function needsAgentBudgetRebalance(
  current: AgentScrollWindow,
  proposed: AgentScrollWindow,
  target: number,
  mountedContentHeight: number,
  viewportHeight: number,
): boolean {
  if (current.start === proposed.start && current.end === proposed.end) return false;
  if (!(current.start <= target && target < current.end)) return true;
  if (proposed.start > current.start || proposed.end < current.end) return true;
  // No speculative expansion above or below a reader-visible viewport. This
  // is what stopped legacy mode from shifting text during its idle phase.
  return Number.isFinite(mountedContentHeight) && Number.isFinite(viewportHeight) &&
    mountedContentHeight + 16 < viewportHeight;
}

/**
 * Budget-independent geometry guard for a small *native* wheel step toward
 * older history. Preload only an immediately adjacent row before its spacer
 * enters the viewport; heavy-row admission stays with PhysicalNeighborWindow.
 * Larger wheel leaps use the existing staged seek instead.
 */
export function agentWheelPrefetchTarget(
  wheelDeltaY: number,
  firstRowViewportTop: number,
  viewportTop: number,
  viewportHeight: number,
  mountedStart: number,
): number | null {
  if (!Number.isFinite(wheelDeltaY) || wheelDeltaY >= 0 || wheelDeltaY < -720 ||
      !Number.isFinite(firstRowViewportTop) || !Number.isFinite(viewportTop) ||
      !Number.isFinite(viewportHeight) || viewportHeight <= 0 ||
      !Number.isSafeInteger(mountedStart) || mountedStart <= 0) return null;
  const lead = Math.max(96, Math.min(200, viewportHeight * 0.18));
  const predictedFirstRowTop = firstRowViewportTop - viewportTop - wheelDeltaY;
  return predictedFirstRowTop >= -lead ? mountedStart - 1 : null;
}
