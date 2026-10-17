// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Normal-flow virtualization: only window boundaries change; rows are never absolutely positioned. */
/** New opt-in preference; 32 remains the legacy capacity/headroom special case. */
import { AGENT_INITIAL_ROWS, AGENT_MAX_ROWS } from "./agent-history-config.ts";
export {
  AGENT_DEFAULT_ROWS, AGENT_INITIAL_ROWS, AGENT_MAX_ROWS,
  AGENT_WINDOW_SIZES, isAgentWindowSize,
} from "./agent-history-config.ts";
export type { AgentWindowSize } from "./agent-history-config.ts";
export const AGENT_ROW_ESTIMATE_PX = 720;
export type AgentScrollWindow = Readonly<{ start: number; end: number }>;

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export function latestAgentWindow(
  count: number,
  windowRows: number = AGENT_INITIAL_ROWS,
): AgentScrollWindow {
  return { start: Math.max(0, count - windowRows), end: count };
}

/** Reuse existing mounted rows until the reader nears an edge; never shift on each token. */
export function agentWindowAtIndex(
  count: number,
  index: number,
  current?: AgentScrollWindow,
  windowRows: number = AGENT_INITIAL_ROWS,
  strict: boolean = false,
): AgentScrollWindow {
  if (count <= windowRows) return { start: 0, end: count };
  const target = clamp(Math.floor(index), 0, count - 1);
  const maxRows = !strict && windowRows === AGENT_INITIAL_ROWS ? AGENT_MAX_ROWS : windowRows;
  // A partially staged window is not a stable reader window. If staging was
  // interrupted, recalculate a full window rather than retaining a lone row.
  if (current && current.start >= 0 && current.end <= count &&
      current.end > current.start && current.end - current.start >= windowRows &&
      current.end - current.start <= maxRows &&
      target >= current.start + Math.min(8, Math.floor((current.end-current.start)/4)) &&
      target < current.end - Math.min(8, Math.floor((current.end-current.start)/4))) {
    return current;
  }
  if (target === 0) return { start: 0, end: windowRows };
  if (target === count - 1) return latestAgentWindow(count, windowRows);
  const lead = windowRows === AGENT_INITIAL_ROWS ? 12 : Math.floor((windowRows - 1) / 2);
  const start = clamp(target - lead, 0, count - windowRows);
  return { start, end: start + windowRows };
}

/**
 * Admit at most two previously unmounted messages per animation frame.
 * One extra DOM/style/layout pass per row raised total work during 200k/400k
 * stress runs; admitting a small batch balances latency against long tasks.
 * Only use for a distant passive scroll seek.
 */
export function nextProgressiveAgentWindow(
  current: AgentScrollWindow,
  destination: AgentScrollWindow,
  maxAddedRows = 2,
): AgentScrollWindow {
  if (!Number.isSafeInteger(maxAddedRows) || maxAddedRows < 1) return current;
  let { start, end } = current;
  for (let added = 0; added < maxAddedRows; added++) {
    if (start > destination.start) start--;
    else if (end < destination.end) end++;
    else break;
  }
  return { start, end };
}

/** Appending at the tail never shifts the mounted rows on the first append. */
export function agentWindowOnAppend(
  previous: AgentScrollWindow,
  oldCount: number,
  count: number,
  following: boolean,
  windowRows: number = AGENT_INITIAL_ROWS,
  strict: boolean = false,
): AgentScrollWindow {
  if (count < oldCount) return latestAgentWindow(count, windowRows);
  if (oldCount === 0) return latestAgentWindow(count, windowRows);
  if (!following || previous.end !== oldCount) return previous;
  // Only the legacy default may grow to 96 rows. Small windows are strict bounds.
  const maxRows = !strict && windowRows === AGENT_INITIAL_ROWS ? AGENT_MAX_ROWS : windowRows;
  if (count - previous.start > maxRows) return latestAgentWindow(count, windowRows);
  return { start: previous.start, end: count };
}

/**
 * Resolve a scroll position into the index of a logical transcript row.
 * The top spacer is the PREFIX of the virtual document: its top is the
 * document origin, not the position of the first mounted row. Subtracting
 * offset(window.start) here double-counts the prefix and pins large histories
 * to the last messages even while the reader scrolls upwards.
 */
export function agentIndexAtScrollPosition(
  heights: AgentHeightIndex,
  topSpacerViewportTop: number,
  viewportTop: number,
  scrollTop: number,
  viewportHeight: number,
): number {
  const documentOrigin = topSpacerViewportTop - viewportTop + scrollTop;
  return heights.indexAt(scrollTop - documentOrigin + viewportHeight / 3);
}

/**
 * Reconcile physical wheel coverage after unknown-height measurements.
 * A mounted range can be taller than the viewport yet start well below its
 * physical top (or end above its physical bottom). Return a target only when
 * significant uncovered space points OUTSIDE the mounted range.
 * Pure selection: this function never writes the native scroll position.
 */
export function agentViewportCoverageRecoveryTarget(
  current: AgentScrollWindow,
  topIndex: number,
  bottomIndex: number,
  topGapPx: number,
  bottomGapPx: number,
  viewportHeightPx: number,
): number | null {
  if (!Number.isFinite(viewportHeightPx) || viewportHeightPx <= 0) return null;
  const threshold = viewportHeightPx * 0.2;
  if (Number.isFinite(topGapPx) && topGapPx > threshold &&
      Number.isSafeInteger(topIndex) && topIndex >= 0 &&
      topIndex < current.start) return topIndex;
  if (Number.isFinite(bottomGapPx) && bottomGapPx > threshold &&
      Number.isSafeInteger(bottomIndex) && bottomIndex >= current.end) return bottomIndex;
  return null;
}

/**
 * Height index: initial estimates + sparse corrections for measured rows.
 * It gives O(log N) offset lookup, and does not re-position mounted rows on append.
 * A growing capacity doubles rather than rebuilding on each agent message.
 */
export class AgentHeightIndex {
  private lengthValue: number;
  private capacity: number;
  private tree: Float64Array;
  private readonly heights = new Map<number, number>();
  readonly estimate: number;

  constructor(count: number, estimate = AGENT_ROW_ESTIMATE_PX) {
    if (!Number.isSafeInteger(count) || count < 0 || !Number.isFinite(estimate) || estimate <= 0) {
      throw new RangeError("Invalid agent history height index");
    }
    this.lengthValue = count;
    this.estimate = estimate;
    this.capacity = Math.max(1, 2 ** Math.ceil(Math.log2(Math.max(1, count))));
    this.tree = new Float64Array(this.capacity + 1);
  }

  get length(): number { return this.lengthValue; }

  resize(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0) throw new RangeError("Invalid count");
    if (count < this.lengthValue) {
      for (const key of this.heights.keys()) if (key >= count) this.heights.delete(key);
      this.rebuild(Math.max(this.capacity, count));
    } else if (count > this.capacity) {
      this.rebuild(Math.max(count, this.capacity * 2));
    }
    this.lengthValue = count;
  }

  /** Returns true only when a measurable height actually changed. */
  measure(index: number, px: number): boolean {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.lengthValue ||
        !Number.isFinite(px) || px < 0) return false;
    const height = Math.max(0, Math.round(px * 10) / 10);
    const old = this.heights.get(index) ?? this.estimate;
    if (Math.abs(old - height) < 0.5) return false;
    this.heights.set(index, height);
    this.add(index, height - old);
    return true;
  }

  height(index: number): number { return this.heights.get(index) ?? this.estimate; }

  offset(index: number): number {
    const end = clamp(Math.floor(index), 0, this.lengthValue);
    let delta = 0;
    for (let i = end; i > 0; i -= i & -i) delta += this.tree[i];
    return end * this.estimate + delta;
  }

  get total(): number { return this.offset(this.lengthValue); }

  /** The original index covering the given estimated document coordinate. */
  indexAt(offsetPx: number): number {
    if (this.lengthValue === 0) return 0;
    const target = Math.max(0, Number.isFinite(offsetPx) ? offsetPx : 0);
    let low = 0;
    let high = this.lengthValue;
    while (low < high) {
      const mid = low + Math.floor((high - low + 1) / 2);
      if (this.offset(mid) <= target) low = mid;
      else high = mid - 1;
    }
    return clamp(low, 0, this.lengthValue - 1);
  }

  private add(index: number, delta: number): void {
    for (let i = index + 1; i <= this.capacity; i += i & -i) this.tree[i] += delta;
  }

  private rebuild(capacity: number): void {
    this.capacity = capacity;
    this.tree = new Float64Array(capacity + 1);
    for (const [index, height] of this.heights) {
      if (index < this.lengthValue) this.add(index, height - this.estimate);
    }
  }
}
