// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The bookkeeping behind the streaming reasoning pane's block window.
 *
 * The pane is 256px tall while a model is thinking and it holds the WHOLE reasoning body, which
 * on a long generation reaches tens of thousands of nodes and about 16,000 Shiki highlight spans.
 * Only a few hundred pixels of that are ever on screen. The window renders a contiguous suffix of
 * the block list and replaces everything above it with one spacer of exactly the height the
 * dropped blocks had.
 *
 * Nothing here touches the Markdown SOURCE STRING. The window is expressed purely as block
 * INDICES, and Streamdown decides what a block is; see block-window-context.tsx. That is the
 * whole point: a windowing scheme that slices the source has to predict where the renderer
 * considers the document divisible, and that judgement changes retroactively (a GFM footnote
 * definition arriving late collapses many blocks into one).
 *
 * COORDINATES. Every offset here is a "content offset": the distance in CSS pixels from the top
 * of the pane's scrollable content to the top of a block's marker element. It is derived from two
 * rectangles read in the same frame,
 *
 *     contentOffset(marker) = marker.getBoundingClientRect().top - origin.getBoundingClientRect().top
 *
 * where `origin` is the spacer, which is always the first thing in the content. Because both
 * rectangles come from the same frame, the value is immune to whatever else changed in that
 * frame, including the text streaming in at the bottom. A `scrollHeight` delta could not be used
 * for the same purpose: streaming appends at the bottom in the same commit, so the delta conflates
 * the removed height with the appended height and the spacer inflates without bound.
 */

/**
 * How much rendered content is kept mounted above the top of the visible pane.
 *
 * The pane is 256px tall, so this is six pane heights. It is the distance a reader can travel
 * upwards between two observations before reaching content that is not mounted, and the only
 * thing that reports a reader scrolling (an IntersectionObserver, since scroll events are not
 * used) fires 400px before the spacer would become visible. Six pane heights is therefore about
 * 1.1k pixels of slack beyond the observer's own warning, which no wheel or trackpad gesture
 * covers in a single frame. A scrollbar drag can, and that case is documented in
 * block-window-context.tsx rather than papered over.
 */
export const RETAIN_ABOVE_PX = 1536;

/**
 * The margin by which the observer's root is grown, so a block mounts BEFORE it enters view.
 *
 * 400px is about one and a half pane heights. It only has to be large enough that an observation
 * arrives before the reader can see the spacer, and RETAIN_ABOVE_PX is already four times it.
 */
export const WINDOW_ROOT_MARGIN_PX = 400;

/**
 * How far behind the newest block a content change is still considered part of the live edge.
 *
 * Streamdown re-parses the tail of the document on every token, so the last few blocks legitimately
 * change shape as text arrives. A change further back than this means the parse was redone behind
 * the live edge and every frozen height is suspect. Matches ROLLBACK_BLOCKS in
 * streaming-render-schedule.ts, which is the margin the incremental cache itself keeps.
 */
export const LIVE_EDGE_BLOCKS = 8;

/**
 * How close to the bottom the pane counts as pinned there.
 *
 * Shared with the reasoning pane's autoscroll, which uses the same number to decide whether a
 * reader who scrolled up has come back. The window's scroll compensation has to agree with it: a
 * correction written while the pane is pinned would read to that handler as the reader scrolling
 * up, and would detach the autoscroll the reader never asked to detach.
 */
export const PANE_BOTTOM_THRESHOLD_PX = 24;

export type BlockWindowState = {
  /** Lowest block index that is rendered. Everything below it is the spacer. */
  start: number;
  /** Height of the spacer, in CSS pixels. Always `contentOffset(start)`. */
  spacerHeight: number;
  /** Content offset per block index, frozen at the last commit that measured it. */
  readonly offsets: Map<number, number>;
  /** Last seen block content per index, used to notice a retroactive re-parse. */
  readonly contents: Map<number, string>;
  /** Highest index that has ever reported content. The live edge. */
  highestIndex: number;
};

export function createBlockWindowState(): BlockWindowState {
  return {
    start: 0,
    spacerHeight: 0,
    offsets: new Map(),
    contents: new Map(),
    highestIndex: -1,
  };
}

/**
 * Drop every frozen height and remount the whole document on the next frame.
 *
 * A frozen height that is never revalidated cannot exist here: the two things that can invalidate
 * one -- the pane changing WIDTH, and the renderer re-segmenting the document behind the live edge
 * -- both invalidate ALL of them at once, and both answer with this.
 */
export function resetBlockWindow(state: BlockWindowState): void {
  state.start = 0;
  state.spacerHeight = 0;
  state.offsets.clear();
  state.contents.clear();
  state.highestIndex = -1;
}

/** Record where a block's marker sits, in content coordinates. */
export function recordBlockOffset(
  state: BlockWindowState,
  index: number,
  offset: number,
): void {
  state.offsets.set(index, offset);
}

/**
 * The lowest block index that has to stay mounted for the reader.
 *
 * `viewTop` is the content offset of the top of the visible pane, i.e. the pane's scroll position
 * expressed in the same coordinates as the recorded offsets. Only indices with a recorded offset
 * are candidates, so the caller can always turn the answer into a spacer height it measured
 * itself. Index 0 is always a candidate, with offset 0 by definition.
 */
export function chooseBlockWindowStart(
  state: BlockWindowState,
  viewTop: number,
  retainAbovePx: number = RETAIN_ABOVE_PX,
): number {
  const ceiling = viewTop - retainAbovePx;
  if (!(ceiling > 0)) {
    return 0;
  }
  let start = 0;
  for (const [index, offset] of state.offsets) {
    if (offset <= ceiling && index > start) {
      start = index;
    }
  }
  return start;
}

/**
 * The spacer height that goes with `start`, taken from the map as it stood BEFORE the blocks are
 * dropped.
 *
 * This is the prescribed "measure then freeze", written absolutely rather than incrementally.
 * `spacerHeight(s') - spacerHeight(s)` is exactly `offset(s') - offset(s)`, the height of the
 * removed range including its collapsed margins, and because both come from the same
 * pre-mutation frame the difference cannot drift. Writing it absolutely also means a window that
 * moves back up restores the earlier height exactly instead of accumulating rounding.
 */
export function blockWindowSpacerHeight(
  state: BlockWindowState,
  start: number,
): number {
  return start <= 0 ? 0 : (state.offsets.get(start) ?? state.spacerHeight);
}

/** Move the window. Returns whether anything actually changed. */
export function setBlockWindowStart(
  state: BlockWindowState,
  start: number,
): boolean {
  const spacerHeight = blockWindowSpacerHeight(state, start);
  if (state.start === start && state.spacerHeight === spacerHeight) {
    return false;
  }
  state.start = start;
  state.spacerHeight = spacerHeight;
  return true;
}

/** Whether a block index is inside the window. The window is a suffix, so this is a comparison. */
export function isBlockMounted(
  state: BlockWindowState,
  index: number,
): boolean {
  return index >= state.start;
}

/**
 * Record a block's content and report whether it proves the parse was redone behind the live edge.
 *
 * The window is keyed on block INDEX, so a re-segmentation moves the document under the frozen
 * heights even though it can never corrupt the text (the source string is never touched). The
 * signal is a block that is not near the live edge changing content. The caller answers by
 * resetting, which revalidates every height at once.
 */
export function recordBlockContent(
  state: BlockWindowState,
  index: number,
  content: string,
  liveEdgeBlocks: number = LIVE_EDGE_BLOCKS,
): boolean {
  const previous = state.contents.get(index);
  const wasBehindLiveEdge =
    previous !== undefined && index < state.highestIndex - liveEdgeBlocks;
  state.contents.set(index, content);
  if (index > state.highestIndex) {
    state.highestIndex = index;
  }
  return wasBehindLiveEdge && previous !== content;
}

/**
 * The block indices whose mounted state differs between two window starts.
 *
 * Only these have to be told to re-render, which is what keeps a window move off the other
 * thousands of block components.
 */
export function blockWindowFlippedRange(
  previousStart: number,
  nextStart: number,
): { from: number; to: number } {
  return {
    from: Math.min(previousStart, nextStart),
    to: Math.max(previousStart, nextStart),
  };
}

/**
 * The contents the window would mount, given the whole block list.
 *
 * Exported for the invariant it exists to state: what is mounted is always a contiguous SUFFIX of
 * the block list the renderer produced, never a re-cut of the source string.
 */
export function mountedBlockContents(
  blocks: readonly string[],
  start: number,
): string[] {
  return blocks.slice(Math.max(0, start));
}

/** Whether `candidate` is a contiguous suffix of `blocks`. */
export function isBlockSuffix(
  blocks: readonly string[],
  candidate: readonly string[],
): boolean {
  if (candidate.length > blocks.length) {
    return false;
  }
  const offset = blocks.length - candidate.length;
  for (let index = 0; index < candidate.length; index += 1) {
    if (blocks[offset + index] !== candidate[index]) {
      return false;
    }
  }
  return true;
}
