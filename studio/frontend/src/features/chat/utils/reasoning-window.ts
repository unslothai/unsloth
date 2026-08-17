// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How much of a streaming thinking block is actually rendered, kept out of the component so the
// slicing rules stay testable.
//
// WHY THERE IS A WINDOW AT ALL
//
// While a reasoning group streams, `ReasoningText` is `max-h-64` (256px) and `overflow-y-auto`,
// and it mounts the entire reasoning body inside that 256px window. A 90,000-character thinking
// block therefore mounts about 16,000 Shiki token spans and 16,000 DOM elements, of which a
// reader can see roughly fifteen lines. Measured with tests/studio/playwright_reasoning_pane.py:
// the main thread's busy time per streamed chunk grows 1.8x from the start of such a generation
// to its end, and on a machine with no headroom left that turns into 47 fps -> 9 fps.
//
// The window keeps the rendered body bounded, so the cost stops growing with the length of the
// thinking block. It does NOT change what the reader can see at any moment, because the pane is
// 256px tall either way, and the window is many pane-heights deep.
//
// WHY THE START ONLY MOVES IN STEPS
//
// Moving the window one chunk at a time would hand the renderer a string that is not an extension
// of the last one on EVERY chunk. `IncrementalMarkdownCache` (components/assistant-ui/
// streaming-render-schedule.ts) answers a non-prefix by dropping its retained blocks and bumping
// `renderGeneration`, which re-keys Streamdown and remounts the whole body. Once per chunk that
// would be far worse than the problem being solved.
//
// So the start is sticky: it stays where it is until the body has grown a whole SLACK beyond the
// window, and only then jumps forward. Over a 90,000-character generation with a 12,000-character
// window that is about a dozen moves rather than 3,750.

/** Characters of thinking text kept mounted while the block is streaming. */
export const REASONING_WINDOW_CHARS = 12_000;

/**
 * How far past the window the body may grow before the start moves.
 *
 * At 0.5 the rendered body sits between 12,000 and 18,000 characters, and the start moves every
 * 6,000 characters of new content.
 */
export const REASONING_WINDOW_SLACK = 0.5;

/**
 * Whether an offset is outside a fenced code block.
 *
 * Slicing a prefix off Markdown can leave the remainder starting INSIDE a ``` fence, and the
 * renderer would then read the fence's closing marker as an opening one and treat the rest of the
 * thinking block as code. Counting the markers before the cut is the whole test: an even count
 * means every fence opened before the cut was also closed before it.
 *
 * Only markers at the start of a line count, because that is the only place CommonMark begins a
 * fence, and inline triple backticks inside prose are common in reasoning text.
 */
export function isOutsideFence(text: string, offset: number): boolean {
  let fences = 0;
  let index = 0;
  while (index < offset) {
    const next = text.indexOf("```", index);
    if (next === -1 || next >= offset) break;
    const lineStart = next === 0 || text[next - 1] === "\n";
    if (lineStart) fences += 1;
    index = next + 3;
  }
  return fences % 2 === 0;
}

/**
 * The first block boundary at or after `target` that leaves the remainder outside a fence.
 *
 * Returns 0 when there is no such boundary, which renders the body whole. That is the right
 * failure: showing everything is correct and merely slow, whereas cutting into a fence is wrong.
 */
export function alignWindowStart(text: string, target: number): number {
  if (target <= 0) return 0;
  let cursor = target;
  while (cursor < text.length) {
    const boundary = text.indexOf("\n\n", cursor);
    if (boundary === -1) return 0;
    const start = boundary + 2;
    if (isOutsideFence(text, start)) return start;
    cursor = start;
  }
  return 0;
}

/**
 * Where the rendered window should start, given where it starts now.
 *
 * Monotone: the result is never less than `currentStart`, so the body a reader is looking at
 * never grows backwards mid-stream and the renderer never sees the string it just rendered plus
 * a prefix.
 */
export function nextReasoningWindowStart(
  text: string,
  currentStart: number,
  windowChars: number = REASONING_WINDOW_CHARS,
  slack: number = REASONING_WINDOW_SLACK,
): number {
  const rendered = text.length - currentStart;
  if (rendered <= windowChars * (1 + slack)) return currentStart;
  const aligned = alignWindowStart(text, text.length - windowChars);
  return Math.max(currentStart, aligned);
}
