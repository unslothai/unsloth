// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How much of a thinking block is actually mounted, kept out of the component so the slicing
// rules stay testable.
//
// WHY A WINDOW, AND WHY UNMOUNTING RATHER THAN SKIPPING PAINT
//
// While a reasoning group streams, `ReasoningText` is `max-h-64` and `overflow-y-auto`: a 256px
// window over a body that reaches tens of thousands of pixels. The whole body is mounted inside
// it. Measured on a reported generation, the page's frame rate tracks the number of mounted
// nodes in that pane with r = -0.88, and it does so in sample windows with 49 to 63 DOM
// mutations in five seconds -- that is, when nothing is being built, tokenized or reconciled.
// The cost is the standing presence of the nodes.
//
// That is why this unmounts rather than applying containment. `content-visibility: auto` on the
// pane's code blocks was measured on the same fixture and did NOTHING: 60.2% late main-thread
// busy against the baseline's 57.4%, with node count unchanged at 17,440 against 17,482.
// Skipping layout and paint for offscreen content buys nothing when the cost is the nodes
// existing. The two look similar from a distance and the measurement says they are not the same
// thing at all.
//
// NOTHING IS EVER LOST, AND THE WINDOW ONLY EXISTS WHILE NOBODY IS READING PAST IT
//
// The window applies in exactly one situation: the block is streaming AND the reader is pinned to
// the bottom of the pane, watching the newest text, unable to see what is above. The moment they
// scroll back, the whole body is restored in one step and the window is off for the rest of the
// round. A finished block is never windowed at all.
//
// It restores in ONE step rather than widening a bit at a time, and that is forced by the
// renderer rather than chosen. Streamdown 2.5.0 keys its blocks positionally --
// `Rn = J.map((k, A) => `${_}-${A}`)` with `_` a `useId()`, applied as `jsx(f, {...}, Rn[A])` --
// and it also passes `index: A` and compares `e.index !== t.index` in the Block memo. Prepending
// blocks shifts both for every block that was already mounted, so a widen remounts the entire
// body no matter how little it adds. Measured: four widens cost frames of 207, 280, 646 and
// 846ms. Neither the React key nor the index is reachable through `BlockComponent` or
// `parseMarkdownIntoBlocksFn`, which are the only seams Streamdown exposes. Appending is free;
// prepending is a full remount. That asymmetry is why the window grows downward or not at all.

/** Characters of thinking text kept mounted while the block streams and the reader is at the end. */
export const REASONING_WINDOW_CHARS = 12_000;

/**
 * How far past the window the body may grow before the start moves.
 *
 * At 0.5 the mounted body sits between 12,000 and 18,000 characters and the start moves every
 * 6,000 characters. It moves in steps rather than continuously because
 * `IncrementalMarkdownCache` answers a string that is not an extension of the last one by
 * dropping its retained blocks and re-keying Streamdown, which remounts the body. Once every
 * 6,000 characters that is affordable; once per 24-character chunk it would be far worse than
 * the problem being solved.
 */
export const REASONING_WINDOW_SLACK = 0.5;

/**
 * Whether an offset is outside a fenced code block.
 *
 * Slicing Markdown at an arbitrary offset can leave the remainder starting INSIDE a ``` fence,
 * and the renderer would then read that fence's closing marker as an opening one and treat the
 * rest of the thinking block as code. Counting the markers before the offset is the whole test:
 * an even count means every fence opened before it was also closed before it.
 *
 * Only markers at the start of a line count, because that is the only place CommonMark begins a
 * fence and inline triple backticks are common in reasoning prose.
 */
export function isOutsideFence(text: string, offset: number): boolean {
  let fences = 0;
  let index = 0;
  while (index < offset) {
    const next = text.indexOf("```", index);
    if (next === -1 || next >= offset) break;
    if (next === 0 || text[next - 1] === "\n") fences += 1;
    index = next + 3;
  }
  return fences % 2 === 0;
}

/**
 * The first block boundary at or after `target` that leaves the remainder outside a fence.
 *
 * Returns 0 when there is none, which mounts the whole body. That is the right failure: showing
 * everything is correct and merely slow, whereas cutting into a fence is wrong.
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
 * Where the mounted window should start, given where it starts now, while the reader is at the
 * end of a streaming block.
 *
 * Monotone: never less than `currentStart`, so the mounted body never grows backwards on its own
 * and the renderer never sees the string it just rendered with a prefix glued back on.
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
