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
// `Rn = J.map((k, A) => `${_}-${A}`)` with `_` a `useId()`, applied as `jsx(f, {...}, Rn[A])`.
//
// The cost of prepending is NOT a remount, and it is worth saying so because the obvious reading
// of positional keys is wrong. Prepend X to [A, B, C] and the keys stay `id-0`..`id-2` and gain
// `id-3`; React matches by key, so every instance survives and only its `content` prop changes.
// Measured directly against the shipped dist with a mount-counting BlockComponent: prepending one
// block costs 2 mounts and ZERO unmounts, exactly the same as appending one. Upstream chose these
// keys deliberately and says so in `index.tsx`: a content-hash key would remount the LAST block on
// every streamed token, which is their common path.
//
// What prepending actually costs is that EVERY block's content prop changes, so every memo returns
// false and every block re-parses and re-renders through the react-markdown pipeline, plus real
// subtree replacement wherever the element type differs at its new position. Measured: four widens
// cost frames of 207, 280, 646 and 846ms.
//
// And it is not only slow. Streamdown's DEFAULT Markdown components are memoised on the POSITION
// of the node, `e.className === t.className && sameNodePosition(e.node, t.node)` over start and
// end line and column only (vercel/streamdown#570, open). A block whose replacement happens to
// occupy the same span keeps the output it already had, so the reader is shown text from elsewhere
// in the reasoning. That is the reason this restores once and then stops, and it is why the
// renderer is re-keyed on every window move rather than merely re-rendered.
//
// Neither the block key nor the index is reachable through `BlockComponent` or
// `parseMarkdownIntoBlocksFn`, which are the only seams Streamdown exposes.

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
 * A line with its container prefixes removed, so a fence opened inside a list item or a quote is
 * still seen as a fence.
 *
 * CommonMark opens a fence on `- ```js` exactly as it does on `  ```js`: the list marker is
 * container structure and the fence begins in the item's content. Matching only the indented form
 * catches the CLOSING marker of such a block and not its opener, which is worse than matching
 * neither, because the scanner then believes a fence opens where one closes.
 */
const CONTAINER_PREFIX = / {0,3}(?:>|(?:[-+*]|\d{1,9}[.)])(?=[ \t]))[ \t]*/y;

function stripContainers(line: string): string {
  let index = 0;
  for (;;) {
    CONTAINER_PREFIX.lastIndex = index;
    const match = CONTAINER_PREFIX.exec(line);
    if (!match || match[0].length === 0) return line.slice(index);
    index += match[0].length;
  }
}

/**
 * The fence marker a line opens or closes, or null if the line is not a fence line.
 *
 * CommonMark, "Fenced code blocks": up to three leading spaces, then at least three backticks or
 * at least three tildes. Counting only bare ``` at column zero misses every shape that occurs in
 * real thinking text -- a fence indented because it sits in a list item, one opened on the list
 * marker line itself, and a ~~~ fence used because the code contains backticks -- and a missed
 * fence is not a missed optimisation, it is a slice into the middle of a code block.
 */
function fenceMarker(rawLine: string): { char: string; length: number; info: string } | null {
  const line = stripContainers(rawLine);
  let index = 0;
  while (index < 3 && line[index] === " ") index += 1;
  const char = line[index];
  if (char !== "`" && char !== "~") return null;
  let length = 0;
  while (line[index + length] === char) length += 1;
  if (length < 3) return null;
  return { char, length, info: line.slice(index + length) };
}

/**
 * `line` with its inline code spans removed.
 *
 * A `$$` inside backticks is prose about display math, not display math, and counting it flips
 * parity so that the REAL opener that follows flips it back. The scan then believes it is outside
 * an equation while inside one, which is the failure this is all trying to avoid, reached by the
 * one line of text most likely to appear in reasoning that discusses formatting.
 * `updateDisplayMathParity` in streaming-render-schedule.ts skips inline code for the same reason.
 */
const INLINE_CODE = /(`+)(?:[^`]|(?!\1)`)*\1/g;

/** How many `$$` markers a line carries outside inline code, which is what flips parity. */
function displayMathMarkers(line: string): number {
  const bare = line.replace(INLINE_CODE, "");
  let count = 0;
  for (let index = bare.indexOf("$$"); index !== -1; index = bare.indexOf("$$", index + 2)) {
    count += 1;
  }
  return count;
}

/** Whether a line is a link-reference definition, `[label]: destination`. */
const LINK_DEFINITION = /^ {0,3}\[[^\]]+\]:/;

/**
 * The state a scan of the text so far leaves the renderer in.
 *
 * Fences and display math are both "the marker that closes me reads as the marker that opens you",
 * so a slice landing inside either one makes the renderer treat everything after it as that
 * construct. `streaming-render-schedule.ts` already refuses to commit a block on non-neutral
 * display-math parity for the same reason; this is the same rule applied to the window.
 */
type ScanState = {
  fence: { char: string; length: number } | null;
  mathOpen: boolean;
};

function advance(state: ScanState, line: string): void {
  const marker = fenceMarker(line);
  if (marker) {
    if (state.fence === null) {
      // A backtick fence's info string may not contain a backtick, which is what keeps inline
      // ``` in prose from opening one.
      if (!(marker.char === "`" && marker.info.includes("`"))) {
        state.fence = { char: marker.char, length: marker.length };
      }
    } else if (
      marker.char === state.fence.char &&
      marker.length >= state.fence.length &&
      marker.info.trim() === ""
    ) {
      state.fence = null;
    }
    return;
  }
  if (state.fence === null && displayMathMarkers(line) % 2 === 1) {
    state.mathOpen = !state.mathOpen;
  }
}

const neutral = (state: ScanState): boolean => state.fence === null && !state.mathOpen;

/**
 * Whether an offset is outside every construct a slice must not land inside.
 *
 * Kept as its own export because it is what the tests assert against; the window itself uses the
 * single pass below rather than calling this per candidate.
 */
export function isOutsideFence(text: string, offset: number): boolean {
  const state: ScanState = { fence: null, mathOpen: false };
  let lineStart = 0;
  while (lineStart < offset) {
    let lineEnd = text.indexOf("\n", lineStart);
    if (lineEnd === -1 || lineEnd > offset) lineEnd = Math.min(offset, text.length);
    advance(state, text.slice(lineStart, lineEnd));
    lineStart = lineEnd + 1;
  }
  return neutral(state);
}

/**
 * Whether the line at `offset` begins a block at the TOP level, with nothing indented about it.
 *
 * A blank line inside a loose list item is a block boundary, but the item is still open across it,
 * and the paragraph after it is INDENTED because that indentation is what keeps it inside the
 * item. Slice there and the marker is gone, so a four-space continuation that was ordinary list
 * text becomes an indented code block, and a two-space one becomes a paragraph that lost its
 * bullet. The container the reader can see is not in the slice, so there is nothing to carry it.
 *
 * Requiring column zero refuses every such boundary without having to model list containers: a
 * line that is indented at all is continuing something, and the window simply waits for the next
 * boundary that is not. Refusing too much only costs a later window; accepting one of these
 * changes what the reader is shown.
 */
function startsTopLevelBlock(text: string, offset: number): boolean {
  const character = text[offset];
  return character !== undefined && character !== " " && character !== "\t";
}

/**
 * The first block boundary at or after `target` that leaves the remainder outside everything.
 *
 * ONE pass over the text, deliberately. The obvious shape -- walk the blank lines and ask
 * `isOutsideFence` about each -- rescans from byte zero for every candidate, and the case where
 * that bites is the exact case the window exists for: inside a still-open fence containing blank
 * lines, NO candidate is ever safe, so every blank line pays a full prefix scan and the whole
 * quadratic sum is repeated on every streamed token. On a 100,000-character unfinished fence that
 * is a frame or more per chunk, which would recreate the slowdown this is here to remove.
 *
 * Returns 0 when there is no safe boundary, which mounts the whole body. That is the right
 * failure: showing everything is correct and merely slow, whereas cutting into a fence is wrong.
 */
export function alignWindowStart(text: string, target: number): number {
  if (target <= 0) return 0;
  const state: ScanState = { fence: null, mathOpen: false };
  let lineStart = 0;
  const length = text.length;
  while (lineStart < length) {
    let lineEnd = text.indexOf("\n", lineStart);
    if (lineEnd === -1) lineEnd = length;
    const line = text.slice(lineStart, lineEnd);
    const nextLine = lineEnd + 1;
    // A blank line is a block boundary, and the boundary the renderer sees is the START of the
    // line after it. Trimming rather than testing for "" also makes a whitespace-only line and a
    // CRLF stream work, both of which the previous "\n\n" search silently declined to window.
    if (
      nextLine >= target &&
      line.trim() === "" &&
      neutral(state) &&
      startsTopLevelBlock(text, nextLine)
    ) {
      return nextLine;
    }
    advance(state, line);
    lineStart = nextLine;
  }
  return 0;
}

/**
 * The link-reference definitions before `start`, so a `[label]` left in the window still resolves.
 *
 * A definition is document-wide and invisible, so slicing it away turns a link in the mounted tail
 * into literal text. `IncrementalMarkdownCache` retains definitions for exactly this reason and
 * cannot recover ones this slice already removed, so they are carried across instead. They render
 * to nothing, so the visible text is unchanged.
 */
export function linkDefinitionsBefore(text: string, start: number): string {
  if (start <= 0) return "";
  const definitions: string[] = [];
  // Carrying the same scan state, because `[label]: value` inside a fence is code that happens to
  // look like a definition, and hoisting it out of its block would put text on screen that the
  // model wrote as an example.
  const state: ScanState = { fence: null, mathOpen: false };
  let lineStart = 0;
  while (lineStart < start) {
    let lineEnd = text.indexOf("\n", lineStart);
    if (lineEnd === -1 || lineEnd > start) lineEnd = start;
    const line = text.slice(lineStart, lineEnd);
    if (neutral(state) && LINK_DEFINITION.test(line)) definitions.push(line);
    advance(state, line);
    lineStart = lineEnd + 1;
  }
  return definitions.length === 0 ? "" : `${definitions.join("\n")}\n\n`;
}

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
