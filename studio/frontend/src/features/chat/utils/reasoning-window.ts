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
//
// WHERE THE WINDOW IS ALLOWED TO START, AND WHY THIS NO LONGER PARSES MARKDOWN ITSELF
//
// It started as a hand-written scan for the constructs a slice must not land inside, and it grew
// one construct per review round: indented fences, tilde fences, fences opened on a list marker,
// fences opened in a quote, longer-run closers, info-string closers, container markers that stop
// applying inside a fence, `$$` display math, `$$` inside inline code, loose list continuations,
// and the five HTML block types a blank line does not end. Every round found another, and several
// of them produced a WRONG boundary rather than merely a missed one.
//
// That is the wrong shape of answer, because the renderer already has the only definition of a
// boundary that matters. Streamdown splits the document with `parseMarkdownIntoBlocks` and renders
// each block INDEPENDENTLY -- that independence is what `BlockComponent` and the block memo are
// for. So if block N is already parsed without reference to blocks 0..N-1, then dropping those
// blocks cannot change how block N renders. Slicing on a block boundary is safe by construction,
// and it is safe in the only sense that counts here: the mounted output matches what the
// unwindowed tree shows, including in the places where Streamdown's own splitting is imperfect,
// because the window and the renderer are then imperfect in exactly the same way.
//
// Verified against the splitter directly: it keeps whole, and rejoins losslessly, the indented
// fence with a blank line in it, the tilde fence, the `10. ```js` fence whose closer is indented
// four spaces, a quoted fence followed by a top-level one, `$$` display math, `<script>`, an HTML
// comment, a loose list, and a four-backtick fence containing three. The two hand-written scans
// this replaces got the ordered-list fence and the quoted-then-top-level fence wrong.
//
// ONE GUARD SURVIVES, because one transform runs BEFORE the split. `preprocessLaTeX` rewrites
// `\[ ... \]` into `$$ ... $$` on the whole string, so bracket math is display math by the time
// blocks are formed but is NOT bracket-delimited any more. A slice taken in raw source space can
// still land inside one, and `preprocessLaTeX` would then meet an orphan `\]`. That is the only
// construct the block boundaries cannot speak for, so it is the only one still tracked here.

import { parseMarkdownIntoBlocks } from "streamdown";

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
 * How far the text must grow before an alignment that found nothing is attempted again.
 *
 * A failed alignment means there is no safe boundary at or after the target, and the target only
 * moves forward, so nothing already scanned can become safe: only newly arrived text can. Retrying
 * on the very next 24-character chunk therefore rescans the whole body to reach the same answer.
 * Measured on a 130,000-character stream that is one unterminated fence, where no boundary is ever
 * safe: 4,667 chunks past the threshold, 1,692ms of scanning in total against 29ms with this
 * backoff. It drops no frame by itself at 0.363ms per chunk against a 73ms chunk interval, but it
 * was pure overhead on the one path where the window delivers nothing at all.
 */
export const REASONING_WINDOW_RETRY_CHARS = 2_000;

/**
 * A link-reference definition, `[label]: destination`, possibly inside a container.
 *
 * The label allows a backslash escape, because CommonMark lets one contain an escaped bracket:
 * `[spec\]]: /url` is one definition, not a label ending at the first `]`. The destination is
 * checked separately by `hasValidDestination`, because a line that merely LOOKS like a definition
 * is rendered visibly as a paragraph rather than registering anything, and carrying one of those
 * puts old text back at the top of the pane.
 */
const LINK_DEFINITION = /^ {0,3}>?\s*\[(?:[^\]\\]|\\.)+\]:\s*(\S[\s\S]*)$/;

/**
 * Whether what follows the colon is a destination CommonMark would actually accept.
 *
 * Deliberately conservative, and it fails toward NOT carrying. Refusing a real definition costs a
 * link its styling inside the window until the reader scrolls back; accepting a fake one puts a
 * paragraph the reader has already read back on screen, which is a visible artefact made by the
 * machinery that exists to avoid one.
 */
function hasValidDestination(rest: string): boolean {
  const trimmed = rest.trim();
  if (trimmed.startsWith("<")) {
    const close = trimmed.indexOf(">");
    // An angle destination may not contain a newline, an unescaped `<`, or spaces past the `>`.
    if (close === -1) return false;
    const inside = trimmed.slice(1, close);
    return !/[<\n]/.test(inside);
  }
  // A bare destination runs to the first whitespace; anything after it must be a title.
  const [destination, ...title] = trimmed.split(/\s+/);
  if (destination.length === 0 || destination.includes("<")) return false;
  if (title.length === 0) return true;
  const rejoined = title.join(" ");
  return /^["'(].*["')]$/.test(rejoined);
}

/** Container prefixes, so `> [spec]: url` is recognised and carried without its quote marker. */
const CONTAINER_PREFIX = /^ {0,3}(?:>|(?:[-+*]|\d{1,9}[.)])(?=[ \t]))[ \t]*/;

function stripContainers(line: string): string {
  let out = line;
  for (;;) {
    const match = CONTAINER_PREFIX.exec(out);
    if (!match || match[0].length === 0) return out;
    out = out.slice(match[0].length);
  }
}

/** Whether a block is fenced code, in which nothing means what it says. */
function isFencedCode(block: string): boolean {
  const body = stripContainers(block.trimStart());
  return body.startsWith("```") || body.startsWith("~~~");
}

/** Inline code spans, which `preprocessLaTeX` also skips. */
const INLINE_CODE = /(`+)(?:[^`]|(?!\1)`)*\1/g;

/**
 * How a block changes `\[ ... \]` parity.
 *
 * Code is skipped, both fenced and inline, for the same reason `preprocessLaTeX` skips it: a `\[`
 * in a code sample is a literal, and treating it as an opener leaves the scanner believing it is
 * inside an equation for the rest of the stream. That failure is quiet and total, because
 * `alignWindowStart` then returns 0 forever and the pane simply never becomes windowed.
 */
function bracketDelta(block: string): number {
  if (isFencedCode(block)) return 0;
  const bare = block.replace(INLINE_CODE, "");
  let depth = 0;
  let index = 0;
  while (index < bare.length - 1) {
    if (bare[index] !== "\\") {
      index += 1;
      continue;
    }
    const next = bare[index + 1];
    if (next === "\\") {
      index += 2;
      continue;
    }
    if (next === "[") depth += 1;
    else if (next === "]") depth -= 1;
    index += 2;
  }
  return depth;
}

/**
 * Whether `offset` sits inside a `\[ ... \]` equation.
 *
 * The one construct the block boundaries cannot speak for, because `preprocessLaTeX` turns it into
 * `$$` before the document is split, so the split never sees it in this form.
 */
export function isOutsideBracketMath(text: string, offset: number): boolean {
  let depth = 0;
  let position = 0;
  for (const block of parseMarkdownIntoBlocks(text)) {
    if (position >= offset) break;
    depth += bracketDelta(block);
    position += block.length;
  }
  return depth <= 0;
}

/**
 * The first block boundary at or after `target`.
 *
 * One pass. Asking `isOutsideBracketMath` per candidate would re-walk the whole prefix for each
 * one, which is the quadratic shape this file has already had to remove twice.
 *
 * Returns 0 when there is none, which mounts the whole body. That is the right failure: showing
 * everything is correct and merely slow, whereas cutting into a construct is wrong.
 */
export function alignWindowStart(text: string, target: number): number {
  if (target <= 0) return 0;
  let offset = 0;
  let depth = 0;
  for (const block of parseMarkdownIntoBlocks(text)) {
    if (offset >= target && offset > 0 && depth <= 0) return offset;
    depth += bracketDelta(block);
    offset += block.length;
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

/**
 * The link-reference definitions before `start`, so a `[label]` left in the window still resolves.
 *
 * A definition is document-wide and invisible, so slicing it away turns a link in the mounted tail
 * into literal text. `IncrementalMarkdownCache` retains definitions for exactly this reason and
 * cannot recover ones this slice already removed, so they are carried across instead. They render
 * to nothing, so the visible text is unchanged.
 *
 * Taken a BLOCK at a time rather than a line at a time, which is what makes it correct for free in
 * three ways a line scan had to be taught one by one: a definition whose destination or title sits
 * on a continuation line is one block and is carried whole, so a bare `[spec]:` can never be
 * hoisted on its own; and a definition written inside a fence or inside an HTML block is part of
 * that larger block rather than a block of its own, so it is never mistaken for a real one.
 */
export function linkDefinitionsBefore(text: string, start: number): string {
  if (start <= 0) return "";
  const definitions: string[] = [];
  let offset = 0;
  for (const block of parseMarkdownIntoBlocks(text)) {
    if (offset >= start) break;
    const bare = stripContainers(block.trim());
    const match = LINK_DEFINITION.exec(bare);
    if (match && hasValidDestination(match[1])) definitions.push(bare);
    offset += block.length;
  }
  return definitions.length === 0 ? "" : `${definitions.join("\n")}\n\n`;
}

/**
 * Where the window starts, when it is worth looking again, and the definitions to carry with it.
 *
 * The definitions live here because they change only when the START does, which is once every
 * 6,000 characters, while renders arrive every frame. Recomputing them per render would rescan the
 * whole immutable prefix each time, which is the quadratic shape this file has already had to
 * remove once.
 */
export type ReasoningWindowState = {
  start: number;
  retryAt: number;
  definitions: string;
};

export const freshReasoningWindow = (): ReasoningWindowState => ({
  start: 0,
  retryAt: 0,
  definitions: "",
});

/**
 * The window state after `text`, skipping the scan when it provably cannot find anything new.
 *
 * The rule itself stays in `nextReasoningWindowStart`, which is pure and is what the tests hold;
 * this only decides whether it is worth asking, and caches what the answer implies.
 */
export function advanceReasoningWindow(
  text: string,
  state: ReasoningWindowState,
): ReasoningWindowState {
  if (text.length < state.retryAt) return state;
  const start = nextReasoningWindowStart(text, state.start);
  if (start > state.start) {
    return { start, retryAt: 0, definitions: linkDefinitionsBefore(text, start) };
  }
  return { ...state, start, retryAt: text.length + REASONING_WINDOW_RETRY_CHARS };
}
