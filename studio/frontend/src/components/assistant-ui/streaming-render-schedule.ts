// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps } from "streamdown";
import {
  parseMarkdownBlockDetails,
  parseMarkdownIntoBlocks,
} from "../../lib/parse-markdown-blocks.ts";

// The block list interleaves "\n\n" separators, so this is about four paragraphs of slack.
const ROLLBACK_BLOCKS = 8;
// Give up on a never-closed marker at a character budget: the boundary scan grows quadratically.
const STALLED_TAIL_CHARACTERS = 8_192;
// remend's whole-string passes are wasted inside an unterminated fence; repair the head plus a
// window of the fence body (last two lines) and splice the untouched middle back in.
const OPEN_FENCE_REPAIR_WINDOW = 4_096;
// Probes whether remend would append a closer for a marker open before the fence. It reads only the
// head; the body marker refusal below covers parity flips later in the body.
const OPEN_FENCE_PROBE = "\nq\n";
// An index resolves only once the next two characters are known, so hold them back.
const FENCE_SCAN_MARGIN = 2;
// Balanced marker prefixes give remend whole-document context without changing parity.
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";
const BOLD_CONTEXT = "**x**\n\n";
const SINGLE_ASTERISK_CONTEXT = "*x*\n\n";
const SINGLE_UNDERSCORE_CONTEXT = "_x_\n\n";
const INLINE_CODE_ASTERISK_CONTEXT = "`a *b* c`\n\n";
const INLINE_CODE_UNDERSCORE_CONTEXT = "`a _b_ c`\n\n";
// Deliberately unbalanced: `\(` is remend's only way into inline LaTeX. The blank line keeps the
// opener off the tail's first line and stops the link-destination scan. No `\[` twin; see
// hasUncarriableMath.
const INLINE_LATEX_CONTEXT = "\\(\n\n";
const FOOTNOTE_REFERENCE_RE = /\[\^[\w-]{1,200}\](?!:)/;
const FOOTNOTE_DEFINITION_RE = /\[\^[\w-]{1,200}\]:/;
// Marked's `def` label; err toward false positives (only costs retention). 999 is CommonMark's
// cap; `u` so the bound counts code points.
const LINK_DEFINITION_RE = /\[(?:\\[\s\S]|[^\]\\]){1,999}\]:/u;
// Widest match in UTF-16 units: 999 times `\` plus an astral code point, plus `[`.
const LINK_DEFINITION_WINDOW = 999 * 3 + 2;

// Odd backslash run: `[a\]b]:` keeps its escaped `]` in the label, `[a]b]:` does not.
function isEscaped(text: string, index: number): boolean {
  let slashes = 0;
  for (let i = index - 1; i >= 0 && text[i] === "\\"; i -= 1) {
    slashes += 1;
  }
  return slashes % 2 === 1;
}
// Same predicate as the regex, scanned from the rare `]:` instead of every `[`. Cursors only
// advance and lookaheads are cached; re-asking indexOf past -1 rescans the tail.
function hasLinkDefinition(text: string): boolean {
  let bracket = text.indexOf("[");
  let nextBracket = bracket < 0 ? -1 : text.indexOf("[", bracket + 1);
  let close = -1;
  let nextClose = text.indexOf("]");
  for (let end = text.indexOf("]:"); end >= 0; end = text.indexOf("]:", end + 1)) {
    while (nextBracket >= 0 && nextBracket <= end) {
      bracket = nextBracket;
      nextBracket = text.indexOf("[", bracket + 1);
    }
    while (nextClose >= 0 && nextClose < end) {
      close = nextClose;
      nextClose = text.indexOf("]", close + 1);
    }
    let start = end < LINK_DEFINITION_WINDOW ? 0 : end - LINK_DEFINITION_WINDOW;
    if (close > start && !isEscaped(text, close)) {
      start = close + 1;
    }
    // `start`, not `bracket`: a label may contain `[`, so an earlier one can match. Skip test only.
    if (bracket < start || bracket > end) {
      continue;
    }
    if (LINK_DEFINITION_RE.test(text.slice(start, end + 2))) {
      return true;
    }
  }
  return false;
}
// A list marker needs trailing whitespace: `-[label]:` is prose, not a bullet.
const CONTAINER_PREFIX = "[ \t]*(?:(?:>[ \t]*)|(?:(?:[-*+]|\\d{1,9}[.)])[ \t]+))*";
const LINK_DEFINITION_LINE_RE = new RegExp(
  `^${CONTAINER_PREFIX}${LINK_DEFINITION_RE.source}`,
  `m${LINK_DEFINITION_RE.flags}`,
);
// Must match exactly what Marked stores after the label: this feeds a React key, so extra capture
// remounts per character. Residual: a wrapped title keeps its old value until the message settles.
const LINK_DEFINITION_DESTINATION = "(?:<[^>\\n]*>?|[^\\s]*)";
const LINK_DEFINITION_TITLE = "[\"'(][^\\n]*[\"')]";
const LINK_DEFINITION_KEY_RE = new RegExp(
  `${LINK_DEFINITION_LINE_RE.source}[ \\t]*(?:\\n${CONTAINER_PREFIX})?${LINK_DEFINITION_DESTINATION}` +
    `(?:[ \\t]+${LINK_DEFINITION_TITLE}|[ \\t]*\\n${CONTAINER_PREFIX}${LINK_DEFINITION_TITLE})?`,
  `g${LINK_DEFINITION_LINE_RE.flags}`,
);
// Literal code bodies: an opening fence, or an indent reaching column four (spaces or a tab).
const CODE_BLOCK_RE = /^(?: {0,3}(?:`{3,}|~{3,})|(?: {4,}| {0,3}\t)[ \t]*[^ \t\r\n])/;
// A backtick opener with a backtick in its info string is not a fence; tildes have no such rule.
const BACKTICK_OPENER_RE = /^ {0,3}`{3,}([^\n]*)/;

function isCodeBlock(block: string): boolean {
  if (!CODE_BLOCK_RE.test(block)) {
    return false;
  }
  const backtick = BACKTICK_OPENER_RE.exec(block);
  return backtick === null || !backtick[1].includes("`");
}
// Must admit exactly what `LINK_DEFINITION_RE` admits, or the wider cap becomes unreachable.
const LINK_REFERENCE_RE =
  /!?\[(?:\\[\s\S]|[^\]\\]){1,999}\]\[(?:\\[\s\S]|[^\]\\]){0,999}\]/u;
// Label side as above, plus `[` and the optional `!`; the reference side needs no `!`.
const LINK_REFERENCE_WINDOW = 999 * 3 + 3;
// The `[` at the seam restarts escape parity, so the label is the text up to the first `]`.
const LINK_REFERENCE_LABEL_RE = /^(?:\\[\s\S]|[^\]\\]){0,999}$/u;

function unescapedClose(text: string, from: number): number {
  for (let i = text.indexOf("]", from); i >= 0; i = text.indexOf("]", i + 1)) {
    if (!isEscaped(text, i)) {
      return i;
    }
  }
  return -1;
}
// Same predicate as the regex, scanned from the rare `][` as hasLinkDefinition scans from `]:`.
function hasLinkReference(text: string): boolean {
  let bracket = text.indexOf("[");
  let nextBracket = bracket < 0 ? -1 : text.indexOf("[", bracket + 1);
  let close = -1;
  let nextClose = text.indexOf("]");
  let after = -1;
  for (let mid = text.indexOf("]["); mid >= 0; mid = text.indexOf("][", mid + 1)) {
    // No empty label, so the opener is at `mid - 2` or earlier. Forward: `lastIndexOf` is not.
    while (nextBracket >= 0 && nextBracket <= mid - 2) {
      bracket = nextBracket;
      nextBracket = text.indexOf("[", bracket + 1);
    }
    while (nextClose >= 0 && nextClose < mid) {
      if (!isEscaped(text, nextClose)) {
        close = nextClose;
      }
      nextClose = text.indexOf("]", nextClose + 1);
    }
    if (after < mid + 2) {
      after = unescapedClose(text, mid + 2);
      if (after < 0) {
        // Return rather than caching -1, which sits behind every later seam and rescans: quadratic.
        return false;
      }
    }
    if (after - mid - 2 > LINK_REFERENCE_WINDOW) {
      continue;
    }
    let start = mid < LINK_REFERENCE_WINDOW ? 0 : mid - LINK_REFERENCE_WINDOW;
    if (close > start) {
      start = close + 1;
    }
    if (bracket < start || bracket > mid - 2) {
      continue;
    }
    if (!LINK_REFERENCE_LABEL_RE.test(text.slice(mid + 2, after))) {
      continue;
    }
    // `start`, not `bracket`, as in `hasLinkDefinition`. Skip test only.
    if (LINK_REFERENCE_RE.test(text.slice(start, after + 1))) {
      return true;
    }
  }
  return false;
}
// A shortcut `[label]` or collapsed `[label][]` resolves against a definition too. Code spans,
// inline links and the like are deliberately not excluded: that only adds false positives.
const SHORTCUT_REFERENCE_RE = /\[((?:\\[\s\S]|[^[\]\\]){1,999})\]/gu;
const DEFINITION_LABEL_RE = /\[((?:\\[\s\S]|[^\]\\]){1,999})\]:/u;

// micromark's `normalizeIdentifier`, so `[SS]` finds `[\u1E9E]:` as the renderer does.
function normalizeLabel(label: string): string {
  return label
    .replace(/[\t\n\r ]+/g, " ")
    .replace(/^ | $/g, "")
    .toLowerCase()
    .toUpperCase();
}

function hasShortcutReference(
  prose: string,
  references: string,
  definitions: readonly string[],
): boolean {
  const labels = new Set<string>();
  // Marked's tokens as well: an unmatched `[` line before a definition widens the regex's label.
  for (const definition of [
    ...definitions,
    ...(prose.match(LINK_DEFINITION_KEY_RE) ?? []),
  ]) {
    const label = DEFINITION_LABEL_RE.exec(definition)?.[1];
    if (label !== undefined) {
      labels.add(normalizeLabel(label));
    }
  }
  labels.delete("");
  if (labels.size === 0) {
    return false;
  }
  // Marked's definitions, not a regex: `[1]: <broken` is prose whose `[1]` is a reference.
  const uses = normalizeLineEndings(references);
  for (const match of uses.matchAll(SHORTCUT_REFERENCE_RE)) {
    if (
      !isEscaped(uses, match.index) &&
      labels.has(normalizeLabel(match[1]))
    ) {
      return true;
    }
  }
  return false;
}
const WORD_CHARACTER_RE = /[\p{L}\p{N}_]/u;
const HTML_TAG_START_RE = /[a-zA-Z/]/;

// One memo slot: markdown-text.tsx asks for the key, then Streamdown splits the same string.
let splitMarkdown: string | null = null;
let splitBlocks: readonly string[] = [];
let splitReferenceProse = "";
let splitDefinitions: readonly string[] = [];

function blocksOf(markdown: string): readonly string[] {
  if (splitMarkdown !== markdown) {
    splitMarkdown = markdown;
    const details = parseMarkdownBlockDetails(markdown);
    splitBlocks = details.blocks;
    splitReferenceProse = details.referenceProse.join("\n\n");
    splitDefinitions = details.definitions;
  }
  return splitBlocks;
}

function referenceProseOf(markdown: string): string {
  blocksOf(markdown);
  return splitReferenceProse;
}

function definitionsOf(markdown: string): readonly string[] {
  blocksOf(markdown);
  return splitDefinitions;
}

// Which replies have to be lexed in one piece.
//
// marked keeps link reference definitions in one document-wide map and emits no token for a
// label it has already seen, so a `[label][ref]` and its `[ref]: url` must reach the lexer
// together or the reference survives as literal text. The question is therefore whether a real
// definition exists outside code -- and the earlier answer, a hand-rolled scan for fences,
// containers and raw HTML, kept disagreeing with marked at the seams: nested fences, the seven
// HTML block shapes, list continuation indentation, lone-CR line endings.
//
// marked has already resolved every one of those by the time it hands back blocks, so the split
// is the answer rather than something to re-derive. A fenced or indented block is code; anything
// else is prose, and a definition line anywhere in the prose counts.
//
// Being wrong is not symmetric, which is why the residual imprecision sits where it does. Saying
// `blocks` when the reply needed one document splits the pair apart and loses content. Saying
// `document` when blocks would have done only costs that reply its per-code-block Copy and
// Download controls -- which is what this path did for EVERY reply containing a `]:` substring
// before. See tests/link-definition-oracle.test.ts, which pins the first case exhaustively.
// Normalised because `\r` counts against `{1,999}` and the `\n` it replaces does
// not, so the scope would otherwise follow the reply's line ending. NOT for
// `blocksOf`, whose one memo slot is shared with `parseMarkdownIntoRenderableBlocks`:
// a normalised copy misses it and costs a CRLF reply two splits per render.
// A shortcut reference can be any `[label]`, so a definition alone is enough to pay for the split.
function documentProse(markdown: string): string | null {
  if (!hasLinkDefinition(normalizeLineEndings(markdown))) {
    return null;
  }
  const prose = normalizeLineEndings(
    blocksOf(markdown)
      .filter((block) => !isCodeBlock(block))
      .join("\n"),
  );
  return LINK_DEFINITION_LINE_RE.test(prose) &&
    (hasLinkReference(prose) ||
      hasShortcutReference(
        prose,
        referenceProseOf(markdown),
        definitionsOf(markdown),
      ))
    ? prose
    : null;
}

export function markdownRenderScope(markdown: string): "blocks" | "document" {
  return documentProse(markdown) === null ? "blocks" : "document";
}

export function markdownRenderKey(markdown: string): string {
  const prose = documentProse(markdown);
  if (prose === null) {
    return "blocks";
  }
  return `document:${(prose.match(LINK_DEFINITION_KEY_RE) ?? []).join("\n")}`;
}

export function parseMarkdownIntoRenderableBlocks(markdown: string): string[] {
  return markdownRenderScope(markdown) === "document"
    ? [markdown]
    : [...blocksOf(markdown)];
}

// Mirrors remend 1.3.1's five-state math machine, since `RepairParity` hand-copies its marker rules.
type EmphasisMathState =
  | "none"
  | "inlineLatex"
  | "blockLatex"
  | "inlineDollar"
  | "blockDollar";

type RepairParity = {
  bold: boolean;
  boldCandidate: boolean;
  boldFence: boolean;
  bracketDepth: number;
  linkDefinition: boolean;
  doubleUnderscore: boolean;
  emphasisInlineCode: boolean;
  emphasisMath: EmphasisMathState;
  // Whether the last counted asterisk was in-word; see `countsAsSingleAsterisk`. Always false at
  // a commit boundary, so it never needs carrying.
  inWordAsteriskChain: boolean;
  firstBoldOrSingleUnderscore: "bold" | "singleUnderscore" | null;
  singleAsterisk: boolean;
  singleAsteriskCandidate: boolean;
  firstSingleAsteriskCandidate: "inlineCode" | "normal" | null;
  singleUnderscore: boolean;
  singleUnderscoreCandidate: boolean;
  firstSingleUnderscoreCandidate: "inlineCode" | "normal" | null;
  tripleAsterisk: boolean;
  displayMathInlineCode: boolean;
  inlineCode: boolean;
  inlineMathInlineCode: boolean;
  strikethrough: boolean;
  displayMath: boolean;
  inlineMath: boolean;
};

// Three open states are excluded because `hasNeutralRepairParity` refuses to commit inside them.
type RetainedLatexState = "none" | "inlineLatex";

const createRepairParity = (
  latex: RetainedLatexState = "none",
): RepairParity => ({
  bold: false,
  boldCandidate: false,
  boldFence: false,
  bracketDepth: 0,
  linkDefinition: false,
  doubleUnderscore: false,
  emphasisInlineCode: false,
  emphasisMath: latex,
  inWordAsteriskChain: false,
  firstBoldOrSingleUnderscore: null,
  singleAsterisk: false,
  singleAsteriskCandidate: false,
  firstSingleAsteriskCandidate: null,
  singleUnderscore: false,
  singleUnderscoreCandidate: false,
  firstSingleUnderscoreCandidate: null,
  tripleAsterisk: false,
  displayMathInlineCode: false,
  inlineCode: false,
  inlineMathInlineCode: false,
  strikethrough: false,
  displayMath: false,
  inlineMath: false,
});

// Keep every link definition in the live tail: marked resolves them document-wide, so a
// retained one would be lexed apart. Same test as `documentProse`.
function updateLinkDefinitionParity(parity: RepairParity, text: string): void {
  if (!isCodeBlock(text) && LINK_DEFINITION_LINE_RE.test(text)) {
    parity.linkDefinition = true;
  }
}

const isTripleBacktick = (text: string, index: number): boolean =>
  (index >= 2 && text.slice(index - 2, index + 1) === "```") ||
  (index >= 1 && text.slice(index - 1, index + 2) === "```") ||
  text.slice(index, index + 3) === "```";

const isWordCharacter = (character: string | undefined): boolean =>
  character !== undefined && WORD_CHARACTER_RE.test(character);

function findLinkDestinationStart(text: string, index: number): number {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    const character = text[cursor];
    if (character === ")" || character === "\n") {
      return -1;
    }
    if (character === "(") {
      return cursor > 0 && text[cursor - 1] === "]" ? cursor : -1;
    }
  }
  return -1;
}

function hasLinkDestinationEnd(text: string, index: number): boolean {
  for (let cursor = index; cursor < text.length; cursor += 1) {
    if (text[cursor] === ")") {
      return true;
    }
    if (text[cursor] === "\n") {
      return false;
    }
  }
  return false;
}

const isWithinLinkDestination = (text: string, index: number): boolean =>
  findLinkDestinationStart(text, index) >= 0 &&
  hasLinkDestinationEnd(text, index);

function isWithinHtmlTag(text: string, index: number): boolean {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (text[cursor] === ">") {
      return false;
    }
    if (text[cursor] === "<") {
      return HTML_TAG_START_RE.test(text[cursor + 1] ?? "");
    }
    if (text[cursor] === "\n") {
      return false;
    }
  }
  return false;
}

// Regions open only from `none` and close from their own state; null means not a delimiter.
function latexMathTransition(
  state: EmphasisMathState,
  next: string | undefined,
): EmphasisMathState | null {
  if (next === "[" && state === "none") {
    return "blockLatex";
  }
  if (next === "]" && state === "blockLatex") {
    return "none";
  }
  if (next === "(" && state === "none") {
    return "inlineLatex";
  }
  if (next === ")" && state === "inlineLatex") {
    return "none";
  }
  return null;
}

// `$$` toggles from any state; a lone `$` inside a block region is absorbed. Matches remend 1.3.0.
function dollarMathTransition(
  state: EmphasisMathState,
  isDouble: boolean,
): EmphasisMathState {
  if (isDouble) {
    return state === "blockDollar" ? "none" : "blockDollar";
  }
  if (state === "blockDollar") {
    return state;
  }
  return state === "inlineDollar" ? "none" : "inlineDollar";
}

const isLatexMathState = (state: EmphasisMathState): boolean =>
  state === "inlineLatex" || state === "blockLatex";

// Incremental version of remend's per-marker math scan; runs before fence handling because
// remend's math pass ignores fences.
function updateEmphasisMathParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (text[index] === "\\") {
    // The escape wins: `\$` is a literal dollar in any scan state.
    if (text[index + 1] === "$") {
      return index + 1;
    }
    const transitioned = latexMathTransition(
      parity.emphasisMath,
      text[index + 1],
    );
    if (transitioned === null) {
      return index;
    }
    parity.emphasisMath = transitioned;
    return index + 1;
  }
  // Inside a LaTeX region a dollar is ordinary text (remend 1.3.1).
  if (text[index] !== "$" || isLatexMathState(parity.emphasisMath)) {
    return index;
  }
  const isDouble = text[index + 1] === "$";
  parity.emphasisMath = dollarMathTransition(parity.emphasisMath, isDouble);
  return isDouble ? index + 1 : index;
}

// Mirrors remend's `!character || isWhitespace(character)`.
const isBoundaryCharacter = (character: string | undefined): boolean =>
  character === undefined ||
  character === " " ||
  character === "\t" ||
  character === "\n";

// remend's skip list for the single asterisk counter, minus the in-word clause 1.3.1 moved below.
function shouldSkipAsterisk(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  if (previous === "\\" || parity.emphasisMath !== "none") {
    return true;
  }
  if (previous !== "*" && next === "*") {
    return text[index + 2] !== "*";
  }
  if (previous === "*") {
    return true;
  }
  return isBoundaryCharacter(previous) && isBoundaryCharacter(next);
}

// remend 1.3.1's rule: an in-word asterisk counts once the count is odd or a chain is running.
function countsAsSingleAsterisk(
  parity: RepairParity,
  text: string,
  index: number,
): { counts: boolean; inWordChain: boolean } {
  const previous = text[index - 1];
  const next = text[index + 1];
  const inWord = isWordCharacter(previous) && isWordCharacter(next);
  // remend's "text" test: present and not whitespace, so punctuation counts.
  const previousIsText = !isBoundaryCharacter(previous);
  const nextIsText = !isBoundaryCharacter(next);
  if (inWord && !parity.singleAsterisk && !parity.inWordAsteriskChain) {
    return { counts: false, inWordChain: false };
  }
  if ((previousIsText && parity.singleAsterisk) || nextIsText) {
    return { counts: true, inWordChain: inWord };
  }
  return { counts: false, inWordChain: false };
}

function isSingleAsteriskCandidate(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  if (
    previous === "\\" ||
    previous === "*" ||
    next === "*" ||
    parity.emphasisMath !== "none"
  ) {
    return false;
  }
  // remend 1.3.1: a marker followed only by whitespace or the end cannot open emphasis.
  return !(
    isBoundaryCharacter(next) ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

function countsAsSingleUnderscore(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  return !(
    previous === "\\" ||
    parity.emphasisMath !== "none" ||
    isWithinLinkDestination(text, index) ||
    isWithinHtmlTag(text, index) ||
    previous === "_" ||
    next === "_" ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

function isSingleUnderscoreCandidate(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  return !(
    previous === "\\" ||
    previous === "_" ||
    next === "_" ||
    parity.emphasisMath !== "none" ||
    isWithinLinkDestination(text, index) ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

// remend decides closers from document-wide parity, so a retained prefix must end neutral.
function updateAsteriskParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (isSingleAsteriskCandidate(parity, text, index)) {
    parity.singleAsteriskCandidate = true;
    parity.firstSingleAsteriskCandidate ??= parity.emphasisInlineCode
      ? "inlineCode"
      : "normal";
  }
  if (!shouldSkipAsterisk(parity, text, index)) {
    const decision = countsAsSingleAsterisk(parity, text, index);
    if (decision.counts) {
      parity.singleAsterisk = !parity.singleAsterisk;
      parity.inWordAsteriskChain = decision.inWordChain;
    }
  }
  if (text[index + 1] === "*") {
    parity.boldCandidate = true;
    parity.firstBoldOrSingleUnderscore ??= "bold";
    parity.bold = !parity.bold;
    return index + 1;
  }
  return index;
}

function updateUnderscoreParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (isSingleUnderscoreCandidate(parity, text, index)) {
    parity.singleUnderscoreCandidate = true;
    parity.firstSingleUnderscoreCandidate ??= parity.emphasisInlineCode
      ? "inlineCode"
      : "normal";
    parity.firstBoldOrSingleUnderscore ??= "singleUnderscore";
  }
  if (text[index + 1] === "_") {
    parity.doubleUnderscore = !parity.doubleUnderscore;
    return index + 1;
  }
  if (countsAsSingleUnderscore(parity, text, index)) {
    parity.singleUnderscore = !parity.singleUnderscore;
  }
  return index;
}

// remend locates `**` with a raw indexOf but counts pairs outside fences only, so a fenced `**`
// seeds the bold context without reaching the counter.
function recordBoldMarker(
  parity: RepairParity,
  text: string,
  index: number,
): void {
  if (text[index] === "*" && text[index + 1] === "*") {
    parity.boldCandidate = true;
    parity.firstBoldOrSingleUnderscore ??= "bold";
  }
}

// remend completes a dangling link at the document end, so an unmatched bracket blocks retention.
function updateBracketDepth(parity: RepairParity, character: string): void {
  if (character === "[") {
    parity.bracketDepth += 1;
  } else if (character === "]" && parity.bracketDepth > 0) {
    parity.bracketDepth -= 1;
  }
}

// remend consumes an escaped backtick before testing for a fence.
function skipEscapeOrFence(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (text[index] === "\\" && text[index + 1] === "`") {
    return index + 1;
  }
  if (text.slice(index, index + 3) === "```") {
    parity.boldFence = !parity.boldFence;
    return index + 2;
  }
  return index;
}

// Mirrors remend: non-asterisk, non-word chars clear the in-word chain, only outside a fence.
function clearInWordAsteriskChain(
  parity: RepairParity,
  character: string | undefined,
): void {
  if (!parity.boldFence && character !== "*" && !isWordCharacter(character)) {
    parity.inWordAsteriskChain = false;
  }
}

function updateEmphasisParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    // Cleared before the multi-character skips: they all start on non-word characters.
    clearInWordAsteriskChain(parity, text[index]);
    index = updateEmphasisMathParity(parity, text, index);
    const skipped = skipEscapeOrFence(parity, text, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    recordBoldMarker(parity, text, index);
    if (parity.boldFence) {
      continue;
    }
    if (text[index] === "`") {
      parity.emphasisInlineCode = !parity.emphasisInlineCode;
      continue;
    }
    if (!parity.emphasisInlineCode) {
      updateBracketDepth(parity, text[index]);
    }
    if (text[index] === "*") {
      index = updateAsteriskParity(parity, text, index);
      continue;
    }
    if (text[index] === "_") {
      index = updateUnderscoreParity(parity, text, index);
    }
  }
}

function updateTripleAsteriskParity(parity: RepairParity, text: string): void {
  let inFence = false;
  let runLength = 0;
  const finishRun = () => {
    if (Math.floor(runLength / 3) % 2 === 1) {
      parity.tripleAsterisk = !parity.tripleAsterisk;
    }
    runLength = 0;
  };
  for (let index = 0; index < text.length; index += 1) {
    if (text.slice(index, index + 3) === "```") {
      finishRun();
      inFence = !inFence;
      index += 2;
    } else if (!inFence && text[index] === "*") {
      runLength += 1;
    } else {
      finishRun();
    }
  }
  finishRun();
}

function updateInlineCodeParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\\" && text[index + 1] === "`") {
      index += 1;
      continue;
    }
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.inlineCode = !parity.inlineCode;
    }
  }
}

function updateStrikethroughParity(parity: RepairParity, text: string): void {
  const strikeMarkers = text.match(/~~/g)?.length ?? 0;
  if (strikeMarkers % 2 === 1) {
    parity.strikethrough = !parity.strikethrough;
  }
}

// Unlike remend, count to text.length: retained block boundaries are interior.
function updateDisplayMathParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.displayMathInlineCode = !parity.displayMathInlineCode;
      continue;
    }
    if (
      !parity.displayMathInlineCode &&
      text[index] === "$" &&
      text[index + 1] === "$"
    ) {
      parity.displayMath = !parity.displayMath;
      index += 1;
    }
  }
}

function updateInlineMathParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\\") {
      index += 1;
      continue;
    }
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.inlineMathInlineCode = !parity.inlineMathInlineCode;
      continue;
    }
    if (parity.inlineMathInlineCode || text[index] !== "$") {
      continue;
    }
    if (text[index + 1] === "$") {
      index += 1;
    } else {
      parity.inlineMath = !parity.inlineMath;
    }
  }
}

function updateRepairParity(parity: RepairParity, text: string): void {
  updateLinkDefinitionParity(parity, text);
  updateEmphasisParity(parity, text);
  updateTripleAsteriskParity(parity, text);
  updateInlineCodeParity(parity, text);
  updateStrikethroughParity(parity, text);
  updateDisplayMathParity(parity, text);
  updateInlineMathParity(parity, text);
}

// Open math regions a boundary may not sit inside. `$`/`$$` would be recounted by the katex
// repairs, and `\[` carries a `[` that remend's link repair reads. `inlineLatex` is safe.
const hasUncarriableMath = (parity: RepairParity): boolean =>
  parity.emphasisMath !== "none" && parity.emphasisMath !== "inlineLatex";

const hasNeutralRepairParity = (parity: RepairParity): boolean =>
  ![
    parity.bracketDepth > 0,
    parity.linkDefinition,
    parity.bold,
    parity.boldFence,
    parity.emphasisInlineCode,
    parity.doubleUnderscore,
    hasUncarriableMath(parity),
    parity.singleAsterisk,
    parity.singleUnderscore,
    parity.tripleAsterisk,
    parity.displayMathInlineCode,
    parity.inlineCode,
    parity.inlineMathInlineCode,
    parity.strikethrough,
    parity.displayMath,
    parity.inlineMath,
  ].includes(true);

export type IncrementalMarkdownRender = {
  markdown: string;
  parseMarkdownIntoBlocks: (markdown: string) => string[];
};

const INCOMPLETE_LINK_REPAIR = "](streamdown:incomplete-link)";

export function hasIncompleteLinkRepair(
  source: string,
  repaired?: string,
): boolean {
  // Only an unclosed `[` gets the placeholder, so bracket-free replies skip the remend pass.
  if (!source.includes("[")) return false;
  const after = repaired ?? remend(source);
  return (
    after.split(INCOMPLETE_LINK_REPAIR).length >
    source.split(INCOMPLETE_LINK_REPAIR).length
  );
}

// Skips only remend's link pass, whose placeholder renders as "[blocked]"; every other repair still runs.
export const LITERAL_LINK_REMEND = { links: false, images: false } as const;

export function repairStreamingMarkdown(source: string): string {
  const repaired = remend(source);
  return hasIncompleteLinkRepair(source, repaired)
    ? remend(source, LITERAL_LINK_REMEND)
    : repaired;
}

type RetainedContext = {
  multilineKatex: boolean;
  bold: boolean;
  singleAsterisk: boolean;
  singleUnderscore: boolean;
  latex: RetainedLatexState;
  firstSingleAsterisk: "inlineCode" | "normal" | null;
  firstSingleUnderscore: "inlineCode" | "normal" | null;
  firstBoldOrSingleUnderscore: "bold" | "singleUnderscore" | null;
};

const createRetainedContext = (): RetainedContext => ({
  multilineKatex: false,
  bold: false,
  singleAsterisk: false,
  singleUnderscore: false,
  latex: "none",
  firstSingleAsterisk: null,
  firstSingleUnderscore: null,
  firstBoldOrSingleUnderscore: null,
});

const singleUnderscoreContext = (context: RetainedContext): string => {
  if (!context.singleUnderscore) {
    return "";
  }
  return context.firstSingleUnderscore === "inlineCode"
    ? INLINE_CODE_UNDERSCORE_CONTEXT
    : SINGLE_UNDERSCORE_CONTEXT;
};

const singleAsteriskContext = (context: RetainedContext): string => {
  if (!context.singleAsterisk) {
    return "";
  }
  return context.firstSingleAsterisk === "inlineCode"
    ? INLINE_CODE_ASTERISK_CONTEXT
    : SINGLE_ASTERISK_CONTEXT;
};

const emphasisContext = (context: RetainedContext): string => {
  const bold = context.bold ? BOLD_CONTEXT : "";
  const underscore = singleUnderscoreContext(context);
  return context.firstBoldOrSingleUnderscore === "singleUnderscore"
    ? underscore + bold
    : bold + underscore;
};

const latexContext = (context: RetainedContext): string =>
  context.latex === "inlineLatex" ? INLINE_LATEX_CONTEXT : "";

// The LaTeX opener must come LAST, right before the tail: everything after it is inside its region.
function repairContextPrefix(context: RetainedContext): string {
  return (
    emphasisContext(context) +
    singleAsteriskContext(context) +
    (context.multilineKatex ? MULTILINE_KATEX_CONTEXT : "") +
    latexContext(context)
  );
}

function repairTail(
  tail: string,
  context: RetainedContext,
  options?: typeof LITERAL_LINK_REMEND,
): string {
  const prefix = repairContextPrefix(context);
  if (!prefix) {
    return remend(tail, options);
  }
  return remend(prefix + tail, options).slice(prefix.length);
}

function repairTailKeepingLinks(
  tail: string,
  context: RetainedContext,
  repaired = repairTail(tail, context),
): string {
  return hasIncompleteLinkRepair(tail, repaired)
    ? repairTail(tail, context, LITERAL_LINK_REMEND)
    : repaired;
}

// Mirrors remend rather than CommonMark: a mid-line ``` closes the fence for remend.
type OpenFenceState = {
  index: number;
  fenceOpen: boolean;
  bodyStart: number;
  // The FIRST ` $ or ~ in the fence body, or -1. Any of them disqualifies the splice, because remend's
  // three fence notions disagree and the synthetic opener cannot reproduce all of them.
  firstBodyMarker: number;
};

const initialOpenFenceState = (): OpenFenceState => ({
  index: 0,
  fenceOpen: false,
  bodyStart: -1,
  firstBodyMarker: -1,
});

function advanceOpenFence(
  state: OpenFenceState,
  text: string,
  limit: number,
): OpenFenceState {
  let { index, fenceOpen, bodyStart, firstBodyMarker } = state;
  while (index < limit) {
    const character = text[index];
    if (character === "\\" && text[index + 1] === "`") {
      // Escaped for `isWithinCodeBlock` only, so ``` run counters still see the backtick.
      if (fenceOpen && firstBodyMarker < 0) {
        firstBodyMarker = index + 1;
      }
      index += 2;
      continue;
    }
    if (
      character === "`" &&
      text[index + 1] === "`" &&
      text[index + 2] === "`"
    ) {
      fenceOpen = !fenceOpen;
      bodyStart = fenceOpen ? index + 3 : -1;
      firstBodyMarker = -1;
      index += 3;
      continue;
    }
    if (
      fenceOpen &&
      firstBodyMarker < 0 &&
      (character === "`" || character === "$" || character === "~")
    ) {
      firstBodyMarker = index;
    }
    index += 1;
  }
  return { index, fenceOpen, bodyStart, firstBodyMarker };
}

// Carries the scan across updates; text that is not an extension rescans from the start.
class OpenFenceTracker {
  private text = "";
  private resolved = initialOpenFenceState();

  spliceBounds(
    text: string,
  ): { bodyStart: number; firstBodyMarker: number } | null {
    if (!hasPrefix(text, this.text)) {
      this.resolved = initialOpenFenceState();
    }
    const limit = Math.max(0, text.length - FENCE_SCAN_MARGIN);
    if (limit > this.resolved.index) {
      this.resolved = advanceOpenFence(this.resolved, text, limit);
    }
    this.text = text;
    const live = advanceOpenFence(this.resolved, text, text.length);
    if (!live.fenceOpen) {
      return null;
    }
    return {
      bodyStart: live.bodyStart,
      firstBodyMarker: live.firstBodyMarker,
    };
  }
}

// An inert head only signals an open fence, which one opener reproduces; keeps repair off it.
const OPEN_FENCE_SYNTHETIC_HEAD = "```\n";

// The spliced repair, or null to repair the whole tail. Refuses when the body is shorter than
// the window, the final or preceding line exceeds it (setext repair reads one line back), or a
// ` $ or ~ appears anywhere in the body.
function repairOpenFenceTail(
  tail: string,
  bodyStart: number,
  firstBodyMarker: number,
): string | null {
  const cut = tail.indexOf("\n", tail.length - OPEN_FENCE_REPAIR_WINDOW);
  if (
    cut < 0 ||
    cut + 1 <= bodyStart ||
    tail.indexOf("\n", cut + 1) < 0 ||
    firstBodyMarker >= 0
  ) {
    return null;
  }
  const repaired = remend(OPEN_FENCE_SYNTHETIC_HEAD + tail.slice(cut + 1));
  if (!hasPrefix(repaired, OPEN_FENCE_SYNTHETIC_HEAD)) {
    return null;
  }
  return (
    tail.slice(0, cut + 1) + repaired.slice(OPEN_FENCE_SYNTHETIC_HEAD.length)
  );
}

// `latex` is a position, not a fact: a later commit can close it, so each commit stores its context.
const retainedLatexState = (parity: RepairParity): RetainedLatexState =>
  parity.emphasisMath === "inlineLatex" ? "inlineLatex" : "none";

const advanceContext = (
  context: RetainedContext,
  parity: RepairParity,
  committedText: string,
): RetainedContext => ({
  multilineKatex: context.multilineKatex || committedText.includes("$$"),
  bold: context.bold || parity.boldCandidate,
  singleAsterisk: context.singleAsterisk || parity.singleAsteriskCandidate,
  singleUnderscore:
    context.singleUnderscore || parity.singleUnderscoreCandidate,
  latex: retainedLatexState(parity),
  firstSingleAsterisk:
    context.firstSingleAsterisk ?? parity.firstSingleAsteriskCandidate,
  firstSingleUnderscore:
    context.firstSingleUnderscore ?? parity.firstSingleUnderscoreCandidate,
  firstBoldOrSingleUnderscore:
    context.firstBoldOrSingleUnderscore ?? parity.firstBoldOrSingleUnderscore,
});

type CommitBoundary = {
  count: number;
  length: number;
  parity: RepairParity | null;
  repairBroke: boolean;
};

// `advanceContext` only adds facts, so each commit's context is stored to allow a rewind.
type CommitPoint = {
  blockCount: number;
  length: number;
  context: RetainedContext;
};

// CommonMark treats LF, CR and CRLF alike, so normalising cannot change the render.
function normalizeLineEndings(text: string): string {
  return text.includes("\r") ? text.replace(/\r\n?/g, "\n") : text;
}

/**
 * `a` begins with `b`. Faster than `startsWith` for long prefixes on V8, slower for short
 * ones, so do not use it for short prefixes.
 */
export const hasPrefix = (a: string, b: string): boolean =>
  a.length >= b.length && a.slice(0, b.length) === b;

function sharedPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left.charCodeAt(index) === right.charCodeAt(index)) {
    index += 1;
  }
  return index;
}

// Marked reads a paragraph plus new text as a lazy continuation, so a block is only stable
// behind a blank line. `\r` counts as line-ending whitespace.
function endsAtBlankLine(text: string, end: number): boolean {
  if (end === 0) {
    return true;
  }
  if (text.charCodeAt(end - 1) !== 10) {
    return false;
  }
  let index = end - 2;
  while (index >= 0) {
    const code = text.charCodeAt(index);
    if (code === 32 || code === 9 || code === 13) {
      index -= 1;
      continue;
    }
    return code === 10;
  }
  return true;
}

// Not sufficient alone; `rewindToRewrite` also keeps a rollback window.
function lastBlankLineEnd(text: string, limit: number): number {
  for (let end = limit; end > 0; end -= 1) {
    if (endsAtBlankLine(text, end)) {
      return end;
    }
  }
  return 0;
}

// Never retain text remend synthesized; record the latest boundary with neutral global parity.
function findCommitBoundary(
  tail: string,
  blocks: string[],
  candidateCount: number,
  latex: RetainedLatexState,
): CommitBoundary {
  // Start from the retained prefix's LaTeX state, the only state that can be non-neutral here.
  const parity = createRepairParity(latex);
  const commit: CommitBoundary = {
    count: 0,
    length: 0,
    parity: null,
    repairBroke: false,
  };
  let exactLength = 0;

  for (let index = 0; index < candidateCount; index += 1) {
    const block = blocks[index];
    if (!tail.startsWith(block, exactLength)) {
      commit.repairBroke = true;
      break;
    }
    exactLength += block.length;
    updateRepairParity(parity, block);
    if (hasNeutralRepairParity(parity)) {
      commit.count = index + 1;
      commit.length = exactLength;
      commit.parity = { ...parity };
    }
  }

  return commit;
}

// Retains blocks safely behind a rollback window and hands Streamdown only the active tail;
// retained blocks are put back in the block list, so output and React keys stay identical.
export class IncrementalMarkdownCache {
  private source = "";
  private tail = "";
  private committedBlocks: string[] = [];
  private committedLength = 0;
  private commitPoints: CommitPoint[] = [];
  private context = createRetainedContext();
  private fenceTracker = new OpenFenceTracker();
  private openFenceHead: string | null = null;
  private openFenceHeadInert = false;
  private fullDocumentMode = false;
  private lastMarkdown: string | null = null;
  private droppedRetainedBlocks = false;
  // Tests read these to hold the rewind path in place.
  private retainedPrefixRebuilds = 0;
  private rewoundCharacters = 0;
  // Bumped only when the Markdown string alone cannot signal a changed render.
  renderGeneration = 0;

  readonly parseMarkdownIntoBlocks = (markdown: string): string[] => [
    ...this.committedBlocks,
    ...parseMarkdownIntoRenderableBlocks(markdown),
  ];

  // Streamdown memoises on the Markdown string and ignores the parser callback, so dropping
  // retained blocks must move the render identity instead.
  private render(markdown: string): IncrementalMarkdownRender {
    if (this.droppedRetainedBlocks && markdown === this.lastMarkdown) {
      this.renderGeneration += 1;
    }
    this.droppedRetainedBlocks = false;
    this.lastMarkdown = markdown;
    return { markdown, parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks };
  }

  // Once the probe shows remend leaves the head alone, it is paid once per fence, not per chunk.
  private repairOpenFence(): string | null {
    const bounds = this.fenceTracker.spliceBounds(this.tail);
    if (bounds === null) {
      return null;
    }
    const head =
      repairContextPrefix(this.context) + this.tail.slice(0, bounds.bodyStart);
    if (head !== this.openFenceHead) {
      this.openFenceHead = head;
      const probed = head + OPEN_FENCE_PROBE;
      this.openFenceHeadInert = remend(probed) === probed;
    }
    return this.openFenceHeadInert
      ? repairOpenFenceTail(this.tail, bounds.bodyStart, bounds.firstBodyMarker)
      : null;
  }

  private resetIncrementalState(markdown: string): void {
    this.droppedRetainedBlocks ||= this.committedBlocks.length > 0;
    this.source = markdown;
    this.tail = markdown;
    this.committedBlocks = [];
    this.committedLength = 0;
    this.commitPoints = [];
    this.context = createRetainedContext();
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {
    this.resetIncrementalState(markdown);
    this.fullDocumentMode = true;
    return this.render(repairStreamingMarkdown(markdown));
  }

  // preprocessLaTeX and closing fences rewrite already emitted spans; rewind to the last commit
  // the rewrite cannot reach. Mutates nothing before returning false.
  private rewindToRewrite(markdown: string): boolean {
    if (this.commitPoints.length === 0) {
      return false;
    }

    const committedPrefix = this.source.slice(0, this.committedLength);
    const shared = hasPrefix(markdown, committedPrefix)
      ? this.committedLength +
        sharedPrefixLength(markdown.slice(this.committedLength), this.tail)
      : sharedPrefixLength(markdown, committedPrefix);

    // Unchanged characters alone are not a safe boundary; stop at the last intact blank line.
    const safeLimit = lastBlankLineEnd(this.source, shared);

    // A blank line is not a wall: Marked merges runs of them and lists reopen across one, so keep
    // ROLLBACK_BLOCKS blocks of margin from the first changed character.
    let blocksBeforeLimit = this.committedBlocks.length;
    let scanned = this.committedLength;
    while (blocksBeforeLimit > 0 && scanned > safeLimit) {
      blocksBeforeLimit -= 1;
      scanned -= this.committedBlocks[blocksBeforeLimit].length;
    }
    const blockLimit = blocksBeforeLimit - ROLLBACK_BLOCKS;
    if (blockLimit <= 0) {
      return false;
    }

    let index = this.commitPoints.length - 1;
    while (
      index >= 0 &&
      (this.commitPoints[index].length > safeLimit ||
        this.commitPoints[index].blockCount > blockLimit)
    ) {
      index -= 1;
    }
    if (index < 0) {
      return false;
    }

    const point = this.commitPoints[index];
    if (point.length < this.committedLength) {
      this.rewoundCharacters += this.committedLength - point.length;
      this.committedBlocks.length = point.blockCount;
      this.committedLength = point.length;
      this.commitPoints.length = index + 1;
      this.droppedRetainedBlocks = true;
    }

    this.context = point.context;
    this.tail = markdown.slice(this.committedLength);
    return true;
  }

  private updateTail(markdown: string): void {
    if (hasPrefix(markdown, this.source)) {
      this.tail += markdown.slice(this.source.length);
    } else if (!this.rewindToRewrite(markdown)) {
      if (this.committedBlocks.length > 0) {
        this.retainedPrefixRebuilds += 1;
      }
      this.resetIncrementalState(markdown);
      this.fullDocumentMode = false;
    }
    this.source = markdown;
  }

  update(rawMarkdown: string): IncrementalMarkdownRender {
    // Streamdown returns LF-normalised blocks; normalise first or a CRLF reply never commits.
    const markdown = normalizeLineEndings(rawMarkdown);

    // The coalescer hands the same text to several renders; skip the repeated work.
    if (markdown === this.source && this.lastMarkdown !== null) {
      return {
        markdown: this.lastMarkdown,
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    if (this.fullDocumentMode && hasPrefix(markdown, this.source)) {
      this.source = markdown;
      return this.renderFullDocument(markdown);
    }

    this.updateTail(markdown);

    const repaired = repairTailKeepingLinks(
      this.tail,
      this.context,
      this.repairOpenFence() ?? undefined,
    );

    // Global definitions must render in the same document as their uses. Computed late because the
    // precise scope costs a lex of everything received so far.
    if (
      FOOTNOTE_REFERENCE_RE.test(repaired) ||
      FOOTNOTE_DEFINITION_RE.test(repaired) ||
      (hasLinkDefinition(this.tail) &&
        markdownRenderScope(markdown) === "document")
    ) {
      return this.renderFullDocument(markdown);
    }
    const blocks = parseMarkdownIntoBlocks(repaired);

    const candidateCount = Math.max(0, blocks.length - ROLLBACK_BLOCKS);
    if (candidateCount === 0) {
      return this.render(repaired);
    }

    const commit = findCommitBoundary(
      this.tail,
      blocks,
      candidateCount,
      this.context.latex,
    );

    // A mid-string repair never becomes a raw prefix, so that fallback is sticky, as is an over-budget
    // tail. Below the budget, an unbalanced marker may still close, so retry.
    if (!commit.parity) {
      if (commit.repairBroke || this.tail.length > STALLED_TAIL_CHARACTERS) {
        return this.renderFullDocument(markdown);
      }
      return this.render(repaired);
    }

    const committedText = this.tail.slice(0, commit.length);
    const nextContext = advanceContext(
      this.context,
      commit.parity,
      committedText,
    );
    const nextTail = this.tail.slice(commit.length);
    const nextMarkdown = repairTailKeepingLinks(nextTail, nextContext);

    // An unchanged string would make Streamdown skip the render; keep the blocks live until it changes.
    if (nextMarkdown === this.lastMarkdown) {
      return this.render(repaired);
    }

    this.committedBlocks.push(...blocks.slice(0, commit.count));
    this.committedLength += commit.length;
    this.context = nextContext;
    this.tail = nextTail;
    this.commitPoints.push({
      blockCount: this.committedBlocks.length,
      length: this.committedLength,
      context: nextContext,
    });

    return this.render(nextMarkdown);
  }
}

export function withoutStreamdownAnimationPlugin(
  rehypePlugins: BlockProps["rehypePlugins"],
  animatePlugin: BlockProps["animatePlugin"],
): BlockProps["rehypePlugins"] {
  const animationPlugin = animatePlugin?.rehypePlugin;
  if (!animationPlugin) {
    return rehypePlugins;
  }

  return rehypePlugins?.filter((plugin) => {
    const pluginFunction = Array.isArray(plugin) ? plugin[0] : plugin;
    return pluginFunction !== animationPlugin;
  });
}
