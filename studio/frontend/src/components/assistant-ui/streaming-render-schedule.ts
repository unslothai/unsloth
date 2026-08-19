// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

// How far behind the live edge a block has to be before it can be retained.
// The block list interleaves "\n\n" separators, so this is about four
// paragraphs of slack for a construct that a later line can still reinterpret.
const ROLLBACK_BLOCKS = 8;
// A marker the reply never closes leaves the tail growing with nothing to
// retain, and the boundary scan is then paid on top of the full repair it was
// meant to replace. Give up at a character budget, since characters are what
// that scan costs, and a transient imbalance closes far below this. The budget
// is spent before giving up and that spending grows with the square of the tail,
// so the value matters. Measured on an emphasis marker that only closes 40,000
// characters later, 420 updates, five repetitions: the median cost against the
// full-document path is +73% at 32,768 and +1.6% at 8,192. 8,192 characters is
// still around 1,300 words of slack, well beyond a marker a later line closes.
const STALLED_TAIL_CHARACTERS = 8_192;
// Balanced marker prefixes preserve the whole-document facts that remend uses
// to decide how an incomplete tail should close, without changing parity.
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";
const BOLD_CONTEXT = "**x**\n\n";
const SINGLE_ASTERISK_CONTEXT = "*x*\n\n";
const SINGLE_UNDERSCORE_CONTEXT = "_x_\n\n";
const INLINE_CODE_ASTERISK_CONTEXT = "`a *b* c`\n\n";
const INLINE_CODE_UNDERSCORE_CONTEXT = "`a _b_ c`\n\n";
const FOOTNOTE_REFERENCE_RE = /\[\^[\w-]{1,200}\](?!:)/;
const FOOTNOTE_DEFINITION_RE = /\[\^[\w-]{1,200}\]:/;
const LINK_DEFINITION_RE = /\[(?:\\.|[^\]\n\\]){1,200}\]:/;
const FENCED_CODE_BLOCK_RE = /^ {0,3}(?:```|~~~)/;
const WORD_CHARACTER_RE = /[\p{L}\p{N}_]/u;
const HTML_TAG_START_RE = /[a-zA-Z/]/;

type RepairParity = {
  bold: boolean;
  boldCandidate: boolean;
  boldFence: boolean;
  bracketDepth: number;
  linkDefinition: boolean;
  doubleUnderscore: boolean;
  emphasisDisplayMath: boolean;
  emphasisInlineCode: boolean;
  emphasisInlineMath: boolean;
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

const createRepairParity = (): RepairParity => ({
  bold: false,
  boldCandidate: false,
  boldFence: false,
  bracketDepth: 0,
  linkDefinition: false,
  doubleUnderscore: false,
  emphasisDisplayMath: false,
  emphasisInlineCode: false,
  emphasisInlineMath: false,
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

// Marked keeps link reference definitions in one document-wide map and emits
// no token for a label it has already seen, so a definition that is retained
// while its twin is still live would be lexed apart and shown as a literal
// line. Keeping every definition in the live tail makes the two lexes agree.
// Marked reads a fenced block as code, so those do not count; anything else
// that merely looks like a definition costs retention, never correctness.
function updateLinkDefinitionParity(parity: RepairParity, text: string): void {
  if (!FENCED_CODE_BLOCK_RE.test(text) && LINK_DEFINITION_RE.test(text)) {
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

function updateEmphasisMathParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (text[index] === "\\" && text[index + 1] === "$") {
    return index + 1;
  }
  if (text[index] !== "$") {
    return index;
  }
  if (text[index + 1] === "$") {
    parity.emphasisDisplayMath = !parity.emphasisDisplayMath;
    parity.emphasisInlineMath = false;
    return index + 1;
  }
  if (!parity.emphasisDisplayMath) {
    parity.emphasisInlineMath = !parity.emphasisInlineMath;
  }
  return index;
}

function countsAsSingleAsterisk(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  if (
    previous === "\\" ||
    parity.emphasisDisplayMath ||
    parity.emphasisInlineMath
  ) {
    return false;
  }
  if (previous !== "*" && next === "*") {
    return text[index + 2] === "*";
  }
  if (
    previous === "*" ||
    (isWordCharacter(previous) && isWordCharacter(next))
  ) {
    return false;
  }
  const previousIsBoundary =
    previous === undefined ||
    previous === " " ||
    previous === "\t" ||
    previous === "\n";
  const nextIsBoundary =
    next === undefined || next === " " || next === "\t" || next === "\n";
  return !(previousIsBoundary && nextIsBoundary);
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
    parity.emphasisDisplayMath ||
    parity.emphasisInlineMath
  ) {
    return false;
  }
  const previousIsBoundary =
    previous === undefined ||
    previous === " " ||
    previous === "\t" ||
    previous === "\n";
  const nextIsBoundary =
    next === undefined || next === " " || next === "\t" || next === "\n";
  return !(
    (previousIsBoundary && nextIsBoundary) ||
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
    parity.emphasisDisplayMath ||
    parity.emphasisInlineMath ||
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
    parity.emphasisDisplayMath ||
    parity.emphasisInlineMath ||
    isWithinLinkDestination(text, index) ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

// Remend decides these closers from marker parity over the whole document, so a
// retained prefix must end with neutral parity: repairing the tail alone could
// otherwise add or omit a closer that a full repair would place differently.
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
  if (countsAsSingleAsterisk(parity, text, index)) {
    parity.singleAsterisk = !parity.singleAsterisk;
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

// Remend finds the marker that orders its closers with a raw indexOf("**") but
// counts pairs only outside fenced code, so a `**` in a fence has to seed the
// bold context without reaching the fence-aware counter in updateAsteriskParity.
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

// Remend completes a dangling link by appending to the end of the whole
// document, so a retained block still holding an unmatched bracket would move
// that completion out of the tail. Brackets in code do not count, as in remend.
function updateBracketDepth(parity: RepairParity, character: string): void {
  if (character === "[") {
    parity.bracketDepth += 1;
  } else if (character === "]" && parity.bracketDepth > 0) {
    parity.bracketDepth -= 1;
  }
}

// Remend consumes an escaped backtick before testing for a fence, so a fence
// that starts one character after a backslash is not a fence. Returns the index
// to resume from, or the same index when neither applies.
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

function updateEmphasisParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
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

// Remend's display-math scan stops one character early: it only reads whole
// documents, where the last character cannot open a pair. This one runs per
// retained block, whose boundaries are interior, so it counts to text.length.
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

const hasNeutralRepairParity = (parity: RepairParity): boolean =>
  ![
    parity.bracketDepth > 0,
    parity.linkDefinition,
    parity.bold,
    parity.boldFence,
    parity.emphasisInlineCode,
    parity.doubleUnderscore,
    parity.emphasisDisplayMath,
    parity.emphasisInlineMath,
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

// Marker facts the retained prefix carries into the tail repair.
type RetainedContext = {
  multilineKatex: boolean;
  bold: boolean;
  singleAsterisk: boolean;
  singleUnderscore: boolean;
  firstSingleAsterisk: "inlineCode" | "normal" | null;
  firstSingleUnderscore: "inlineCode" | "normal" | null;
  firstBoldOrSingleUnderscore: "bold" | "singleUnderscore" | null;
};

const createRetainedContext = (): RetainedContext => ({
  multilineKatex: false,
  bold: false,
  singleAsterisk: false,
  singleUnderscore: false,
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

// Taking the context as a value lets a candidate commit be priced before it is
// applied, which the repeated-Markdown check in update() needs.
function repairTail(tail: string, context: RetainedContext): string {
  const prefix =
    emphasisContext(context) +
    singleAsteriskContext(context) +
    (context.multilineKatex ? MULTILINE_KATEX_CONTEXT : "");
  if (!prefix) {
    return remend(tail);
  }
  return remend(prefix + tail).slice(prefix.length);
}

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
  // The parity at the boundary, or null when no block can be retained.
  parity: RepairParity | null;
  repairBroke: boolean;
};

// Where one commit left the retained prefix. `advanceContext` only ever adds
// facts and cannot be undone, so each commit's context is stored to allow a
// rewind to an earlier boundary.
type CommitPoint = {
  blockCount: number;
  length: number;
  context: RetainedContext;
};

// CommonMark counts a line feed, a lone carriage return, and a carriage return
// followed by a line feed as the same line ending, and reference parsers
// normalise to LF before parsing, so this cannot change what is rendered. The
// scan runs per frame and costs nothing next to the repair and lex it protects
// (0.4 us on an 88,000 character reply); the replace only runs for a reply that
// actually carries a carriage return.
function normalizeLineEndings(text: string): string {
  return text.includes("\r") ? text.replace(/\r\n?/g, "\n") : text;
}

/**
 * `a` begins with `b`.
 *
 * `String.prototype.startsWith` is the obvious spelling and is far slower here,
 * because it scans where slicing to the prefix length and comparing lets the
 * engine reject on length and then compare natively.
 *
 * The win is in the PREFIX length, not in how the strings are represented. Swept
 * on V8: at an 8 character prefix `startsWith` is FASTER (0.4x), at 1,024 it is
 * 105x slower and at 60,000 it is 250x slower, and a cons receiver behaves the
 * same as a flat one (250x against 254x). So do not reach for this helper for
 * short prefixes; it earns its place on a reply-length one.
 *
 * Semantically identical: `slice` clamps to the string length, so a `b` longer
 * than `a` yields a short slice that cannot equal it.
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

// Does `text` end at a blank line, counting the start of the document as one?
// Unchanged characters alone cannot keep a block across a rewrite: Marked reads
// `paragraph\n` plus new text as a lazy continuation, so an edit that closes up
// a blank line re-segments the paragraph before it even though that paragraph's
// characters never moved. `\r` counts as line-ending whitespace, so CRLF reads
// the same as LF.
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

// The last blank line at or before `limit`, or 0 for none. An untouched blank
// line between the boundary and the rewrite carries most of the insulation, so
// the boundary need not land on the blank line itself. Not all of it: see
// `rewindToRewrite`, which also keeps a rollback window of blocks between the
// boundary and the first changed character.
function lastBlankLineEnd(text: string, limit: number): number {
  for (let end = limit; end > 0; end -= 1) {
    if (endsAtBlankLine(text, end)) {
      return end;
    }
  }
  return 0;
}

// Remend may synthesize closing syntax at the end of an incomplete tail.
// Never retain synthetic or mid-string repaired text. Scan forward once,
// recording the latest exact boundary whose global repair parity is neutral.
function findCommitBoundary(
  tail: string,
  blocks: string[],
  candidateCount: number,
): CommitBoundary {
  const parity = createRepairParity();
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

// Streamdown normally repairs and lexes the entire growing reply on every
// update. Retain blocks that are safely behind a rollback window and give
// Streamdown only the active tail. The parser callback puts the retained blocks
// back into its block list, so output and React keys stay identical.
export class IncrementalMarkdownCache {
  private source = "";
  private tail = "";
  private committedBlocks: string[] = [];
  private committedLength = 0;
  private commitPoints: CommitPoint[] = [];
  private context = createRetainedContext();
  private fullDocumentMode = false;
  private lastMarkdown: string | null = null;
  private droppedRetainedBlocks = false;
  // How often a rewrite discarded the whole retained prefix, and how many
  // characters a rewind handed back to the live tail. Both redo work with an
  // identical result, so time is the only other evidence they happened. Tests
  // read these to hold the rewind path in place.
  private retainedPrefixRebuilds = 0;
  private rewoundCharacters = 0;
  // Bumped only when the Markdown string alone cannot signal a changed render.
  renderGeneration = 0;

  readonly parseMarkdownIntoBlocks = (markdown: string): string[] => [
    ...this.committedBlocks,
    ...parseMarkdownIntoBlocks(markdown),
  ];

  // Streamdown memoises the whole component on the Markdown string and ignores
  // the parser callback, so the string is the only thing that can schedule a
  // render. Retaining a block can wait for a string that differs, but dropping
  // retained blocks cannot, so that case moves the render identity instead.
  private render(markdown: string): IncrementalMarkdownRender {
    if (this.droppedRetainedBlocks && markdown === this.lastMarkdown) {
      this.renderGeneration += 1;
    }
    this.droppedRetainedBlocks = false;
    this.lastMarkdown = markdown;
    return { markdown, parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks };
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
    return this.render(remend(markdown));
  }

  // The text handed to the cache is not always an extension of the last one:
  // `preprocessLaTeX` rewrites an already emitted span when a `\(...\)` closes,
  // a `\[...\]` becomes a `$$` block or a currency `$` turns out not to open
  // math, and a closing fence rewrites its own body. Each edits one span, so
  // rewind to the last commit the rewrite can neither reach nor re-segment
  // rather than discarding the whole prefix. Returns false when nothing
  // survives and the caller has to start over; mutates nothing before then, so
  // the reset fallback never sees a half-rewound cache. `fullDocumentMode`
  // cannot be set here: only `renderFullDocument` raises it, and it clears
  // `commitPoints` first, so the guard below has already returned.
  private rewindToRewrite(markdown: string): boolean {
    if (this.commitPoints.length === 0) {
      return false;
    }

    // Characters the rewrite left alone. Usually the rewrite is inside the live
    // tail, and `slice` shares the original's characters, so that case costs
    // one native comparison plus a scan of the tail (about to be re-lexed
    // anyway) rather than a scan of the whole reply.
    const committedPrefix = this.source.slice(0, this.committedLength);
    const shared = hasPrefix(markdown, committedPrefix)
      ? this.committedLength +
        sharedPrefixLength(markdown.slice(this.committedLength), this.tail)
      : sharedPrefixLength(markdown, committedPrefix);

    // Unchanged characters alone do not make a boundary safe to keep, so stop
    // at the last blank line the rewrite left intact.
    const safeLimit = lastBlankLineEnd(this.source, shared);

    // A blank line is not a wall either: Marked merges a run of them into one
    // separator block, so inserting a newline (as `\[...\]` -> `\n$$\n` does)
    // re-segments the separator in front of it, and a list or indented code
    // block reopens across one. So demand the same margin the append path
    // requires -- ROLLBACK_BLOCKS blocks behind the boundary -- measured from
    // the first changed character. Counting backwards costs only the blocks
    // near the rewrite.
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
    // Every boundary this class finds is a byte offset into the text it was
    // handed, but the blocks it compares against come back from Streamdown with
    // their line endings already normalised. On a CRLF reply the two disagree
    // one block in, `findCommitBoundary` sees `"para"` where the source holds
    // `"para\r"`, nothing is ever committed, and the whole reply re-repairs and
    // re-lexes on every frame. Normalise first so both sides speak LF.
    const markdown = normalizeLineEndings(rawMarkdown);

    // Tokens arrive faster than frames, so the coalescer hands the same text to
    // several renders. Nothing about the result can differ, and repeating the
    // work would be the whole reply again once the document path is in use.
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

    const repaired = repairTail(this.tail, this.context);

    // Streamdown deliberately turns a repaired document containing footnotes
    // into one block so definitions can resolve references anywhere in the
    // document. Such a construct is globally scoped and cannot retain a prefix.
    if (
      FOOTNOTE_REFERENCE_RE.test(repaired) ||
      FOOTNOTE_DEFINITION_RE.test(repaired)
    ) {
      return this.renderFullDocument(markdown);
    }

    const blocks = parseMarkdownIntoBlocks(repaired);

    const candidateCount = Math.max(0, blocks.length - ROLLBACK_BLOCKS);
    if (candidateCount === 0) {
      return this.render(repaired);
    }

    const commit = findCommitBoundary(this.tail, blocks, candidateCount);

    // A mid-string repair can never become a raw prefix on a later append, so
    // make that fallback sticky, and do the same once the tail grows past the
    // budget with nothing to show for it. A temporarily unbalanced marker can
    // close in a later block, so below the budget keep the tail live and retry.
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
    const nextMarkdown = repairTail(nextTail, nextContext);

    // A repeating reply can leave the tail unchanged once a block is retained.
    // Streamdown would then see the Markdown it already holds and skip the
    // render, so the retained blocks would never be displayed. Keep them in the
    // live tail instead; the next update commits them with a longer string.
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
