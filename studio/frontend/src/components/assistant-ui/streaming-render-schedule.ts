// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCompletedFencedCodeOccurrences } from "../../lib/fenced-code-provenance.ts";

import { fromMarkdown } from "mdast-util-from-markdown";

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

import {
  getTerminalStreamingCodeFence,
  hasClosingFenceLine,
  isClosingFenceLine,
  normalizeCodeFenceLanguage,
  readFenceMarker,
} from "./streaming-code-policy.ts";

// Retain blocks only after they are outside the rollback window.
const ROLLBACK_BLOCKS = 8;
// Group retained blocks so token updates touch only the final partial chunk.
const COMMITTED_CHUNK_BLOCKS = 8;
// Bound stalled-tail scans; 8,192 characters preserves enough context for late closures.
const STALLED_TAIL_CHARACTERS = 8_192;
// remend repairs the trailing incomplete construct, but it reaches that
// decision with whole-string passes: one walks the text once per `[` looking
// for an unterminated link, another escapes comparison operators line by line.
// Inside an unterminated fence every such character is literal and every pass
// declines, so those walks cost the length of the fence and change nothing. The
// fence also lexes into a single block, so `candidateCount` in update() is 0,
// nothing is ever retained, and the tail grows with the fence: appending one
// chunk then costs O(fence) and a reply costs the square of it. Repair the head
// plus a window of the fence body instead, and splice the untouched middle back
// in. The window carries the last line and the line before it, which is all the
// trailing repairs read.
const OPEN_FENCE_REPAIR_WINDOW = 4_096;
// A plain two-line probe. When remend leaves the head alone with ordinary text
// after it, no marker is pending for it to close, so it cannot append anything
// beyond the window either. Without it a marker left open before the fence gets
// its closer at the wrong offset.
//
// Its verdict is a property of the HEAD, while remend decides from the whole
// string, so on its own it is not enough: text later in the body can flip a
// global parity the probe assumed and revive a marker it cleared. The body
// marker refusal below is what closes that gap, and the two only work together.
const OPEN_FENCE_PROBE = "\nq\n";
// An index can only be resolved once the two characters after it are known, so
// the scan keeps that many back from the live edge and re-reads them.
const FENCE_SCAN_MARGIN = 2;
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

// Keep link definitions in the live tail so Marked's document-wide map stays consistent.
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

export type IncrementalMarkdownCodeFence = Readonly<{
  id: string;
  language: string | null;
  meta: string | null;
  openingOffset: number;
  source: string;
  // Set only when live source preservation or a validated cold overlay owns
  // the otherwise ambiguous line feed immediately before the closing fence.
  terminalLineFeedIsExact?: true;

}>;

export type IncrementalMarkdownBlock = Readonly<{
  codeFences: readonly IncrementalMarkdownCodeFence[];
  content: string;
  id: string;
}>;

export type IncrementalMarkdownChunk = Readonly<{
  blocks: readonly IncrementalMarkdownBlock[];
  id: string;
  startIndex: number;
}>;

export type IncrementalMarkdownTerminalCodeTail = Readonly<{
  blockId: string;
  fenceMarkdown: string;
  id: string;
  isClosed: boolean;
  language: string | null;
  openingLine: string;
  openingOffset: number;
  prefixBlocks: readonly IncrementalMarkdownBlock[];
  prefixMarkdown: string;
  source: string;
  sourceOffset: number;
}>;

export type IncrementalMarkdownRender = Readonly<{
  chunks: readonly IncrementalMarkdownChunk[];
  committedBlockCount: number;
  epoch: number;
  // The repaired Markdown still owned by the mutable rollback tail.
  markdown: string;
  // Streamdown parses this one block only so its public BlockComponent receives
  // the dependency's exact incomplete-fence state beneath all private providers.
  shellMarkdown: string;
  // The same shell before remend appends synthetic closing markers. Presentation
  // and copy paths use this exact source; Streamdown still parses shellMarkdown.
  sourceShellMarkdown: string;
  tail: readonly IncrementalMarkdownBlock[];
  terminalCodeTail: IncrementalMarkdownTerminalCodeTail | null;
}>;


type IncrementalMarkdownRenderObservation = Readonly<{
  codeFenceSourceLengths: readonly number[];
  isStreaming: boolean;
  sourceLength: number;
  terminalSourceLength: number | null;
}>;

let incrementalMarkdownRenderObserver:
  | ((observation: IncrementalMarkdownRenderObservation) => void)
  | null = null;

export function observeIncrementalMarkdownRenders(
  observer: (observation: IncrementalMarkdownRenderObservation) => void,
): () => void {
  incrementalMarkdownRenderObserver = observer;
  return () => {
    if (incrementalMarkdownRenderObserver === observer) {
      incrementalMarkdownRenderObserver = null;
    }
  };
}

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
function repairContextPrefix(context: RetainedContext): string {
  return (
    emphasisContext(context) +
    singleAsteriskContext(context) +
    (context.multilineKatex ? MULTILINE_KATEX_CONTEXT : "")
  );
}

function repairTail(tail: string, context: RetainedContext): string {
  const prefix = repairContextPrefix(context);
  if (!prefix) {
    return remend(tail);
  }
  return remend(prefix + tail).slice(prefix.length);
}

// Where remend believes a fence is open. It toggles on any ``` run, wherever on
// the line that run sits, and a backslash escapes the backtick after it, so this
// deliberately mirrors remend rather than CommonMark: a mid-line ``` closes the
// fence for remend and must close it here too.
type OpenFenceState = {
  index: number;
  fenceOpen: boolean;
  bodyStart: number;
  // The FIRST ` $ or ~ in the current fence body, or -1. Its mere presence
  // disqualifies the splice, so only whether one exists ever matters; the index
  // is kept because it reads better in a test than a bare boolean.
  //
  // remend does not have one notion of "inside a fence"; it has at least three,
  // and they disagree. `isWithinCodeBlock` walks the text and honours a
  // backslash before a backtick, the emphasis counters toggle on ``` without
  // honouring it, and the inline-code repair just counts /```/g. An escaped
  // \``` or a ```` run flips the last of those and not the first. The opener
  // that stands in for the elided text can only reproduce all three when the
  // elided part holds none of the characters any of them counts.
  //
  // The first such character, not the last: a marker sitting in the window
  // would otherwise mask an earlier one sitting in the elided middle, which is
  // the half that matters. First is also monotone, so it costs nothing to keep.
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
      // Escaped for `isWithinCodeBlock`, not for the passes that only count
      // ``` runs, so the backtick still counts as a marker.
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

// Carries the scan across updates so a growing tail costs the characters that
// arrived, not the characters it holds. Any text that is not an extension of the
// last one rescans from the start, which is what a commit or a rewrite hands it.
class OpenFenceTracker {
  private text = "";
  private resolved = initialOpenFenceState();

  // Where the open fence's body starts and where its first marker sits, or null
  // when the whole-tail repair applies.
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

// A head the probe found inert contributes nothing but "a fence is open", which
// one opener reproduces. Standing in for it keeps the repair off the head as
// well as off the elided body.
const OPEN_FENCE_SYNTHETIC_HEAD = "```\n";

// The spliced repair, or null when it cannot be shown to match repairTail. All
// four refusals fall back to repairing the whole tail:
//
//   a body still shorter than the window has nothing to elide;
//   a final line longer than the window leaves no line boundary to cut on,
//     which is the only place the splice can start without changing what the
//     trailing repairs read;
//   a PRECEDING line longer than the window puts the cut on the newline that
//     ends it, so the window holds the final line alone. remend's setext repair
//     reads exactly one line back, and the synthetic opener standing in for it
//     is never blank, so a whitespace-only line elided that way turns a tail
//     ending in `-`, `--`, `=` or `==` into one carrying a zero-width space
//     that repairing the whole tail does not add, and that the copy button
//     would then put on the clipboard;
//   a ` $ or ~ ANYWHERE in the body, elided or retained, is a character one of
//     remend's fence notions counts, and the opener cannot stand in for it.
//
// The last one deliberately covers the retained window and not just the elided
// middle, because the probe that licenses the opener reads the head ALONE while
// remend decides from the whole string. An escaped \``` in the window flips the
// global triple-run parity, so the whole-tail repair closes an unmatched inline
// backtick that sits before the opener and the spliced output does not; a `$$`
// there moves the math parity the other way. Neither is reachable from a
// refusal on the cut, since the probe runs before the cut is chosen. Keeping
// the body free of all three characters is what makes the probe's verdict a
// property of the whole tail rather than of the head it was handed.
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
/*
 * Is there a footnote reference or definition here, ignoring fenced code?
 *
 * A character class in ordinary code is shaped exactly like a footnote
 * reference: `/[^a-z]/` matches `FOOTNOTE_REFERENCE_RE`. Testing the repaired
 * tail as one string therefore reads a regex literal inside an open fence as a
 * footnote and switches to full-document mode, which is sticky, so every later
 * token re-repairs and re-lexes the whole reply. That is the quadratic this
 * class exists to remove, restored by one line of the reply's own code.
 *
 * Fenced code is not Markdown, so scan lines and skip the fenced ones. Neither
 * pattern can span a newline, so testing per line is exactly the whole-string
 * test with the fenced lines removed.
 */
export function hasFootnoteConstruct(markdown: string): boolean {
  let fence: { marker: "`" | "~"; markerLength: number } | null = null;
  for (let lineStart = 0; lineStart <= markdown.length; ) {
    const lf = markdown.indexOf("\n", lineStart);
    const rawEnd = lf < 0 ? markdown.length : lf;
    const end =
      rawEnd > lineStart && markdown[rawEnd - 1] === "\r" ? rawEnd - 1 : rawEnd;
    const line = markdown.slice(lineStart, end);

    if (fence) {
      if (isClosingFenceLine(line, fence.marker, fence.markerLength)) {
        fence = null;
      }
    } else {
      const opening = readFenceMarker(markdown, lineStart);
      if (opening) {
        fence = opening;
      } else if (
        FOOTNOTE_REFERENCE_RE.test(line) ||
        FOOTNOTE_DEFINITION_RE.test(line)
      ) {
        return true;
      }
    }

    if (lf < 0) break;
    lineStart = lf + 1;
  }
  return false;
}

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
// update. Retain blocks that are safely behind a rollback window and expose
// them as immutable chunks plus one mutable tail. MarkdownText renders this
// plan through a single Streamdown provider shell, so neither Streamdown nor
// React walks the complete growing block list for every token.
type MarkdownCodeNode = {
  children?: readonly MarkdownCodeNode[];
  lang?: string | null;
  meta?: string | null;
  position?: { start: { offset?: number } };
  type: string;
  value?: string;
};

// A globally scoped Streamdown block (notably one containing footnotes) can
// hold many completed nodes. Preserve canonical fenced-code facts before a
// terminal tail is promoted or followed by prose.
export function getCompletedCodeFences(
  content: string,
  blockId = "detached",
): IncrementalMarkdownCodeFence[] {
  const fences: IncrementalMarkdownCodeFence[] = [];
  const visit = (node: MarkdownCodeNode): void => {
    if (node.type === "code") {
      const start = node.position?.start.offset;
      if (start !== undefined) {
        const lineEnd = content.indexOf("\n", start);
        const openingLine = content.slice(
          start,
          lineEnd < 0 ? content.length : lineEnd,
        );
        // Slicing to "\n" leaves the CR of a CRLF opening line in place.
        if (/^ {0,3}(?:`{3,}|~{3,})[^\r\n]*\r?$/.test(openingLine)) {
          fences.push({
            id: `${blockId}:fence:${start}`,
            language: normalizeCodeFenceLanguage(node.lang ?? null),
            meta: node.meta ?? null,
            openingOffset: start,
            source: node.value ?? "",
          });
        }
      }
    }
    node.children?.forEach(visit);
  };
  visit(fromMarkdown(content) as MarkdownCodeNode);
  return fences;
}

/*
 * The same answer for a block that only grew, without re-parsing it.
 *
 * A block holding an unclosed fence is the whole live tail, and it is re-lexed
 * on every token, so charging it a whole-document CommonMark parse each time
 * makes one chunk cost the length of the fence and a reply the square of it.
 * That is the exact shape #9517 removed from the repair path.
 *
 * While the last fence is open nothing else in the block can change: CommonMark
 * ends a fenced code block only at a matching closing line or at the end of the
 * document, so every earlier fence is settled and the appended characters can
 * only extend that one body. So carry the previous result forward and re-parse
 * only when a closing line actually arrives, which is once per fence.
 */
type OpenFenceMemo = {
  blockId: string;
  content: string;
  fences: IncrementalMarkdownCodeFence[];
  marker: "`" | "~";
  markerLength: number;
};

// What mdast reports as the body of a still-open fence: everything after the
// opening line, less the single trailing line ending it drops.
const openFenceBody = (content: string, openingOffset: number): string => {
  const lf = content.indexOf("\n", openingOffset);
  if (lf < 0) return "";
  const body = content.slice(lf + 1);
  return body.endsWith("\n") ? body.slice(0, -1) : body;
};

// A block whose last fence runs unclosed to its end, or null. A closed fence
// cannot pass: its body stops before a closer that is still in the content. Nor
// can a half-written opening line, where the next character joins the info
// string rather than the body.
const openFenceOf = (
  content: string,
  fences: readonly IncrementalMarkdownCodeFence[],
): Pick<OpenFenceMemo, "marker" | "markerLength"> | null => {
  const last = fences.at(-1);
  if (
    !last ||
    content.indexOf("\n", last.openingOffset) < 0 ||
    last.source !== openFenceBody(content, last.openingOffset)
  ) {
    return null;
  }
  return readFenceMarker(content, last.openingOffset);
};

export function createCompletedCodeFenceCache(): (
  content: string,
  blockId: string,
) => IncrementalMarkdownCodeFence[] {
  let memo: OpenFenceMemo | null = null;

  return (content, blockId) => {
    if (
      memo !== null &&
      memo.blockId === blockId &&
      content.length > memo.content.length &&
      hasPrefix(content, memo.content) &&
      // A closer has to start a line, and the appended text may continue the
      // block's last partial one, so rescan from that line's start.
      !hasClosingFenceLine(
        content,
        memo.content.lastIndexOf("\n") + 1,
        memo.marker,
        memo.markerLength,
      )
    ) {
      const previous = memo.fences;
      const last = previous[previous.length - 1];
      const fences = [
        ...previous.slice(0, -1),
        { ...last, source: openFenceBody(content, last.openingOffset) },
      ];
      memo = { ...memo, content, fences };
      return fences;
    }

    const fences = getCompletedCodeFences(content, blockId);
    const open = openFenceOf(content, fences);
    memo = open && { ...open, blockId, content, fences };
    return fences;
  };
}


export class IncrementalMarkdownCache {
  private source = "";
  private tail = "";
  private committedBlocks: IncrementalMarkdownBlock[] = [];
  private committedChunks: IncrementalMarkdownChunk[] = [];
  private tailBlocks: IncrementalMarkdownBlock[] = [];
  private committedLength = 0;
  private commitPoints: CommitPoint[] = [];
  private context = createRetainedContext();
  private fenceTracker = new OpenFenceTracker();
  private openFenceHead: string | null = null;
  private openFenceHeadInert = false;
  private fullDocumentMode = false;
  private lastRender: IncrementalMarkdownRender | null = null;
  private lastCanonicalRender: IncrementalMarkdownRender | null = null;
  private nextBlockIdentity = 0;
  private epoch = 0;

  private persistedTrailingLfOrdinals: readonly number[];
  constructor(persistedTrailingLfOrdinals: readonly number[] = []) {
    this.persistedTrailingLfOrdinals = [...persistedTrailingLfOrdinals];
  }

  private updatePersistedFenceProvenance(
    next: readonly number[],
  ): boolean {
    if (
      next.length === this.persistedTrailingLfOrdinals.length &&
      next.every(
        (ordinal, index) =>
          ordinal === this.persistedTrailingLfOrdinals[index],
      )
    ) {
      return false;
    }
    this.persistedTrailingLfOrdinals = [...next];
    return true;
  }

  private isStreaming = true;
  // How often a rewrite discarded the whole retained prefix, and how many
  // characters a rewind handed back to the live tail. Both redo work with an
  // identical result, so time is the only other evidence they happened. Tests
  // read these to hold the rewind path in place.
  private retainedPrefixRebuilds = 0;
  private rewoundCharacters = 0;

  private completedCodeFences = createCompletedCodeFenceCache();

  private createBlock(content: string): IncrementalMarkdownBlock {
    const id = `${this.epoch}:${this.nextBlockIdentity}`;
    const block = {
      codeFences: this.completedCodeFences(content, id),
      content,
      id,
    } satisfies IncrementalMarkdownBlock;
    this.nextBlockIdentity += 1;
    return block;
  }

  // Streamdown itself uses positional keys inside the bounded live tail. Match
  // that behavior while preserving an exact block object when its content did
  // not change. A promoted block carries this same identity into a chunk.
  private reconcileTailBlocks(contents: string[]): IncrementalMarkdownBlock[] {
    const previous = this.tailBlocks;
    const next = contents.map((content, index) => {
      const existing = previous[index];
      if (!existing) {
        return this.createBlock(content);
      }
      if (existing.content === content) {
        return existing;
      }
      return {
        codeFences: this.completedCodeFences(content, existing.id),
        content,
        id: existing.id,
      } satisfies IncrementalMarkdownBlock;
    });
    this.tailBlocks = next;
    return next;
  }

  // Preserve every closed chunk object. The one open chunk changes only when a
  // block is committed, never for ordinary token-only tail updates.
  private syncCommittedChunks(): void {
    const previous = this.committedChunks;
    const next: IncrementalMarkdownChunk[] = [];
    for (
      let start = 0, chunkIndex = 0;
      start < this.committedBlocks.length;
      start += COMMITTED_CHUNK_BLOCKS, chunkIndex += 1
    ) {
      const blocks = this.committedBlocks.slice(
        start,
        start + COMMITTED_CHUNK_BLOCKS,
      );
      const existing = previous[chunkIndex];
      if (
        existing &&
        existing.blocks.length === blocks.length &&
        existing.blocks.every((block, index) => block === blocks[index])
      ) {
        next.push(existing);
      } else {
        next.push({
          blocks,
          id: `chunk:${blocks[0].id}`,
          startIndex: start,
        });
      }
    }
    this.committedChunks = next;
  }

  // A closing fence followed immediately by prose is no longer terminal, so
  // mdast owns only its semantic code value and drops the separator newline.
  // Carry the exact source observed while the fence was open into that newly
  // completed block; later chunk promotion then retains the same actions.
  private preserveTerminalCodeSource(
    blocks: IncrementalMarkdownBlock[],
    previous: IncrementalMarkdownTerminalCodeTail | null,
  ): IncrementalMarkdownBlock[] {
    if (!previous) return blocks;
    let changed = false;
    let blockOffset = this.committedLength;
    const next = blocks.map((block) => {
      const currentBlockOffset = blockOffset;
      blockOffset += block.content.length;
      let blockChanged = false;
      const codeFences = block.codeFences.map((fence) => {
        const samePersistentOccurrence =
          (block.id === previous.blockId &&
            fence.openingOffset === previous.openingOffset) ||
          currentBlockOffset + fence.openingOffset === previous.sourceOffset;
        if (
          !samePersistentOccurrence ||
          fence.language !== previous.language ||
          fence.source === previous.source ||
          (`${fence.source}\n` !== previous.source &&
            `${fence.source}\r\n` !== previous.source)
        ) {
          return fence;
        }
        blockChanged = true;
        return {
          ...fence,
          source: previous.source,
          terminalLineFeedIsExact: true as const,
        };
      });
      if (!blockChanged) return block;
      changed = true;
      return { ...block, codeFences };
    });
    if (changed) this.tailBlocks = next;
    return changed ? next : blocks;
  }
  private overlayPersistedFenceProvenance(
    render: IncrementalMarkdownRender,
  ): IncrementalMarkdownRender {
    const ordinals = this.persistedTrailingLfOrdinals;
    if (ordinals.length === 0) return render;

    const blocks = [
      ...render.chunks.flatMap((chunk) => chunk.blocks),
      ...render.tail,
    ];
    const markdown = blocks.map((block) => block.content).join("");
    const occurrences = getCompletedFencedCodeOccurrences(markdown);
    const byOrdinal = new Map(occurrences.map((entry) => [entry.ordinal, entry]));
    const byOpeningOffset = new Map(
      occurrences.map((entry) => [entry.openingOffset, entry]),
    );
    if (
      ordinals.some((ordinal) => {
        const occurrence = byOrdinal.get(ordinal);
        return !occurrence || !occurrence.bodyWithSeparator.endsWith("\n");
      })
    ) {
      return render;
    }

    const selectedOrdinals = new Set(ordinals);
    let blockOffset = 0;
    const targets = new Set<string>();
    for (const block of blocks) {
      for (const fence of block.codeFences) {
        const openingOffset = blockOffset + fence.openingOffset;
        const occurrence = byOpeningOffset.get(openingOffset);
        if (occurrence && selectedOrdinals.has(occurrence.ordinal)) {
          targets.add(fence.id);
        }
      }
      blockOffset += block.content.length;
    }
    if (targets.size !== ordinals.length) return render;

    let changed = false;
    const replaceBlock = (
      block: IncrementalMarkdownBlock,
    ): IncrementalMarkdownBlock => {
      let blockChanged = false;
      const codeFences = block.codeFences.map((fence) => {
        if (!targets.has(fence.id) || fence.terminalLineFeedIsExact) {
          return fence;
        }
        blockChanged = true;
        return {
          ...fence,
          source: `${fence.source}\n`,
          terminalLineFeedIsExact: true as const,
        };
      });
      if (!blockChanged) return block;
      changed = true;
      return { ...block, codeFences };
    };

    const chunks = render.chunks.map((chunk) => {
      const chunkBlocks = chunk.blocks.map(replaceBlock);
      return chunkBlocks.every((block, index) => block === chunk.blocks[index])
        ? chunk
        : { ...chunk, blocks: chunkBlocks };
    });
    const tail = render.tail.map(replaceBlock);
    if (!changed) return render;

    return { ...render, chunks, tail };
  }




  private publishCanonicalRender(
    canonicalRender: IncrementalMarkdownRender,
  ): IncrementalMarkdownRender {
    const render = this.overlayPersistedFenceProvenance(canonicalRender);
    incrementalMarkdownRenderObserver?.({
      codeFenceSourceLengths: render.tail.flatMap((block) =>
        block.codeFences.map((fence) => fence.source.length),
      ),
      isStreaming: this.isStreaming,
      sourceLength: this.source.length,
      terminalSourceLength:
        canonicalRender.terminalCodeTail?.source.length ?? null,
    });

    this.lastCanonicalRender = canonicalRender;
    this.lastRender = render;
    return render;
  }


  private render(
    markdown: string,
    blockContents: string[],
    previousCodeTail = this.lastRender?.terminalCodeTail ?? null,
  ): IncrementalMarkdownRender {
    const tail = this.preserveTerminalCodeSource(
      this.reconcileTailBlocks(blockContents),
      previousCodeTail,
    );
    const shellMarkdown = tail.at(-1)?.content ?? "";
    const shellStart = markdown.endsWith(shellMarkdown)
      ? markdown.length - shellMarkdown.length
      : -1;
    const sourceShellIsExact =
      shellStart >= 0 &&
      shellStart <= this.tail.length &&
      markdown.slice(0, shellStart) === this.tail.slice(0, shellStart);
    const sourceShellMarkdown = sourceShellIsExact
      ? this.tail.slice(shellStart)
      : shellMarkdown;
    // Fence syntax belongs to the source, not the transport status. A stopped or
    // cold malformed reply still needs the same bounded terminal-code plan; only
    // animation and action availability follow `isStreaming`.

    // Same argument as the completed-fence cache above: while the fence stays
    // open its opening offset cannot move, so the whole-tail parse that finds it
    // is recomputing a constant once per token.
    const settledOpeningOffset =
      previousCodeTail !== null &&
      !previousCodeTail.isClosed &&
      hasPrefix(sourceShellMarkdown, previousCodeTail.prefixMarkdown) &&
      sourceShellMarkdown.startsWith(
        previousCodeTail.openingLine,
        previousCodeTail.openingOffset,
      )
        ? previousCodeTail.openingOffset
        : undefined;
    const candidate = sourceShellIsExact
      ? getTerminalStreamingCodeFence(sourceShellMarkdown, settledOpeningOffset)
      : null;
    const sourceOffset = candidate
      ? this.committedLength + shellStart + candidate.openingOffset
      : -1;
    const keepsIdentity =
      candidate !== null &&
      previousCodeTail !== null &&
      previousCodeTail.sourceOffset === sourceOffset &&
      previousCodeTail.openingLine === candidate.openingLine;
    const usesTerminalCodeTail =
      candidate !== null && (!candidate.isClosed || keepsIdentity);

    let terminalCodeTail: IncrementalMarkdownTerminalCodeTail | null = null;
    const shellBlock = tail.at(-1);
    if (usesTerminalCodeTail && candidate && shellBlock) {
      const prefixMarkdown = sourceShellMarkdown.slice(
        0,
        candidate.openingOffset,
      );
      const prefixBlocks =
        keepsIdentity && previousCodeTail?.prefixMarkdown === prefixMarkdown
          ? previousCodeTail.prefixBlocks
          : parseMarkdownIntoBlocks(remend(prefixMarkdown)).map((content) =>
              this.createBlock(content),
            );
      const source =
        candidate.isClosed &&
        keepsIdentity &&
        previousCodeTail &&
        (candidate.rawSource === `${previousCodeTail.source}\n` ||
          candidate.rawSource === `${previousCodeTail.source}\r\n`)
          ? previousCodeTail.source
          : candidate.rawSource;

      terminalCodeTail = {
        blockId: shellBlock.id,
        fenceMarkdown: candidate.fenceMarkdown,
        id: keepsIdentity
          ? previousCodeTail.id
          : `terminal-code:${this.epoch}:${this.nextBlockIdentity++}`,
        isClosed: candidate.isClosed,
        language: candidate.language,
        openingLine: candidate.openingLine,
        openingOffset: candidate.openingOffset,
        prefixBlocks,
        prefixMarkdown,
        source,
        sourceOffset,
      };
    }

    const canonicalRender = {
      chunks: this.committedChunks,
      committedBlockCount: this.committedBlocks.length,
      epoch: this.epoch,
      markdown,
      shellMarkdown,
      sourceShellMarkdown,
      tail,
      terminalCodeTail,
    } satisfies IncrementalMarkdownRender;
    return this.publishCanonicalRender(canonicalRender);
  }

  // The repair of a tail sitting inside an open fence, or null to repair the
  // whole tail as before. The head is everything the repair would have to keep:
  // once the probe shows remend leaves it alone, it holds nothing that could
  // change the repair of the body, and it stays fixed until the fence closes,
  // so the probe is paid once per fence rather than once per chunk.
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
    const invalidatesRenderedBlocks =
      this.lastRender !== null ||
      this.committedBlocks.length > 0 ||
      this.tailBlocks.length > 0;
    if (invalidatesRenderedBlocks) {
      this.epoch += 1;
    }
    this.source = markdown;
    this.tail = markdown;
    this.committedBlocks = [];
    this.committedChunks = [];
    this.tailBlocks = [];
    this.committedLength = 0;
    this.commitPoints = [];
    this.context = createRetainedContext();
    this.lastCanonicalRender = null;
    this.lastRender = null;
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {

    const previousCodeTail = this.lastRender?.terminalCodeTail ?? null;
    if (!this.fullDocumentMode) {
      this.resetIncrementalState(markdown);
      this.fullDocumentMode = true;
    } else {
      this.source = markdown;
      this.tail = markdown;
    }
    const repaired = remend(markdown);
    return this.render(
      repaired,
      parseMarkdownIntoBlocks(repaired),
      previousCodeTail,
    );
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
      scanned -= this.committedBlocks[blocksBeforeLimit].content.length;
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
      const rewoundBlocks = this.committedBlocks.slice(point.blockCount);
      this.committedBlocks.length = point.blockCount;
      this.committedLength = point.length;
      this.commitPoints.length = index + 1;
      // Let blocks handed back to the mutable tail retain their identities when
      // the repaired sequence still has the same positional shape.
      this.tailBlocks = [...rewoundBlocks, ...this.tailBlocks];
      this.syncCommittedChunks();
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

  update(
    rawMarkdown: string,
    isStreaming = true,
    persistedTrailingLfOrdinals = this.persistedTrailingLfOrdinals,
  ): IncrementalMarkdownRender {
    const streamingChanged = this.isStreaming !== isStreaming;
    const provenanceChanged = this.updatePersistedFenceProvenance(
      persistedTrailingLfOrdinals,
    );
    this.isStreaming = isStreaming;
    // Every boundary this class finds is a byte offset into the text it was
    // handed, but the blocks it compares against come back from Streamdown with
    // their line endings already normalised. On a CRLF reply the two disagree
    // one block in, `findCommitBoundary` sees `"para"` where the source holds
    // `"para\r"`, nothing is ever committed, and the whole reply re-repairs and
    // re-lexes on every frame. Normalise first so both sides speak LF.
    const markdown = normalizeLineEndings(rawMarkdown);
    // Tokens arrive faster than frames, so the coalescer hands the same text to
    // several renders. Nothing about the result can differ; return the exact
    // same plan object so context consumers also do no work.
    if (
      !streamingChanged &&
      markdown === this.source &&
      this.lastRender !== null
    ) {
      if (!provenanceChanged) return this.lastRender;
      if (this.lastCanonicalRender !== null) {
        return this.publishCanonicalRender(this.lastCanonicalRender);
      }
    }

    if (this.fullDocumentMode && hasPrefix(markdown, this.source)) {
      return this.renderFullDocument(markdown);
    }

    this.updateTail(markdown);

    const repaired =
      this.repairOpenFence() ?? repairTail(this.tail, this.context);

    // Streamdown deliberately turns a repaired document containing footnotes
    // into one block so definitions can resolve references anywhere in the
    // document. Such a construct is globally scoped and cannot retain a prefix.
    if (hasFootnoteConstruct(repaired)) {
      return this.renderFullDocument(markdown);
    }

    const blocks = parseMarkdownIntoBlocks(repaired);
    const candidateCount = Math.max(0, blocks.length - ROLLBACK_BLOCKS);
    if (candidateCount === 0) {
      return this.render(repaired, blocks);
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
      return this.render(repaired, blocks);
    }

    const committedText = this.tail.slice(0, commit.length);
    const nextContext = advanceContext(
      this.context,
      commit.parity,
      committedText,
    );
    const nextTail = this.tail.slice(commit.length);
    const nextMarkdown = repairTail(nextTail, nextContext);
    // Preserve the previous terminal occurrence while every reconciled block is
    // still present. A close can promote that block in this same update, so doing
    // this after the committed/tail split loses its exact terminal line ending.
    const reconciled = this.preserveTerminalCodeSource(
      this.reconcileTailBlocks(blocks),
      this.lastRender?.terminalCodeTail ?? null,
    );
    this.committedBlocks.push(...reconciled.slice(0, commit.count));
    this.tailBlocks = reconciled.slice(commit.count);
    this.committedLength += commit.length;
    this.context = nextContext;
    this.tail = nextTail;
    this.commitPoints.push({
      blockCount: this.committedBlocks.length,
      length: this.committedLength,
      context: nextContext,
    });
    this.syncCommittedChunks();

    return this.render(nextMarkdown, parseMarkdownIntoBlocks(nextMarkdown));
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
