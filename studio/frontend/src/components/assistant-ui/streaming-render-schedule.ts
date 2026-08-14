// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

// How far behind the live edge a block has to be before it can be retained.
// The block list interleaves "\n\n" separators, so this is about four
// paragraphs of slack for a construct that a later line can still reinterpret.
const ROLLBACK_BLOCKS = 8;
// A marker the reply never closes leaves the tail growing with nothing to
// retain, and the scan for a boundary is then paid on top of the full repair it
// was meant to replace. Give up at a character budget, because characters are
// what that scan costs, and a transient imbalance closes far below this.
const STALLED_TAIL_CHARACTERS = 32_768;
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
const LINK_DEFINITION_RE = /\[[^\]\n]{1,200}\]:/;
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

// Remend decides these closers from marker parity over the whole document.
// A retained prefix must therefore end with neutral parity, otherwise repairing
// the remaining tail alone could add or omit a closer that a full repair would
// handle differently.
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

// Remend finds the bold marker that orders its closers with a raw
// indexOf("**") while counting pairs only outside fenced code, so a `**` in a
// fence still has to seed the bold context without reaching the fence-aware
// counter in updateAsteriskParity.
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

// Remend's own display-math scan stops one character early because it only ever
// reads a whole document, where the last character cannot open a pair. Here the
// scan runs per retained block, and every block boundary is interior to the
// document, so the last character has to be counted.
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
  private context = createRetainedContext();
  private fullDocumentMode = false;
  private lastMarkdown: string | null = null;
  private droppedRetainedBlocks = false;
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
    this.context = createRetainedContext();
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {
    this.resetIncrementalState(markdown);
    this.fullDocumentMode = true;
    return this.render(remend(markdown));
  }

  private updateTail(markdown: string): void {
    if (markdown.startsWith(this.source)) {
      this.tail += markdown.slice(this.source.length);
    } else {
      this.resetIncrementalState(markdown);
      this.fullDocumentMode = false;
    }
    this.source = markdown;
  }

  update(markdown: string): IncrementalMarkdownRender {
    // Tokens arrive faster than frames, so the coalescer hands the same text to
    // several renders. Nothing about the result can differ, and repeating the
    // work would be the whole reply again once the document path is in use.
    if (markdown === this.source && this.lastMarkdown !== null) {
      return {
        markdown: this.lastMarkdown,
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    if (this.fullDocumentMode && markdown.startsWith(this.source)) {
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
    // make that fallback sticky, and do the same once the tail has grown past
    // the budget with nothing to show for it. A temporarily unbalanced marker
    // can close in a later block, so below that keep the repaired tail live and
    // retry on the next update.
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
    this.context = nextContext;
    this.tail = nextTail;

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
