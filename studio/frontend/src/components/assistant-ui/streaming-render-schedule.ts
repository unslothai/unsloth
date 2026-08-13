// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

const ROLLBACK_BLOCKS = 8;
// Balanced marker prefixes preserve the whole-document facts that remend uses
// to decide how an incomplete tail should close, without changing parity.
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";
const BOLD_CONTEXT = "**x**\n\n";
const SINGLE_ASTERISK_CONTEXT = "*x*\n\n";
const SINGLE_UNDERSCORE_CONTEXT = "_x_\n\n";
const INLINE_CODE_ASTERISK_CONTEXT = "`a *b`\n\n";
const INLINE_CODE_UNDERSCORE_CONTEXT = "`a _b`\n\n";
const FOOTNOTE_REFERENCE_RE = /\[\^[\w-]{1,200}\](?!:)/;
const FOOTNOTE_DEFINITION_RE = /\[\^[\w-]{1,200}\]:/;
const WORD_CHARACTER_RE = /[\p{L}\p{N}_]/u;
const HTML_TAG_START_RE = /[a-zA-Z/]/;

type RepairParity = {
  bold: boolean;
  boldCandidate: boolean;
  boldFence: boolean;
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
  if (text[index + 1] === "*") {
    parity.boldCandidate = true;
    parity.firstBoldOrSingleUnderscore ??= "bold";
    parity.bold = !parity.bold;
    return index + 1;
  }
  if (countsAsSingleAsterisk(parity, text, index)) {
    parity.singleAsterisk = !parity.singleAsterisk;
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

function updateEmphasisParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    index = updateEmphasisMathParity(parity, text, index);
    if (text.slice(index, index + 3) === "```") {
      parity.boldFence = !parity.boldFence;
      index += 2;
      continue;
    }
    if (parity.boldFence) {
      continue;
    }
    if (text[index] === "`" && (index === 0 || text[index - 1] !== "\\")) {
      parity.emphasisInlineCode = !parity.emphasisInlineCode;
      continue;
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

function updateDisplayMathParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length - 1; index += 1) {
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
  updateEmphasisParity(parity, text);
  updateTripleAsteriskParity(parity, text);
  updateInlineCodeParity(parity, text);
  updateStrikethroughParity(parity, text);
  updateDisplayMathParity(parity, text);
  updateInlineMathParity(parity, text);
}

const hasNeutralRepairParity = (parity: RepairParity): boolean =>
  ![
    parity.bold,
    parity.boldFence,
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

// Streamdown normally repairs and lexes the entire growing reply on every
// update. Retain blocks that are safely behind a rollback window and give
// Streamdown only the active tail. The parser callback puts the retained blocks
// back into its block list, so output and React keys stay identical.
export class IncrementalMarkdownCache {
  private source = "";
  private tail = "";
  private committedBlocks: string[] = [];
  private hasMultilineKatexContext = false;
  private hasBoldContext = false;
  private hasSingleAsteriskContext = false;
  private hasSingleUnderscoreContext = false;
  private firstSingleAsteriskContext: "inlineCode" | "normal" | null = null;
  private firstSingleUnderscoreContext: "inlineCode" | "normal" | null = null;
  private firstBoldOrSingleUnderscoreContext:
    | "bold"
    | "singleUnderscore"
    | null = null;
  private fullDocumentMode = false;

  readonly parseMarkdownIntoBlocks = (markdown: string): string[] => [
    ...this.committedBlocks,
    ...parseMarkdownIntoBlocks(markdown),
  ];

  private getSingleUnderscoreContext(): string {
    if (!this.hasSingleUnderscoreContext) {
      return "";
    }
    return this.firstSingleUnderscoreContext === "inlineCode"
      ? INLINE_CODE_UNDERSCORE_CONTEXT
      : SINGLE_UNDERSCORE_CONTEXT;
  }

  private getSingleAsteriskContext(): string {
    if (!this.hasSingleAsteriskContext) {
      return "";
    }
    return this.firstSingleAsteriskContext === "inlineCode"
      ? INLINE_CODE_ASTERISK_CONTEXT
      : SINGLE_ASTERISK_CONTEXT;
  }

  private getEmphasisContext(): string {
    const bold = this.hasBoldContext ? BOLD_CONTEXT : "";
    const underscore = this.getSingleUnderscoreContext();
    return this.firstBoldOrSingleUnderscoreContext === "singleUnderscore"
      ? underscore + bold
      : bold + underscore;
  }

  private repairTail(): string {
    const context =
      this.getEmphasisContext() +
      this.getSingleAsteriskContext() +
      (this.hasMultilineKatexContext ? MULTILINE_KATEX_CONTEXT : "");
    if (!context) {
      return remend(this.tail);
    }
    return remend(context + this.tail).slice(context.length);
  }

  private resetIncrementalState(markdown: string): void {
    this.tail = markdown;
    this.committedBlocks = [];
    this.hasMultilineKatexContext = false;
    this.hasBoldContext = false;
    this.hasSingleAsteriskContext = false;
    this.hasSingleUnderscoreContext = false;
    this.firstSingleAsteriskContext = null;
    this.firstSingleUnderscoreContext = null;
    this.firstBoldOrSingleUnderscoreContext = null;
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {
    this.resetIncrementalState(markdown);
    this.fullDocumentMode = true;
    return {
      markdown: remend(markdown),
      parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
    };
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
    if (this.fullDocumentMode && markdown.startsWith(this.source)) {
      this.source = markdown;
      return this.renderFullDocument(markdown);
    }

    this.updateTail(markdown);

    const repaired = this.repairTail();

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
      return {
        markdown: repaired,
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    const parity = createRepairParity();
    let exactLength = 0;
    let commitCount = 0;
    let committedLength = 0;
    let committedHasBoldCandidate = false;
    let committedHasSingleAsteriskCandidate = false;
    let committedHasSingleUnderscoreCandidate = false;
    let committedFirstSingleAsterisk: "inlineCode" | "normal" | null = null;
    let committedFirstSingleUnderscore: "inlineCode" | "normal" | null = null;
    let committedFirstBoldOrSingleUnderscore:
      | "bold"
      | "singleUnderscore"
      | null = null;
    let repairBroke = false;

    // Remend may synthesize closing syntax at the end of an incomplete tail.
    // Never retain synthetic or mid-string repaired text. Scan forward once,
    // recording the latest exact boundary whose global repair parity is neutral.
    for (let index = 0; index < candidateCount; index += 1) {
      const block = blocks[index];
      if (!this.tail.startsWith(block, exactLength)) {
        repairBroke = true;
        break;
      }
      exactLength += block.length;
      updateRepairParity(parity, block);
      if (hasNeutralRepairParity(parity)) {
        commitCount = index + 1;
        committedLength = exactLength;
        committedHasBoldCandidate = parity.boldCandidate;
        committedHasSingleAsteriskCandidate = parity.singleAsteriskCandidate;
        committedHasSingleUnderscoreCandidate =
          parity.singleUnderscoreCandidate;
        committedFirstSingleAsterisk = parity.firstSingleAsteriskCandidate;
        committedFirstSingleUnderscore = parity.firstSingleUnderscoreCandidate;
        committedFirstBoldOrSingleUnderscore =
          parity.firstBoldOrSingleUnderscore;
      }
    }

    // A mid-string repair can never become a raw prefix on a later append, so
    // make that fallback sticky. A temporarily unbalanced marker can close in a
    // later block, so keep its repaired tail live and retry on the next update.
    if (commitCount === 0) {
      if (repairBroke) {
        return this.renderFullDocument(markdown);
      }
      return {
        markdown: repaired,
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    this.committedBlocks.push(...blocks.slice(0, commitCount));
    this.hasBoldContext ||= committedHasBoldCandidate;
    this.hasSingleAsteriskContext ||= committedHasSingleAsteriskCandidate;
    this.hasSingleUnderscoreContext ||= committedHasSingleUnderscoreCandidate;
    this.firstSingleAsteriskContext ??= committedFirstSingleAsterisk;
    this.firstSingleUnderscoreContext ??= committedFirstSingleUnderscore;
    this.firstBoldOrSingleUnderscoreContext ??=
      committedFirstBoldOrSingleUnderscore;
    const committedText = this.tail.slice(0, committedLength);
    if (committedText.includes("$$")) {
      this.hasMultilineKatexContext = true;
    }
    this.tail = this.tail.slice(committedLength);

    return {
      markdown: this.repairTail(),
      parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
    };
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
