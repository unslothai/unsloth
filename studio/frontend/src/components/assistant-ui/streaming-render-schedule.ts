// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

const ROLLBACK_BLOCKS = 8;
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";
const FOOTNOTE_REFERENCE_RE = /\[\^[\w-]{1,200}\](?!:)/;
const FOOTNOTE_DEFINITION_RE = /\[\^[\w-]{1,200}\]:/;

type RepairParity = {
  bold: boolean;
  boldFence: boolean;
  doubleUnderscore: boolean;
  displayMathInlineCode: boolean;
  inlineCode: boolean;
  inlineMathInlineCode: boolean;
  strikethrough: boolean;
  displayMath: boolean;
  inlineMath: boolean;
};

const createRepairParity = (): RepairParity => ({
  bold: false,
  boldFence: false,
  doubleUnderscore: false,
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

// Remend decides these closers from marker parity over the whole document.
// A retained prefix must therefore end with neutral parity, otherwise repairing
// the remaining tail alone could add or omit a closer that a full repair would
// handle differently.
function updateEmphasisParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text.slice(index, index + 3) === "```") {
      parity.boldFence = !parity.boldFence;
      index += 2;
      continue;
    }
    if (!parity.boldFence && text.slice(index, index + 2) === "**") {
      parity.bold = !parity.bold;
      index += 1;
      continue;
    }
    if (!parity.boldFence && text.slice(index, index + 2) === "__") {
      parity.doubleUnderscore = !parity.doubleUnderscore;
      index += 1;
    }
  }
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
  updateInlineCodeParity(parity, text);
  updateStrikethroughParity(parity, text);
  updateDisplayMathParity(parity, text);
  updateInlineMathParity(parity, text);
}

const hasNeutralRepairParity = (parity: RepairParity): boolean =>
  Object.values(parity).every((value) => !value);

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
  private fullDocumentMode = false;

  readonly parseMarkdownIntoBlocks = (markdown: string): string[] => [
    ...this.committedBlocks,
    ...parseMarkdownIntoBlocks(markdown),
  ];

  private repairTail(): string {
    if (!this.hasMultilineKatexContext) {
      return remend(this.tail);
    }
    return remend(MULTILINE_KATEX_CONTEXT + this.tail).slice(
      MULTILINE_KATEX_CONTEXT.length,
    );
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {
    this.tail = markdown;
    this.committedBlocks = [];
    this.hasMultilineKatexContext = false;
    this.fullDocumentMode = true;
    return {
      markdown: remend(markdown),
      parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
    };
  }

  update(markdown: string): IncrementalMarkdownRender {
    if (markdown.startsWith(this.source)) {
      this.tail += markdown.slice(this.source.length);
    } else {
      this.tail = markdown;
      this.committedBlocks = [];
      this.hasMultilineKatexContext = false;
      this.fullDocumentMode = false;
    }
    this.source = markdown;

    if (this.fullDocumentMode) {
      return this.renderFullDocument(markdown);
    }

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

    // Remend may synthesize closing syntax at the end of an incomplete tail.
    // Never retain synthetic or mid-string repaired text. Scan forward once,
    // recording the latest exact boundary whose global repair parity is neutral.
    for (let index = 0; index < candidateCount; index += 1) {
      const block = blocks[index];
      if (!this.tail.startsWith(block, exactLength)) {
        break;
      }
      exactLength += block.length;
      updateRepairParity(parity, block);
      if (hasNeutralRepairParity(parity)) {
        commitCount = index + 1;
        committedLength = exactLength;
      }
    }

    // A repair before the rollback window or an unbalanced global marker can
    // pin every following block behind it. Make that fallback sticky for this
    // append-only message so later updates pay the normal full-repair cost once,
    // rather than repeatedly rescanning and walking the same failed prefix.
    if (commitCount === 0) {
      return this.renderFullDocument(markdown);
    }

    this.committedBlocks.push(...blocks.slice(0, commitCount));
    const committedText = this.tail.slice(0, committedLength);
    const katex = committedText.indexOf("$$");
    if (katex >= 0 && committedText.indexOf("\n", katex) >= 0) {
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
