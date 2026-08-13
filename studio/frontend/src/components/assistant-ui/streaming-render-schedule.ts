// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

const ROLLBACK_BLOCKS = 8;
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";

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

  update(markdown: string): IncrementalMarkdownRender {
    if (markdown.startsWith(this.source)) {
      this.tail += markdown.slice(this.source.length);
    } else {
      this.tail = markdown;
      this.committedBlocks = [];
      this.hasMultilineKatexContext = false;
    }
    this.source = markdown;

    // Streamdown deliberately turns a document containing footnotes into one
    // block so definitions can resolve references anywhere in the document.
    // Such a construct is globally scoped and cannot retain a parsed prefix.
    if (markdown.includes("[^")) {
      this.tail = markdown;
      this.committedBlocks = [];
      this.hasMultilineKatexContext = false;
      return {
        markdown: remend(markdown),
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    const repaired = this.repairTail();
    const blocks = parseMarkdownIntoBlocks(repaired);
    let commitCount = Math.max(0, blocks.length - ROLLBACK_BLOCKS);
    let committedText = blocks.slice(0, commitCount).join("");

    // Remend may synthesize closing syntax at the end of an incomplete tail.
    // Never retain synthetic text. Walk backward until the candidate is an
    // exact prefix of the real source tail.
    while (commitCount > 0 && !this.tail.startsWith(committedText)) {
      commitCount -= 1;
      committedText = blocks.slice(0, commitCount).join("");
    }

    if (commitCount > 0) {
      this.committedBlocks.push(...blocks.slice(0, commitCount));
      const katex = committedText.indexOf("$$");
      if (katex >= 0 && committedText.indexOf("\n", katex) >= 0) {
        this.hasMultilineKatexContext = true;
      }
      this.tail = this.tail.slice(committedText.length);
    }

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
