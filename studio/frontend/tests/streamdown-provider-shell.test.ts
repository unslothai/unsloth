// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement, Fragment, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import remend from "remend";
import {
  Block,
  type BlockProps,
  Streamdown,
  parseMarkdownIntoBlocks,
} from "streamdown";

const renderMonolithic = (markdown: string): string =>
  renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        controls: { code: true },
        isAnimating: true,
        mode: "streaming",
      },
      markdown,
    ),
  );

function renderThroughProviderShell(markdown: string): {
  html: string;
  shellIsIncomplete: boolean;
} {
  const repaired = remend(markdown);
  const blocks = parseMarkdownIntoBlocks(repaired);
  const shellMarkdown = blocks.at(-1) ?? "";
  let shellIsIncomplete = false;

  const ShellBlock = (shellProps: BlockProps): ReactNode => {
    shellIsIncomplete = shellProps.isIncomplete;
    return createElement(
      Fragment,
      null,
      ...blocks.map((content, index) =>
        createElement(Block, {
          ...shellProps,
          content,
          index,
          isIncomplete:
            index === blocks.length - 1 && shellProps.isIncomplete,
          key: index,
        }),
      ),
    );
  };

  const html = renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        BlockComponent: ShellBlock,
        controls: { code: true },
        isAnimating: true,
        mode: "streaming",
        parseIncompleteMarkdown: false,
        parseMarkdownIntoBlocksFn: (content) => [content],
      },
      shellMarkdown,
    ),
  );
  return { html, shellIsIncomplete };
}

test("one custom block can fan out under Streamdown's provider shell without changing layout", () => {
  for (const markdown of [
    "# Heading\n\nParagraph with **bold** and a [link](https://example.com).",
    "> quoted\n\n- one\n- two\n\n| a | b |\n| - | - |\n| 1 | 2 |",
    "Before\n\n```ts\nconst answer = 42;\n```\n\nAfter",
  ]) {
    assert.equal(
      renderThroughProviderShell(markdown).html,
      renderMonolithic(markdown),
      JSON.stringify(markdown),
    );
  }
});

test("the provider shell receives and preserves incomplete-fence state", () => {
  const markdown = "Before\n\n```ts\nconst answer = 42;";
  const monolithic = renderMonolithic(markdown);
  const shell = renderThroughProviderShell(markdown);

  assert.equal(shell.shellIsIncomplete, true);
  assert.match(shell.html, /data-incomplete="true"/);
  assert.equal(shell.html, monolithic);
});
