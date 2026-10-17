// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown, type BlockProps } from "streamdown";
import ts from "typescript";
import { markdownBlockFallback } from "../src/components/assistant-ui/markdown-block-fallback.ts";
import { IncrementalMarkdownCache } from "../src/components/assistant-ui/streaming-render-schedule.ts";
import {
  getCodeFence,
  isHtmlFence,
  isSvgFence,
} from "../src/features/chat/artifacts/html-fences.ts";
import { readSrc } from "./helpers/kit.ts";

const parsed = ts.createSourceFile(
  "markdown-text.tsx",
  readSrc("components/assistant-ui/markdown-text.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const names = [
  "StreamdownBlockContent",
  "PlainFenceBlock",
  "PlainThinkingFence",
  "CodeBlockActions",
];
const declarations = names.map((name) => {
  const declaration = parsed.statements.find(
    (node) => ts.isFunctionDeclaration(node) && node.name?.text === name,
  );
  assert.ok(declaration, `${name} must exist`);
  return declaration.getText(parsed);
});
const { outputText } = ts.transpileModule(
  `${declarations.join("\n")}\nexports.render = StreamdownBlockContent;`,
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.React,
    },
  },
);
const context = {
  exports: {} as { render: React.ComponentType<BlockProps> },
  React,
  useContext: React.useContext,
  CodeHighlightModeContext: React.createContext("plain"),
  RenderHtmlToolPresenceContext: React.createContext(false),
  useAnimationFreeBlockProps: (props: BlockProps) => props,
  useChatRuntimeStore: () => false,
  findMermaidFence: () => ({ open: false }),
  mermaidSourceOf: () => null,
  getCodeFence,
  isHtmlFence,
  isSvgFence,
  markdownBlockFallback,
  PlainFenceShell: ({ source }: { source: string }) =>
    React.createElement("pre", null, React.createElement("code", null, source)),
  MarkdownRendererBoundary: ({ children }: { children: React.ReactNode }) =>
    children,
  MarkdownBlockFallbackView: () => null,
  Block: () => null,
  useCopiedState: () => ({ copied: false, showCopied: () => {} }),
  ACTION_PANEL_CLASS: "",
  ACTION_BUTTON_CLASS: "",
  HugeiconsIcon: () => null,
  Tick02Icon: null,
  Copy01Icon: null,
  Download01Icon: null,
};
runInNewContext(outputText, context);

function render(markdown: string, isAnimating = true) {
  const incremental = new IncrementalMarkdownCache().update(markdown);
  const blocks: BlockProps[] = [];
  const html = renderToStaticMarkup(
    React.createElement(Streamdown, {
      mode: "streaming",
      isAnimating,
      parseIncompleteMarkdown: false,
      parseMarkdownIntoBlocksFn: incremental.parseMarkdownIntoBlocks,
      BlockComponent: (props: BlockProps) => {
        blocks.push(props);
        return React.createElement(context.exports.render, props);
      },
      children: incremental.markdown,
    }),
  );
  return { html, blocks };
}

test("closed Thinking fences enable copy and download while reasoning streams", () => {
  for (const body of ["x = 1", "```js\nx = 1"]) {
    const markdown = `\`\`\`text\n${body}\n\`\`\``;
    for (const [text, active] of [
      [markdown, true],
      [`${markdown}\n\nContinuing my reasoning.`, true],
      [markdown, false],
    ] as const) {
      const { html } = render(text, active);
      const buttons = html.match(/<button\b[^>]*>/g) ?? [];
      assert.equal(
        buttons.length,
        2,
        `${JSON.stringify(text)}: both actions exist`,
      );
      assert.match(buttons[0], /title="Copy code"/);
      assert.match(buttons[1], /title="Download file"/);
      for (const button of buttons) {
        assert.doesNotMatch(button, /\bdisabled\b/);
      }
      assert.ok(
        html.includes(`<code>${body}</code>`),
        "the complete code survives",
      );
    }
  }
});

test("an open Thinking fence stays plain without completed-fence actions", () => {
  const { html, blocks } = render("```python\nx = 1");
  assert.equal(blocks[0].isIncomplete, true);
  assert.equal(getCodeFence(blocks[0].content), null);
  assert.match(html, /<code>x = 1<\/code>/);
  assert.doesNotMatch(html, /<button\b/);
});
