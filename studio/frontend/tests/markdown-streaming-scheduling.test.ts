// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const MARKDOWN_TEXT_PATH = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

const REASONING_PATH = new URL(
  "../src/components/assistant-ui/reasoning.tsx",
  import.meta.url,
);
const reasoningSource = readFileSync(REASONING_PATH, "utf8");

function findImmediateUpdateConfig(): ts.ObjectLiteralExpression | null {
  let config: ts.ObjectLiteralExpression | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "STREAMDOWN_IMMEDIATE_UPDATES" &&
      node.initializer &&
      ts.isSatisfiesExpression(node.initializer) &&
      ts.isObjectLiteralExpression(node.initializer.expression)
    ) {
      config = node.initializer.expression;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return config;
}

function findChatStreamdown(): ts.JsxOpeningLikeElement | null {
  let streamdown: ts.JsxOpeningLikeElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) === "Streamdown"
    ) {
      streamdown = node;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return streamdown;
}

function jsxAttribute(
  opening: ts.JsxOpeningLikeElement,
  name: string,
): ts.JsxAttribute | null {
  for (const attribute of opening.attributes.properties) {
    if (
      ts.isJsxAttribute(attribute) &&
      attribute.name.getText(source) === name
    ) {
      return attribute;
    }
  }
  return null;
}

test("chat streaming bypasses Streamdown's starvable transition without visible animation", () => {
  const config = findImmediateUpdateConfig();
  assert.ok(config, "STREAMDOWN_IMMEDIATE_UPDATES is missing");

  const values = new Map(
    config.properties.flatMap((property) => {
      if (
        !(
          ts.isPropertyAssignment(property) &&
          ts.isNumericLiteral(property.initializer)
        )
      ) {
        return [];
      }
      return [
        [property.name.getText(source), Number(property.initializer.text)],
      ];
    }),
  );
  assert.equal(values.get("duration"), 0);
  assert.equal(values.get("stagger"), 0);

  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "mode")?.initializer?.getText(source),
    '"streaming"',
  );
  assert.equal(
    jsxAttribute(streamdown, "animated")?.initializer?.getText(source),
    "{STREAMDOWN_IMMEDIATE_UPDATES}",
  );
});

test("token updates keep Streamdown's expensive configuration props stable", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");

  for (const [attribute, constant] of [
    ["plugins", "isStreaming ? STREAMDOWN_STREAMING_PLUGINS : STREAMDOWN_PLUGINS"],
    ["controls", "STREAMDOWN_CONTROLS"],
    ["shikiTheme", "STREAMDOWN_SHIKI_THEME"],
  ]) {
    assert.equal(
      jsxAttribute(streamdown, attribute)
        ?.initializer?.getText(source)
        .replace(/\s+/g, ""),
      `{${constant}}`.replace(/\s+/g, ""),
      `${attribute} must retain object identity while raw tokens arrive`,
    );
  }
});


test("syntax highlighting is deferred until the stream completes", () => {
  const markdownSource = source.getText();
  assert.ok(
    markdownSource.includes(
      "const STREAMDOWN_STREAMING_PLUGINS = { math, mermaid }",
    ),
  );
  assert.ok(
    markdownSource.includes("!isStreaming && shouldVirtualizeCode(block)"),
  );
});

test("stream updates are paint-aligned and rate-limited", () => {
  const markdownSource = source.getText();
  const hookStart = markdownSource.indexOf(
    "function useCoalescedStreamingText",
  );
  const hookEnd = markdownSource.indexOf("const MarkdownTextImpl", hookStart);
  assert.ok(hookStart >= 0 && hookEnd > hookStart);
  const hook = markdownSource.slice(hookStart, hookEnd);

  assert.ok(hook.includes("requestAnimationFrame"));
  assert.ok(hook.includes("setTimeout"));
  assert.ok(hook.includes("STREAM_RENDER_INTERVAL_MS"));

  // A running message can be replaced rather than appended to, as the audio
  // path does when it swaps its placeholder for the player, so holding the last
  // painted text has to be gated on the new text extending it.
  assert.ok(hook.includes("text.startsWith(displayed.text)"));
});

test("virtual rows avoid nested transformed layers and browser culling", () => {
  const markdownSource = source.getText();
  const codeStart = markdownSource.indexOf("function VirtualizedCodeLines");
  const markdownStart = markdownSource.indexOf("function VirtualizedMarkdown");
  const implementationEnd = markdownSource.indexOf(
    "const MarkdownTextImpl",
    markdownStart,
  );
  assert.ok(
    codeStart >= 0 && markdownStart > codeStart && implementationEnd > markdownStart,
  );
  const virtualized = markdownSource.slice(codeStart, implementationEnd);

  assert.ok(virtualized.includes("top: virtualLine.start - scrollMargin"));
  assert.ok(virtualized.includes("top: virtualBlock.start - scrollMargin"));
  assert.ok(!virtualized.includes("transform: `translateY"));
  assert.ok(!virtualized.includes('contentVisibility: "auto"'));
});

test("streaming reparses only the active Markdown tail", () => {
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "parseIncompleteMarkdown")?.initializer?.getText(
      source,
    ),
    "{!incrementalRender}",
  );
  assert.equal(
    jsxAttribute(streamdown, "parseMarkdownIntoBlocksFn")?.initializer?.getText(
      source,
    ),
    "{incrementalRender?.parseMarkdownIntoBlocks}",
  );
});

test("dropping retained blocks moves Streamdown's render identity", () => {
  // Streamdown compares only the Markdown string, so an edit that clears the
  // retained blocks while leaving the live tail alone has to remount instead.
  const streamdown = findChatStreamdown();
  assert.ok(streamdown, "chat <Streamdown> is missing");
  assert.equal(
    jsxAttribute(streamdown, "key")?.initializer?.getText(source),
    "{`${messageId}:${incrementalCache.renderGeneration}`}",
  );
});


test("an active code fence stays plain until the closing fence arrives", () => {
  const markdownSource = source.getText();
  const blockStart = markdownSource.indexOf("function StreamdownBlockContent");
  const blockEnd = markdownSource.indexOf("const StreamdownBlock = memo", blockStart);
  assert.ok(blockStart >= 0 && blockEnd > blockStart);
  const block = markdownSource.slice(blockStart, blockEnd);

  const plainBranch = block.indexOf(
    "(props.isIncomplete || activeStreamingBlock) && codeFence",
  );
  const completedBranch = block.indexOf("if (codeFence)");
  assert.ok(plainBranch >= 0, "the live tail needs a plain streaming branch");
  assert.ok(
    completedBranch > plainBranch,
    "the plain branch must run before syntax-highlighted code rendering",
  );
  assert.ok(block.includes("<StreamingPlainCodeBlock"));
  assert.ok(markdownSource.includes('data-streaming-code="true"'));

  assert.ok(markdownSource.includes("ActiveStreamingBlockContext.Provider"));
  assert.ok(!markdownSource.includes("value={isStreaming && isLast}"));
  assert.ok(markdownSource.includes("value={isStreaming}"));
});

test("reasoning streams stable plain-text chunks and formats Markdown on completion", () => {
  const markdownSource = source.getText();
  assert.ok(markdownSource.includes("data-streaming-plain-text"));
  assert.ok(markdownSource.includes("PLAIN_STREAM_CHUNK_SIZE"));
  assert.ok(markdownSource.includes("const blocks = plainStreaming"));
  assert.ok(reasoningSource.includes("PlainStreamingMarkdownContext.Provider"));
});


test("reasoning scroll pinning happens before paint and only user input can reattach", () => {
  const textStart = reasoningSource.indexOf("function ReasoningText");
  const textEnd = reasoningSource.indexOf("const ReasoningImpl", textStart);
  assert.ok(textStart >= 0 && textEnd > textStart);
  const implementation = reasoningSource.slice(textStart, textEnd);

  assert.ok(implementation.includes("pointerScrollIntentRef"));
  assert.ok(implementation.includes("hasUserScrollIntent && movedTowardBottom"));
  assert.ok(implementation.includes("!pointerScrollIntentRef.current"));
  assert.ok(!implementation.includes("requestAnimationFrame"));
  assert.ok(implementation.includes("[overflow-anchor:none]"));

  assert.ok(implementation.includes("max-h-64 overflow-y-auto"));
  assert.ok(!implementation.includes('streaming ? "max-h-64'));
});
