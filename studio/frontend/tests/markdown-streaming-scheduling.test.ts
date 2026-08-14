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
    ["plugins", "STREAMDOWN_PLUGINS"],
    ["controls", "STREAMDOWN_CONTROLS"],
    ["shikiTheme", "STREAMDOWN_SHIKI_THEME"],
  ]) {
    assert.equal(
      jsxAttribute(streamdown, attribute)?.initializer?.getText(source),
      `{${constant}}`,
      `${attribute} must retain object identity while raw tokens arrive`,
    );
  }
});

test("stream updates are paint-coalesced without a time or length throttle", () => {
  const markdownSource = source.getText();
  const hookStart = markdownSource.indexOf(
    "function useCoalescedStreamingText",
  );
  const hookEnd = markdownSource.indexOf("const MarkdownTextImpl", hookStart);
  assert.ok(hookStart >= 0 && hookEnd > hookStart);
  const hook = markdownSource.slice(hookStart, hookEnd);

  assert.ok(hook.includes("requestAnimationFrame"));
  assert.ok(!hook.includes("setTimeout"));
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
