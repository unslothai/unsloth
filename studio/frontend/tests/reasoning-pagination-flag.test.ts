// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { selectReasoningMarkdownPage } from "../src/components/assistant-ui/reasoning-pagination.ts";
import { REASONING_PAGINATION_ENABLED } from "../src/components/assistant-ui/thread-feature-flags.ts";

const parse = (path: URL): ts.SourceFile =>
  ts.createSourceFile(
    path.pathname,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );

const reasoning = parse(
  new URL("../src/components/assistant-ui/reasoning.tsx", import.meta.url),
);
const markdownText = parse(
  new URL("../src/components/assistant-ui/markdown-text.tsx", import.meta.url),
);

type OpeningLike = ts.JsxOpeningElement | ts.JsxSelfClosingElement;

const openingElements = (source: ts.SourceFile): OpeningLike[] => {
  const found: OpeningLike[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) {
      found.push(node);
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return found;
};

const attributeText = (
  element: OpeningLike,
  name: string,
  source: ts.SourceFile,
): string | undefined =>
  element.attributes.properties
    .find(
      (property): property is ts.JsxAttribute =>
        ts.isJsxAttribute(property) && property.name.getText(source) === name,
    )
    ?.initializer?.getText(source);

// The nearest `left && <element/>` this element renders under, or undefined if
// it is rendered unconditionally.
const renderGuard = (
  element: OpeningLike,
  source: ts.SourceFile,
): string | undefined => {
  for (let node: ts.Node = element; node.parent; node = node.parent) {
    const parent = node.parent;
    if (
      ts.isBinaryExpression(parent) &&
      parent.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken &&
      parent.right === node
    ) {
      return parent.left.getText(source);
    }
  }
  return undefined;
};

test("reasoning pagination ships off", () => {
  assert.equal(
    REASONING_PAGINATION_ENABLED,
    false,
    "pagination costs select-all, find-in-page, print and deep links, so it ships off",
  );
});

test("the reasoning pane routes pagination through the flag", () => {
  const element = openingElements(reasoning).find(
    (node) => node.tagName.getText(reasoning) === "MarkdownText",
  );
  assert.ok(element, "reasoning.tsx must still render <MarkdownText>");

  assert.equal(
    attributeText(element, "paginateReasoning", reasoning),
    "{REASONING_PAGINATION_ENABLED}",
    "pagination must be read from the flag, not hardcoded",
  );
  // The plain-code policy is a separate, shipping change and stays on.
  assert.equal(
    attributeText(element, "codeHighlighting", reasoning),
    '"plain"',
  );
});

test("the flag off renders the whole trace with no page containers", () => {
  const trace = Array.from(
    { length: 4_000 },
    (_, index) => `- step ${index}: ${"reasoning ".repeat(8)}`,
  ).join("\n");

  for (const streaming of [false, true]) {
    const page = selectReasoningMarkdownPage(trace, {
      enabled: REASONING_PAGINATION_ENABLED,
      streaming,
    });
    assert.deepEqual(
      {
        canonicalCodeSources: page.canonicalCodeSources,
        end: page.end,
        hasEarlier: page.hasEarlier,
        hasNewer: page.hasNewer,
        start: page.start,
        wholeTrace: page.markdown === trace,
      },
      {
        canonicalCodeSources: [],
        end: trace.length,
        hasEarlier: false,
        hasNewer: false,
        start: 0,
        wholeTrace: true,
      },
      `streaming=${streaming}`,
    );
  }

  // Both page controls are the right operand of the page flags the call above
  // pins false, so the off arm emits neither of them.
  const guards = new Map(
    openingElements(markdownText)
      .map((element): [string, string | undefined] => [
        attributeText(element, "data-slot", markdownText) ?? "",
        renderGuard(element, markdownText),
      ])
      .filter(([slot]) => slot.startsWith('"reasoning-show-')),
  );
  assert.deepEqual([...guards].sort(), [
    ['"reasoning-show-earlier"', "reasoningPage.hasEarlier"],
    ['"reasoning-show-newer"', "reasoningPage.hasNewer"],
  ]);
});
