// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import ts from "typescript";
import type * as ProgressiveRowsModule from "../src/components/progressive-rows.tsx";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const { ProgressiveRows } = loadWithStubs<typeof ProgressiveRowsModule>(
  new URL("../src/components/progressive-rows.tsx", import.meta.url),
  { react: React, "react/jsx-runtime": jsxRuntime },
);

function render(count: number, pageSize: number): string {
  const items = Array.from({ length: count }, (_, i) => ({ id: `chat-${i}` }));
  return renderToStaticMarkup(
    React.createElement(
      "ul",
      null,
      React.createElement(ProgressiveRows<{ id: string }>, {
        items,
        pageSize,
        renderItem: ({ id }) => React.createElement("li", { "data-row": id }),
        end: React.createElement("li", { "data-end": "" }),
      }),
    ),
  );
}

test("a list longer than a page mounts one page, then the sentinel, and no end", () => {
  const html = render(5000, 50);
  assert.equal(html.match(/data-row=/g)?.length, 50);
  assert.match(html, /data-row="chat-49"/);
  assert.doesNotMatch(html, /data-row="chat-50"/);
  assert.doesNotMatch(html, /data-end/);
  const last = html.slice(html.lastIndexOf("<li"));
  assert.match(last, /^<li aria-hidden="true"/);
  assert.doesNotMatch(last, /data-row/);
});

test("a list that fits in a page mounts every row and then the end", () => {
  for (const count of [0, 1, 50]) {
    const html = render(count, 50);
    assert.equal(html.match(/data-row=/g)?.length ?? 0, count);
    assert.match(html, /<li data-end=""><\/li><\/ul>$/);
  }
});

function attrText(
  tag: ts.JsxOpeningLikeElement,
  name: string,
): string | undefined {
  const prop = tag.attributes.properties.find(
    (p): p is ts.JsxAttribute =>
      ts.isJsxAttribute(p) && p.name.getText() === name,
  );
  return prop?.initializer?.getText();
}

function isRecentsTail(node: ts.Node): boolean {
  if (!ts.isObjectLiteralExpression(node)) return false;
  const value = (name: string) =>
    node.properties
      .find(
        (p): p is ts.PropertyAssignment =>
          ts.isPropertyAssignment(p) && p.name.getText() === name,
      )
      ?.initializer.getText();
  return value("scope") === "SIDEBAR_TAIL_SCOPE" && value("id") === '"recents"';
}

test("Recents pages its rows and draws its end strip only after the last one", () => {
  const path = "components/app-sidebar.tsx";
  const file = ts.createSourceFile(
    path,
    readSrc(path),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let recents: ts.JsxOpeningLikeElement | undefined;
  const tails: ts.Node[] = [];
  const visit = (node: ts.Node) => {
    const tag = openingTag(node);
    if (
      tag?.tagName.getText() === "ProgressiveRows" &&
      attrText(tag, "items") === "{sortedRecentChatItems}"
    ) {
      recents = tag;
    }
    if (isRecentsTail(node)) tails.push(node);
    ts.forEachChild(node, visit);
  };
  visit(file);
  assert.ok(recents, "Recents is not a ProgressiveRows");
  assert.match(attrText(recents, "renderItem") ?? "", /RECENTS_ORDER_SCOPE/);
  // sidebar-drag.ts drops the strip after Recents' last chat, so it belongs only in end.
  const end = recents.attributes.properties.find(
    (p) => ts.isJsxAttribute(p) && p.name.getText() === "end",
  );
  assert.ok(end, "Recents passes no `end`");
  assert.equal(tails.length, 1);
  assert.ok(tails[0].pos >= end.pos && tails[0].end <= end.end);
});
