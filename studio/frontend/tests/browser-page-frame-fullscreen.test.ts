// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "@typescript/typescript6";

import { openingTag } from "./helpers/tsx-ast.ts";

const PAGE_FRAME = "../src/features/browser/page-frame.tsx";

function iframeAttributes(): Map<string, string | null> {
  const path = fileURLToPath(new URL(PAGE_FRAME, import.meta.url));
  const source = ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
  const tags: ts.JsxOpeningLikeElement[] = [];
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "iframe") tags.push(opening);
    node.forEachChild(visit);
  };
  visit(source);
  assert.equal(tags.length, 1);
  const attributes = new Map<string, string | null>();
  for (const attribute of tags[0].attributes.properties) {
    if (!ts.isJsxAttribute(attribute)) continue;
    const value = attribute.initializer;
    attributes.set(
      attribute.name.getText(),
      value ? (ts.isStringLiteral(value) ? value.text : value.getText()) : null,
    );
  }
  return attributes;
}

test("the panel's page frame may go fullscreen without leaving its sandbox", () => {
  const attributes = iframeAttributes();
  assert.ok(attributes.has("allowFullScreen"));
  assert.equal(attributes.get("allowFullScreen"), null);
  assert.equal(attributes.get("sandbox"), "allow-scripts allow-forms");
});

test("pages in the panel may copy to the clipboard but not read it", () => {
  assert.equal(iframeAttributes().get("allow"), "clipboard-write");
});
