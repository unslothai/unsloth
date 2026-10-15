// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * No descendant-argument `:has()` on an ancestor of the thread: Chromium re-walks the whole
 * thread on every insertion (measured as the entire per-append cost). Child form `has-[>...]`
 * is cheap. Source test only; other ancestors are guarded by the perf ladder.
 */

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

import { readText } from "./helpers/kit.ts";

const parse = (rel: string): ts.SourceFile =>
  ts.createSourceFile(rel, readText(rel), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

const stringLiterals = (source: ts.SourceFile): string[] => {
  const out: string[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      out.push(node.text);
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return out;
};

/*
 * has-[>[x]] is child (cheap); has-[[x]], has-data-*, has-aria-*, group-has-* are descendant.
 * Match on the class token so `has-[>...]` is not misread by a substring test.
 */
const descendantHasUtilities = (className: string): string[] =>
  className
    .split(/\s+/)
    .filter((token) => {
      const variant = token.startsWith("group-has-") ? token.slice("group-".length) : token;
      if (!variant.startsWith("has-")) return false;
      if (variant.startsWith("has-[>")) return false;
      return true;
    });

test("the sidebar wrapper, an ancestor of the whole app, has no descendant-argument :has()", () => {
  const source = parse("../src/components/ui/sidebar.tsx");
  const wrappers = stringLiterals(source).filter((s) => s.includes("group/sidebar-wrapper"));
  assert.equal(
    wrappers.length,
    1,
    "expected exactly one className carrying group/sidebar-wrapper; if this element was "
      + "restructured the assertion below is no longer looking at the ancestor of the thread",
  );
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `sidebar-wrapper className carries a descendant-argument :has(): ${wrappers[0]}`,
  );
  // The rule must remain in child form; dropping it would change the inset sidebar.
  assert.ok(
    wrappers[0].includes("has-[>[data-variant=inset]]:bg-sidebar"),
    `sidebar-wrapper lost the inset background rule: ${wrappers[0]}`,
  );
});

test("the chat wrapper that contains the thread has no descendant-argument :has()", () => {
  const source = parse("../src/features/chat/chat-page.tsx");
  const wrappers: string[] = [];
  const walk = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag?.attributes.properties.some((attr) =>
      ts.isJsxAttribute(attr) && attr.name.getText() === "data-chat-content-root",
    )) {
      for (const attr of tag.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.getText() === "className") {
          assert.ok(attr.initializer && ts.isStringLiteral(attr.initializer));
          wrappers.push(attr.initializer.text);
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  assert.equal(wrappers.length, 1, "expected exactly one chat content root");
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `the chat wrapper carries a descendant-argument :has(): ${wrappers[0]}`,
  );
});

test("index.css has no descendant-argument :has() on the panels that hold the thread", () => {
  const css = readText("../src/index.css");
  const offenders =
    css.match(/(?:#chat-thread|\.chat-artifact-split|\.chat-thread-pane)[^{},]*:has\((?!\s*>)/g) ?? [];
  assert.deepEqual(offenders, [], `descendant-argument :has() on a thread ancestor: ${offenders.join(" | ")}`);
});
