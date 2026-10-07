// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The message action menu must stay non-modal: Radix sets inherited `pointer-events: none` on
 * <body> for modal menus, which restyled the whole document (42s vs 0.5s on a 500-message thread).
 * `modal={false}` looks like a stray prop, so it is pinned here.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const THREAD = new URL(
  "../src/components/assistant-ui/thread.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  "thread.tsx",
  readFileSync(THREAD, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

const menuRoots = (): ts.JsxOpeningLikeElement[] => {
  const found: ts.JsxOpeningLikeElement[] = [];
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText() === "ActionBarMorePrimitive.Root"
    ) {
      found.push(node);
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(source, visit);
  return found;
};

test("every message action menu is non-modal", () => {
  const roots = menuRoots();
  assert.ok(roots.length > 0, "no ActionBarMorePrimitive.Root in thread.tsx");

  for (const root of roots) {
    const modal = root.attributes.properties.find(
      (p): p is ts.JsxAttribute =>
        ts.isJsxAttribute(p) && p.name.getText() === "modal",
    );
    assert.ok(
      modal,
      "ActionBarMorePrimitive.Root has no modal prop; it defaults to modal, " +
        "which puts the whole document on the modal layer on every open",
    );
    const value = modal.initializer;
    assert.ok(
      value &&
        ts.isJsxExpression(value) &&
        value.expression?.kind === ts.SyntaxKind.FalseKeyword,
      "modal must be exactly {false}",
    );
  }
});

test("the prop reaches Radix rather than being swallowed by the wrapper", () => {
  // ActionBarMorePrimitive.Root only honors modal because it spreads ...rest onto DropdownMenu.Root;
  // if that stops, modal={false} silently becomes a no-op.
  const wrapper = new URL(
    "../node_modules/@assistant-ui/react/dist/primitives/actionBarMore/ActionBarMoreRoot.js",
    import.meta.url,
  );
  const text = readFileSync(wrapper, "utf8");
  assert.match(
    text,
    /DropdownMenuPrimitive\.Root,\s*\{[^}]*\.\.\.rest/,
    "ActionBarMorePrimitive.Root no longer forwards unknown props to Radix",
  );
  assert.doesNotMatch(
    text,
    /\bmodal\b/,
    "the wrapper now names modal itself; check it still forwards false",
  );
});
