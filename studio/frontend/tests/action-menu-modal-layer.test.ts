// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The message action menu must stay non-modal.
 *
 * Radix writes `pointer-events: none` onto <body> for a modal menu. That is an
 * INHERITED property, so the write invalidates style for every element in the
 * document, and the menu re-renders the thread's tooltips on the way back out.
 * Measured on a 500-message thread at 4x CPU throttle, one open plus close:
 *
 *              modal    non-modal
 *   windows   42462ms       506ms
 *   ubuntu    31651ms       356ms
 *   macos     16505ms       231ms
 *
 * The cost is also flat in thread length afterwards (1.5x from 10 to 500
 * messages, against 24x to 32x before), which is the property worth keeping.
 *
 * `modal={false}` is one word and reads like a stray prop, so it is pinned here
 * rather than left to survive the next tidy-up.
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

/** Every `<ActionBarMorePrimitive.Root ...>` in the file, opening element only. */
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
  // ActionBarMorePrimitive.Root spreads ...rest onto Radix's DropdownMenu.Root,
  // which is the only reason a prop it does not name has any effect. If the
  // pinned version stops doing that, modal={false} silently becomes a no-op and
  // the test above keeps passing.
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
