// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Long-chat menus must not be modal: scroll lock and pointer-events writes scale with the thread.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const NON_MODAL = [
  ["components/assistant-ui/thread.tsx", "thinkEffortAriaLabel({"],
  ["components/app-sidebar.tsx", "aria-label={options.ariaLabel}"],
  ["components/app-sidebar.tsx", 'aria-label="Chat options"'],
  ["components/app-sidebar.tsx", 'aria-label="Project options"'],
  ["components/app-sidebar.tsx", 't("shell.aria.runOptions")'],
  ["components/app-sidebar.tsx", 't("shell.accountMenu"'],
  ["features/chat/chat-page.tsx", 'aria-label="Project options"'],
  ["features/chat/chat-page.tsx", 'aria-label="Chat options"'],
  ["features/chat/shared-composer.tsx", "thinkEffortAriaLabel({"],
] as const;

const MENU_ROOTS = new Set(["NonModalDropdownMenu", "DropdownMenu"]);

const parse = (relative: string): ts.SourceFile =>
  ts.createSourceFile(
    relative,
    readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );

const tagOf = (node: ts.Node): string | undefined =>
  ts.isJsxElement(node)
    ? node.openingElement.tagName.getText()
    : ts.isJsxSelfClosingElement(node)
      ? node.tagName.getText()
      : undefined;

function enclosingMenuRoot(
  source: ts.SourceFile,
  position: number,
): string | undefined {
  let found: string | undefined;
  const visit = (node: ts.Node): void => {
    if (node.getStart() > position || node.getEnd() < position) return;
    const tag = tagOf(node);
    if (tag && MENU_ROOTS.has(tag)) found = tag;
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(source, visit);
  return found;
}

for (const [file, marker] of NON_MODAL) {
  test(`${file}: the menu at ${marker} is non-modal`, () => {
    const source = parse(file);
    const positions: number[] = [];
    for (let at = source.text.indexOf(marker); at !== -1; ) {
      positions.push(at);
      at = source.text.indexOf(marker, at + marker.length);
    }
    assert.ok(
      positions.length > 0,
      "marker not found; the menu moved or was renamed",
    );
    for (const at of positions) {
      const root = enclosingMenuRoot(source, at);
      assert.ok(root, `no menu root encloses ${marker}`);
      assert.equal(
        root,
        "NonModalDropdownMenu",
        `this trigger sits inside <${root}>; a modal menu locks body scroll and ` +
          "aria-hides the document on every open",
      );
    }
  });
}

test("NonModalDropdownMenu is non-modal and guards its own dismissal", () => {
  const source = parse("components/ui/non-modal-dropdown-menu.tsx");
  const text = source.text;
  assert.match(text, /<DropdownMenu[^>]*\bmodal=\{false\}/);
  assert.match(text, /<MenuDismissGuard triggerRef=\{triggerRef\} \/>/);
  assert.match(text, /const triggerRef = useRef<HTMLButtonElement>\(null\)/);
  assert.match(text, /trigger\(triggerRef\)/);
});

test("the dismiss guard is mounted only while the menu is open", () => {
  // The content outlives the close by its exit animation; an ungated guard swallows the next click.
  const source = parse("components/ui/non-modal-dropdown-menu.tsx");
  const text = source.text;
  assert.match(
    text,
    /\{open \? <MenuDismissGuard triggerRef=\{triggerRef\} \/> : null\}/,
    "the guard must be gated on the open state, not mounted for the content's lifetime",
  );
  assert.match(
    text,
    /<DropdownMenu[^>]*\bonOpenChange=\{setOpen\}/,
    "the open state must come from the menu itself",
  );
});

test("the menu content still animates out, which is why the guard is gated", () => {
  const content = readSrc("components/ui/dropdown-menu.tsx");
  assert.match(content, /data-closed:animate-out/);
});
