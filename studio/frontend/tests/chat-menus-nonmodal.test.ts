// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The menus a long chat opens constantly must not be modal: the body scroll lock and the inherited
// `pointer-events` write cost a visible pause and a layout shift that scale with the thread.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (relative: string): string =>
  readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8");

/** Keyed by a marker on the menu's trigger. */
const NON_MODAL = [
  ["components/assistant-ui/thread.tsx", "thinkEffortAriaLabel({"],
  ["components/app-sidebar.tsx", "aria-label={options.ariaLabel}"],
  ["components/app-sidebar.tsx", 'aria-label="Chat options"'],
  ["components/app-sidebar.tsx", 'aria-label="Project options"'],
  ["components/app-sidebar.tsx", 't("shell.aria.runOptions")'],
  ["components/app-sidebar.tsx", 't("shell.accountMenu"'],
  ["features/chat/chat-page.tsx", 'aria-label="Project options"'],
  ["features/chat/chat-page.tsx", 'aria-label="Chat options"'],
] as const;

function element(source: string, at: number, tag: string): string {
  let depth = 0;
  let i = at;
  while (i < source.length) {
    const open = source.indexOf(`<${tag}`, i + 1);
    const close = source.indexOf(`</${tag}>`, i + 1);
    if (close === -1) break;
    if (open !== -1 && open < close) {
      depth++;
      i = open;
      continue;
    }
    if (depth === 0) return source.slice(at, close);
    depth--;
    i = close;
  }
  throw new Error(`unbalanced <${tag}> at ${at}`);
}

for (const [file, marker] of NON_MODAL) {
  test(`${file}: the menu at ${marker} is non-modal`, () => {
    const source = read(file);
    const marks = source.indexOf(marker);
    assert.ok(marks > 0, `marker not found; the menu moved or was renamed`);
    const start = source.lastIndexOf("<NonModalDropdownMenu", marks);
    const modal = source.lastIndexOf("<DropdownMenu>", marks);
    assert.ok(
      start > modal,
      `this trigger still sits inside a modal <DropdownMenu>; every open locks body scroll and aria-hides the document`,
    );
    assert.ok(element(source, start, "NonModalDropdownMenu").includes(marker));
  });
}

test("NonModalDropdownMenu is non-modal and guards its own dismissal", () => {
  const source = read("components/ui/non-modal-dropdown-menu.tsx");
  assert.match(source, /<DropdownMenu modal=\{false\}>/);
  assert.match(source, /<MenuDismissGuard triggerRef=\{triggerRef\} \/>/);
  // Each mount owns its ref, or a per-row menu restores focus to another row's trigger.
  assert.match(source, /const triggerRef = useRef<HTMLButtonElement>\(null\)/);
  assert.match(source, /trigger\(triggerRef\)/);
});
