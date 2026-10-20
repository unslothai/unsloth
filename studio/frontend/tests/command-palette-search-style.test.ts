// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The command palette is drawn like chat search: same surface, header row and pill rows.

const PALETTE = readSrc("components/command-palette.tsx");
const SEARCH = readSrc("features/chat/components/chat-search-dialog.tsx");

function dialogClasses(source: string): string[] {
  const match = /<CommandDialog[\s\S]*?className="([^"]+)"\s*overlayClassName="([^"]+)"/.exec(source);
  assert.ok(match);
  return [match[1], match[2]];
}

test("the palette dialog uses the chat search surface", () => {
  assert.deepEqual(dialogClasses(PALETTE), dialogClasses(SEARCH));
  assert.match(PALETTE, /<Command className="rounded-3xl p-0">/);
});

test("the palette header is chat search's search, input and close row", () => {
  const header = 'className="flex items-center gap-3 border-b border-border/40 px-4 py-3"';
  assert.ok(SEARCH.includes(header));
  assert.ok(PALETTE.includes(header));
  assert.doesNotMatch(PALETTE, /<CommandInput\b/);
  assert.match(PALETTE, /<CommandPrimitive\.Input\s+placeholder=\{t\("shell\.commandPalette\.placeholder"\)\}/);
  assert.match(PALETTE, /icon=\{Cancel01Icon\}/);
  assert.match(PALETTE, /aria-label=\{t\("common\.close"\)\}/);
});

test("Enter on the close button closes instead of running the selected row", () => {
  // The button sits inside cmdk, whose root handles Enter by selecting the highlighted row.
  const button = /<button\s+type="button"\s+onClick=\{close\}([\s\S]*?)aria-label=/.exec(PALETTE);
  assert.ok(button);
  assert.match(button[1], /onKeyDown=\{\(e\) => \{\s*if \(e\.key === "Enter"\) e\.stopPropagation\(\);\s*\}\}/);
});

test("palette rows are chat search's pill rows", () => {
  assert.match(PALETTE, /const ROW_CLASS =\s*"gap-3 rounded-full px-3 py-2\.5 text-ui-13 font-medium data-selected:bg-muted/);
  assert.match(SEARCH, /rounded-full px-3 py-2\.5 text-sm outline-hidden data-selected:bg-muted/);
  // Every row goes through PaletteItem.
  assert.doesNotMatch(PALETTE.split("function PaletteContent()")[1], /<CommandItem\b/);
});

test("the list keeps chat search's fixed height and scrollbar", () => {
  assert.match(
    PALETTE,
    /<CommandList className="cmd-native-scrollbar hover-scrollbar h-\[calc\(420px\*var\(--ui-space-scale,1\)\)\] max-h-\[60dvh\] p-1">/,
  );
});
