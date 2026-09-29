// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const REGISTRY_SHORTCUT = /useShortcut\("openCommandPalette"/;
const RAW_KEY_LISTENER = /addEventListener\("keydown"/;
const SETTINGS_INDEX_IMPORT =
  /\bDIALOG_SETTINGS_SEARCH_INDEX,\n[^;]*from "@\/features\/settings"/;
const LOCALIZED_SETTINGS_KEYWORDS =
  /keywords=\{DIALOG_SETTINGS_SEARCH_INDEX\[tab\]\.map\(\(key\) =>\s*t\(key\),?\s*\)\}/;
const HARDCODED_SETTINGS_KEYWORDS = /keywords:\s*\[/;
const PALETTE_SEARCH_OPENER =
  /useChatSearchStore\.getState\(\)\.open\(\{\s*opener: useCommandPaletteStore\.getState\(\)\.opener/s;
const SEARCH_STORE_OPENER = /opener: HTMLElement \| null/;
const EXPLICIT_SEARCH_OPENER = /options\?\.opener !== undefined/;
const SEARCH_CLOSE_FOCUS = /onCloseAutoFocus=\{\(event\) => \{/;
const RESTORE_SEARCH_OPENER = /opener\.focus\(\{ preventScroll: true \}\)/;

const palette = await readFile(
  new URL("../src/components/command-palette.tsx", import.meta.url),
  "utf8",
);
const chatSearchStore = await readFile(
  new URL("../src/features/chat/stores/chat-search-store.ts", import.meta.url),
  "utf8",
);
const chatSearchDialog = await readFile(
  new URL(
    "../src/features/chat/components/chat-search-dialog.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the command-palette chord comes from the rebindable shortcut registry", () => {
  assert.match(palette, REGISTRY_SHORTCUT);
  assert.doesNotMatch(palette, RAW_KEY_LISTENER);
});

test("settings commands search the localized settings index", () => {
  assert.match(palette, SETTINGS_INDEX_IMPORT);
  assert.match(palette, LOCALIZED_SETTINGS_KEYWORDS);
  assert.doesNotMatch(palette, HARDCODED_SETTINGS_KEYWORDS);
});

test("chat search receives and restores the palette opener", () => {
  assert.match(palette, PALETTE_SEARCH_OPENER);
  assert.match(chatSearchStore, SEARCH_STORE_OPENER);
  assert.match(chatSearchStore, EXPLICIT_SEARCH_OPENER);
  assert.match(chatSearchDialog, SEARCH_CLOSE_FOCUS);
  assert.match(chatSearchDialog, RESTORE_SEARCH_OPENER);
});

test("reopening during the exit animation clears the previous query", () => {
  assert.match(
    palette,
    /if \(isOpen !== wasOpen\) \{\s*setWasOpen\(isOpen\);\s*if \(isOpen\) setQuery\(""\);/,
  );
});

test("a chord that navigates behind the palette closes it", () => {
  assert.match(
    palette,
    /useRouterState\(\{ select: \(s\) => s\.location\.href \}\)/,
  );
  assert.match(
    palette,
    /useEffect\(\(\) => \{\s*useCommandPaletteStore\.getState\(\)\.close\(\);\s*\}, \[href\]\);/,
  );
});
