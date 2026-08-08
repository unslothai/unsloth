// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const PHYSICAL_P_SHORTCUT = /e\.code !== "KeyP"/;
const LOCALIZED_P_SHORTCUT = /e\.key\.toLowerCase\(\) !== "p"/;
const SETTINGS_INDEX_IMPORT =
  /import \{ SETTINGS_SEARCH_INDEX \} from "@\/features\/settings\/settings-search"/;
const LOCALIZED_SETTINGS_KEYWORDS =
  /keywords=\{SETTINGS_SEARCH_INDEX\[tab\.id\]\.map\(\(key\) => t\(key\)\)\}/;
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

test("the command-palette shortcut follows the physical P key", () => {
  assert.match(palette, PHYSICAL_P_SHORTCUT);
  assert.doesNotMatch(palette, LOCALIZED_P_SHORTCUT);
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
