// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { en } from "../src/i18n/locales/en.ts";

// The component reaches the hub barrel and cannot be imported here, so this
// asserts on source, like ~50 sibling tests.
const COMBOBOX = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/settings/components/embedding-model-combobox.tsx",
      import.meta.url,
    ),
  ),
  "utf-8",
);
const GENERAL_TAB = readFileSync(
  fileURLToPath(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
  ),
  "utf-8",
);

test("only typed text searches the Hub", () => {
  // Searching for the saved model returns that model, so the list held one row
  // and the other embedding models were unreachable without clearing the field.
  assert.match(COMBOBOX, /const query = typed !== null && typed === value \? typed : ""/);
  assert.match(COMBOBOX, /useDebouncedValue\(query\)/);
  assert.ok(
    !COMBOBOX.includes("useDebouncedValue(value)"),
    "the controlled value is no longer the query",
  );
});

test("a selection stops filtering by what was typed to find it", () => {
  const start = COMBOBOX.indexOf("onInputValueChange");
  const handler = COMBOBOX.slice(start, COMBOBOX.indexOf("}}", start));
  // The selecting branch runs on pick; leaving the query would reopen the list
  // still filtered by the old search.
  assert.ok(handler.includes("selectingRef.current"));
  assert.ok(handler.includes("setTyped(null)"));
  assert.ok(handler.includes("setTyped(next)"));
});

test("the field says what it does", () => {
  assert.match(
    GENERAL_TAB,
    /placeholder=\{t\("settings.general.rag.searchPlaceholder"\)\}/,
  );
  assert.equal(
    en.settings.general.rag.searchPlaceholder,
    "Search embedding models",
  );
});
