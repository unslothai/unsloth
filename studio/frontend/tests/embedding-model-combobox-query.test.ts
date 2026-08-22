// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { en } from "../src/i18n/locales/en.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

// These reach the hub and chat barrels and cannot be imported here, so this
// asserts on source, like ~50 sibling tests.
const COMBOBOX = read(
  "../src/features/settings/components/embedding-model-combobox.tsx",
);
const SECTION = read(
  "../src/features/settings/components/documents-rag-section.tsx",
);
const GENERAL_TAB = read("../src/features/settings/tabs/general-tab.tsx");
const DATA_TAB = read("../src/features/settings/tabs/data-tab.tsx");

test("only typed text searches the Hub", () => {
  // Searching for the saved model returns that model, so the list held one row
  // and the other embedding models were unreachable without clearing the field.
  assert.match(
    COMBOBOX,
    /const query = typed !== null && typed === value \? typed : ""/,
  );
  assert.match(COMBOBOX, /useDebouncedValue\(query\)/);
  assert.ok(
    !COMBOBOX.includes("useDebouncedValue(value)"),
    "the controlled value is no longer the query",
  );
});

test("only a keystroke counts as typing", () => {
  const start = COMBOBOX.indexOf("onInputValueChange");
  const handler = COMBOBOX.slice(start, COMBOBOX.indexOf("}}", start));
  // base-ui writes the input for the settings load ("none") and for a pick
  // ("item-press") too. Treating those as a search collapses the list to the
  // model already set, which is what the field did before.
  assert.ok(handler.includes('details.reason !== "input-change"'));
  assert.ok(handler.includes("setTyped(null)"));
  assert.ok(handler.includes("setTyped(next)"));
});

test("closing the list ends the search", () => {
  const start = COMBOBOX.indexOf("onOpenChange");
  const handler = COMBOBOX.slice(start, COMBOBOX.indexOf("}}", start));
  // Saving writes the same string back, so the value never changes and the
  // equality guard cannot see it. Without this, reopening the list after a
  // save is still filtered by what was typed to enter the model.
  assert.ok(handler.includes("if (!open) setTyped(null)"));
});

test("the field says what it does", () => {
  assert.match(
    SECTION,
    /placeholder=\{t\("settings.general.rag.searchPlaceholder"\)\}/,
  );
  assert.equal(
    en.settings.general.rag.searchPlaceholder,
    "Search embedding models",
  );
});

test("General and Data show the same section, not two copies of it", () => {
  assert.ok(SECTION.includes("export function DocumentsRagSection"));
  for (const [name, tab] of [
    ["general", GENERAL_TAB],
    ["data", DATA_TAB],
  ] as const) {
    assert.ok(tab.includes("<DocumentsRagSection />"), `${name} renders it`);
    // A second copy of the load and save logic would let the two tabs disagree.
    assert.ok(
      !tab.includes("loadEmbeddingModelSettings"),
      `${name} has no embedding logic of its own`,
    );
  }
});
