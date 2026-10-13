// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");

const API = read("../src/features/hub/inventory/api.ts");
const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const HUB_ROWS = read("../src/features/hub/catalog/models-catalog-rows.tsx");
const CARD = read("../src/features/hub/catalog/local-on-device-card.tsx");

test("local reveal and delete go through the path-checked hub routes", () => {
  assert.match(API, /authFetch\("\/api\/hub\/local\/reveal"/);
  assert.match(API, /authFetch\("\/api\/hub\/local\/delete", \{\s*method: "DELETE"/);
  // Mirrors DELETABLE_SOURCES in local_model_actions.py: Ollama blobs stay read-only.
  assert.match(
    API,
    /DELETABLE_LOCAL_SOURCES[^=]*= new Set\(\[\s*"models_dir",\s*"lmstudio",\s*"custom",\s*\]\)/,
  );
});

test("picker local rows reveal by path and offer delete only for deletable sources", () => {
  assert.match(PICKERS, /onReveal=\{\(\) => revealLocalModel\(m\.path\)\}/);
  assert.match(PICKERS, /DELETABLE_LOCAL_SOURCES\.has\(m\.source\)\s*\?/);
  assert.match(PICKERS, /onConfirm: \(\) => deleteLocalModel\(m\.path\)/);
});

test("hub rows and the on-device card act on non-cache rows by path", () => {
  assert.match(HUB_ROWS, /row\.source !== "hf_cache" &&\s*row\.path/);
  assert.match(HUB_ROWS, /onReveal=\{\(\) => revealLocalModel\(localActionPath\)\}/);
  assert.match(CARD, /await deleteLocalModel\(path\)/);
  assert.match(CARD, /DELETABLE_LOCAL_SOURCES\.has\(source\) && !!path/);
  assert.match(CARD, /: revealLocalModel\(path\)/);
});
