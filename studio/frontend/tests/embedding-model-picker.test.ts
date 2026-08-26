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
const PICKER = read(
  "../src/features/settings/components/embedding-model-picker.tsx",
);
const SECTION = read(
  "../src/features/settings/components/documents-rag-section.tsx",
);
const API = read("../src/features/settings/api/embedding-model.ts");
const GENERAL_TAB = read("../src/features/settings/tabs/general-tab.tsx");
const DATA_TAB = read("../src/features/settings/tabs/data-tab.tsx");

test("the field says it reaches the whole Hub", () => {
  assert.equal(
    en.settings.general.rag.searchPlaceholder,
    "Search any model on HF",
  );
  assert.match(
    PICKER,
    /placeholder=\{t\("settings.general.rag.searchPlaceholder"\)\}/,
  );
});

test("an empty query lists unsloth, a typed one searches everything", () => {
  // The global top-downloads page holds no unsloth mirrors to float, so an
  // unscoped empty query buries the models this install actually ships with.
  assert.match(PICKER, /ownerScope: debouncedQuery \? "all" : "unsloth"/);
  assert.match(PICKER, /useDebouncedValue\(query\.trim\(\)\)/);
});

test("only the query searches; the saved model never becomes one", () => {
  // The old combobox searched for whatever was in the field, so opening it on a
  // saved model returned that one row and hid every other embedder.
  assert.ok(
    !PICKER.includes("useDebouncedValue(value)"),
    "the controlled value is not the query",
  );
  // It is still reachable as a row, just not as a search term.
  assert.match(PICKER, /rows\.push\(\{ id: selected, sizeBytes: null \}\)/);
});

test("picking applies straight away, with no Save button", () => {
  assert.match(SECTION, /onSelect=\{\(model\) => void applyEmbeddingModel\(model, false\)\}/);
  assert.ok(
    !SECTION.includes('t("common.save")'),
    "selection is the apply action, as it is for dictation models",
  );
  // "Save anyway" survives: a 409 is still forceable, per model.
  assert.match(SECTION, /applyEmbeddingModel\(forceCandidate, true\)/);
});

test("a model that is not on disk is offered as a real download", () => {
  // The whole point of the change: before this, saving only wrote the setting
  // and the weights arrived invisibly at the first index.
  assert.match(SECTION, /resolveEmbeddingModel\(trimmed, \{/);
  assert.match(SECTION, /setPendingDownload\(resolution\)/);
  // Same manager the Hub cards use, so progress, cancel and transport are shared.
  assert.match(SECTION, /downloadManager\.requestStart\(\{/);
  assert.match(SECTION, /kind: DOWNLOAD_KIND\.MODEL/);
});

test("the resolve runs before the save, not after it", () => {
  // Saving first meant a model with no same-owner GGUF 409'd and the user saw a
  // wall of red text instead of a download.
  const fn = SECTION.slice(
    SECTION.indexOf("const applyEmbeddingModel"),
    SECTION.indexOf("const startDownload"),
  );
  assert.ok(
    fn.indexOf("resolveEmbeddingModel") < fn.indexOf("persist(trimmed, resolution"),
    "resolve decides, then the save records what it found",
  );
  // Only the explicit override skips it.
  assert.match(fn, /if \(force\) \{\s*await persist\(trimmed, null, true\);/);
});

test("the repo the resolve picked is what gets stored", () => {
  // A conversion under another owner follows no naming rule, so the loader has
  // to be told rather than left to re-derive it.
  assert.match(SECTION, /persist\(\s*resolution\.embeddingModel,\s*resolution\.downloadRepo,/);
  assert.match(API, /gguf_repo: options\?\.ggufRepo \?\? null/);
});

test("the dialog names the repo it will download from", () => {
  assert.match(SECTION, /settings\.general\.rag\.downloadSource/);
  assert.equal(en.settings.general.rag.downloadSource, "From {repo}");
});

test("only the embedder's own GGUF is fetched, not every quant", () => {
  assert.match(SECTION, /scopeId: scoped \? EMBEDDING_DOWNLOAD_SCOPE : null/);
  assert.match(SECTION, /variant: scoped \? scopedVariant\(EMBEDDING_DOWNLOAD_SCOPE\) : null/);
});

test("a gated-repo token never rides in the URL", () => {
  const start = API.indexOf("export async function resolveEmbeddingModel");
  const fn = API.slice(start, API.indexOf("\n}", start));
  assert.ok(!fn.includes('params.set("hf_token"'), "not a query parameter");
  assert.match(fn, /headers: hubTokenHeader\(options\?\.hfToken\)/);
});

test("on-device rows carry the Hub's green dot", () => {
  assert.match(PICKER, /rounded-full bg-status-success/);
  assert.match(PICKER, /cachedModels\?\.has\(item\.id\)/);
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
