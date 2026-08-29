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
  assert.match(
    SECTION,
    /onSelect=\{\(model\) => void applyEmbeddingModel\(model, false\)\}/,
  );
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
  assert.match(SECTION, /setResolution\(resolution\)/);
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
    fn.indexOf("resolveEmbeddingModel") <
      fn.indexOf("persist(trimmed, resolution"),
    "resolve decides, then the save records what it found",
  );
  // Only the explicit override skips it, and it persists no plan.
  assert.match(
    fn,
    /if \(force\) \{[\s\S]*persist\(trimmed, null, true, reservation\)/,
  );
  const forceBranch = fn.slice(fn.indexOf("if (force) {"), fn.indexOf("let resolution"));
  assert.ok(
    !forceBranch.includes("resolveEmbeddingModel"),
    "the force path must not re-enter the resolver before saving",
  );
});

test("cross-surface save order is claimed before model resolution", () => {
  const fn = SECTION.slice(
    SECTION.indexOf("const applyEmbeddingModel"),
    SECTION.indexOf("const startDownload"),
  );
  assert.ok(
    fn.indexOf("const reservation = beginSave()") <
      fn.indexOf("resolveEmbeddingModel(trimmed"),
  );
  assert.match(fn, /isSaveCurrent\(reservation\)/);
  assert.match(fn, /persist\(trimmed, resolution, false, reservation\)/);
});

test("the repo the resolve picked is what gets stored", () => {
  // A GGUF repo need not follow a naming rule, so the loader has to be told
  // rather than left to re-derive it. The backend rides along: a model with no
  // GGUF runs on safetensors, and llama-server would have nothing to open.
  assert.match(
    SECTION,
    /plan\?\.backend === "llama" \? \(plan\.downloadRepo \?\? null\) : null/,
  );
  assert.match(SECTION, /backend: plan\?\.backend \?\? null/);
  assert.match(API, /gguf_repo: options\?\.ggufRepo \?\? null/);
  assert.match(API, /backend: options\?\.backend \?\? null/);
});

test("a missing model gets a Download button, not a popup", () => {
  // A modal for a one-click action was noise, and voice already had the shape.
  assert.ok(!SECTION.includes("AlertDialog"), "no confirmation modal");
  assert.match(SECTION, /const canDownload = Boolean\(/);
  assert.match(
    SECTION,
    /onClick=\{\(\) => resolution && void startDownload\(resolution\)\}/,
  );
});

test("the action slot offers Download or Unload, not Reset to default", () => {
  assert.ok(
    !SECTION.includes("resetEmbeddingModelSettings"),
    "reset is reachable by picking the default in the list",
  );
  assert.match(SECTION, /settings\.general\.rag\.unload/);
  // Gated on backendLoaded, not loaded, and outside the Download chain: saving a
  // new model does not release the old one, so the control that frees the
  // previous model was unreachable while Download showed.
  assert.match(SECTION, /embeddingModel\?\.backendLoaded \? \(/);
  assert.ok(
    !SECTION.includes("): embeddingModel?.loaded ? ("),
    "Unload is not an alternative to Download",
  );
});

test("the button follows a transfer started anywhere", () => {
  // Keyed off the shared manager, so a download begun from the Hub disables it too.
  assert.match(SECTION, /useDownloadManagerStore\(\(state\) =>/);
  assert.match(SECTION, /jobKeyOf\(/);
  assert.match(SECTION, /const fullSnapshotJobKey =/);
  assert.match(SECTION, /fullSnapshotDownloadState === "running"/);
});

test("download completion refreshes the resolved cache state", () => {
  assert.match(
    SECTION,
    /downloadState !== "complete" &&\s*fullSnapshotDownloadState !== "complete"/,
  );
  assert.match(
    SECTION,
    /Promise\.all\(\[\s*resolveEmbeddingModel\(savedModel,[\s\S]*refreshCachedRepos\(\)/,
  );
});

test("a saved-model change clears every previous model-scoped action", () => {
  const effect = SECTION.slice(
    SECTION.indexOf("const savedModel"),
    SECTION.indexOf("/** Persist the pick"),
  );
  assert.ok(
    effect.indexOf("setResolution(null)") <
      effect.indexOf("resolveEmbeddingModel(savedModel"),
  );
  assert.ok(
    effect.indexOf("setForceCandidate(null)") <
      effect.indexOf("resolveEmbeddingModel(savedModel"),
  );
  assert.ok(
    effect.indexOf("setSaveError(null)") <
      effect.indexOf("resolveEmbeddingModel(savedModel"),
  );
});

test("a new save cannot retain another model's force action", () => {
  const apply = SECTION.slice(
    SECTION.indexOf("const applyEmbeddingModel"),
    SECTION.indexOf("const startDownload"),
  );
  assert.match(apply, /setForceCandidate\(null\);[\s\S]*const trimmed/);
});

test("a rejected save cannot retain its download plan", () => {
  const apply = SECTION.slice(
    SECTION.indexOf("const applyEmbeddingModel"),
    SECTION.indexOf("const startDownload"),
  );
  assert.ok(
    apply.lastIndexOf("setResolution(resolution)") >
      apply.indexOf("await persist(trimmed, resolution"),
    "the accepted persistence result publishes the download plan",
  );
});

test("every download-manager non-start outcome gets feedback", () => {
  // requestStart refuses by returning an outcome rather than throwing, so a
  // non-start is silent unless this caller speaks. Every branch must, and the
  // three do not mean the same thing: "conflict" is resumable from the Hub and
  // "busy" is a sibling transfer already running, so reporting either as
  // "couldn't start the download" sends the user hunting for a fault that the
  // downloads panel is, at that moment, showing them the answer to.
  assert.match(SECTION, /if \(outcome === "started"\)[\s\S]*else \{/);
  assert.match(
    SECTION,
    /outcome === "conflict"[\s\S]*toast\.info\(t\("settings\.general\.rag\.downloadConflict"\)\)/,
  );
  assert.match(
    SECTION,
    /outcome === "busy"[\s\S]*toast\.info\(t\("settings\.general\.rag\.downloadBusy"\)\)/,
  );
  assert.match(
    SECTION,
    /toast\.error\(t\("settings\.general\.rag\.downloadFailed"\)\)/,
  );
});

test("only the embedder's own GGUF is fetched, not every quant", () => {
  assert.match(SECTION, /scopeId: scoped \? EMBEDDING_DOWNLOAD_SCOPE : null/);
  assert.match(
    SECTION,
    /variant: scoped \? scopedVariant\(EMBEDDING_DOWNLOAD_SCOPE\) : null/,
  );
  assert.match(SECTION, /inventoryKind: scoped \? "gguf" : undefined/);
});

test("the current row can be retried and arbitrary relative paths submit", () => {
  assert.match(PICKER, /onSelect\(model\);/);
  assert.ok(!PICKER.includes("if (model !== value.trim())"));
  assert.match(
    PICKER,
    /const typed = query\.trim\(\);\s*if \(typed\) \{\s*pick\(typed\);/,
  );
  assert.ok(!PICKER.includes("isDirectModelReference"));
});

test("a gated-repo token never rides in the URL", () => {
  const start = API.indexOf("export async function resolveEmbeddingModel");
  const fn = API.slice(start, API.indexOf("\n}", start));
  assert.ok(!fn.includes('params.set("hf_token"'), "not a query parameter");
  assert.match(fn, /headers: hubTokenHeader\(options\?\.hfToken\)/);
});

test("on-device rows carry the Hub's green dot", () => {
  assert.match(PICKER, /rounded-full bg-status-success/);
  // The membership test moved off the raw id and onto the resolved repo; see
  // "the on-device dot follows the resolved repo, not the displayed id".
  assert.match(PICKER, /isOnDevice\(cachedModels, item\.id\)/);
  assert.match(PICKER, /cached\.has\(repo\)/);
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

const VOICE_TAB = read("../src/features/settings/tabs/voice-tab.tsx");

test("a dictation download is reported once, not twice", () => {
  // Settings drew its own bar and Cancel beside the shared downloads panel,
  // so one transfer showed two identical progress readouts.
  assert.ok(
    !VOICE_TAB.includes("DownloadProgressBar"),
    "no second progress bar",
  );
  assert.ok(
    !VOICE_TAB.includes("sttCancelDownload"),
    "cancelling belongs to the panel",
  );
  // The status line still names the state; only the duplicate readout went.
  assert.match(VOICE_TAB, /\{sttModelStatusText\}/);
  assert.match(VOICE_TAB, /sttDownloading/);
});

test("the rate estimator went with the bar it fed", () => {
  for (const dead of [
    "downloadBytesPerSec",
    "downloadEtaSeconds",
    "computeTransferStats",
    "downloadSamplesRef",
  ]) {
    assert.ok(!VOICE_TAB.includes(dead), `${dead} is unused now`);
  }
});

test("a force save re-resolves even though the model string did not change", () => {
  // Save anyway on the model already saved leaves savedModel identical, so the
  // effect keyed on it never re-runs and nothing restores the plan the apply
  // cleared. The backend still marks an uncached model pending, so the row would
  // sit with no Download while the loader refuses to index.
  assert.match(
    SECTION,
    /if \(await persist\(trimmed, null, true, reservation\)\) \{\s*setResolveNonce\(\(n\) => n \+ 1\);/,
  );
  assert.match(SECTION, /\}, \[savedModel, hfToken, resolveNonce\]\);/);
});

test("the configured default stays reachable when the listing drops it", () => {
  // The empty query is scoped to `unsloth`, so a private, other-owner or local
  // default had no row, and "Reset to default" is gone.
  assert.match(PICKER, /rows\.push\(\{ id: fallback, sizeBytes: null \}\)/);
  assert.match(PICKER, /const fallback = defaultModel\?\.trim\(\)/);
  // A stale memo would pin the row to whatever the default was on first render.
  assert.match(PICKER, /\}, \[results, value, defaultModel\]\)/);
  // And the section still hands the default down for it to be found.
  assert.match(SECTION, /defaultModel=\{embeddingModel\?\.defaultEmbeddingModel\}/);
});

test("backend residency is re-read, not just loaded once on mount", () => {
  // A running job reaching its first encode makes a backend resident with no
  // settings mutation, and the store loads only on mount. No lifecycle event to
  // subscribe to, so this re-reads.
  assert.match(SECTION, /const RESIDENCY_POLL_MS = \d+;/);
  assert.match(SECTION, /window\.setInterval\(refresh, RESIDENCY_POLL_MS\)/);
  // A hidden tab must not poll, and must catch up the moment it is shown.
  assert.match(SECTION, /if \(document\.hidden\) return;/);
  assert.match(SECTION, /addEventListener\("visibilitychange", refresh\)/);
  assert.match(SECTION, /removeEventListener\("visibilitychange", refresh\)/);
  assert.match(SECTION, /window\.clearInterval\(timer\)/);
});

test("the on-device dot follows the resolved repo, not the displayed id", () => {
  // The inventory records what was fetched, not what was picked, so an exact-id
  // lookup left the dot off a fully downloaded model.
  assert.match(PICKER, /export function cachedRepoCandidates\(model: string\): string\[\]/);
  assert.match(PICKER, /`\$\{id\}-GGUF`/);
  assert.match(PICKER, /`sentence-transformers\/\$\{id\}`/);
  assert.match(PICKER, /if \(!id\.includes\("\/"\)\)/);
  assert.match(PICKER, /isOnDevice\(cachedModels, item\.id\)/);
  assert.ok(
    !PICKER.includes("cachedModels?.has(item.id)"),
    "the raw exact-id lookup is gone",
  );
});

test("an unquantized re-upload's GGUF companion counts as on device", () => {
  // The backend strips the quant suffix, so
  // unsloth/embeddinggemma-300m-qat-q8_0-unquantized resolves under
  // unsloth/embeddinggemma-300m-GGUF; checking only <literal>-GGUF left the dot
  // off a fully downloaded companion.
  assert.match(PICKER, /\(\?:-qat\)\?\(\?:-q\\d\+_\\d\+\[a-z\]\*\)\?-unquantized\$/i);
  assert.match(PICKER, /if \(base !== name\) candidates\.push\(`\$\{owner\}\$\{base\}-GGUF`\)/);
  // No lookbehind: this build target ships regex verbatim, so anything Safari 16
  // cannot parse would break the bundle rather than fail a test.
  assert.ok(!PICKER.includes("(?<="), "no lookbehind in shipped regex");
});
