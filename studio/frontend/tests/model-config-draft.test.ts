// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearExtraArgsEditForDraft,
  clearModelConfigDraftEdited,
  isExtraArgsHydratedForDraft,
  isModelConfigDraftEdited,
  markModelConfigDraftEdited,
  markExtraArgsHydratedForDraft,
  readExtraArgsEditForDraft,
  setExtraArgsEditForDraft,
  setExtraArgsEditLoadableForDraft,
  modelConfigDraftKey,
  patchModelConfigDraft,
  primeModelConfigDraft,
  readModelConfigDraft,
  replaceModelConfigDraft,
  retainModelConfigDraft,
  setModelConfigDraftRemember,
} from "../src/features/model-picker/model-config/model-config-draft.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";

const MODEL = "unsloth/Qwen3-8B-GGUF";
const VARIANT = "Q4_K_M";
const KEY = modelConfigDraftKey(MODEL, VARIANT);

const SEED: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: "f16",
  speculativeType: "auto",
  specDraftNMax: null,
  nParallel: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "auto",
  gpuLayers: -1,
  nCpuMoe: 0,
  selectedGpuIds: null,
};

test("two hosts share one draft for the same model identity", () => {
  primeModelConfigDraft(KEY, { config: SEED, remembered: true }, "none");
  patchModelConfigDraft(KEY, { kvCacheDtype: "q8_0" });
  const shared = readModelConfigDraft(KEY);
  assert.equal(shared?.config.kvCacheDtype, "q8_0");
  patchModelConfigDraft(KEY, { nParallel: 2 });
  assert.equal(readModelConfigDraft(KEY)?.config.nParallel, 2);
});

test("a new live signature re-seeds the draft from the resident process", () => {
  primeModelConfigDraft(KEY, { config: SEED, remembered: true }, "sig-a");
  patchModelConfigDraft(KEY, { kvCacheDtype: "q8_0" });
  const live: PerModelConfig = { ...SEED, kvCacheDtype: "q4_0", nParallel: 4 };
  primeModelConfigDraft(
    KEY,
    { config: live, remembered: true },
    "sig-b",
  );
  const reseeded = readModelConfigDraft(KEY);
  assert.equal(reseeded?.config.kvCacheDtype, "q4_0");
  assert.equal(reseeded?.config.nParallel, 4);
  assert.equal(reseeded?.appliedLiveSignature, "sig-b");
});

test("remember toggle keeps saved baseline until save", () => {
  const key = modelConfigDraftKey(MODEL, "Q8_0");
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "none");
  setModelConfigDraftRemember(key, true);
  const draft = readModelConfigDraft(key);
  assert.equal(draft?.remember, true);
  assert.equal(draft?.savedRemember, false);
});

// The sidebar keys off the checkpoint as /status spells it, the picker off the row's id.
const SAME_MODEL_SPELLINGS: [string, string | null, string, string | null][] = [
  ["C:\\models\\Qwen3.gguf", null, "C:/models/Qwen3.gguf", null],
  ["c:\\models\\Qwen3.gguf", null, "C:\\models\\Qwen3.gguf", null],
  ["/models/qwen3-dir/", "Q4_K_M", "/models/qwen3-dir", "Q4_K_M"],
  ["Unsloth/Qwen3-8B-GGUF", "Q4_K_M", "unsloth/Qwen3-8B-GGUF", "q4_k_m"],
  ["\\\\share\\models\\Qwen3.gguf", null, "//SHARE/models/qwen3.gguf", null],
];

for (const [
  leftId,
  leftVariant,
  rightId,
  rightVariant,
] of SAME_MODEL_SPELLINGS) {
  test(`one draft for ${leftId} and ${rightId}`, () => {
    assert.equal(
      modelConfigDraftKey(leftId, leftVariant),
      modelConfigDraftKey(rightId, rightVariant),
    );
  });
}

test("quants of one repo keep their own drafts", () => {
  assert.notEqual(
    modelConfigDraftKey(MODEL, "Q4_K_M"),
    modelConfigDraftKey(MODEL, "Q8_0"),
  );
});

test("a colon in the model id is not read as a quant", () => {
  // A `${id}:${variant}` join gave "ollama/qwen3:8b" and "ollama/qwen3" at 8b one draft.
  assert.notEqual(
    modelConfigDraftKey("ollama/qwen3:8b", null),
    modelConfigDraftKey("ollama/qwen3", "8b"),
  );
});

test("the draft outlives one host but not the last one", () => {
  const key = modelConfigDraftKey("unsloth/Lifetime-GGUF", VARIANT);
  const sidebar = retainModelConfigDraft(key);
  const dropdown = retainModelConfigDraft(key);
  primeModelConfigDraft(key, { config: SEED, remembered: true }, "none");
  patchModelConfigDraft(key, { nParallel: 6 });
  markExtraArgsHydratedForDraft(key);
  dropdown();
  assert.equal(readModelConfigDraft(key)?.config.nParallel, 6);
  // Or a value typed and never applied comes back as this model's settings.
  sidebar();
  assert.equal(readModelConfigDraft(key), undefined);
  assert.equal(isExtraArgsHydratedForDraft(key), false);
});

test("a release that fires twice does not drop another host's draft", () => {
  // StrictMode replays effects: a second decrement would go past the host still showing it.
  const key = modelConfigDraftKey("unsloth/Double-Release-GGUF", VARIANT);
  const dropdown = retainModelConfigDraft(key);
  const sidebar = retainModelConfigDraft(key);
  primeModelConfigDraft(
    key,
    { config: { ...SEED, nParallel: 3 }, remembered: false },
    "none",
  );
  dropdown();
  dropdown();
  assert.equal(readModelConfigDraft(key)?.config.nParallel, 3);
  sidebar();
  assert.equal(readModelConfigDraft(key), undefined);
});

test("the Extra Arguments box is shared, and an external replacement supersedes it", () => {
  const key = modelConfigDraftKey("unsloth/Extra-Args-GGUF", VARIANT);
  const release = retainModelConfigDraft(key);
  // Half-typed: parseExtraArgs still yields tokens, which the other editor would re-quote.
  setExtraArgsEditForDraft(key, {
    text: '--chat-template "a b',
    source: '--chat-template "a b"',
  });
  assert.equal(readExtraArgsEditForDraft(key)?.text, '--chat-template "a b');
  assert.equal(
    readExtraArgsEditForDraft(key)?.source,
    '--chat-template "a b"',
  );
  release();
  assert.equal(readExtraArgsEditForDraft(key), undefined);
});

test("the row's verdict travels with the edit, and a keystroke retires it", () => {
  const key = modelConfigDraftKey("unsloth/Verdict-GGUF", VARIANT);
  const release = retainModelConfigDraft(key);
  setExtraArgsEditForDraft(key, {
    text: '--chat-template "a b',
    source: '--chat-template "a b"',
  });
  // Only the row reads the raw text; the other editor judges the TOKENS, which format back
  // balanced, and would offer to load the unfinished line.
  setExtraArgsEditLoadableForDraft(key, false);
  assert.equal(readExtraArgsEditForDraft(key)?.loadable, false);
  assert.equal(readExtraArgsEditForDraft(key)?.text, '--chat-template "a b');
  // A keystroke publishes a new edit with no verdict, so a refusal cannot outlive its text.
  setExtraArgsEditForDraft(key, {
    text: '--chat-template "a b"',
    source: '--chat-template "a b"',
  });
  assert.equal(readExtraArgsEditForDraft(key)?.loadable, undefined);
  setExtraArgsEditLoadableForDraft(key, true);
  assert.equal(readExtraArgsEditForDraft(key)?.loadable, true);
  release();
  assert.equal(readExtraArgsEditForDraft(key), undefined);
});

test("a verdict without an edit is dropped rather than inventing one", () => {
  const key = modelConfigDraftKey("unsloth/No-Edit-GGUF", VARIANT);
  const release = retainModelConfigDraft(key);
  setExtraArgsEditLoadableForDraft(key, false);
  assert.equal(readExtraArgsEditForDraft(key), undefined);
  release();
});

test("a fresh editor re-reads the stored row, but not over an unsaved edit", () => {
  const key = modelConfigDraftKey("unsloth/Refresh-GGUF", VARIANT);
  const sidebar = retainModelConfigDraft(key);
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "none");
  markExtraArgsHydratedForDraft(key);

  // The sidebar never unmounts while a model is resident, so without this retirement the tab
  // never sees a save made by another origin.
  const picker = retainModelConfigDraft(key);
  assert.equal(isExtraArgsHydratedForDraft(key), false);
  markExtraArgsHydratedForDraft(key);
  picker();

  // The mark stands: that edit is already in the config a second read compares itself against.
  markModelConfigDraftEdited(key);
  const pickerAgain = retainModelConfigDraft(key);
  assert.equal(isExtraArgsHydratedForDraft(key), true);
  pickerAgain();

  clearModelConfigDraftEdited(key);
  const pickerOnceMore = retainModelConfigDraft(key);
  assert.equal(isExtraArgsHydratedForDraft(key), false);
  pickerOnceMore();
  sidebar();
  assert.equal(isModelConfigDraftEdited(key), false);
});

test("re-seeding or replacing a draft retires the edited mark", () => {
  const key = modelConfigDraftKey("unsloth/Edited-GGUF", VARIANT);
  const release = retainModelConfigDraft(key);
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "sig-a");
  markModelConfigDraftEdited(key);
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "sig-b");
  assert.equal(isModelConfigDraftEdited(key), false);
  markModelConfigDraftEdited(key);
  replaceModelConfigDraft(key, SEED, { remember: true, savedRemember: true });
  assert.equal(isModelConfigDraftEdited(key), false);
  release();
});

test("an external replacement retires the raw edit, so an A to B to A round trip cannot resurrect it", () => {
  const key = modelConfigDraftKey("unsloth/Aba-GGUF", VARIANT);
  const release = retainModelConfigDraft(key);
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "sig-a");
  // Noncanonical raw text whose tokens format back to something else, carrying a refusal.
  setExtraArgsEditForDraft(key, {
    text: '--chat-template "a b',
    source: '--chat-template "a b"',
  });
  setExtraArgsEditLoadableForDraft(key, false);

  replaceModelConfigDraft(key, SEED, { remember: true, savedRemember: true });
  // Gone, not stale: a later value formatting to the same tokens would make it current again.
  assert.equal(readExtraArgsEditForDraft(key), undefined);

  setExtraArgsEditForDraft(key, { text: "--verbose", source: "--verbose" });
  primeModelConfigDraft(key, { config: SEED, remembered: false }, "sig-b");
  assert.equal(readExtraArgsEditForDraft(key), undefined);

  setExtraArgsEditForDraft(key, { text: "--verbose", source: "--verbose" });
  assert.equal(clearExtraArgsEditForDraft(key), true);
  assert.equal(readExtraArgsEditForDraft(key), undefined);
  release();
});
