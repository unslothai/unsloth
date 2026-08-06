// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { settingsGgufVariantForRow } from "../src/features/hub/inventory/settings-identity.ts";
import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "../src/features/hub/inventory/types.ts";
import {
  isOllamaLinkPath,
  isStandaloneGgufPath,
  modelIdsMatch,
  publicModelId,
  residentModelIdMatches,
} from "../src/features/hub/lib/model-identity.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store, storage } = installLocalStorageFake();

const REPO_KEY = 'v2:["unsloth/repo-gguf","q4_k_m"]';

// The legacy import runs once on the first read, so it must be staged before the import.
store.set(
  "unsloth_model_configs",
  JSON.stringify({ [REPO_KEY]: { version: 1, maxSeqLength: 32768 } }),
);
store.set(
  "unsloth_load_settings",
  JSON.stringify({ "Unsloth/Repo-GGUF::Q4_K_M": { contextLength: 8192 } }),
);

const { listPerModelConfigs, resolveInitialConfig, savePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");
const { modelStorageKey, splitQuantSuffix } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);

function config(maxSeqLength: number, kvCacheDtype: string | null = null) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    nBatch: null,
    nUbatch: null,
    tensorParallel: false,
    chatTemplateOverride: null,
  };
}

function storedKeys(): string[] {
  return Object.keys(
    JSON.parse(storage.getItem("unsloth_model_configs") ?? "{}"),
  );
}

test("publicModelId mirrors what /status reports for a path-loaded model", () => {
  // Mirrors public_model_id in studio/backend/core/inference/model_ids.py.
  assert.equal(
    publicModelId("/srv/models/Qwen3-8B-Q4_K_M.gguf"),
    "Qwen3-8B-Q4_K_M",
  );
  assert.equal(
    publicModelId(
      "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123",
    ),
    "unsloth/Qwen3-8B-GGUF",
  );
  assert.equal(publicModelId("C:\\models\\Foo-Q4_K_M.gguf"), "Foo-Q4_K_M");
  assert.equal(publicModelId("~/models/Foo.gguf"), "Foo");
  assert.equal(publicModelId("/srv/models/repo/"), "repo");
  // A repo id and an already-clean name come back untouched.
  assert.equal(publicModelId("unsloth/Qwen3-8B-GGUF"), "unsloth/Qwen3-8B-GGUF");
  assert.equal(publicModelId("Qwen3-8B-Q4_K_M"), "Qwen3-8B-Q4_K_M");
  // "models--" alone is not the cache layout; only the snapshots sibling is.
  assert.equal(publicModelId("models--only--nosnapshots/blobs/x"), "x");
});

test("a resident path-loaded model is matched by the id /status reports", () => {
  // A loose .gguf: the row is keyed by path and the Hub records the loadable
  // identifier, so the literal pass answers.
  assert.equal(
    modelIdsMatch("Qwen3-8B-Q4_K_M", "/srv/models/Qwen3-8B-Q4_K_M.gguf"),
    false,
  );
  assert.equal(
    residentModelIdMatches(
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
    ),
    true,
  );
  // A repo in an inactive cache keeps the repo id as its settings identity.
  assert.equal(
    residentModelIdMatches(
      "unsloth/Qwen3-8B-GGUF",
      "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123",
      "unsloth/Qwen3-8B-GGUF",
    ),
    true,
  );
  // The raw identifier is still matched literally.
  assert.equal(
    residentModelIdMatches(
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      null,
    ),
    true,
  );
  // Another model is still not the loaded one.
  assert.equal(
    residentModelIdMatches(
      "Qwen3-8B-Q4_K_M",
      "/srv/models/Llama-3-8B-Q4_K_M.gguf",
      null,
    ),
    false,
  );
  assert.equal(
    residentModelIdMatches(
      "unsloth/Qwen3-8B-GGUF",
      "/mnt/old-cache/models--unsloth--Llama-3-GGUF/snapshots/abc123",
      "unsloth/Llama-3-GGUF",
    ),
    false,
  );
  assert.equal(residentModelIdMatches(null, "/srv/models/x.gguf"), false);
  assert.equal(residentModelIdMatches("Qwen3-8B-Q4_K_M"), false);
});

test("a shared filename or folder name never marks a row resident", () => {
  // Same filename in two folders collapses onto one public id, so a stem cannot say which.
  const loaded = "/srv/models/alpha/model.gguf";
  const other = "/srv/models/beta/model.gguf";
  assert.equal(publicModelId(loaded), publicModelId(other));
  assert.equal(residentModelIdMatches(publicModelId(loaded), other, other), false);
  // The loadable identifier names exactly one of them.
  assert.equal(residentModelIdMatches(loaded, loaded, loaded), true);
  assert.equal(residentModelIdMatches(loaded, other, other), false);

  // Same collapse one level up: two model directories sharing a basename.
  const loadedDir = "/srv/lmstudio/publisher-a/Llama-3-8B-GGUF";
  const otherDir = "/srv/models/publisher-b/Llama-3-8B-GGUF";
  assert.equal(publicModelId(loadedDir), publicModelId(otherDir));
  assert.equal(
    residentModelIdMatches(publicModelId(loadedDir), otherDir, otherDir),
    false,
  );

  // A cache snapshot still collapses onto its repo id, which names one model.
  assert.equal(
    residentModelIdMatches(
      "unsloth/Qwen3-8B-GGUF",
      "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123",
      null,
    ),
    true,
  );
});

test("Ollama link paths are recognised the way the resolver excludes them", () => {
  // core/inference/local_model_resolver.py refuses any path with these segments.
  assert.equal(
    isOllamaLinkPath("/home/u/.ollama/models/.studio_links/q/qwen3-Q4_K_M.gguf"),
    true,
  );
  assert.equal(
    isOllamaLinkPath("/home/u/.cache/unsloth/ollama_links/ab12/llama3.gguf"),
    true,
  );
  assert.equal(
    isOllamaLinkPath("C:\\Users\\u\\.ollama\\models\\.studio_links\\q\\a.gguf"),
    true,
  );
  // Only those exact segments, not a directory that merely contains the name.
  assert.equal(isOllamaLinkPath("/srv/studio_links_backup/a.gguf"), false);
  assert.equal(isOllamaLinkPath("/srv/models/Qwen3-8B-Q4_K_M.gguf"), false);
  assert.equal(isOllamaLinkPath("unsloth/Qwen3-8B-GGUF"), false);
  assert.equal(isOllamaLinkPath(null), false);
});

test("a standalone gguf keeps one settings identity across surfaces", () => {
  const loose = {
    kind: "local",
    path: "/srv/models/Qwen3-8B-Q4_K_M.gguf",
    // What hub/services/models/common.py emits for a single scanned file.
    formatVariant: "Q4_K_M",
  } as LocalInventoryRow;
  // The Chat picker opens the same file with no variant, so adopting the filename
  // label would leave the two editing different configs.
  assert.equal(settingsGgufVariantForRow(loose), null);

  // A GGUF directory still has a variant slot for the quant lookup to fill.
  const repoDir = {
    kind: "local",
    path: "/srv/models/Qwen3-8B-GGUF",
    formatVariant: null,
  } as LocalInventoryRow;
  assert.equal(settingsGgufVariantForRow(repoDir), null);
  const lmStudioDir = {
    kind: "local",
    path: "/srv/lmstudio/Qwen3-8B-GGUF",
    formatVariant: "Q8_0",
  } as LocalInventoryRow;
  assert.equal(settingsGgufVariantForRow(lmStudioDir), "Q8_0");

  // Cached repo rows are unaffected (cache_inventory.py never sets one).
  const cached = { kind: "cache", formatVariant: null } as CachedInventoryRow;
  assert.equal(settingsGgufVariantForRow(cached), null);
});

// The backfill matches on the folded identity, which is only unambiguous because storage
// holds one record per model. These pin that rule rather than the backfill.
test("importing the legacy load settings never doubles up a model", () => {
  // The legacy casing names the model the v2 record already holds, so the import must
  // leave it alone rather than add a second record.
  assert.deepEqual(listPerModelConfigs().length, 1);
  assert.deepEqual(storedKeys(), [REPO_KEY]);
  assert.equal(
    resolveInitialConfig("unsloth/repo-gguf", "q4_k_m").config
      .customContextLength,
    null,
  );
});

test("two spellings of one model id keep a single stored record", () => {
  store.clear();
  savePerModelConfig("Unsloth/Repo-GGUF", "Q4_K_M", config(4096));
  savePerModelConfig("unsloth/repo-gguf", "q4_k_m", config(32768, "q8_0"));

  assert.deepEqual(storedKeys(), [REPO_KEY]);
  const listed = listPerModelConfigs();
  assert.equal(listed.length, 1);
  assert.equal(listed[0]?.config.maxSeqLength, 32768);
  // What the picker applies and the only thing the backfill can see agree.
  assert.equal(
    resolveInitialConfig("Unsloth/Repo-GGUF", "Q4_K_M").config.maxSeqLength,
    32768,
  );
});

test("two spellings of one Windows path keep a single stored record", () => {
  store.clear();
  savePerModelConfig("C:\\Models\\Foo.gguf", null, config(4096));
  savePerModelConfig("c:/models/foo.gguf", null, config(32768, "q8_0"));

  assert.deepEqual(storedKeys(), ['v2:["c:/models/foo.gguf",""]']);
  assert.equal(listPerModelConfigs().length, 1);
});

test("a POSIX path is case sensitive, so its two spellings stay separate", () => {
  store.clear();
  savePerModelConfig("/models/Foo.gguf", null, config(4096));
  savePerModelConfig("/models/foo.gguf", null, config(32768, "q8_0"));

  assert.equal(storedKeys().length, 2);
  assert.equal(
    resolveInitialConfig("/models/Foo.gguf", null).config.maxSeqLength,
    4096,
  );
});

// Every answer below is the one the backend's split_quant_suffix gives. The backfill folds a
// stored key with this before comparing, so a disagreement collapses two models onto one key.
const CASES: [string, [string, string] | null][] = [
  // A known quant label, with and without the optional bpw modifier.
  ["org/Repo-GGUF:Q4_K_M", ["org/Repo-GGUF", "Q4_K_M"]],
  ["org/Repo-GGUF:IQ4_XS-3.53bpw", ["org/Repo-GGUF", "IQ4_XS-3.53bpw"]],
  ["org/Repo-GGUF:UD-Q4_K_XL", ["org/Repo-GGUF", "UD-Q4_K_XL"]],
  // A .gguf with no quant token is labelled by its stem, lowercased in storage while
  // the scanner keeps the filename's casing.
  ["/models/CustomModel.gguf:custommodel", ["/models/CustomModel.gguf", "custommodel"]],
  ["/models/CustomModel.gguf:CustomModel", ["/models/CustomModel.gguf", "CustomModel"]],
  ["C:\\models\\CustomModel.gguf:custommodel", ["C:\\models\\CustomModel.gguf", "custommodel"]],
  // A shard suffix is not part of the label.
  [
    "/models/Custom-00001-of-00003.gguf:custom",
    ["/models/Custom-00001-of-00003.gguf", "custom"],
  ],
  ["/models/Custom-00001-of-00003.gguf:custom-00001-of-00003", null],
  // An extensionless .gguf still has a label.
  ["/models/.gguf:gguf", ["/models/.gguf", "gguf"]],
  // A quant token inside the filename wins over the stem.
  ["/models/tinyllama-Q4_K_M.gguf:q4_k_m", ["/models/tinyllama-Q4_K_M.gguf", "q4_k_m"]],
  ["/models/tinyllama-Q4_K_M.gguf:tinyllama-q4_k_m", null],
  // Only the basename is labelled, never the directories above it.
  [
    "/models/dir/CustomModel.gguf:custommodel",
    ["/models/dir/CustomModel.gguf", "custommodel"],
  ],
  ["/models/dir/CustomModel.gguf:dir/custommodel", null],
  // A colon is legal in a POSIX filename: reading it as a variant folds two real files.
  ["/models/foo:Bar.gguf", null],
  ["/models/foo:bar.gguf", null],
  ["/models/llama.gguf:Bar.gguf", null],
  ["/models/llama.gguf:bar.gguf", null],
  ["/models/CustomModel.gguf:othermodel", null],
  ["/models/model.gguf:notalabel", null],
  ["/models/plain.gguf:plain:extra", null],
  // A Windows drive letter is not a separator either.
  ["C:\\models\\foo.gguf", null],
  ["C:/models/foo.gguf", null],
  // Nothing to split.
  ["org/Repo-GGUF", null],
  ["/models/foo.gguf", null],
  ["org/Repo:", null],
  [":Q4_K_M", null],
];

test("splitQuantSuffix answers exactly as the backend's split_quant_suffix", () => {
  for (const [value, expected] of CASES) {
    assert.deepEqual(splitQuantSuffix(value), expected, value);
  }
});

test("a .gguf filename carrying a colon is not folded into a variant", () => {
  // Two real, distinct files: POSIX allows a colon and is case sensitive. The variant
  // half of a key is stored lowercased, so folding these strands one file's settings.
  const upper = "/models/llama.gguf:Bar.gguf";
  const lower = "/models/llama.gguf:bar.gguf";
  assert.equal(splitQuantSuffix(upper), null);
  assert.equal(splitQuantSuffix(lower), null);
  assert.notEqual(modelStorageKey(upper, null), modelStorageKey(lower, null));
});

// Repo ids ending in .gguf are real on the Hub, an iMat repo among them, and those hold every
// quant. Reading one as a single file drops the variant, so Q4 and Q8 save under one key.
const STANDALONE_GGUF_CASES: [string, boolean][] = [
  ["/models/llama.gguf", true],
  ["/mnt/c/models/llama.gguf", true],
  ["C:\\models\\llama.gguf", true],
  ["\\\\server\\share\\llama.gguf", true],
  ["./models/llama.gguf", true],
  ["~/models/llama.gguf", true],
  // A dropped or picked file, which /status echoes back by bare name.
  ["llama.gguf", true],
  // Repo ids: one separator, no anchor. These are repos, not files.
  ["lex-au/Orpheus-3b-FT-Q8_0.gguf", false],
  ["NexesQuants/TeeZee_Kyllene-Yi-34B-v1.1-iMat.GGUF", false],
  ["Joshua65535/qwen2.5-1.5b-instruct-q4_k_m.gguf", false],
  ["unsloth/Qwen3-8B-GGUF", false],
  ["", false],
];

test("only a file on this machine counts as a standalone gguf", () => {
  for (const [modelId, expected] of STANDALONE_GGUF_CASES) {
    assert.equal(isStandaloneGgufPath(modelId), expected, modelId);
  }
  assert.equal(isStandaloneGgufPath(null), false);
  assert.equal(isStandaloneGgufPath(undefined), false);
});
