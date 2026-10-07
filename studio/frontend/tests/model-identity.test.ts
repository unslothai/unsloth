// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isOllamaLinkPath,
  isOllamaModelId,
  isStandaloneGgufPath,
  modelDisplayName,
  modelIdsMatch,
  normalizeModelIdentity,
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
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    nBatch: null,
    nUbatch: null,
    tensorParallel: false,
    disableVision: false,
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
  assert.equal(
    publicModelId(
      "C:\\Users\\u\\.cache\\huggingface\\hub\\models--unsloth--gemma-4-12B-it-qat-GGUF\\snapshots\\7102bdea",
    ),
    "unsloth/gemma-4-12B-it-qat-GGUF",
  );
  assert.equal(publicModelId("~/models/Foo.gguf"), "Foo");
  assert.equal(publicModelId("/srv/models/repo/"), "repo");
  assert.equal(publicModelId("unsloth/Qwen3-8B-GGUF"), "unsloth/Qwen3-8B-GGUF");
  assert.equal(publicModelId("Qwen3-8B-Q4_K_M"), "Qwen3-8B-Q4_K_M");
  assert.equal(publicModelId("models--only--nosnapshots/blobs/x"), "x");
});

test("a resident path-loaded model is matched by the id /status reports", () => {
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
  assert.equal(
    residentModelIdMatches(
      "unsloth/Qwen3-8B-GGUF",
      "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123",
      "unsloth/Qwen3-8B-GGUF",
    ),
    true,
  );
  const pinnedSnapshot =
    "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123";
  assert.equal(
    residentModelIdMatches(
      pinnedSnapshot,
      "unsloth/Qwen3-8B-GGUF",
      "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF",
    ),
    false,
  );
  assert.equal(
    residentModelIdMatches(
      pinnedSnapshot,
      pinnedSnapshot,
      "unsloth/Qwen3-8B-GGUF",
    ),
    true,
  );
  assert.equal(
    residentModelIdMatches(
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      null,
    ),
    true,
  );
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
  assert.equal(residentModelIdMatches(loaded, loaded, loaded), true);
  assert.equal(residentModelIdMatches(loaded, other, other), false);

  const loadedDir = "/srv/lmstudio/publisher-a/Llama-3-8B-GGUF";
  const otherDir = "/srv/models/publisher-b/Llama-3-8B-GGUF";
  assert.equal(publicModelId(loadedDir), publicModelId(otherDir));
  assert.equal(
    residentModelIdMatches(publicModelId(loadedDir), otherDir, otherDir),
    false,
  );

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
  assert.equal(isOllamaLinkPath("/srv/studio_links_backup/a.gguf"), false);
  assert.equal(isOllamaLinkPath("/srv/models/Qwen3-8B-Q4_K_M.gguf"), false);
  assert.equal(isOllamaLinkPath("unsloth/Qwen3-8B-GGUF"), false);
  assert.equal(isOllamaLinkPath(null), false);
  assert.equal(isOllamaLinkPath("ollama-manifest:%2Fh%2Fllama3"), false);
  assert.equal(isOllamaModelId("ollama-manifest:%2Fh%2Fllama3"), true);
});

// The backfill matches on the folded identity, unambiguous only because storage holds one record per model.
test("importing the legacy load settings never doubles up a model", () => {
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

test("a Windows drive root stays distinct from a drive-relative identity", () => {
  assert.equal(normalizeModelIdentity("C:\\"), "c:/");
  assert.equal(normalizeModelIdentity("C:"), "c:");
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

test("symlink alias paths keep independent remembered settings", () => {
  store.clear();
  savePerModelConfig("/models/alias-a", "Q4_K_M", config(4096));
  savePerModelConfig("/models/alias-b", "Q4_K_M", config(32768, "q8_0"));

  assert.equal(storedKeys().length, 2);
  assert.equal(
    resolveInitialConfig("/models/alias-a", "Q4_K_M").config.maxSeqLength,
    4096,
  );
  assert.equal(
    resolveInitialConfig("/models/alias-b", "Q4_K_M").config.maxSeqLength,
    32768,
  );
  assert.equal(
    resolveInitialConfig("/models/alias-b", "Q4_K_M").config.kvCacheDtype,
    "q8_0",
  );
});

// Mirrors the backend's split_quant_suffix; disagreement collapses two models onto one key.
const CASES: [string, [string, string] | null][] = [
  ["org/Repo-GGUF:Q4_K_M", ["org/Repo-GGUF", "Q4_K_M"]],
  ["org/Repo-GGUF:IQ4_XS-3.53bpw", ["org/Repo-GGUF", "IQ4_XS-3.53bpw"]],
  ["org/Repo-GGUF:UD-Q4_K_XL", ["org/Repo-GGUF", "UD-Q4_K_XL"]],
  ["/models/CustomModel.gguf:custommodel", ["/models/CustomModel.gguf", "custommodel"]],
  ["/models/CustomModel.gguf:CustomModel", ["/models/CustomModel.gguf", "CustomModel"]],
  ["C:\\models\\CustomModel.gguf:custommodel", ["C:\\models\\CustomModel.gguf", "custommodel"]],
  [
    "/models/Custom-00001-of-00003.gguf:custom",
    ["/models/Custom-00001-of-00003.gguf", "custom"],
  ],
  ["/models/Custom-00001-of-00003.gguf:custom-00001-of-00003", null],
  ["/models/.gguf:gguf", ["/models/.gguf", "gguf"]],
  ["/models/tinyllama-Q4_K_M.gguf:q4_k_m", ["/models/tinyllama-Q4_K_M.gguf", "q4_k_m"]],
  ["/models/tinyllama-Q4_K_M.gguf:tinyllama-q4_k_m", null],
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
  ["C:\\models\\foo.gguf", null],
  ["C:/models/foo.gguf", null],
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
  // POSIX allows colons and is case sensitive; the lowercased variant would fold these.
  const upper = "/models/llama.gguf:Bar.gguf";
  const lower = "/models/llama.gguf:bar.gguf";
  assert.equal(splitQuantSuffix(upper), null);
  assert.equal(splitQuantSuffix(lower), null);
  assert.notEqual(modelStorageKey(upper, null), modelStorageKey(lower, null));
});

// Hub repo ids can end in .gguf, so they must not be read as single files.
const STANDALONE_GGUF_CASES: [string, boolean][] = [
  ["/models/llama.gguf", true],
  ["/mnt/c/models/llama.gguf", true],
  ["C:\\models\\llama.gguf", true],
  ["\\\\server\\share\\llama.gguf", true],
  ["./models/llama.gguf", true],
  ["~/models/llama.gguf", true],
  ["llama.gguf", true],
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

test("normalizes relative Windows separators without changing path case", () => {
  assert.equal(
    normalizeModelIdentity(String.raw`.\models\demo`),
    normalizeModelIdentity("./models/demo"),
  );
  assert.equal(
    normalizeModelIdentity(String.raw`..\Models\Demo\\`),
    "../Models/Demo",
  );
  assert.notEqual(
    normalizeModelIdentity("./Models/Demo"),
    normalizeModelIdentity("./models/demo"),
  );
});

test("normalizes tilde path separators without changing path case", () => {
  assert.equal(
    normalizeModelIdentity(String.raw`~\Models\Demo\\`),
    "~/Models/Demo",
  );
  assert.equal(
    normalizeModelIdentity(String.raw`~user\Models\Demo`),
    "~user/Models/Demo",
  );
  assert.notEqual(
    normalizeModelIdentity(String.raw`~\Models\Demo`),
    normalizeModelIdentity(String.raw`~\models\demo`),
  );
});

test("keeps backslashes in POSIX-shaped paths as filename characters", () => {
  assert.notEqual(
    normalizeModelIdentity(String.raw`./models/foo\bar`),
    normalizeModelIdentity("./models/foo/bar"),
  );
  assert.notEqual(
    normalizeModelIdentity(String.raw`/models/foo\bar`),
    normalizeModelIdentity("/models/foo/bar"),
  );
});

test("preserves existing platform and Hub identity rules", () => {
  assert.equal(
    normalizeModelIdentity(String.raw`C:\Models\Demo\\`),
    "c:/models/demo",
  );
  assert.equal(
    normalizeModelIdentity(String.raw`C:Models\Demo\\`),
    "c:models/demo",
  );
  assert.equal(
    normalizeModelIdentity(String.raw`\Models\Demo\\`),
    "/models/demo",
  );
  assert.equal(
    normalizeModelIdentity(String.raw`\\Server\Share\Models\Demo\\`),
    "//server/share/models/demo",
  );
  assert.equal(normalizeModelIdentity("Org/Model"), "org/model");
});


test("labels a model id with the repo leaf, never the raw path", () => {
  // A Windows HF cache path has no "/", so splitting the raw id shows the home dir.
  assert.equal(
    modelDisplayName(
      String.raw`C:\Users\An\.cache\huggingface\hub\models--unsloth--DeepSeek-V4-Flash-0731-GGUF\snapshots\57326b941c4603e24d1a5e71c22520c66e086eb8`,
    ),
    "DeepSeek-V4-Flash-0731-GGUF",
  );
  assert.equal(
    modelDisplayName(
      "/home/u/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/57326b941c4603e24d1a5e71c22520c66e086eb8",
    ),
    "DeepSeek-V4-Flash-0731-GGUF",
  );
  assert.equal(
    modelDisplayName("/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf"),
    "Qwen3-30B-A3B-Q4_K_M",
  );
  assert.equal(modelDisplayName("unsloth/Qwen3-30B-A3B-GGUF"), "Qwen3-30B-A3B-GGUF");
  assert.equal(modelDisplayName("Qwen3-30B-A3B"), "Qwen3-30B-A3B");
  assert.equal(modelDisplayName(""), "");
});

test("keeps .gguf on Hub repo ids, which are not file paths", () => {
  assert.equal(
    modelDisplayName("lex-au/Orpheus-3b-FT-Q8_0.gguf"),
    "Orpheus-3b-FT-Q8_0.gguf",
  );
  assert.equal(modelDisplayName("lex-au/Orpheus-3b-FT/Q8_0.gguf"), "Q8_0");
  assert.equal(modelDisplayName("/srv/Qwen3-Q4.gguf"), "Qwen3-Q4");
  assert.equal(modelDisplayName(String.raw`C:\models\Qwen3-Q4.gguf`), "Qwen3-Q4");
});
