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
  modelIdsMatch,
  publicModelId,
  residentModelIdMatches,
} from "../src/features/hub/lib/model-identity.ts";

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
  // A loose .gguf: the catalog row is keyed by the path, /status by the stem.
  assert.equal(
    modelIdsMatch("Qwen3-8B-Q4_K_M", "/srv/models/Qwen3-8B-Q4_K_M.gguf"),
    false,
  );
  assert.equal(
    residentModelIdMatches(
      "Qwen3-8B-Q4_K_M",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
      "/srv/models/Qwen3-8B-Q4_K_M.gguf",
    ),
    true,
  );
  // A repo in an inactive HF cache loads by snapshot path but keeps the repo id
  // as its settings identity, so the configId alias already covers it.
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
  // The Chat picker opens the same file with no variant, so the Hub row must not
  // adopt the filename-derived label or the two edit different configs.
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
