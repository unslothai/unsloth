// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const TAB = readFileSync(
  fileURLToPath(
    new URL("../src/features/settings/tabs/agents-tab.tsx", import.meta.url),
  ),
  "utf-8",
);

test("the default model uses its recommended thinking settings", () => {
  assert.ok(
    TAB.includes('const EXAMPLE_MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF";'),
  );
  assert.ok(TAB.includes('const EXAMPLE_MODEL_VARIANT = "UD-Q4_K_XL";'));
  const start = TAB.indexOf("const EXAMPLE_MODEL_OPTIONS");
  const options = TAB.slice(start, TAB.indexOf("].join", start));
  for (const option of [
    "--context-length 32768",
    "--temperature 1.0",
    "--top-p 0.95",
    "--top-k 20",
    "--min-p 0.0",
    "--reasoning-effort medium",
  ]) {
    assert.ok(options.includes(option), `${option} is in the default command`);
  }
  assert.ok(
    TAB.includes("modelKey(selectedModel) === modelKey(EXAMPLE_MODEL_REPO)"),
  );
  assert.ok(!options.includes("--presence-penalty"));
});

test("the model dropdown loads live trending GGUFs", () => {
  const start = TAB.indexOf('useHubModelSearch("", {');
  const request = TAB.slice(start, TAB.indexOf("});", start));
  assert.ok(request.includes('owner: "unsloth"'));
  assert.ok(request.includes('tags: ["gguf"]'));
  assert.ok(request.includes('sortBy: "trendingScore"'));
  assert.ok(request.includes('sortDirection: "desc"'));
  assert.ok(request.includes("keepUnsupportedTags: false"));
  assert.ok(TAB.includes("!isEmbeddingHubModel(model)"));
  assert.ok(TAB.includes("EMBEDDING_TAGS.has(tag.toLowerCase())"));
  assert.ok(TAB.includes("mergeModelOrder(trendingModels, models)"));
  assert.ok(TAB.includes("[...primary, ...fallback]"));
});

test("restored Hub selections remain valid while uncached", () => {
  assert.ok(TAB.includes("isHuggingFaceRepo(restored)"));
});

test("model selection matching ignores Hub repository casing", () => {
  assert.ok(TAB.includes("modelKey(model) === selectedKey"));
  assert.ok(TAB.includes("modelKey(model) === modelKey(selectedModel)"));
});
