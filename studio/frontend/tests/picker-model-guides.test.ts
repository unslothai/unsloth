// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The table is ordered, so a family with a more specific sibling must not fall through.

import assert from "node:assert/strict";
import test from "node:test";

import { modelGuide } from "../src/features/model-picker/components/model-selector/model-guides.ts";

test("a repo resolves to its own family's guide", () => {
  assert.equal(
    modelGuide("unsloth/Qwen3.8-27B-GGUF")?.url,
    "https://unsloth.ai/docs/models/qwen3.8",
  );
  assert.equal(
    modelGuide("unsloth/gpt-oss-20b-GGUF")?.url,
    "https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune",
  );
  assert.equal(
    modelGuide("unsloth/gemma-4-E2B-it-GGUF")?.url,
    "https://unsloth.ai/docs/models/gemma-4",
  );
});

// Qwen is the family where a loose match would be wrong most often.
test("a more specific family wins over the general one", () => {
  assert.equal(modelGuide("unsloth/Qwen3.8-Flash-Next")?.title, "Qwen3.8-Flash-Next");
  assert.equal(modelGuide("unsloth/Qwen3-Coder-Next-GGUF")?.title, "Qwen3-Coder-Next");
  assert.equal(modelGuide("unsloth/Qwen3-Coder-30B-GGUF")?.title, "Qwen3-Coder");
  assert.equal(modelGuide("unsloth/Qwen3-Next-80B")?.title, "Qwen3-Next");
  assert.equal(modelGuide("unsloth/Qwen3-VL-8B")?.title, "Qwen3-VL");
  assert.equal(modelGuide("unsloth/Qwen3-8B-GGUF")?.title, "Qwen3");
  // 3n is not 3, and 4 is not 3.
  assert.equal(modelGuide("unsloth/gemma-3n-E4B-it")?.title, "Gemma 3n");
  assert.equal(modelGuide("unsloth/gemma-3-27b-it-GGUF")?.title, "Gemma 3");
  assert.equal(modelGuide("google/gemma-4-12b")?.title, "Gemma 4");
  // GLM point releases, where 5.3 must not read as 5.
  assert.equal(modelGuide("zai-org/GLM-5.3-Flash")?.title, "GLM-5.3-Flash");
  assert.equal(modelGuide("zai-org/GLM-5.3")?.title, "GLM-5.3");
  assert.equal(modelGuide("zai-org/GLM-5")?.title, "GLM-5");
});

test("matching ignores case and the owner prefix", () => {
  assert.equal(modelGuide("UNSLOTH/QWEN3.8-27B")?.title, "Qwen3.8");
  assert.equal(modelGuide("Qwen3.8-27B")?.title, "Qwen3.8");
  assert.equal(modelGuide("unsloth/gpt_oss_20b")?.title, "gpt-oss");
});

test("a family with no guide gets no link, rather than a guessed one", () => {
  assert.equal(modelGuide("sentence-transformers/all-MiniLM-L6-v2"), null);
  assert.equal(modelGuide("some-org/a-model-nobody-wrote-about"), null);
  assert.equal(modelGuide(""), null);
  assert.equal(modelGuide(null), null);
  assert.equal(modelGuide(undefined), null);
});

// Every URL was checked against the live docs, so a malformed one here is a typo.
test("every guide points at an unsloth docs URL", () => {
  const repos = [
    "unsloth/Qwen3.8-27B",
    "unsloth/gpt-oss-20b",
    "unsloth/gemma-4-E2B-it",
    "unsloth/DeepSeek-R1",
    "unsloth/Llama-4-Scout",
    "unsloth/Kimi-K2-Thinking",
    "unsloth/Mistral-3.5",
    "unsloth/phi-4-reasoning",
    "unsloth/QwQ-32B",
  ];
  for (const repo of repos) {
    const guide = modelGuide(repo);
    assert.ok(guide, `${repo} has a guide`);
    assert.match(guide.url, /^https:\/\/unsloth\.ai\/docs\/[\w./-]+$/);
    assert.ok(guide.title.trim().length > 0, `${repo} guide has a title`);
  }
});
