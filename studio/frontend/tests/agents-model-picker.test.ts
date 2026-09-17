// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  isChatGenerativeHubModel,
  isClassifierOrRerankerHubModel,
  isSpeechOnlyHubModel,
} from "../src/features/settings/lib/agent-hub-model.ts";
import { en } from "../src/i18n/locales/en.ts";

const TAB = readFileSync(
  fileURLToPath(
    new URL("../src/features/settings/tabs/agents-tab.tsx", import.meta.url),
  ),
  "utf-8",
);

test("the default model demonstrates reasoning effort without sampling flags", () => {
  assert.ok(
    TAB.includes('const EXAMPLE_MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF";'),
  );
  assert.ok(TAB.includes('const EXAMPLE_MODEL_VARIANT = "UD-Q4_K_XL";'));
  const start = TAB.indexOf("const EXAMPLE_MODEL_FLAGS");
  const flags = TAB.slice(start, TAB.indexOf(";", start));
  assert.ok(flags.includes("--reasoning-effort medium"));
  for (const samplingFlag of [
    "--temperature",
    "--top-p",
    "--top-k",
    "--min-p",
    "--presence-penalty",
  ]) {
    assert.ok(!flags.includes(samplingFlag));
  }
  assert.ok(
    TAB.includes("modelKey(selectedModel) === modelKey(EXAMPLE_MODEL_REPO)"),
  );
  assert.equal(
    en.settings.agents.automaticSettingsNote,
    "Unsloth automatically applies the model’s recommended settings if you have not set any flags.",
  );
  assert.equal(
    en.settings.agents.configurationNote,
    "You can also adjust any configuration. See further below or",
  );
  assert.equal(en.settings.agents.configurationDocs, "docs");
  assert.equal(en.settings.agents.configurationFlagsSuffix, "for flags.");
  assert.ok(TAB.includes("href={FLAGS_DOCS_URL}"));
  assert.ok(TAB.includes("#flags--options"));
});

test("the model dropdown loads live trending GGUFs", () => {
  const start = TAB.indexOf('useHubModelSearch("", {');
  const request = TAB.slice(start, TAB.indexOf("});", start));
  assert.ok(request.includes('owner: "unsloth"'));
  assert.ok(request.includes('tags: ["gguf"]'));
  assert.ok(request.includes('sortBy: "trendingScore"'));
  assert.ok(request.includes('sortDirection: "desc"'));
  assert.ok(request.includes("keepUnsupportedTags: false"));
  assert.ok(TAB.includes("isChatGenerativeHubModel(model)"));
  assert.ok(TAB.includes("!isEmbeddingHubModel(model)"));
  assert.ok(TAB.includes("EMBEDDING_TAGS.has(tag.toLowerCase())"));
  assert.ok(TAB.includes("!isSpeechOnlyHubModel(model)"));
  assert.ok(TAB.includes("!isClassifierOrRerankerHubModel(model)"));
  assert.ok(TAB.includes("mergeModelOrder(trendingModels, models)"));
  assert.ok(TAB.includes("[...primary, ...fallback]"));
});

test("the agent feed admits only declared chat-generation pipelines", () => {
  for (const pipelineTag of [
    "text-generation",
    "conversational",
    "image-text-to-text",
    "audio-text-to-text",
    "any-to-any",
  ]) {
    assert.equal(isChatGenerativeHubModel({ pipelineTag }), true);
  }
  for (const pipelineTag of [
    "fill-mask",
    "audio-classification",
    "voice-activity-detection",
    "feature-extraction",
    "question-answering",
    "image-classification",
    "text-to-speech",
    "text-classification",
  ]) {
    assert.equal(isChatGenerativeHubModel({ pipelineTag }), false);
  }
  assert.equal(
    isChatGenerativeHubModel({ pipelineTag: "  TEXT-GENERATION " }),
    true,
  );
  assert.equal(isChatGenerativeHubModel({}), true);
});

test("the agent feed excludes speech-only model tasks", () => {
  for (const pipelineTag of [
    "text-to-speech",
    "automatic-speech-recognition",
  ]) {
    assert.equal(isSpeechOnlyHubModel({ pipelineTag }), true);
  }
  assert.equal(
    isSpeechOnlyHubModel({
      tags: ["GGUF", " TEXT-TO-SPEECH "],
    }),
    true,
  );
  assert.equal(
    isSpeechOnlyHubModel({
      pipelineTag: "text-generation",
      tags: ["gguf", "text-to-speech"],
    }),
    false,
  );
  assert.equal(
    isSpeechOnlyHubModel({
      pipelineTag: "audio-text-to-text",
      tags: ["gguf", "automatic-speech-recognition"],
    }),
    false,
  );
  assert.equal(
    isSpeechOnlyHubModel({ pipelineTag: "image-text-to-text" }),
    false,
  );
});

test("the agent feed excludes classifier and reranker models", () => {
  for (const pipelineTag of [
    "text-classification",
    "token-classification",
    "zero-shot-classification",
    "text-ranking",
  ]) {
    assert.equal(isClassifierOrRerankerHubModel({ pipelineTag }), true);
  }
  assert.equal(
    isClassifierOrRerankerHubModel({ id: "unsloth/Qwen3-Reranker-GGUF" }),
    true,
  );
  assert.equal(
    isClassifierOrRerankerHubModel({ tags: ["gguf", "cross-encoder"] }),
    true,
  );
  assert.equal(
    isClassifierOrRerankerHubModel({
      pipelineTag: "text-generation",
      tags: ["text-classification"],
    }),
    false,
  );
  assert.equal(
    isClassifierOrRerankerHubModel({
      id: "unsloth/Qwen3.8-27B-GGUF",
      pipelineTag: "text-generation",
    }),
    false,
  );
});

test("restored Hub selections remain valid while uncached", () => {
  assert.ok(TAB.includes("isHuggingFaceRepo(restored)"));
});

test("model selection matching ignores Hub repository casing", () => {
  assert.ok(TAB.includes("modelKey(model) === selectedKey"));
  assert.ok(TAB.includes("modelKey(model) === modelKey(selectedModel)"));
});

test("an adopted resident model does not use a cached load ID", () => {
  assert.ok(TAB.includes("const selectedModelIsActive"));
  assert.ok(
    TAB.includes("const cachedLoadId = selectedModelIsActive\n    ? null"),
  );
});
