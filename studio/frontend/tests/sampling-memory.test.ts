// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  fromApiSampling,
  recommendedSampling,
  sameSampling,
  samplingPayload,
  samplingSaveStatus,
  toApiSampling,
} = await import("../src/features/chat/lib/sampling-memory.ts");
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const params = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  minPMode: "server-default" as const,
  repetitionPenalty: 1.1,
  presencePenalty: 0.2,
};

test("Save stores Min P only when it is set by hand", () => {
  assert.equal("minP" in samplingPayload(params), false);
  assert.equal(samplingPayload({ ...params, minPMode: "custom" }).minP, 0.05);
});

test("the request uses the API's names and forgets with nulls", () => {
  assert.deepEqual(toApiSampling(samplingPayload(params)), {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    min_p: null,
    repetition_penalty: 1.1,
    presence_penalty: 0.2,
  });
  assert.ok(Object.values(toApiSampling(null)).every((v) => v === null));
  assert.equal(fromApiSampling({}), null);
  assert.deepEqual(fromApiSampling({ top_p: 0.8, junk: 1 }), { topP: 0.8 });
});

test("the status line follows Remember and whether the values are saved", () => {
  const current = samplingPayload(params);
  assert.equal(samplingSaveStatus(false, null, current), null);
  assert.equal(samplingSaveStatus(true, null, current), "Unsaved changes");
  assert.equal(
    samplingSaveStatus(true, { ...current }, current),
    "Saved for this model",
  );
  assert.equal(
    samplingSaveStatus(true, { ...current, temperature: 1 }, current),
    "Unsaved changes",
  );
  assert.equal(
    samplingSaveStatus(false, current, current),
    "Saved settings will be forgotten",
  );
  assert.equal(sameSampling({ topK: 40 }, { topK: 40, minP: 0.1 }), false);
});

test("Reset takes the model's recommendation, app defaults for the rest", () => {
  assert.deepEqual(
    recommendedSampling(
      { temperature: 1, top_p: 0.95 },
      DEFAULT_INFERENCE_PARAMS,
    ),
    {
      temperature: 1,
      topP: 0.95,
      topK: DEFAULT_INFERENCE_PARAMS.topK,
      minP: DEFAULT_INFERENCE_PARAMS.minP,
      repetitionPenalty: DEFAULT_INFERENCE_PARAMS.repetitionPenalty,
      presencePenalty: DEFAULT_INFERENCE_PARAMS.presencePenalty,
    },
  );
});

test("the Sampling section ends with the footer, not a lone Save button", () => {
  const sheet = readFileSync(
    new URL("../src/features/chat/chat-settings-sheet.tsx", import.meta.url),
    "utf8",
  );
  const sampling = sheet.slice(
    sheet.indexOf('<CollapsibleSection label="Sampling"'),
    sheet.indexOf('<CollapsibleSection label="Tools"'),
  );
  assert.match(
    sampling,
    /<SamplingFooter\s+store=\{useChatRuntimeStore\}\s+notify=\{toast\}\s+Checkbox=\{Checkbox\}\s+defaults=\{DEFAULT_INFERENCE_PARAMS\}\s+qwenThinking=\{resolveQwenThinkingParams\}\s+\/>/,
  );
  assert.doesNotMatch(sheet, /SaveForModelButton|Save for this model/);
});

test("a save or forget that finishes after a model switch leaves the new model's state alone", () => {
  const footer = readFileSync(
    new URL(
      "../src/features/chat/components/sampling-footer.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    footer,
    /const stillOn = \(\) => store\.getState\(\)\.params\.checkpoint === modelId;/,
  );
  assert.match(
    footer,
    /await writeSavedSampling\(modelId, sampling\);\s*if \(stillOn\(\)\) \{\s*setSaved\(sampling\);/,
  );
  assert.match(
    footer,
    /await writeSavedSampling\(modelId, null\);\s*if \(stillOn\(\)\) \{\s*setSaved\(null\);/,
  );
});

test("Reset layers Qwen3's thinking-mode table over the base recommendation", async () => {
  const { resolveQwenThinkingParams } = await import(
    "../src/features/chat/utils/qwen-sampling-table.ts"
  );
  assert.deepEqual(
    {
      ...recommendedSampling(
        { temperature: 0.6, top_p: 0.95 },
        DEFAULT_INFERENCE_PARAMS,
      ),
      ...resolveQwenThinkingParams("unsloth/Qwen3-8B-GGUF", false),
    },
    {
      ...recommendedSampling({}, DEFAULT_INFERENCE_PARAMS),
      temperature: 0.7,
      topP: 0.8,
      topK: 20,
      minP: 0,
    },
  );
  const footer = readFileSync(
    new URL(
      "../src/features/chat/components/sampling-footer.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    footer,
    /const qwen = status\.supports_reasoning\s*\?\s*qwenThinking\(\s*modelId,\s*status\.reasoning_always_on === true \|\| state\.reasoningEnabled,?\s*\)\s*:\s*null;/,
  );
  assert.match(
    footer,
    /\.\.\.recommendedSampling\([\s\S]*?\),\s*\.\.\.qwen,\s*\}\);/,
  );
});

test("Remember and Save wait for the first read, so its older result cannot overwrite them", () => {
  const footer = readFileSync(
    new URL(
      "../src/features/chat/components/sampling-footer.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    footer,
    /setLoaded\(false\);\s*readSavedSampling\(modelId, controller\.signal\)/,
  );
  assert.match(
    footer,
    /\.finally\(\(\) => \{\s*if \(!controller\.signal\.aborted\) \{\s*setLoaded\(true\);/,
  );
  assert.match(footer, /checked=\{remember\}\s*disabled=\{!loaded\}/);
  assert.match(footer, /disabled=\{!loaded \|\| busy\}/);
  assert.match(footer, /\(remember \? !unchanged : saved !== null\) &&/);
});
