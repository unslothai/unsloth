// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import {
  normalizeSamplerChoice,
  readSamplerChoices,
  samplerRecipeLabel,
  samplerRequestFields,
  withSamplerChoice,
} from "../src/features/images/lib/sampler-choice.ts";

const SDXL = [
  { id: "default", label: "Model default", schedule_types: ["automatic", "uniform", "karras", "exponential"] },
  { id: "euler_a", label: "Euler a", schedule_types: ["automatic", "uniform"] },
  { id: "dpmpp_2m", label: "DPM++ 2M", schedule_types: ["automatic", "uniform", "karras", "exponential"] },
];

test("the shipped scheduler sends the request it always sent", () => {
  assert.deepEqual(samplerRequestFields(SDXL, { sampler: "default", scheduleType: "automatic" }), {});
  assert.deepEqual(samplerRequestFields([], { sampler: "dpmpp_2m", scheduleType: "karras" }), {});
  assert.deepEqual(samplerRequestFields(SDXL, { sampler: "dpmpp_2m", scheduleType: "karras" }), {
    sampler: "dpmpp_2m",
    schedule_type: "karras",
  });
});

test("a choice the loaded model lacks falls back instead of failing the generation", () => {
  assert.deepEqual(normalizeSamplerChoice(SDXL, { sampler: "euler_a", scheduleType: "karras" }), {
    sampler: "euler_a",
    scheduleType: "automatic",
  });
  assert.deepEqual(normalizeSamplerChoice(SDXL, { sampler: "dpmpp_2m_sde" }), {
    sampler: "default",
    scheduleType: "automatic",
  });
  assert.deepEqual(normalizeSamplerChoice([], { sampler: "euler_a" }), {
    sampler: "default",
    scheduleType: "automatic",
  });
});

test("choices are remembered per model and the default is not stored", () => {
  const { store } = installLocalStorageFake();
  let all = withSamplerChoice({}, "sdxl-a", { sampler: "dpmpp_2m", scheduleType: "karras" });
  all = withSamplerChoice(all, "sdxl-b", { sampler: "euler_a", scheduleType: "automatic" });
  assert.deepEqual(readSamplerChoices(), all);
  all = withSamplerChoice(all, "sdxl-b", { sampler: "default", scheduleType: "automatic" });
  assert.deepEqual(Object.keys(readSamplerChoices()), ["sdxl-a"]);
  store.set("unsloth:images:sampler-by-model", "not json");
  assert.deepEqual(readSamplerChoices(), {});
});

test("recipe label", () => {
  assert.equal(samplerRecipeLabel("dpmpp_2m", "karras"), "DPM++ 2M Karras");
  assert.equal(samplerRecipeLabel("euler_a", null), "Euler a");
  assert.equal(samplerRecipeLabel(null, null), null);
});

test("the Images page sends, restores and shows the sampler only where the model offers one", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /\.\.\.samplerRequestFields\(samplerOptions, samplerChoice\)/);
  assert.match(page, /samplerOptions\.length > 1 && \(/);
  assert.match(page, /scheduleType: image\.schedule_type \?\? AUTOMATIC_SCHEDULE/);
});
