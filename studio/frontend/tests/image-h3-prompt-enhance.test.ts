// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import {
  ENHANCE_PROMPT_HINT,
  enhancePromptHint,
  enhancePromptRequestField,
  H3_IMAGE_ENHANCE_PROMPT_HINT,
  H3_IMAGE_FAMILY,
  PROMPT_UPSAMPLER_ROUTES,
  promptUpsamplerCapability,
  readEnhanceChoices,
  resolveEnhancePrompt,
  upsamplerRouteFor,
  withEnhanceChoice,
} from "../src/features/images/lib/prompt-enhance.ts";

// What the backend advertises for MiniMax-H3-Image: supported, on by default (both reference templates ship it on).
const H3 = { loaded: true, family: H3_IMAGE_FAMILY, supports_prompt_upsampler: true, prompt_upsampler_default: true };

test("MiniMax-H3-Image shows the Enhance toggle, on by default", () => {
  installLocalStorageFake();
  const cap = promptUpsamplerCapability(H3);
  assert.deepEqual(cap, { supported: true, defaultOn: true });
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: readEnhanceChoices()[H3_IMAGE_FAMILY] }), true);
  assert.deepEqual(enhancePromptRequestField(cap, true), { enhance_prompt: true });
  // Turning it off is stored for this family only; back on drops the stored choice (equal to the default).
  let choices = withEnhanceChoice({}, H3_IMAGE_FAMILY, false, cap.defaultOn);
  assert.deepEqual(choices, { [H3_IMAGE_FAMILY]: false });
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: choices[H3_IMAGE_FAMILY] }), false);
  choices = withEnhanceChoice(choices, H3_IMAGE_FAMILY, true, cap.defaultOn);
  assert.deepEqual(choices, {});
});

test("the hint names the family's own enhancer", () => {
  assert.equal(enhancePromptHint(H3_IMAGE_FAMILY), H3_IMAGE_ENHANCE_PROMPT_HINT);
  assert.equal(enhancePromptHint("flux.3-klein"), ENHANCE_PROMPT_HINT);
  assert.equal(enhancePromptHint(null), ENHANCE_PROMPT_HINT);
  assert.match(H3_IMAGE_ENHANCE_PROMPT_HINT, /Qwen3\.5-9B/);
  assert.doesNotMatch(H3_IMAGE_ENHANCE_PROMPT_HINT, /\u2014/);
});

test("status, download and cancel name the family; no family keeps the old URL", () => {
  assert.equal(
    upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.status, H3_IMAGE_FAMILY),
    "/api/inference/images/prompt-upsampler/status?family=minimax-h3-image",
  );
  assert.equal(upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.status, null), PROMPT_UPSAMPLER_ROUTES.status);
  assert.equal(upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.cancel, ""), PROMPT_UPSAMPLER_ROUTES.cancel);
  const api = readSrc("features/images/prompt-upsampler-api.ts");
  for (const route of ["status", "download", "cancel"]) {
    assert.match(api, new RegExp(`upsamplerRouteFor\\(PROMPT_UPSAMPLER_ROUTES\\.${route}, family\\)`), route);
  }
  const panel = readSrc("features/images/prompt-enhance-panel.tsx");
  // The status is re-read when the family changes (FLUX.3 to H3-Image without a reload).
  assert.match(panel, /getPromptUpsamplerStatus\(controller\.signal, family\)[\s\S]{0,200}\[active, tick, family\]/);
  assert.match(panel, /startPromptUpsamplerDownload\(hfApiToken\(getHfToken\(\)\), family\)/);
  assert.match(panel, /cancelPromptUpsamplerDownload\(family\)/);
  assert.match(panel, /<InfoHint>\{enhancePromptHint\(family\)\}<\/InfoHint>/);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /family=\{status\?\.family \?\? null\}/);
  assert.match(page, /\.\.\.\(status\?\.family \? \{ family: status\.family \} : \{\}\)/);
});
