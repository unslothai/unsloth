// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ALLOW_OVERSIZED_LABEL,
  allowOversizedField,
  isMemoryEstimateRefusal,
  MEMORY_REFUSAL_HEADER,
  MEMORY_REFUSAL_KIND,
  MemoryEstimateRefusalError,
  shouldOfferGenerateAnyway,
  shouldRunQueuedOversizedRetry,
} from "../src/features/images/lib/memory-refusal.ts";
import { readSrc, readText } from "./helpers/kit.ts";

test("only a 400 tagged as the memory estimate is the memory refusal", () => {
  assert.equal(isMemoryEstimateRefusal(400, MEMORY_REFUSAL_KIND), true);
  assert.equal(isMemoryEstimateRefusal(400, " Memory-Estimate "), true);
  assert.equal(isMemoryEstimateRefusal(400, null), false);
  assert.equal(isMemoryEstimateRefusal(400, "something-else"), false);
  assert.equal(isMemoryEstimateRefusal(500, MEMORY_REFUSAL_KIND), false);
  assert.equal(isMemoryEstimateRefusal(409, MEMORY_REFUSAL_KIND), false);
});

test("Generate anyway is offered for the refusal, and not again once the override was sent", () => {
  const refusal = new MemoryEstimateRefusalError("Generating at 2048x2048 needs about 34 GB");
  assert.equal(shouldOfferGenerateAnyway({ error: refusal, allowOversizedSent: false }), true);
  assert.equal(shouldOfferGenerateAnyway({ error: refusal, allowOversizedSent: true }), false);
  assert.equal(
    shouldOfferGenerateAnyway({ error: new Error("width is required"), allowOversizedSent: false }),
    false,
  );
});

test("allow_oversized is sent only when the setting or the one-shot retry asks for it", () => {
  assert.equal(allowOversizedField(false, false), undefined);
  assert.equal(allowOversizedField(true, false), true);
  assert.equal(allowOversizedField(false, true), true);
});

test("the setting label matches the one the backend quotes in its refusal", () => {
  const backend = readText("../../backend/core/inference/diffusion_memory.py");
  assert.ok(
    backend.includes(`OVERSIZED_GENERATE_SETTING_LABEL = "${ALLOW_OVERSIZED_LABEL}"`),
    "the refusal would point at a setting the page does not show",
  );
  assert.ok(backend.includes(`IMAGE_REFUSAL_HEADER = "${MEMORY_REFUSAL_HEADER}"`));
  assert.ok(backend.includes(`IMAGE_REFUSAL_MEMORY_ESTIMATE = "${MEMORY_REFUSAL_KIND}"`));
  const main = readText("../../backend/main.py");
  assert.match(main, /expose_headers = \[[^\]]*"X-Unsloth-Refusal"/);
});

test("the page wires the setting, the request field and the toast action", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /usePersistedToggle\(\s*"unsloth_images_allow_oversized"/);
  assert.match(page, /allow_oversized: allowOversizedSent \? true : undefined/);
  assert.match(page, /label: GENERATE_ANYWAY_LABEL/);
  assert.match(page, /checked=\{allowOversized\}/);
  const api = readSrc("features/images/api.ts");
  assert.match(api, /allow_oversized\?: boolean;/);
  assert.match(api, /throw new MemoryEstimateRefusalError\(/);
});

test("Generate anyway clicked during the refused run's cleanup waits for busy to clear", () => {
  assert.equal(shouldRunQueuedOversizedRetry({ queued: true, busy: "generating" }), false);
  assert.equal(shouldRunQueuedOversizedRetry({ queued: true, busy: null }), true);
  assert.equal(shouldRunQueuedOversizedRetry({ queued: false, busy: null }), false);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /onClick: \(\) => setOversizedRetryQueued\(true\)/);
  assert.match(
    page,
    /shouldRunQueuedOversizedRetry\(\{ queued: oversizedRetryQueued, busy \}\)[\s\S]*?\}, \[oversizedRetryQueued, busy, handleGenerateWithRecall\]\);/,
  );
});

test("Generate anyway on an unloaded model keeps the override for the recalled generation", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /load: loadSeq\.current \+ 1,[\s\S]*?allowOversized: oversizedOnce\.current,/);
  assert.match(
    page,
    /oversizedOnce\.current = pending\.allowOversized === true;\s*void handleGenerate\(\)/,
  );
});
