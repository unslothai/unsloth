// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reported from Unsloth Desktop: Upscale was refused as too big for the GPU, and the only way past
// the refusal was an environment variable a desktop install has no terminal to set. The override
// now travels with the request, from a persisted setting or from the refusal toast.

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
} from "../src/features/images/lib/memory-refusal.ts";
import { readSrc, readText } from "./helpers/kit.ts";

test("only a 400 tagged as the memory estimate is the memory refusal", () => {
  assert.equal(isMemoryEstimateRefusal(400, MEMORY_REFUSAL_KIND), true);
  assert.equal(isMemoryEstimateRefusal(400, " Memory-Estimate "), true);
  // Every other 400, and any other status, stays an ordinary error.
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
  // The desktop app is cross-origin, so the header is unreadable unless CORS exposes it.
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
