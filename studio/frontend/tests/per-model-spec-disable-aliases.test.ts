// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const specOf = (value: string) =>
  normalizePerModelConfig({ speculativeType: value }).speculativeType;

/**
 * A server-side override reaches this canonicalizer with whatever the API caller
 * wrote, and `/settings` stores `speculative_type` without canonicalizing it. The
 * backend reads llama.cpp's own "none", plus "disable" / "disabled", as off. Here
 * null does not mean off, it means follow the global preference, so a spelling that
 * fell through would turn an explicit disable into Auto and hand the load a drafter.
 */
test("a stored disable alias is an override, not a fall-through to the global default", () => {
  for (const spelling of [
    "off",
    "none",
    "None",
    "NONE",
    "  none  ",
    "disable",
    "Disabled",
    "disabled",
  ]) {
    assert.equal(specOf(spelling), "off", `${spelling} must read as off`);
  }
});

test("the rest of the mapping still resolves as before", () => {
  // "auto" and an unknown value are the follow-global sentinel, which is what the
  // aliases above must not be confused with.
  assert.equal(specOf("auto"), null);
  assert.equal(specOf("default"), null);
  assert.equal(specOf("bogus"), null);
  assert.equal(specOf("mtp"), "mtp");
  assert.equal(specOf("draft-mtp"), "mtp");
  assert.equal(specOf("ngram-mod"), "ngram");
  assert.equal(specOf("mtp+ngram"), "mtp+ngram");
});
