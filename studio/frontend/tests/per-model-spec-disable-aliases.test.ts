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

/** null means follow the global preference, so every disable alias must map to off. */
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
  assert.equal(specOf("auto"), null);
  assert.equal(specOf("default"), null);
  assert.equal(specOf("bogus"), null);
  assert.equal(specOf("mtp"), "mtp");
  assert.equal(specOf("draft-mtp"), "mtp");
  assert.equal(specOf("ngram-mod"), "ngram");
  assert.equal(specOf("mtp+ngram"), "mtp+ngram");
});
