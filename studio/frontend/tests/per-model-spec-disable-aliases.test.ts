// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizePerModelConfig, storedSpeculativeAuto } = await import(
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
  // An unknown value is the follow-global sentinel the aliases must not become; Auto is kept.
  assert.equal(specOf("auto"), "auto");
  assert.equal(specOf("default"), "auto");
  assert.equal(specOf("bogus"), null);
  assert.equal(specOf("mtp"), "mtp");
  assert.equal(specOf("draft-mtp"), "mtp");
  assert.equal(specOf("ngram-mod"), "ngram");
  assert.equal(specOf("mtp+ngram"), "mtp+ngram");
});

test("MLX modes and an explicit Auto survive storage; the drafter only with a drafter mode", () => {
  for (const mode of ["eagle3", "dflash+ngram"]) {
    assert.equal(specOf(mode), mode);
  }
  const drafterOf = (speculativeType: string, specDraftModel: string) =>
    normalizePerModelConfig({ speculativeType, specDraftModel }).specDraftModel;
  assert.deepEqual(
    [drafterOf("auto", " o/d "), drafterOf("ngram", "o/d"), drafterOf("mtp", " ")],
    ["o/d", null, null],
  );
  // GGUF's Auto is "follow the standing preference", which storage spells null.
  assert.equal(storedSpeculativeAuto({ speculativeType: "auto" }, false).speculativeType, null);
  assert.equal(storedSpeculativeAuto({ speculativeType: "auto" }, true).speculativeType, "auto");
});
