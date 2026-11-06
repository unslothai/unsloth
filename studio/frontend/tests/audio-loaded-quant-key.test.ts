// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { audioLoadedSelection } = await import("../src/features/audio/audio-page-policy.ts");
const { ggufQuantLabel } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);

test("a quant picked from the list reads as its quant key, not the file name llama echoes", () => {
  // The loader sends the file name, llama status echoes it, and picker rows key the quant.
  const loaded = (gguf_variant: string | null) =>
    audioLoadedSelection(
      { active_model: "unsloth/orpheus-3b-0.1-ft-GGUF", gguf_variant },
      ggufQuantLabel,
    ).variant;
  assert.equal(loaded("orpheus-3b-0.1-ft-q4_k_m.gguf"), "Q4_K_M");
  assert.equal(loaded("Orpheus-3B-UD-Q4_K_XL.GGUF"), "UD-Q4_K_XL");
  assert.equal(loaded("distilled/model-q8_0.gguf"), "distilled/Q8_0");
  assert.equal(loaded("Q8_0"), "Q8_0");
  assert.equal(loaded(null), null);
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /const speakLoaded = audioLoadedSelection\(status, ggufQuantLabel\);/,
  );
});
