// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every model the Generate picker offers must be one the backend can actually decode.
// Llasa was offered and could not be: it speaks XCodec2 (65,536 <|s_N|> tokens), which is
// in neither _AUDIO_TOKEN_PATTERNS nor AudioCodecManager, so selecting it loaded the model
// and then failed with "not a supported TTS model". Probed live against a running Unsloth,
// unsloth/Llasa-1B reports is_audio=false while every other curated TTS row reports its
// codec, which is the same shape as the Orpheus defect this PR was opened to fix.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const catalog = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/model-catalog.ts",
    import.meta.url,
  ),
  "utf8",
);
const policy = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/audio-picker-policy.ts",
    import.meta.url,
  ),
  "utf8",
);

// The four the main-slot TTS backend decodes: snac, csm, bicodec, dac.
const DECODABLE = ["orpheus", "csm", "spark", "outetts"];

test("the curated audio catalog offers no model the backend cannot decode", () => {
  const audio = catalog.slice(catalog.indexOf("export const AUDIO_CATALOG"));
  const body = audio.slice(0, audio.indexOf("\n];"));
  assert.doesNotMatch(body, /canonicalId: "[^"]*Llasa[^"]*"/i);
  // The rows that remain are still there, so this cannot pass by emptying the catalog.
  for (const family of ["orpheus", "csm-1b", "Spark-TTS", "OuteTTS", "whisper"]) {
    assert.match(body, new RegExp(family, "i"), family);
  }
});

test("the community TTS family list matches what can be decoded", () => {
  const match = policy.match(/\/\(\?:\^\|\[-_\.\/\]\)\(([^)]+)\)/);
  assert.ok(match, "could not find the family regex");
  const families = match[1].split("|");
  assert.ok(!families.some((f) => /llasa/i.test(f)), families.join("|"));
  for (const decodable of DECODABLE) {
    assert.ok(
      families.some((f) => f.replace(/[-?]/g, "").includes(decodable.replace(/[-?]/g, ""))),
      `${decodable} missing from ${families.join("|")}`,
    );
  }
});
