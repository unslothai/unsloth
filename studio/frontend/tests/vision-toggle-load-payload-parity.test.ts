// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The toggle's plumbing, pinned at the source level because none of it can be
// imported into bare node: applyPerModelConfigToRuntime and the load payload
// builders all reach the chat barrel, which re-exports JSX that
// --experimental-strip-types cannot compile. The rules below are exactly the
// ones that were wrong or missing, so a weaker check than a unit test still
// earns its place.
//
// The failure these prevent is silent and expensive to find: the switch saves,
// displays, round-trips through the config store and never reaches
// /api/inference/load, so Vision looks off and the projector loads anyway.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

const count = (haystack: string, needle: RegExp) =>
  (haystack.match(needle) ?? []).length;

// Hoisted for biome's useTopLevelRegex. Safe to share across the loop below:
// String.match on a global regex resets lastIndex itself.
const TENSOR_PARALLEL_KEY = /tensor_parallel[:?]/g;
const DISABLE_VISION_KEY = /disable_vision[:?]/g;
const VISION_WRITE = /disableVision:/;
const VISION_COMPARE = /disableVision/;
const DIFFUSION_GUARDED_VISION =
  /(?:disableVision|DisableVision)\s*[:=][\s\S]{0,80}?Diffusion/;

// Every file that builds a /load or /validate payload. tensor_parallel is the
// sibling knob with exactly the same lifecycle -- per-model, remembered,
// diffusion-inert -- so it is the reference each payload is measured against.
const PAYLOAD_BUILDERS = [
  "src/features/chat/api/chat-api.ts",
  "src/features/chat/api/chat-adapter.ts",
  "src/features/chat/shared-composer.tsx",
  "src/features/chat/hooks/use-chat-model-runtime.ts",
];

for (const path of PAYLOAD_BUILDERS) {
  test(`${path} sends the toggle everywhere it sends tensor_parallel`, () => {
    const src = read(path);
    const tp = count(src, TENSOR_PARALLEL_KEY);
    const dv = count(src, DISABLE_VISION_KEY);
    assert.ok(tp > 0, "reference knob vanished; this test needs re-anchoring");
    assert.equal(
      dv,
      tp,
      `${path} names tensor_parallel ${tp} time(s) but disable_vision ${dv}: a load or validate path is not carrying the vision toggle`,
    );
  });
}

test("the runtime store is written, snapshotted and compared on the toggle", () => {
  // applyPerModelConfigToRuntime is what a model switch and a preset apply both
  // go through. It read the key and never wrote it, so a remembered setting was
  // silently dropped the moment the user changed models.
  const src = read(
    "src/features/model-picker/model-config/apply-per-model-config.ts",
  );
  const apply = src.slice(
    src.indexOf("export function applyPerModelConfigToRuntime"),
    src.indexOf("export function currentRuntimePerModelConfig"),
  );
  assert.ok(
    VISION_WRITE.test(apply),
    "applyPerModelConfigToRuntime does not write disableVision to the store",
  );

  const snapshot = src.slice(
    src.indexOf("export function currentRuntimePerModelConfig"),
    src.indexOf("export function perModelConfigsEqual"),
  );
  assert.ok(
    VISION_WRITE.test(snapshot),
    "the runtime snapshot drops disableVision, so a rollback loses it",
  );

  const equals = src.slice(src.indexOf("export function perModelConfigsEqual"));
  assert.ok(
    VISION_COMPARE.test(equals),
    "two configs differing only in Vision would compare equal",
  );
});

test("the toggle is a diffusion no-op wherever tensor_parallel is", () => {
  // The diffusion runner has no projector, so sending the flag would record a
  // setting the process never got -- the same reason tensorParallel is forced
  // off on these paths.
  for (const path of [
    "src/features/model-picker/model-config/apply-per-model-config.ts",
    "src/features/chat/api/chat-adapter.ts",
    "src/features/chat/shared-composer.tsx",
    "src/features/chat/hooks/use-chat-model-runtime.ts",
  ]) {
    const src = read(path);
    // The value bound for the load has to consult a diffusion flag within the
    // same expression, not merely somewhere in the file.
    assert.ok(
      DIFFUSION_GUARDED_VISION.test(src),
      `${path} does not force the vision toggle off for a diffusion load`,
    );
  }
});

test("the preset round-trip carries the toggle at every stage", () => {
  // PresetLoadConfig has its own key list, normalizer, snapshot, apply and
  // summary; a knob added to PerModelConfig alone does not round-trip, so a
  // preset saved with Vision off applied with Vision on.
  const src = read("src/features/chat/presets/preset-load-config.ts");
  for (const [label, marker] of [
    ["the saved key list", '| "disableVision"'],
    ["the empty default", "disableVision: false,"],
    ["the normalizer", 'typeof partial.disableVision === "boolean"'],
    ["the snapshot", "disableVision: snapshot.disableVision"],
    ["the apply", "disableVision: config.disableVision"],
    ["the summary", "if (config.disableVision)"],
  ] as const) {
    assert.ok(
      src.includes(marker),
      `preset ${label} does not carry the toggle`,
    );
  }
});
