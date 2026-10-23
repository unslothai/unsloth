// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// diffusers get_scheduler ignores num_warmup_steps under "constant", so the pair is seeded together.

import assert from "node:assert/strict";
import test from "node:test";

import {
  LR_SCHEDULERS,
  lrSchedulePreset,
} from "../src/features/images/train/diffusion-train-lr-schedule.ts";

import { readSrcAsync } from "./helpers/kit.ts";

const source = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");

test("a family that recommends a ramp carries both halves of it", () => {
  assert.deepEqual(
    lrSchedulePreset({
      lora_rank: 16,
      learning_rate: 0.0001,
      resolution: 512,
      lr_scheduler: "constant_with_warmup",
      lr_warmup_steps: 20,
    }),
    { lrScheduler: "constant_with_warmup", lrWarmupSteps: 20 },
  );
});

test("a family that recommends none contributes nothing to seed", () => {
  assert.deepEqual(
    lrSchedulePreset({ lora_rank: 16, learning_rate: 0.0001, resolution: 1024 }),
    {},
  );
  assert.deepEqual(lrSchedulePreset(null), {});
  assert.deepEqual(lrSchedulePreset(undefined), {});
});

test("half a pair is dropped rather than seeded", () => {
  assert.deepEqual(lrSchedulePreset({ lr_warmup_steps: 20 }), {});
  assert.deepEqual(lrSchedulePreset({ lr_scheduler: "constant_with_warmup" }), {});
});

test("a scheduler the panel cannot show is not seeded into its Select", () => {
  for (const name of ["cosine_with_restarts", "polynomial", "", "linear "]) {
    assert.deepEqual(lrSchedulePreset({ lr_scheduler: name, lr_warmup_steps: 20 }), {});
  }
  for (const name of LR_SCHEDULERS) {
    assert.deepEqual(lrSchedulePreset({ lr_scheduler: name, lr_warmup_steps: 20 }), {
      lrScheduler: name,
      lrWarmupSteps: 20,
    });
  }
});

test("a warmup count that is not a usable step number is dropped", () => {
  for (const warmup of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(
      lrSchedulePreset({ lr_scheduler: "constant_with_warmup", lr_warmup_steps: warmup }),
      {},
    );
  }
  assert.deepEqual(
    lrSchedulePreset({ lr_scheduler: "constant_with_warmup", lr_warmup_steps: 20.7 }),
    { lrScheduler: "constant_with_warmup", lrWarmupSteps: 20 },
  );
});

test("mergeFamilies carries the reported ramp instead of narrowing it away", () => {
  assert.equal(source.match(/\.\.\.lrSchedulePreset\(r\.defaults\),/g)?.length, 2);
  assert.doesNotMatch(source, /lrScheduler:\s*r\.defaults\?\.lr_scheduler\s*\?\?\s*p\./);
});

test("the family re-seed writes the ramp, and resets it for a family without one", () => {
  assert.match(source, /setLrScheduler\(family\.defaults\.lrScheduler \?\? "constant"\);/);
  assert.match(source, /setLrWarmupSteps\(family\.defaults\.lrWarmupSteps \?\? 0\);/);
});

test("an edit to an unrelated setting cannot suppress the family ramp", () => {
  // The LR schedule pair has its own dirty flag; settingsDirty covers Steps and Seed too.
  assert.match(source, /const lrScheduleDirty = useRef\(false\);/);
  assert.match(
    source,
    /\}\s*\n\s*\/\/[\s\S]{0,400}?if \(!lrScheduleDirty\.current\) \{\s*\n\s*setLrScheduler\(/,
  );
  assert.doesNotMatch(source, /settingsDirty\.current = true;\s*\n\s*setLrScheduler\(/);
});

test("a hand-edited ramp survives a family switch", () => {
  assert.match(
    source,
    /onValueChange=\{\(v\) => \{[\s\S]{0,400}?lrScheduleDirty\.current = true;\s*\n\s*setLrScheduler\(v as LrScheduler\);/,
  );
  assert.match(
    source,
    /markDirty: \(\) => \{\s*\n\s*lrScheduleDirty\.current = true;\s*\n\s*\},\s*\n\s*\}\)\}/,
  );
});

test("tuning the ramp does not freeze the other family-seeded settings", () => {
  assert.match(source, /if \(extra\?\.markDirty\) extra\.markDirty\(\);\s*\n\s*else settingsDirty\.current = true;/);
  assert.match(source, /markDirty\?: \(\) => void/);
});
