// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The six flow-matching DiT families recommend a 20-step LR ramp, and the backend now pairs that
// count with `constant_with_warmup` because diffusers' get_scheduler returns before it reads
// num_warmup_steps under plain "constant". The Train panel is where that recommendation has to
// land: it seeds its own state from the family and always sends lr_scheduler + lr_warmup_steps
// explicitly, so a pair it never reads is a pair the primary training flow never applies.
// mergeFamilies and the seeding effect are inline in the panel, so the wiring is asserted against
// the source the same way the batch cap is in diffusion-train-batch-cap.test.ts; the pair's
// normalization lives in its own module and is exercised directly.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  LR_SCHEDULERS,
  lrSchedulePreset,
} from "../src/features/images/train/diffusion-train-lr-schedule.ts";

const source = await readFile(
  new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
  "utf8",
);

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
  // sdxl / z-image / krea-2: no warmup preset, so the panel keeps its own "constant" default.
  assert.deepEqual(
    lrSchedulePreset({ lora_rank: 16, learning_rate: 0.0001, resolution: 1024 }),
    {},
  );
  assert.deepEqual(lrSchedulePreset(null), {});
  assert.deepEqual(lrSchedulePreset(undefined), {});
});

test("half a pair is dropped rather than seeded", () => {
  // A warmup count with no scheduler is the bug the backend pairing fixes: under "constant" it
  // ramps nothing. Seeding it alone would put the count back in the UI with the same outcome.
  assert.deepEqual(lrSchedulePreset({ lr_warmup_steps: 20 }), {});
  // And a scheduler with no count would advertise a ramp of zero steps.
  assert.deepEqual(lrSchedulePreset({ lr_scheduler: "constant_with_warmup" }), {});
});

test("a scheduler the panel cannot show is not seeded into its Select", () => {
  // The backend's Literal is wider than the four options the panel offers. Seeding one of the
  // others leaves the Select on a value with no item, which renders blank.
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
  // The field is an integer step count on the way out; a float would be sent as typed.
  assert.deepEqual(
    lrSchedulePreset({ lr_scheduler: "constant_with_warmup", lr_warmup_steps: 20.7 }),
    { lrScheduler: "constant_with_warmup", lrWarmupSteps: 20 },
  );
});

test("mergeFamilies carries the reported ramp instead of narrowing it away", () => {
  // Both merge arms: the preset-matched family and the backend-only one that goes last.
  assert.equal(source.match(/\.\.\.lrSchedulePreset\(r\.defaults\),/g)?.length, 2);
  // No preset fallback for the ramp, unlike rank/lr/resolution: a reported family owns it, so a
  // backend that drops its warmup preset drops the ramp here rather than a stale copy of it.
  assert.doesNotMatch(source, /lrScheduler:\s*r\.defaults\?\.lr_scheduler\s*\?\?\s*p\./);
});

test("the family re-seed writes the ramp, and resets it for a family without one", () => {
  assert.match(source, /setLrScheduler\(family\.defaults\.lrScheduler \?\? "constant"\);/);
  assert.match(source, /setLrWarmupSteps\(family\.defaults\.lrWarmupSteps \?\? 0\);/);
});

test("an edit to an unrelated setting cannot suppress the family ramp", () => {
  // settingsDirty is one flag over every numeric control, Steps and Seed included. Gating the
  // ramp on it meant typing a step count and then switching to a flow-matching DiT left the
  // panel on "constant" with the Warmup steps field hidden, so the ramp this panel exists to
  // seed silently never ran. The pair gets its own flag, outside the settingsDirty block.
  assert.match(source, /const lrScheduleDirty = useRef\(false\);/);
  assert.match(
    source,
    /\}\s*\n\s*\/\/[\s\S]{0,400}?if \(!lrScheduleDirty\.current\) \{\s*\n\s*setLrScheduler\(/,
  );
  // And it is NOT re-gated on settingsDirty anywhere.
  assert.doesNotMatch(source, /settingsDirty\.current = true;\s*\n\s*setLrScheduler\(/);
});

test("a hand-edited ramp survives a family switch", () => {
  // Both halves mark the pair dirty, or the re-seed replaces the user's pick on the next switch.
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
  // "Warmup steps" is newly visible by default for the six flow families, so it now reaches
  // numberField's dirty mark in normal use. Charging it to the shared settingsDirty would mean
  // typing a ramp length pinned rank/LR/resolution to the previous family: switch flux.1 ->
  // qwen-image afterwards and the LR stays 1e-4 instead of re-seeding to 5e-5.
  assert.match(source, /if \(extra\?\.markDirty\) extra\.markDirty\(\);\s*\n\s*else settingsDirty\.current = true;/);
  assert.match(source, /markDirty\?: \(\) => void/);
});
