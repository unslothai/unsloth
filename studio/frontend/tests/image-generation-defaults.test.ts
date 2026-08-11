// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { defaultsFor } from "../src/features/images/image-generation-defaults.ts";

test("distinguishes Klein base checkpoints from distilled checkpoints", () => {
  for (const size of ["4B", "9B"]) {
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-base-${size}`), {
      steps: 50,
      guidance: 4,
    });
    assert.deepEqual(defaultsFor(`unsloth/FLUX.2-klein-${size}`), {
      steps: 4,
      guidance: 1,
    });
  }
});

test("keeps the existing family defaults and fallback", () => {
  assert.deepEqual(defaultsFor("krea/Krea-2-Raw"), {
    steps: 52,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("black-forest-labs/FLUX.1-dev"), {
    steps: 28,
    guidance: 3.5,
  });
  assert.deepEqual(defaultsFor("local/unknown-image-model"), {
    steps: 9,
    guidance: 0,
  });
});
