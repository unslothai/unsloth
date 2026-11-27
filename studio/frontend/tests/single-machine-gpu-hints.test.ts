// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { en } from "../src/i18n/locales/en.ts";

// Each GPU hint must say multi-machine (e.g. two DGX Sparks) is unsupported in this release.
const NOT_RELEASED = /not part of this release/;
const TWO_MACHINES = /two DGX Sparks/;

test("the engine GPU hint says serving across machines is not part of this release", () => {
  const help = en.managedEngines.gpuHelp;
  assert.match(help, /this machine/);
  assert.match(help, TWO_MACHINES);
  assert.match(help, NOT_RELEASED);
});

test("the training GPU override hint says multi-node training is not part of this release", () => {
  const hint = en.studio.gpus.overrideHint;
  assert.match(hint, /multi-node training/);
  assert.match(hint, TWO_MACHINES);
  assert.match(hint, NOT_RELEASED);
});

for (const page of ["images/images-page.tsx", "video/video-page.tsx"]) {
  test(`the ${page} device hint says one job across machines is not part of this release`, () => {
    const source = readFileSync(new URL(`../src/features/${page}`, import.meta.url), "utf8");
    // A plain string, or a template that appends the shared multi-GPU batch-split note.
    const hint = source.match(/hint=(?:"|\{`)(Which card this model loads on\.[^"`]*)["`]/);
    assert.ok(hint, "device hint not found");
    assert.match(hint[1], TWO_MACHINES);
    assert.match(hint[1], NOT_RELEASED);
  });
}
