// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { INFO_RECIPE_KEYS } from "../src/features/images/lib/image-metadata.ts";
import { recipeTimeLabel } from "../src/features/images/lib/recipe-time.ts";

test("a first render's Recipe time says it includes compile and tuning", () => {
  assert.equal(
    recipeTimeLabel({ generation_seconds: 21.9, warmup_render: true }),
    "21.9 s (first render, includes compile and tuning)",
  );
});

test("a steady-state Recipe time keeps the old text", () => {
  assert.equal(recipeTimeLabel({ generation_seconds: 1.519 }), "1.5 s");
  assert.equal(recipeTimeLabel({ generation_seconds: 0.42 }), "0.42 s");
  assert.equal(
    recipeTimeLabel({ generation_seconds: 6.2, generation_count: 4, warmup_render: false }),
    "6.2 s (batch of 4)",
  );
});

test("warmup_render is a known recipe key and the Recipe row uses the label", () => {
  assert.ok((INFO_RECIPE_KEYS as readonly string[]).includes("warmup_render"));
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /value=\{recipeTimeLabel\(image\)\}/);
});
