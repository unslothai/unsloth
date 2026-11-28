// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/recipe-studio/dialogs/models/local-recipe-model-selector.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the Recipe local model selector drops partial rows", () => {
  assert.match(
    source,
    /setModels\(\s*response\.models\.filter\(\(model\) => !model\.partial\)\s*\)/,
  );
});
