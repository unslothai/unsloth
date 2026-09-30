// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const picker = readSrc("features/model-picker/components/model-selector/pickers.tsx");

test("every quant list and cached GGUF row marks the quant each loaded model runs", () => {
  const expanders = picker.match(/<GgufVariantExpander\b/g)?.length ?? 0;
  assert.equal(picker.match(/loadedQuants=\{/g)?.length ?? 0, expanders);
  assert.match(
    picker,
    /loadedQuants\?\.some\(\(q\) => ggufVariantsMatchForPicker\(q, v\.quant\)\) \? \(\s*<span[^>]*>\s*loaded\s*<\/span>/,
  );
  // Every loaded model, not just the one chat uses.
  assert.match(picker, /loadedModels\s*\.filter\(\(m\) => m\.quant && modelIdsMatchForPicker\(m\.id, repoId\)\)/);
  assert.match(
    picker,
    /if \(activeGgufVariant && modelIdsMatchForPicker\(loadedModelId, repoId\)\) \{\s*quants\.push\(activeGgufVariant\);/,
  );
  assert.match(picker, /meta="GGUF"\s*quantChip=\{loadedQuants\.map\(ggufQuantChipLabel\)\.join\(", "\) \|\| undefined\}/);
});
