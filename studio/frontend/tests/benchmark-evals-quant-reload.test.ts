// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("picking another quant of the loaded GGUF repo loads it before the eval runs", () => {
  const src = readFileSync(
    new URL("../src/features/benchmark/use-evals-model.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /meta\.ggufVariant\.toLowerCase\(\) !== \(activeGgufVariant \?\? ""\)\.toLowerCase\(\)/,
  );
  assert.match(
    src,
    /value !== \(inferenceParams\.checkpoint \?\? undefined\) \|\| otherQuant/,
  );
});
