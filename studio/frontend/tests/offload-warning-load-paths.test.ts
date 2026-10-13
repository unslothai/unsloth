// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Every load path must surface the offload warning, not a plain success toast.
const LOAD_PATHS = [
  "../src/features/chat/hooks/use-chat-model-runtime.ts",
  "../src/features/chat/api/chat-adapter.ts",
  "../src/features/chat/shared-composer.tsx",
  "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
  "../src/features/audio/hooks/use-audio-model-slot.ts",
];

test("every user-facing load path consults the offload warning", () => {
  for (const path of LOAD_PATHS) {
    const source = readFileSync(
      fileURLToPath(new URL(path, import.meta.url)),
      "utf8",
    );
    assert.match(source, /offloadWarning\(/, `${path} ignores the split`);
  }
});

test("a recipe run no longer claims plain success unconditionally", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(source, /offloadNotice \? toast\.warning : toast\.success/);
  assert.doesNotMatch(source, /toast\.success\(`Loaded \$\{modelLabel\}`/);
});
