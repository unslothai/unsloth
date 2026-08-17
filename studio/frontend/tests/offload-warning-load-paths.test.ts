// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Five separate flows load a model, and each grew its own success toast. A
// silent split announced as success in any one of them is the whole defect, so
// the list is asserted rather than left to whoever touches a load path next.
const LOAD_PATHS = [
  "../src/features/chat/hooks/use-chat-model-runtime.ts",
  "../src/features/chat/api/chat-adapter.ts",
  "../src/features/chat/shared-composer.tsx",
  "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
  // The Audio page loads a GGUF TTS variant, which fits like any other GGUF.
  "../src/features/audio/audio-page.tsx",
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
  // It loads with no panel open, so before this the run simply came out slow.
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
