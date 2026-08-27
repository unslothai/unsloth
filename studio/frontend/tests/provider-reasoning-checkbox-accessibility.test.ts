// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
  "utf8",
);
const reasoningModelCheckbox =
  /id={`provider-reasoning-model-\${index}`}\s+aria-label={`Mark \${model} as a reasoning model`}/;

test("each Ollama reasoning model checkbox has an accessible name", () => {
  assert.match(source, reasoningModelCheckbox);
});

test("switching provider types clears draft reasoning state", () => {
  const start = source.indexOf("if (editingProviderId) return;");
  const end = source.indexOf("if (isCustomProviderType(value))", start);
  assert.ok(start >= 0 && end > start);
  const providerTypeChange = source.slice(start, end);
  assert.match(providerTypeChange, /setIsReasoningModel\(false\)/);
  assert.match(providerTypeChange, /setReasoningModelIds\(\[\]\)/);
});
