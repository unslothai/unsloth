// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { mergeLocalProviderOptions } = await import(
  "../src/features/chat/sync-external-providers.ts"
);

const provider = {
  id: "ollama-local",
  providerType: "ollama",
  name: "Ollama",
  baseUrl: "http://127.0.0.1:11434/v1",
  models: ["reasoning-model", "removed-model"],
  isReasoningModel: true,
  reasoningModelIds: ["reasoning-model", "removed-model"],
  createdAt: 1,
  updatedAt: 1,
};

test("provider sync drops reasoning pins for removed models", () => {
  const merged = mergeLocalProviderOptions(provider, {
    ...provider,
    models: ["reasoning-model"],
    reasoningModelIds: undefined,
  });
  assert.deepEqual(merged.reasoningModelIds, ["reasoning-model"]);
});

test("provider sync preserves an explicit empty reasoning pin list", () => {
  const merged = mergeLocalProviderOptions(
    { ...provider, reasoningModelIds: [] },
    { ...provider, models: ["reasoning-model"] },
  );
  assert.deepEqual(merged.reasoningModelIds, []);
});
