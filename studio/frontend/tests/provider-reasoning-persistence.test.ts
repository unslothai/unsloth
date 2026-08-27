// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const { loadExternalProviders } = await import(
  "../src/features/chat/external-providers.ts"
);

const storedProvider = {
  id: "ollama-local",
  providerType: "ollama",
  name: "Ollama",
  baseUrl: "http://127.0.0.1:11434/v1",
  models: ["reasoning-model", "instruct-model"],
  isReasoningModel: true,
  createdAt: 1,
  updatedAt: 1,
};

test("a legacy Ollama reasoning flag pins every enabled model", () => {
  store.set("unsloth_chat_external_providers", JSON.stringify([storedProvider]));
  const [loaded] = loadExternalProviders();
  assert.deepEqual(loaded.reasoningModelIds, storedProvider.models);
});

test("a malformed Ollama pin list fails closed", () => {
  store.set(
    "unsloth_chat_external_providers",
    JSON.stringify([{ ...storedProvider, reasoningModelIds: "reasoning-model" }]),
  );
  const [loaded] = loadExternalProviders();
  assert.deepEqual(loaded.reasoningModelIds, []);
});
