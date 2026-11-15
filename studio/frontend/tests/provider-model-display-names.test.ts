// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";

type Learn = (
  providerType: string,
  models: { id: string; display_name?: string | null }[],
) => void;
type Name = (
  providerType: string | null | undefined,
  modelId: string,
) => string;

let vite: ViteDevServer;
let learn: Learn;
let name: Name;
let compareName: (id: string) => string;
let setProviders: (providers: unknown[]) => void;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const names = await vite.ssrLoadModule(
    "/src/features/chat/model-display-names.ts",
  );
  learn = names.learnModelDisplayNames as Learn;
  name = names.providerModelDisplayName as Name;
  const labels = await vite.ssrLoadModule(
    "/src/features/chat/lib/external-model-label.ts",
  );
  compareName = labels.compareModelDisplayName as (id: string) => string;
  const store = await vite.ssrLoadModule(
    "/src/features/chat/stores/external-providers-store.ts",
  );
  setProviders = (providers) =>
    store.useExternalProvidersStore.setState({ providers });
});

after(async () => {
  await vite.close();
});

test("a listed provider name labels the id; unknown ids show as they are", () => {
  learn("anthropic", [
    { id: "claude-opus-5-5", display_name: "Claude Opus 5.5" },
    { id: "claude-haiku-5-5", display_name: "claude-haiku-5-5" },
  ]);
  assert.equal(name("anthropic", "claude-opus-5-5"), "Claude Opus 5.5");
  assert.equal(name("anthropic", "claude-haiku-5-5"), "claude-haiku-5-5");
  assert.equal(name("anthropic", "claude-unknown"), "claude-unknown");
  assert.equal(name("openai", "claude-opus-5-5"), "claude-opus-5-5");
  assert.equal(name(undefined, "claude-opus-5-5"), "claude-opus-5-5");
});

test("a listing that no longer names a model drops the old name", () => {
  learn("gemini", [{ id: "gemini-3-pro", display_name: "Gemini 3 Pro" }]);
  learn("gemini", [{ id: "gemini-3-pro", display_name: "" }]);
  assert.equal(name("gemini", "gemini-3-pro"), "gemini-3-pro");
});

test("compare labels use the connection's name, else the id without its org prefix", () => {
  setProviders([
    { id: "conn-a", providerType: "anthropic" },
    { id: "conn-r", providerType: "openrouter" },
  ]);
  learn("anthropic", [
    { id: "claude-opus-5-5", display_name: "Claude Opus 5.5" },
  ]);
  assert.equal(
    compareName("external::conn-a::claude-opus-5-5"),
    "Claude Opus 5.5",
  );
  assert.equal(
    compareName(
      `external::conn-r::${encodeURIComponent("anthropic/claude-opus-5.5")}`,
    ),
    "claude-opus-5.5",
  );
  assert.equal(compareName("unsloth/Qwen3-8B-GGUF"), "Qwen3-8B-GGUF");
});
