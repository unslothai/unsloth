// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";

interface SubscriptionModels {
  models: { id: string }[];
  known?: { id: string }[];
  source: "subscription" | "curated" | "reauthorization_required";
}

type WithNewlyOffered = (
  savedModels: string[],
  previouslyOffered: string[],
  listed: SubscriptionModels | null,
) => { models: string[]; availableModels: string[]; added: string[] } | null;

type WithNewlyListed = (
  savedModels: string[],
  previouslyListed: string[],
  listedIds: string[],
  options?: { knownIds?: string[]; maxAdded?: number },
) => { models: string[]; availableModels: string[]; added: string[] } | null;

type Connection = {
  providerType: string;
  authKind?: string;
  sharedByOwner?: boolean;
};

let vite: ViteDevServer;
let withNewlyOfferedCodexModels: WithNewlyOffered;
let withNewlyListedModels: WithNewlyListed;
let autoTicksNewlyListedModels: (provider: Connection) => boolean;
let maxAutoTicked: number;

const plan = (...ids: string[]): SubscriptionModels => ({
  models: ids.map((id) => ({ id })),
  source: "subscription",
});

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  withNewlyOfferedCodexModels =
    loaded.withNewlyOfferedCodexModels as WithNewlyOffered;
  withNewlyListedModels = loaded.withNewlyListedModels as WithNewlyListed;
  autoTicksNewlyListedModels = loaded.autoTicksNewlyListedModels as (
    provider: Connection,
  ) => boolean;
  maxAutoTicked = loaded.MAX_AUTO_TICKED_MODELS as number;
});

after(async () => {
  await vite.close();
});

test("a model the plan starts offering after connect is ticked and saved", () => {
  assert.deepEqual(
    withNewlyOfferedCodexModels(
      ["gpt-6-sol", "gpt-6-luna"],
      ["gpt-6-sol", "gpt-6-luna"],
      plan("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"),
    ),
    {
      models: ["gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"],
      availableModels: ["gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"],
      added: ["gpt-6.1-sol"],
    },
  );
});

test("a model offered before and unticked by the user stays unticked", () => {
  assert.equal(
    withNewlyOfferedCodexModels(
      ["gpt-6-sol"],
      ["gpt-6-sol", "gpt-6-luna"],
      plan("gpt-6-sol", "gpt-6-luna"),
    ),
    null,
  );
  const update = withNewlyOfferedCodexModels(
    ["gpt-6-sol"],
    ["gpt-6-sol", "gpt-6-luna"],
    plan("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"),
  );
  assert.deepEqual(update?.models, ["gpt-6-sol", "gpt-6.1-sol"]);
});

test("saved models the plan no longer knows are dropped alongside the new ticks", () => {
  const update = withNewlyOfferedCodexModels(
    ["gpt-5.4", "gpt-6-sol"],
    ["gpt-5.4", "gpt-6-sol"],
    plan("gpt-6.1-sol", "gpt-6-sol"),
  );
  assert.deepEqual(update?.models, ["gpt-6-sol", "gpt-6.1-sol"]);
});

test("no plan catalog or no saved baseline changes nothing", () => {
  const saved = ["gpt-6-sol"];
  assert.equal(
    withNewlyOfferedCodexModels(saved, saved, {
      models: [{ id: "gpt-6.1-sol" }],
      source: "curated",
    }),
    null,
  );
  assert.equal(withNewlyOfferedCodexModels(saved, saved, null), null);
  assert.equal(
    withNewlyOfferedCodexModels(saved, [], plan("gpt-6.1-sol", "gpt-6-sol")),
    null,
  );
  assert.equal(
    withNewlyOfferedCodexModels([], saved, plan("gpt-6.1-sol", "gpt-6-sol")),
    null,
  );
});

test("applying the result again is a no-op", () => {
  const listed = plan("gpt-6.1-sol", "gpt-6-sol");
  const first = withNewlyOfferedCodexModels(
    ["gpt-6-sol"],
    ["gpt-6-sol"],
    listed,
  );
  assert.ok(first);
  assert.equal(
    withNewlyOfferedCodexModels(first.models, first.availableModels, listed),
    null,
  );
});

test("an API connection ticks a model its provider newly lists and keeps unlisted saved ones", () => {
  assert.deepEqual(
    withNewlyListedModels(
      ["claude-opus-4-7", "my-pinned-id"],
      ["claude-opus-4-7", "claude-haiku-4-5"],
      ["claude-opus-5", "claude-opus-4-7", "claude-haiku-4-5"],
      { maxAdded: maxAutoTicked },
    ),
    {
      models: ["claude-opus-4-7", "my-pinned-id", "claude-opus-5"],
      availableModels: [
        "claude-opus-5",
        "claude-opus-4-7",
        "claude-haiku-4-5",
        "my-pinned-id",
      ],
      added: ["claude-opus-5"],
    },
  );
});

test("a jump past the limit adopts the listing as the baseline without ticking", () => {
  const listed = ["a", "b", "c", "d", "e", "f"];
  const update = withNewlyListedModels(["a"], ["a"], listed, {
    maxAdded: maxAutoTicked,
  });
  assert.deepEqual(update, {
    models: ["a"],
    availableModels: listed,
    added: [],
  });
  assert.ok(update);
  assert.equal(
    withNewlyListedModels(update.models, update.availableModels, listed, {
      maxAdded: maxAutoTicked,
    }),
    null,
  );
});

test("gateways, custom endpoints and local servers are not auto ticked", () => {
  for (const providerType of ["anthropic", "openai", "gemini"]) {
    assert.equal(
      autoTicksNewlyListedModels({ providerType }),
      true,
      providerType,
    );
  }
  for (const providerType of [
    "openrouter",
    "huggingface",
    "llama_cpp",
    "ollama",
    "vllm",
  ]) {
    assert.equal(
      autoTicksNewlyListedModels({ providerType }),
      false,
      providerType,
    );
  }
  assert.equal(
    autoTicksNewlyListedModels({
      providerType: "anthropic",
      sharedByOwner: true,
    }),
    false,
  );
  assert.equal(
    autoTicksNewlyListedModels({
      providerType: "openai_codex",
      authKind: "chatgpt_oauth",
    }),
    false,
  );
});
