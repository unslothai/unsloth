// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

interface SubscriptionModels {
  models: { id: string; vision?: boolean | null }[];
  known?: { id: string; vision?: boolean | null }[];
  source: "subscription" | "curated" | "reauthorization_required";
}

type Resolve = (
  curated: string[],
  savedModels: string[],
  listed: SubscriptionModels | null,
) => { catalog: string[]; selected: string[] };

let vite: ViteDevServer;
let resolveCodexPickerModels: Resolve;

const CURATED = ["gpt-5.4", "gpt-5.5"];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/chat-providers-dialog.tsx",
  );
  resolveCodexPickerModels = loaded.resolveCodexPickerModels as Resolve;
});

after(async () => {
  await vite.close();
});

test("a plan catalog drives the picker and retires slugs it no longer lists", () => {
  const { catalog, selected } = resolveCodexPickerModels(
    CURATED,
    ["gpt-5.4", "gpt-5.7-nova"],
    { models: [{ id: "gpt-5.7-nova" }], source: "subscription" },
  );
  assert.deepEqual(catalog, ["gpt-5.7-nova"]);
  assert.deepEqual(selected, ["gpt-5.7-nova"]);
});

test("a curated fallback keeps every saved model selected", () => {
  // The backend answers with the seed when it cannot reach upstream. Dropping the
  // saved dynamic slug here would lose it on the next unrelated save.
  const { catalog, selected } = resolveCodexPickerModels(
    CURATED,
    ["gpt-5.4", "gpt-5.7-nova"],
    { models: CURATED.map((id) => ({ id })), source: "curated" },
  );
  assert.ok(catalog.includes("gpt-5.7-nova"));
  assert.deepEqual(selected, ["gpt-5.4", "gpt-5.7-nova"]);
});

test("an unreachable backend keeps every saved model selected", () => {
  const { selected } = resolveCodexPickerModels(
    CURATED,
    ["gpt-5.4", "gpt-5.7-nova"],
    null,
  );
  assert.deepEqual(selected, ["gpt-5.4", "gpt-5.7-nova"]);
});

test("an empty plan catalog is not treated as an empty account", () => {
  const { selected } = resolveCodexPickerModels(
    CURATED,
    ["gpt-5.4"],
    { models: [], source: "subscription" },
  );
  assert.deepEqual(selected, ["gpt-5.4"]);
});

interface RegistryEntry {
  model_capabilities?: Record<string, { vision?: boolean; studio_tools?: boolean }>;
  supports_studio_tools?: boolean;
}

type Capabilities = (
  entry: RegistryEntry | undefined,
  listed: (SubscriptionModels & { models: { id: string; vision?: boolean | null }[] }) | null,
  stored: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
) => Record<string, { vision?: boolean; studio_tools?: boolean }> | null;

const ENTRY: RegistryEntry = {
  model_capabilities: { "gpt-5.4": { vision: true, studio_tools: true } },
  supports_studio_tools: true,
};

test("a plan-listed slug carries its own vision flag into the capability map", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const codexCapabilitiesWithPlanModels = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  const capabilities = codexCapabilitiesWithPlanModels(
    ENTRY,
    {
      source: "subscription",
      models: [{ id: "gpt-5.7-nova", vision: false }, { id: "gpt-5.7-eos", vision: true }],
    },
    undefined,
  );
  // Without this the composer reads "unknown" as allowed and offers attachments the
  // backend refuses on every send.
  assert.equal(capabilities?.["gpt-5.7-nova"].vision, false);
  assert.equal(capabilities?.["gpt-5.7-eos"].vision, true);
  // The registry's own entries survive untouched.
  assert.deepEqual(capabilities?.["gpt-5.4"], { vision: true, studio_tools: true });
});

test("a curated fallback never rewrites the capability map", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const codexCapabilitiesWithPlanModels = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  assert.equal(
    codexCapabilitiesWithPlanModels(
      ENTRY,
      { source: "curated", models: [{ id: "gpt-5.7-nova", vision: true }] },
      undefined,
    ),
    null,
  );
  assert.equal(codexCapabilitiesWithPlanModels(ENTRY, null, undefined), null);
});

test("a plan that describes nothing new never overrides the registry", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const codexCapabilitiesWithPlanModels = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  const capabilities = codexCapabilitiesWithPlanModels(
    ENTRY,
    { source: "subscription", models: [{ id: "gpt-5.4", vision: false }, { id: "gpt-5.7-nova" }] },
    undefined,
  );
  assert.deepEqual(capabilities?.["gpt-5.4"], { vision: true, studio_tools: true });
  // No modality list normalizes to null upstream and the backend gate is bool(vision),
  // so the UI has to say false too or it offers attachments that every send refuses.
  assert.deepEqual(capabilities?.["gpt-5.7-nova"], { vision: false });
});


test("another connection's learned slug survives this one's catalog", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const codexCapabilitiesWithPlanModels = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  // The map is keyed by provider type, so a second ChatGPT connection's catalog must
  // not erase what the first one taught it.
  const capabilities = codexCapabilitiesWithPlanModels(
    ENTRY,
    { source: "subscription", models: [{ id: "gpt-5.7-eos", vision: true }] },
    { "gpt-5.7-nova": { vision: false } },
  );
  assert.deepEqual(capabilities?.["gpt-5.7-nova"], { vision: false });
  assert.deepEqual(capabilities?.["gpt-5.7-eos"], { vision: true });
});


test("a saved slug the plan still returns survives losing its picker slot", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const resolve = loaded.resolveCodexPickerModels as Resolve;
  // "hide" retires a model from the picker; it does not revoke one already in use, and
  // dropping it here would make the next save delete it from the connection.
  const { catalog, selected } = resolve(
    CURATED,
    ["gpt-5.7-nova"],
    {
      models: [{ id: "gpt-5.4" }],
      known: [{ id: "gpt-5.4" }, { id: "gpt-5.7-nova" }],
      source: "subscription",
    },
  );
  assert.deepEqual(selected, ["gpt-5.7-nova"]);
  assert.ok(catalog.includes("gpt-5.7-nova"));
});

test("a slug the plan no longer returns at all is retired", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const resolve = loaded.resolveCodexPickerModels as Resolve;
  const { selected } = resolve(
    CURATED,
    ["gpt-5.6-sol"],
    { models: [{ id: "gpt-5.4" }], known: [{ id: "gpt-5.4" }], source: "subscription" },
  );
  assert.deepEqual(selected, []);
});


test("a hidden saved slug still contributes its capability", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const capabilities = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  // Hidden slugs stay selectable, so a fresh browser has to learn their modalities from
  // here or the composer guesses and offers what the chat route refuses.
  const resolved = capabilities(
    ENTRY,
    {
      source: "subscription",
      models: [{ id: "gpt-5.4", vision: true }],
      known: [{ id: "gpt-5.4", vision: true }, { id: "gpt-5.7-nova", vision: false }],
    },
    undefined,
  );
  assert.deepEqual(resolved?.["gpt-5.7-nova"], { vision: false });
});


test("a reauthorization answer retires nothing and describes nothing", async () => {
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const resolve = loaded.resolveCodexPickerModels as Resolve;
  const capabilities = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  // It carries the seed, but the connection is dead: nothing about it is authoritative.
  const { selected } = resolve(
    CURATED,
    ["gpt-5.4", "gpt-5.7-nova"],
    { models: CURATED.map((id) => ({ id })), source: "reauthorization_required" },
  );
  assert.deepEqual(selected, ["gpt-5.4", "gpt-5.7-nova"]);
  assert.equal(
    capabilities(ENTRY, {
      source: "reauthorization_required",
      models: [{ id: "gpt-5.7-nova", vision: true }],
    }, undefined),
    null,
  );
});

test("a catalog that arrives before the registry does not rewrite what is learned", async () => {
  // registryByType is empty until the mount fetch resolves, and stays empty when it
  // fails, while the Edit button is gated only on a pending mutation. The plan catalog
  // can therefore land with no registry row behind it. Writing then would drop the
  // wildcard that carries studio_tools for the whole provider type and overwrite the
  // registry's own vision flags with the plan's, and both are persisted.
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  const codexCapabilitiesWithPlanModels = loaded.codexCapabilitiesWithPlanModels as Capabilities;
  const stored = {
    "gpt-5.4": { vision: true, studio_tools: true },
    "*": { studio_tools: true },
  };
  const capabilities = codexCapabilitiesWithPlanModels(
    undefined,
    {
      source: "subscription",
      models: [{ id: "gpt-5.4", vision: false }],
    },
    stored,
  );
  assert.equal(capabilities, null);
});
