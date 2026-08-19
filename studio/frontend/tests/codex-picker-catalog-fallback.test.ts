// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

interface SubscriptionModels {
  models: { id: string }[];
  source: "subscription" | "curated";
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
