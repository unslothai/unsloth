// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Old backends without the codex/models route must land on the seed with the selection intact.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

interface SubscriptionModels {
  models: { id: string; vision?: boolean | null }[];
  known?: { id: string; vision?: boolean | null }[];
  source: "subscription" | "curated" | "reauthorization_required";
}

type Fetch = (
  providerId: string,
  options?: { refresh?: boolean },
) => Promise<SubscriptionModels>;
type Resolve = (
  curated: string[],
  savedModels: string[],
  listed: SubscriptionModels | null,
) => { catalog: string[]; selected: string[] };

let vite: ViteDevServer;
let fetchCodexSubscriptionModels: Fetch;
let resolveCodexPickerModels: Resolve;

const CURATED = ["gpt-5.4", "gpt-5.5"];
const SAVED = ["gpt-5.4", "gpt-5.7-nova"];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const api = await vite.ssrLoadModule("/src/features/chat/api/providers-api.ts");
  fetchCodexSubscriptionModels = api.fetchCodexSubscriptionModels as Fetch;
  const dialog = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  resolveCodexPickerModels = dialog.resolveCodexPickerModels as Resolve;
});

after(async () => {
  await vite.close();
});

function stubFetch(response: Response): () => void {
  const original = globalThis.fetch;
  globalThis.fetch = (async () => response.clone()) as typeof globalThis.fetch;
  return () => {
    globalThis.fetch = original;
  };
}

async function listedOrNull(providerId: string): Promise<SubscriptionModels | null> {
  try {
    return await fetchCodexSubscriptionModels(providerId);
  } catch {
    return null;
  }
}

test("an old backend's 404 degrades to the curated seed and keeps the selection", async () => {
  const restore = stubFetch(
    new Response(JSON.stringify({ detail: "Not Found" }), {
      status: 404,
      headers: { "content-type": "application/json" },
    }),
  );
  try {
    const listed = await listedOrNull("provider-1");
    assert.equal(listed, null);
    const { catalog, selected } = resolveCodexPickerModels(CURATED, SAVED, listed);
    assert.deepEqual(selected, SAVED);
    for (const model of CURATED) assert.ok(catalog.includes(model));
  } finally {
    restore();
  }
});

test("an old backend's SPA index.html degrades to the curated seed", async () => {
  const restore = stubFetch(
    new Response("<!doctype html><html><body></body></html>", {
      status: 200,
      headers: { "content-type": "text/html" },
    }),
  );
  try {
    const listed = await listedOrNull("provider-1");
    assert.equal(listed, null);
    const { selected } = resolveCodexPickerModels(CURATED, SAVED, listed);
    assert.deepEqual(selected, SAVED);
  } finally {
    restore();
  }
});

test("a body without a source field is not mistaken for a plan catalog", async () => {
  const restore = stubFetch(
    new Response(JSON.stringify({ models: [{ id: "gpt-5.4" }] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
  try {
    const listed = await fetchCodexSubscriptionModels("provider-1");
    assert.notEqual(listed?.source, "subscription");
    const { selected } = resolveCodexPickerModels(CURATED, SAVED, listed);
    assert.deepEqual(selected, SAVED);
  } finally {
    restore();
  }
});

test("a gateway 401 on the unknown path still keeps the selection", async () => {
  // Kept last: authFetch treats every 401 as an expired session and runs refresh-and-retry.
  const location = { pathname: "/chat", href: "/chat" };
  const globals = globalThis as { window?: unknown; localStorage?: unknown };
  const originalWindow = globals.window;
  const originalStorage = globals.localStorage;
  const store = new Map<string, string>();
  globals.localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, String(value)),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
    key: () => null,
    length: 0,
  };
  globals.window = { location, localStorage: globals.localStorage };
  const restore = stubFetch(
    new Response(JSON.stringify({ detail: "Unauthorized" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    }),
  );
  try {
    const listed = await listedOrNull("provider-1");
    assert.equal(listed, null);
    const { selected } = resolveCodexPickerModels(CURATED, SAVED, listed);
    assert.deepEqual(selected, SAVED);
    await new Promise((resolve) => setTimeout(resolve, 50));
  } finally {
    restore();
    globals.window = originalWindow;
    globals.localStorage = originalStorage;
  }
});
