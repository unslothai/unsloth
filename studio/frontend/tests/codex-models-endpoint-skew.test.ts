// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// New frontend against an OLD backend that has no /api/providers/{id}/codex/models.
// Three shapes that skew actually produces -- 404 JSON, a 200 SPA index.html, and a
// 401 from an auth gateway -- must all land on the curated seed with the saved
// selection intact. A wipe here is not cosmetic: the next unrelated Save persists the
// emptied picker and the connection loses models the account can still reach.

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

/** What applyCodexSubscriptionModels does with the call: any throw degrades to null. */
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
  // A dev proxy or a single-page fallback answers an unknown /api path with 200 and
  // the app shell. response.json() rejects and parseJsonOrThrow hands back null on an
  // ok response, so the picker must read that the same way it reads a throw.
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
  // An intermediate backend that grew the route before the source discriminator would
  // otherwise be read as authoritative and retire every saved slug it omits.
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
  // Kept last: authFetch reads every 401 as an expired Unsloth session and runs the
  // refresh-and-retry path, which is why the backend answers a dead ChatGPT connection
  // with 200 + source:"reauthorization_required" instead of a 401. Whatever that path
  // decides, the picker must still land on the seed with the selection intact.
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
    // The session-expiry path may also navigate; either way it must not have
    // rewritten the picker.
    await new Promise((resolve) => setTimeout(resolve, 50));
  } finally {
    restore();
    globals.window = originalWindow;
    globals.localStorage = originalStorage;
  }
});
