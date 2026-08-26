// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend owns the remembered model; localStorage is a shadow so old
// bundles, old backends and dropped writes all still behave.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();

const LEGACY_KEY = "unsloth.last-local-model-load.v1";

const { readLastLocalModelLoad, recordLastLocalModelLoad } = await import(
  "../src/features/chat/utils/last-local-model-load.ts"
);

type Backend = { record: Record<string, unknown> | null; status?: number };

let backend: Backend = { record: null };
let puts: Record<string, unknown>[] = [];
let serverSkewMs = 0;

globalThis.fetch = (async (_input: string, init?: RequestInit) => {
  if (init?.signal?.aborted) {
    throw Object.assign(new Error("aborted"), { name: "AbortError" });
  }
  const status = backend.status ?? 200;
  const serverNow = Date.now() + serverSkewMs;
  if (status !== 200) {
    return { ok: false, status, json: async () => ({}) } as Response;
  }
  if (init?.method === "PUT") {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    puts.push(body);
    // Mirror the server: shift into its frame, then keep the newer stamp.
    const shifted =
      typeof body.loaded_at === "number" && typeof body.client_now === "number"
        ? body.loaded_at + (serverNow - body.client_now)
        : body.loaded_at;
    const current = backend.record;
    const beaten =
      current &&
      typeof current.loaded_at === "number" &&
      typeof shifted === "number" &&
      shifted < current.loaded_at;
    if (!beaten) {
      backend.record = { ...body, loaded_at: shifted };
      delete (backend.record as { client_now?: unknown }).client_now;
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({ ...(backend.record ?? {}), server_now: serverNow }),
    } as Response;
  }
  return {
    ok: true,
    status: 200,
    json: async () => ({ ...(backend.record ?? {}), server_now: serverNow }),
  } as Response;
}) as typeof fetch;

function reset(): void {
  store.clear();
  backend = { record: null };
  puts = [];
  serverSkewMs = 0;
}

function legacy(): Record<string, unknown> | null {
  const raw = store.get(LEGACY_KEY);
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : null;
}

test("a new frontend against an old backend falls back to the shadow", async () => {
  reset();
  backend.status = 404;
  recordLastLocalModelLoad({ id: "u/m", kind: "gguf", ggufVariant: "Q4" });
  const got = await readLastLocalModelLoad();
  assert.deepEqual(got, { id: "u/m", kind: "gguf", ggufVariant: "Q4" });
  // The write could not land, so the shadow stays flagged for a later sync.
  assert.equal(legacy()?.pendingSync, true);
});

test("the shadow is written synchronously so a teardown cannot lose it", () => {
  reset();
  // No await: the record must already be stored when this returns.
  recordLastLocalModelLoad({ id: "u/m", kind: "model", ggufVariant: null });
  assert.equal(legacy()?.id, "u/m");
  assert.equal(typeof legacy()?.loadedAt, "number");
});

test("the shadow keeps loadedAt so a still-open old tab can still read it", () => {
  reset();
  recordLastLocalModelLoad({ id: "u/m", kind: "model", ggufVariant: null });
  // Pre-backend readers reject a record without a numeric loadedAt.
  assert.equal(typeof legacy()?.loadedAt, "number");
});

test("a newer backend record wins over an older shadow", async () => {
  reset();
  const now = Date.now();
  store.set(
    LEGACY_KEY,
    JSON.stringify({ id: "old", kind: "model", ggufVariant: null, loadedAt: now - 60_000 }),
  );
  backend.record = { id: "new", kind: "model", gguf_variant: null, loaded_at: now };
  const got = await readLastLocalModelLoad();
  assert.equal(got?.id, "new");
});

test("a newer shadow wins over an older backend record and is re-synced", async () => {
  reset();
  const now = Date.now();
  store.set(
    LEGACY_KEY,
    JSON.stringify({ id: "fresh", kind: "model", ggufVariant: null, loadedAt: now }),
  );
  backend.record = { id: "stale", kind: "model", gguf_variant: null, loaded_at: now - 60_000 };
  const got = await readLastLocalModelLoad();
  assert.equal(got?.id, "fresh");
  // Re-issued so the server stops handing the stale one to other surfaces.
  assert.equal(puts.at(-1)?.id, "fresh");
});

test("a stale shadow loses to an unstamped backend record", async () => {
  reset();
  store.set(
    LEGACY_KEY,
    JSON.stringify({
      id: "shadow",
      kind: "model",
      ggufVariant: null,
      loadedAt: Date.now() - 60_000,
    }),
  );
  // Written by a pre-loaded_at client, so there is no stamp to compare.
  backend.record = { id: "backend", kind: "model", gguf_variant: null, loaded_at: null };
  const got = await readLastLocalModelLoad();
  assert.equal(got?.id, "backend");
});

test("a pending shadow beats an unstamped backend record", async () => {
  reset();
  store.set(
    LEGACY_KEY,
    JSON.stringify({
      id: "pending",
      kind: "model",
      ggufVariant: null,
      loadedAt: Date.now(),
      pendingSync: true,
    }),
  );
  backend.record = { id: "backend", kind: "model", gguf_variant: null, loaded_at: null };
  assert.equal((await readLastLocalModelLoad())?.id, "pending");
});

for (const skew of [-86_400_000, 86_400_000]) {
  test(`a server clock ${skew < 0 ? "behind" : "ahead"} by a day still orders correctly`, async () => {
    reset();
    serverSkewMs = skew;
    const now = Date.now();
    store.set(
      LEGACY_KEY,
      JSON.stringify({ id: "fresh", kind: "model", ggufVariant: null, loadedAt: now }),
    );
    // The same instant, expressed in the server's frame.
    backend.record = {
      id: "stale",
      kind: "model",
      gguf_variant: null,
      loaded_at: now + skew - 60_000,
    };
    assert.equal((await readLastLocalModelLoad())?.id, "fresh");
  });
}

test("a corrupt shadow is ignored rather than thrown", async () => {
  reset();
  store.set(LEGACY_KEY, "{not json");
  assert.equal(await readLastLocalModelLoad(), null);
});

test("a gguf record with no variant and a repo id is rejected", async () => {
  reset();
  backend.record = { id: "u/m", kind: "gguf", gguf_variant: null, loaded_at: Date.now() };
  // Names no file to load, so it cannot be acted on.
  assert.equal(await readLastLocalModelLoad(), null);
});

test("a gguf record with no variant but a filesystem path is kept", async () => {
  reset();
  backend.record = {
    id: "/models/m.gguf",
    kind: "gguf",
    gguf_variant: null,
    loaded_at: Date.now(),
  };
  assert.equal((await readLastLocalModelLoad())?.id, "/models/m.gguf");
});

test("an aborted read rejects rather than falling back", async () => {
  reset();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(() => readLastLocalModelLoad(controller.signal));
});
