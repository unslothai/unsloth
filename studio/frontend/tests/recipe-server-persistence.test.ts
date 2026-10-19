// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Call = { url: string; method: string; body: unknown };

function fakeServer(respond: (call: Call) => number) {
  const calls: Call[] = [];
  const release: (() => void)[] = [];
  const authFetch = (url: string, init: RequestInit = {}) => {
    const call = {
      url,
      method: init.method ?? "GET",
      body: init.body ? JSON.parse(String(init.body)) : null,
    };
    calls.push(call);
    const status = respond(call);
    return new Promise<Response>((resolve) => {
      release.push(() =>
        resolve(new Response(status === 204 ? null : "{}", { status })),
      );
    });
  };
  const flush = async () => {
    while (release.length) {
      release.shift()?.();
      await new Promise((r) => setTimeout(r, 0));
    }
  };
  return { calls, authFetch, flush };
}

test("a run saved many times sends one write in flight and then only the newest", async () => {
  const server = fakeServer(() => 204);
  const { saveRecipeExecution } = loadWithStubs<{
    saveRecipeExecution: (record: Record<string, unknown>) => Promise<void>;
  }>(
    new URL(
      "../src/features/recipe-studio/data/executions-db.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: server.authFetch,
        getAuthSessionEpoch: () => 1,
      },
    },
  );

  const save = (done: number) =>
    saveRecipeExecution({ id: "e1", recipeId: "r1", done });
  const saves = [save(1)];
  await new Promise((r) => setTimeout(r, 0));
  saves.push(save(2), save(3), save(4));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(server.calls.length, 1);
  await server.flush();
  await Promise.all(saves);

  assert.deepEqual(
    server.calls.map((c) => [
      c.method,
      c.url,
      (c.body as { done: number }).done,
    ]),
    [
      ["PUT", "/api/data-recipe/recipes/r1/executions/e1", 1],
      ["PUT", "/api/data-recipe/recipes/r1/executions/e1", 4],
    ],
  );
});

test("every coalesced caller hears about the failed write that carried its record", async () => {
  let status = 204;
  const server = fakeServer(() => status);
  const { saveRecipeExecution } = loadWithStubs<{
    saveRecipeExecution: (record: Record<string, unknown>) => Promise<void>;
  }>(
    new URL(
      "../src/features/recipe-studio/data/executions-db.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: server.authFetch,
        getAuthSessionEpoch: () => 1,
      },
    },
  );
  const first = saveRecipeExecution({ id: "e1", recipeId: "r1", done: 1 });
  const later = [2, 3].map((done) =>
    saveRecipeExecution({ id: "e1", recipeId: "r1", done }),
  );
  const settled = Promise.allSettled([first, ...later]);
  status = 422;
  await server.flush();
  const outcomes = await settled;
  assert.deepEqual(
    outcomes.map((o) => o.status),
    ["fulfilled", "rejected", "rejected"],
  );
});

test("a transient failure retries the newest snapshot instead of dropping it", async () => {
  const statuses = [503, 204];
  const server = fakeServer(() => statuses.shift() ?? 204);
  const { saveRecipeExecution } = loadWithStubs<{
    saveRecipeExecution: (record: Record<string, unknown>) => Promise<void>;
  }>(
    new URL(
      "../src/features/recipe-studio/data/executions-db.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: server.authFetch,
        getAuthSessionEpoch: () => 1,
      },
    },
  );
  const saved = saveRecipeExecution({ id: "e1", recipeId: "r1", done: 9 });
  await server.flush();
  await new Promise((r) => setTimeout(r, 1100));
  await server.flush();
  await saved;
  assert.deepEqual(
    server.calls.map((c) => (c.body as { done: number }).done),
    [9, 9],
  );
});

test("legacy import isolates a rejected record, keeps the rest, and runs once", async () => {
  const storage = new Map<string, string>();
  globalThis.localStorage = {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => void storage.set(k, v),
  } as Storage;
  const rejected = "bad";
  const imported: string[] = [];
  const recipeRequest = (_path: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<
      string,
      { id: string }[]
    >;
    const rows = body.recipes ?? body.executions;
    if (rows.some((row) => row.id === rejected)) {
      return Promise.reject(new RecipeApiError("too large", 422));
    }
    imported.push(...rows.map((row) => row.id));
    return Promise.resolve({});
  };
  class RecipeApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  }
  const stores: Record<string, Record<string, unknown>[]> = {
    "unsloth-data-recipes": [
      { id: "a", name: "A", payload: {}, createdAt: 1, updatedAt: 2 },
      { id: rejected, name: "Bad", payload: {}, createdAt: 1, updatedAt: 2 },
      { id: "no-payload", name: "Skipped client-side" },
    ],
    "unsloth-data-recipe-executions": [
      { id: "e1", recipeId: "a", createdAt: 1 },
    ],
  };
  class FakeDexie {
    static exists = (name: string) => Promise.resolve(name in stores);
    tables = [{ name: "recipes" }, { name: "executions" }];
    dbName: string;
    constructor(dbName: string) {
      this.dbName = dbName;
    }
    open = () => Promise.resolve();
    close = () => undefined;
    table = () => ({ toArray: () => Promise.resolve(stores[this.dbName]) });
  }
  const { importLegacyRecipes } = loadWithStubs<{
    importLegacyRecipes: () => Promise<void>;
  }>(
    new URL(
      "../src/features/data-recipes/data/legacy-import.ts",
      import.meta.url,
    ),
    {
      "@/lib/account-transition": {
        accountDatabaseName: (name: string) => name,
      },
      "@/utils": {
        normalizeNonEmptyName: (name: string) => name.trim() || "Unnamed",
      },
      dexie: { default: FakeDexie, __esModule: true },
      "./recipes-api": { recipeRequest, RecipeApiError },
    },
  );

  const warn = console.warn;
  console.warn = () => undefined;
  try {
    await importLegacyRecipes();
    await importLegacyRecipes();
  } finally {
    console.warn = warn;
  }
  assert.deepEqual(imported, ["a", "e1"]);
  assert.ok(storage.get("unsloth-data-recipes:server-import.v1"));
});

test("an auth or throttling failure leaves the legacy import to retry", async () => {
  const storage = new Map<string, string>();
  globalThis.localStorage = {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => void storage.set(k, v),
  } as Storage;
  class RecipeApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  }
  let calls = 0;
  const recipeRequest = () => {
    calls += 1;
    return Promise.reject(new RecipeApiError("Unauthorized", 401));
  };
  class FakeDexie {
    static exists = () => Promise.resolve(true);
    tables = [{ name: "recipes" }];
    open = () => Promise.resolve();
    close = () => undefined;
    table = () => ({
      toArray: () =>
        Promise.resolve([
          { id: "a", name: "A", payload: {}, createdAt: 1, updatedAt: 2 },
          { id: "b", name: "B", payload: {}, createdAt: 1, updatedAt: 2 },
        ]),
    });
  }
  const { importLegacyRecipes } = loadWithStubs<{
    importLegacyRecipes: () => Promise<void>;
  }>(
    new URL(
      "../src/features/data-recipes/data/legacy-import.ts",
      import.meta.url,
    ),
    {
      "@/lib/account-transition": {
        accountDatabaseName: (name: string) => name,
      },
      "@/utils": { normalizeNonEmptyName: (name: string) => name },
      dexie: { default: FakeDexie, __esModule: true },
      "./recipes-api": { recipeRequest, RecipeApiError },
    },
  );
  const error = console.error;
  console.error = () => undefined;
  try {
    await importLegacyRecipes();
    await importLegacyRecipes();
  } finally {
    console.error = error;
  }
  assert.equal(calls, 2);
  assert.equal(storage.get("unsloth-data-recipes:server-import.v1"), undefined);
});
