// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

let vite: ViteDevServer;
before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true, hmr: false },
  });
});
after(async () => {
  await vite.close();
});

test("reload preference defaults off, only applies to llama.cpp, and survives backend sync", async () => {
  const { mergeLocalProviderOptions } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  assert.equal(
    mergeLocalProviderOptions(undefined, { providerType: "llama_cpp" })
      .autoReloadModels,
    undefined,
  );
  assert.equal(
    mergeLocalProviderOptions(
      { autoReloadModels: true },
      { providerType: "llama_cpp" },
    ).autoReloadModels,
    true,
  );
  assert.equal(
    mergeLocalProviderOptions(
      { autoReloadModels: true },
      { providerType: "ollama" },
    ).autoReloadModels,
    undefined,
  );
});

test("monitor refreshes once per connection, retains selections, retries failures and ignores retired requests", async () => {
  const storage = new Map<string, string>([
    ["unsloth_auth_token", "test-token"],
  ]);
  const originalWindow = globalThis.window;
  const originalStorage = globalThis.localStorage;
  const originalFetch = globalThis.fetch;
  let storageListener: ((event: Partial<StorageEvent>) => void) | undefined;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      addEventListener(type: string, listener: typeof storageListener) {
        if (type === "storage") storageListener = listener;
      },
      removeEventListener(type: string) {
        if (type === "storage") storageListener = undefined;
      },
    },
  });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    },
  });
  const { startLlamaCppAutoReload } = await vite.ssrLoadModule(
    "/src/features/chat/llama-cpp-auto-reload.ts",
  );
  const { useExternalProvidersStore: store } = await vite.ssrLoadModule(
    "/src/features/chat/stores/external-providers-store.ts",
  );
  let healthy = true;
  let models = ["kept", "disabled", "new"];
  let failSave = false;
  let failCatalog = false;
  let delayCatalog: Promise<void> | undefined;
  const counts = { probes: 0, catalogs: 0, saves: 0 };
  const savedConfigs = new Map<
    string,
    { models: string[]; available_models: string[] }
  >();
  const provider = {
    id: "llama",
    providerType: "llama_cpp",
    name: "llama.cpp",
    baseUrl: "http://localhost:8080/v1",
    models: ["kept", "removed"],
    availableModels: ["kept", "disabled", "removed"],
    createdAt: 1,
    updatedAt: 1,
    autoReloadModels: false,
  };
  globalThis.fetch = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ) => {
    const url = String(input);
    if (url.endsWith("/test")) {
      counts.probes++;
      return Response.json({ success: healthy });
    }
    if (url.endsWith("/models")) {
      counts.catalogs++;
      await delayCatalog;
      return failCatalog
        ? Response.json({ detail: "offline" }, { status: 502 })
        : Response.json(models.map((id) => ({ id })));
    }
    if (url.replace(/\/$/, "").endsWith("/providers")) {
      return Response.json(store.getState().providers.map((item: typeof provider) => {
        if (!savedConfigs.has(item.id)) {
          savedConfigs.set(item.id, { models: item.models, available_models: item.availableModels });
        }
        return { id: item.id, base_url: item.baseUrl, ...savedConfigs.get(item.id) };
      }));
    }
    if (init?.method === "PUT") {
      counts.saves++;
      if (failSave) return Response.json({ detail: "save failed" }, { status: 500 });
      const payload = JSON.parse(String(init.body));
      savedConfigs.set(url.split("/").at(-1)!, payload);
      return Response.json(payload);
    }
    throw new Error(`Unexpected request ${url}`);
  }) as typeof fetch;
  const waitFor = async (predicate: () => boolean) => {
    for (let attempt = 0; attempt < 200 && !predicate(); attempt++)
      await new Promise((resolve) => setTimeout(resolve, 5));
    assert.ok(predicate(), JSON.stringify(counts));
  };
  const pause = () => new Promise((resolve) => setTimeout(resolve, 45));
  const setEnabled = (value: boolean) =>
    store.getState().setProviders(
      store.getState().providers.map((p: typeof provider) => ({
        ...p,
        autoReloadModels: value,
      })),
    );
  store.getState().setProviders([provider]);
  store.getState().setConnectionsEnabled(true);
  const stop = startLlamaCppAutoReload(10);
  try {
    await pause();
    assert.equal(counts.probes, 0, "default-off connections must not poll");
    setEnabled(true);
    await waitFor(() => counts.saves === 1);
    assert.deepEqual(store.getState().providers[0].models, ["kept", "new"]);
    await pause();
    assert.equal(counts.catalogs, 1, "healthy probes must not reload again");
    healthy = false;
    await pause();
    assert.equal(counts.catalogs, 1);
    healthy = true;
    models = ["replacement"];
    failSave = true;
    await waitFor(() => counts.saves >= 2);
    assert.deepEqual(store.getState().providers[0].models, ["kept", "new"]);
    failSave = false;
    await waitFor(
      () => store.getState().providers[0].models[0] === "replacement",
    );
    const successfulCatalogs = counts.catalogs;
    await pause();
    assert.equal(counts.catalogs, successfulCatalogs);
    healthy = false;
    await pause();
    healthy = true;
    models = [];
    await pause();
    assert.deepEqual(store.getState().providers[0].models, ["replacement"]);
    failCatalog = true;
    models = ["after-outage"];
    await pause();
    assert.deepEqual(store.getState().providers[0].models, ["replacement"]);
    failCatalog = false;
    await waitFor(
      () => store.getState().providers[0].models[0] === "after-outage",
    );
    store.getState().setConnectionsEnabled(false);
    const disabledProbes = counts.probes;
    await pause();
    assert.equal(counts.probes, disabledProbes);
    let release!: () => void;
    delayCatalog = new Promise<void>((resolve) => {
      release = resolve;
    });
    const beforeDelay = counts.catalogs;
    store.getState().setConnectionsEnabled(true);
    await waitFor(() => counts.catalogs > beforeDelay);
    const beforeSave = counts.saves;
    setEnabled(false);
    models = ["late"];
    release();
    await pause();
    assert.equal(
      counts.saves,
      beforeSave,
      "disabling retires in-flight requests",
    );
    assert.deepEqual(store.getState().providers[0].models, ["after-outage"]);
    setEnabled(true);
    await waitFor(() => store.getState().providers[0].models[0] === "late");
    models = ["kept", "new"];
    store.getState().setProviders([
      {
        ...provider,
        id: "manual",
        autoReloadModels: true,
        models: ["kept", "manual-id"],
        availableModels: ["kept", "disabled"],
      },
    ]);
    await waitFor(() => store.getState().providers[0].models.includes("new"));
    assert.deepEqual(store.getState().providers[0].models, [
      "manual-id",
      "kept",
      "new",
    ]);
    models = ["only-remaining"];
    store.getState().setProviders([
      {
        ...provider,
        id: "replaced",
        autoReloadModels: true,
        models: ["gone"],
        availableModels: ["gone", "only-remaining"],
      },
    ]);
    await waitFor(
      () => store.getState().providers[0].models[0] === "only-remaining",
    );
    healthy = false;
    await pause();
    savedConfigs.set("replaced", {
      models: ["selected-elsewhere"],
      available_models: ["only-remaining", "selected-elsewhere"],
    });
    models = ["only-remaining", "selected-elsewhere", "new-from-reconnect"];
    healthy = true;
    await waitFor(() => store.getState().providers[0].models.includes("new-from-reconnect"));
    assert.deepEqual(store.getState().providers[0].models, [
      "selected-elsewhere", "new-from-reconnect",
    ], "a completed save in another tab must not resurrect a disabled model");
    healthy = false;
    await pause();
    const savesBeforeLocalSync = counts.saves;
    store.getState().setProviders(store.getState().providers.map((item: typeof provider) => ({
      ...item,
      models: ["stale-local-selection"],
    })));
    healthy = true;
    await waitFor(() => !store.getState().providers[0].models.includes("stale-local-selection"));
    assert.equal(counts.saves, savesBeforeLocalSync, "a current server catalog only needs local reconciliation");
    const preferenceKey = "unsloth_chat_external_providers";
    const disabledProviders = store.getState().providers.map((item: typeof provider) => ({
      ...item, autoReloadModels: false,
    }));
    const persisted = JSON.stringify(disabledProviders);
    storage.set(preferenceKey, persisted);
    assert.ok(storageListener);
    storageListener({ key: preferenceKey, storageArea: localStorage });
    assert.equal(store.getState().providers[0].autoReloadModels, false);
    const probesBeforeStorageDisable = counts.probes;
    await pause();
    assert.equal(counts.probes, probesBeforeStorageDisable);
    assert.equal(storage.get(preferenceKey), persisted, "cross-tab sync must not write back stale state");
    storage.delete("unsloth_auth_token");
    const beforeLogout = counts.probes;
    await pause();
    assert.equal(counts.probes, beforeLogout);
  } finally {
    stop();
    assert.equal(storageListener, undefined, "stopping removes the storage listener");
    globalThis.fetch = originalFetch;
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: originalWindow,
    });
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: originalStorage,
    });
  }
});

test("manual saves follow delayed automatic saves, and a failed save does not block later updates", async () => {
  const { withProviderModelUpdate } = await vite.ssrLoadModule(
    "/src/features/chat/stores/external-providers-store.ts",
  );
  let release!: () => void;
  const delayed = new Promise<void>((resolve) => {
    release = resolve;
  });
  const writes: string[] = [];
  const automatic = withProviderModelUpdate("race", async () => {
    await delayed;
    writes.push("automatic");
  });
  const manual = withProviderModelUpdate("race", async () => {
    writes.push("manual");
  });
  await withProviderModelUpdate("other", async () => {
    writes.push("other");
  });
  assert.deepEqual(writes, ["other"]);
  release();
  await Promise.all([automatic, manual]);
  assert.deepEqual(writes, ["other", "automatic", "manual"]);
  await assert.rejects(
    withProviderModelUpdate("race", async () => {
      throw new Error("offline");
    }),
  );
  await withProviderModelUpdate("race", async () => {
    writes.push("retry");
  });
  assert.equal(writes.at(-1), "retry");
});

test("late settings sync preserves refreshed models while accepting unrelated server changes", async () => {
  const { preserveConcurrentProviderUpdates } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const previous = {
    id: "llama",
    providerType: "llama_cpp",
    models: ["old"],
    availableModels: ["old"],
    autoReloadModels: true,
  };
  const synced = {
    ...previous,
    name: "Renamed on server",
    models: ["old"],
    availableModels: ["old"],
  };
  const current = { ...previous, models: ["new"], availableModels: ["new"] };
  const merged = preserveConcurrentProviderUpdates(
    [synced],
    [previous],
    [current],
  )[0];
  assert.deepEqual(merged.models, ["new"]);
  assert.deepEqual(merged.availableModels, ["new"]);
  assert.equal(merged.name, "Renamed on server");
  assert.equal(merged.autoReloadModels, true);
  const external = {
    ...synced,
    models: ["external"],
    availableModels: ["external"],
  };
  assert.deepEqual(
    preserveConcurrentProviderUpdates([external], [previous], [previous])[0]
      .models,
    ["external"],
  );
  assert.deepEqual(
    preserveConcurrentProviderUpdates([], [previous], [current]),
    [],
  );
});
