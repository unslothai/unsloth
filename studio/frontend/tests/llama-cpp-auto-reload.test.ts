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

test("the preference is llama.cpp only and survives a backend sync", async () => {
  const { mergeLocalProviderOptions } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const merge = (existing: object | undefined, providerType: string) =>
    mergeLocalProviderOptions(existing, { providerType }).autoReloadModels;
  assert.equal(merge(undefined, "llama_cpp"), undefined);
  assert.equal(merge({ autoReloadModels: true }, "llama_cpp"), true);
  assert.equal(merge({ autoReloadModels: true }, "ollama"), undefined);
});

test("a reload keeps manual IDs and picks, drops removed IDs and enables new ones", async () => {
  const { mergeReloadedModels } = await vite.ssrLoadModule(
    "/src/features/chat/llama-cpp-auto-reload.ts",
  );
  // beta was deselected, alpha is gone, gamma is new, manual was typed in.
  assert.deepEqual(
    mergeReloadedModels(["manual", "alpha"], ["alpha", "beta"], ["beta", "gamma"]),
    ["manual", "gamma"],
  );
  // A manual ID the server now lists stays selected once.
  assert.deepEqual(
    mergeReloadedModels(["manual"], [], ["manual", "alpha"]),
    ["manual", "alpha"],
  );
  // Nothing left selected: enable the whole catalog rather than an empty connection.
  assert.deepEqual(mergeReloadedModels(["alpha"], ["alpha", "beta"], ["beta"]), ["beta"]);
});

test("the monitor reloads on first contact and each reconnect only", async () => {
  const storage = new Map<string, string>([["unsloth_auth_token", "t"]]);
  const saved = { models: ["manual", "alpha"], available_models: ["alpha"] };
  let served: string[] | null = ["alpha", "beta"];
  const puts: Array<{ models: string[]; available_models: string[] }> = [];
  let calls = 0;
  const originals = [globalThis.window, globalThis.localStorage, globalThis.fetch] as const;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { addEventListener() {}, removeEventListener() {} },
  });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    },
  });
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    calls++;
    const url = String(input);
    if (url.includes("/api/providers/models")) {
      return served === null
        ? Response.json({ detail: "connection refused" }, { status: 502 })
        : Response.json(served.map((id) => ({ id })));
    }
    if (init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      puts.push(body);
      Object.assign(saved, body);
      return Response.json({ id: "p", ...saved });
    }
    if (url.replace(/\/$/, "").endsWith("/api/providers")) {
      return Response.json([{ id: "p", base_url: "http://llama/v1", ...saved }]);
    }
    throw new Error(`unexpected ${url}`);
  }) as typeof fetch;

  const { startLlamaCppAutoReload } = await vite.ssrLoadModule(
    "/src/features/chat/llama-cpp-auto-reload.ts",
  );
  const { useExternalProvidersStore: store } = await vite.ssrLoadModule(
    "/src/features/chat/stores/external-providers-store.ts",
  );
  store.setState({
    connectionsEnabled: true,
    providers: [{
      id: "p",
      providerType: "llama_cpp",
      name: "llama.cpp",
      baseUrl: "http://llama/v1",
      models: ["manual", "alpha"],
      availableModels: ["alpha"],
      hasApiKey: false,
      autoReloadModels: true,
      createdAt: 1,
      updatedAt: 1,
    }],
  });
  const row = () => store.getState().providers[0];
  const settle = async (condition: () => boolean) => {
    for (let i = 0; i < 200 && !condition(); i++) await new Promise((r) => setTimeout(r, 5));
    assert.ok(condition());
  };
  const idle = () => new Promise((r) => setTimeout(r, 60));

  const stop = startLlamaCppAutoReload(10);
  try {
    await settle(() => puts.length === 1);
    assert.deepEqual(puts[0], {
      models: ["manual", "alpha", "beta"],
      available_models: ["alpha", "beta"],
    });
    await settle(() => row().models.length === 3);
    await idle();
    assert.equal(puts.length, 1, "no rewrite while the server stays up");

    // Another tab deselected beta; this tab's copy still has it. The reload must follow the saved row.
    saved.models = ["manual", "alpha"];
    served = null;
    await idle();
    served = ["alpha", "beta", "gamma"];
    await settle(() => puts.length === 2);
    assert.deepEqual(puts[1].models, ["manual", "alpha", "gamma"]);
    await settle(() => row().models.join() === "manual,alpha,gamma");

    // A server still starting lists nothing: keep the selection, reload once it lists models.
    served = null;
    await idle();
    served = [];
    await idle();
    assert.equal(puts.length, 2);
    served = ["gamma"];
    await settle(() => puts.length === 3);
    assert.deepEqual(puts[2].models, ["manual", "gamma"]);

    // Switched off: a reconnect changes nothing.
    store.setState({ providers: [{ ...row(), autoReloadModels: false }] });
    served = null;
    await idle();
    served = ["gamma", "delta"];
    await idle();
    assert.equal(puts.length, 3);
  } finally {
    stop();
  }
  const callsAtStop = calls;
  await idle();
  assert.equal(calls, callsAtStop, "stopped monitor makes no requests");
  Object.defineProperty(globalThis, "window", { configurable: true, value: originals[0] });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: originals[1] });
  globalThis.fetch = originals[2];
});
