// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


import type { ProviderConfig } from "../src/features/chat/api/providers-api.ts";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();

const {
  reconcileLegacyHfToken,
  reconcileLegacyProviderKeys,
  runCredentialBootstrap,

  settleTasksIfCurrent,
} = await import("../src/features/credentials/reconciliation.ts");

const { resolveProviderCredentialEdit } = await import(
  "../src/features/chat/provider-credential-edit.ts"
);


type UiProviderConfig = ReturnType<
  Parameters<typeof runCredentialBootstrap>[0]["getProviders"]
>[number];

function provider(id: string, hasApiKey: boolean): ProviderConfig {
  return {
    id,
    provider_type: "openai",
    display_name: id,
    base_url: "https://api.openai.com/v1",
    is_enabled: true,
    has_api_key: hasApiKey,
    models: [],
    available_models: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}


function uiProvider(id: string, hasApiKey: boolean): UiProviderConfig {
  return {
    id,
    providerType: "openai",
    name: id,
    baseUrl: "https://api.openai.com/v1",
    hasApiKey,
    models: [],
    createdAt: 1,
    updatedAt: 1,
  };
}

test("provider migration is server-first and removes only confirmed keys", async () => {
  const removed: string[] = [];
  const saved: string[] = [];
  const result = await reconcileLegacyProviderKeys(
    [provider("server", true), provider("legacy", false)],
    {
      getLegacyKey: (id) => (id === "legacy" ? " old-key " : "stale-key"),
      saveLegacyKey: async (id, key) => {
        saved.push(`${id}:${key}`);
        return provider(id, true);
      },
      removeLegacyKey: (id, key) => removed.push(`${id}:${key}`),
    },
  );

  assert.deepEqual(saved, ["legacy:old-key"]);
  assert.deepEqual(removed, ["server:stale-key", "legacy: old-key "]);
  assert.equal(result.every((row) => row.has_api_key), true);
});

test("provider migration retains failed and orphaned legacy input", async () => {
  const removed: string[] = [];
  const requested: string[] = [];
  const result = await reconcileLegacyProviderKeys([provider("existing", false)], {
    getLegacyKey: (id) => {
      requested.push(id);
      return "retry-me";
    },
    saveLegacyKey: async () => {
      throw new Error("offline");
    },
    removeLegacyKey: (id) => removed.push(id),
  });

  assert.equal(result[0].has_api_key, false);
  assert.deepEqual(requested, ["existing"]);
  assert.deepEqual(removed, []);
  assert.equal(requested.includes("orphan"), false);
});


test("provider migration retains plaintext when a successful response has no saved key", async () => {
  const removed: string[] = [];
  const result = await reconcileLegacyProviderKeys([provider("legacy", false)], {
    getLegacyKey: () => "retry-me",
    saveLegacyKey: async () => provider("legacy", false),
    removeLegacyKey: (id) => removed.push(id),
  });

  assert.equal(result[0].has_api_key, false);
  assert.deepEqual(removed, []);
});

test("provider migration is retry-safe after partial success", async () => {
  const removed = new Set<string>();
  let attempts = 0;
  const dependencies = {
    getLegacyKey: (id: string) => (removed.has(id) ? "" : `key-${id}`),
    saveLegacyKey: async (id: string) => {
      attempts += 1;
      if (id === "second" && attempts === 2) throw new Error("offline");
      return provider(id, true);
    },
    removeLegacyKey: (id: string) => removed.add(id),
  };

  const first = await reconcileLegacyProviderKeys(
    [provider("first", false), provider("second", false)],
    dependencies,
  );
  assert.deepEqual([...removed], ["first"]);
  const second = await reconcileLegacyProviderKeys(first, dependencies);
  assert.equal(second.every((row) => row.has_api_key), true);
  assert.deepEqual([...removed], ["first", "second"]);
});

test("superseded provider backfills are not launched", async () => {
  let launched = 0;
  await settleTasksIfCurrent(
    [async () => { launched += 1; }],
    () => false,
  );
  assert.equal(launched, 0);

  await settleTasksIfCurrent([async () => { launched += 1; }], () => true);
  assert.equal(launched, 1);
});


test("provider backfills are finished, not merely started, when the batch resolves", async () => {
  // The credential bootstrap gate releases app content on this promise, so a batch that is
  // only started leaves the writes racing an immediate close. Timer-backed tasks are the
  // only way to tell "awaited" from "fired and forgotten": a task that counts synchronously
  // is already done by the time the helper returns either way.
  const finished: string[] = [];
  const delayed = (name: string, ms: number) => () =>
    new Promise((resolve) => {
      setTimeout(() => {
        finished.push(name);
        resolve(null);
      }, ms);
    });

  // The rejection in the middle must not sink the two around it, which is why the batch
  // settles rather than rejecting on the first failure.
  await settleTasksIfCurrent(
    [delayed("first", 20), () => Promise.reject(new Error("backfill failed")), delayed("last", 40)],
    () => true,
  );
  assert.deepEqual(finished, ["first", "last"]);
});


test("authoritative provider cleanup removes only orphaned legacy keys", async () => {
  const { store } = installLocalStorageFake();
  store.set(
    "unsloth_chat_external_provider_keys",
    JSON.stringify({ retained: "sk-retained", orphan: "sk-orphan" }),
  );
  try {
    const { pruneExternalProviderApiKeys } = await import(
      "../src/features/chat/external-providers.ts"
    );
    pruneExternalProviderApiKeys(["retained"]);
    assert.deepEqual(
      JSON.parse(store.get("unsloth_chat_external_provider_keys") ?? "{}"),
      { retained: "sk-retained" },
    );
  } finally {
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("browser credential migration uses insert-if-absent endpoints", () => {
  const hfSource = readFileSync(
    new URL("../src/features/hub/stores/hf-token-store.ts", import.meta.url),
    "utf8",
  );
  const providerSource = readFileSync(
    new URL("../src/features/chat/sync-external-providers.ts", import.meta.url),
    "utf8",
  );
  assert.match(hfSource, /migrateHfToken\(token\)/);
  assert.match(providerSource, /saveLegacyKey: migrateProviderApiKey/);
});



test("HF migration is server-first, retry-safe, and idempotent", async () => {
  const applied: string[] = [];
  const removed: string[] = [];
  let saves = 0;
  const server = await reconcileLegacyHfToken({
    loadSavedToken: async () => ({ has_token: true, token: "hf_server" }),
    getLegacyToken: () => "hf_legacy",
    saveLegacyToken: async () => {
      saves += 1;
      return { has_token: true, token: "hf_legacy" };
    },
    removeLegacyToken: (token) => {
      removed.push(token);
    },
    applyToken: (token) => applied.push(token),
  });
  assert.equal(server.token, "hf_server");
  assert.deepEqual(applied, ["hf_server"]);
  assert.equal(saves, 0);
  assert.deepEqual(removed, ["hf_legacy"]);


  await assert.rejects(
    reconcileLegacyHfToken({
      loadSavedToken: async () => {
        throw new Error("backend unavailable");
      },
      getLegacyToken: () => "hf_read_retry",
      saveLegacyToken: async () => {
        throw new Error("must not upload without server status");
      },
      removeLegacyToken: (token) => {
        removed.push(token);
      },
      applyToken: (token) => applied.push(token),
    }),
    /backend unavailable/,
  );

  await assert.rejects(
    reconcileLegacyHfToken({
      loadSavedToken: async () => ({ has_token: false, token: null }),
      getLegacyToken: () => "hf_retry",
      saveLegacyToken: async () => {
        throw new Error("offline");
      },
      removeLegacyToken: (token) => {
        removed.push(token);
      },
      applyToken: (token) => applied.push(token),
    }),
    /offline/,
  );
  assert.deepEqual(removed, ["hf_legacy"]);
  assert.deepEqual(applied, ["hf_server", "hf_read_retry", "hf_retry"]);
});


test("HF cleanup removes only the value observed before the request", async () => {
  let legacyToken = "hf_old";
  const removed: string[] = [];
  await reconcileLegacyHfToken({
    loadSavedToken: async () => {
      legacyToken = "hf_new";
      return { has_token: true, token: "hf_server" };
    },
    getLegacyToken: () => legacyToken,
    saveLegacyToken: async () => {
      throw new Error("server token must win");
    },
    removeLegacyToken: (expectedToken) => {
      removed.push(expectedToken);
      if (legacyToken === expectedToken) legacyToken = "";
    },
    applyToken: () => undefined,
  });

  assert.deepEqual(removed, ["hf_old"]);
  assert.equal(legacyToken, "hf_new");
});

test("credential bootstrap releases providers only after both attempts settle", async () => {
  let releaseHf!: () => void;
  let releaseProviders!: (value: UiProviderConfig[]) => void;
  const hf = new Promise<void>((resolve) => {
    releaseHf = resolve;
  });
  const providers = new Promise<UiProviderConfig[]>((resolve) => {
    releaseProviders = resolve;
  });
  const applied: UiProviderConfig[][] = [];
  const boot = runCredentialBootstrap({
    hydrateHfToken: () => hf,
    getProviders: () => [uiProvider("local", false)],
    syncProviders: () => providers,
    setProviders: (rows) => applied.push(rows),
  });

  releaseProviders([uiProvider("server", true)]);
  await Promise.resolve();
  assert.deepEqual(applied, []);
  releaseHf();
  await boot;
  assert.deepEqual(applied, [[uiProvider("server", true)]]);
});


test("provider bootstrap cannot publish a previous authentication session", async () => {
  let current = true;
  let releaseProviders!: (value: UiProviderConfig[]) => void;
  const providers = new Promise<UiProviderConfig[]>((resolve) => {
    releaseProviders = resolve;
  });
  const applied: UiProviderConfig[][] = [];
  const boot = runCredentialBootstrap({
    hydrateHfToken: async () => undefined,
    getProviders: () => [uiProvider("first-session", false)],
    syncProviders: () => providers,
    setProviders: (rows) => applied.push(rows),
    isCurrent: () => current,
  });

  current = false;
  releaseProviders([uiProvider("stale-response", true)]);
  await boot;
  assert.deepEqual(applied, []);
});

test("provider migration does not consume local input after a session change", async () => {
  let current = true;
  const removed: string[] = [];
  const result = await reconcileLegacyProviderKeys(
    [provider("provider-1", false)],
    {
      getLegacyKey: () => "legacy-key",
      saveLegacyKey: async () => {
        current = false;
        return provider("provider-1", true);
      },
      removeLegacyKey: (id) => removed.push(id),
      isCurrent: () => current,
    },
  );

  assert.equal(result[0].has_api_key, false);
  assert.deepEqual(removed, []);
});

test("legacy migration remains installation-wide and retry-safe", () => {
  const bootstrapSource = readFileSync(
    new URL("../src/features/credentials/bootstrap.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(bootstrapSource, /migration-owner|legacy_credential_owner/);
  assert.doesNotMatch(bootstrapSource, /authSubjectFromJwt|currentOwner/);
});


test("HF credential API rejects a successful non-JSON response", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response("not json", {
      status: 200,
      headers: { "Content-Type": "text/plain" },
    })) as typeof fetch;

  try {
    const { loadSavedHfToken } = await import(
      "../src/features/hub/api/hf-token-api.ts"
    );
    await assert.rejects(loadSavedHfToken(), /returned an invalid response/);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("legacy training HF token gets a durable migration copy", async () => {
  const { store } = installLocalStorageFake();
  try {
    const hfStore = await import(
      "../src/features/hub/stores/hf-token-store.ts"
    );
    hfStore.stageLegacyHfTokenForMigration(" hf_durable ");
    assert.equal(store.get("unsloth_hf_token_migration_v1"), "hf_durable");
  } finally {
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("HF store hydrates from the API without recreating plaintext storage", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  store.set("unsloth_hf_token", "hf_legacy");
  const originalFetch = globalThis.fetch;
  const requests: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    requests.push(String(input));
    return new Response(
      JSON.stringify({ token: "hf_server", has_token: true }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }) as typeof fetch;

  try {
    const hfStore = await import(
      "../src/features/hub/stores/hf-token-store.ts"
    );
    await hfStore.hydrateHfTokenFromBackend();
    assert.equal(hfStore.getHfToken(), "hf_server");
    assert.deepEqual(requests, ["/api/settings/hugging-face-token"]);
    assert.equal(store.has("unsloth_hf_token"), false);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("HF hydration does not overwrite an in-flight user edit", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let resolveLoad!: (response: Response) => void;
  let resolveSave!: (response: Response) => void;
  const loadResponse = new Promise<Response>((resolve) => {
    resolveLoad = resolve;
  });
  const saveResponse = new Promise<Response>((resolve) => {
    resolveSave = resolve;
  });

  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) =>
    init?.method === "PUT" ? saveResponse : loadResponse) as typeof fetch;

  try {
    const hfStore = await import(
      "../src/features/hub/stores/hf-token-store.ts"
    );
    const hydration = hfStore.hydrateHfTokenFromBackend();
    hfStore.useHfTokenStore.getState().setToken("hf_user_edit");

    resolveLoad(
      new Response(JSON.stringify({ token: "hf_server", has_token: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await hydration;
    assert.equal(hfStore.getHfToken(), "hf_user_edit");
    assert.equal(hfStore.useHfTokenStore.getState().isPersisting, true);

    resolveSave(
      new Response(JSON.stringify({ token: "hf_user_edit", has_token: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    for (let attempt = 0; attempt < 10; attempt += 1) {
      if (!hfStore.useHfTokenStore.getState().isPersisting) break;
      await new Promise((resolve) => setImmediate(resolve));
    }
    assert.equal(hfStore.getHfToken(), "hf_user_edit");
    assert.equal(hfStore.useHfTokenStore.getState().isPersisting, false);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("cross-tab refresh replays after an active HF hydration", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let resolveFirst!: (response: Response) => void;
  let firstStarted!: () => void;
  const firstResponse = new Promise<Response>((resolve) => { resolveFirst = resolve; });
  const started = new Promise<void>((resolve) => { firstStarted = resolve; });
  let reads = 0;
  globalThis.fetch = (async () => {
    reads += 1;
    if (reads === 1) {
      firstStarted();
      return firstResponse;
    }
    return new Response(JSON.stringify({ token: "hf_new_tab", has_token: true }), {
      status: 200, headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const hfStore = await import("../src/features/hub/stores/hf-token-store.ts");
    const hydration = hfStore.hydrateHfTokenFromBackend();
    await started;
    const refresh = hfStore.refreshHfTokenFromBackend();
    resolveFirst(new Response(JSON.stringify({ token: "hf_old", has_token: true }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    await Promise.all([hydration, refresh]);
    assert.equal(reads, 2);
    assert.equal(hfStore.getHfToken(), "hf_new_tab");
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("a delayed HF write response reconciles a newer cross-tab commit", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let resolveWrite!: (response: Response) => void;
  let writeStarted!: () => void;
  const writeResponse = new Promise<Response>((resolve) => { resolveWrite = resolve; });
  const startedWrite = new Promise<void>((resolve) => { writeStarted = resolve; });
  let reads = 0;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "PUT") {
      writeStarted();
      return writeResponse;
    }
    reads += 1;
    return new Response(JSON.stringify({ token: "hf_new_tab", has_token: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const hfStore = await import("../src/features/hub/stores/hf-token-store.ts");
    hfStore.useHfTokenStore.getState().setToken("hf_delayed_tab");
    await startedWrite;

    // Another tab commits while this write's response is still delayed.
    await hfStore.refreshHfTokenFromBackend();
    resolveWrite(new Response(JSON.stringify({ token: "hf_delayed_tab", has_token: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    for (let attempt = 0; attempt < 10; attempt += 1) {
      if (!hfStore.useHfTokenStore.getState().isPersisting) break;
      await new Promise((resolve) => setImmediate(resolve));
    }

    assert.equal(reads, 2);
    assert.equal(hfStore.getHfToken(), "hf_new_tab");
    assert.equal(hfStore.useHfTokenStore.getState().isPersisting, false);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});



test("pending HF edits can be drained before navigation", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let resolveSave!: (response: Response) => void;
  let saveStarted!: () => void;
  const response = new Promise<Response>((resolve) => { resolveSave = resolve; });
  const started = new Promise<void>((resolve) => { saveStarted = resolve; });
  globalThis.fetch = (async () => { saveStarted(); return response; }) as typeof fetch;

  try {
    const hfStore = await import("../src/features/hub/stores/hf-token-store.ts");
    hfStore.useHfTokenStore.getState().setToken("hf_before_exit");
    await started;
    let drained = false;
    const drain = hfStore.waitForHfTokenPersistence().then(() => { drained = true; });
    await Promise.resolve();
    assert.equal(drained, false);
    resolveSave(new Response(JSON.stringify({ token: "hf_before_exit", has_token: true }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    await drain;
    assert.equal(drained, true);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});


test("a superseded successful HF write advances the rollback baseline", async () => {
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let resolveFirst!: (response: Response) => void;
  let resolveSecond!: (response: Response) => void;
  let firstStarted!: () => void;
  let secondStarted!: () => void;
  const firstRequest = new Promise<void>((resolve) => { firstStarted = resolve; });
  const secondRequest = new Promise<void>((resolve) => { secondStarted = resolve; });
  const firstResponse = new Promise<Response>((resolve) => { resolveFirst = resolve; });
  const secondResponse = new Promise<Response>((resolve) => { resolveSecond = resolve; });
  let requestCount = 0;
  globalThis.fetch = (async () => {
    requestCount += 1;
    if (requestCount === 1) { firstStarted(); return firstResponse; }
    secondStarted();
    return secondResponse;
  }) as typeof fetch;

  try {
    const hfStore = await import("../src/features/hub/stores/hf-token-store.ts");
    hfStore.useHfTokenStore.getState().setToken("hf_first");
    await firstRequest;
    hfStore.useHfTokenStore.getState().setToken("hf_second");
    resolveFirst(new Response(JSON.stringify({ token: "hf_first", has_token: true }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    await secondRequest;
    resolveSecond(new Response(JSON.stringify({ detail: "failed" }), {
      status: 500, headers: { "Content-Type": "application/json" },
    }));
    for (let attempt = 0; attempt < 10; attempt += 1) {
      if (!hfStore.useHfTokenStore.getState().isPersisting) break;
      await new Promise((resolve) => setImmediate(resolve));
    }
    assert.equal(hfStore.getHfToken(), "hf_first");
    assert.match(hfStore.useHfTokenStore.getState().persistenceError ?? "", /failed/);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});




test("new HF edits never write the token back to localStorage", () => {
  const source = readFileSync(
    new URL("../src/features/hub/stores/hf-token-store.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /localStorage\.setItem\(HF_TOKEN_KEY/);

  assert.match(source, /persistenceError:/);
  assert.match(source, /HF_TOKEN_SYNC_KEY/);
});


test("legacy training HF tokens never merge back into persisted state", () => {
  const source = readFileSync(
    new URL(
      "../src/features/training/stores/training-config-persistence.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /if \(key === "hfToken"\) return false/);
  assert.match(source, /delete persistedRecord\.hfToken/);
});

test("provider edit state keeps, replaces, and explicitly clears saved keys", () => {
  assert.deepEqual(resolveProviderCredentialEdit(true, "", false), {
    action: "keep",
  });
  assert.deepEqual(resolveProviderCredentialEdit(true, "", true), {
    action: "clear",
  });
  assert.deepEqual(resolveProviderCredentialEdit(true, " replacement ", true), {
    action: "replace",
    apiKey: "replacement",
  });
  assert.deepEqual(resolveProviderCredentialEdit(false, "", false), {
    action: "missing",
  });

  const source = readFileSync(
    new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /Saved securely\. Leave blank to keep it\./);
  assert.match(source, /Remove saved key/);

  assert.match(source, /removeExternalProviderApiKey\(editingProviderId\)/);
});

test("credential gate follows authentication session transitions", () => {
  const rootSource = readFileSync(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  const sessionSource = readFileSync(
    new URL("../src/features/auth/session.ts", import.meta.url),
    "utf8",
  );

  assert.match(rootSource, /AUTH_SESSION_CLEARED_EVENT, reconcile/);
  assert.match(rootSource, /AUTH_SESSION_STORED_EVENT, reconcile/);
  assert.match(rootSource, /\{!isAuthFlowRoute \? \(/);
  assert.match(sessionSource, /const sessionStarted = !localStorage\.getItem\(AUTH_TOKEN_KEY\)/);
  assert.match(sessionSource, /dispatchEvent\(new Event\(AUTH_SESSION_STORED_EVENT\)\)/);
  const bootstrapSource = readFileSync(
    new URL("../src/features/credentials/bootstrap.ts", import.meta.url),
    "utf8",
  );
  assert.match(sessionSource, /authSessionEpoch \+= 1/);
  assert.match(bootstrapSource, /const sessionEpoch = getAuthSessionEpoch\(\)/);
  assert.match(
    bootstrapSource,
    /hasAuthToken\(\) && getAuthSessionEpoch\(\) === sessionEpoch/,
  );

});
