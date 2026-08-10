// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


import type { ProviderConfig } from "../src/features/chat/api/providers-api.ts";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  reconcileLegacyHfToken,
  reconcileLegacyProviderKeys,
  runCredentialBootstrap,
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
      removeLegacyKey: (id) => removed.push(id),
    },
  );

  assert.deepEqual(saved, ["legacy:old-key"]);
  assert.deepEqual(removed, ["server", "legacy"]);
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

test("HF migration is server-first, retry-safe, and idempotent", async () => {
  const applied: string[] = [];
  let removed = 0;
  let saves = 0;
  const server = await reconcileLegacyHfToken({
    loadSavedToken: async () => ({ has_token: true, token: "hf_server" }),
    getLegacyToken: () => "hf_legacy",
    saveLegacyToken: async () => {
      saves += 1;
      return { has_token: true, token: "hf_legacy" };
    },
    removeLegacyToken: () => {
      removed += 1;
    },
    applyToken: (token) => applied.push(token),
  });
  assert.equal(server.token, "hf_server");
  assert.deepEqual(applied, ["hf_server"]);
  assert.equal(saves, 0);
  assert.equal(removed, 1);

  await assert.rejects(
    reconcileLegacyHfToken({
      loadSavedToken: async () => ({ has_token: false, token: null }),
      getLegacyToken: () => "hf_retry",
      saveLegacyToken: async () => {
        throw new Error("offline");
      },
      removeLegacyToken: () => {
        removed += 1;
      },
      applyToken: (token) => applied.push(token),
    }),
    /offline/,
  );
  assert.equal(removed, 1);
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

test("new HF edits never write the token back to localStorage", () => {
  const source = readFileSync(
    new URL("../src/features/hub/stores/hf-token-store.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /localStorage\.setItem\(HF_TOKEN_KEY/);
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
  assert.match(rootSource, /!isAuthFlowRoute \? \(/);
  assert.match(sessionSource, /dispatchEvent\(new Event\(AUTH_SESSION_STORED_EVENT\)\)/);
});
