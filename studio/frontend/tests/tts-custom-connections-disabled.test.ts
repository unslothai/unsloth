// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9214: "Enable connections" is a frontend-only flag, so the request guard is the
// whole enforcement; custom TTS posted assistant text with connections switched off.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Adapter = {
  generateCustomTtsAudio: (
    text: string,
    signal?: AbortSignal,
  ) => Promise<string>;
};

type AuthApi = {
  authFetch: (
    input: string,
    init?: RequestInit,
    options?: { beforeRetry?: () => void },
  ) => Promise<Response>;
};

type StubProvider = {
  id: string;
  hasApiKey: boolean;
  baseUrl?: string;
  providerType?: string;
  backendProviderType?: string;
  updatedAt?: number;
};

/** The adapter with both stores faked and a fetch that only logs. */
function load(
  connectionsEnabled: boolean,
  {
    providers = [{ id: "conn-1", hasApiKey: true }],
    legacyKey = "",
    voice = "af_sky",
    encrypt = async (key: string) => `enc(${key})`,
    beforeAuthenticatedRetry,
  }: {
    providers?: StubProvider[];
    legacyKey?: string;
    voice?: string;
    encrypt?: (key: string) => Promise<string>;
    beforeAuthenticatedRetry?: () => void;
  } = {},
): {
  adapter: Adapter;
  posted: string[];
  clearedProviderIds: string[];
  setConnectionsEnabled: (enabled: boolean) => void;
  setProviders: (providers: StubProvider[]) => void;
} {
  const posted: string[] = [];
  const clearedProviderIds: string[] = [];
  let currentConnectionsEnabled = connectionsEnabled;
  let currentProviders = providers;
  const adapter = loadWithStubs<Adapter>(
    new URL(
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (
          _input: string,
          init: { body: string },
          options?: { beforeRetry?: () => void },
        ) => {
          if (beforeAuthenticatedRetry) {
            beforeAuthenticatedRetry();
            options?.beforeRetry?.();
          }
          posted.push(init.body);
          return {
            ok: true,
            blob: async () => "audio-blob",
          };
        },
      },
      "../search-images/search-images": {
        stripSearchImageTokens: (text: string) => text,
      },
      "../stores/external-providers-store": {
        useExternalProvidersStore: {
          getState: () => ({
            connectionsEnabled: currentConnectionsEnabled,
            providers: currentProviders,
          }),
        },
      },
      "../api/providers-api": {
        encryptProviderApiKey: encrypt,
      },
      "../external-providers": {
        getExternalProviderApiKey: () => legacyKey,
      },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: {
          getState: () => ({
            ttsProviderId: "conn-1",
            ttsProviderModel: "kokoro",
            ttsProviderVoice: voice,
            setTtsProviderId: (value: string) => clearedProviderIds.push(value),
          }),
        },
      },
      "@/lib/toast": { toast: { error: () => {} } },
    },
  );
  return {
    adapter,
    posted,
    clearedProviderIds,
    setConnectionsEnabled: (enabled) => {
      currentConnectionsEnabled = enabled;
    },
    setProviders: (nextProviders) => {
      currentProviders = nextProviders;
    },
  };
}

Object.assign(globalThis, {
  URL: Object.assign(URL, { createObjectURL: () => "blob:tts" }),
});

test("custom TTS sends nothing once connections are switched off", async () => {
  const { adapter, posted } = load(false);
  await assert.rejects(
    adapter.generateCustomTtsAudio("secret assistant reply"),
    /Connections are disabled/,
  );
  assert.deepEqual(posted, []);
});

test("custom TTS still reaches the saved connection while connections are on", async () => {
  const { adapter, posted } = load(true);
  assert.equal(await adapter.generateCustomTtsAudio("hello"), "blob:tts");
  assert.equal(posted.length, 1);
  assert.deepEqual(JSON.parse(posted[0]), {
    input: "hello",
    provider_id: "conn-1",
    model: "kokoro",
    voice: "af_sky",
  });
});

test("custom TTS defaults a blank voice for strict OpenAI-compatible endpoints", async () => {
  const { adapter, posted } = load(true, { voice: "  " });
  await adapter.generateCustomTtsAudio("hello");
  assert.equal(JSON.parse(posted[0]).voice, "alloy");
});

test("custom TTS rejects a deleted provider even when Voice settings is unmounted", async () => {
  const { adapter, posted, clearedProviderIds } = load(true, { providers: [] });
  await assert.rejects(
    adapter.generateCustomTtsAudio("must stay local"),
    /connection.*no longer exists/i,
  );
  assert.deepEqual(posted, []);
  assert.deepEqual(clearedProviderIds, [""]);
});

test("custom TTS rechecks the global switch after legacy-key encryption", async () => {
  let releaseEncryption: ((value: string) => void) | undefined;
  const encryption = new Promise<string>((resolve) => {
    releaseEncryption = resolve;
  });
  const { adapter, posted, setConnectionsEnabled } = load(true, {
    providers: [{ id: "conn-1", hasApiKey: false }],
    legacyKey: "sk-legacy",
    encrypt: async () => encryption,
  });

  const pending = adapter.generateCustomTtsAudio("secret assistant reply");
  await Promise.resolve();
  setConnectionsEnabled(false);
  releaseEncryption?.("enc(sk-legacy)");

  await assert.rejects(pending, /Connections are disabled/);
  assert.deepEqual(posted, []);
});

test("custom TTS does not send a captured legacy key after the connection changes", async () => {
  let releaseEncryption: ((value: string) => void) | undefined;
  const encryption = new Promise<string>((resolve) => {
    releaseEncryption = resolve;
  });
  const { adapter, posted, setProviders } = load(true, {
    providers: [
      {
        id: "conn-1",
        hasApiKey: false,
        baseUrl: "https://old-tts.example/v1",
        providerType: "custom",
        backendProviderType: "custom",
        updatedAt: 1,
      },
    ],
    legacyKey: "sk-legacy",
    encrypt: async () => encryption,
  });

  const pending = adapter.generateCustomTtsAudio("secret assistant reply");
  await Promise.resolve();
  setProviders([
    {
      id: "conn-1",
      hasApiKey: true,
      baseUrl: "https://new-tts.example/v1",
      providerType: "custom",
      backendProviderType: "custom",
      updatedAt: 2,
    },
  ]);
  releaseEncryption?.("enc(sk-legacy)");

  await assert.rejects(pending, /connection changed/i);
  assert.deepEqual(posted, []);
});

test("custom TTS rechecks the connection switch before an authenticated retry", async () => {
  let disableConnections = () => {};
  const loaded = load(true, {
    beforeAuthenticatedRetry: () => disableConnections(),
  });
  disableConnections = () => loaded.setConnectionsEnabled(false);

  await assert.rejects(
    loaded.adapter.generateCustomTtsAudio("secret assistant reply"),
    /Connections are disabled/,
  );
  assert.deepEqual(loaded.posted, []);
});

test("authFetch invokes its policy guard after refresh and before retry", async () => {
  let accessToken: string | null = "expired-access";
  let refreshToken: string | null = "refresh-token";
  const fetched: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    fetched.push(url);
    if (url === "/api/auth/refresh") {
      return new Response(
        JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "fresh-refresh",
          must_change_password: false,
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    return new Response(null, { status: fetched.length === 1 ? 401 : 200 });
  };

  try {
    const authApi = loadWithStubs<AuthApi>(
      new URL("../src/features/auth/api.ts", import.meta.url),
      {
        "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
        "./session": {
          clearAuthTokens: () => {
            accessToken = null;
            refreshToken = null;
          },
          getAuthToken: () => accessToken,
          getRefreshToken: () => refreshToken,
          mustChangePassword: () => false,
          setMustChangePassword: () => {},
          storeAuthTokens: (access: string, refresh: string) => {
            accessToken = access;
            refreshToken = refresh;
          },
        },
      },
    );

    await assert.rejects(
      authApi.authFetch(
        "/api/inference/audio/speech",
        { method: "POST" },
        {
          beforeRetry: () => {
            throw new Error("Connections are disabled");
          },
        },
      ),
      /Connections are disabled/,
    );
    assert.deepEqual(fetched, [
      "/api/inference/audio/speech",
      "/api/auth/refresh",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// #9214: a failed key migration leaves the connection selectable on the retained key.
test("custom TTS forwards a retained legacy key when the connection has none saved", async () => {
  const { adapter, posted } = load(true, {
    providers: [
      {
        id: "conn-1",
        hasApiKey: false,
        baseUrl: "https://tts.example/v1",
      },
    ],
    legacyKey: "sk-legacy",
  });
  await adapter.generateCustomTtsAudio("hello");
  assert.deepEqual(JSON.parse(posted[0]), {
    input: "hello",
    provider_id: "conn-1",
    provider_base_url: "https://tts.example/v1",
    model: "kokoro",
    voice: "af_sky",
    encrypted_api_key: "enc(sk-legacy)",
  });
});

test("custom TTS sends no key when the connection already has one saved", async () => {
  const { adapter, posted } = load(true, {
    providers: [{ id: "conn-1", hasApiKey: true }],
    legacyKey: "sk-legacy",
  });
  await adapter.generateCustomTtsAudio("hello");
  assert.equal("encrypted_api_key" in JSON.parse(posted[0]), false);
});

// #9214: a deleted connection left ttsProviderId persisted, so the select pointed at a
// missing item and every read aloud posted the stale id. Mirrors the dictation guard.
test("the custom TTS selection is dropped when its connection disappears", () => {
  const voiceTab = readFileSync(
    new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    voiceTab,
    /if \(ttsProviderId && !hasSelectedTtsConnection\) \{\s*setTtsProviderId\(""\);/,
  );
  assert.match(
    voiceTab,
    /value=\{hasSelectedTtsConnection \? ttsProviderId : ""\}/,
  );
});
