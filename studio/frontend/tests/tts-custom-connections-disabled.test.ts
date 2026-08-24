// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9214: "Enable connections" is a frontend-only flag, so the request guard is the
// whole enforcement; custom TTS posted assistant text with connections switched off.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Adapter = {
  generateCustomTtsAudio: (text: string, signal?: AbortSignal) => Promise<string>;
};

/** The adapter with both stores faked and a fetch that only logs. */
function load(
  connectionsEnabled: boolean,
  {
    providers = [{ id: "conn-1", hasApiKey: true }],
    legacyKey = "",
    voice = "af_sky",
    encrypt = async (key: string) => `enc(${key})`,
  }: {
    providers?: { id: string; hasApiKey: boolean }[];
    legacyKey?: string;
    voice?: string;
    encrypt?: (key: string) => Promise<string>;
  } = {},
): {
  adapter: Adapter;
  posted: string[];
  clearedProviderIds: string[];
  setConnectionsEnabled: (enabled: boolean) => void;
} {
  const posted: string[] = [];
  const clearedProviderIds: string[] = [];
  let currentConnectionsEnabled = connectionsEnabled;
  const adapter = loadWithStubs<Adapter>(
    new URL(
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (_input: string, init: { body: string }) => {
          posted.push(init.body);
          return {
            ok: true,
            blob: async () => "audio-blob",
          };
        },
      },
      "../stores/external-providers-store": {
        useExternalProvidersStore: {
          getState: () => ({
            connectionsEnabled: currentConnectionsEnabled,
            providers,
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

// #9214: a failed key migration leaves the connection selectable on the retained key.
test("custom TTS forwards a retained legacy key when the connection has none saved", async () => {
  const { adapter, posted } = load(true, {
    providers: [{ id: "conn-1", hasApiKey: false }],
    legacyKey: "sk-legacy",
  });
  await adapter.generateCustomTtsAudio("hello");
  assert.equal(JSON.parse(posted[0]).encrypted_api_key, "enc(sk-legacy)");
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
