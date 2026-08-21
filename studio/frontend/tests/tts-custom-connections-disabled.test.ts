// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9214: "Enable connections" is a frontend-only flag, so the request guard is the
// whole enforcement; custom TTS posted assistant text with connections switched off.

import assert from "node:assert/strict";
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
  }: { providers?: { id: string; hasApiKey: boolean }[]; legacyKey?: string } = {},
): {
  adapter: Adapter;
  posted: string[];
} {
  const posted: string[] = [];
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
          getState: () => ({ connectionsEnabled, providers }),
        },
      },
      "../api/providers-api": {
        encryptProviderApiKey: async (key: string) => `enc(${key})`,
      },
      "../external-providers": {
        getExternalProviderApiKey: () => legacyKey,
      },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: {
          getState: () => ({
            ttsProviderId: "conn-1",
            ttsProviderModel: "kokoro",
            ttsProviderVoice: "af_sky",
          }),
        },
      },
      "@/lib/toast": { toast: { error: () => {} } },
    },
  );
  return { adapter, posted };
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
