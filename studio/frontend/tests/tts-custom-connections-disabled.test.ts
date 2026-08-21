// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9214: "Enable connections" is a frontend-only flag, so the request guard is the
// whole enforcement. Custom TTS read the persisted engine and posted assistant text
// to the saved connection even with connections switched off.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Adapter = {
  generateCustomTtsAudio: (text: string, signal?: AbortSignal) => Promise<string>;
};

/** The adapter with both stores faked and a fetch that only logs. */
function load(connectionsEnabled: boolean): {
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
        useExternalProvidersStore: { getState: () => ({ connectionsEnabled }) },
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
