// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read aloud on the "Load TTS model" engine speaks in a saved Audio voice and keeps its clips
// out of Speak's history.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

import { readSrc } from "./helpers/kit.ts";

type Adapter = {
  generateStudioTtsAudio: (
    text: string,
    signal?: AbortSignal,
  ) => Promise<string>;
};

type Reply = { status: number; body: unknown };

/** The adapter with the voice setting faked and a fetch that logs each body. */
function load(
  voiceId: string,
  reply: Reply = { status: 200, body: { audio: { data: "UklGRg==" } } },
) {
  const posted: Record<string, unknown>[] = [];
  const setVoiceIds: string[] = [];
  const adapter = loadWithStubs<Adapter>(
    new URL(
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (_input: string, init: { body: string }) => {
          posted.push(JSON.parse(init.body));
          return {
            ok: reply.status < 400,
            status: reply.status,
            json: async () => reply.body,
          };
        },
      },
      "../search-images/search-images": {
        stripSearchImageTokens: (text: string) => text,
      },
      "../utils/speech-text": {
        markdownToSpeechText: (text: string) => text,
      },
      "../stores/external-providers-store": {
        useExternalProvidersStore: { getState: () => ({}) },
      },
      "../api/providers-api": { encryptProviderApiKey: async () => "" },
      "../external-providers": { getExternalProviderApiKey: () => "" },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: {
          getState: () => ({
            ttsStudioVoiceId: voiceId,
            setTtsStudioVoiceId: (value: string) => setVoiceIds.push(value),
          }),
        },
      },
      "@/lib/toast": { toast: { error: () => {} } },
    },
  );
  return { adapter, posted, setVoiceIds };
}

test("read aloud keeps its clip out of history and speaks in the model's voice by default", async () => {
  const { adapter, posted } = load("");
  assert.equal(
    await adapter.generateStudioTtsAudio("hello"),
    "data:audio/wav;base64,UklGRg==",
  );
  assert.deepEqual(posted, [
    {
      messages: [{ role: "user", content: "hello" }],
      stream: false,
      persist: false,
    },
  ]);
});

test("read aloud sends the saved voice chosen in Settings", async () => {
  const { adapter, posted } = load("8f2c");
  await adapter.generateStudioTtsAudio("hello");
  assert.equal(posted[0].voice_id, "8f2c");
  assert.equal(posted[0].persist, false);
});

test("a deleted saved voice goes back to the model's own voice", async () => {
  const { adapter, setVoiceIds } = load("8f2c", {
    status: 404,
    body: { detail: "That saved voice no longer exists." },
  });
  await assert.rejects(
    adapter.generateStudioTtsAudio("hello"),
    /no longer exists.*model's own voice/,
  );
  assert.deepEqual(setVoiceIds, [""]);
});

test("a 404 that is not about the voice keeps the choice", async () => {
  const { adapter, setVoiceIds } = load("8f2c", {
    status: 404,
    body: { detail: "Model not found" },
  });
  await assert.rejects(
    adapter.generateStudioTtsAudio("hello"),
    /Model not found/,
  );
  assert.deepEqual(setVoiceIds, []);
});

test("a model that cannot clone says how to fix it, and keeps the choice", async () => {
  const { adapter, setVoiceIds } = load("8f2c", {
    status: 400,
    body: { detail: "Load a model that can clone a voice." },
  });
  await assert.rejects(
    adapter.generateStudioTtsAudio("hello"),
    /can't use saved voices.*Settings → Voice/,
  );
  assert.deepEqual(setVoiceIds, []);
});

test("no model loaded points at Audio instead of a model name", async () => {
  const { adapter } = load("", {
    status: 400,
    body: { detail: "No model loaded." },
  });
  await assert.rejects(
    adapter.generateStudioTtsAudio("hello"),
    /^Error: No speech model is loaded\. Open Audio/,
  );
});

test("the voice setting persists and Settings lists saved voices for the studio engine", () => {
  const store = readSrc("features/settings/stores/voice-settings-store.ts");
  assert.match(
    store,
    /ttsStudioVoiceId: asString\(saved\?\.ttsStudioVoiceId, ""\),/,
  );
  const tab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    tab,
    /if \(effectiveTtsEngine === "studio"\) \{\s*void useAudioVoicesStore\.getState\(\)\.refresh\(\);/,
  );
  // A voice deleted on the Audio page is cleared once the list is known.
  assert.match(
    tab,
    /if \(ttsStudioVoiceId && savedVoicesListed && !hasSelectedStudioVoice\) \{\s*setTtsStudioVoiceId\(""\);/,
  );
  assert.match(
    tab,
    /setTtsStudioVoiceId\(value === "model" \? "" : value\)[\s\S]*?studioVoiceDefault[\s\S]*?savedVoices\.map\(/,
  );
  // The stored choice stays visible while the list is loading or failed to load.
  assert.match(tab, /value=\{ttsStudioVoiceId \|\| "model"\}/);
  assert.match(
    tab,
    /\{ttsStudioVoiceId && !hasSelectedStudioVoice \? \(\s*<SelectItem value=\{ttsStudioVoiceId\}>\s*\{t\("settings\.voice\.readAloud\.studioVoiceSaved"\)\}/,
  );
});

test("no read-aloud copy names a model the user may not have", () => {
  const dir = new URL("../src/i18n/locales/", import.meta.url);
  for (const file of readdirSync(dir)) {
    assert.doesNotMatch(readSrc(`i18n/locales/${file}`), /Orpheus/, file);
  }
  assert.doesNotMatch(
    readSrc("features/chat/adapters/studio-speech-synthesis-adapter.ts"),
    /Orpheus/,
  );
});
