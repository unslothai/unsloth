// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read aloud on the "Unsloth model" engine speaks with the loaded speech model (named, so
// one kept beside the chat model answers), in a saved Audio voice, and keeps its clips out of
// Speak's history.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

import * as modelCatalog from "../src/features/settings/api/openai-model-catalog.ts";
import * as formatError from "../src/lib/format-fastapi-error.ts";
import { readSrc } from "./helpers/kit.ts";

type Adapter = {
  generateStudioTtsAudio: (
    text: string,
    signal?: AbortSignal,
  ) => Promise<string>;
};

type Reply = { status: number; body: unknown };

const KOKORO = "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF";
const CLONER = "audio-cpp/audio.cpp-gguf/PocketTTS-GGUF";

// Not the real ceiling, so a copy of the number in the adapter fails the body check.
const MAX_TOKENS = 1234;

/** The adapter with the voice setting faked and a fetch that logs each body. */
function load(
  voiceId: string,
  reply: Reply = { status: 200, body: { audio: { data: "UklGRg==" } } },
  // GET /v1/models: the listing, or an Error for a failed request.
  models: unknown = { data: [] },
) {
  const posted: Record<string, unknown>[] = [];
  const setVoiceIds: string[] = [];
  const warnings: string[] = [];
  const adapter = loadWithStubs<Adapter>(
    new URL(
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (input: string, init?: { body?: string }) => {
          if (input === "/v1/models") {
            if (models instanceof Error) throw models;
            return { ok: true, status: 200, json: async () => models };
          }
          posted.push(JSON.parse(init?.body ?? "{}"));
          return {
            ok: reply.status < 400,
            status: reply.status,
            json: async () => reply.body,
          };
        },
      },
      "../../settings/api/openai-model-catalog": modelCatalog,
      "@/lib/format-fastapi-error": formatError,
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
      "@/lib/toast": {
        toast: {
          error: () => {},
          warning: (message: string) => warnings.push(message),
        },
      },
      "@/features/audio/audio-workspace-constants": {
        TTS_MAX_TOKENS: MAX_TOKENS,
      },
    },
  );
  return { adapter, posted, setVoiceIds, warnings };
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
      max_tokens: MAX_TOKENS,
    },
  ]);
});

test("read aloud warns only when the speech stops before the end of the reply", async () => {
  const spoken = async (finishReason: string) => {
    const { adapter, warnings } = load("", {
      status: 200,
      body: {
        audio: { data: "UklGRg==" },
        choices: [{ finish_reason: finishReason }],
      },
    });
    assert.equal(
      await adapter.generateStudioTtsAudio("a reply"),
      "data:audio/wav;base64,UklGRg==",
    );
    return warnings;
  };
  const cut = await spoken("length");
  assert.equal(cut.length, 1);
  assert.match(cut[0], /before the end of the reply/);
  assert.deepEqual(await spoken("stop"), []);
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
    /^Error: No Unsloth speech model is loaded\. Open Audio › Text to Speech/,
  );
});

test("read aloud names the loaded speech model, so one beside the chat model answers", async () => {
  const { adapter, posted } = load("", undefined, {
    data: [
      { id: "unsloth/Qwen3-1.7B-GGUF", loaded: true, task: "text-generation" },
      {
        id: "music",
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["music"],
      },
      {
        id: "on-disk",
        loaded: false,
        task: "text-to-speech",
        audio_workflows: ["speak"],
      },
      {
        id: KOKORO,
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["speak"],
      },
    ],
  });
  await adapter.generateStudioTtsAudio("hello");
  assert.equal(posted[0].model, KOKORO);
});

test("a saved voice prefers a loaded model that can clone", async () => {
  const { adapter, posted } = load("8f2c", undefined, {
    data: [
      {
        id: KOKORO,
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["speak"],
      },
      {
        id: CLONER,
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["speak", "clone"],
      },
    ],
  });
  await adapter.generateStudioTtsAudio("hello");
  assert.equal(posted[0].model, CLONER);
  assert.equal(posted[0].voice_id, "8f2c");
});

test("no listing or no loaded speech model leaves the request on the active model", async () => {
  for (const models of [
    new Error("offline"),
    { data: [] },
    { detail: "nope" },
  ]) {
    const { adapter, posted } = load("", undefined, models);
    await adapter.generateStudioTtsAudio("hello");
    assert.equal("model" in posted[0], false);
  }
});

test("a named speech model that was unloaded says to load it again", async () => {
  // The server nests this refusal ({detail: {error: {message}}}); a plain string read showed
  // "[object Object]".
  const { adapter } = load(
    "",
    {
      status: 404,
      body: {
        detail: {
          error: {
            message: `The model '${KOKORO}' is downloaded but not loaded, and 'Switch model by request' is off.`,
            type: "invalid_request_error",
          },
        },
      },
    },
    {
      data: [
        {
          id: KOKORO,
          loaded: true,
          task: "text-to-speech",
          audio_workflows: ["speak"],
        },
      ],
    },
  );
  await assert.rejects(
    adapter.generateStudioTtsAudio("hello"),
    /^Error: The Unsloth speech model for read aloud is no longer loaded\. Load it again in Audio › Text to Speech/,
  );
});

test("read-aloud errors name the Text to Speech page, not Speak", () => {
  const source = readSrc(
    "features/chat/adapters/studio-speech-synthesis-adapter.ts",
  );
  assert.doesNotMatch(source, /load one in Speak|in Audio, or/);
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

test("a saved voice can read aloud with a loaded clone-only model", async () => {
  const { adapter, posted } = load("8f2c", undefined, {
    data: [
      {
        id: CLONER,
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["clone"],
      },
    ],
  });
  await adapter.generateStudioTtsAudio("hello");
  assert.equal(posted[0].model, CLONER);
});
