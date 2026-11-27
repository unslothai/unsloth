// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const created: { engine: string; options: Record<string, unknown> }[] = [];
let dictationEngine = "browser";

function fakeAdapter(engine: string) {
  return class {
    static isSupported() {
      return true;
    }
    constructor(options: Record<string, unknown>) {
      created.push({ engine, options });
    }
    listen() {
      return { onSpeechEnd() {}, onEnd() {}, cancel() {} };
    }
  };
}

const { StudioDictationAdapter } = loadWithStubs<{
  StudioDictationAdapter: new (options?: {
    chatId?: string | null;
    language?: string;
    detectLanguage?: boolean;
  }) => { listen(): unknown };
}>(
  new URL(
    "../src/features/chat/adapters/studio-dictation-adapter.tsx",
    import.meta.url,
  ),
  {
    "@/features/settings/stores/stt-download-prompt-store": {
      requestSttDownload() {},
    },
    "@/features/settings/stores/voice-settings-store": {
      sttModelVariant: () => null,
      useVoiceSettingsStore: {
        getState: () => ({ dictationEngine, micTarget: "transcribe" }),
      },
    },
    "@/lib/toast": { toast: { error() {} } },
    "../stores/chat-runtime-store": {
      useChatRuntimeStore: {
        getState: () => ({ params: {}, models: [], modelLoading: false }),
      },
    },
    "../stores/external-providers-store": {
      useExternalProvidersStore: {
        getState: () => ({ connectionsEnabled: false, providers: [] }),
      },
    },
    "@/lib/audio-utils": { maxAudioFilesFor: () => 0 },
    "../utils/mic-target": {
      micClipRoom: () => 0,
      micSendsAudioToModel: () => false,
    },
    "../utils/dictation-entry": {
      currentDictationEntryMode: () => "browser",
      insecureDictationGuidance: () => "",
    },
    "./studio-model-dictation-adapter": {
      StudioModelDictationAdapter: fakeAdapter("model"),
      fetchSttStatus: async () => null,
      sttEngineStatusFor: () => null,
      sttQuantDownloaded: () => false,
    },
    "./pcm-recorder": { PcmRecorder: { isSupported: () => true } },
    "./studio-audio-clip-adapter": {
      discardClipOnRelease() {},
      discardClipWhenModelChanges() {},
      listenForAudioClip() {},
    },
    "./studio-web-speech-dictation-adapter": {
      StudioWebSpeechDictationAdapter: fakeAdapter("browser"),
    },
  },
);

for (const engine of ["browser", "model"]) {
  test(`a session language reaches the ${engine} dictation engine`, () => {
    created.length = 0;
    dictationEngine = engine;
    new StudioDictationAdapter({ chatId: null, language: "es-ES" }).listen();
    assert.equal(created.length, 1);
    assert.equal(created[0].engine, engine);
    assert.equal(created[0].options.language, "es-ES");
  });

  test(`without a session language the ${engine} engine keeps Voice settings`, () => {
    created.length = 0;
    dictationEngine = engine;
    new StudioDictationAdapter({ chatId: null }).listen();
    assert.equal(created[0].options.language, undefined);
  });
}

test("asked to detect the language, only a recording engine is told to", () => {
  created.length = 0;
  dictationEngine = "model";
  new StudioDictationAdapter({ chatId: null, detectLanguage: true }).listen();
  assert.equal(created[0].options.detectLanguage, true);
  created.length = 0;
  dictationEngine = "browser";
  new StudioDictationAdapter({ chatId: null, detectLanguage: true }).listen();
  assert.equal(created[0].options.detectLanguage, undefined);
});

registerBundlerResolver();
installLocalStorageFake();
const { isModelDictationLanguage } = await import(
  "../src/features/settings/stores/voice-settings-store.ts"
);

test("the local engine takes a source language only when Whisper knows it", () => {
  assert.equal(isModelDictationLanguage("small", "es-ES"), true);
  assert.equal(isModelDictationLanguage("small", "fil"), true);
  assert.equal(isModelDictationLanguage("small", "ug"), false);
});
