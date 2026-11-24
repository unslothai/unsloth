// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Result = [transcript: string, isFinal: boolean];

class FakeRecognition extends EventTarget {
  static last: FakeRecognition | null = null;
  lang = "";
  continuous = false;
  interimResults = false;

  constructor() {
    super();
    FakeRecognition.last = this;
  }

  start() {
    this.dispatchEvent(new Event("start"));
  }

  stop() {
    this.dispatchEvent(new Event("end"));
  }

  abort() {}

  emit(resultIndex: number, results: Result[]) {
    const event = new Event("result");
    Object.assign(event, {
      resultIndex,
      results: results.map(([transcript, isFinal]) =>
        Object.assign([{ transcript, confidence: 1 }], { isFinal }),
      ),
    });
    this.dispatchEvent(event);
  }
}

const recent: string[] = [];

const track = { readyState: "live", stop() {} };
Object.defineProperty(globalThis, "window", {
  value: {
    SpeechRecognition: FakeRecognition,
    setTimeout,
    clearTimeout,
  },
  configurable: true,
  writable: true,
});
Object.defineProperty(globalThis, "navigator", {
  value: {
    mediaDevices: {
      getUserMedia: async () => ({
        getTracks: () => [track],
        getAudioTracks: () => [track],
      }),
    },
  },
  configurable: true,
  writable: true,
});

const { StudioWebSpeechDictationAdapter } = loadWithStubs<{
  StudioWebSpeechDictationAdapter: new () => {
    listen(): {
      stop(): Promise<void>;
      onSpeech(callback: (result: { transcript: string }) => void): void;
    };
  };
}>(
  new URL(
    "../src/features/chat/adapters/studio-web-speech-dictation-adapter.ts",
    import.meta.url,
  ),
  {
    "@/features/settings/stores/settings-dialog-store": {
      useSettingsDialogStore: { getState: () => ({ openDialog() {} }) },
    },
    "@/features/settings/stores/voice-settings-store": {
      applyDictationDictionary: (text: string) => text,
      recordRecentDictation: (text: string) => recent.push(text),
      resolveDictationLanguage: () => "en-US",
      useVoiceSettingsStore: { getState: () => ({ micDeviceId: "default" }) },
    },
    "@/lib/api-base": { isTauri: false },
    sonner: { toast: { error() {} } },
    "../stores/chat-runtime-store": {
      useChatRuntimeStore: { getState: () => ({ activeThreadId: "chat-1" }) },
    },
    "./dictation-level": { startDictationLevelMeter: () => () => {} },
    "./dictation-outcome": {
      beginDictationSession() {},
      markDictationFailed() {},
      markDictationTranscript() {},
    },
  },
);

async function dictate(events: [number, Result[]][]) {
  recent.length = 0;
  const session = new StudioWebSpeechDictationAdapter().listen();
  const spoken: string[] = [];
  session.onSpeech(({ transcript }) => spoken.push(transcript));
  await new Promise((resolve) => setImmediate(resolve));
  const recognition = FakeRecognition.last;
  assert.ok(recognition);
  for (const [resultIndex, results] of events) {
    recognition.emit(resultIndex, results);
  }
  await session.stop();
  return { spoken, recent: [...recent] };
}

test("a split interim merged into one final result is not repeated on stop", async () => {
  const { spoken, recent } = await dictate([
    [
      0,
      [
        ["what is the capital of", false],
        [" France", false],
      ],
    ],
    [0, [["what is the capital of France", false]]],
    [0, [["what is the capital of France", true]]],
  ]);

  assert.deepEqual(spoken, ["what is the capital of France"]);
  assert.deepEqual(recent, ["what is the capital of France"]);
});

test("a merge after an earlier final utterance keeps both utterances once", async () => {
  const { spoken } = await dictate([
    [0, [["hello there", true]]],
    [
      1,
      [
        ["hello there", true],
        ["book a table for", false],
        [" four", false],
      ],
    ],
    [
      1,
      [
        ["hello there", true],
        ["book a table for four", true],
      ],
    ],
  ]);

  assert.deepEqual(spoken, ["hello there book a table for four"]);
});

test("interim text still left at stop is kept once", async () => {
  const { spoken } = await dictate([
    [
      0,
      [
        ["what is the capital of", false],
        [" France", false],
      ],
    ],
  ]);

  assert.deepEqual(spoken, ["what is the capital of France"]);
});
