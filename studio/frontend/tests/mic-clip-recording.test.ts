// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import * as dictationOutcome from "../src/features/chat/adapters/dictation-outcome.ts";
import * as micTarget from "../src/features/chat/utils/mic-target.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const MAX = 25 * 1024 * 1024;
const toasts: string[] = [];
let lastRecorder: FakeRecorder | null = null;

class FakeRecorder {
  state: RecordingState = "inactive";
  private data: ((event: { data: Blob }) => void)[] = [];
  constructor() {
    lastRecorder = this;
  }
  secondsWithin(maxBytes: number) {
    assert.equal(maxBytes, MAX);
    return 30;
  }
  addEventListener(type: string, listener: (event: { data: Blob }) => void) {
    if (type === "dataavailable") this.data.push(listener);
  }
  start() {
    this.state = "recording";
  }
  stopCalls = 0;
  discarded = false;
  discard() {
    this.state = "inactive";
    this.discarded = true;
  }
  stop() {
    this.stopCalls += 1;
    this.state = "inactive";
    const data = new Blob([new Uint8Array(64)], { type: "audio/wav" });
    for (const listener of this.data) listener({ data });
  }
}

const stream = { getTracks: () => [{ stop() {} }] };
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: { getUserMedia: async () => stream } },
  configurable: true,
});

const { listenForAudioClip } = loadWithStubs<
  typeof import("../src/features/chat/adapters/studio-audio-clip-adapter.ts")
>(
  new URL("../src/features/chat/adapters/studio-audio-clip-adapter.ts", import.meta.url),
  {
    "@/features/settings/stores/voice-settings-store": {
      useVoiceSettingsStore: { getState: () => ({ micDeviceId: "default" }) },
    },
    "@/lib/audio-utils": { MAX_AUDIO_SIZE: MAX, MAX_AUDIO_SIZE_LABEL: "25MB" },
    sonner: {
      toast: {
        info: (message: string) => toasts.push(message),
        error: (message: string) => toasts.push(message),
      },
    },
    "../utils/mic-target": micTarget,
    "./dictation-level": { startDictationLevelMeter: () => () => {} },
    "./dictation-outcome": dictationOutcome,
    "./pcm-recorder": { PcmRecorder: FakeRecorder },
    "./studio-web-speech-dictation-adapter": { isMissingDeviceError: () => false },
  },
);

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("a recording stops itself at the audio attachment limit and keeps the clip", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const clips: File[] = [];
    const session = listenForAudioClip((file) => clips.push(file));
    await flush();
    assert.equal(session.status.type, "running");
    mock.timers.tick(29_999);
    assert.equal(lastRecorder?.state, "recording");
    mock.timers.tick(1);
    await flush();
    assert.equal(lastRecorder?.state, "inactive");
    assert.equal(clips.length, 1);
    assert.equal(clips[0].type, "audio/wav");
    assert.equal(session.status.type, "ended");
    assert.equal(dictationOutcome.dictationProducedClip(), true);
    assert.deepEqual(toasts, ["Recording stopped at the 25MB audio limit."]);
  } finally {
    mock.timers.reset();
  }
});

test("a stop before the limit clears the auto-stop", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    toasts.length = 0;
    const clips: File[] = [];
    const session = listenForAudioClip((file) => clips.push(file));
    await flush();
    await session.stop();
    mock.timers.tick(60_000);
    await flush();
    assert.equal(clips.length, 1);
    assert.deepEqual(toasts, []);
  } finally {
    mock.timers.reset();
  }
});

test("cancel discards the recording", async () => {
  const clips: File[] = [];
  const session = listenForAudioClip((file) => clips.push(file));
  await flush();
  session.cancel();
  await flush();
  assert.equal(clips.length, 0);
  assert.equal(lastRecorder?.stopCalls, 0);
  assert.equal(lastRecorder?.discarded, true);
  assert.deepEqual(session.status, { type: "ended", reason: "cancelled" });
});
