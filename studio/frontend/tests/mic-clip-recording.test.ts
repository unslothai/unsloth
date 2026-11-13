// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import * as dictationOutcome from "../src/features/chat/adapters/dictation-outcome.ts";
import * as micClipState from "../src/features/chat/adapters/mic-clip-state.ts";
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
  static rate = 16_000;
  sampleRate = FakeRecorder.rate;
  limitBytes = 0;
  bytesFor(seconds: number) {
    return 44 + Math.ceil(seconds * this.sampleRate) * 2;
  }
  maxBytes = Number.POSITIVE_INFINITY;
  secondsWithin(maxBytes: number) {
    this.limitBytes = maxBytes;
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
let micOpens = 0;
let micDeviceId = "default";
Object.defineProperty(globalThis, "navigator", {
  value: {
    mediaDevices: {
      getUserMedia: async () => {
        micOpens += 1;
        return stream;
      },
    },
  },
  configurable: true,
});

const { discardClipOnRelease, discardClipWhenModelChanges, listenForAudioClip } =
  loadWithStubs<
  typeof import("../src/features/chat/adapters/studio-audio-clip-adapter.ts")
>(
  new URL("../src/features/chat/adapters/studio-audio-clip-adapter.ts", import.meta.url),
  {
    "@/features/settings/stores/voice-settings-store": {
      useVoiceSettingsStore: { getState: () => ({ micDeviceId }) },
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
    "./mic-clip-state": micClipState,
    "./pcm-recorder": { PcmRecorder: FakeRecorder },
    "./studio-web-speech-dictation-adapter": {
      isMissingDeviceError: (error: unknown) =>
        (error as Error)?.name === "NotFoundError",
    },
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
    assert.equal(lastRecorder?.limitBytes, MAX);
    mock.timers.tick(29_999);
    assert.equal(lastRecorder?.state, "recording");
    mock.timers.tick(1);
    await flush();
    assert.equal(lastRecorder?.state, "inactive");
    assert.equal(clips.length, 1);
    assert.equal(clips[0].type, "audio/wav");
    assert.equal(session.status.type, "ended");
    assert.equal(dictationOutcome.dictationProducedClip(), clips[0]);
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

test("a recording stops within the room left by earlier attachments", async () => {
  const session = listenForAudioClip(() => {}, { maxBytes: 1_000_000 });
  await flush();
  assert.equal(lastRecorder?.limitBytes, 1_000_000);
  session.cancel();
});

test("no room left refuses before opening the mic", async () => {
  toasts.length = 0;
  lastRecorder = null;
  const opensBefore = micOpens;
  let ended = false;
  const session = listenForAudioClip(() => {}, { reason: "No room." });
  session.onEnd?.(() => {
    ended = true;
  });
  await flush();
  assert.equal(micOpens, opensBefore);
  assert.equal(lastRecorder, null);
  assert.equal(ended, true);
  assert.deepEqual(session.status, { type: "ended", reason: "error" });
  assert.deepEqual(toasts, ["No room."]);
});

test("other audio is held off while recording, and the clip is capped at the room", async () => {
  let recordingAtClip: boolean | null = null;
  const session = listenForAudioClip(
    () => {
      recordingAtClip = micClipState.micClipRecording();
    },
    { maxBytes: 500_000 },
  );
  assert.equal(micClipState.micClipRecording(), true);
  await flush();
  assert.equal(lastRecorder?.maxBytes, 500_000);
  await session.stop();
  assert.equal(recordingAtClip, false);
  assert.equal(micClipState.micClipRecording(), false);
});

test("cancel and a refused start leave other audio free to attach", async () => {
  const session = listenForAudioClip(() => {});
  await flush();
  session.cancel();
  assert.equal(micClipState.micClipRecording(), false);
  listenForAudioClip(() => {}, { reason: "No room." });
  assert.equal(micClipState.micClipRecording(), false);
  await flush();
});

test("a stale session's late cleanup keeps a newer recording's hold", async () => {
  let resolveFirst: (value: unknown) => void = () => {};
  const nav = globalThis.navigator as unknown as {
    mediaDevices: { getUserMedia: () => Promise<unknown> };
  };
  const original = nav.mediaDevices.getUserMedia;
  nav.mediaDevices.getUserMedia = () =>
    new Promise((resolve) => {
      resolveFirst = resolve;
    });
  const first = listenForAudioClip(() => {});
  first.cancel();
  nav.mediaDevices.getUserMedia = original;
  const second = listenForAudioClip(() => {});
  await flush();
  assert.equal(micClipState.micClipRecording(), true);
  // The first mic request resolves after the user already started again.
  resolveFirst(stream);
  await flush();
  assert.equal(micClipState.micClipRecording(), true);
  second.cancel();
  assert.equal(micClipState.micClipRecording(), false);
});

test("a cancel during the saved-mic request skips the default-mic fallback", async () => {
  const nav = globalThis.navigator as unknown as {
    mediaDevices: { getUserMedia: () => Promise<unknown> };
  };
  const original = nav.mediaDevices.getUserMedia;
  let rejectSaved: (error: Error) => void = () => {};
  let requests = 0;
  nav.mediaDevices.getUserMedia = () => {
    requests += 1;
    return requests === 1
      ? new Promise((_resolve, reject) => {
          rejectSaved = reject;
        })
      : Promise.resolve(stream);
  };
  micDeviceId = "unplugged-usb-mic";
  toasts.length = 0;
  try {
    const session = listenForAudioClip(() => {});
    session.cancel();
    rejectSaved(Object.assign(new Error("gone"), { name: "NotFoundError" }));
    await flush();
    assert.equal(requests, 1);
    assert.deepEqual(toasts, []);
    assert.deepEqual(session.status, { type: "ended", reason: "cancelled" });
  } finally {
    nav.mediaDevices.getUserMedia = original;
    micDeviceId = "default";
  }
});

test("a cancelled session's late mic error leaves the next session's outcome alone", async () => {
  const nav = globalThis.navigator as unknown as {
    mediaDevices: { getUserMedia: () => Promise<unknown> };
  };
  const original = nav.mediaDevices.getUserMedia;
  let rejectFirst: (error: Error) => void = () => {};
  let requests = 0;
  nav.mediaDevices.getUserMedia = () => {
    requests += 1;
    return requests === 1
      ? new Promise((_resolve, reject) => {
          rejectFirst = reject;
        })
      : Promise.resolve(stream);
  };
  toasts.length = 0;
  try {
    const first = listenForAudioClip(() => {});
    first.cancel();
    const second = listenForAudioClip(() => {});
    await flush();
    assert.equal(second.status.type, "running");
    rejectFirst(Object.assign(new Error("busy"), { name: "NotReadableError" }));
    await flush();
    assert.equal(dictationOutcome.dictationFailed(), false);
    assert.deepEqual(toasts, []);
    second.cancel();
  } finally {
    nav.mediaDevices.getUserMedia = original;
  }
});

function fakeCheckpointStore(checkpoint: string) {
  let state: {
    params: { checkpoint: string };
    modelLoading?: boolean;
    loadingModelPick?: unknown;
  } = { params: { checkpoint } };
  const listeners = new Set<(s: typeof state) => void>();
  return {
    getState: () => state,
    subscribe: (listener: (s: typeof state) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    set(next: string, extra: { modelLoading?: boolean; loadingModelPick?: unknown } = {}) {
      state = { params: { checkpoint: next }, ...extra };
      for (const listener of [...listeners]) listener(state);
    },
    listeners,
  };
}

test("switching the chat model mid-recording discards the clip with a toast", async () => {
  toasts.length = 0;
  const clips: File[] = [];
  const store = fakeCheckpointStore("gemma-4-e2b");
  const session = listenForAudioClip((file) => clips.push(file));
  discardClipWhenModelChanges(session, store);
  await flush();
  store.set("gemma-4-e2b");
  assert.equal(session.status.type, "running");
  store.set("qwen3-text-only");
  await flush();
  assert.deepEqual(session.status, { type: "ended", reason: "cancelled" });
  assert.equal(lastRecorder?.discarded, true);
  assert.equal(clips.length, 0);
  assert.deepEqual(toasts, ["Recording discarded because the chat model changed."]);
  assert.equal(store.listeners.size, 0);
});

test("a finished recording stops watching the chat model", async () => {
  toasts.length = 0;
  const store = fakeCheckpointStore("gemma-4-e2b");
  const session = listenForAudioClip(() => {});
  discardClipWhenModelChanges(session, store);
  await flush();
  await session.stop();
  assert.equal(store.listeners.size, 0);
  store.set("qwen3-text-only");
  assert.deepEqual(toasts, []);
});

test("a model load starting mid-recording discards the clip before checkpoint moves", async () => {
  for (const extra of [
    { modelLoading: true },
    { loadingModelPick: { id: "qwen3-text-only" } },
  ]) {
    toasts.length = 0;
    const store = fakeCheckpointStore("gemma-4-e2b");
    const session = listenForAudioClip(() => {});
    discardClipWhenModelChanges(session, store);
    await flush();
    store.set("gemma-4-e2b", extra);
    await flush();
    assert.deepEqual(session.status, { type: "ended", reason: "cancelled" });
    assert.deepEqual(toasts, ["Recording discarded because the chat model changed."]);
  }
});

test("a room sized for 16 kHz is refused when the recorder opens at device rate", async () => {
  toasts.length = 0;
  FakeRecorder.rate = 48_000;
  try {
    const clips: File[] = [];
    const session = listenForAudioClip((file) => clips.push(file), {
      maxBytes: 44 + 16_000 * 2,
    });
    await flush();
    assert.deepEqual(session.status, { type: "ended", reason: "error" });
    assert.equal(lastRecorder?.discarded, true);
    assert.equal(lastRecorder?.stopCalls, 0);
    assert.equal(clips.length, 0);
    assert.deepEqual(toasts, ["Audio files together exceed the 25MB per-message limit"]);
    assert.equal(micClipState.micClipRecording(), false);
  } finally {
    FakeRecorder.rate = 16_000;
  }
});

test("the same room is enough at 16 kHz", async () => {
  const session = listenForAudioClip(() => {}, { maxBytes: 44 + 16_000 * 2 });
  await flush();
  assert.equal(session.status.type, "running");
  session.cancel();
});

test("switching chats mid-recording discards the clip instead of attaching it elsewhere", async () => {
  toasts.length = 0;
  const clips: File[] = [];
  const onRelease = new Set<() => void>();
  const session = listenForAudioClip((file) => clips.push(file));
  discardClipOnRelease(session, onRelease);
  await flush();
  assert.equal(session.status.type, "running");
  for (const listener of [...onRelease]) listener();
  await flush();
  assert.deepEqual(session.status, { type: "ended", reason: "cancelled" });
  assert.equal(lastRecorder?.discarded, true);
  assert.equal(clips.length, 0);
  assert.deepEqual(toasts, ["Recording discarded because the chat changed."]);
  assert.equal(onRelease.size, 0);
});

test("a finished recording is not discarded by a later chat switch", async () => {
  toasts.length = 0;
  const clips: File[] = [];
  const onRelease = new Set<() => void>();
  const session = listenForAudioClip((file) => clips.push(file));
  discardClipOnRelease(session, onRelease);
  await flush();
  await session.stop();
  assert.equal(clips.length, 1);
  assert.equal(onRelease.size, 0);
  assert.deepEqual(toasts, []);
});
