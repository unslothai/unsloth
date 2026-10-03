// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  audioFileProblem,
  audioSourceReducer,
  INITIAL_AUDIO_SOURCE_STATE,
  REFERENCE_EXPIRED_MESSAGE,
} = await import("../src/features/audio/hooks/audio-source-state.ts");
const { AUDIO_INPUT_MAX_BYTES, selectionExpired } = await import(
  "../src/features/audio/audio-run-request.ts"
);

const card = readSrc("features/audio/components/audio-source-input.tsx");
const hook = readSrc("features/audio/hooks/use-audio-source.ts");

test("a picked file uploads with progress, draws at once, then is ready", () => {
  let state = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "voice.wav",
    key: "local:1",
  });
  assert.deepEqual(state.status, {
    phase: "uploading",
    name: "voice.wav",
    progress: 0,
  });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local:1",
    peaks: [0.2, 1],
    durationS: 4.7,
    url: "blob:x",
  });
  assert.equal(state.status.phase, "uploading");
  assert.equal(state.preview.durationS, 4.7);
  state = audioSourceReducer(state, { type: "upload-progress", progress: 0.4 });
  assert.deepEqual(state.status, {
    phase: "uploading",
    name: "voice.wav",
    progress: 0.4,
  });
  state = audioSourceReducer(state, { type: "upload-progress", progress: 7 });
  assert.equal(state.status.phase === "uploading" && state.status.progress, 1);
  state = audioSourceReducer(state, { type: "upload-done", key: "input:abc" });
  assert.equal(state.status.phase, "ready");
  assert.equal(state.preview.key, "input:abc");
  assert.deepEqual(state.preview.peaks, [0.2, 1]);
});

test("a local decode that finishes after a fast upload still draws the card", () => {
  let state = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "voice.wav",
    key: "local:1",
  });
  state = audioSourceReducer(state, { type: "upload-done", key: "input:abc" });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local:1",
    peaks: [0.5, 1],
    durationS: 4.7,
    url: "blob:x",
  });
  assert.equal(state.preview.key, "input:abc");
  assert.deepEqual(state.preview.peaks, [0.5, 1]);
  assert.equal(state.preview.durationS, 4.7);
  state = audioSourceReducer(state, { type: "load-start", key: "voice:v1" });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local:1",
    peaks: [0.1],
    durationS: 1,
    url: "blob:y",
  });
  assert.equal(state.preview.key, "voice:v1");
  assert.equal(state.preview.peaks, null);
});

test("a slow decode of an earlier pick does not draw on the next one", () => {
  let state = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "first.wav",
    key: "local:1",
  });
  state = audioSourceReducer(state, { type: "upload-done", key: "input:a" });
  state = audioSourceReducer(state, {
    type: "upload-start",
    name: "second.wav",
    key: "local:2",
  });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local:1",
    peaks: [0.9],
    durationS: 9,
    url: "blob:first",
  });
  assert.equal(state.preview.key, "local:2");
  assert.equal(state.preview.peaks, null);
  assert.equal(state.preview.url, null);
});

test("a failed upload shows its reason; an expired one says so", () => {
  const uploading = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "a.mp3",
    key: "local:1",
  });
  assert.deepEqual(
    audioSourceReducer(uploading, {
      type: "fail",
      message: "Audio is too large.",
    }).status,
    {
      phase: "error",
      message: "Audio is too large.",
    },
  );
  const expired = audioSourceReducer(uploading, { type: "expire" });
  assert.equal(expired.status.phase, "expired");
  assert.equal(
    REFERENCE_EXPIRED_MESSAGE,
    "This reference expired. Add it again.",
  );
  assert.match(card, /REFERENCE_EXPIRED_MESSAGE/);
  assert.match(card, /Add it again/);
  assert.match(card, /Try another file/);
});

test("a decode that lands after the card moved on draws nothing", () => {
  const loading = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "load-start",
    key: "clip:1",
  });
  const stale = audioSourceReducer(loading, {
    type: "preview",
    key: "clip:0",
    peaks: [1],
    durationS: 1,
    url: "blob:old",
  });
  assert.equal(stale, loading);
});

test("recording starts and stops back to the empty card", () => {
  const recording = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "record-start",
    now: 5,
  });
  assert.deepEqual(recording.status, { phase: "recording", startedAt: 5 });
  assert.equal(
    audioSourceReducer(recording, { type: "record-stop" }).status.phase,
    "idle",
  );
  assert.match(hook, /createAudioRecorder\(stream\)/);
});

test("a too-large or empty file is refused before any upload, at the server's 200 MiB", () => {
  assert.equal(AUDIO_INPUT_MAX_BYTES, 200 * 1024 * 1024);
  assert.equal(
    audioFileProblem({ size: AUDIO_INPUT_MAX_BYTES, type: "audio/wav" }),
    null,
  );
  assert.equal(
    audioFileProblem({ size: AUDIO_INPUT_MAX_BYTES + 1, type: "audio/wav" }),
    "Audio is too large. Keep it under 200 MB.",
  );
  assert.equal(
    audioFileProblem({ size: 0, type: "audio/wav" }),
    "This file is empty.",
  );
  assert.match(
    audioFileProblem({ size: 10, type: "image/png", name: "a.png" }) ?? "",
    /not an audio file/,
  );
  assert.equal(
    audioFileProblem({ size: 10, type: "", name: "take.m4a" }),
    null,
  );
  assert.ok(
    hook.indexOf("audioFileProblem(") < hook.indexOf("uploadAudioInput(file"),
  );
});

test("an upload past its keep-until time is expired", () => {
  const reference = {
    kind: "input" as const,
    id: "i",
    name: "a",
    durationS: 1,
    expiresAt: "2026-10-01T00:00:00Z",
  };
  assert.equal(
    selectionExpired(reference, Date.parse("2026-10-02T00:00:00Z")),
    true,
  );
  assert.equal(
    selectionExpired(reference, Date.parse("2026-09-30T00:00:00Z")),
    false,
  );
  assert.equal(
    selectionExpired(
      { ...reference, kind: "voice" },
      Date.parse("2027-01-01T00:00:00Z"),
    ),
    false,
  );
});

test("the whole card is the drop target, with every way in", () => {
  const section = card.slice(card.indexOf("<section"), card.indexOf("<input"));
  assert.match(section, /onDragOver=/);
  assert.match(section, /onDrop=/);
  assert.match(section, /nativeDropRef\(element\)/);
  assert.match(card, /type="file"/);
  for (const tab of ["Upload", "Record", "From history", "Saved voice"]) {
    assert.ok(card.includes(`label: "${tab}"`), tab);
  }
  assert.match(
    card,
    /corner-squircle grid gap-3 rounded-4xl bg-card p-4 ring-1/,
  );
  assert.doesNotMatch(card, /shadow-/);
  assert.ok(
    hook.indexOf('void drawBlob("local", file)') <
      hook.indexOf("await uploadAudioInput("),
  );
  assert.match(hook, /await decodePeaks\(blob\)/);
});

test("a superseded microphone request releases its stream", () => {
  const hook = readSrc("features/audio/hooks/use-audio-source.ts");
  assert.match(
    hook,
    /if \(recorderRef\.current \|\| acquiring\.current\) return;/,
  );
  assert.match(
    hook,
    /if \(ticket !== acquisition\.current \|\| !activeRef\.current\) \{\s*for \(const track of stream\.getTracks\(\)\) track\.stop\(\);/,
  );
  // Leaving the Audio page ends a recording.
  assert.match(
    hook,
    /activeRef\.current = active;\s*if \(!active\) stopRecording\(\);/,
  );
  assert.match(
    card,
    /useAudioSource\(\{ value, onChange, maxRecordSeconds, active \}\)/,
  );
  assert.match(
    hook,
    /const abortAll = useCallback\(\(\) => \{\s*acquisition\.current \+= 1;/,
  );
});

test("the chooser accepts MP4 like the drop check does", () => {
  const input = readSrc("features/audio/components/audio-source-input.tsx");
  assert.match(input, /const AUDIO_EXTS = "[^"]*\bmp4\b/);
});

test("picking a new source clears an earlier error", () => {
  const hook = readSrc("features/audio/hooks/use-audio-source.ts");
  assert.match(
    hook,
    /seenKey\.current = valueKey;\s*if \(valueKey && \(phase === "error" \|\| phase === "expired"\)\)\s*dispatch\(\{ type: "reset" \}\);/,
  );
});

test("a history pick carries a transcript only from Speak and Clone clips", async () => {
  assert.match(card, /onChange\(clipReference\(clip, clipWorkflow\(clip\)\)\)/);
  const { clipReference } = await import(
    "../src/features/audio/audio-run-request.ts"
  );
  const clip = { id: "c", prompt: "Hello there.", duration_s: 2 };
  assert.equal(clipReference(clip, "speak").transcript, "Hello there.");
  assert.equal(clipReference(clip, "clone").transcript, "Hello there.");
  // Convert and Music prompts are labels, not what the clip says.
  assert.equal(clipReference(clip, "convert").transcript, null);
  assert.equal(clipReference(clip, "music").transcript, null);
});

test("a new selection clears the old one's error or expiry so it loads", () => {
  assert.match(
    hook,
    /if \(valueKey && \(phase === "error" \|\| phase === "expired"\)\)\s*dispatch\(\{ type: "reset" \}\);/,
  );
  const failed = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "fail",
    message: "x",
  });
  assert.deepEqual(
    audioSourceReducer(failed, { type: "reset" }),
    INITIAL_AUDIO_SOURCE_STATE,
  );
});

test("an upload that replaces a selection clears it, so a failed upload cannot run the old one", () => {
  assert.match(
    hook,
    /dispatch\(\{ type: "upload-start", name: fileName, key: localKey \}\);\s*(\/\/[^\n]*\n\s*)?onChangeRef\.current\(null\);/,
  );
});
