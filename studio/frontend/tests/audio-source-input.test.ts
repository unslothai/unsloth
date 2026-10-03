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
  });
  assert.deepEqual(state.status, {
    phase: "uploading",
    name: "voice.wav",
    progress: 0,
  });
  // Drawn from the local decode before the upload finishes.
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local",
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
  // The drawing carries over to the uploaded id, so it is not fetched again.
  assert.equal(state.preview.key, "input:abc");
  assert.deepEqual(state.preview.peaks, [0.2, 1]);
});

test("a local decode that finishes after a fast upload still draws the card", () => {
  let state = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "voice.wav",
  });
  // The upload wins the race against the browser's decode.
  state = audioSourceReducer(state, { type: "upload-done", key: "input:abc" });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local",
    peaks: [0.5, 1],
    durationS: 4.7,
    url: "blob:x",
  });
  assert.equal(state.preview.key, "input:abc");
  assert.deepEqual(state.preview.peaks, [0.5, 1]);
  assert.equal(state.preview.durationS, 4.7);
  // A later pick moves on, and an old local decode no longer draws over it.
  state = audioSourceReducer(state, { type: "load-start", key: "voice:v1" });
  state = audioSourceReducer(state, {
    type: "preview",
    key: "local",
    peaks: [0.1],
    durationS: 1,
    url: "blob:y",
  });
  assert.equal(state.preview.key, "voice:v1");
  assert.equal(state.preview.peaks, null);
});

test("a failed upload shows its reason; an expired one says so", () => {
  const uploading = audioSourceReducer(INITIAL_AUDIO_SOURCE_STATE, {
    type: "upload-start",
    name: "a.mp3",
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
  // The hook checks before uploading.
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
  // The shared card: squircle, 4xl radius, a one-pixel ring and no shadow.
  assert.match(
    card,
    /corner-squircle grid gap-3 rounded-4xl bg-card p-4 ring-1/,
  );
  assert.doesNotMatch(card, /shadow-/);
  // Duration and waveform come from a local decode, started before the upload.
  assert.ok(
    hook.indexOf('void drawBlob("local", file)') <
      hook.indexOf("await uploadAudioInput("),
  );
  assert.match(hook, /decodeAudioData/);
});

test("a history pick carries a transcript only from Speak and Clone clips", () => {
  assert.match(
    card,
    /SPOKEN_PROMPT_WORKFLOWS[^=]*=\s*new Set\(\[\s*"speak",\s*"clone",?\s*\]\)/,
  );
  assert.match(
    card,
    /transcript: SPOKEN_PROMPT_WORKFLOWS\.has\(\s*clipWorkflow\(clip\),?\s*\)\s*\?\s*clip\.prompt \|\| null\s*:\s*null/,
  );
});

test("a new selection clears the old one's error or expiry so it loads", () => {
  assert.match(
    hook,
    /if \(changed && \(phase === "error" \|\| phase === "expired"\)\) \{\s*dispatch\(\{ type: "reset" \}\);/,
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

test("a microphone granted after the card unmounted is released, not recorded", () => {
  assert.match(
    hook,
    /if \(!mounted\.current\) \{\s*for \(const track of stream\.getTracks\(\)\) track\.stop\(\);\s*return;/,
  );
  assert.match(hook, /return \(\) => \{\s*mounted\.current = false;/);
});

test("an upload that replaces a selection clears it, so a failed upload cannot run the old one", () => {
  assert.match(
    hook,
    /dispatch\(\{ type: "upload-start", name: fileName \}\);\s*(\/\/[^\n]*\n\s*)?onChangeRef\.current\(null\);/,
  );
});
