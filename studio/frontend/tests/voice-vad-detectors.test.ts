// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CPP_REPO,
  AUDIO_CPP_UNOFFERED_FOLDERS,
  audioCppModelFor,
} from "../src/features/audio/audio-cpp-catalog.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const PULSEVAD = `${AUDIO_CPP_REPO}/PulseVAD-GGUF`;

const load = (status: unknown = {}, ok = true) =>
  loadWithStubs<typeof import("../src/features/chat/voice/vad-client.ts")>(
    new URL("../src/features/chat/voice/vad-client.ts", import.meta.url),
    {
      "@/features/audio/audio-cpp-catalog": { AUDIO_CPP_REPO },
      "@/features/auth": {
        authFetch: () =>
          Promise.resolve(new Response(JSON.stringify(status), { status: ok ? 200 : 503 })),
      },
    },
  );

test("a detector is named by the server's label, and readably before the server answers", () => {
  const vad = load();
  assert.equal(vad.VOICE_DEFAULT_VAD_MODEL, "builtin/earshot-vad");
  const listed = [{ id: "builtin/earshot-vad", label: "Earshot VAD", available: true }];
  assert.equal(vad.vadDisplayName("builtin/earshot-vad", listed), "Earshot VAD");
  assert.equal(vad.vadDisplayName("builtin/earshot-vad"), "Earshot VAD");
  assert.equal(vad.vadDisplayName("builtin/silero-vad"), "Silero VAD");
  assert.equal(vad.vadDisplayName(PULSEVAD), "PulseVAD");
});

test("the built-in list comes from the status route; a failed call lists none", async () => {
  const detectors = [
    { id: "builtin/earshot-vad", label: "Earshot VAD", available: true },
    { id: "builtin/silero-vad", label: "Silero VAD", available: false },
  ];
  assert.deepEqual(await load({ detectors }).fetchVadDetectors(), detectors);
  assert.deepEqual(await load({ loaded_model: null }).fetchVadDetectors(), []);
  assert.deepEqual(await load({ detail: "down" }, false).fetchVadDetectors(), []);
});

test("a built-in this install cannot run falls back; a Hub pick that fails does not", () => {
  const vad = load();
  // An audio runtime older than Earshot still has the Hub detector.
  assert.equal(vad.vadFallback("builtin/earshot-vad"), PULSEVAD);
  // A saved pick whose weights went away with a runtime update.
  assert.equal(vad.vadFallback("builtin/silero-vad"), "builtin/earshot-vad");
  assert.equal(vad.vadFallback("Builtin/Silero-VAD"), "builtin/earshot-vad");
  assert.equal(vad.vadFallback(PULSEVAD), null);
});

test("a clip is held to its detector's own bar, the shorter one for a barge-in probe", () => {
  const vad = load();
  const earshot = { utterance_ms: 200, barge_ms: 150 };
  assert.equal(vad.minSpeechMs(earshot, false), 200);
  assert.equal(vad.minSpeechMs(earshot, true), 150);
  // A server that sends no bars: the Hub detectors', and the probe's is the one its clip can meet.
  assert.equal(vad.minSpeechMs(undefined, false), 300);
  assert.equal(vad.minSpeechMs(undefined, true), 200);
});

test("the picker offers the audio.cpp detectors the runtime loads", () => {
  for (const folder of ["PulseVAD-GGUF", "FireRedVAD-GGUF"]) {
    assert.equal(audioCppModelFor(`${AUDIO_CPP_REPO}/${folder}`)?.task, "vad", folder);
    assert.ok(!(folder in AUDIO_CPP_UNOFFERED_FOLDERS), folder);
  }
});
