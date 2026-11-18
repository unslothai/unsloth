// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const { changedOptionCount, voiceModeOptionGroups, voiceModeRequestOptions } = await import(
  "../src/features/chat/voice/voice-tts-options.ts"
);
const { parseAudioOptions } = await import("../src/features/audio/audio-options.ts");

// As the server reports them (audio_cpp_backend.model_info_fields), trimmed.
const KOKORO = parseAudioOptions([
  { name: "voice", type: "enum", values: ["af_heart", "am_adam"], default: "af_heart" },
  { name: "speed", type: "float", min: 0.5, max: 2, default: 1 },
  { name: "text_chunk_size", type: "int", min: 1, max: 4000 },
  { name: "seed", type: "int", min: 0, max: 2147483647 },
]);
const OMNIVOICE = parseAudioOptions([
  { name: "num_inference_steps", type: "int", min: 1, max: 64, default: 32 },
  { name: "guidance_scale", type: "float", min: 0, max: 10, default: 2 },
  { name: "speed", type: "float", min: 0.5, max: 2, default: 1 },
  { name: "duration_sec", type: "float", min: 0, max: 60 },
  { name: "audio_chunk_duration_sec", type: "float", min: 1, max: 60, default: 15 },
  { name: "audio_chunk_threshold_sec", type: "float", min: 1, max: 60, default: 30 },
  { name: "text_chunk_mode", type: "enum", values: ["word_budget", "tag_aware"], default: "tag_aware" },
  { name: "denoise", type: "bool", default: true },
]);

const names = (specs: { name: string }[]) => specs.map((s) => s.name);

test("Kokoro shows only its speed: the voice is the speaker list, seed and chunking are voice mode's", () => {
  const { primary, more } = voiceModeOptionGroups(KOKORO);
  assert.deepEqual(names(primary), ["speed"]);
  assert.deepEqual(names(more), []);
});

test("an open-ended speed becomes a 0.5 to 2 slider resting at 1, without the spec's API wording", () => {
  const [speed] = voiceModeOptionGroups(
    parseAudioOptions([{ name: "speed", type: "float", default: 1, description: "Must be finite and positive." }]),
  ).primary;
  assert.equal(speed.min, 0.5);
  assert.equal(speed.max, 2);
  assert.equal(speed.default, 1);
  assert.equal(speed.description, null);
  // A declared range is the model's own and stays.
  const [own] = voiceModeOptionGroups(parseAudioOptions([{ name: "speed", type: "float", min: 0.7, max: 1.3, default: 1 }])).primary;
  assert.equal(own.min, 0.7);
  assert.equal(own.max, 1.3);
});

test("a model with many knobs keeps speed up front and the rest under More options", () => {
  const { primary, more } = voiceModeOptionGroups(OMNIVOICE);
  assert.deepEqual(names(primary), ["speed"]);
  assert.deepEqual(names(more), ["num_inference_steps", "guidance_scale", "denoise"]);
});

test("a required option with no default is never tucked under More options", () => {
  const specs = parseAudioOptions([
    { name: "style", type: "string", required: true },
    { name: "temperature", type: "float", min: 0, max: 2, default: 0.7 },
  ]);
  const { primary, more } = voiceModeOptionGroups(specs);
  assert.deepEqual(names(primary), ["style"]);
  assert.deepEqual(names(more), ["temperature"]);
});

test("what voice mode sends drops the options it owns and sends nothing when nothing is left", () => {
  assert.deepEqual(voiceModeRequestOptions({ speed: 1.25, voice: "am_adam", seed: 7, language: "fr" }), {
    speed: 1.25,
    language: "fr",
  });
  assert.equal(voiceModeRequestOptions({ voice: "am_adam" }), undefined);
  assert.equal(voiceModeRequestOptions(undefined), undefined);
});

test("only values away from the default count as changed", () => {
  assert.equal(changedOptionCount(OMNIVOICE, { guidance_scale: 2, denoise: false, speed: 1.5 }), 2);
  assert.equal(changedOptionCount(OMNIVOICE, undefined), 0);
});
