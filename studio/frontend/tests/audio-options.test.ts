// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
const {
  AUDIO_OPTIONS_STORAGE_KEY,
  audioOptionDisplayValue,
  audioOptionFloatStep,
  audioOptionLabel,
  audioOptionsForRequest,
  audioVoiceLabel,
  coerceAudioOptionValue,
  missingRequiredAudioOptions,
  parseAudioOptions,
  readAudioOptionValues,
  saveAudioOptionValues,
} = await import("../src/features/audio/audio-options.ts");

const YUE2_OPTIONS = [
  {
    name: "num_inference_steps",
    type: "int",
    description: "Diffusion steps.",
    default: 50,
    min: 1,
    max: 200,
    required: false,
  },
  { name: "guidance_scale", type: "float", default: 1.5, min: 0, max: 10 },
  { name: "cot", type: "enum", values: ["off", "melody", "full"], default: "off" },
  { name: "abc", type: "bool", default: false },
  { name: "stop_after", type: "string" },
  { name: "voice_id", type: "string", required: true },
];

test("the status schema parses, dropping what cannot be rendered", () => {
  const specs = parseAudioOptions([
    ...YUE2_OPTIONS,
    { name: "voice_ref", type: "audio_path" },
    { name: "empty_enum", type: "enum", values: [] },
    { name: "", type: "int" },
    { name: "cot", type: "string" },
    "junk",
    null,
  ]);
  assert.deepEqual(
    specs.map((spec) => [spec.name, spec.type]),
    [
      ["num_inference_steps", "int"],
      ["guidance_scale", "float"],
      ["cot", "enum"],
      ["abc", "bool"],
      ["stop_after", "string"],
      ["voice_id", "string"],
    ],
  );
  assert.equal(specs[0].default, 50);
  assert.equal(specs[2].default, "off");
  assert.equal(specs[3].default, false);
  assert.equal(specs[4].default, null);
  assert.deepEqual(parseAudioOptions(undefined), []);
  assert.deepEqual(parseAudioOptions({ name: "x" }), []);
  const [steps, cot] = parseAudioOptions([
    { name: "steps", type: "int", default: 500, max: 200 },
    { name: "cot", type: "enum", values: ["off"], default: "full" },
  ]);
  assert.equal(steps.default, 200);
  assert.equal(cot.default, null);
});

test("values are coerced to the option's type and range", () => {
  const [steps, guidance, cot, abc, stopAfter] = parseAudioOptions(YUE2_OPTIONS);
  assert.equal(coerceAudioOptionValue(steps, 12.6), 13);
  assert.equal(coerceAudioOptionValue(steps, "400"), 200);
  assert.equal(coerceAudioOptionValue(steps, 0), 1);
  assert.equal(coerceAudioOptionValue(steps, "x"), undefined);
  assert.equal(coerceAudioOptionValue(guidance, 2.25), 2.25);
  assert.equal(coerceAudioOptionValue(cot, "melody"), "melody");
  assert.equal(coerceAudioOptionValue(cot, "bogus"), undefined);
  assert.equal(coerceAudioOptionValue(abc, true), true);
  assert.equal(coerceAudioOptionValue(abc, "true"), undefined);
  assert.equal(coerceAudioOptionValue(stopAfter, "verse"), "verse");
  assert.equal(audioOptionDisplayValue(steps, {}), 50);
  assert.equal(audioOptionDisplayValue(steps, { num_inference_steps: 20 }), 20);
});

test("only set, valid values are sent; unknown names are dropped", () => {
  const specs = parseAudioOptions(YUE2_OPTIONS);
  assert.deepEqual(
    audioOptionsForRequest(specs, {
      num_inference_steps: 999,
      cot: "full",
      abc: true,
      stop_after: "",
      guidance_scale: "high" as unknown as number,
      removed_option: 3,
    }),
    { num_inference_steps: 200, cot: "full", abc: true },
  );
  assert.deepEqual(audioOptionsForRequest(specs, {}), {});
});

test("a required option with no value or default blocks generation", () => {
  const specs = parseAudioOptions(YUE2_OPTIONS);
  assert.deepEqual(
    missingRequiredAudioOptions(specs, {}).map((spec) => spec.name),
    ["voice_id"],
  );
  assert.deepEqual(missingRequiredAudioOptions(specs, { voice_id: "alba" }), []);
  assert.deepEqual(
    missingRequiredAudioOptions(specs, { voice_id: "" }).map((spec) => spec.name),
    ["voice_id"],
  );
});

test("labels and slider steps read naturally", () => {
  assert.equal(audioOptionLabel("num_inference_steps"), "Num inference steps");
  assert.equal(audioOptionLabel("minimax_music3.duration_sec"), "Duration sec");
  assert.equal(audioOptionFloatStep(0, 10), 0.1);
  assert.equal(audioOptionFloatStep(0, 1), 0.01);
  assert.equal(audioOptionFloatStep(0, 300), 5);
  assert.equal(audioOptionFloatStep(1, 1), 0.01);
});

test("values are remembered per model id, whatever its casing", () => {
  saveAudioOptionValues("audio-cpp/Yue2-3B-GGUF", { cot: "full", num_inference_steps: 30 });
  saveAudioOptionValues("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", { speed: 1.2 });
  assert.deepEqual(readAudioOptionValues("AUDIO-CPP/yue2-3b-gguf/"), {
    cot: "full",
    num_inference_steps: 30,
  });
  assert.deepEqual(readAudioOptionValues("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"), {
    speed: 1.2,
  });
  saveAudioOptionValues("audio-cpp/Yue2-3B-GGUF", {});
  assert.deepEqual(readAudioOptionValues("audio-cpp/Yue2-3B-GGUF"), {});
  assert.deepEqual(readAudioOptionValues(null), {});
  globalThis.localStorage.setItem(AUDIO_OPTIONS_STORAGE_KEY, "{not json");
  assert.deepEqual(readAudioOptionValues("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"), {});
});

test("built-in voices read as names, and Kokoro ids also name their language", () => {
  assert.equal(audioVoiceLabel("af_heart", "kokoro_tts"), "Heart (American English, female)");
  assert.equal(audioVoiceLabel("bm_george", "kokoro_tts"), "George (British English, male)");
  assert.equal(audioVoiceLabel("zf_xiaoxiao", "kokoro_tts"), "Xiaoxiao (Mandarin Chinese, female)");
  assert.equal(audioVoiceLabel("uncle_fu", "qwen3_tts"), "Uncle Fu");
  assert.equal(audioVoiceLabel("af_heart", "kitten_tts"), "Af Heart");
  assert.equal(audioVoiceLabel("Custom Voice.wav", null), "Custom Voice.wav");
});
