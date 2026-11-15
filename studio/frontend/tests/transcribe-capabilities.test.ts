// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SttCapabilities,
  type TranscribeSwitch,
  connectionTranscribeModels,
  connectionTranscribeSwitches,
  transcribeSourceHint,
  transcribeSwitches,
} from "../src/features/audio/transcribe-capabilities.ts";
import {
  TRANSCRIBE_LANGUAGES,
  transcribeLanguageFor,
  transcribeLanguagesFor,
} from "../src/features/audio/transcribe-languages.ts";

const UNSUPPORTED: SttCapabilities = {
  engine: "audiocpp",
  family: "x",
  timestamps: "unsupported",
  speakers: false,
  aligner: null,
  cpu_only: false,
};
const off = { timestamps: false, speakers: false, speakerAccent: "" };
const on = { timestamps: true, speakers: true, speakerAccent: "" };
const switches = (
  over: Partial<SttCapabilities> | null,
  prefs: Parameters<typeof transcribeSwitches>[1] = on,
) =>
  transcribeSwitches(over && { ...UNSUPPORTED, ...over }, prefs, {
    loading: false,
    hasModel: true,
  });
const flags = (value: TranscribeSwitch) => [value.checked, value.disabled];

test("unsupported: both switches off, each saying why and offering the speakers model", () => {
  const { timestamps, speakers, request, notice } = switches({});
  assert.deepEqual(
    [...flags(timestamps), ...flags(speakers)],
    [false, true, false, true],
  );
  assert.match(timestamps.hint, /plain text/);
  assert.match(speakers.hint, /MOSS-Transcribe-Diarize and VibeVoice-ASR/);
  assert.equal(speakers.suggestSpeakersModel, true);
  assert.deepEqual([request, notice], [off, null]);
});

test("on_request: opt-in timestamps that say they download the aligner and reload", () => {
  const aligner = { downloaded: false, size_bytes: 1_100_000_000 };
  const qwen = { timestamps: "on_request", aligner } as const;
  const optedOut = switches(qwen, off);
  assert.deepEqual(flags(optedOut.timestamps), [false, false]);
  assert.match(optedOut.timestamps.hint, /downloads it \(1\.1 GB\)/);
  assert.deepEqual([optedOut.request, optedOut.notice], [off, null]);
  const optedIn = switches(qwen);
  assert.deepEqual(optedIn.request, {
    timestamps: true,
    speakers: false,
    speakerAccent: "",
  });
  assert.match(
    optedIn.notice ?? "",
    /Downloads the timing aligner \(1\.1 GB\) and reloads/,
  );
  const loaded = switches({
    ...qwen,
    aligner: { ...aligner, downloaded: true },
  });
  assert.doesNotMatch(loaded.timestamps.hint, /downloads/);
  assert.match(loaded.notice ?? "", /Reloads the model/);
});

test("always: timestamps read as always on, and the request only asks for speakers", () => {
  const moss = { timestamps: "always", speakers: true } as const;
  const result = switches(moss, { timestamps: false, speakers: true });
  assert.deepEqual(
    [flags(result.timestamps), result.timestamps.always],
    [[true, true], true],
  );
  assert.deepEqual(result.request, {
    timestamps: false,
    speakers: true,
    speakerAccent: "",
  });
  assert.deepEqual(switches(moss, off).request, off);
});

test("cpu_only says so before the run, after any aligner notice", () => {
  assert.equal(
    switches({ cpu_only: true }).notice,
    "This model runs on the CPU.",
  );
  const qwen = { timestamps: "on_request", cpu_only: true } as const;
  assert.match(
    switches(qwen).notice ?? "",
    /not loaded yet\. Runs on the CPU\.$/,
  );
});

test("while unknown the switches wait, and without a model they say what to do", () => {
  for (const [loading, hasModel, hint] of [
    [true, true, /Checking/],
    [false, false, /Pick a speech-to-text model/],
    [false, true, /Transcribing still works/],
  ] as const) {
    const result = transcribeSwitches(null, on, { loading, hasModel });
    assert.match(result.speakers.hint, hint);
    assert.deepEqual(
      [flags(result.timestamps), result.request],
      [[false, true], off],
    );
  }
});

test("languages are ISO codes with Auto first; a saved one the model lacks is sent as detect", () => {
  const names = TRANSCRIBE_LANGUAGES.map((entry) => entry.name);
  assert.match(
    names.join(),
    /^Detect automatically,English,Chinese,.*,Cantonese$/,
  );
  assert.equal(transcribeLanguagesFor(undefined), TRANSCRIBE_LANGUAGES);
  const canary = transcribeLanguagesFor(["en", "de", "es", "fr"]);
  assert.deepEqual(canary.map((entry) => entry.code).join(), ",en,es,fr,de");
  assert.equal(transcribeLanguageFor("ja", canary), "");
  assert.equal(transcribeLanguageFor("de", canary), "de");
  assert.equal(transcribeLanguageFor("de", transcribeLanguagesFor(["en"])), "");
});

test("a connection lists its speech-to-text models and the one dictation uses", () => {
  const custom = {
    providerType: "custom",
    models: ["llama-3.1-8b", "whisper-1"],
    availableModels: ["Systran/faster-whisper-small", "unsloth/Qwen3-ASR-0.6B-GGUF", "whisper-1"],
  };
  assert.deepEqual(connectionTranscribeModels(custom), [
    "whisper-1",
    "Systran/faster-whisper-small",
    "unsloth/Qwen3-ASR-0.6B-GGUF",
  ]);
  assert.deepEqual(connectionTranscribeModels({ ...custom, models: [], availableModels: [] }, "large-v3"), [
    "large-v3",
  ]);
  assert.deepEqual(connectionTranscribeModels({ providerType: "openai", models: ["gpt-5.5"] }), [
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
  ]);
});

test("on a connection, timestamps follow the switch and speakers stay off", () => {
  const result = connectionTranscribeSwitches({ timestamps: true }, "whisper-1");
  assert.deepEqual(result.request, {
    timestamps: true,
    speakers: false,
    speakerAccent: "",
  });
  assert.equal(result.timestamps.disabled, false);
  assert.equal(result.speakers.disabled, true);
  assert.equal(result.speakers.suggestSpeakersModel, undefined);
  assert.equal(
    connectionTranscribeSwitches({ timestamps: false }, "whisper-1").request.timestamps,
    false,
  );
});

test("OpenAI's gpt-4o transcribe models never ask for timestamps", () => {
  for (const model of ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]) {
    const result = connectionTranscribeSwitches({ timestamps: true }, model);
    assert.equal(result.timestamps.disabled, true);
    assert.equal(result.timestamps.checked, false);
    assert.equal(result.request.timestamps, false);
  }
});

test("dated and newer GPT transcription models never ask for timestamps either", () => {
  for (const model of [
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-mini-transcribe-2025-03-20",
    "gpt-4o-transcribe-2025-03-20",
    "gpt-transcribe",
    "gpt-4o-transcribe-diarize",
  ]) {
    const result = connectionTranscribeSwitches({ timestamps: true }, model);
    assert.equal(result.timestamps.disabled, true, model);
    assert.equal(result.request.timestamps, false, model);
  }
  for (const model of ["whisper-1", "whisper-large-v3", "parakeet-tdt", "my-gpt-transcriber"]) {
    assert.equal(connectionTranscribeSwitches({ timestamps: true }, model).request.timestamps, true, model);
  }
});

test("an owner-shared connection lists only the models the owner allowed", () => {
  const shared = { providerType: "openai", models: ["gpt-5.5", "whisper-1"], sharedByOwner: true };
  assert.deepEqual(connectionTranscribeModels(shared, "gpt-4o-transcribe"), ["whisper-1"]);
  assert.deepEqual(connectionTranscribeModels({ ...shared, models: ["gpt-5.5"] }), []);
});

test("an OpenAI-typed Azure connection offers its deployments, not OpenAI's model IDs", () => {
  const azure = {
    providerType: "openai",
    baseUrl: "https://contoso.openai.azure.com/openai/v1",
    models: ["gpt-5.5", "whisper-deploy"],
  };
  assert.deepEqual(connectionTranscribeModels(azure), ["whisper-deploy"]);
  assert.deepEqual(
    connectionTranscribeModels({ ...azure, baseUrl: "https://api.openai.com/v1", models: [] }),
    ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
  );
});

test("a connection's Voxtral TTS model stays out of Transcribe; its realtime ASR does not", () => {
  const custom = {
    providerType: "custom",
    models: [
      "voxtral-mini-tts-latest",
      "voxtral-small-latest",
      "voxtral-mini-latest",
      "voxtral-mini-transcribe-realtime",
    ],
  };
  assert.deepEqual(connectionTranscribeModels(custom), [
    "voxtral-mini-latest",
    "voxtral-mini-transcribe-realtime",
  ]);
});

test("the source hint names how long this model's audio can run", () => {
  assert.equal(
    transcribeSourceHint(4 * 3600),
    "Up to 4 hours with this model. Drop a file, record, or reuse a clip.",
  );
  assert.equal(
    transcribeSourceHint(30 * 60),
    "Up to 30 minutes with this model. Drop a file, record, or reuse a clip.",
  );
  assert.equal(
    transcribeSourceHint(90 * 60),
    "Up to 90 minutes with this model. Drop a file, record, or reuse a clip.",
  );
  const today = "Up to 30 minutes. Drop a file, record, or reuse a clip.";
  assert.equal(transcribeSourceHint(null), today);
  assert.equal(transcribeSourceHint(), today);
  assert.equal(
    switches({ max_seconds: 14400 }).sourceHint,
    transcribeSourceHint(14400),
  );
  assert.equal(switches({}).sourceHint, today);
  assert.equal(switches(null).sourceHint, today);
  assert.equal(
    connectionTranscribeSwitches({ timestamps: true }, "whisper-1").sourceHint,
    today,
  );
});

test("speaker accent: sent only to a model that takes the hint, tidied", () => {
  const prefs = { ...off, speakerAccent: "  Chinese-accented\n  English " };
  const qwen = switches({ engine: "mtmd", accent_hint: true }, prefs);
  assert.equal(qwen.accent.disabled, false);
  assert.match(qwen.accent.hint, /stays in the spoken language/);
  assert.equal(qwen.request.speakerAccent, "Chinese-accented English");
  assert.equal(
    switches({ accent_hint: true }, { ...off, speakerAccent: "x".repeat(300) })
      .request.speakerAccent.length,
    200,
  );
});

test("speaker accent: a model without the hint says so and sends nothing", () => {
  const prefs = { ...off, speakerAccent: "Japanese-accented English" };
  // An older backend omits accent_hint: the hint is never sent on a guess.
  for (const over of [{}, { accent_hint: false }]) {
    const { accent, request } = switches(over, prefs);
    assert.equal(accent.disabled, true);
    assert.match(accent.hint, /cannot use an accent hint, so none is sent/);
    assert.equal(request.speakerAccent, "");
  }
  for (const loading of [true, false]) {
    const result = transcribeSwitches(null, prefs, { loading, hasModel: true });
    assert.deepEqual([result.accent.disabled, result.request.speakerAccent], [true, ""]);
  }
});

test("speaker accent: a connection passes it on as the transcription prompt", () => {
  for (const model of ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]) {
    const result = connectionTranscribeSwitches(
      { timestamps: false, speakerAccent: "Indian English" },
      model,
    );
    assert.equal(result.accent.disabled, false);
    assert.equal(result.request.speakerAccent, "Indian English");
  }
});

test("speaker accent: Whisper and diarizing connection models take no prompt", () => {
  for (const model of ["gpt-4o-transcribe-diarize", "whisper-1", "Systran/faster-whisper-small"]) {
    const result = connectionTranscribeSwitches(
      { timestamps: false, speakerAccent: "Indian English" },
      model,
    );
    assert.equal(result.accent.disabled, true);
    assert.equal(result.request.speakerAccent, "");
  }
});
