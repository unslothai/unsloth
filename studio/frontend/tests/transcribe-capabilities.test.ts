// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SttCapabilities,
  transcribeSwitches,
} from "../src/features/audio/transcribe-capabilities.ts";
import {
  TRANSCRIBE_LANGUAGES,
  transcribeLanguageFor,
  transcribeLanguagesFor,
} from "../src/features/audio/transcribe-languages.ts";

const caps = (over: Partial<SttCapabilities>): SttCapabilities => ({
  engine: "audiocpp",
  family: "x",
  timestamps: "unsupported",
  speakers: false,
  aligner: null,
  cpu_only: false,
  ...over,
});
const ready = { loading: false, hasModel: true };
const on = { timestamps: true, speakers: true };
const off = { timestamps: false, speakers: false };

test("a plain-text model disables both switches with the reason and a way out", () => {
  const result = transcribeSwitches(
    caps({ engine: "transformers" }),
    on,
    ready,
  );
  assert.equal(result.timestamps.disabled, true);
  assert.equal(result.timestamps.checked, false);
  assert.match(result.timestamps.hint, /plain text/);
  assert.equal(result.timestamps.suggestSpeakersModel, true);
  assert.equal(result.speakers.disabled, true);
  assert.match(
    result.speakers.hint,
    /MOSS-Transcribe-Diarize and VibeVoice-ASR/,
  );
  assert.deepEqual(result.request, { timestamps: false, speakers: false });
  assert.equal(result.notice, null);
});

test("Qwen3-ASR timestamps are opt-in and say they download and reload", () => {
  const qwen = caps({
    family: "qwen3_asr",
    timestamps: "on_request",
    aligner: { downloaded: false, size_bytes: 1_100_000_000 },
  });
  const optedOut = transcribeSwitches(qwen, off, ready);
  assert.equal(optedOut.timestamps.disabled, false);
  assert.equal(optedOut.timestamps.checked, false);
  assert.match(optedOut.timestamps.hint, /downloads it \(1\.1 GB\)/);
  assert.equal(optedOut.notice, null);
  assert.equal(optedOut.request.timestamps, false);

  const optedIn = transcribeSwitches(qwen, on, ready);
  assert.equal(optedIn.request.timestamps, true);
  assert.match(
    optedIn.notice ?? "",
    /Downloads the timing aligner \(1\.1 GB\) and reloads/,
  );
  // Qwen3 cannot tell speakers apart, so asking for them sends nothing.
  assert.equal(optedIn.request.speakers, false);

  const downloaded = transcribeSwitches(
    caps({ ...qwen, aligner: { downloaded: true, size_bytes: 1 } }),
    on,
    ready,
  );
  assert.doesNotMatch(downloaded.timestamps.hint, /downloads/);
  assert.match(downloaded.notice ?? "", /Reloads the model/);
});

test("MOSS always times its lines and lets speakers be switched off", () => {
  const moss = caps({
    family: "moss_transcribe_diarize",
    timestamps: "always",
    speakers: true,
  });
  const result = transcribeSwitches(
    moss,
    { timestamps: false, speakers: true },
    ready,
  );
  assert.deepEqual(
    [result.timestamps.checked, result.timestamps.disabled],
    [true, true],
  );
  assert.match(result.timestamps.hint, /always adds timestamps/);
  assert.equal(result.timestamps.always, true);
  assert.equal(result.speakers.disabled, false);
  // The server times every MOSS run anyway; the request only asks for what is optional.
  assert.deepEqual(result.request, { timestamps: false, speakers: true });
  const quiet = transcribeSwitches(
    moss,
    { timestamps: false, speakers: false },
    ready,
  );
  assert.equal(quiet.request.speakers, false);
});

test("a CPU-only model says so before the run", () => {
  const result = transcribeSwitches(
    caps({ family: "niagara_asr", cpu_only: true }),
    off,
    ready,
  );
  assert.equal(result.notice, "This model runs on the CPU.");
});

test("while unknown the switches wait, and without a model they say what to do", () => {
  const loading = transcribeSwitches(null, on, {
    loading: true,
    hasModel: true,
  });
  assert.match(loading.timestamps.hint, /Checking/);
  assert.equal(loading.timestamps.disabled, true);
  const none = transcribeSwitches(null, on, {
    loading: false,
    hasModel: false,
  });
  assert.match(none.speakers.hint, /Pick a speech-to-text model/);
  const failed = transcribeSwitches(null, on, {
    loading: false,
    hasModel: true,
  });
  assert.match(failed.speakers.hint, /Transcribing still works/);
  assert.deepEqual(failed.request, { timestamps: false, speakers: false });
});

test("languages are ISO codes with Auto first; English-only models keep only Auto and English", () => {
  assert.deepEqual(TRANSCRIBE_LANGUAGES[0], {
    code: "",
    name: "Detect automatically",
  });
  assert.ok(
    TRANSCRIBE_LANGUAGES.every((entry) => /^[a-z]{0,3}$/.test(entry.code)),
  );
  assert.equal(transcribeLanguagesFor(undefined), TRANSCRIBE_LANGUAGES);
  assert.deepEqual(
    transcribeLanguagesFor(["en"]).map((entry) => entry.code),
    ["", "en"],
  );
  assert.deepEqual(
    transcribeLanguagesFor(["en", "de", "es", "fr"]).map((entry) => entry.code),
    ["", "en", "es", "fr", "de"],
  );
});

test("a saved language the model does not list is sent as detect, as the rail shows it", () => {
  const canary = transcribeLanguagesFor(["en", "de", "es", "fr"]);
  assert.equal(transcribeLanguageFor("ja", canary), "");
  assert.equal(transcribeLanguageFor("de", canary), "de");
  // An English-only model shows no picker, and a saved code is not sent to it.
  assert.equal(transcribeLanguageFor("de", transcribeLanguagesFor(["en"])), "");
  assert.equal(transcribeLanguageFor("", canary), "");
});
