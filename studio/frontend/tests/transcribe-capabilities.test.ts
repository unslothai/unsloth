// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SttCapabilities,
  type TranscribeSwitch,
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
const off = { timestamps: false, speakers: false };
const on = { timestamps: true, speakers: true };
const switches = (over: Partial<SttCapabilities> | null, prefs = on) =>
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
  assert.deepEqual(optedIn.request, { timestamps: true, speakers: false });
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
  assert.deepEqual(result.request, { timestamps: false, speakers: true });
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
