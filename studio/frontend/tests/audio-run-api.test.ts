// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const {
  buildAudioRunBody,
  buildVoiceCreateBody,
  sourceFileUrl,
  sourceRefOf,
  transcribeUrl,
} = await import("../src/features/audio/audio-run-request.ts");

const api = readSrc("features/audio/api.ts");

const PATHLIKE = /"(path|voice_ref|file|url|audio_path)"\s*:/;

test("a clone run sends ids only, with the contract's keys", () => {
  const body = buildAudioRunBody({
    workflow: "clone",
    text: "Hello",
    language: "English",
    instructions: "  ",
    inputs: {
      reference: { input_id: "a".repeat(32) },
      reference_text: " Okay, I'm Cemo. ",
      emotion: { clip_id: "c1" },
    },
    options: { x_vector_only_mode: true, bad: Number.NaN },
    speed: 1.3,
    seed: 7,
  });
  assert.deepEqual(body, {
    workflow: "clone",
    text: "Hello",
    language: "English",
    inputs: {
      reference: { input_id: "a".repeat(32) },
      reference_text: "Okay, I'm Cemo.",
      emotion: { clip_id: "c1" },
    },
    options: { x_vector_only_mode: true },
    speed: 1.3,
    seed: 7,
  });
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
});

test("a cleared transcript is sent blank, an omitted one is left out", () => {
  const cleared = buildAudioRunBody({
    workflow: "clone",
    text: "Hi",
    inputs: { reference: { voice_id: "v1" }, reference_text: "  " },
  });
  assert.deepEqual(cleared.inputs, {
    reference: { voice_id: "v1" },
    reference_text: "",
  });
  const omitted = buildAudioRunBody({
    workflow: "clone",
    text: "Hi",
    inputs: { reference: { voice_id: "v1" }, reference_text: null },
  });
  assert.deepEqual(omitted.inputs, { reference: { voice_id: "v1" } });
});

test("whatever a caller spreads in, no path or bytes reach the body", () => {
  const sneaky = {
    workflow: "speak" as const,
    text: "Hi",
    path: "/etc/passwd",
    voice_ref: "/abs/ref.wav",
    inputs: {
      reference: { voice_id: "v1", path: "/x.wav", url: "/y" } as never,
      emotion: { input_id: "i", clip_id: "c" } as never,
    },
  };
  const body = buildAudioRunBody(sneaky);
  assert.deepEqual(body, {
    workflow: "speak",
    text: "Hi",
    inputs: { reference: { voice_id: "v1" } },
  });
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
});

test("a convert run sends the source, the target and the convert settings, and no text", () => {
  const body = buildAudioRunBody({
    workflow: "convert",
    inputs: {
      source: { clip_id: "c1" },
      target: { voice_id: "v1" },
      source_text: "  Hello there.  ",
    },
    convert: { mode: "speech", pitch: 3.4, pitch_auto: false, style: "target" },
    options: { route: "v2_vc", length_adjust: 1.1, bad: Number.NaN },
    seed: 7,
  });
  assert.deepEqual(body, {
    workflow: "convert",
    inputs: {
      source: { clip_id: "c1" },
      target: { voice_id: "v1" },
      source_text: "Hello there.",
    },
    convert: { mode: "speech", pitch: 3, pitch_auto: false, style: "target" },
    options: { route: "v2_vc", length_adjust: 1.1 },
    seed: 7,
  });
  assert.equal("text" in body, false);
});

test("a convert run to a built-in voice has no target, and Auto keeps its extra shift", () => {
  const body = buildAudioRunBody({
    workflow: "convert",
    inputs: { source: { input_id: "i1" }, target: null, source_text: " " },
    convert: {
      mode: "singing",
      pitch: 5,
      pitch_auto: true,
      voice: " manthos ",
    },
  });
  assert.deepEqual(body, {
    workflow: "convert",
    inputs: { source: { input_id: "i1" } },
    convert: {
      mode: "singing",
      pitch: 5,
      pitch_auto: true,
      voice: "manthos",
    },
  });
  const low = buildAudioRunBody({
    workflow: "convert",
    inputs: { source: { input_id: "i1" } },
    convert: { mode: "speech", pitch: -40, pitch_auto: false },
  });
  assert.equal((low.convert as { pitch: number }).pitch, -24);
});

test("whatever a caller spreads into a convert run, no path or unknown key reaches the body", () => {
  const sneaky = {
    workflow: "convert" as const,
    text: "should not be sent",
    voice_ref: "/abs/ref.wav",
    inputs: {
      source: { input_id: "i1", path: "/x.wav" } as never,
      target: { voice_id: "v1", clip_id: "c2" } as never,
      reference: { voice_id: "v9" },
      target_voice: "/abs/target.wav",
    } as never,
    convert: {
      mode: "humming",
      pitch: null,
      pitch_auto: false,
      route: "v1_svc",
      source_audio: "/abs/src.wav",
    } as never,
    options: { retrieval_blend: 0.5, nested: { path: "/x" } } as never,
  };
  const body = buildAudioRunBody(sneaky);
  assert.deepEqual(body, {
    workflow: "convert",
    inputs: { source: { input_id: "i1" } },
    convert: { mode: "speech", pitch: null, pitch_auto: false },
    options: { retrieval_blend: 0.5 },
  });
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
  assert.doesNotMatch(JSON.stringify(body), /target_voice|source_audio|"text"/);
});

test("selections become one-id refs and server URLs", () => {
  assert.deepEqual(
    sourceRefOf({ kind: "input", id: "i1", name: "a", durationS: 1 }),
    { input_id: "i1" },
  );
  assert.deepEqual(
    sourceRefOf({ kind: "clip", id: "c1", name: "a", durationS: 1 }),
    { clip_id: "c1" },
  );
  assert.deepEqual(
    sourceRefOf({ kind: "voice", id: "v1", name: "a", durationS: 1 }),
    { voice_id: "v1" },
  );
  assert.equal(
    sourceFileUrl({ kind: "input", id: "i1", name: "", durationS: null }),
    "/api/inference/audio/inputs/i1/file",
  );
  assert.equal(
    sourceFileUrl({ kind: "voice", id: "v1", name: "", durationS: null }),
    "/api/inference/audio/voices/v1/file",
  );
  assert.equal(
    transcribeUrl({ input_id: "i1" }),
    "/api/inference/audio/inputs/i1/transcribe",
  );
  assert.match(transcribeUrl({ clip_id: "c 1" }), /\?clip_id=c%201$/);
});

test("a saved voice is made from an upload or a clip, within the limits", () => {
  const body = buildVoiceCreateBody({
    source: { clip_id: "c1" },
    name: ` ${"n".repeat(100)} `,
    transcript: "",
    language: "English",
  });
  assert.deepEqual(body, {
    source: { clip_id: "c1" },
    name: "n".repeat(80),
    language: "English",
  });
  assert.throws(() =>
    buildVoiceCreateBody({ source: { voice_id: "v" }, name: "x" }),
  );
});

test("api.ts sends run and voice bodies only through the builders", () => {
  assert.match(api, /body: JSON\.stringify\(buildAudioRunBody\(request\)\)/);
  assert.match(api, /body: JSON\.stringify\(buildVoiceCreateBody\(request\)\)/);
  assert.match(api, /"\/api\/inference\/audio\/run"/);
  assert.match(api, /xhr\.send\(blob\)/);
  assert.match(api, /body: blob,/);
  assert.doesNotMatch(api, /voice_ref/);
});

test("an edit run sends the source by id and only the edit part's allowed keys", () => {
  const markup = 'A <sub targ="robot">human</sub> voice.';
  const body = buildAudioRunBody({
    workflow: "edit",
    text: "A robot voice.",
    inputs: {
      source: {
        input_id: "in_1",
        trim: { start_s: 0, end_s: 4 },
        path: "/tmp/x.wav",
      } as never,
      reference_text: " A human voice. ",
    },
    edit: {
      mode: "words",
      markup,
      instructions: ["", 3 as never],
      speed: Number.NaN,
      pitch_steps: 1.5,
      path: "/tmp/y.wav",
    } as never,
  });
  assert.deepEqual(body, {
    workflow: "edit",
    text: "A robot voice.",
    inputs: {
      source: { input_id: "in_1" },
      reference_text: "A human voice.",
    },
    edit: { mode: "words", markup },
  });
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
  const editOf = (edit: object) =>
    buildAudioRunBody({ workflow: "edit", text: "x", edit: edit as never })
      .edit;
  assert.equal(editOf({ mode: "bogus" }), undefined);
  const delivery = { mode: "delivery", speed: 1.5, pitch_steps: 3 };
  assert.deepEqual(editOf(delivery), delivery);
});
