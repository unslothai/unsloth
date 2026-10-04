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
