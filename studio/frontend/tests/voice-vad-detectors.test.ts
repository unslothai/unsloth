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

test("a built-in this install cannot run falls back; a Hub pick that fails does not", () => {
  const vad = load();
  assert.equal(vad.vadFallback("builtin/earshot-vad"), PULSEVAD);
  assert.equal(vad.vadFallback("builtin/silero-vad"), "builtin/earshot-vad");
  assert.equal(vad.vadFallback("Builtin/Silero-VAD"), "builtin/earshot-vad");
  assert.equal(vad.vadFallback(PULSEVAD), null);
});

test("a clip is held to its detector's own bar, the shorter one for a barge-in probe", () => {
  const vad = load();
  const earshot = { utterance_ms: 200, barge_ms: 150 };
  assert.equal(vad.minSpeechMs(earshot, false), 200);
  assert.equal(vad.minSpeechMs(earshot, true), 150);
  // The probe's shorter bar must fit its ~250 ms clip.
  assert.equal(vad.minSpeechMs(undefined, false), 300);
  assert.equal(vad.minSpeechMs(undefined, true), 200);
});
