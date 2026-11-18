// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 0 tracing: each spoken turn carries one id from speech to first audio, through the
// turn.* marks, so a latency baseline can be read off the trace.

import assert from "node:assert/strict";
import test from "node:test";

import {
  dropVoiceTurn,
  endVoiceTurns,
  liveVoiceTurn,
  noteVoiceTurn,
  sendVoiceTurn,
  settleVoiceTurn,
} from "../src/features/chat/voice/voice-loop-bridge.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

test.beforeEach(() => {
  endVoiceTurns();
});

test("utterances coalesced into one run keep the first one's id", () => {
  noteVoiceTurn("first");
  noteVoiceTurn("second");
  assert.equal(sendVoiceTurn(), "first");
  assert.equal(liveVoiceTurn(), "first");
  // Nothing waits now: a typed send that starts the next run carries no id.
  assert.equal(sendVoiceTurn(), null);
  assert.equal(liveVoiceTurn(), "first");
});

test("a turn queued behind a live reply keeps its id until its own run", () => {
  noteVoiceTurn("answered");
  sendVoiceTurn();
  noteVoiceTurn("queued");
  noteVoiceTurn("joined the queue");
  settleVoiceTurn("answered");
  assert.equal(liveVoiceTurn(), null);
  assert.equal(sendVoiceTurn(), "queued");
});

test("a barge-in ends only the live turn; a stop ends the waiting one too", () => {
  noteVoiceTurn("cut");
  sendVoiceTurn();
  noteVoiceTurn("cutting in");
  assert.deepEqual(endVoiceTurns(true), ["cut"]);
  assert.equal(sendVoiceTurn(), "cutting in");
  noteVoiceTurn("waiting");
  assert.deepEqual(endVoiceTurns(), ["cutting in", "waiting"]);
  assert.deepEqual(endVoiceTurns(), []);
});

test("an ignored utterance drops its id only when it is the waiting one", () => {
  noteVoiceTurn("queued first");
  assert.equal(dropVoiceTurn("ignored"), false);
  assert.equal(sendVoiceTurn(), "queued first");
  noteVoiceTurn("ignored");
  assert.equal(dropVoiceTurn("ignored"), true);
  assert.equal(sendVoiceTurn(), null);
});

test("settling another turn's id leaves the live one", () => {
  noteVoiceTurn("live");
  sendVoiceTurn();
  settleVoiceTurn("stale");
  settleVoiceTurn(null);
  assert.equal(liveVoiceTurn(), "live");
});

type Player = {
  audibleAt: (
    start: number,
    ctx: {
      currentTime: number;
      outputLatency?: number;
      baseLatency?: number;
      getOutputTimestamp?: () => { contextTime?: number; performanceTime?: number };
    },
    now: number,
  ) => number;
};

const { audibleAt } = loadWithStubs<Player>(
  new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url),
  Object.fromEntries(
    [
      "@/features/chat/adapters/studio-speech-synthesis-adapter",
      "@/features/chat/stores/chat-runtime-store",
      "../../settings/stores/voice-settings-store",
      "@/i18n",
      "@/lib/toast",
      "react",
      "../voice/speech-text",
      "../voice/tts-clip-clean",
      "../voice/tts-chunk-plan",
      "../voice/voice-loop-bridge",
      "../voice/voice-spectrum",
      "../voice/voice-trace",
      "../voice/voice-models",
      "../voice/voice-pin",
      "../voice/tts-clip-ledger",
      "../voice/speech-normalize",
      "../voice/voice-tts-options",
    ].map((name) => [name, {}]),
  ),
);

test("the first audible sample is placed on the performance clock by the output timestamp", () => {
  const ctx = {
    currentTime: 1.4,
    outputLatency: 0.2,
    getOutputTimestamp: () => ({ contextTime: 1.5, performanceTime: 1000 }),
  };
  // The speaker plays context time 1.5 at 1000 ms, so 2.0 sounds half a second later.
  assert.equal(audibleAt(2, ctx, 900), 1500);
});

test("without an output timestamp the output latency is added to now", () => {
  const ctx = { currentTime: 1, outputLatency: 0.05, baseLatency: 0.01 };
  assert.equal(Math.round(audibleAt(1.2, ctx, 500)), 760);
  // A suspended context reports zeros, which would place the sample at the epoch.
  const suspended = { ...ctx, getOutputTimestamp: () => ({ contextTime: 0, performanceTime: 0 }) };
  assert.equal(Math.round(audibleAt(1.2, suspended, 500)), 760);
});
