// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 0 tracing: each spoken turn carries one id from speech to first audio, through the
// turn.* marks, so a latency baseline can be read off the trace.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

const src = (path: string) => readFileSync(new URL(`../src/features/chat/${path}`, import.meta.url), "utf8");
const adapter = src("adapters/studio-voice-loop-adapter.ts");
const engine = src("voice/voice-engine.tsx");
const player = src("hooks/use-tts-player.ts");

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

test("a continuation keeps the id of the fragment it joins", () => {
  assert.match(adapter, /turnId \?\?= heldTurn\?\.turnId \?\? newTurnId\(\);/);
  // Held, the fragment remembers it; flushed alone, it is sent under it.
  assert.match(adapter, /heldTurn = \{\s*text: transcript,\s*at: heldTurn\?\.at \?\? performance\.now\(\),\s*turnId,\s*\};/);
  assert.match(adapter, /const heldId = heldTurn\.turnId \?\? undefined;[\s\S]*?submitLater\(text, \{ turnId: heldId \}\)/);
  // Both ways speech starts open the turn: plain speech and a committed barge-in.
  assert.equal(adapter.match(/heardSpeech = true;\s*startTurn\(\);/g)?.length, 2);
});

test("the adapter marks each stage of a spoken turn", () => {
  for (const mark of ["turn.speechEnd", "turn.vadDone", "turn.transcript", "turn.judgeDone", "turn.confirmed"]) {
    assert.ok(adapter.includes(`voiceTrace("${mark}", { turnId`), mark);
  }
  assert.match(adapter, /voiceTrace\("turn\.confirmed", \{ turnId, early: soundedFinished \}\);\s*submitLater\(transcript, \{ turnId: turnId \?\? undefined, confirmedEarly: soundedFinished \}\);/);
});

test("the bridge's submit reaches the engine as a spoken send, never a typed one", () => {
  assert.match(engine, /registerVoiceSubmit\(\(text, meta\) => \{\s*submitTranscript\(text, false, meta\);/);
  assert.match(engine, /if \(typed\) endTurns\("typed"\);\s*else if \(text\) noteVoiceTurn\(meta\?\.turnId\);/);
});

test("the chat request is marked where a run starts, not where a message is only appended", () => {
  assert.match(engine, /markChatSent\(\);\s*composer\.send\(\);/);
  assert.match(engine, /markChatSent\(\);\s*t\.startRun\(\{ parentId: last\.id \}\);/);
  assert.doesNotMatch(engine, /markChatSent\(\);\s*composer\.send\(\{ startRun: false \}\)/);
  assert.match(engine, /beginStreamRef\.current\(\{ turnId: liveVoiceTurn\(\) \?\? undefined \}\)/);
});

test("every way a turn ends without audio is recorded with its reason", () => {
  const reasons = new Set([
    ...[...engine.matchAll(/endTurns\("(\w+)"/g)].map((m) => m[1]),
    ...[...(engine + player).matchAll(/"turn\.end", \{ turnId: [^,]+, reason: "(\w+)" \}/g)].map((m) => m[1]),
  ]);
  assert.deepEqual([...reasons].sort(), ["barged", "cancelled", "noTts", "superseded", "typed"]);
  // A run that never starts lets its utterance's id go, so the next utterance is traced as its own turn.
  assert.equal(engine.match(/dropUnsentTurn\(\);/g)?.length, 3);
  // A cut reply ends only its own turn: the utterance that cut it is the next one.
  assert.match(engine, /endTurns\("barged", true\)/);
  for (const stop of ["const turnOff = useCallback", "const threadReset = useCallback", "engine.stopSpeaking"]) {
    const at = engine.indexOf(stop);
    assert.ok(at >= 0, stop);
    assert.ok(engine.slice(at, at + 600).includes('endTurns("cancelled")'), stop);
  }
});

test("the first token is marked once per turn, with the poll's resolution", () => {
  // Sampled by the 150 ms stream poll: it can be up to that late, which the event says.
  assert.match(engine, /if \(live && text && _firstTokenTurn !== live\) \{\s*_firstTokenTurn = live;\s*voiceTrace\("turn\.firstToken", \{ turnId: live, approxMs: 150 \}\);/);
});

test("the player marks the first chunk, each request and response, and the first audible sample", () => {
  for (const mark of ["turn.firstChunk", "turn.ttsSent", "turn.ttsReceived", "turn.firstAudible"]) {
    assert.ok(player.includes(`"${mark}"`), mark);
  }
  // Reply clips and the chat model's own clips end the turn; asides do not.
  assert.equal(player.match(/markAudible\(st, schedule\(/g)?.length, 1);
  assert.match(player, /const start = schedule\(buffer, text, entries\);[\s\S]{0,200}markAudible\(st, start\);/);
  assert.doesNotMatch(player, /tts\.aside", \{ text: said, cached \}\);\s*markAudible/);
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
      "../voice/tts-clip-ledger",
      "../voice/speech-normalize",
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
