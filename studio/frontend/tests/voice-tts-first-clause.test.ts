// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 1: the player speaks the first clause before the first sentence settles, then only the
// rest of that sentence, and never loses or repeats a splitter sentence while doing it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as speechText from "../src/features/chat/voice/speech-text.ts";
import * as chunkPlan from "../src/features/chat/voice/tts-chunk-plan.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Entry = { text: string; source: number; from: number; to: number };
type Queue = { sentences: Entry[]; sources: number; head: string | null };
type Player = {
  queueSentences: (
    q: Queue,
    settled: readonly string[],
    partial: string,
    clause: boolean,
  ) => { mismatch: boolean; head: string | null };
};

const { queueSentences } = loadWithStubs<Player>(
  new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url),
  {
    ...Object.fromEntries(
      [
        "@/features/chat/adapters/studio-speech-synthesis-adapter",
        "@/features/chat/stores/chat-runtime-store",
        "../../settings/stores/voice-settings-store",
        "@/i18n",
        "@/lib/toast",
        "react",
        "../voice/tts-clip-clean",
        "../voice/voice-loop-bridge",
        "../voice/voice-spectrum",
        "../voice/voice-trace",
        "../voice/voice-models",
        "../voice/voice-pin",
        "../voice/tts-clip-ledger",
        "../voice/speech-normalize",
      ].map((name) => [name, {}]),
    ),
    "../voice/speech-text": speechText,
    "../voice/tts-chunk-plan": chunkPlan,
  },
);

const newQueue = (): Queue => ({ sentences: [], sources: 0, head: null });

/** Streams `reply` a few characters at a time as the engine's poll does, then ends it. */
function stream(reply: string, clause: boolean, step = 3) {
  const q = newQueue();
  let headAt: number | null = null;
  let firstSettledAt: number | null = null;
  let mismatches = 0;
  for (let n = step; n < reply.length + step; n += step) {
    const { complete, partial } = speechText.splitStreaming(reply.slice(0, n));
    const r = queueSentences(q, complete, partial, clause);
    if (r.head !== null && headAt === null) headAt = n;
    if (complete.length > 0 && firstSettledAt === null) firstSettledAt = n;
    if (r.mismatch) mismatches++;
  }
  const all = speechText.splitIntoSentences(speechText.speakableReply(reply));
  if (queueSentences(q, all, "", false).mismatch) mismatches++;
  return { q, all, headAt, firstSettledAt, mismatches };
}

/** Each splitter sentence, rebuilt from the entries that carry it. */
const bySource = (q: Queue, count: number) =>
  Array.from({ length: count }, (_, s) => {
    const parts = q.sentences.filter((e) => e.source === s);
    return parts.map((e) => e.text).join(" ");
  });

const sentences = (n: number) =>
  Array.from({ length: n }, (_, i) => `Sentence number ${i + 1} talks about the weather, and then it ends.`).join(" ");

test("the head goes out from the partial, before the first sentence settles", () => {
  const { q, headAt, firstSettledAt } = stream(sentences(2), true);
  assert.equal(q.sentences[0]?.text, "Sentence number 1 talks about the weather,");
  assert.ok(headAt !== null && firstSettledAt !== null && headAt < firstSettledAt);
  // Then only the rest of that sentence, with its place in the sentence.
  assert.deepEqual(q.sentences[1], {
    text: "and then it ends.",
    source: 0,
    from: "Sentence number 1 talks about the weather, ".length,
    to: "Sentence number 1 talks about the weather, and then it ends.".length,
  });
});

for (const n of [1, 2, 10]) {
  test(`no sentence is skipped or repeated across ${n} sentence${n === 1 ? "" : "s"}`, () => {
    for (const clause of [true, false]) {
      const { q, all, mismatches } = stream(sentences(n), clause);
      assert.equal(all.length, n);
      assert.equal(q.sources, n);
      assert.equal(mismatches, 0);
      assert.deepEqual(bySource(q, n), all);
      // Entries stay in sentence order.
      const order = q.sentences.map((e) => e.source);
      assert.deepEqual(order, [...order].sort((a, b) => a - b));
    }
  });
}

test("a first sentence that settles differently is sent whole and reported", () => {
  const q = newQueue();
  assert.equal(queueSentences(q, [], "Okay so the plan for today is simple, we start", true).head, "Okay so the plan for today is simple,");
  const r = queueSentences(q, ["Okay, so the plan for today is simple."], "", true);
  assert.equal(r.mismatch, true);
  assert.deepEqual(
    q.sentences.map((e) => e.text),
    ["Okay so the plan for today is simple,", "Okay, so the plan for today is simple."],
  );
  assert.equal(q.head, null);
});

test("a sentence that ends exactly at the head adds nothing after it", () => {
  const q = newQueue();
  queueSentences(q, [], "This first clause is long enough, ", true);
  queueSentences(q, ["This first clause is long enough,"], "", true);
  assert.deepEqual(q.sentences.map((e) => e.text), ["This first clause is long enough,"]);
  assert.equal(q.sources, 1);
});

test("with the setting off, each settled sentence is queued whole, as before", () => {
  const reply = sentences(3);
  const { q, all, headAt } = stream(reply, false);
  assert.equal(headAt, null);
  assert.deepEqual(q.sentences, all.map((text, source) => ({ text, source, from: 0, to: text.length })));
});

test("only the first sentence is ever cut early", () => {
  const q = newQueue();
  queueSentences(q, ["Hello there."], "And now a second sentence that is long, with a comma", true);
  assert.deepEqual(q.sentences.map((e) => e.text), ["Hello there."]);
  assert.equal(q.head, null);
});

test("the setting is read once per stream, and the final read never cuts", () => {
  const player = readFileSync(new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url), "utf8");
  assert.match(player, /clause: useVoiceSettingsStore\.getState\(\)\.clauseFirstChunk,/);
  assert.match(player, /queueSentences\(st, settled, partial, st\.clause && !st\.final\)/);
  assert.doesNotMatch(player, /\bextra\b/);
});
