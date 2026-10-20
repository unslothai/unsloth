// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  EARLY_RESTART_MAX_AGE_MS,
  createEarlyInterject,
  type EarlyAnswer,
} from "../src/features/chat/voice/early-interject.ts";

// The adapter's hold flushes a fragment as the whole turn after this long with no
// more speech; the early cut has to land well before it.
const JUDGE_FLUSH_MS = 4000;

function rig(answer: EarlyAnswer | null) {
  let t = 0;
  const log: string[] = [];
  const cancels: { gapMs: number; kind: string | null }[] = [];
  const asked: string[] = [];
  const submitted: { text: string; replace: boolean }[] = [];
  let release: () => void = () => {};
  const early = createEarlyInterject({
    ask: (fragment) => {
      asked.push(fragment);
      return new Promise((resolve) => {
        release = () => resolve(answer);
      });
    },
    cancel: (info) => {
      cancels.push(info);
      log.push(`cancel@${t}`);
    },
    now: () => t,
    trace: (event, data) => log.push(`${event}:${JSON.stringify(data)}`),
  });
  // The engine's submit: an early restart sends the whole utterance as that restart;
  // otherwise it asks interject about the whole utterance, as before.
  const submit = (text: string) => {
    const done = early.complete();
    if (done.restarted) submitted.push({ text, replace: done.kind === "replace" });
    else log.push(`reask:${text}`);
  };
  return {
    early,
    log,
    cancels,
    asked,
    submitted,
    submit,
    advance: (ms: number) => {
      t += ms;
    },
    answer: async () => {
      release();
      await new Promise((r) => setImmediate(r));
    },
  };
}

test("a held fragment over a thinking run cuts it before the judge's flush, then sends the whole turn once", async () => {
  const r = rig({ verdict: "restart", kind: "amend" });
  r.early.fragment("Okay, now tell me about like the mountain");
  r.advance(400);
  await r.answer();
  assert.deepEqual(r.asked, ["Okay, now tell me about like the mountain"]);
  assert.equal(r.cancels.length, 1);
  assert.equal(r.cancels[0].gapMs, 400);
  assert.ok(r.cancels[0].gapMs < JUDGE_FLUSH_MS);
  assert.ok(r.log.includes('interject.early:{"verdict":"restart","words":8}'));

  // A continuation is still unfinished: no second ask, no second cut.
  r.early.fragment("Okay, now tell me about like the mountain that");
  await r.answer();
  assert.equal(r.asked.length, 1);
  assert.equal(r.cancels.length, 1);

  r.advance(3000);
  r.submit("Okay, now tell me about like the mountain that we climbed");
  assert.deepEqual(r.submitted, [{ text: "Okay, now tell me about like the mountain that we climbed", replace: false }]);
  assert.ok(!r.log.some((l) => l.startsWith("reask:")));

  // The next utterance is a fresh one.
  r.submit("thanks");
  assert.equal(r.submitted.length, 1);
  assert.ok(r.log.includes("reask:thanks"));
});

test("a replace sends only the new words as the restart", async () => {
  const r = rig({ verdict: "restart", kind: "replace" });
  r.early.fragment("actually forget that, what is");
  await r.answer();
  r.submit("actually forget that, what is the capital of Peru");
  assert.deepEqual(r.submitted, [{ text: "actually forget that, what is the capital of Peru", replace: true }]);
});

for (const verdict of ["queue", "ignore", null] as const) {
  test(`a ${verdict ?? "missing"} verdict on a fragment does nothing until the whole utterance is asked about`, async () => {
    const r = rig(verdict ? { verdict } : null);
    r.early.fragment("and also what about");
    await r.answer();
    assert.equal(r.cancels.length, 0);
    r.submit("and also what about the weather");
    assert.deepEqual(r.submitted, []);
    assert.ok(r.log.includes("reask:and also what about the weather"));
  });
}

test("an utterance that finishes before the early answer leaves the decision to the normal path", async () => {
  const r = rig({ verdict: "restart", kind: "amend" });
  r.early.fragment("wait, can you");
  r.submit("wait, can you also add tests");
  await r.answer();
  assert.equal(r.cancels.length, 0);
  assert.deepEqual(r.submitted, []);
});

test("a reset (the user took the turn, a run started) drops a pending early cut", async () => {
  const r = rig({ verdict: "restart", kind: "amend" });
  r.early.fragment("so the thing is");
  await r.answer();
  r.early.reset();
  r.submit("hello again");
  assert.deepEqual(r.submitted, []);
});

test("a stale early cut is not carried into a much later utterance", async () => {
  const r = rig({ verdict: "restart", kind: "amend" });
  r.early.fragment("so the thing is");
  await r.answer();
  r.advance(EARLY_RESTART_MAX_AGE_MS);
  r.submit("hello again");
  assert.deepEqual(r.submitted, []);
});
