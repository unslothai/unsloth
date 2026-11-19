// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type TurnEnd = typeof import("../src/features/chat/voice/turn-end.ts");
const load = (authFetch: (url: string, init?: RequestInit) => Promise<Response>) =>
  loadWithStubs<TurnEnd>(new URL("../src/features/chat/voice/turn-end.ts", import.meta.url), {
    "@/features/auth": { authFetch },
  });
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const wav = () => new Blob(["x"]);
const settle = () => new Promise((r) => setTimeout(r, 0));

test("an answer counts only for the speech it was asked about", async () => {
  const { createTurnEndCheck } = await load(() => Promise.resolve(json({})));
  const asked: AbortSignal[] = [];
  const answers: unknown[] = [];
  let reply: (finished: boolean) => void = () => {};
  const check = createTurnEndCheck(
    (_wav, signal) => {
      asked.push(signal);
      return new Promise((resolve) => {
        reply = (finished) => resolve({ finished, p: finished ? 0.98 : 0.01 });
      });
    },
    (answer) => answers.push(answer),
  );
  check.ask("0:10", wav);
  check.ask("0:10", wav);
  assert.equal(asked.length, 1, "one question per pause");
  assert.equal(check.finished("0:10"), false, "no answer yet is not a yes");
  const first = reply;
  check.ask("0:25", wav);
  assert.equal(asked[0].aborted, true);
  first(true);
  await settle();
  assert.equal(check.finished("0:10"), false);
  assert.equal(check.finished("0:25"), false);
  assert.deepEqual(answers, []);
  reply(true);
  await settle();
  assert.equal(check.finished("0:25"), true);
  assert.equal(check.finished("0:40"), false);
  assert.deepEqual(answers, [{ finished: true, p: 0.98 }]);
  check.clear();
  assert.equal(check.finished("0:25"), false);
  assert.equal(asked[1].aborted, true);
});

test("a failed or unfinished answer leaves the silence wait in charge", async () => {
  const { createTurnEndCheck } = await load(() => Promise.resolve(json({})));
  const failing = createTurnEndCheck(() => Promise.reject(new Error("offline")));
  failing.ask("0:10", wav);
  await settle();
  assert.equal(failing.finished("0:10"), false);
  const unfinished = createTurnEndCheck(() => Promise.resolve({ finished: false, p: 0.02 }));
  unfinished.ask("0:10", wav);
  await settle();
  assert.equal(unfinished.finished("0:10"), false);
});

test("the clip keeps the quiet end of the last word and none of the silence after it", async () => {
  const { turnClipSpan } = await load(() => Promise.resolve(json({})));
  const voiced = [false, false, true, true, false, false, false, false];
  const notSilent = [false, true, true, true, true, false, false, false];
  assert.deepEqual(turnClipSpan(voiced, notSilent), { start: 1, end: 4 });
  assert.deepEqual(
    turnClipSpan(voiced, [false, true, true, true, true, false, true, false]),
    { start: 1, end: 4 },
  );
  assert.equal(turnClipSpan([false, false], [true, true]), null);
  assert.deepEqual(turnClipSpan([true], [true]), { start: 0, end: 0 });
});

test("a transcript reads cut off on an open last word, or on a missing mark from a transcriber that writes them", async () => {
  const { readsUnfinished, hasClosingMark } = await load(() => Promise.resolve(json({})));
  for (const text of ["I went to the store and", "I went to the store and.", "Because", "so I asked the"]) {
    assert.equal(readsUnfinished(text, false), true, text);
  }
  for (const text of ["How are you?", "Yes, I do.", "I think so.", "What is it about?", "Chips or?", "Come on!"]) {
    assert.equal(readsUnfinished(text, true), false, text);
  }
  // A missing closing mark means nothing unless this transcriber writes them.
  assert.equal(readsUnfinished("I was wondering whether you could", true), true);
  assert.equal(readsUnfinished("I was wondering whether you could", false), false);
  assert.equal(readsUnfinished("i want to go home", false), false);
  assert.equal(readsUnfinished("HELLO WORLD", false), false);
  for (const answer of ["10", "21", "Room 12", "Yes", "Not really"]) {
    assert.equal(readsUnfinished(answer, true), false, answer);
  }
  assert.equal(readsUnfinished("", true), false);
  assert.equal(readsUnfinished("你好吗？", true), false);
  assert.equal(hasClosingMark("Sure. "), true);
  assert.equal(hasClosingMark("Sure"), false);
});

test("a reply waits out the gap after speech heard while it was about to start", async () => {
  const { stillInTurn, TURN_CONTINUE_GRACE_MS } = await load(() => Promise.resolve(json({})));
  assert.equal(stillInTurn(true, 0, 5000), true, "speaking now");
  assert.equal(stillInTurn(false, 0, 5000), false, "nothing was heard: the reply starts");
  assert.equal(stillInTurn(false, 5000, 5000 + TURN_CONTINUE_GRACE_MS - 1), true, "the turn is still being recorded");
  assert.equal(stillInTurn(false, 5000, 5000 + TURN_CONTINUE_GRACE_MS), false);
});
