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
  // The speaker went on: the first question is dropped, and its late yes ends nothing.
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

test("the model loads once for the page, and again after an unload", async () => {
  const calls: string[] = [];
  let ok = true;
  const { loadTurnModel, unloadTurnModel, askTurnEnd } = await load((url) => {
    calls.push(url.replace("/api/inference/audio/turn/", ""));
    if (url.endsWith("/detect")) return Promise.resolve(json({ finished: true, p_complete: 0.97 }));
    return Promise.resolve(json({ loaded_model: "m" }, ok ? 200 : 501));
  });
  assert.equal(await loadTurnModel(), true);
  assert.equal(await loadTurnModel(), true);
  assert.deepEqual(calls, ["load"]);
  assert.deepEqual(await askTurnEnd(wav()), { finished: true, p: 0.97 });
  // The language travels only when the user set one.
  await askTurnEnd(wav(), undefined, "auto");
  await askTurnEnd(wav(), undefined, "es-419");
  assert.deepEqual(calls.splice(1), ["detect", "detect", "detect?language=es-419"]);
  await unloadTurnModel();
  ok = false;
  // An install that cannot run it says so once; the loop then never asks.
  assert.equal(await loadTurnModel(), false);
  assert.deepEqual(calls, ["load", "unload", "load"]);
});

test("a refused check is an error, not a no", async () => {
  const { askTurnEnd } = await load(() => Promise.resolve(json({ detail: "busy" }, 503)));
  await assert.rejects(askTurnEnd(wav()));
});

test("a language the model is not trusted with comes back as not finished, with no score", async () => {
  const { askTurnEnd } = await load(() =>
    Promise.resolve(json({ finished: false, p_complete: null, bar: null, model: null })),
  );
  assert.deepEqual(await askTurnEnd(wav(), undefined, "zh-CN"), { finished: false, p: null });
});

test("the clip keeps the quiet end of the last word and none of the silence after it", async () => {
  const { turnClipSpan } = await load(() => Promise.resolve(json({})));
  //               0      1      2     3     4      5      6      7
  const voiced = [false, false, true, true, false, false, false, false];
  const notSilent = [false, true, true, true, true, false, false, false];
  assert.deepEqual(turnClipSpan(voiced, notSilent), { start: 1, end: 4 });
  // A noise after the pause is not part of the utterance.
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
  // Everyday endings are not open ones, and a question mark settles it.
  for (const text of ["How are you?", "Yes, I do.", "I think so.", "What is it about?", "Chips or?", "Come on!"]) {
    assert.equal(readsUnfinished(text, true), false, text);
  }
  // No closing mark means something only once this transcriber has written one.
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

test("only the end of a long turn is uploaded", async () => {
  const { turnClipTail, TURN_CLIP_MAX_S } = await load(() => Promise.resolve(json({})));
  const rate = 1000;
  const long = Float32Array.from({ length: 30 * rate }, (_, i) => i);
  const tail = turnClipTail(long, rate);
  assert.equal(tail.length, TURN_CLIP_MAX_S * rate);
  assert.equal(tail[tail.length - 1], long[long.length - 1]);
  const short = new Float32Array(3 * rate);
  assert.equal(turnClipTail(short, rate), short);
});
