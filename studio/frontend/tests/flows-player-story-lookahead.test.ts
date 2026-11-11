// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { NextSceneRequest as Req } from "../src/features/flows/player/story/lookahead.ts";
import { flushMicrotasks as tick, storyManifest } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/story/reducer.ts");
const L = await import("../src/features/flows/player/story/lookahead.ts");
type Scene = import("../src/features/flows/types.ts").StoryScene;
type State = import("../src/features/flows/player/story/reducer.ts").StoryState;

const m = storyManifest(
  [{
    id: "door",
    lines: [{ text: "Dark." }, { speaker: "g", text: "Go on." }],
    choices: [
      { id: "c1", text: "In", goto: "gen_1", set: { hp: 9 } },
      { id: "c2", text: "Back", goto: "gen_2" },
      { id: "c3", text: "Wait", goto: "gen_3" },
    ],
  }],
  {
    title: "Dungeon",
    start: "door",
    vars: { hp: 10 },
    characters: [{ id: "g", name: "Grim" }],
    mode: "infinite",
    entry_points: { next_scene: "next" },
  },
);

function writtenScene(tag: string, choices = 3): Scene {
  return {
    id: `w_${tag}`,
    lines: [{ text: `${tag} happens.` }],
    choices: Array.from({ length: choices }, (_, i) => ({
      id: `${tag}_${i}`,
      text: `${tag} option ${i}`,
      goto: `gen_${tag}_${i}`,
    })),
  } as Scene;
}

interface Call {
  req: Req;
  signal: AbortSignal;
  resolve: (s: Scene) => void;
  reject: (e: unknown) => void;
}

/** A WriteAhead over a fake writer whose calls the test answers; `notify` is its change callback. */
function writeAhead(notify?: () => void) {
  const calls: Call[] = [];
  const write = (req: Req, signal: AbortSignal) =>
    new Promise<Scene>((resolve, reject) => {
      calls.push({ req, signal, resolve, reject });
      signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
  return { calls, wa: new L.WriteAhead(write, notify) };
}

const start = () => R.initialState(m);
/** Depth-1 plan with nothing written yet. */
const plan1 = (s: State) => L.planWriteAhead(m, s, 1, () => undefined);
/** Read `from` to its choices and pick `choiceId`: the live state waiting on the next scene. */
const pick = (from: State, choiceId = "c1") => R.reduce(m, L.readToStop(m, from), { type: "choose", choiceId });
const reqOf = (live: State) => L.nextSceneRequest(m, live) as Req;
/** The live call for `live`'s next scene, through the write-ahead cache. */
const take = (wa: InstanceType<typeof L.WriteAhead>, live: State) => {
  const req = reqOf(live);
  return { req, scene: wa.take(req, L.writeKey(live, req)) };
};
/** A hold the test releases; `release()` is valid once the hold's task has started. */
function heldTask() {
  let release: () => void = () => undefined;
  return { task: () => new Promise<void>((r) => (release = r)), release: () => release() };
}

test("the request written ahead is exactly the one the live call sends after the pick", () => {
  const planned = plan1(start());
  assert.equal(planned.length, 3, "one scene per offered choice");
  const live = pick(start());
  assert.equal(live.status, "generating");
  const req = L.nextSceneRequest(m, live);
  assert.ok(req);
  const hit = planned.find((p) => p.key === L.writeKey(live, req));
  assert.ok(hit, "the picked branch was written ahead under the live key");
  assert.deepEqual(hit.req, req);
  assert.equal(req.state.vars.hp, 9, "the choice's effects are in the request");
  assert.deepEqual(req.state.history.slice(-3), ["Narrator: Dark.", "Grim: Go on.", "(Player chose: In)"]);
});

test("a different path never reuses a written scene", () => {
  const s = start();
  const a = plan1(s);
  const b = plan1(R.reduce(m, s, { type: "set", vars: { gold: 3 } }));
  const c = plan1(R.reduce(m, R.reduce(m, s, { type: "advance" }), { type: "start" }));
  for (const x of a) {
    assert.ok(!b.some((y) => y.key === x.key), "vars are part of the key");
    assert.ok(!c.some((y) => y.key === x.key), "a new playthrough is too");
  }
});

test("planning ahead leaves the real state untouched", () => {
  const s = start();
  const before = structuredClone(s);
  L.planWriteAhead(m, s, 2, () => writtenScene("x"));
  assert.deepEqual(s, before);
});

test("a written scene is used at once; an unchosen branch is dropped", async () => {
  const { calls, wa } = writeAhead();
  const plan = plan1(start());
  wa.plan(plan);
  assert.equal(calls.length, 1, "one background call at a time");
  calls[0].resolve(writtenScene("a"));
  await tick();
  assert.equal(calls.length, 2);
  const live = pick(start());
  const scene = await take(wa, live).scene;
  assert.equal(scene.id, "w_a");
  assert.equal(calls.length, 2, "a hit makes no call");
  assert.equal(wa.stats.hits, 1);
  const next = R.reduce(m, live, { type: "addScene", scene, asId: "gen_1", rev: live.rev });
  wa.plan(plan1(next));
  assert.ok(calls[1].signal.aborted, "the running sibling write is stopped");
  assert.equal(wa.written(plan[1].key), undefined);
  assert.equal(wa.written(plan[2].key), undefined);
  assert.ok(wa.stats.dropped >= 2);
});

test("a cached scene for another request is a miss", async () => {
  const { calls, wa } = writeAhead();
  wa.plan(plan1(start()));
  calls[0].resolve(writtenScene("a"));
  await tick();
  const other = R.reduce(m, L.readToStop(m, start()), { type: "set", vars: { gold: 1 } });
  const { req, scene } = take(wa, R.reduce(m, other, { type: "choose", choiceId: "c1" }));
  await tick();
  assert.equal(wa.stats.misses, 1);
  const own = calls.at(-1) as Call;
  assert.equal(own.req, req, "the live call sends its own request");
  own.resolve(writtenScene("live"));
  assert.equal((await scene).id, "w_live");
});

test("background writing waits for the player and yields the GPU to a live call", async () => {
  const { calls, wa } = writeAhead();
  const plan = plan1(start());
  const player = heldTask();
  const held = wa.hold(player.task, false);
  wa.plan(plan);
  assert.equal(calls.length, 0, "nothing starts while a player call waits");
  player.release();
  await held;
  assert.equal(calls.length, 1);
  const running = calls[0];
  const elsewhere = R.reduce(m, L.readToStop(m, start()), { type: "set", vars: { hp: 2 } });
  const { req, scene } = take(wa, R.reduce(m, elsewhere, { type: "choose", choiceId: "c2" }));
  assert.ok(running.signal.aborted);
  await tick();
  const own = calls.at(-1) as Call;
  assert.equal(own.req, req);
  own.resolve(writtenScene("live"));
  await scene;
  await tick();
  assert.equal(calls.length, 2, "nothing restarts before the next plan");
  wa.plan(plan);
  assert.equal(calls.at(-1)?.req, running.req, "the stopped write is retried when it is still wanted");
});

test("painting the scene on screen stops a write already running, which runs again after", async () => {
  const { calls, wa } = writeAhead();
  wa.plan(plan1(start()));
  const running = calls[0];
  const paint = heldTask();
  const art = wa.hold(paint.task);
  assert.ok(running.signal.aborted, "the art does not wait behind a speculative write");
  await tick();
  assert.equal(calls.length, 1, "nothing starts while the art paints");
  paint.release();
  await art;
  await tick();
  assert.equal(calls.at(-1)?.req, running.req, "the stopped write is retried");
});

test("a pick while its scene is being written waits on that write", async () => {
  const { calls, wa } = writeAhead();
  wa.plan(plan1(start()));
  const { scene } = take(wa, pick(start()));
  assert.equal(calls[0].signal.aborted, false, "the same write is kept");
  calls[0].resolve(writtenScene("a"));
  assert.equal((await scene).id, "w_a");
  assert.equal(calls.length, 1);
});

test("load and restart cancel everything written ahead", async () => {
  let notified = 0;
  const { calls, wa } = writeAhead(() => notified++);
  const plan = plan1(start());
  wa.plan(plan);
  calls[0].resolve(writtenScene("a"));
  await tick();
  assert.equal(notified, 1);
  const inFlight = calls[1];
  wa.reset();
  assert.ok(inFlight.signal.aborted);
  assert.equal(wa.written(plan[0].key), undefined);
  inFlight.resolve(writtenScene("late"));
  await tick();
  assert.equal(wa.written(plan[1].key), undefined, "a late answer is dropped");
  assert.equal(notified, 1);
});

test("a writer closed by an effect cleanup works again once reopened", async () => {
  const { calls, wa } = writeAhead();
  wa.close();
  wa.open();
  wa.plan(plan1(start()));
  await tick();
  assert.equal(calls.length, 1);
});

test("an ending drops every branch still queued or being written", async () => {
  const { calls, wa } = writeAhead();
  wa.plan(plan1(start()));
  const running = calls[0];
  wa.plan([]);
  assert.ok(running.signal.aborted);
  running.reject(new Error("stopped"));
  await tick();
  assert.equal(calls.length, 1, "nothing queued is started after the ending");
});

test("depth 2 writes below written scenes, capped", () => {
  const one = L.planWriteAhead(m, start(), 1, () => writtenScene("x"));
  assert.equal(one.length, 3);
  const two = L.planWriteAhead(m, start(), 2, () => writtenScene("x", 4));
  assert.equal(two.length, L.MAX_WRITE_AHEAD);
  assert.deepEqual(two.slice(0, 3).map((p) => p.key), one.map((p) => p.key), "nearest scenes first");
  assert.match(two[3].req.state.choice, /^x option/);
  assert.deepEqual(L.planWriteAhead(m, start(), 0, () => undefined), []);
});

test("scene text never waits behind painting", async () => {
  const lanes = new L.Lanes();
  const paint = heldTask();
  const art = lanes.run("art", paint.task);
  let wrote = false;
  await lanes.run("text", async () => {
    wrote = true;
  });
  assert.ok(wrote, "text ran while the art call was still open");
  let second = false;
  const art2 = lanes.run("art", async () => {
    second = true;
  });
  await tick();
  assert.equal(second, false, "art stays one call at a time");
  paint.release();
  await art;
  await art2;
  assert.ok(second);
});

test("restarting during a live miss cancels it and the next playthrough writes ahead at once", async () => {
  const { calls, wa } = writeAhead();
  const missed = take(wa, pick(start())).scene.catch(() => "stopped");
  assert.equal(calls.length, 1);
  wa.close();
  assert.ok(calls[0].signal.aborted, "the live request is cancelled");
  wa.open();
  wa.plan(plan1(start()));
  assert.equal(calls.length, 2, "the new playthrough is not held by the old miss");
  assert.equal(await missed, "stopped");
  await tick();
  wa.plan([]);
});

test("a new playthrough cancels a live miss without letting it end a later hold early", async () => {
  const { calls, wa } = writeAhead();
  const missed = take(wa, pick(start())).scene.catch(() => "stopped");
  wa.reset();
  assert.ok(calls[0].signal.aborted, "the live request is cancelled");
  const player = heldTask();
  const held = wa.hold(player.task, false);
  assert.equal(await missed, "stopped");
  wa.plan(plan1(start()));
  assert.equal(calls.length, 1, "the hold still keeps background writing waiting");
  player.release();
  await held;
  assert.equal(calls.length, 2);
});

test("a failed request is asked again on a later visit, a finished one is not", async () => {
  const requested = new Set<string>();
  let calls = 0;
  const failing = () => {
    calls++;
    return Promise.reject(new Error("no GPU"));
  };
  const first = L.requestOnce(requested, "0:art:a", failing);
  assert.ok(first);
  assert.equal(L.requestOnce(requested, "0:art:a", failing), null, "no second call while the first is out");
  await first.catch(() => undefined);
  await tick();
  assert.equal(requested.has("0:art:a"), false);
  await L.requestOnce(requested, "0:art:a", failing)?.catch(() => undefined);
  assert.equal(calls, 2);
  await L.requestOnce(requested, "0:art:b", () => Promise.resolve(1));
  await tick();
  assert.equal(L.requestOnce(requested, "0:art:b", () => Promise.resolve(2)), null);
});
