// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { NextSceneRequest as Req } from "../src/features/flows/player/story/lookahead.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/story/reducer.ts");
const L = await import("../src/features/flows/player/story/lookahead.ts");
type Manifest = import("../src/features/flows/types.ts").StoryManifest;
type Scene = import("../src/features/flows/types.ts").StoryScene;
type State = import("../src/features/flows/player/story/reducer.ts").StoryState;

const m: Manifest = {
  title: "Dungeon",
  start: "door",
  vars: { hp: 10 },
  characters: [{ id: "g", name: "Grim" }],
  mode: "infinite",
  entry_points: { next_scene: "next" },
  scenes: [
    {
      id: "door",
      lines: [{ text: "Dark." }, { speaker: "g", text: "Go on." }],
      choices: [
        { id: "c1", text: "In", goto: "gen_1", set: { hp: 9 } },
        { id: "c2", text: "Back", goto: "gen_2" },
        { id: "c3", text: "Wait", goto: "gen_3" },
      ],
    },
  ],
} as Manifest;

function readToChoice(from: State): State {
  return L.readToStop(m, from);
}

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

function fakeWriter() {
  const calls: Call[] = [];
  const write = (req: Req, signal: AbortSignal) =>
    new Promise<Scene>((resolve, reject) => {
      calls.push({ req, signal, resolve, reject });
      signal.addEventListener("abort", () =>
        reject(new DOMException("aborted", "AbortError")),
      );
    });
  return { calls, write };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("the request written ahead is exactly the one the live call sends after the pick", () => {
  const start = R.initialState(m);
  const planned = L.planWriteAhead(m, start, 1, () => undefined);
  assert.equal(planned.length, 3, "one scene per offered choice");
  const live = R.reduce(m, readToChoice(start), {
    type: "choose",
    choiceId: "c1",
  });
  assert.equal(live.status, "generating");
  const req = L.nextSceneRequest(m, live);
  assert.ok(req);
  const key = L.writeKey(live, req);
  const hit = planned.find((p) => p.key === key);
  assert.ok(hit, "the picked branch was written ahead under the live key");
  assert.deepEqual(hit.req, req);
  assert.equal(req.state.vars.hp, 9, "the choice's effects are in the request");
  assert.deepEqual(req.state.history.slice(-3), [
    "Narrator: Dark.",
    "Grim: Go on.",
    "(Player chose: In)",
  ]);
});

test("a different path never reuses a written scene", () => {
  const start = R.initialState(m);
  const a = L.planWriteAhead(m, start, 1, () => undefined);
  const hurt = R.reduce(m, start, { type: "set", vars: { gold: 3 } });
  const b = L.planWriteAhead(m, hurt, 1, () => undefined);
  const later = R.reduce(m, R.reduce(m, start, { type: "advance" }), {
    type: "start",
  });
  const c = L.planWriteAhead(m, later, 1, () => undefined);
  for (const x of a) {
    assert.ok(!b.some((y) => y.key === x.key), "vars are part of the key");
    assert.ok(!c.some((y) => y.key === x.key), "a new playthrough is too");
  }
});

test("planning ahead leaves the real state untouched", () => {
  const start = R.initialState(m);
  const before = structuredClone(start);
  L.planWriteAhead(m, start, 2, () => writtenScene("x"));
  assert.deepEqual(start, before);
});

test("a written scene is used at once; an unchosen branch is dropped", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const start = R.initialState(m);
  const plan = L.planWriteAhead(m, start, 1, () => undefined);
  wa.plan(plan);
  assert.equal(calls.length, 1, "one background call at a time");
  calls[0].resolve(writtenScene("a"));
  await tick();
  assert.equal(calls.length, 2);
  const live = R.reduce(m, readToChoice(start), {
    type: "choose",
    choiceId: "c1",
  });
  const req = L.nextSceneRequest(m, live) as Req;
  const scene = await wa.take(req, L.writeKey(live, req));
  assert.equal(scene.id, "w_a");
  assert.equal(calls.length, 2, "a hit makes no call");
  assert.equal(wa.stats.hits, 1);
  const next = R.reduce(m, live, {
    type: "addScene",
    scene,
    asId: "gen_1",
    rev: live.rev,
  });
  wa.plan(L.planWriteAhead(m, next, 1, () => undefined));
  assert.ok(calls[1].signal.aborted, "the running sibling write is stopped");
  assert.equal(wa.written(plan[1].key), undefined);
  assert.equal(wa.written(plan[2].key), undefined);
  assert.ok(wa.stats.dropped >= 2);
});

test("a cached scene for another request is a miss", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const start = R.initialState(m);
  wa.plan(L.planWriteAhead(m, start, 1, () => undefined));
  calls[0].resolve(writtenScene("a"));
  await tick();
  const other = R.reduce(m, readToChoice(start), {
    type: "set",
    vars: { gold: 1 },
  });
  const live = R.reduce(m, other, { type: "choose", choiceId: "c1" });
  const req = L.nextSceneRequest(m, live) as Req;
  const p = wa.take(req, L.writeKey(live, req));
  await tick();
  assert.equal(wa.stats.misses, 1);
  const own = calls.at(-1) as Call;
  assert.equal(own.req, req, "the live call sends its own request");
  own.resolve(writtenScene("live"));
  assert.equal((await p).id, "w_live");
});

test("background writing waits for the player and yields the GPU to a live call", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const plan = L.planWriteAhead(m, R.initialState(m), 1, () => undefined);
  let release: () => void = () => undefined;
  const held = wa.hold(() => new Promise<void>((r) => (release = r)), false);
  wa.plan(plan);
  assert.equal(calls.length, 0, "nothing starts while a player call waits");
  release();
  await held;
  assert.equal(calls.length, 1);
  const running = calls[0];
  const elsewhere = R.reduce(m, readToChoice(R.initialState(m)), {
    type: "set",
    vars: { hp: 2 },
  });
  const live = R.reduce(m, elsewhere, { type: "choose", choiceId: "c2" });
  const req = L.nextSceneRequest(m, live) as Req;
  const p = wa.take(req, L.writeKey(live, req));
  assert.ok(running.signal.aborted);
  await tick();
  const own = calls.at(-1) as Call;
  assert.equal(own.req, req);
  own.resolve(writtenScene("live"));
  await p;
  await tick();
  assert.equal(calls.length, 2, "nothing restarts before the next plan");
  wa.plan(plan);
  assert.equal(
    calls.at(-1)?.req,
    running.req,
    "the stopped write is retried when it is still wanted",
  );
});

test("painting the scene on screen stops a write already running, which runs again after", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  wa.plan(L.planWriteAhead(m, R.initialState(m), 1, () => undefined));
  const running = calls[0];
  let release: () => void = () => undefined;
  const art = wa.hold(() => new Promise<void>((r) => (release = r)));
  assert.ok(
    running.signal.aborted,
    "the art does not wait behind a speculative write",
  );
  await tick();
  assert.equal(calls.length, 1, "nothing starts while the art paints");
  release();
  await art;
  await tick();
  assert.equal(calls.at(-1)?.req, running.req, "the stopped write is retried");
});

test("a pick while its scene is being written waits on that write", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const start = R.initialState(m);
  wa.plan(L.planWriteAhead(m, start, 1, () => undefined));
  const live = R.reduce(m, readToChoice(start), {
    type: "choose",
    choiceId: "c1",
  });
  const req = L.nextSceneRequest(m, live) as Req;
  const p = wa.take(req, L.writeKey(live, req));
  assert.equal(calls[0].signal.aborted, false, "the same write is kept");
  calls[0].resolve(writtenScene("a"));
  assert.equal((await p).id, "w_a");
  assert.equal(calls.length, 1);
});

test("load and restart cancel everything written ahead", async () => {
  const { calls, write } = fakeWriter();
  let notified = 0;
  const wa = new L.WriteAhead(write, () => notified++);
  const start = R.initialState(m);
  const plan = L.planWriteAhead(m, start, 1, () => undefined);
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
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  wa.close();
  wa.open();
  wa.plan(L.planWriteAhead(m, R.initialState(m), 1, () => undefined));
  await tick();
  assert.equal(calls.length, 1);
});

test("an ending drops every branch still queued or being written", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  wa.plan(L.planWriteAhead(m, R.initialState(m), 1, () => undefined));
  const running = calls[0];
  wa.plan([]);
  assert.ok(running.signal.aborted);
  running.reject(new Error("stopped"));
  await tick();
  assert.equal(calls.length, 1, "nothing queued is started after the ending");
});

test("depth 2 writes below written scenes, capped", () => {
  const start = R.initialState(m);
  const one = L.planWriteAhead(m, start, 1, () => writtenScene("x"));
  assert.equal(one.length, 3);
  const two = L.planWriteAhead(m, start, 2, () => writtenScene("x", 4));
  assert.equal(two.length, L.MAX_WRITE_AHEAD);
  assert.deepEqual(
    two.slice(0, 3).map((p) => p.key),
    one.map((p) => p.key),
    "nearest scenes first",
  );
  assert.match(two[3].req.state.choice, /^x option/);
  assert.deepEqual(
    L.planWriteAhead(m, start, 0, () => undefined),
    [],
  );
});

test("scene text never waits behind painting", async () => {
  const lanes = new L.Lanes();
  let paintDone: () => void = () => undefined;
  const art = lanes.run("art", () => new Promise<void>((r) => (paintDone = r)));
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
  paintDone();
  await art;
  await art2;
  assert.ok(second);
});

test("the story player routes scenes through the write-ahead and the two lanes", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/story/story-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.ok(!src.includes("queue.current"), "no shared text and art queue");
  assert.match(src, /lanes\.current\.run\("art"/);
  assert.match(src, /lanes\.current\.run\("text"/);
  assert.match(src, /\.take\(req, writeKey\(snapshot, req\)\)/);
  assert.match(
    src,
    /useEffect\(\(\) => \{\s*writeAhead\.reset\(\);\s*\}, \[writeAhead, state\.rev\]\)/,
  );
  assert.match(src, /current \? writeAhead\.hold\(job\) : job\(\)/);
  assert.match(
    src,
    /writeAhead\.open\(\);\s*return \(\) => writeAhead\.close\(\);/,
  );
  assert.match(
    src,
    /state\.status === "ended" \|\| state\.status === "error"\) \{\s*writeAhead\.plan\(\[\]\);/,
  );
});

test("restarting during a live miss cancels it and the next playthrough writes ahead at once", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const live = R.reduce(m, readToChoice(R.initialState(m)), {
    type: "choose",
    choiceId: "c1",
  });
  const req = L.nextSceneRequest(m, live) as Req;
  const missed = wa.take(req, L.writeKey(live, req)).catch(() => "stopped");
  assert.equal(calls.length, 1);
  wa.close();
  assert.ok(calls[0].signal.aborted, "the live request is cancelled");
  wa.open();
  wa.plan(L.planWriteAhead(m, R.initialState(m), 1, () => undefined));
  assert.equal(
    calls.length,
    2,
    "the new playthrough is not held by the old miss",
  );
  assert.equal(await missed, "stopped");
  await tick();
  wa.plan([]);
});

test("a new playthrough cancels a live miss without letting it end a later hold early", async () => {
  const { calls, write } = fakeWriter();
  const wa = new L.WriteAhead(write);
  const live = R.reduce(m, readToChoice(R.initialState(m)), {
    type: "choose",
    choiceId: "c1",
  });
  const req = L.nextSceneRequest(m, live) as Req;
  const missed = wa.take(req, L.writeKey(live, req)).catch(() => "stopped");
  wa.reset();
  assert.ok(calls[0].signal.aborted, "the live request is cancelled");
  let release: () => void = () => undefined;
  const held = wa.hold(() => new Promise<void>((r) => (release = r)), false);
  assert.equal(await missed, "stopped");
  wa.plan(L.planWriteAhead(m, R.initialState(m), 1, () => undefined));
  assert.equal(
    calls.length,
    1,
    "the hold still keeps background writing waiting",
  );
  release();
  await held;
  assert.equal(calls.length, 2);
});
