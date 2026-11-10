// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { storyFixture, storyManifest } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/story/reducer.ts");
type Manifest = import("../src/features/flows/types.ts").StoryManifest;
type State = import("../src/features/flows/player/story/reducer.ts").StoryState;

const picnic = storyFixture<Manifest>("lantern-picnic");
const moon = storyFixture<Manifest>("sloth-moon");
const live = { mode: "live", entry_points: { next_scene: "next" } } as const;

function readToChoice(m: Manifest, from: State): State {
  let s = from;
  for (let guard = 0; s.status === "reading" && guard < 500; guard++) s = R.reduce(m, s, { type: "advance" });
  return s;
}

function must<T>(v: T | null | undefined, what: string): T {
  assert.ok(v !== null && v !== undefined, what);
  return v as T;
}

function chooseText(m: Manifest, s: State, text: string): State {
  const choices = R.visibleChoices(must(R.sceneOf(m, s, s.sceneId), "scene"), s.vars);
  const c = choices.find((x) => x.text.startsWith(text));
  assert.ok(c, `no choice "${text}" in ${s.sceneId}: ${choices.map((x) => x.text)}`);
  return R.reduce(m, s, { type: "choose", choiceId: c.id });
}

const lineText = (m: Manifest, s: State) => R.currentLine(m, s)?.text;

test("the picnic plays to the cake ending through the creek", () => {
  let s = R.initialState(picnic, 1);
  assert.equal(s.status, "reading");
  assert.equal(lineText(picnic, s)?.startsWith("The sun is melting"), true);
  s = readToChoice(picnic, s);
  assert.equal(s.status, "choosing");
  s = chooseText(picnic, s, "The creek");
  assert.equal(s.vars.snacks, 2);
  s = readToChoice(picnic, s);
  assert.equal(s.status, "ended");
  assert.equal(s.ending?.title, "The Cake Incident");
  assert.ok(s.backlog.some((b) => b.speaker === R.CHOICE_SPEAKER));
});

test("the market route makes friends and reaches the party ending", () => {
  let s = readToChoice(picnic, R.initialState(picnic, 1));
  s = chooseText(picnic, s, "The lantern market");
  assert.equal(s.vars.friends, 2);
  s = readToChoice(picnic, s);
  assert.equal(s.ending?.title, "The Biggest Picnic");
});

test("random events are deterministic per seed and both outcomes occur", () => {
  const endings = new Set<string>();
  const firstScenes = new Set<string>();
  for (let seed = 0; seed < 24; seed++) {
    const play = () => {
      let s = chooseText(picnic, readToChoice(picnic, R.initialState(picnic, seed)), "The hill");
      s = R.reduce(picnic, s, { type: "advance" });
      while (s.status === "reading" && s.sceneId === "s_hill") s = R.reduce(picnic, s, { type: "advance" });
      firstScenes.add(s.sceneId);
      return readToChoice(picnic, s);
    };
    const a = play();
    assert.deepEqual(a, play(), "same seed, same playthrough");
    endings.add(must(a.ending, "ending").title);
  }
  assert.deepEqual([...firstScenes].sort(), ["s_goat", "s_stars"]);
  assert.ok(endings.has("The Biggest Picnic") && endings.has("Just the Two of Us"));
});

test("conditions gate the moon's secret ending and save/load round-trips", () => {
  let s = readToChoice(moon, R.initialState(moon, 3));
  s = readToChoice(moon, chooseText(moon, s, "Climb down"));
  s = chooseText(moon, s, "Drink the tea");
  assert.equal(s.vars.kindness, 1);
  const saved = JSON.parse(JSON.stringify(s)) as State;
  const kindEnough = readToChoice(moon, R.reduce(moon, s, { type: "set", vars: { kindness: 2 } }));
  assert.equal(kindEnough.ending?.title, "Starlit Friends");
  const plain = readToChoice(moon, s);
  assert.equal(plain.ending?.title, "The Moon Comes Home");
  const loaded = R.reduce(moon, plain, { type: "load", state: saved });
  assert.equal(loaded.rev, plain.rev + 1);
  assert.equal(readToChoice(moon, loaded).ending?.title, "The Moon Comes Home");
  assert.equal(R.reduce(moon, plain, { type: "load", state: { ...saved, v: 99 } }), plain);
});

test("interpolation and hidden lines", () => {
  const m = storyManifest(
    [{
      id: "a",
      lines: [
        { text: "HP is {{hp}}." },
        { text: "secret", if: { op: ">", args: [{ var: "hp" }, 5] } },
        { text: "bye {{missing}}!" },
      ],
      choices: [],
    }],
    { vars: { hp: 3 } },
  );
  let s = R.initialState(m);
  assert.equal(R.interpolate(must(R.currentLine(m, s), "line").text, s.vars), "HP is 3.");
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(R.interpolate(must(R.currentLine(m, s), "line").text, s.vars), "bye !");
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(s.status, "ended");
  assert.equal(s.ending?.title, "The End");
});

test("infinite mode waits for a written scene and drops stale ones", () => {
  const m = storyManifest(
    [{
      id: "door",
      lines: [{ text: "Dark." }],
      choices: [{ id: "c1", text: "In", goto: "gen_c_1", set: { hp: 9 } }],
    }],
    { start: "door", vars: { hp: 10 }, mode: "infinite", entry_points: { next_scene: "next" } },
  );
  let s = R.reduce(m, readToChoice(m, R.initialState(m)), { type: "choose", choiceId: "c1" });
  assert.equal(s.status, "generating");
  assert.deepEqual(s.pending, { sceneId: "gen_c_1", choice: "In" });
  assert.equal(s.vars.hp, 9);
  const choices = [{ id: "x", text: "Run", goto: "gen_x" }];
  const scene = { id: "gen_abc", lines: [{ text: "A goblin!" }], choices };
  assert.equal(
    R.reduce(m, s, { type: "addScene", scene, asId: "gen_c_1", rev: s.rev + 1 }),
    s,
    "stale rev ignored",
  );
  s = R.reduce(m, s, { type: "addScene", scene, asId: "gen_c_1", rev: s.rev });
  assert.equal(s.status, "reading");
  assert.equal(lineText(m, s), "A goblin!");
  assert.equal(R.sceneOf(m, s, "gen_c_1")?.id, "gen_c_1");
});

test("a router loop is reported instead of hanging", () => {
  const m = storyManifest([
    { id: "a", lines: [], choices: [], next: "b" },
    { id: "b", lines: [], choices: [], next: "a" },
  ]);
  assert.equal(R.initialState(m).status, "error");
});

test("a scene whose dialogue is still being written waits for it", () => {
  const m = storyManifest(
    [
      { id: "a", lines: [{ text: "Hello." }], choices: [{ id: "go", text: "", goto: "b" }] },
      { id: "b", lines: [], generate_lines: true, choices: [{ id: "end", text: "", goto: "c" }] },
      { id: "c", lines: [{ text: "Bye." }], choices: [] },
    ],
    live,
  );
  let s = R.reduce(m, R.initialState(m), { type: "advance" });
  assert.equal(s.sceneId, "b");
  assert.equal(s.status, "reading");
  assert.equal(lineText(m, s), "...");
  const before = s.backlog.length;
  s = R.reduce(m, s, { type: "setLines", sceneId: "b", lines: [{ text: "Written." }], rev: s.rev });
  assert.equal(lineText(m, s), "Written.");
  assert.equal(s.backlog.length, before + 1, "the earlier line stays in the history");
});

test("the placeholder cannot be read past until the lines arrive or fail", () => {
  const m = storyManifest(
    [
      { id: "a", lines: [], generate_lines: true, choices: [{ id: "go", text: "", goto: "b" }] },
      { id: "b", lines: [{ text: "Next." }], choices: [] },
    ],
    live,
  );
  let s = R.initialState(m);
  assert.equal(lineText(m, s), "...");
  assert.equal(R.reduce(m, s, { type: "advance" }), s, "advance waits for the lines");
  s = R.reduce(m, s, { type: "linesFailed", sceneId: "a", rev: s.rev });
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(s.sceneId, "b");
});

test("a re-roll rebuilds only the story or world, never a side branch", async () => {
  const { rerollTargets } = await import("../src/features/flows/player/player-api.ts");
  const doc = {
    nodes: [
      { id: "gen", type: "story.generator" },
      { id: "art", type: "story.scene_art" },
      { id: "train", type: "training.finetune" },
      { id: "save", type: "output.library" },
      { id: "off", type: "story.illustrate", disabled: true },
    ],
  } as never;
  assert.deepEqual(rerollTargets(doc), ["gen"]);
});

test("a save whose scene no longer exists is refused", () => {
  const s = R.initialState(picnic);
  assert.equal(R.saveFits(picnic, s), true);
  const stale = { ...s, sceneId: "a_scene_from_before_the_reroll" };
  assert.equal(R.saveFits(picnic, stale), false);
  assert.equal(R.reduce(picnic, s, { type: "load", state: stale }), s);
});

test("lines written after the reader moved on do not replace the scene under them", () => {
  const m = storyManifest(
    [{ id: "a", lines: [{ text: "One." }, { text: "Two." }, { text: "Three." }], generate_lines: true, choices: [] }],
    live,
  );
  const s = R.reduce(m, R.initialState(m), { type: "advance" });
  assert.equal(lineText(m, s), "Two.");
  const after = R.reduce(m, s, { type: "setLines", sceneId: "a", lines: [{ text: "Written." }], rev: s.rev });
  assert.equal(after, s);
  assert.equal(lineText(m, after), "Two.");
});

test("dialogue written while the reader is mid-scene is held for the next visit", () => {
  const manifest = {
    start: "a",
    scenes: [{ id: "a", lines: [{ text: "one" }, { text: "two" }, { text: "three" }], generate_lines: true }],
  } as never;
  const start = R.initialState(manifest);
  const lines = [{ text: "new" }];
  const waits = (s: typeof start, rev: number) => R.linesWaitForNextVisit(manifest, s, rev, "a");
  assert.equal(waits(start, start.rev), false, "at the scene start lines land now");
  const midScene = { ...start, line: 2 };
  assert.equal(waits(midScene, start.rev), true);
  assert.equal(
    R.reduce(manifest, midScene, { type: "setLines", sceneId: "a", lines, rev: start.rev }),
    midScene,
    "the reducer refuses them mid-scene, so the player must hold them",
  );
  assert.equal(waits(midScene, start.rev + 1), false, "a stale revision is dropped");
});
