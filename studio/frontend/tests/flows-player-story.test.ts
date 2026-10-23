// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/story/reducer.ts");
type Manifest = import("../src/features/flows/types.ts").StoryManifest;
type State = import("../src/features/flows/player/story/reducer.ts").StoryState;

const picnic = JSON.parse(
  readFileSync(
    new URL("./fixtures/flows-lantern-picnic.story.json", import.meta.url),
    "utf8",
  ),
) as Manifest;
const moon = JSON.parse(
  readFileSync(
    new URL("./fixtures/flows-sloth-moon.story.json", import.meta.url),
    "utf8",
  ),
) as Manifest;

function readToChoice(m: Manifest, from: State): State {
  let s = from;
  let guard = 0;
  while (s.status === "reading" && guard++ < 500) {
    s = R.reduce(m, s, { type: "advance" });
  }
  return s;
}

function must<T>(v: T | null | undefined, what: string): T {
  assert.ok(v !== null && v !== undefined, what);
  return v as T;
}

function chooseText(m: Manifest, s: State, text: string): State {
  const scene = must(R.sceneOf(m, s, s.sceneId), "scene");
  const c = R.visibleChoices(scene, s.vars).find((x) =>
    x.text.startsWith(text),
  );
  assert.ok(
    c,
    `no choice "${text}" in ${s.sceneId}: ${R.visibleChoices(scene, s.vars).map((x) => x.text)}`,
  );
  return R.reduce(m, s, { type: "choose", choiceId: c.id });
}

test("the picnic plays to the cake ending through the creek", () => {
  let s = R.initialState(picnic, 1);
  assert.equal(s.status, "reading");
  assert.equal(
    R.currentLine(picnic, s)?.text.startsWith("The sun is melting"),
    true,
  );
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
      let s = readToChoice(picnic, R.initialState(picnic, seed));
      s = chooseText(picnic, s, "The hill");
      s = R.reduce(picnic, s, { type: "advance" });
      while (s.status === "reading" && s.sceneId === "s_hill")
        s = R.reduce(picnic, s, { type: "advance" });
      firstScenes.add(s.sceneId);
      return readToChoice(picnic, s);
    };
    const a = play();
    const b = play();
    assert.deepEqual(a, b, "same seed, same playthrough");
    endings.add(must(a.ending, "ending").title);
  }
  assert.deepEqual([...firstScenes].sort(), ["s_goat", "s_stars"]);
  assert.ok(
    endings.has("The Biggest Picnic") && endings.has("Just the Two of Us"),
  );
});

test("conditions gate the moon's secret ending and save/load round-trips", () => {
  let s = readToChoice(moon, R.initialState(moon, 3));
  s = chooseText(moon, s, "Climb down");
  s = readToChoice(moon, s);
  s = chooseText(moon, s, "Drink the tea");
  assert.equal(s.vars.kindness, 1);
  const saved = JSON.parse(JSON.stringify(s)) as State;
  const kindEnough = readToChoice(
    moon,
    R.reduce(moon, s, { type: "set", vars: { kindness: 2 } }),
  );
  assert.equal(kindEnough.ending?.title, "Starlit Friends");
  const plain = readToChoice(moon, s);
  assert.equal(plain.ending?.title, "The Moon Comes Home");
  const loaded = R.reduce(moon, plain, { type: "load", state: saved });
  assert.equal(loaded.rev, plain.rev + 1);
  assert.equal(readToChoice(moon, loaded).ending?.title, "The Moon Comes Home");
  assert.equal(
    R.reduce(moon, plain, { type: "load", state: { ...saved, v: 99 } }),
    plain,
  );
});

test("interpolation and hidden lines", () => {
  const m: Manifest = {
    title: "T",
    start: "a",
    vars: { hp: 3 },
    characters: [],
    mode: "fixed",
    scenes: [
      {
        id: "a",
        lines: [
          { text: "HP is {{hp}}." },
          { text: "secret", if: { op: ">", args: [{ var: "hp" }, 5] } },
          { text: "bye {{missing}}!" },
        ],
        choices: [],
      },
    ],
  };
  let s = R.initialState(m);
  assert.equal(
    R.interpolate(must(R.currentLine(m, s), "line").text, s.vars),
    "HP is 3.",
  );
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(
    R.interpolate(must(R.currentLine(m, s), "line").text, s.vars),
    "bye !",
  );
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(s.status, "ended");
  assert.equal(s.ending?.title, "The End");
});

test("infinite mode waits for a written scene and drops stale ones", () => {
  const m: Manifest = {
    title: "D",
    start: "door",
    vars: { hp: 10 },
    characters: [],
    mode: "infinite",
    entry_points: { next_scene: "next" },
    scenes: [
      {
        id: "door",
        lines: [{ text: "Dark." }],
        choices: [{ id: "c1", text: "In", goto: "gen_c_1", set: { hp: 9 } }],
      },
    ],
  };
  let s = readToChoice(m, R.initialState(m));
  s = R.reduce(m, s, { type: "choose", choiceId: "c1" });
  assert.equal(s.status, "generating");
  assert.deepEqual(s.pending, { sceneId: "gen_c_1", choice: "In" });
  assert.equal(s.vars.hp, 9);
  const scene = {
    id: "gen_abc",
    lines: [{ text: "A goblin!" }],
    choices: [{ id: "x", text: "Run", goto: "gen_x" }],
  };
  assert.equal(
    R.reduce(m, s, {
      type: "addScene",
      scene,
      asId: "gen_c_1",
      rev: s.rev + 1,
    }),
    s,
    "stale rev ignored",
  );
  s = R.reduce(m, s, { type: "addScene", scene, asId: "gen_c_1", rev: s.rev });
  assert.equal(s.status, "reading");
  assert.equal(R.currentLine(m, s)?.text, "A goblin!");
  assert.equal(R.sceneOf(m, s, "gen_c_1")?.id, "gen_c_1");
});

test("a router loop is reported instead of hanging", () => {
  const m: Manifest = {
    title: "L",
    start: "a",
    vars: {},
    characters: [],
    mode: "fixed",
    scenes: [
      { id: "a", lines: [], choices: [], next: "b" },
      { id: "b", lines: [], choices: [], next: "a" },
    ],
  };
  const s = R.initialState(m);
  assert.equal(s.status, "error");
});

test("a scene whose dialogue is still being written waits for it", () => {
  const m: Manifest = {
    title: "W",
    start: "a",
    vars: {},
    characters: [],
    mode: "live",
    entry_points: { next_scene: "next" },
    scenes: [
      {
        id: "a",
        lines: [{ text: "Hello." }],
        choices: [{ id: "go", text: "", goto: "b" }],
      },
      {
        id: "b",
        lines: [],
        generate_lines: true,
        choices: [{ id: "end", text: "", goto: "c" }],
      },
      { id: "c", lines: [{ text: "Bye." }], choices: [] },
    ],
  };
  let s = R.initialState(m);
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(s.sceneId, "b");
  assert.equal(s.status, "reading");
  assert.equal(R.currentLine(m, s)?.text, "...");
  const before = s.backlog.length;
  s = R.reduce(m, s, {
    type: "setLines",
    sceneId: "b",
    lines: [{ text: "Written." }],
    rev: s.rev,
  });
  assert.equal(R.currentLine(m, s)?.text, "Written.");
  assert.equal(
    s.backlog.length,
    before + 1,
    "the earlier line stays in the history",
  );
});

test("the placeholder cannot be read past until the lines arrive or fail", () => {
  const m: Manifest = {
    title: "W",
    start: "a",
    vars: {},
    characters: [],
    mode: "live",
    entry_points: { next_scene: "next" },
    scenes: [
      {
        id: "a",
        lines: [],
        generate_lines: true,
        choices: [{ id: "go", text: "", goto: "b" }],
      },
      { id: "b", lines: [{ text: "Next." }], choices: [] },
    ],
  };
  let s = R.initialState(m);
  assert.equal(R.currentLine(m, s)?.text, "...");
  assert.equal(
    R.reduce(m, s, { type: "advance" }),
    s,
    "advance waits for the lines",
  );
  s = R.reduce(m, s, { type: "linesFailed", sceneId: "a", rev: s.rev });
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(s.sceneId, "b");
});

test("a re-roll rebuilds only the story or world, never a side branch", async () => {
  const { rerollTargets } = await import(
    "../src/features/flows/player/player-api.ts"
  );
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
  const m: Manifest = {
    title: "W",
    start: "a",
    vars: {},
    characters: [],
    mode: "live",
    entry_points: { next_scene: "next" },
    scenes: [
      {
        id: "a",
        lines: [{ text: "One." }, { text: "Two." }, { text: "Three." }],
        generate_lines: true,
        choices: [],
      },
    ],
  };
  let s = R.initialState(m);
  s = R.reduce(m, s, { type: "advance" });
  assert.equal(R.currentLine(m, s)?.text, "Two.");
  const after = R.reduce(m, s, {
    type: "setLines",
    sceneId: "a",
    lines: [{ text: "Written." }],
    rev: s.rev,
  });
  assert.equal(after, s);
  assert.equal(R.currentLine(m, after)?.text, "Two.");
});
