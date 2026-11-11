// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { flushMicrotasks, storyFixture, storyManifest } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/story/reducer.ts");
type Manifest = import("../src/features/flows/types.ts").StoryManifest;
type Scene = import("../src/features/flows/types.ts").StoryScene;
type State = import("../src/features/flows/player/story/reducer.ts").StoryState;

const moon = storyFixture<Manifest>("sloth-moon");

const staged: Manifest = storyManifest(
  [
    {
      id: "a",
      on_stage: ["lu"],
      transition: "fade",
      lines: [
        { text: "", stage: { show: "mo", expression: "happy", at: "left", with: "dissolve" } },
        { speaker: "mo", text: "Hi.", expression: "surprised" },
        { text: "", stage: { sound: "door" } },
        { text: "", stage: { hide: "lu", with: "slideright" } },
        { speaker: "mo", text: "Lu left." },
        { text: "", stage: { pause: 1.5 } },
        { text: "", stage: { show: "mo", at: "center" } },
        { speaker: null, text: "Quiet." },
        { text: "", stage: { hide: "mo" } },
      ],
      choices: [],
      next: "b",
    },
    { id: "b", lines: [{ text: "", stage: { sound: "door" } }], choices: [], next: "c" },
    { id: "c", on_stage: ["mo"], lines: [{ speaker: "mo", text: "Bye." }], choices: [] },
  ],
  {
    title: "Stage",
    sounds: { door: { type: "audio", ref: "door" } as never },
    characters: [
      { id: "mo", name: "Mo", side: "left" },
      { id: "lu", name: "Lu", side: "right" },
    ],
  },
);
/** `staged` with its characters and sounds, but other scenes. */
const restaged = (scenes: Scene[], extra: Partial<Manifest> = {}): Manifest => ({ ...staged, ...extra, scenes });

function step(s: State, n = 1): State {
  let out = s;
  for (let i = 0; i < n; i++) out = R.reduce(staged, out, { type: "advance" });
  return out;
}
const lineText = (m: Manifest, s: State) => R.currentLine(m, s)?.text;

test("direction lines are applied on the way past and never shown", () => {
  const s = R.initialState(staged, 1);
  assert.equal(s.line, 1);
  assert.equal(lineText(staged, s), "Hi.");
  const view = R.stageAt(staged, s);
  assert.deepEqual(
    view.cast.map((c) => [c.id, c.at, c.expression, c.enter]),
    [["lu", null, null, null], ["mo", "left", "happy", "dissolve"]],
  );
  assert.deepEqual(view.sounds, []);
  assert.deepEqual(s.backlog.map((b) => b.text), ["Hi."]);
});

test("hide takes a character off with its exit, and sounds cue once at the next stop", () => {
  const s = step(R.initialState(staged, 1));
  assert.equal(lineText(staged, s), "Lu left.");
  const view = R.stageAt(staged, s);
  assert.deepEqual(view.cast.map((c) => [c.id, c.enter]), [["mo", null]]);
  assert.deepEqual(view.leaving.map((c) => [c.id, c.exit]), [["lu", "slideright"]]);
  assert.deepEqual(view.sounds, ["door"]);
  const later = R.stageAt(staged, step(s, 2));
  assert.deepEqual(later.sounds, []);
  assert.deepEqual(later.leaving, []);
  assert.equal(later.cast[0].expression, "happy");
  assert.equal(later.cast[0].at, "center");
});

test("pause stops the reader on a text-less line until advanced", () => {
  const s = step(R.initialState(staged, 1), 2);
  assert.equal(s.status, "reading");
  const line = R.currentLine(staged, s);
  assert.equal(line?.text, "");
  assert.equal(R.pauseOf(line), 1.5);
  const next = step(s);
  assert.equal(lineText(staged, next), "Quiet.");
  assert.equal(R.pauseOf(R.currentLine(staged, next)), null);
});

test("trailing directions run before leaving, and a directions-only scene routes on", () => {
  const s = step(R.initialState(staged, 1), 4);
  assert.equal(s.sceneId, "c");
  assert.equal(lineText(staged, s), "Bye.");
  assert.equal(s.backlog.some((b) => b.text === ""), false);
});

test("a save pointing at a direction line loads onto the next readable line", () => {
  const old: State = { ...R.initialState(staged, 1), line: 2 };
  const loaded = R.reduce(staged, R.initialState(staged, 1), { type: "load", state: old });
  assert.equal(loaded.line, 4);
  assert.equal(lineText(staged, loaded), "Lu left.");
});

test("old manifests and v1 saves are untouched: stage is just on_stage", () => {
  assert.equal(R.STORY_SAVE_VERSION, 1);
  let s = R.initialState(moon, 3);
  const saved = JSON.parse(JSON.stringify(s)) as State;
  const loaded = R.reduce(moon, s, { type: "load", state: saved });
  assert.equal(loaded.line, saved.line);
  assert.equal(loaded.sceneId, saved.sceneId);
  for (let i = 0; i < 40 && s.status === "reading"; i++) {
    const view = R.stageAt(moon, s);
    assert.deepEqual(view.cast.map((c) => c.id), R.sceneOf(moon, s, s.sceneId)?.on_stage ?? []);
    assert.equal(view.cast.every((c) => c.enter === null && c.at === null), true);
    assert.deepEqual(view.sounds, []);
    s = R.reduce(moon, s, { type: "advance" });
  }
});

test("written lines replace a generated scene that opens with a direction", () => {
  const m = storyManifest(
    [{
      id: "a",
      lines: [{ text: "", stage: { show: "mo", at: "left" } }, { speaker: "mo", text: "Placeholder." }],
      generate_lines: true,
      choices: [],
    }],
    { characters: [{ id: "mo", name: "Mo", side: "left" }], mode: "live", entry_points: { next_scene: "next" } },
  );
  const s = R.initialState(m);
  assert.equal(s.line, 1);
  const lines = [{ speaker: "mo", text: "Written." }];
  const after = R.reduce(m, s, { type: "setLines", sceneId: "a", lines, rev: s.rev });
  assert.equal(lineText(m, after), "Written.");
  assert.equal(R.atSceneStart(m, s), true);
});

test("a cue before the last line plays once, not again when the choices open", () => {
  const m = restaged([{
    id: "a",
    lines: [{ text: "One." }, { text: "", stage: { sound: "door" } }, { text: "Two." }],
    choices: [{ id: "x", text: "Go", goto: "a" }],
  }]);
  const last = R.reduce(m, R.initialState(m, 1), { type: "advance" });
  assert.equal(lineText(m, last), "Two.");
  assert.deepEqual(R.stageAt(m, last).sounds, ["door"]);
  const choosing = R.reduce(m, last, { type: "advance" });
  assert.equal(choosing.status, "choosing");
  assert.deepEqual(R.stageAt(m, choosing).sounds, []);
});

test("cues passed while leaving a scene play at the next stop", () => {
  const arrived = step(R.initialState(staged, 1), 4);
  assert.equal(lineText(staged, arrived), "Bye.");
  assert.deepEqual(R.stageAt(staged, arrived).sounds, ["door"]);
  assert.deepEqual(R.stageAt(staged, step(arrived)).sounds, []);
  const m = restaged([{
    id: "a",
    lines: [{ text: "Hello." }, { text: "", stage: { sound: "door" } }],
    choices: [],
    ending: { title: "Out", kind: "neutral" },
  }]);
  const ended = R.reduce(m, R.initialState(m, 1), { type: "advance" });
  assert.equal(ended.status, "ended");
  assert.deepEqual(R.stageAt(m, ended).sounds, ["door"]);
  const loaded = R.reduce(staged, ended, { type: "load", state: arrived });
  assert.deepEqual(R.stageAt(staged, loaded).sounds, []);
});

for (const [name, loadedMeanwhile, expected] of [
  ["a cue whose media resolves after a save load stays silent", true, []],
  ["a cue whose media resolves in the same playthrough plays", false, ["door-url"]],
] as const) {
  test(name, async () => {
    const S = await import("../src/features/flows/player/story/sound-cues.ts");
    let resolve = (_url: string) => {};
    const lookup = () => new Promise<string>((r) => (resolve = r));
    const played: string[] = [];
    let rev = 0;
    const cued = rev;
    S.playCues(["door"], staged.sounds, lookup, () => rev === cued, async (url) => played.push(url));
    if (loadedMeanwhile) rev = 1;
    resolve("door-url");
    await flushMicrotasks();
    assert.deepEqual(played, expected);
  });
}

test("a carried cue plays once while the next scene generates", () => {
  const m = restaged(
    [{
      id: "a",
      lines: [{ text: "", stage: { sound: "door" } }],
      choices: [{ id: "c", text: "", goto: "unwritten" }],
    }],
    { mode: "infinite", entry_points: { next_scene: "next" } },
  );
  const s = R.initialState(m, 1);
  assert.equal(s.status, "generating");
  assert.deepEqual(R.stageAt(m, s).sounds, ["door"]);
});
