// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const W = await import("../src/features/flows/player/world/world-state.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type State = import("../src/features/flows/player/world/world-state.ts").WorldState;

const SPEC = {
  title: "T",
  seed: 1,
  width: 20,
  height: 20,
  tile_size: 32,
  biome: "meadow",
  tileset: {},
  player: { name: "Pip" },
  npcs: [{ id: "tom", name: "Tom", persona: "", greeting: "hi" }],
  items: [],
  quests: [],
  regions: [],
  vars: {},
  intro_text: "",
} as unknown as Spec;

function talk(n: number, summarising: boolean, s = W.initialWorldState(SPEC)) {
  for (let i = 0; i < n; i++)
    s = W.addTurn(
      s,
      "tom",
      { role: i % 2 ? "npc" : "player", text: `t${i}` },
      summarising,
    );
  return s;
}

const texts = (s: State) => (s.talks.tom ?? []).map((t) => t.text);

test("turns are numbered and a summary is due only past the threshold", () => {
  const s = talk(W.SUMMARY_AFTER, true);
  assert.deepEqual(
    s.talks.tom.map((t) => t.seq),
    [...Array(W.SUMMARY_AFTER).keys()],
  );
  assert.equal(W.summaryDue(s, "tom"), null);
  const due = W.summaryDue(talk(W.SUMMARY_AFTER + 1, true), "tom");
  assert.ok(due);
  assert.deepEqual(
    due.turns.map((t) => t.text),
    ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
  );
  assert.equal(due.upto, 9);
});

test("with summaries on, a turn is never dropped before it is summarised", () => {
  let s = talk(100, true);
  assert.equal(s.talks.tom.length, 100);
  assert.equal(texts(s)[0], "t0");
  s = W.foldSummary(s, "tom", "Pip asked about fish.", 79);
  assert.equal(s.talks.tom.length, 40);
  assert.equal(texts(s)[0], "t60");
  assert.deepEqual(
    W.talkHistory(s, "tom").map((t) => t.text),
    Array.from({ length: 20 }, (_, i) => `t${80 + i}`),
  );
  assert.equal(W.foldSummary(s, "tom", "old", 40), s);
  s = W.addTurn(s, "tom", { role: "player", text: "next" }, true);
  assert.equal(s.talks.tom.at(-1)?.seq, 100);
});

test("with summaries off, conversations are cut as before", () => {
  const s = talk(50, false);
  assert.equal(s.talks.tom.length, 40);
  assert.equal(texts(s)[0], "t10");
  assert.equal(W.talkHistory(s, "tom").length, W.MAX_HISTORY);
});

test("saves from before summaries load and keep talking", () => {
  const old = {
    v: 2,
    player: null,
    inventory: [],
    picked: [],
    activeQuests: [],
    completedQuests: [],
    vars: {},
    talks: {
      tom: Array.from({ length: 30 }, (_, i) => ({
        role: i % 2 ? "npc" : "player",
        text: `t${i}`,
      })),
    },
    clock: 0,
    npc: {},
  };
  const s = W.deserialize(SPEC, old);
  assert.ok(s);
  assert.deepEqual(s.summaries, {});
  assert.equal(W.summaryDue(s, "tom")?.upto, 9);
  const next = W.addTurn(s, "tom", { role: "player", text: "x" }, true);
  assert.equal(next.talks.tom.at(-1)?.seq, 30);
  const folded = W.foldSummary(next, "tom", "We met.", 9);
  const back = W.deserialize(SPEC, JSON.parse(JSON.stringify(folded)));
  assert.deepEqual(back?.summaries, { tom: { text: "We met.", upto: 9 } });
});

test("the client sends as many turns as the backend reads", () => {
  const py = readFileSync(
    new URL("../../backend/core/flows/world.py", import.meta.url),
    "utf8",
  );
  assert.equal(Number(/^MAX_HISTORY = (\d+)$/m.exec(py)?.[1]), W.MAX_HISTORY);
});

test("every turn the world player adds keeps unsummarised turns when summaries are on", () => {
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/world/world-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const calls = src.split("addTurn(").slice(1);
  assert.ok(calls.length >= 4);
  for (const rest of calls) {
    let depth = 1;
    let i = 0;
    while (depth > 0 && i < rest.length) {
      if (rest[i] === "(") depth++;
      else if (rest[i] === ")") depth--;
      i++;
    }
    assert.match(
      rest.slice(0, i),
      /summarising/,
      `addTurn(${rest.slice(0, 60)}`,
    );
  }
});
