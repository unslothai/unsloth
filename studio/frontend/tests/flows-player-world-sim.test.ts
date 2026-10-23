// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const S = await import("../src/features/flows/player/world/world-sim.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;

function spec(simulation?: Spec["simulation"]): Spec {
  return {
    title: "Smallville",
    seed: 1,
    width: 48,
    height: 36,
    tile_size: 32,
    biome: "town",
    tileset: {},
    player: { name: "Pip" },
    npcs: [
      {
        id: "isabella",
        name: "Isabella",
        persona: "",
        greeting: "hi",
        seed_memories: [{ text: "Party at Hobbs Cafe", importance: 8 }],
      },
      { id: "tom", name: "Tom", persona: "", greeting: "hey" },
    ],
    items: [{ id: "pie", name: "Pie", description: "" }],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    simulation,
  };
}

const SIM = {
  enabled: true,
  minutes_per_second: 2,
  start_minute: 480,
  places: [{ id: "cafe", name: "Hobbs Cafe", x: 2, y: 2, w: 4, h: 3 }],
  chat_cooldown_min: 120,
  calls_per_minute: 4,
};

test("v1 saves upgrade to v2 and classic worlds get no minds", () => {
  const classic = spec();
  const v1 = {
    v: 1,
    player: { x: 3, y: 4 },
    inventory: ["pie", "ghost"],
    picked: ["pie"],
    activeQuests: [],
    completedQuests: [],
    vars: { a: 1 },
    talks: { tom: [{ role: "npc", text: "hey" }] },
  };
  const up = W.deserialize(classic, v1);
  assert.ok(up);
  assert.equal(up.v, 2);
  assert.equal(up.clock, 0);
  assert.deepEqual(up.npc, {});
  assert.deepEqual(up.inventory, ["pie"]);
  assert.deepEqual(up.talks, v1.talks);
  assert.deepEqual(W.initialWorldState(classic).npc, {});

  const village = spec(SIM);
  const upSim = W.deserialize(village, v1);
  assert.ok(upSim);
  assert.equal(upSim.clock, 480);
  assert.equal(upSim.npc.isabella.memory[0].text, "Party at Hobbs Cafe");
  assert.equal(upSim.npc.isabella.memory[0].importance, 8);
  assert.equal(upSim.npc.tom.memory.length, 0);
  assert.equal(W.deserialize(village, { v: 3 }), null);
});

test("v2 saves round-trip clock, memories, plans and cooldowns", () => {
  const village = spec(SIM);
  let s = W.initialWorldState(village);
  s = W.withMind(s, "tom", (m) =>
    S.remember(m, [{ text: "Isabella invited me", importance: 7 }], 900),
  );
  s = {
    ...s,
    clock: 900,
    npc: S.markChat(s.npc, "tom", "isabella", 900),
  };
  s = W.withMind(s, "tom", (m) => ({
    ...m,
    plan: [{ start: 420, minutes: 60, activity: "Coffee", place: "cafe" }],
    planDay: 0,
  }));
  const back = W.deserialize(
    village,
    JSON.parse(JSON.stringify(W.serialize(s))),
  );
  assert.deepEqual(back, s);
});

test("memories cap by importance and recency, and recall refreshes them", () => {
  let m = S.emptyMind();
  m = S.remember(m, [{ text: "keep", importance: 9 }], 0);
  for (let i = 0; i < S.MAX_MEMORIES + 5; i++)
    m = S.remember(m, [{ text: `chatter ${i}`, importance: 2 }], i);
  assert.equal(m.memory.length, S.MAX_MEMORIES);
  assert.ok(m.memory.some((r) => r.text === "keep"));
  assert.ok(!m.memory.some((r) => r.text === "chatter 0"));
  const touched = S.touch(m, ["m0"], 600);
  assert.equal(touched.memory.find((r) => r.id === "m0")?.t_access, 10);
  assert.equal(S.remember(m, [{ text: "  ", importance: 3 }], 0), m);
});

test("clock, plan blocks, encounters and cooldowns", () => {
  assert.equal(S.clockLabel(1440 + 17 * 60 + 5), "Day 2 · 17:05");
  const plan = [
    { start: 480, minutes: 60, activity: "Bake bread", place: "cafe" },
    { start: 600, minutes: 120, activity: "Party", place: "cafe" },
  ];
  assert.equal(S.currentBlock(plan, 500)?.activity, "Bake bread");
  assert.equal(S.currentBlock(plan, 590), null);
  assert.equal(S.activityEmoji("Bake bread"), "🍞");
  assert.equal(S.activityEmoji("ponder"), "");
  const tiles = { a: { x: 1, y: 1 }, b: { x: 3, y: 2 }, c: { x: 20, y: 20 } };
  let minds: Record<string, ReturnType<typeof S.emptyMind>> = {
    a: S.emptyMind(),
    b: S.emptyMind(),
    c: S.emptyMind(),
  };
  assert.deepEqual(S.findEncounter(tiles, new Set(), minds, 0, 120), [
    "a",
    "b",
  ]);
  assert.equal(S.findEncounter(tiles, new Set(["a"]), minds, 0, 120), null);
  minds = S.markChat(minds, "a", "b", 100);
  assert.equal(S.findEncounter(tiles, new Set(), minds, 150, 120), null);
  assert.deepEqual(S.findEncounter(tiles, new Set(), minds, 220, 120), [
    "a",
    "b",
  ]);
  assert.equal(S.simOf(spec()), null);
  assert.equal(S.simOf(spec({ ...SIM, enabled: false })), null);
});

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

test("scheduler: one call in flight, plans first, a per-minute budget", async () => {
  let now = 0;
  const q = new S.BackgroundQueue(4, () => now);
  const started: string[] = [];
  const gates: Record<string, ReturnType<typeof deferred>> = {};
  const job = (key: string, kind: "plan" | "chat") => ({
    key,
    kind,
    run: () => {
      started.push(key);
      gates[key] = deferred();
      return gates[key].promise;
    },
  });
  q.push(job("chat:1", "chat"));
  q.push(job("plan:a", "plan"));
  q.push(job("plan:a", "plan"));
  for (let i = 2; i <= 6; i++) q.push(job(`chat:${i}`, "chat"));
  assert.equal(q.pending, 7);
  q.pump();
  q.pump();
  assert.deepEqual(started, ["plan:a"]);
  for (const key of ["plan:a", "chat:1", "chat:2"]) {
    gates[key].resolve();
    await flush();
  }
  assert.deepEqual(started, ["plan:a", "chat:1", "chat:2", "chat:3"]);
  gates["chat:3"].resolve();
  await flush();
  // Four starts inside the minute: the fifth waits for the window to move.
  assert.equal(started.length, 4);
  assert.equal(q.allows(), false);
  now = 60_001;
  q.pump();
  assert.deepEqual(started.at(-1), "chat:4");
});

test("scheduler: paused while the player talks, and the call in flight is aborted", async () => {
  const q = new S.BackgroundQueue(10, () => 0);
  const seen: string[] = [];
  let aborted = false;
  q.push({
    key: "chat:a:b",
    kind: "chat",
    run: (signal) =>
      new Promise<void>((_, reject) => {
        seen.push("chat");
        signal.addEventListener("abort", () => {
          aborted = true;
          reject(new Error("aborted"));
        });
      }),
  });
  q.push({
    key: "plan:a",
    kind: "plan",
    run: async () => {
      seen.push("plan");
    },
  });
  q.pump();
  assert.deepEqual(seen, ["plan"]);
  await flush();
  assert.deepEqual(seen, ["plan", "chat"]);
  q.setPaused(true);
  await flush();
  assert.equal(aborted, true);
  assert.equal(q.busy, null);
  q.push({
    key: "chat:c:d",
    kind: "chat",
    run: async () => {
      seen.push("later");
    },
  });
  q.pump();
  await flush();
  assert.deepEqual(seen, ["plan", "chat"]);
  q.setPaused(false);
  q.pump();
  await flush();
  assert.deepEqual(seen, ["plan", "chat", "later"]);
  const off = new S.BackgroundQueue(0, () => 0);
  off.push({
    key: "x",
    kind: "plan",
    run: async () => seen.push("never") as never,
  });
  off.pump();
  assert.ok(!seen.includes("never"));
});

test("a re-roll skips every player entry point, plan and chat included", async () => {
  const { rerollTargets, PLAYER_ENTRYPOINTS } = await import(
    "../src/features/flows/player/player-api.ts"
  );
  const { readFileSync } = await import("node:fs");
  const catalog = JSON.parse(
    readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"),
  ) as { nodes: { type: string; entrypoint?: boolean }[] };
  const entry = catalog.nodes.filter((n) => n.entrypoint).map((n) => n.type);
  assert.ok(entry.includes("world.npc_plan") && entry.includes("world.npc_chat"));
  for (const t of entry) assert.ok(PLAYER_ENTRYPOINTS.has(t), t);
  const doc = {
    nodes: [
      { id: "world", type: "world.generator" },
      { id: "plan", type: "world.npc_plan" },
      { id: "chat", type: "world.npc_chat" },
      { id: "reply", type: "world.npc_reply" },
    ],
  } as never;
  assert.deepEqual(rerollTargets(doc), ["world"]);
});
