// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const N = await import("../src/features/flows/player/world/needs.ts");
const A = await import("../src/features/flows/player/world/world-actions.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Area = { x: number; y: number; w: number; h: number };

const PLACES = [
  { id: "home", name: "Cottage Row", x: 2, y: 2, w: 4, h: 3 },
  { id: "cafe", name: "Hobbs Cafe", x: 10, y: 2, w: 4, h: 3 },
  { id: "square", name: "Town Square", x: 10, y: 10, w: 5, h: 5 },
  { id: "park", name: "Johnson Park", x: 20, y: 10, w: 6, h: 6 },
  { id: "forge", name: "Forge", x: 20, y: 2, w: 3, h: 3 },
];

function spec(simulation?: Partial<NonNullable<Spec["simulation"]>>): Spec {
  return {
    title: "Smallville",
    seed: 7,
    width: 48,
    height: 36,
    tile_size: 32,
    biome: "town",
    tileset: {},
    player: { name: "Pip" },
    npcs: [
      { id: "isabella", name: "Isabella", persona: "", greeting: "hi" },
      { id: "tom", name: "Tom", persona: "", greeting: "hey" },
    ],
    items: [],
    quests: [],
    regions: [
      { id: "pond", name: "Pond", x: 30, y: 20, w: 4, h: 4, on_enter_text: "" },
    ],
    vars: {},
    intro_text: "",
    simulation: simulation
      ? {
          enabled: true,
          minutes_per_second: 2,
          start_minute: 480,
          places: PLACES,
          chat_cooldown_min: 120,
          calls_per_minute: 4,
          ...simulation,
        }
      : undefined,
  } as Spec;
}

function fakeHandle(npcAt: Record<string, { x: number; y: number }> = {}) {
  const calls: {
    place: Record<string, Area | null>;
    status: Record<string, string>;
    said: string[];
  } = {
    place: {},
    status: {},
    said: [],
  };
  let player = { x: 15, y: 15 };
  return {
    calls,
    movePlayer(p: { x: number; y: number }) {
      player = p;
    },
    playerTile: () => player,
    npcTile: (id: string) => npcAt[id] ?? { x: 0, y: 0 },
    setNpcPlace(id: string, area: Area | null) {
      calls.place[id] = area;
    },
    setNpcStatus(id: string, icon: string) {
      calls.status[id] = icon;
    },
    say(id: string, text: string) {
      calls.said.push(`${id}: ${text}`);
    },
  };
}

test("places advertise actions from their names, with defaults for the missing ones", () => {
  const offers = N.advertisements(PLACES);
  const at = (kind: string) =>
    offers.filter((o) => o.kind === kind).map((o) => o.place);
  assert.deepEqual(at("sleep"), ["home"]);
  assert.deepEqual(at("eat"), ["cafe"]);
  assert.deepEqual(at("socialise"), ["cafe", "square"]);
  assert.deepEqual(at("wander"), ["park"]);
  assert.deepEqual(at("work"), ["forge"]);
  // A village with no named kitchen still lets everyone eat anywhere, at half strength.
  const plain = N.advertisements([
    { id: "a", name: "North", x: 0, y: 0, w: 2, h: 2 },
  ]);
  assert.ok(plain.some((o) => o.kind === "eat" && o.strength === 0.5));
  assert.ok(plain.some((o) => o.kind === "sleep"));
});

test("needs decay with time, restore at the right place, and stay in 0..100", () => {
  let needs = { energy: 50, hunger: 50, social: 50, fun: 50 };
  needs = N.tickNeeds(needs, 60, null, false);
  assert.ok(needs.hunger < 50 && needs.energy < 50);
  const walking = N.tickNeeds({ ...needs }, 30, "eat", false);
  const eating = N.tickNeeds({ ...needs }, 30, "eat", true);
  assert.ok(eating.hunger > needs.hunger && walking.hunger < needs.hunger);
  let n = { energy: 1, hunger: 99, social: 0, fun: 100 };
  for (let i = 0; i < 2000; i++) {
    n = N.tickNeeds(n, 7, i % 2 ? "eat" : null, true);
    for (const k of N.NEEDS)
      assert.ok(n[k] >= 0 && n[k] <= 100, `${k}=${n[k]}`);
  }
  assert.deepEqual(
    N.clampNeeds(
      { energy: 400, hunger: "x", social: -3 },
      { energy: 1, hunger: 2, social: 3, fun: 4 },
    ),
    { energy: 100, hunger: 2, social: 0, fun: 4 },
  );
});

test("a hungry villager goes to eat, a rested fed one follows the day plan, and draws are seeded", () => {
  const offers = N.advertisements(PLACES);
  const block = {
    start: 480,
    minutes: 120,
    activity: "work at the forge",
    place: "forge",
  };
  const pick = (needs: Record<string, number>, seed: number) =>
    N.chooseAction({
      needs: needs as never,
      offers,
      clock: 540,
      block,
      rnd: N.seeded(1, "isabella", seed),
    });
  const hungry = pick({ energy: 90, hunger: 5, social: 90, fun: 90 }, 1);
  assert.equal(hungry.kind, "eat");
  assert.equal(hungry.place, "cafe");
  const content = pick({ energy: 95, hunger: 95, social: 95, fun: 95 }, 1);
  assert.equal(content.place, "forge");
  assert.equal(content.until, 600);
  // The same seed always draws the same action; across seeds the top choices vary.
  const mid = { energy: 60, hunger: 45, social: 40, fun: 45 };
  const draws = () => Array.from({ length: 30 }, (_, i) => pick(mid, i));
  assert.deepEqual(draws(), draws());
  const kinds = new Set(
    Array.from(
      { length: 40 },
      (_, i) => `${pick(mid, i).kind}@${pick(mid, i).place}`,
    ),
  );
  assert.ok(kinds.size > 1, [...kinds].join());
  // Bed time doubles the pull of sleep.
  const late = N.chooseAction({
    needs: { energy: 40, hunger: 80, social: 80, fun: 80 },
    offers,
    clock: 23 * 60,
    block: null,
    rnd: () => 0,
  });
  assert.equal(late.kind, "sleep");
});

test("villagerStep restores the need at the place, then picks the next action through the handle", () => {
  const s = spec({});
  const sim = s.simulation as NonNullable<Spec["simulation"]>;
  const areas = A.placeAreas(s);
  const offers = N.advertisements(PLACES);
  const g = fakeHandle({ isabella: { x: 11, y: 3 } });
  const mind = {
    memory: [],
    nextId: 0,
    plan: [],
    planDay: 0,
    lastChat: {},
    needs: { energy: 80, hunger: 20, social: 80, fun: 80 },
    doing: { kind: "eat" as const, place: "cafe", until: 600 },
  };
  const out = A.villagerStep({
    g,
    spec: s,
    sim,
    npcId: "isabella",
    mind,
    clock: 520,
    offers,
    areas,
  });
  assert.ok(out);
  assert.ok(out.needs.hunger > 20);
  assert.equal(out.doing.kind, "eat");
  assert.deepEqual(g.calls.place.isabella, areas.get("cafe"));
  assert.equal(g.calls.status.isabella, "🍞");
  const next = A.villagerStep({
    g,
    spec: s,
    sim,
    npcId: "isabella",
    mind: { ...mind, needs: { energy: 80, hunger: 99, social: 10, fun: 80 } },
    clock: 520,
    offers,
    areas,
  });
  assert.ok(next && next.doing.kind !== "eat");
  // No needs (a world that turned them off) means the classic plan path handles the NPC.
  assert.equal(
    A.villagerStep({
      g,
      spec: s,
      sim,
      npcId: "isabella",
      mind: { ...mind, needs: undefined },
      clock: 520,
      offers,
      areas,
    }),
    null,
  );
});

test("follow me: a reply makes the NPC walk to the player through the handle, stop sends them home", () => {
  const s = spec();
  const g = fakeHandle();
  const areas = A.placeAreas(s);
  let ws = W.initialWorldState(s);
  ws = W.applyReply(s, ws, "tom", {
    text: "Lead the way.",
    action: "follow_player",
  }).state;
  assert.deepEqual(ws.following, ["tom"]);
  let r = A.steerActions(g, areas, ws, new Set());
  assert.deepEqual(g.calls.place.tom, { x: 14, y: 14, w: 3, h: 3 });
  assert.equal(g.calls.status.tom, "👣");
  g.movePlayer({ x: 30, y: 5 });
  r = A.steerActions(g, areas, ws, r.steered);
  assert.deepEqual(g.calls.place.tom, { x: 29, y: 4, w: 3, h: 3 });

  ws = W.applyReply(s, ws, "tom", {
    text: "I'll wait at the pond.",
    action: "go_to",
    place: "pond",
  }).state;
  assert.deepEqual(ws.following, []);
  assert.deepEqual(ws.errands, { tom: { place: "pond", until: null } });
  r = A.steerActions(g, areas, ws, r.steered);
  assert.deepEqual(g.calls.place.tom, areas.get("pond"));

  ws = W.applyReply(s, ws, "tom", {
    text: "Bye.",
    action: "stop_following",
  }).state;
  r = A.steerActions(g, areas, ws, r.steered);
  assert.equal(g.calls.place.tom, null);
  assert.equal(g.calls.status.tom, "");
  assert.equal(r.steered.size, 0);

  const same = W.applyReply(s, ws, "tom", {
    text: "x",
    action: "dance" as never,
    place: "pond",
  }).state;
  assert.deepEqual(same.errands, ws.errands);
  assert.deepEqual(same.following, ws.following);
  const nowhere = W.applyReply(s, ws, "tom", {
    text: "x",
    action: "go_to",
    place: "moon",
  }).state;
  assert.deepEqual(nowhere.errands, ws.errands);
});

test("a villager who stays put takes no movement action", () => {
  const s = spec();
  s.npcs[1] = { ...s.npcs[1], wander: false };
  const ws = W.applyReply(s, W.initialWorldState(s), "tom", {
    text: "Sure.",
    action: "follow_player",
  }).state;
  assert.deepEqual(ws.following ?? [], []);
});

test("needs keep dropping while a villager follows or runs an errand", async () => {
  const mind = { needs: { energy: 80, hunger: 80, social: 80, fun: 80 } };
  const after = A.steeredNeeds(mind as never, 60) as Record<string, number>;
  assert.ok(after.hunger < 80 && after.energy < 80);
  assert.equal(A.steeredNeeds({} as never, 60), null);
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL("../src/features/flows/player/world/world-player.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /if \(steered\.current\.has\(n\.id\)[^{]*\) \{[\s\S]{0,200}steeredNeeds\(mind, sim\.minutes_per_second\)/);
});

test("errands end with the game clock in simulated worlds", () => {
  const s = spec({});
  const g = fakeHandle();
  const areas = A.placeAreas(s);
  let ws = W.initialWorldState(s);
  ws = W.applyAction(s, ws, "tom", "go_to", "park");
  assert.deepEqual(ws.errands?.tom, { place: "park", until: 600 });
  assert.deepEqual(A.steerActions(g, areas, ws, new Set()).expired, []);
  assert.deepEqual(
    A.steerActions(g, areas, { ...ws, clock: 600 }, new Set()).expired,
    ["tom"],
  );
});

test("saves keep needs, actions, followers and errands; old saves load with fresh needs", () => {
  const s = spec({});
  let ws = W.initialWorldState(s);
  assert.ok(ws.npc.isabella.needs && ws.npc.tom.needs);
  assert.notDeepEqual(ws.npc.isabella.needs, ws.npc.tom.needs);
  ws = W.withMind(ws, "isabella", (m) => ({
    ...m,
    needs: { energy: 12, hunger: 34, social: 56, fun: 78 },
    doing: { kind: "eat", place: "cafe", until: 700 },
  }));
  ws = W.applyAction(s, ws, "tom", "follow_player", undefined);
  const back = W.deserialize(s, W.serialize(ws));
  assert.ok(back);
  assert.deepEqual(back.npc.isabella.needs, {
    energy: 12,
    hunger: 34,
    social: 56,
    fun: 78,
  });
  assert.deepEqual(back.npc.isabella.doing, {
    kind: "eat",
    place: "cafe",
    until: 700,
  });
  assert.deepEqual(back.following, ["tom"]);

  const old = {
    v: 2,
    inventory: [],
    picked: [],
    activeQuests: [],
    completedQuests: [],
    vars: {},
    talks: {},
    clock: 500,
    npc: { isabella: { memory: [], plan: [] } },
  };
  const up = W.deserialize(s, old);
  assert.ok(up);
  assert.deepEqual(up.following, []);
  assert.deepEqual(up.errands, {});
  assert.deepEqual(
    up.npc.isabella.needs,
    W.initialWorldState(s).npc.isabella.needs,
  );
  assert.equal(up.npc.isabella.doing, undefined);
  const junk = W.deserialize(s, {
    ...old,
    following: ["ghost", "tom"],
    errands: { tom: { place: "moon" } },
    npc: { isabella: { needs: { energy: 900 }, doing: { kind: "teleport" } } },
  });
  assert.ok(junk);
  assert.deepEqual(junk.following, ["tom"]);
  assert.deepEqual(junk.errands, {});
  assert.equal(junk.npc.isabella.needs?.energy, 100);
  assert.equal(junk.npc.isabella.doing, undefined);
  // Worlds that turn needs off, and classic worlds, carry none.
  assert.equal(
    W.initialWorldState(spec({ needs: false })).npc.isabella.needs,
    undefined,
  );
  assert.deepEqual(W.initialWorldState(spec()).npc, {});
});
