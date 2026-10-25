// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const A = await import("../src/features/flows/player/world/autonomy.ts");
const R = await import("../src/features/flows/player/world/roles.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
const P = await import("../src/features/flows/player/world/procgen.ts");
const X = await import("../src/features/flows/player/world/world-actions.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type State = import("../src/features/flows/player/world/world-state.ts").WorldState;
type Point = { x: number; y: number };

const BASE = {
  title: "Lantern Village",
  seed: 11,
  width: 40,
  height: 30,
  tile_size: 32,
  biome: "meadow",
  tileset: {},
  player: { name: "Pip" },
  npcs: [
    { id: "tom", name: "Tom the Baker", persona: "bakes bread", greeting: "hi" },
    { id: "ada", name: "Ada", persona: "a carpenter", greeting: "hi" },
    { id: "bo", name: "Bo", persona: "a stone mason", greeting: "hi" },
    { id: "grim", name: "Grim", persona: "a sour drifter", greeting: "what" },
  ],
  items: [{ id: "rod", name: "Fishing Rod", description: "" }],
  quests: [
    {
      id: "lost_rod",
      title: "The Lost Rod",
      description: "",
      giver: "tom",
      requires_item: "rod",
      reward_text: "Thanks",
    },
  ],
  regions: [
    { id: "cottage", name: "Rose Cottage", x: 2, y: 2, w: 4, h: 4, on_enter_text: "" },
    { id: "square", name: "Town Square", x: 28, y: 20, w: 6, h: 5, on_enter_text: "" },
  ],
  vars: {},
  intro_text: "",
} as unknown as Spec;

const ALL = { trade: true, homes: true, quests: true, rivals: true, enemies: true };
const spec = (autonomy: object | undefined = ALL) =>
  ({ ...BASE, autonomy }) as Spec;

function ground(s: Spec) {
  const map = P.generateMap(s);
  return {
    spec: s,
    map,
    areas: X.placeAreas(s),
    items: { ...map.items },
    homes: { ...map.npcs },
  };
}

function crowd(g: ReturnType<typeof ground>): { player: Point; npcs: Record<string, Point> } {
  const spot = P.reachable(g.map, g.map.spawn);
  const open: Point[] = [];
  for (let y = 10; y < 20 && open.length < 4; y++)
    for (let x = 10; x < 30 && open.length < 4; x++)
      if (spot[y * g.map.width + x] && !P.isSolid(g.map, x, y)) open.push({ x, y });
  return {
    player: { x: 0, y: 0 },
    npcs: { tom: open[0], ada: open[1], bo: open[2], grim: open[3] },
  };
}

function play(s: Spec, beats: number, start?: State, seen?: ReturnType<typeof crowd>) {
  const g = ground(s);
  const at = seen ?? crowd(g);
  let st = start ?? W.initialWorldState(s);
  const notes: string[] = [];
  const coins = A.totals(st.auto!).coins;
  for (let i = 0; i < beats; i++) {
    const out = A.autoStep(g, st, at);
    st = out.state;
    notes.push(...out.notes);
    assert.equal(A.totals(st.auto!).coins, coins);
    for (const stock of Object.values(st.auto!.goods))
      for (const n of Object.values(stock)) assert.ok((n ?? 0) > 0);
    for (const c of Object.values(st.auto!.coins)) assert.ok(c >= 0);
  }
  return { st, notes };
}

test("autonomy is off unless the world opts in, and old saves load as before", () => {
  const plain = BASE;
  assert.equal(R.autonomyOn(plain), false);
  const s = W.initialWorldState(plain);
  assert.equal(s.auto, undefined);
  const back = W.deserialize(plain, JSON.parse(JSON.stringify(W.serialize(s))));
  assert.equal(back?.auto, undefined);
  assert.equal(back?.npcAt, undefined);
  assert.deepEqual(R.hostilesOf(plain), []);
  assert.equal(R.rivalFor(plain, plain.quests[0]), null);
  const on = W.deserialize(spec(), JSON.parse(JSON.stringify(W.serialize(s))));
  assert.equal(on?.auto?.coins.player, 15);
});

test("villagers make goods, shop from each other and claim or build homes, with coins conserved", () => {
  const { st, notes } = play(spec({ trade: true, homes: true }), 400);
  assert.ok(notes.some((n) => / bought .* from /.test(n)), notes.join("\n"));
  assert.ok(notes.includes("Tom the Baker moved into Rose Cottage") || notes.some((n) => /moved into Rose Cottage/.test(n)));
  assert.ok(notes.some((n) => /built a home/.test(n)), notes.join("\n"));
  const built = Object.values(st.auto!.homes).find((h) => h.at);
  assert.ok(built?.at);
  const g = ground(spec());
  const reach = P.reachable(g.map, g.map.spawn);
  assert.equal(reach[built.at.y * g.map.width + built.at.x], 1);
});

test("a run replays the same, and a save and load in the middle changes nothing", () => {
  const s = spec({ trade: true, homes: true, quests: true });
  const a = play(s, 300);
  const b = play(s, 300);
  assert.deepEqual(a.st.auto, b.st.auto);
  assert.deepEqual(a.notes, b.notes);
  const half = play(s, 150);
  assert.ok(half.notes.some((n) => / bought /.test(n)), half.notes.join("\n"));
  const saved = W.deserialize(s, JSON.parse(JSON.stringify(W.serialize(half.st))));
  assert.ok(saved);
  const rest = play(s, 150, saved);
  assert.deepEqual(rest.st.auto, a.st.auto);
});

test("a rival races for an open quest's item; the first to reach it keeps it, and the player can buy it back", () => {
  const s = spec({ quests: true });
  const g = ground(s);
  const rival = R.rivalFor(s, s.quests[0]);
  assert.ok(rival && rival !== "tom");
  let st: State = { ...W.initialWorldState(s), activeQuests: ["lost_rod"] };
  const seen = crowd(g);
  for (let i = 0; i < 20; i++) st = A.autoStep(g, st, seen).state;
  assert.equal(st.auto!.held.rod, undefined);
  let out = A.autoStep(g, st, seen);
  for (let i = 0; i < 10; i++) {
    out = A.autoStep(g, out.state, seen);
  }
  assert.ok(out.cmds.some((c) => c.do === "place" && c.id === rival));
  const racing = out.state;
  out = A.autoStep(g, out.state, { ...seen, npcs: { ...seen.npcs, [rival]: g.items.rod } });
  assert.equal(out.state.auto!.held.rod, rival);
  assert.ok(out.cmds.some((c) => c.do === "remove" && c.id === "rod"));
  assert.deepEqual(W.pickUp(out.state, "rod").inventory, []);
  const poor = { ...out.state, auto: { ...out.state.auto!, coins: { ...out.state.auto!.coins, player: 2 } } };
  const total = A.totals(poor.auto).coins;
  const back = A.buyBack(s, poor, rival, "rod");
  assert.ok(back);
  assert.deepEqual(back.state.inventory, ["rod"]);
  assert.equal(back.state.auto!.held.rod, undefined);
  assert.equal(back.state.auto!.coins.player, 0);
  assert.equal(A.totals(back.state.auto!).coins, total);
  const mine = W.pickUp(racing, "rod");
  const after = A.autoStep(g, mine, { ...seen, npcs: { ...seen.npcs, [rival]: g.items.rod } });
  assert.equal(after.state.auto!.held.rod, undefined);
  assert.deepEqual(after.state.inventory, ["rod"]);
});

test("a hostile villager chases only nearby and never into a safe place; the encounter is non-lethal", () => {
  const s = spec({ enemies: true });
  const g = ground(s);
  const [enemy] = R.hostilesOf(s);
  assert.ok(enemy && enemy !== "tom");
  const home = g.homes[enemy];
  const st: State = { ...W.initialWorldState(s), inventory: ["rod"] };
  const near = { x: home.x + 2, y: home.y };
  const seen = (player: Point, me: Point) => ({ player, npcs: { ...g.homes, [enemy]: me } });
  let out = A.autoStep(g, st, seen({ x: home.x + 5, y: home.y }, home));
  assert.ok(out.cmds.some((c) => c.do === "status" && c.id === enemy && c.icon === "💢"));
  for (const directed of [
    { ...st, following: [enemy] },
    { ...st, errands: { [enemy]: { place: "square", until: null } } },
  ]) {
    const mine = A.autoStep(g, directed, seen({ x: home.x + 5, y: home.y }, home));
    assert.ok(!mine.cmds.some((c) => (c.do === "place" || c.do === "status") && c.id === enemy));
    assert.ok(!mine.steered.includes(enemy));
  }
  out = A.autoStep(g, st, seen({ x: home.x + 30, y: home.y }, { x: home.x + 25, y: home.y }));
  assert.ok(out.cmds.some((c) => c.do === "status" && c.id === enemy && c.icon === "😠"));
  assert.ok(!out.cmds.some((c) => c.do === "status" && c.id === enemy && c.icon === "💢"));
  const square = { x: 30, y: 22 };
  const bySquare = { ...g, homes: { ...g.homes, [enemy]: { x: 30, y: 18 } } };
  out = A.autoStep(bySquare, st, { player: square, npcs: { ...g.homes, [enemy]: { x: 29, y: 22 } } });
  assert.equal(out.state.auto!.encounter, null);
  out = A.autoStep(bySquare, st, { player: { x: 30, y: 19 }, npcs: { ...g.homes, [enemy]: { x: 29, y: 19 } } });
  assert.equal(out.state.auto!.encounter, enemy);
  out = A.autoStep(g, st, seen(near, home));
  assert.equal(out.state.auto!.encounter, enemy);
  const total = A.totals(out.state.auto!).coins;
  const paid = A.resolveEncounter(s, out.state, "negotiate");
  assert.equal(paid.state.auto!.encounter, null);
  assert.equal(paid.state.auto!.coins.player, 12);
  let lost = out.state;
  for (let beat = 0; beat < 200; beat++) {
    const tryEvade = A.resolveEncounter(s, { ...out.state, auto: { ...out.state.auto!, beat } }, "evade");
    if (/took/.test(tryEvade.note)) {
      lost = tryEvade.state;
      break;
    }
  }
  assert.ok(lost.auto!.coins.player < 15);
  assert.deepEqual(lost.inventory, ["rod"]);
  assert.equal(A.totals(lost.auto!).coins, total);
  let won = null;
  for (let beat = 0; beat < 200 && !won; beat++) {
    const r = A.resolveEncounter(s, { ...lost, auto: { ...lost.auto!, beat, encounter: enemy } }, "defend");
    if (/got your things back/.test(r.note)) won = r.state;
  }
  assert.ok(won);
  assert.equal(won.auto!.coins.player, 15);
  const again = A.autoStep(g, { ...out.state, auto: { ...paid.state.auto! } }, seen(near, home));
  assert.equal(again.state.auto!.encounter, null);
});

test("the player buys and sells at bounded prices, and cannot spend coins they do not have", () => {
  const s = spec({ trade: true });
  const st = W.initialWorldState(s);
  const bought = A.playerBuy(s, st, "tom", "bread");
  assert.ok(bought);
  assert.equal(bought.state.auto!.goods.player.bread, 1);
  assert.equal(bought.state.auto!.goods.tom.bread, 2);
  assert.equal(A.totals(bought.state.auto!).coins, A.totals(st.auto!).coins);
  const sold = A.playerSell(s, bought.state, "ada", "bread");
  assert.ok(sold);
  assert.equal(sold.state.auto!.goods.player.bread, undefined);
  const broke = { ...st, auto: { ...st.auto!, coins: { ...st.auto!.coins, player: 0 } } };
  assert.equal(A.playerBuy(s, broke, "tom", "bread"), null);
  assert.equal(A.playerBuy(s, st, "tom", "tools"), null);
  for (let stock = 0; stock < 10; stock++)
    for (const g of A.GOOD_IDS) {
      const p = A.priceOf(g, stock);
      assert.ok(p >= Math.ceil(A.GOODS[g].price / 2) && p <= A.GOODS[g].price * 2);
    }
});

test("an item a villager holds is only theirs to give", () => {
  const s = spec({ quests: true });
  const st: State = {
    ...W.initialWorldState(s),
    picked: ["rod"],
    auto: { ...A.initialAuto(s), held: { rod: "ada" } },
  };
  const other = W.applyReply(s, st, "bo", { text: "here", give_item: "rod" });
  assert.deepEqual(other.state.inventory, []);
  const holder = W.applyReply(s, st, "ada", { text: "fine", give_item: "rod" });
  assert.deepEqual(holder.state.inventory, ["rod"]);
  assert.equal(holder.state.auto!.held.rod, undefined);
});

test("the player drives autonomy through the game handle and pauses it while talking", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/world-player.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /if \(!g \|\| lockedRef\.current\) return;\s*const played = specRef\.current;/);
  assert.match(src, /run\(g, out\.cmds\);/);
  assert.match(src, /npcAt: game\.current\.npcTiles\(\)/);
  assert.match(src, /steered\.current\.has\(n\.id\) \|\| autoSteered\.current\.has\(n\.id\)/);
  const game = readFileSync(
    new URL("../src/features/flows/player/world/phaser-game.ts", import.meta.url),
    "utf8",
  );
  const wander = game.slice(game.indexOf("n.rest -= dt"), game.indexOf("this.animate(\n          n,"));
  assert.ok(wander.length > 100);
  assert.doesNotMatch(wander, /Math\.random/);
  assert.match(game, /new Prng\(seedFrom\(spec\.seed, "wander", id\)\)/);
  assert.match(game, /if \(sceneRef\) drawMarker\(sceneRef, id, at, icon, label\);\s*else pendingMarkers\.set\(id, \[at, icon, label\]\);/);
  assert.match(game, /for \(const \[id, m\] of pendingMarkers\) drawMarker\(this, id, \.\.\.m\);/);
});

test("the director's new villager keeps a rival or hostile role only when the world allows it", async () => {
  const D = await import("../src/features/flows/player/world/world-director.ts");
  const s = spec({ enemies: true });
  const g = ground(s);
  const done = { ...W.initialWorldState(s), completedQuests: ["lost_rod"] };
  const patch = {
    quest: { id: "q2", title: "Q", description: "", giver: "kit", requires_item: "rod2", reward_text: "" },
    npcs: [{ id: "kit", name: "Kit", persona: "", greeting: "", role: "hostile" as const }],
    items: [{ id: "rod2", name: "Rod 2", description: "" }],
  };
  const occupied = [...Object.values(g.map.npcs), ...Object.values(g.map.items)];
  const ok = D.applyPatch(s, g.map, done, patch as never, occupied);
  assert.equal(ok?.npcs[0].role, "hostile");
  assert.equal(D.directorRequest(s, done, []).allow_hostile, true);
  assert.equal(D.directorRequest(s, done, []).allow_rival, false);
  const asRival = D.applyPatch(s, g.map, done, { ...patch, npcs: [{ ...patch.npcs[0], role: "rival" }] } as never, occupied);
  assert.equal(asRival?.npcs[0].role, undefined);
  const plain = D.applyPatch(BASE, g.map, { ...W.initialWorldState(BASE), completedQuests: ["lost_rod"] }, patch as never, occupied);
  assert.equal(plain?.npcs[0].role, undefined);
  const played = D.withAdditions(s, ok!.state.additions);
  assert.deepEqual(R.hostilesOf(played), ["kit"]);
});
