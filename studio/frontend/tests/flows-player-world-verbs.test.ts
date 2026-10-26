// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const A = await import("../src/features/flows/player/world/action3d.ts");
const V = await import("../src/features/flows/player/world/action-verbs.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
type Prop = import("../src/features/flows/types.ts").WorldProp3d;
type Spec = import("../src/features/flows/types.ts").WorldSpec;

const flat = () => 0;
const edge = (x: number, y: number) => x < 0 || y < 0 || x >= 20 || y >= 20;
const world = (props: Prop[] = [], solid = edge) =>
  A.buildActionWorld(20, 20, flat, solid, props);
const prop = (
  id: string,
  kind: Prop["kind"],
  x: number,
  y: number,
  extra: Partial<Prop> = {},
): Prop => ({
  id,
  kind,
  x,
  y,
  rot: 0,
  scale: 1,
  solid: kind !== "lever",
  ...extra,
});
const gate = prop("gate", "door", 5.5, 5.5, { rot: 90, locked_by: "key" });
const shed = prop("shed", "door", 9.5, 5.5);
const box = prop("box", "chest", 3.5, 8.5, { contents: ["key", "map", "key"] });
const pull = prop("pull", "lever", 7.5, 8.5, {
  targets: ["gate", "shed", "box"],
});
const PROPS = [gate, shed, box, pull];
const names = {
  name: (id: string) => ({ key: "Brass key", map: "Old map" })[id] ?? id,
};

test("a locked door needs its key, then stays unlocked; it opens and closes", () => {
  const no = V.interactProp(gate, PROPS, {}, [], names);
  assert.equal(no.text, "Locked. It needs the Brass key");
  assert.deepEqual(no.changed, []);
  assert.equal(V.isOpen(gate, no.states), false);
  const yes = V.interactProp(gate, PROPS, {}, ["key"], names);
  assert.equal(yes.text, "Unlocked with the Brass key");
  assert.deepEqual(yes.states.gate, { unlocked: true, open: true });
  assert.ok(yes.loud);
  const shut = V.interactProp(gate, PROPS, yes.states, [], names);
  assert.equal(shut.text, "The door closes");
  const again = V.interactProp(gate, PROPS, shut.states, [], names);
  assert.equal(again.text, "The door opens", "the key is not needed twice");
});

test("a door does not close on someone standing in it", () => {
  const open = { shed: { open: true } };
  const out = V.interactProp(shed, PROPS, open, [], { occupied: () => true });
  assert.equal(out.text, "Something is in the way");
  assert.equal(V.isOpen(shed, out.states), true);
  assert.equal(
    V.isOpen({ ...shed, open: true }, {}),
    true,
    "the spec's open flag is the start state",
  );
});

test("a chest gives its contents once", () => {
  assert.deepEqual(V.heldInChests(PROPS, {}), ["key", "map", "key"]);
  const first = V.interactProp(box, PROPS, {}, ["map"], names);
  assert.deepEqual(first.gained, ["key"]);
  assert.equal(first.text, "Found Brass key");
  assert.deepEqual(first.states.box, { looted: true, open: true });
  assert.deepEqual(V.heldInChests(PROPS, first.states), []);
  const second = V.interactProp(box, PROPS, first.states, [], names);
  assert.deepEqual(second.gained, []);
  assert.equal(second.text, "The chest is empty");
});

test("a lever toggles its linked doors, locked ones too, and nothing else", () => {
  const out = V.interactProp(pull, PROPS, {}, []);
  assert.deepEqual(out.changed, ["pull", "gate", "shed"]);
  assert.equal(V.isOpen(gate, out.states), true);
  assert.equal(out.states.gate.unlocked, true);
  assert.equal(V.isOpen(shed, out.states), true);
  assert.equal(out.states.box, undefined);
  const back = V.interactProp(pull, PROPS, out.states, []);
  assert.equal(V.isOpen(gate, back.states), false);
  assert.equal(V.isOpen(pull, back.states), false);
  const stuck = V.interactProp(pull, PROPS, out.states, [], {
    occupied: (p) => p.id === "shed",
  });
  assert.equal(V.isOpen(shed, stuck.states), true);
  assert.match(stuck.text, /blocks a door/);
});

test("a closed door blocks the capsule and an open one lets it through", () => {
  const door = prop("d", "door", 5.5, 5.5, { rot: 90 });
  const east = { dx: 1, dy: 0, sprint: false, crouch: false, jump: false };
  const walk = (st: Record<string, { open?: boolean }>) => {
    const w = world(V.closedProps([door], st));
    const b = A.newBody(w, 3.5, 5.5);
    for (let i = 0; i < 120; i++) A.advance(w, b, east, 1 / 60, 3);
    return b.x;
  };
  assert.ok(walk({}) < 5.5, `${walk({})}`);
  assert.ok(walk({ d: { open: true } }) > 7, `${walk({ d: { open: true } })}`);
});

test("door, chest and lever states survive a save and old saves load without them", () => {
  const spec: Spec = {
    title: "Keep",
    seed: 1,
    width: 20,
    height: 20,
    tile_size: 32,
    biome: "town",
    tileset: {},
    player: { name: "Pip" },
    npcs: [],
    items: [
      { id: "key", name: "Brass key", description: "" },
      { id: "map", name: "Old map", description: "" },
    ],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    props3d: PROPS,
  };
  let s = W.initialWorldState(spec);
  s = W.applyUse(s, V.interactProp(box, PROPS, s.props ?? {}, s.inventory));
  s = W.applyUse(s, V.interactProp(gate, PROPS, s.props ?? {}, s.inventory));
  assert.deepEqual(s.inventory, ["key", "map"]);
  assert.deepEqual(s.picked, ["key", "map"]);
  const raw = JSON.parse(JSON.stringify(W.serialize(s)));
  raw.props.ghost = { open: true };
  raw.props.gate.open = "yes";
  const back = W.deserialize(spec, raw);
  assert.ok(back);
  assert.deepEqual(back.props, {
    box: { looted: true, open: true },
    gate: { unlocked: true },
  });
  assert.deepEqual(back.inventory, ["key", "map"]);
  const old = W.deserialize(spec, {
    ...W.serialize(W.initialWorldState(spec)),
  });
  assert.ok(old && !("props" in old));
});

test("after a reload a looted chest gives nothing and a pulled lever is not applied twice", () => {
  const spec = {
    title: "Keep",
    seed: 1,
    width: 20,
    height: 20,
    tile_size: 32,
    biome: "town",
    tileset: {},
    player: { name: "Pip" },
    npcs: [],
    items: [
      { id: "key", name: "Brass key", description: "" },
      { id: "map", name: "Old map", description: "" },
    ],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    props3d: PROPS,
  } as Spec;
  const reload = (st: ReturnType<typeof W.initialWorldState>) => {
    const back = W.deserialize(
      spec,
      JSON.parse(JSON.stringify(W.serialize(st))),
    );
    assert.ok(back);
    return back;
  };
  let s = W.initialWorldState(spec);
  s = W.applyUse(s, V.interactProp(box, PROPS, s.props ?? {}, s.inventory));
  s = W.applyUse(s, V.interactProp(pull, PROPS, s.props ?? {}, s.inventory));
  s = { ...s, inventory: s.inventory.filter((i) => i !== "key") };
  s = reload(s);
  assert.equal(V.isOpen(gate, s.props ?? {}), true);
  assert.equal(V.isOpen(pull, s.props ?? {}), true);
  const again = V.interactProp(box, PROPS, s.props ?? {}, s.inventory);
  assert.deepEqual(again.gained, []);
  assert.equal(again.text, "The chest is empty");
  s = W.applyUse(s, again);
  assert.deepEqual(s.inventory, ["map"]);
  s = reload(
    W.applyUse(s, V.interactProp(pull, PROPS, s.props ?? {}, s.inventory)),
  );
  assert.equal(
    V.isOpen(gate, s.props ?? {}),
    false,
    "one pull after the reload, one toggle",
  );
  assert.equal(V.isOpen(pull, s.props ?? {}), false);
  assert.equal(s.props?.gate.unlocked, true);
});

const still = { dx: 0, dy: 0, sprint: false, crouch: false, jump: false };

function dashAt(fps: number, w = world()) {
  const b = A.newBody(w, 2.5, 5.5);
  const v = V.newVerbs();
  assert.ok(V.startDash(v, b, 1, 0));
  let frames = 0;
  while (V.dashing(v) && frames < 200) {
    V.verbsTick(v, 1 / fps);
    V.dashAdvance(w, b, v, 1 / fps);
    frames++;
  }
  for (let i = 0; i < fps; i++) A.advance(w, b, still, 1 / fps, 3);
  return { b, v, x: b.x };
}

test("a dash covers the same distance at 30, 60 and 144 frames a second", () => {
  const ends = [30, 60, 144].map((fps) => dashAt(fps).x - 2.5);
  for (const d of ends)
    assert.ok(Math.abs(d - V.DASH_DISTANCE) < 1e-6, `${ends}`);
});

test("a dash into a wall still ends on time, and dashing gives no invulnerability", () => {
  const w = world([], (x, y) => edge(x, y) || x === 4);
  const b = A.newBody(w, 3.5, 5.5);
  const v = V.newVerbs();
  const c = A.newCombat();
  assert.ok(V.startDash(v, b, 1, 0));
  let frames = 0;
  let hurtWhileDashing = false;
  while (V.dashing(v) && frames < 100) {
    V.verbsTick(v, 1 / 60);
    A.tick(c, 1 / 60);
    V.dashAdvance(w, b, v, 1 / 60);
    const before = c.health;
    A.enemiesAct(
      w,
      c,
      { x: b.x, y: b.y, z: b.z },
      [{ id: "h", x: 2.5, y: 5.5, z: 0, hostile: true }],
      () => false,
    );
    hurtWhileDashing ||= c.health < before;
    frames++;
  }
  assert.ok(frames <= Math.ceil(V.DASH_S * 60) + 1, `${frames} frames`);
  assert.ok(b.x < 4 - A.RADIUS + 1e-6);
  assert.ok(hurtWhileDashing, "a dash must not dodge damage");
  assert.equal(V.startDash(v, b, -1, 0), false, "still cooling down");
});

test("a dash cools down, needs a direction, is refused while crouched and stops at a thin prop", () => {
  const w = world();
  const b = A.newBody(w, 2.5, 5.5);
  const v = V.newVerbs();
  assert.equal(V.startDash(v, b, 0, 0), false);
  assert.ok(V.startDash(v, b, 0, 1));
  assert.equal(V.startDash(v, b, 0, 1), false, "cooling down");
  V.verbsTick(v, V.DASH_S + V.DASH_COOLDOWN_S);
  b.crouched = true;
  assert.equal(V.startDash(v, b, 0, 1), false);
  const lamp = prop("lamp", "lamp", 3.6, 5.5, { solid: true });
  const r = dashAt(30, world([lamp]));
  assert.ok(r.x < 3.6 - 0.05 - A.RADIUS + 1e-6, `tunnelled to ${r.x}`);
});

const hostile = (id: string, x: number, y: number, h = true) => ({
  id,
  x,
  y,
  z: 0,
  hostile: h,
});

test("melee hits the nearest hostile in a short box ahead, pushes it back and cools down", () => {
  const w = world();
  const c = A.newCombat();
  const v = V.newVerbs();
  const eye = { x: 5, y: 5, z: 1 };
  const targets = [
    hostile("far", 7, 5),
    hostile("near", 5.9, 5.2),
    hostile("behind", 4.2, 5),
    hostile("kind", 5.5, 5, false),
  ];
  const hit = V.melee(w, c, v, eye, 1, 0, targets);
  assert.ok(hit);
  assert.equal(hit.hit, "near");
  assert.equal(c.enemies.near.health, A.ENEMY_HEALTH - V.MELEE_DAMAGE);
  assert.ok(hit.kx > 0 && hit.ky === 0);
  assert.equal(V.melee(w, c, v, eye, 1, 0, targets), null, "cooling down");
  V.verbsTick(v, V.MELEE_COOLDOWN_S);
  const again = V.melee(w, c, v, eye, 1, 0, targets);
  assert.equal(again?.downed, true);
  assert.ok(c.enemies.near.down > c.clock);
  V.verbsTick(v, V.MELEE_COOLDOWN_S);
  assert.equal(
    V.melee(w, c, v, eye, -1, 0, [hostile("side", 5, 6.2)])?.hit,
    null,
  );
});

test("melee does not reach through a wall", () => {
  const w = world([], (x, y) => edge(x, y) || x === 5);
  const c = A.newCombat();
  const v = V.newVerbs();
  const out = V.melee(w, c, v, { x: 4.7, y: 5.5, z: 1 }, 1, 0, [
    hostile("h", 5.6, 5.5),
  ]);
  assert.equal(out?.hit, null);
  assert.equal(c.enemies.h, undefined);
});

test("melee only reaches a hostile at the player's own height", () => {
  const w = world();
  const c = A.newCombat();
  const v = V.newVerbs();
  assert.equal(
    V.melee(w, c, v, { x: 5, y: 5, z: 10 }, 1, 0, [hostile("below", 6, 5)])
      ?.hit,
    null,
  );
  assert.equal(c.enemies.below, undefined);
  V.verbsTick(v, V.MELEE_COOLDOWN_S);
  assert.equal(
    V.melee(w, c, v, { x: 5, y: 5, z: 1 }, 1, 0, [hostile("level", 6, 5)])?.hit,
    "level",
  );
});

function fly(
  w: ReturnType<typeof world>,
  dir = { x: 1, y: 0, z: 0 },
  hits = (_x: number, _y: number, _z: number) => false,
) {
  const c = A.newCombat();
  const v = V.newVerbs();
  const eye = { x: 2, y: 5.5, z: 1.5 };
  assert.ok(V.throwRock(c, v, eye, dir));
  assert.equal(c.ammo, A.START_AMMO - V.THROW_AMMO);
  const trail: { x: number; z: number; vz: number }[] = [];
  const bursts: { x: number; y: number; z: number; t: number }[] = [];
  for (let i = 0; i < 300 && v.live; i++) {
    V.verbsTick(v, 1 / 60);
    V.stepProjectiles(w, v, 1 / 60, hits, (p) =>
      bursts.push({ x: p.x, y: p.y, z: p.z, t: v.clock }),
    );
    trail.push({ x: v.shots[0].x, z: v.shots[0].z, vz: v.shots[0].vz });
  }
  return { c, v, trail, bursts };
}

test("a thrown rock follows a ballistic arc, bounces, and bursts on its fuse with a loud noise", () => {
  const { v, trail, bursts } = fly(world());
  const r = V.PROJECTILE_R;
  const z0 = 1.5 - 0.15;
  // First touch of the ground: z0 + lift t - g t^2 / 2 = r.
  const t =
    (V.THROW_LIFT + Math.sqrt(V.THROW_LIFT ** 2 + 2 * 16 * (z0 - r))) / 16;
  const land = trail.findIndex(
    (p, i) => i > 0 && trail[i - 1].vz < -2 && p.vz > 0,
  );
  assert.ok(
    Math.abs(trail[land].x - (2.3 + V.THROW_SPEED * t)) < 0.2,
    `${trail[land].x}`,
  );
  const apex = Math.max(...trail.map((p) => p.z));
  assert.ok(Math.abs(apex - (z0 + V.THROW_LIFT ** 2 / 32)) < 0.03, `${apex}`);
  assert.equal(bursts.length, 1);
  assert.ok(Math.abs(bursts[0].t - V.THROW_FUSE_S) < 0.02);
  assert.equal(v.noiseSeq, 1);
  assert.equal(v.noises[0].radius, V.NOISE.burst);
  assert.ok(Math.abs(v.noises[0].x - bursts[0].x) < 1e-9);
});

test("a rock bounces back off a wall instead of passing it", () => {
  const { trail } = fly(world([], (x, y) => edge(x, y) || x === 4));
  assert.ok(Math.max(...trail.map((p) => p.x)) < 4, "went through the wall");
});

test("a rock bursts early on a hostile; the splash falls off with distance and stops at walls", () => {
  const hit = fly(world(), { x: 1, y: 0, z: 0 }, (x) => x > 3);
  assert.ok(hit.bursts[0].t < 0.3);
  const w = world([], (x, y) => edge(x, y) || (x === 12 && y === 10));
  const c = A.newCombat();
  const at = { x: 10.5, y: 10.5, z: 0.1 };
  const out = V.splash(w, c, at, [
    hostile("close", 10.9, 10.5),
    hostile("edge", 12.3, 11.5),
    hostile("walled", 12.6, 10.5),
    hostile("out", 14, 10.5),
    hostile("kind", 10.6, 10.5, false),
  ]);
  assert.deepEqual(
    out.map((h) => h.id),
    ["close", "edge"],
  );
  assert.ok(c.enemies.close.health < c.enemies.edge.health);
  assert.equal(c.enemies.walled, undefined);
});

test("throwing needs ammo and a free rock; at most four fly at once", () => {
  const c = A.newCombat();
  const v = V.newVerbs();
  const eye = { x: 2, y: 2, z: 1.5 };
  const up = { x: 0, y: 0, z: 1 };
  let n = 0;
  for (let i = 0; i < 6; i++) {
    if (V.throwRock(c, v, eye, up)) n++;
    V.verbsTick(v, V.THROW_COOLDOWN_S);
  }
  assert.equal(n, 4);
  c.ammo = V.THROW_AMMO - 1;
  v.shots[0].live = false;
  v.live--;
  assert.equal(V.throwRock(c, v, eye, up), false);
});

test("the noise ring is bounded and keeps the newest", () => {
  const v = V.newVerbs();
  for (let i = 0; i < 40; i++) V.makeNoise(v, i, 0, 3);
  assert.equal(v.noises.length, 16);
  assert.equal(Math.max(...v.noises.map((n) => n.seq)), 40);
  assert.equal(Math.min(...v.noises.map((n) => n.seq)), 25);
});
