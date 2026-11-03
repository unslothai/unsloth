// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const A = await import("../src/features/flows/player/world/action3d.ts");
type Prop = import("../src/features/flows/types.ts").WorldProp3d;

const flat = () => 0;
const wallAt = (x: number, y: number) =>
  x < 0 || y < 0 || x >= 20 || y >= 20 || (x === 10 && y !== 18);
const world = (props: Prop[] = [], solid = (x: number, y: number) => x < 0 || y < 0 || x >= 20 || y >= 20) =>
  A.buildActionWorld(20, 20, flat, solid, props);
const crate = (x: number, y: number, extra: Partial<Prop> = {}): Prop => ({
  id: `crate-${x}-${y}`,
  kind: "crate",
  x,
  y,
  rot: 0,
  scale: 1,
  solid: true,
  ...extra,
});
const still = { dx: 0, dy: 0, sprint: false, crouch: false, jump: false };
const east = { ...still, dx: 1 };

function run(w: ReturnType<typeof world>, b: ReturnType<typeof A.newBody>, fps: number, seconds: number, input = east) {
  for (let i = 0; i < Math.round(fps * seconds); i++) A.advance(w, b, input, 1 / fps, 3);
}

test("the same input moves the same distance at 30, 60 and 144 frames a second", () => {
  const w = world();
  const ends = [30, 60, 144].map((fps) => {
    const b = A.newBody(w, 2.5, 5.5);
    run(w, b, fps, 1.5);
    return b.x;
  });
  for (const x of ends) assert.ok(Math.abs(x - ends[1]) < 0.06, `${ends}`);
  assert.ok(Math.abs(ends[1] - (2.5 + 4.5)) < 0.06, `${ends[1]}`);
});

test("a jump needs the ground (or a moment after leaving it), and gravity brings the body back", () => {
  const w = world();
  const b = A.newBody(w, 5.5, 5.5);
  A.advance(w, b, { ...still, jump: true }, A.STEP_S, 3);
  assert.ok(b.vz > 0 && !b.grounded);
  let peak = 0;
  for (let i = 0; i < 20; i++) {
    A.advance(w, b, still, A.STEP_S, 3);
    peak = Math.max(peak, b.z);
  }
  const vz = b.vz;
  A.advance(w, b, { ...still, jump: true }, A.STEP_S, 3);
  assert.ok(b.vz < vz, "no jump in mid-air");
  for (let i = 0; i < 120; i++) A.advance(w, b, still, A.STEP_S, 3);
  assert.ok(peak > 0.6 && peak < 1.2, `${peak}`);
  assert.equal(b.z, 0);
  assert.ok(b.grounded);
});

test("a jump pressed on a short frame that runs no physics step still jumps", () => {
  const w = world();
  const b = A.newBody(w, 5.5, 5.5);
  assert.equal(A.advance(w, b, { ...still, jump: true }, 1 / 144, 3), 0);
  for (let i = 0; i < 3; i++) A.advance(w, b, still, 1 / 144, 3);
  assert.ok(b.vz > 0 && !b.grounded, "the press was kept for the next step");
});

test("sprinting into a wall never passes through it, and the body slides along it", () => {
  const w = world([], wallAt);
  const b = A.newBody(w, 8.5, 5.5);
  run(w, b, 20, 3, { ...still, dx: 1, dy: 0.3, sprint: true });
  assert.ok(b.x < 10 - A.RADIUS + 1e-6, `${b.x}`);
  assert.ok(b.y > 6.5, "slid along the wall");
  const c = A.newBody(w, 9.5, 2.5);
  for (let i = 0; i < 30; i++) A.advance(w, c, { ...still, dx: 1, sprint: true }, 0.25, 3);
  assert.ok(c.x < 10, `${c.x}`);
});

test("a low step is climbed, a tall box blocks, and a jump gets onto a crate", () => {
  const rock = world([{ ...crate(6.5, 5.5), kind: "rock", scale: 0.7 }]);
  const b = A.newBody(rock, 4.5, 5.5);
  run(rock, b, 60, 1.2);
  assert.ok(b.x > 7, "walked over the 0.38 rock");
  const tall = world([{ ...crate(6.5, 5.5), scale: 2 }]);
  const c = A.newBody(tall, 4.5, 5.5);
  run(tall, c, 60, 2);
  assert.ok(c.x < 6.5 - 0.52 - A.RADIUS + 0.05, `${c.x}`);
  const box = world([crate(6.5, 5.5)]);
  const d = A.newBody(box, 4.5, 5.5);
  run(box, d, 60, 1);
  assert.ok(d.x < 6.5, "blocked without a jump");
  A.advance(box, d, { ...east, jump: true }, A.STEP_S, 3);
  for (let i = 0; i < 12; i++) A.advance(box, d, east, A.STEP_S, 3);
  for (let i = 0; i < 6 && d.x < 6.5; i++) A.advance(box, d, east, A.STEP_S, 3);
  for (let i = 0; i < 60; i++) A.advance(box, d, still, A.STEP_S, 3);
  assert.ok(Math.abs(d.z - 0.51) < 1e-6 && d.grounded, `${d.x},${d.z}`);
});

test("crouching passes under a low roof, and standing up waits for headroom", () => {
  const slab = A.buildActionWorld(20, 20, flat, (x, y) => x < 0 || y < 0 || x >= 20 || y >= 20, []);
  const roof: import("../src/features/flows/player/world/action3d.ts").Box = {
    cx: 7.5,
    cy: 5.5,
    hx: 1,
    hy: 1,
    cos: 1,
    sin: 0,
    bottom: 1,
    top: 1.4,
  };
  slab.boxes.push(roof);
  for (let y = 3; y <= 8; y++) for (let x = 5; x <= 10; x++) slab.cells.set(y * 20 + x, [roof]);
  const b = A.newBody(slab, 4.5, 5.5);
  run(slab, b, 60, 1);
  assert.ok(b.x < 6.5 - A.RADIUS + 0.01, `standing is blocked at ${b.x}`);
  run(slab, b, 60, 0.6, { ...east, crouch: true });
  assert.ok(b.x > 7, `crouched under at ${b.x}`);
  assert.ok(b.crouched);
  run(slab, b, 60, 0.1, still);
  assert.ok(b.crouched);
  run(slab, b, 60, 1.5, east);
  assert.ok(b.x > 8.6 && !b.crouched, `stood up past the roof at ${b.x}`);
});

test("rotated props block along their rotated footprint", () => {
  const fence = (rot: number) => world([{ ...crate(6.5, 5.5), kind: "fence", rot }]);
  const south = { ...still, dy: 1 };
  const a = A.newBody(fence(0), 7.2, 4.5);
  run(fence(0), a, 60, 1, south);
  assert.ok(a.y < 5.5, `${a.y}`);
  const b = A.newBody(fence(90), 7.2, 4.5);
  run(fence(90), b, 60, 1, south);
  assert.ok(b.y > 6.5, `${b.y}`);
});

test("a prop's collider turns the same way three.js turns the rendered prop", async () => {
  const three = await import("three");
  const fence = A.propBox({ ...crate(5, 5), kind: "fence", rot: 45, scale: 3 }, () => 0)!;
  const end = new three.Vector3(1.2, 0, 0).applyAxisAngle(new three.Vector3(0, 1, 0), Math.PI / 4);
  assert.ok(A.circleHitsBox(fence, 5 + end.x, 5 + end.z, 0.05));
  assert.ok(!A.circleHitsBox(fence, 5 + end.x, 5 - end.z, 0.05));
});

test("shots hit the first hostile in line, walls stop them, friendlies are never hurt", () => {
  const w = world([], wallAt);
  const c = A.newCombat();
  const eye = { x: 2.5, y: 5.5, z: 1.4 };
  const dir = { x: 1, y: 0, z: -0.12 };
  const grim = { id: "grim", x: 6.5, y: 5.5, z: 0, hostile: true };
  const tom = { id: "tom", x: 4.5, y: 5.5, z: 0, hostile: false };
  assert.deepEqual(A.shoot(w, c, eye, dir, [grim, tom]), { fired: true, hit: null, downed: false });
  assert.equal(c.enemies.tom, undefined);
  A.tick(c, 1);
  const hit = A.shoot(w, c, eye, dir, [grim]);
  assert.equal(hit.hit, "grim");
  assert.equal(c.enemies.grim.health, A.ENEMY_HEALTH - A.SHOT_DAMAGE);
  assert.equal(A.shoot(w, c, eye, dir, [grim]).fired, false);
  A.tick(c, 1);
  const behind = { ...grim, x: 12.5 };
  assert.equal(A.shoot(w, c, eye, dir, [behind]).hit, null);
  let downed = false;
  for (let i = 0; i < 3; i++) {
    A.tick(c, 1);
    downed = A.shoot(w, c, eye, dir, [grim]).downed || downed;
  }
  assert.ok(downed);
  A.tick(c, 1);
  assert.equal(A.shoot(w, c, eye, dir, [grim]).hit, null, "a knocked-out villager is not a target");
  A.tick(c, A.KNOCKOUT_S);
  assert.equal(A.shoot(w, c, eye, dir, [grim]).hit, "grim");
  c.ammo = 0;
  A.tick(c, 1);
  assert.equal(A.shoot(w, c, eye, dir, [grim]).fired, false);
});

test("level aim still hits a hostile a little below or above, not one far overhead", () => {
  const slope = (x: number) => -Math.max(0, x - 2.5) * 0.25;
  const w = A.buildActionWorld(20, 20, slope, (x, y) => x < 0 || y < 0 || x >= 20 || y >= 20, []);
  const level = { x: 1, y: 0, z: 0 };
  const eye = { x: 2.5, y: 5.5, z: 1.1 };
  const c = A.newCombat();
  assert.equal(A.shoot(w, c, eye, level, [{ id: "g", x: 8.5, y: 5.5, z: slope(8.5), hostile: true }]).hit, "g");
  A.tick(c, 1);
  assert.equal(A.shoot(w, c, eye, level, [{ id: "g", x: 8.5, y: 5.5, z: 4, hostile: true }]).hit, null);
});

test("hostiles shoot back only in range, in sight and outside safe places; defeat costs only ammo", () => {
  const w = world([], wallAt);
  const c = A.newCombat();
  const me = { x: 8.5, y: 5.5, z: 0 };
  const grim = { id: "grim", x: 5.5, y: 5.5, z: 0, hostile: true };
  const none = () => false;
  assert.deepEqual(A.enemiesAct(w, c, me, [grim], none), ["grim"]);
  assert.equal(c.health, A.MAX_HEALTH - A.ENEMY_DAMAGE);
  assert.deepEqual(A.enemiesAct(w, c, me, [grim], none), [], "cooldown");
  A.tick(c, A.ENEMY_COOLDOWN_S + 0.01);
  assert.deepEqual(A.enemiesAct(w, c, { ...me, x: 12.5 }, [grim], none), [], "wall in between");
  assert.deepEqual(A.enemiesAct(w, c, { ...me, x: 5.5 + A.ENEMY_RANGE + 1, y: 18.5 }, [{ ...grim, y: 18.5 }], none), [], "out of range");
  assert.deepEqual(A.enemiesAct(w, c, me, [grim], (x) => x > 8), [], "player safe");
  assert.deepEqual(A.enemiesAct(w, c, me, [grim], (x) => x < 6), [], "hostile in a safe place");
  assert.deepEqual(A.enemiesAct(w, c, me, [{ ...grim, hostile: false }], none), []);
  c.health = 0;
  c.ammo = A.START_AMMO + 12;
  A.respawn(c);
  assert.equal(c.health, A.MAX_HEALTH);
  assert.equal(c.ammo, A.START_AMMO, "picked-up ammo is lost");
  c.ammo = 0;
  A.respawn(c);
  assert.equal(c.ammo, A.START_AMMO, "never left unable to fight");
});

test("pickups are taken once, only when useful, from fixed seeded spots", () => {
  const open = Array.from({ length: 50 }, (_, i) => ({ x: i % 10, y: Math.floor(i / 10) }));
  const a = A.placePickups(open, 7);
  assert.deepEqual(a, A.placePickups(open, 7));
  assert.equal(new Set(a.map((p) => `${p.x},${p.y}`)).size, a.length);
  const c = A.newCombat();
  const health = a.find((p) => p.kind === "health")!;
  assert.equal(A.takePickup(c, a, health.x, health.y), null, "full health leaves it");
  c.health = 50;
  assert.equal(A.takePickup(c, a, health.x, health.y)?.id, health.id);
  assert.equal(c.health, 50 + A.PICKUP_HEALTH);
  assert.equal(A.takePickup(c, a, health.x, health.y), null, "taken once");
});

test("combat saves round-trip and anything else starts fresh", () => {
  const c = A.newCombat();
  c.health = 40;
  c.ammo = 7;
  c.taken = ["pickup-1"];
  c.enemies.grim = { health: 30, down: 120, cooldown: 99 };
  c.clock = 100;
  const back = A.combatOf(JSON.parse(JSON.stringify(c)));
  assert.equal(back.health, 40);
  assert.equal(back.ammo, 7);
  assert.deepEqual(back.taken, ["pickup-1"]);
  assert.equal(back.enemies.grim.down, 120);
  assert.deepEqual(A.combatOf(undefined), A.newCombat());
  assert.equal(A.combatOf({ health: -5, ammo: 1e9 }).ammo, A.MAX_AMMO);
});

test("action play is wired only when the world turns it on", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/three-world.ts", import.meta.url),
    "utf8",
  );
  assert.match(src, /const action = spec\.action\?\.movement === true;/);
  assert.match(src, /const shooter = action && spec\.action\?\.shooter === true;/);
  assert.match(src, /if \(action && body && aw\) \{/);
  assert.match(src, /\[me\.x, me\.y\] = slide\(map, me\.x, me\.y, nx - me\.x, ny - me\.y\);/);
});

test("a save keeps action state only for a world with action play on; old saves load as before", async () => {
  const W = await import("../src/features/flows/player/world/world-state.ts");
  type Spec = import("../src/features/flows/types.ts").WorldSpec;
  const spec = {
    title: "Arena",
    seed: 3,
    width: 24,
    height: 18,
    tile_size: 32,
    biome: "meadow",
    tileset: {},
    player: { name: "Pip" },
    npcs: [],
    items: [],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
  } as unknown as Spec;
  const armed = { ...spec, action: { movement: true, shooter: true } } as Spec;
  const action = { combat: { health: 40, ammo: 3 }, pose: { z: 0.6 } };
  const saved = { ...W.serialize(W.initialWorldState(armed)), action };
  assert.deepEqual(W.deserialize(armed, JSON.parse(JSON.stringify(saved)))?.action, action);
  assert.equal(W.deserialize(spec, saved)?.action, undefined);
  const old = W.serialize(W.initialWorldState(armed)) as unknown as Record<string, unknown>;
  delete old.action;
  assert.equal(W.deserialize(armed, old)?.action, undefined);
});

test("switching views keeps the fight, while a loaded save brings its own", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL("../src/features/flows/player/world/world-player.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /const bootedFor = loads\.current;/);
  assert.match(src, /const action = handle\.actionState\?\.\(\)[^;]*;\s*if \(action && loads\.current === bootedFor\)/);
});

test("switching views keeps scenery edits and villager positions, and pickups stay where the world put them", () => {
  const read = (f: string) =>
    readFileSync(new URL(`../src/features/flows/player/world/${f}`, import.meta.url), "utf8");
  const player = read("world-player.tsx");
  assert.match(player, /carried\.current = \{\s*map,\s*props: world\.props,\s*blocked: world\.blocked,/);
  assert.match(player, /npcs: loads\.current === bootedFor \? handle\.npcTiles\(\) : undefined/);
  assert.match(player, /const kept = carried\.current\?\.map === map \? carried\.current : null;/);
  assert.match(player, /\(switched && kept\?\.npcs\) \|\| wsRef\.current\.npcAt/);
  assert.match(read("three-world.ts"), /Math\.hypot\(x - map\.spawn\.x, y - map\.spawn\.y\) > 3/);
});

test("a prop dropped on the player stays passable for them, director hostiles fight, and carried props load their models", () => {
  const read = (f: string) =>
    readFileSync(new URL(`../src/features/flows/player/world/${f}`, import.meta.url), "utf8");
  const three = read("three-world.ts");
  assert.match(three, /circleHitsBox\(b, me\.x, me\.y, RADIUS\)/);
  assert.match(three, /all\.filter\(\(p\) => !over\.has\(p\.id\)\)/);
  assert.match(three, /addNpc\(npc: WorldNpc, at: Point\) \{\s*if \(npc\.role === "hostile"\) hostile\.add\(npc\.id\);/);
  assert.match(read("world-player.tsx"), /\(\(kept \? kept\.props : map\.props\) \?\? \[\]\)/);
  const house = A.propBox({ id: "h", kind: "house", x: 5, y: 5, rot: 0, scale: 1, solid: true } as Prop, flat);
  assert.ok(house && A.circleHitsBox(house, 5.2, 5.1, A.RADIUS));
  assert.ok(house && !A.circleHitsBox(house, 8, 8, A.RADIUS));
});

test("a knocked-out hostile is never the interaction target", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/three-world.ts", import.meta.url),
    "utf8",
  );
  assert.match(src, /if \(!n\.group\.visible\) continue;\s*const d = Math\.hypot\(n\.pos\.x - me\.x/);
  assert.match(src, /npcs\.filter\(\(n\) => n\.group\.visible\)\.map\(\(n\) => n\.group\)/);
});

test("a fast sprint cannot pass through a thin fence between two steps", () => {
  const fence = world([{ ...crate(5, 5), kind: "fence", rot: 90 }]);
  const b = A.newBody(fence, 4.709, 5);
  for (let i = 0; i < 30; i++) A.advance(fence, b, { ...east, sprint: true }, A.STEP_S, 20);
  assert.ok(b.x < 5, `${b.x}`);
});

test("hostile encounters give way to combat only in the 3D view", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/world-player.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /const shooterOn = baseSpec\.action\?\.shooter === true && view === "3d";/);
});

test("a route step stops on its waypoint instead of overshooting at high walk speeds", () => {
  const w = world();
  const b = A.newBody(w, 2.5, 5.5);
  for (let i = 0; i < 30; i++) {
    const dx = 3.5 - b.x;
    if (Math.abs(dx) < 0.15) break;
    A.advance(w, b, { ...still, dx, limit: Math.abs(dx) }, 1 / 30, 20);
  }
  assert.ok(Math.abs(b.x - 3.5) < 0.15, `${b.x}`);
});

test("a prop the player stands on keeps holding them up when colliders are rebuilt", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/three-world.ts", import.meta.url),
    "utf8",
  );
  assert.match(src, /b\.top > z \+ 1e-3 &&\s*b\.bottom < z \+ STAND_H &&\s*circleHitsBox\(b, me\.x, me\.y, RADIUS\)/);
  assert.match(src, /let aw: ActionWorld \| null = action \? collide\(feet\) : null;/);
});

test("safe places hold for the player's shots as well as the hostiles'", () => {
  const src = readFileSync(
    new URL("../src/features/flows/player/world/three-world.ts", import.meta.url),
    "utf8",
  );
  assert.match(src, /if \(firePressed && safe\(body\.x, body\.y\)\) \{\s*firePressed = false;/);
  assert.match(src, /targets\(\)\.filter\(\(t\) => !safe\(t\.x, t\.y\)\)/);
});

test("crouching behind cover hides the player from hostile fire", () => {
  const w = world([{ ...crate(8, 5.5), scale: 1.86 }]);
  const me = { x: 9.5, y: 5.5, z: 0 };
  const grim = { id: "grim", x: 5.5, y: 5.5, z: 0, hostile: true };
  const none = () => false;
  assert.deepEqual(A.enemiesAct(w, A.newCombat(), { ...me, h: A.STAND_H }, [grim], none), ["grim"]);
  assert.deepEqual(A.enemiesAct(w, A.newCombat(), { ...me, h: A.CROUCH_H }, [grim], none), []);
});

test("a door the player's capsule reaches into from the next tile cannot be closed", async () => {
  const V = await import("../src/features/flows/player/world/action-verbs.ts");
  const door = { id: "d", kind: "door", x: 5.5, y: 5.5 } as Prop;
  const flat = () => 0;
  const me = { x: 4.9, y: 5.5 };
  // Tile 4 vs the door's tile 5: a tile check sees nothing in the way.
  assert.notEqual(Math.floor(me.x), Math.floor(door.x));
  assert.equal(A.capsuleOverlapsProp(door, flat, me.x, me.y), true);
  assert.equal(A.capsuleOverlapsProp(door, flat, 3.5, 5.5), false);
  const out = V.interactProp(door, [door], { d: { open: true } }, [], {
    occupied: (p) => A.capsuleOverlapsProp(p, flat, me.x, me.y),
  });
  assert.deepEqual(out.changed, []);
  const player = readFileSync(
    new URL("../src/features/flows/player/world/world-player.tsx", import.meta.url),
    "utf8",
  );
  assert.match(player, /occupied: \(p\) =>\s*g\.blocksProp\?\.\(p\) === true/);
});
