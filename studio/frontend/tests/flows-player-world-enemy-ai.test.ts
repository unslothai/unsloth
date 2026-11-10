// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const A = await import("../src/features/flows/player/world/action3d.ts");
const V = await import("../src/features/flows/player/world/action-verbs.ts");
const E = await import("../src/features/flows/player/world/enemy-ai.ts");
type Prop = import("../src/features/flows/types.ts").WorldProp3d;
type Noise = ReturnType<typeof V.newVerbs>["noises"][number];
type Sense = Parameters<typeof E.canSee>[2];

const N = 24;
function arena(walls: [number, number][] = [], props: Prop[] = []) {
  const wall = new Set(walls.map(([x, y]) => y * N + x));
  const solid = (x: number, y: number) =>
    x < 0 || y < 0 || x >= N || y >= N || wall.has(y * N + x);
  const aw = A.buildActionWorld(N, N, () => 0, solid, props);
  const grid = E.gridOf(
    N,
    N,
    (x, y) =>
      solid(x, y) ||
      props.some((p) => Math.floor(p.x) === x && Math.floor(p.y) === y),
  );
  const brain = E.createBrain(
    grid,
    (ox, oy, oz, dx, dy, dz, max) =>
      A.wallDistance(aw, ox, oy, oz, dx, dy, dz, max),
    () => 0,
    (x, y) => props.some((p) => Math.floor(p.x) === x && Math.floor(p.y) === y),
  );
  return { brain, grid };
}
const player = (x: number, y: number, extra: Partial<Sense> = {}) => ({
  x,
  y,
  z: 0,
  h: A.STAND_H,
  fx: 1,
  fy: 0,
  crouched: false,
  hidden: false,
  ...extra,
});
function run(
  brain: ReturnType<typeof arena>["brain"],
  c: ReturnType<typeof A.newCombat>,
  p: ReturnType<typeof player>,
  seconds: number,
  noises: Noise[] = [],
  each?: () => void,
) {
  for (let i = 0; i < Math.round(seconds * 60); i++) {
    A.tick(c, 1 / 60);
    E.brainStep(brain, 1 / 60, p, c, noises);
    each?.();
  }
}

function bfs(
  grid: ReturnType<typeof E.gridOf>,
  sx: number,
  sy: number,
  tx: number,
  ty: number,
) {
  const dist = new Int32Array(grid.width * grid.height).fill(-1);
  const q = [sy * grid.width + sx];
  dist[q[0]] = 0;
  for (let h = 0; h < q.length; h++) {
    const i = q[h];
    const x = i % grid.width;
    const y = (i / grid.width) | 0;
    for (const [nx, ny] of [
      [x + 1, y],
      [x - 1, y],
      [x, y + 1],
      [x, y - 1],
    ]) {
      if (nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height) continue;
      const j = ny * grid.width + nx;
      if (grid.blocked[j] || dist[j] >= 0) continue;
      dist[j] = dist[i] + 1;
      q.push(j);
    }
  }
  return dist[ty * grid.width + tx];
}

test("A* finds paths as short as breadth-first search, on open and cluttered maps, and says when there is none", () => {
  let seed = 7;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const finder = new E.Pathfinder(32 * 32);
  const out = new Int32Array(256);
  for (const density of [0, 0, 0.12, 0.25]) {
    const blocked = new Uint8Array(32 * 32);
    for (let i = 0; i < blocked.length; i++)
      blocked[i] = rnd() < density ? 1 : 0;
    const grid = { width: 32, height: 32, blocked };
    for (let k = 0; k < 40; k++) {
      const s = [Math.floor(rnd() * 32), Math.floor(rnd() * 32)];
      const t = [Math.floor(rnd() * 32), Math.floor(rnd() * 32)];
      blocked[s[1] * 32 + s[0]] = 0;
      const want = blocked[t[1] * 32 + t[0]]
        ? -1
        : bfs(grid, s[0], s[1], t[0], t[1]);
      const got = finder.find(grid, s[0], s[1], t[0], t[1], out);
      assert.equal(got, want, `density ${density} ${s} -> ${t}`);
      let prev = s[1] * 32 + s[0];
      for (let i = 0; i < got; i++) {
        const step = out[i];
        assert.equal(
          Math.abs((step % 32) - (prev % 32)) +
            Math.abs(((step / 32) | 0) - ((prev / 32) | 0)),
          1,
        );
        assert.equal(blocked[step], 0);
        prev = step;
      }
      if (got > 0) assert.equal(prev, t[1] * 32 + t[0]);
    }
  }
  const wall = { width: 8, height: 8, blocked: new Uint8Array(64) };
  for (let y = 0; y < 8; y++) wall.blocked[y * 8 + 4] = 1;
  assert.equal(finder.find(wall, 1, 1, 6, 6, out), -1);
  assert.equal(finder.find(wall, 1, 1, 1, 1, out), 0);
});

test("a long path fills only the agent's buffer and still reports its full length", () => {
  const grid = { width: 64, height: 4, blocked: new Uint8Array(256) };
  const out = new Int32Array(8);
  assert.equal(new E.Pathfinder(256).find(grid, 0, 0, 60, 0, out), 60);
  assert.deepEqual([...out], [1, 2, 3, 4, 5, 6, 7, 8]);
});

test("a hostile sees the player in its view cone, not behind a wall, not behind its back, and always up close", () => {
  const { brain } = arena([
    [10, 5],
    [10, 6],
    [10, 7],
  ]);
  const a = E.addAgent(brain, "h", 5.5, 6.5);
  a.fx = 1;
  a.fy = 0;
  assert.equal(E.canSee(brain, a, player(9.5, 6.5)), true);
  assert.equal(
    E.canSee(brain, a, player(12.5, 6.5)),
    false,
    "seen through a wall",
  );
  assert.equal(
    E.canSee(brain, a, player(2.5, 6.5)),
    false,
    "seen behind its back",
  );
  assert.equal(E.canSee(brain, a, player(4.8, 6.5)), true, "felt up close");
  assert.equal(
    E.canSee(brain, a, player(6.5, 12.5)),
    false,
    "outside the cone",
  );
  assert.equal(
    E.canSee(brain, a, player(9.5, 6.5, { hidden: true })),
    false,
    "safe places hide the player",
  );
  a.state = "chase";
  a.lastAt = brain.clock;
  assert.equal(
    E.canSee(brain, a, player(2.5, 6.5)),
    true,
    "right after contact it tracks all around",
  );
  a.lastAt = brain.clock - 2;
  assert.equal(
    E.canSee(brain, a, player(2.5, 6.5)),
    false,
    "a hostile that lost the player looks where it faces",
  );
});

test("crouching shrinks how far a hostile can see you", () => {
  const { brain } = arena();
  const a = E.addAgent(brain, "h", 2.5, 6.5);
  a.fx = 1;
  a.fy = 0;
  assert.equal(E.canSee(brain, a, player(10.5, 6.5)), true);
  assert.equal(
    E.canSee(brain, a, player(10.5, 6.5, { crouched: true })),
    false,
  );
});

test("sprinting is heard, crouch-walking is not, and noise fades with distance", () => {
  const v = V.newVerbs();
  const a = { ...E.addAgent(arena().brain, "h", 5.5, 5.5) };
  V.makeNoise(v, 9.5, 5.5, V.NOISE.crouch);
  assert.equal(E.hear(a, v.noises), false);
  V.makeNoise(v, 9.5, 5.5, V.NOISE.sprint);
  assert.equal(E.hear(a, v.noises), true);
  assert.deepEqual([a.lastX, a.lastY], [9.5, 5.5]);
  assert.equal(E.hear(a, v.noises), false, "a noise is heard once");
  V.makeNoise(v, 5.5 + V.NOISE.walk + 0.5, 5.5, V.NOISE.walk);
  assert.equal(E.hear(a, v.noises), false, "a step out of earshot");
  V.makeNoise(v, 5.5 + V.NOISE.shot - 1, 5.5, V.NOISE.shot);
  assert.equal(E.hear(a, v.noises), true);
});

test("idle, alert after a reaction delay, attack in range, and the player takes damage", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const a = E.addAgent(brain, "h", 5.5, 6.5, true);
  a.fx = 1;
  a.fy = 0;
  const seen: string[] = [a.state];
  let attackAt = -1;
  run(brain, c, player(9.5, 6.5), 2.5, [], () => {
    if (seen.at(-1) !== a.state) seen.push(a.state);
    if (a.state === "attack" && attackAt < 0) attackAt = brain.clock;
  });
  assert.deepEqual(seen.slice(0, 3), ["idle", "alert", "attack"]);
  assert.ok(attackAt >= E.REACT_S, `attacked after ${attackAt} s`);
  assert.ok(c.health < A.MAX_HEALTH);
  assert.ok(a.attacks >= 1);
  assert.ok(E.BARKS.see.includes(a.bark) || E.BARKS.attack.includes(a.bark));
});

test("up close a wall still hides the player, and a hostile in a safe place never attacks", () => {
  const { brain } = arena([
    [6, 5],
    [6, 6],
    [6, 7],
  ]);
  const a = E.addAgent(brain, "h", 5.5, 6.5);
  assert.equal(
    E.canSee(brain, a, player(6.9, 6.5)),
    false,
    "felt through a wall",
  );
  const open = arena();
  const c = A.newCombat();
  const b = E.addAgent(open.brain, "s", 5.5, 6.5, true);
  b.fx = 1;
  b.fy = 0;
  open.brain.safe = (x, y) => Math.floor(x) === 5 && Math.floor(y) === 6;
  run(open.brain, c, player(6.5, 6.5), 3);
  assert.equal(c.health, A.MAX_HEALTH);
  assert.equal(b.attacks, 0);
});

test("a hostile waking from a knockout does not chase noises made while it was out", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const a = E.addAgent(brain, "h", 5.5, 6.5, true);
  c.enemies.h = { health: A.ENEMY_HEALTH, down: c.clock + 2, cooldown: 0 };
  const noises = [{ seq: 1, x: 8.5, y: 6.5, radius: 10, at: 0 }];
  const states = new Set<string>();
  run(brain, c, player(20.5, 20.5, { hidden: true }), 4, noises, () =>
    states.add(a.state),
  );
  assert.deepEqual(
    [...states].filter((x) => x !== "idle" && x !== "patrol"),
    [],
  );
});

test("a hostile added mid-play does not hear noises made before it existed", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const v = V.newVerbs();
  V.makeNoise(v, 6.5, 5.5, V.NOISE.shot);
  const hidden = player(22, 22, { hidden: true });
  run(brain, c, hidden, 30, v.noises);
  const late = E.addAgent(brain, "late", 5.5, 5.5, true);
  const roused = () => {
    const states = new Set<string>();
    run(brain, c, hidden, 0.5, v.noises, () => states.add(late.state));
    return [...states].filter((x) => x !== "idle" && x !== "patrol");
  };
  assert.deepEqual(roused(), []);
  V.makeNoise(v, 6.5, 5.5, V.NOISE.shot);
  assert.notDeepEqual(roused(), [], "a fresh noise still reaches it");
});

test("it decides at 10 Hz while it moves every frame", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const v = V.newVerbs();
  const a = E.addAgent(brain, "h", 5.5, 5.5);
  let heardChanges = 0;
  let last = 0;
  for (let i = 0; i < 60; i++) {
    V.makeNoise(v, 30, 30, 1);
    A.tick(c, 1 / 60);
    E.brainStep(brain, 1 / 60, player(20.5, 20.5), c, v.noises);
    if (a.heardSeq !== last) heardChanges++;
    last = a.heardSeq;
  }
  assert.ok(heardChanges >= 9 && heardChanges <= 11, `${heardChanges}`);
  const b2 = arena().brain;
  const m = E.addAgent(b2, "m", 2.5, 2.5);
  E.brainStep(b2, 1 / 60, player(20.5, 2.5), A.newCombat(), []);
  m.state = "chase";
  m.lastX = 12.5;
  m.lastY = 2.5;
  m.lastAt = 0;
  const xs: number[] = [];
  for (let i = 0; i < 30; i++) {
    E.brainStep(b2, 1 / 60, player(20.5, 20.5), A.newCombat(), []);
    xs.push(m.x);
  }
  const steps = xs
    .slice(8)
    .map((x, i, all) => (i ? x - all[i - 1] : 0))
    .slice(1);
  for (const s of steps) assert.ok(Math.abs(s - 2.4 / 60) < 1e-6, `${steps}`);
});

test("a noise sends a hostile to look, and it gives up and goes home when it finds nothing", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const v = V.newVerbs();
  const a = E.addAgent(brain, "h", 5.5, 5.5, true);
  a.fx = -1;
  a.fy = 0;
  V.makeNoise(v, 11.5, 5.5, V.NOISE.door);
  run(brain, c, player(20.5, 20.5, { crouched: true }), 0.5, v.noises);
  assert.equal(a.state, "chase");
  assert.ok(E.BARKS.hear.includes(a.bark));
  let returned = false;
  run(brain, c, player(20.5, 20.5, { crouched: true }), 8, v.noises, () => {
    returned ||= a.state === "return";
  });
  assert.ok(returned);
  run(brain, c, player(20.5, 20.5, { crouched: true }), 8, v.noises);
  assert.equal(a.state, "idle");
  assert.ok(Math.hypot(a.x - 5.5, a.y - 5.5) < 0.01);
});

function escape(sneak: boolean) {
  const walls: [number, number][] = [];
  for (let x = 0; x < 16; x++) walls.push([x, 12]);
  const { brain } = arena(walls);
  const c = A.newCombat();
  const v = V.newVerbs();
  const a = E.addAgent(brain, "h", 4.5, 16.5, true);
  a.fx = 1;
  a.fy = 0;
  const p = player(13.5, 14.5);
  run(brain, c, p, 0.5, v.noises);
  const spotted = a.state;
  const route: [number, number][] = [
    [16.5, 14.5],
    [16.5, 10.5],
    [6.5, 10.5],
  ];
  let at = 0;
  const states = new Set<string>();
  for (let i = 0; i < 60 * 12; i++) {
    if (sneak && at < route.length) {
      p.crouched = true;
      const [tx, ty] = route[at];
      const d = Math.hypot(tx - p.x, ty - p.y);
      const step = 1.6 / 60;
      if (d <= step) at++;
      else {
        p.x += ((tx - p.x) / d) * step;
        p.y += ((ty - p.y) / d) * step;
      }
      if (i % 24 === 0) V.makeNoise(v, p.x, p.y, V.NOISE.crouch);
    }
    A.tick(c, 1 / 60);
    E.brainStep(brain, 1 / 60, p, c, v.noises);
    states.add(a.state);
  }
  return { spotted, states, a };
}

test("a hostile loses track when the player breaks line of sight and crouches away, not while they stand in view", () => {
  const away = escape(true);
  assert.equal(away.spotted, "chase");
  assert.ok(away.states.has("return"), [...away.states].join());
  assert.equal(away.a.state, "idle");
  assert.deepEqual([away.a.x, away.a.y], [4.5, 16.5]);
  assert.ok(E.BARKS.lose.includes(away.a.bark));
  const stay = escape(false);
  assert.equal(stay.states.has("return"), false);
  assert.ok(
    stay.a.state === "attack" || stay.a.state === "chase",
    stay.a.state,
  );
});

test("pain on a hit, flee when badly hurt, and a knocked-out hostile wakes up at home", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const a = E.addAgent(brain, "h", 5.5, 5.5, true);
  a.fx = 1;
  a.fy = 0;
  const p = player(9.5, 5.5);
  run(brain, c, p, 1);
  V.hurt(c, "h", 30);
  run(brain, c, p, 0.15);
  assert.equal(a.state, "pain");
  assert.ok(E.BARKS.pain.includes(a.bark));
  run(brain, c, p, 0.5);
  assert.notEqual(a.state, "pain");
  V.hurt(c, "h", 50);
  run(brain, c, p, 0.6);
  assert.equal(a.state, "flee");
  const d0 = Math.hypot(a.x - p.x, a.y - p.y);
  run(brain, c, p, 1.5);
  assert.ok(Math.hypot(a.x - p.x, a.y - p.y) > d0 + 1);
  V.hurt(c, "h", 50);
  run(brain, c, p, 0.2);
  assert.ok(a.down);
  assert.deepEqual([a.x, a.y, a.state], [5.5, 5.5, "idle"]);
});

test("utility: healthy hostiles attack in range and chase from afar; hurt ones seek cover behind a solid prop", () => {
  const crate: Prop = {
    id: "c",
    kind: "crate",
    x: 12.5,
    y: 10.5,
    rot: 0,
    scale: 1,
    solid: true,
  };
  const { brain } = arena([], [crate]);
  const a = E.addAgent(brain, "h", 11.5, 12.5);
  assert.equal(E.choose(brain, a, player(9.5, 12.5)), "attack");
  assert.equal(E.choose(brain, a, player(20.5, 12.5)), "chase");
  a.health = A.ENEMY_HEALTH * 0.45;
  assert.equal(E.choose(brain, a, player(6.5, 10.5)), "attack");
  assert.equal(a.tactic, "cover");
  const tile = E.pickCover(brain, a, player(6.5, 10.5));
  assert.equal(tile, 10 * N + 13, `${tile % N},${(tile / N) | 0}`);
  a.health = A.ENEMY_HEALTH * 0.2;
  assert.equal(E.choose(brain, a, player(6.5, 10.5)), "flee");
});

test("a second hostile flanks: it picks a tile with a clear shot, out of the player's gaze and away from its ally", () => {
  const { brain } = arena();
  const ally = E.addAgent(brain, "ally", 10.5, 6.5);
  ally.state = "attack";
  // It starts right in front of the player, where the shortest walk would leave it.
  const a = E.addAgent(brain, "flanker", 16.5, 10.5);
  const p = player(10.5, 10.5, { fx: 1, fy: 0 });
  const tile = E.pickFlank(brain, a, p);
  const tx = (tile % N) + 0.5;
  const ty = ((tile / N) | 0) + 0.5;
  const d = Math.hypot(tx - p.x, ty - p.y);
  assert.ok(d >= 2.5 && d <= 6, `${tx},${ty}`);
  assert.ok((tx - p.x) / d < 0.5, `in the player's gaze at ${tx},${ty}`);
  assert.ok(
    Math.hypot(tx - ally.x, ty - ally.y) > 3,
    `crowding the ally at ${tx},${ty}`,
  );
  assert.ok(ty > p.y, `the far side from the ally: ${tx},${ty}`);
});

test("guards hold their post; others patrol near home", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const guard = E.addAgent(brain, "g", 5.5, 5.5, true);
  const walker = E.addAgent(brain, "w", 15.5, 15.5);
  let patrolled = false;
  run(brain, c, player(1.5, 22.5, { crouched: true }), 12, [], () => {
    patrolled ||= walker.state === "patrol";
    assert.ok(Math.hypot(walker.x - 15.5, walker.y - 15.5) <= 4.8);
  });
  assert.ok(patrolled);
  assert.deepEqual([guard.x, guard.y, guard.state], [5.5, 5.5, "idle"]);
});

test("the world may bring its own barks; built-in lines fill the rest", () => {
  const pool = E.barkPool({
    see: ["Raider spotted!", 3, ""],
    lose: "nope",
    bogus: ["x"],
  });
  assert.deepEqual(pool.see, ["Raider spotted!"]);
  assert.deepEqual(pool.lose, E.BARKS.lose);
  assert.equal("bogus" in pool, false);
  const brain = E.createBrain(
    arena().grid,
    () => Number.POSITIVE_INFINITY,
    () => 0,
    () => false,
    { see: ["Raider spotted!"] },
  );
  const a = E.addAgent(brain, "h", 5.5, 5.5, true);
  a.fx = 1;
  a.fy = 0;
  run(brain, A.newCombat(), player(8.5, 5.5), 0.2);
  assert.equal(a.bark, "Raider spotted!");
});

test("crouching lowers the body and quiets footsteps, so a nearby hostile hears a walk but not a crouch", () => {
  const w = A.buildActionWorld(
    N,
    N,
    () => 0,
    () => false,
    [],
  );
  const b = A.newBody(w, 5.5, 5.5);
  A.advance(
    w,
    b,
    { dx: 0, dy: 0, sprint: false, crouch: true, jump: false },
    1 / 60,
    3,
  );
  assert.equal(A.bodyHeight(b), A.CROUCH_H);
  assert.ok(A.CROUCH_H < A.STAND_H);
  assert.equal(
    V.stepNoise(b, true),
    V.NOISE.crouch,
    "crouching beats sprinting",
  );
  assert.equal(V.stepNoise({ crouched: false }, true), V.NOISE.sprint);
  assert.equal(V.stepNoise({ crouched: false }, false), V.NOISE.walk);
  const heardAt = (crouched: boolean) => {
    const v = V.newVerbs();
    const a = E.addAgent(arena().brain, "h", 5.5, 5.5);
    V.makeNoise(v, 7.5, 5.5, V.stepNoise({ crouched }, false));
    return E.hear(a, v.noises);
  };
  assert.equal(heardAt(false), true);
  assert.equal(heardAt(true), false);
});

test("a calm hostile follows village life; combat owns it while alerted and hands it back after returning", () => {
  const { brain } = arena();
  const c = A.newCombat();
  const a = E.addAgent(brain, "h", 5.5, 5.5, true);
  assert.equal(E.follow(a, 7.5, 5.5), false);
  assert.deepEqual([a.x, a.homeX], [7.5, 7.5]);
  a.fx = 1;
  a.fy = 0;
  const p = player(10.5, 5.5);
  run(brain, c, p, 0.5);
  assert.ok(E.engaged(a));
  assert.equal(
    E.follow(a, 1.5, 1.5),
    true,
    "village life cannot pull it away mid-fight",
  );
  assert.notEqual(a.x, 1.5);
  p.hidden = true;
  run(brain, c, p, 15);
  assert.equal(a.state, "idle");
  assert.deepEqual([a.x, a.y], [7.5, 5.5], "back where it was taken over");
  assert.equal(E.follow(a, 8.5, 5.5), false);
});

test("a raid's tagged bark list feeds the pool", () => {
  const pool = E.barkPool([
    { event: "spot", text: "Over here, lads!" },
    { event: "lost", text: "Keep looking." },
    { event: "retreat", text: "Run!" },
    { event: "dance", text: "no" },
    { event: "hurt", text: 5 },
  ]);
  assert.deepEqual(pool.see, ["Over here, lads!"]);
  assert.deepEqual(pool.lose, ["Keep looking."]);
  assert.deepEqual(pool.flee, ["Run!"]);
  assert.deepEqual(pool.pain, E.BARKS.pain);
});
