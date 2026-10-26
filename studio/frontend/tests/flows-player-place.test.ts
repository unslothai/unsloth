// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const P = await import("../src/features/flows/player/world/procgen.ts");
const E = await import("../src/features/flows/player/world/props3d.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Prop = import("../src/features/flows/types.ts").WorldProp3d;

const spec = (extra: Partial<Spec> = {}): Spec =>
  ({
    title: "T",
    seed: 7,
    width: 30,
    height: 20,
    tile_size: 32,
    biome: "meadow",
    tileset: {},
    player: { name: "You" },
    npcs: [{ id: "n0", name: "N0", persona: "", greeting: "hi" }],
    items: [{ id: "i0", name: "I0", description: "" }],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    ...extra,
  }) as Spec;

test("the place request marks walls, props and kept-clear tiles", () => {
  const map = P.generateMap(spec());
  const at = E.freeTileNear(
    map,
    { x: map.spawn.x + 2, y: map.spawn.y },
    (x, y) => !P.isSolid(map, x, y),
  )!;
  const crate: Prop = {
    id: "crate-1",
    kind: "crate",
    x: at.x + 0.5,
    y: at.y + 0.5,
    solid: true,
  };
  const free = (x: number, y: number) => !P.isSolid(map, x, y);
  const req = E.placeRequest(map, [crate], free, "  a market  ", at);
  assert.equal(req.text, "a market");
  assert.equal(req.rows.length, map.height);
  assert.ok(req.rows.every((r) => r.length === map.width));
  assert.equal(req.rows[at.y][at.x], "#", "a solid prop blocks its tile");
  assert.equal(req.rows[map.spawn.y][map.spawn.x], "R");
  const npc = map.npcs.n0;
  assert.equal(req.rows[npc.y][npc.x], "R");
  for (let y = 0; y < map.height; y++)
    for (let x = 0; x < map.width; x++)
      if (req.rows[y][x] === ".") assert.ok(free(x, y), `${x},${y} is free`);
  assert.deepEqual(req.taken, ["crate-1"]);
});

test("placed props merge with fresh ids and unknown kinds dropped", () => {
  const have: Prop[] = [{ id: "well-1", kind: "well", x: 3.5, y: 3.5 }];
  const out = E.mergePlaced(
    have,
    [
      {
        id: "well-1",
        kind: "well",
        x: 5.5,
        y: 6.5,
        rot: 92,
        solid: true,
        name: "town well",
      },
      { id: "bench-1", kind: "bench", x: 6.5, y: 6.5, rot: 0, solid: false },
      { id: "m", kind: "model", x: 1, y: 1 },
      { id: "x", kind: "tree", x: Number.NaN, y: 1 },
      null,
    ],
    { width: 10, height: 10 },
  );
  assert.equal(out.length, 3);
  assert.equal(out[0], have[0]);
  assert.deepEqual(
    out.slice(1).map((p) => [p.id, p.kind, p.x, p.y, p.rot, p.solid]),
    [
      ["well-2", "well", 5.5, 6.5, 90, true],
      ["bench-1", "bench", 6.5, 6.5, 0, false],
    ],
  );
  assert.equal(out[1].name, "town well");
  assert.equal(
    E.mergePlaced(have, "nope", { width: 10, height: 10 }).length,
    1,
  );
});
