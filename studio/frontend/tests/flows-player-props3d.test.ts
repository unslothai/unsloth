// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const P = await import("../src/features/flows/player/world/procgen.ts");
const E = await import("../src/features/flows/player/world/props3d.ts");
const V = await import("../src/features/flows/player/world/world3d.ts");
const A = await import("../src/features/flows/player/player-api.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Prop = import("../src/features/flows/types.ts").WorldProp3d;
type Doc = import("../src/features/flows/types.ts").FlowDocument;

function spec(extra: Partial<Spec> = {}): Spec {
  return {
    title: "T",
    seed: 7,
    width: 40,
    height: 30,
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
  } as Spec;
}

/** A walkable tile next to spawn that is not reserved, to put a prop on. */
function openNear(map: ReturnType<typeof P.generateMap>) {
  const at = E.freeTileNear(
    map,
    { x: map.spawn.x + 2, y: map.spawn.y },
    (x, y) => !P.isSolid(map, x, y),
  );
  assert.ok(at);
  return at;
}

test("a world without props builds the same map as before and blocks nothing", () => {
  const map = P.generateMap(spec());
  assert.equal(map.blocked, undefined);
  assert.equal(map.props, undefined);
  const again = P.generateMap(spec({ props3d: [] }));
  assert.deepEqual(again.tiles, map.tiles);
  assert.equal(again.blocked, undefined);
});

test("a solid prop blocks its tile for walking (3D) and paths (2D and villagers)", () => {
  const base = P.generateMap(spec());
  const at = openNear(base);
  const prop: Prop = {
    id: "crate-1",
    kind: "crate",
    x: at.x + 0.5,
    y: at.y + 0.5,
    solid: true,
  };
  const map = P.generateMap(spec({ props3d: [prop] }));
  assert.deepEqual(map.tiles, base.tiles, "terrain is untouched");
  assert.equal(P.isSolid(base, at.x, at.y), false);
  assert.equal(P.isSolid(map, at.x, at.y), true);
  // 3D walking: the collision circle cannot stand on the prop's tile.
  assert.equal(V.canStand(base, at.x + 0.5, at.y + 0.5), true);
  assert.equal(V.canStand(map, at.x + 0.5, at.y + 0.5), false);
  // Click to walk and villager paths go around it.
  const path = P.findPath(map, map.spawn, at);
  assert.ok(!path.some((p) => p.x === at.x && p.y === at.y));
});

test("walk-past props and reserved tiles never block", () => {
  const base = P.generateMap(spec());
  const at = openNear(base);
  const lamp = E.makeProp("lamp", at, []);
  assert.equal(lamp.solid, false);
  const onSpawn: Prop = {
    id: "rock-1",
    kind: "rock",
    x: base.spawn.x + 0.5,
    y: base.spawn.y + 0.5,
    solid: true,
  };
  const npc = base.npcs.n0;
  const onNpc: Prop = {
    id: "house-1",
    kind: "house",
    x: npc.x + 0.2,
    y: npc.y + 0.7,
  };
  const map = P.generateMap(spec({ props3d: [lamp, onSpawn, onNpc] }));
  assert.equal(P.isSolid(map, at.x, at.y), false);
  assert.equal(P.isSolid(map, base.spawn.x, base.spawn.y), false);
  assert.equal(P.isSolid(map, npc.x, npc.y), false);
});

test("setMapProps updates collision in place, so a live edit and a view switch see it", () => {
  const map = P.generateMap(spec());
  const at = openNear(map);
  E.setMapProps(map, [E.makeProp("well", at, [])]);
  assert.equal(P.isSolid(map, at.x, at.y), true);
  E.setMapProps(map, []);
  assert.equal(P.isSolid(map, at.x, at.y), false);
  assert.equal(map.blocked, undefined);
});

test("a solid prop dropped where the player stands never traps them", async () => {
  const map = P.generateMap(spec());
  const at = openNear(map);
  E.setMapProps(map, [E.makeProp("well", at, [])], [at]);
  assert.equal(P.isSolid(map, at.x, at.y), false);
  E.setMapProps(map, [E.makeProp("well", at, [])]);
  assert.equal(
    P.isSolid(map, at.x, at.y),
    true,
    "once the player has left it blocks again",
  );
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/world/three-world.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    src,
    /onProps: \(next\) => setMapProps\(map, next, \[tileOf\(me\)\]\)/,
  );
  const player = readFileSync(
    new URL(
      "../src/features/flows/player/world/world-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    player,
    /await saveUntilCurrent\(current, savedProps, saveScene\)/,
  );
  // Done waits for model imports still uploading, so an imported model is saved rather than added after close.
  assert.match(
    player,
    /while \(imports\.current\.size\)\s*await Promise\.allSettled\(\[\.\.\.imports\.current\]\);/,
  );
  assert.match(player, /imports\.current\.add\(run\);/);
});

test("Done keeps saving until an edit made during a save is saved too", async () => {
  let shown = "a";
  const saves: string[] = [];
  const save = async () => {
    saves.push(shown);
    // The player edits again while this save is in flight.
    if (saves.length === 1) shown = "b";
    return true;
  };
  assert.equal(await E.saveUntilCurrent(() => shown, "", save), true);
  assert.deepEqual(saves, ["a", "b"]);
  assert.equal(
    await E.saveUntilCurrent(
      () => "x",
      "",
      async () => false,
    ),
    false,
  );
  assert.equal(await E.saveUntilCurrent(() => "same", "same", save), true);
  assert.equal(saves.length, 2, "nothing to save, no call");
});

test("new props get unique ids and land on a free tile, never on another prop", () => {
  const map = P.generateMap(spec());
  const a = E.makeProp("tree", openNear(map), []);
  E.setMapProps(map, [a]);
  const next = E.freeTileNear(
    map,
    E.propTile(a),
    (x, y) => !P.isSolid(map, x, y),
  );
  assert.ok(next);
  assert.notDeepEqual(next, E.propTile(a));
  const b = E.makeProp("tree", next, [a.id]);
  assert.equal(a.id, "tree-1");
  assert.equal(b.id, "tree-2");
});

test("snapping: tile centres, 15 degree turns, clamped scale", () => {
  assert.deepEqual(E.snapToTile(3.9, 4.1, 40, 30), { x: 3.5, y: 4.5 });
  assert.deepEqual(E.snapToTile(-2, 99, 40, 30), { x: 0.5, y: 29.5 });
  assert.equal(E.snapRotation(44), 45);
  assert.equal(E.snapRotation(-90), 270);
  assert.equal(E.snapRotation(359), 0);
  assert.equal(E.clampScale(9), 5);
  assert.equal(E.clampScale(0.01), 0.2);
  assert.equal(E.clampScale(Number.NaN), 1);
});

test("undo and redo step through whole prop lists; a new edit clears redo", () => {
  const one: Prop = { id: "a", kind: "crate", x: 1.5, y: 1.5 };
  let h = E.startHistory([]);
  h = E.commit(h, [one]);
  h = E.commit(h, E.updateProp(h.props, "a", { x: 2.5 }));
  assert.equal(h.props[0].x, 2.5);
  h = E.undo(h);
  assert.equal(h.props[0].x, 1.5);
  h = E.undo(h);
  assert.deepEqual(h.props, []);
  assert.equal(E.undo(h), h, "nothing left to undo");
  h = E.redo(h);
  assert.deepEqual(h.props, [one]);
  h = E.commit(h, E.removeProp(h.props, "a"));
  assert.deepEqual(h.future, []);
  assert.equal(E.redo(h), h);
  // An edit that changes nothing is not a step.
  assert.equal(E.commit(h, [...h.props]), h);
});

// A minimal binary glTF: one JSON chunk (padded to 4 bytes) and an optional BIN chunk.
function glb(json: object, bin = new Uint8Array(0)): ArrayBuffer {
  let text = JSON.stringify(json);
  while (text.length % 4) text += " ";
  const j = new TextEncoder().encode(text);
  const total = 12 + 8 + j.length + (bin.length ? 8 + bin.length : 0);
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, j.length, true);
  dv.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(buf, 20, j.length).set(j);
  if (bin.length) {
    dv.setUint32(20 + j.length, bin.length, true);
    dv.setUint32(24 + j.length, 0x004e4942, true);
    new Uint8Array(buf, 28 + j.length, bin.length).set(bin);
  }
  return buf;
}

const TRI = {
  asset: { version: "2.0" },
  buffers: [{ byteLength: 36 }],
  bufferViews: [{ buffer: 0, byteLength: 36 }],
  accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }],
  meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
};

test("a self-contained .glb passes with its triangle count", () => {
  assert.deepEqual(E.inspectGlb(glb(TRI, new Uint8Array(36))), {
    bytes: glb(TRI, new Uint8Array(36)).byteLength,
    triangles: 1,
  });
});

test("models that point elsewhere, are too big, too dense or need a codec are refused", () => {
  const refuse = (buf: ArrayBuffer, re: RegExp) =>
    assert.throws(
      () => E.inspectGlb(buf),
      (e: unknown) => e instanceof E.GlbError && re.test((e as Error).message),
    );
  refuse(
    glb({
      ...TRI,
      buffers: [{ byteLength: 36, uri: "https://example.com/x.bin" }],
    }),
    /outside files/,
  );
  refuse(glb({ ...TRI, images: [{ uri: "texture.png" }] }), /outside files/);
  refuse(new ArrayBuffer(21 * 1024 * 1024), /at most 20 MB/);
  refuse(
    glb({ ...TRI, accessors: [{ count: 3_000_000 }] }),
    /1,000,000 triangles/,
  );
  refuse(
    glb({ ...TRI, extensionsRequired: ["KHR_draco_mesh_compression"] }),
    /Draco/,
  );
  refuse(
    new TextEncoder().encode("not a model at all, sorry").buffer as ArrayBuffer,
    /Not a binary glTF/,
  );
  const cut = glb(TRI).slice(0, 40);
  refuse(cut, /cut short/);
});

test("scenery saves onto every enabled World Player block, and only there", () => {
  const doc = {
    id: "f",
    name: "F",
    kind: "world",
    nodes: [
      { id: "gen", type: "world.generate", params: { a: 1 } },
      { id: "play", type: "world.play", params: { view: "3d" } },
      { id: "off", type: "world.play", params: {}, disabled: true },
    ],
    edges: [],
  } as unknown as Doc;
  const props: Prop[] = [{ id: "a", kind: "crate", x: 1.5, y: 1.5 }];
  const out = A.withWorldProps(doc, props);
  assert.ok(out);
  const byId = Object.fromEntries(out.nodes.map((n) => [n.id, n.params]));
  assert.deepEqual(byId.play, { view: "3d", props3d: props });
  assert.deepEqual(byId.gen, { a: 1 });
  assert.deepEqual(byId.off, {});
  assert.equal(
    A.withWorldProps({ ...doc, nodes: [doc.nodes[0]] } as Doc, props),
    null,
  );
});
