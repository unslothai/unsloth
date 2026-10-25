// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const three = await import("three");
const S = await import("../src/features/flows/player/world/props-scene3d.ts");
type Prop = import("../src/features/flows/types.ts").WorldProp3d;

/** A unit box as binary glTF (8 vertices, 12 triangles), the shape of a small Blender export. */
function boxGlb(): ArrayBuffer {
  const pos: number[] = [];
  for (const x of [-0.5, 0.5])
    for (const y of [0, 1]) for (const z of [-0.5, 0.5]) pos.push(x, y, z);
  const quads = [
    [0, 1, 3, 2],
    [4, 6, 7, 5],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 5, 7, 3],
  ];
  const idx = quads.flatMap(([a, b, c, d]) => [a, b, c, a, c, d]);
  const v = new Float32Array(pos);
  const i = new Uint16Array(idx);
  const bin = new Uint8Array(v.byteLength + i.byteLength);
  bin.set(new Uint8Array(v.buffer), 0);
  bin.set(new Uint8Array(i.buffer), v.byteLength);
  let text = JSON.stringify({
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: v.byteLength },
      { buffer: 0, byteOffset: v.byteLength, byteLength: i.byteLength },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 8,
        type: "VEC3",
        min: [-0.5, 0, -0.5],
        max: [0.5, 1, 0.5],
      },
      { bufferView: 1, componentType: 5123, count: idx.length, type: "SCALAR" },
    ],
  });
  while (text.length % 4) text += " ";
  const j = new TextEncoder().encode(text);
  const total = 12 + 8 + j.length + 8 + bin.length;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, j.length, true);
  dv.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(buf, 20, j.length).set(j);
  dv.setUint32(20 + j.length, bin.length, true);
  dv.setUint32(24 + j.length, 0x004e4942, true);
  new Uint8Array(buf, 28 + j.length, bin.length).set(bin);
  return buf;
}

const meshes = (o: import("three").Object3D) => {
  const out: import("three").Mesh[] = [];
  o.traverse((x) => {
    if ((x as import("three").Mesh).isMesh) out.push(x as import("three").Mesh);
  });
  return out;
};

const model = (id: string, ref = "sha256:boat"): Prop => ({
  id,
  kind: "model",
  x: 3.5,
  y: 4.5,
  asset: { kind: "model", ref },
});

test("built-in props stand on the ground, turn and scale, and go when removed", () => {
  const layer = S.createPropLayer((x, y) => x * 0.1 + y * 0.01);
  layer.sync([
    { id: "h", kind: "house", x: 2.5, y: 3.5, rot: 90, scale: 2 },
    { id: "w", kind: "well", x: 5.5, y: 5.5 },
  ]);
  const h = layer.objectOf("h");
  assert.ok(h);
  assert.deepEqual(h.position.toArray(), [2.5, 0.25 + 0.035, 3.5]);
  assert.ok(Math.abs(h.rotation.y - Math.PI / 2) < 1e-9);
  assert.equal(h.scale.x, 2);
  assert.equal(layer.idOf(meshes(h)[0]), "h");
  layer.sync([{ id: "w", kind: "well", x: 5.5, y: 5.5 }]);
  assert.equal(layer.objectOf("h"), undefined);
  assert.equal(layer.root.children.length, 1);
});

test("an imported model is kept until it is placed, released when its last prop goes, and redrawn on undo", async () => {
  const layer = S.createPropLayer(() => 0);
  await layer.addModel("sha256:boat", boxGlb());
  // Nothing uses it yet: an import must not be thrown away before the editor places it.
  layer.sync([]);
  assert.equal(layer.modelCount(), 1);
  layer.sync([model("m-1")]);
  const placed = layer.objectOf("m-1");
  assert.ok(placed);
  const geo = meshes(placed)[0].geometry;
  assert.equal(
    geo.getAttribute("position").count,
    8,
    "the model, not the placeholder box",
  );
  let disposed = 0;
  geo.addEventListener("dispose", () => disposed++);
  layer.sync([]);
  assert.equal(layer.modelCount(), 0);
  assert.equal(
    disposed,
    1,
    "the model's geometry is released with its last prop",
  );
  // Undo brings the prop back: the file is still at hand, so the model loads again.
  layer.sync([model("m-1")]);
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 50 && layer.modelCount() === 0; i++)
    await new Promise((r) => setTimeout(r, 20));
  assert.equal(layer.modelCount(), 1);
  assert.equal(
    meshes(
      layer.objectOf("m-1") as import("three").Object3D,
    )[0].geometry.getAttribute("position").count,
    8,
  );
  layer.dispose();
  assert.equal(layer.modelCount(), 0);
});

test("a model on glTF's default (fully metallic) material is drawn as a plain surface", () => {
  const plain = new three.MeshStandardMaterial({
    metalness: 1,
    color: "#ff0000",
  });
  const metal = new three.MeshStandardMaterial({ metalness: 1 });
  metal.metalnessMap = new three.Texture();
  const root = new three.Group();
  root.add(new three.Mesh(new three.BoxGeometry(), plain));
  root.add(new three.Mesh(new three.BoxGeometry(), metal));
  S.unmetal(root);
  assert.equal(plain.metalness, 0);
  assert.equal(plain.color.getHexString(), "ff0000", "colours are kept");
  assert.equal(metal.metalness, 1, "a model's own metal texture is respected");
});

test("a prop keeps its object when its model finishes loading, so a selection stays attached", async () => {
  const layer = S.createPropLayer(() => 0, { "sha256:boat": boxGlb() });
  layer.sync([model("m-1")]);
  const selected = layer.objectOf("m-1");
  for (let i = 0; i < 50 && layer.modelCount() === 0; i++)
    await new Promise((r) => setTimeout(r, 20));
  assert.equal(layer.modelCount(), 1);
  assert.equal(
    layer.objectOf("m-1"),
    selected,
    "the same object the gizmo holds",
  );
  assert.equal(
    meshes(selected as import("three").Object3D)[0].geometry.getAttribute(
      "position",
    ).count,
    8,
    "now drawing the model, not the placeholder",
  );
  assert.equal(
    layer.idOf(meshes(selected as import("three").Object3D)[0]),
    "m-1",
  );
  layer.dispose();
});

test("a saved world's models load from the files handed in; an unknown one stays a placeholder", async () => {
  const layer = S.createPropLayer(() => 0, { "sha256:boat": boxGlb() });
  layer.sync([model("m-1"), model("m-2", "sha256:missing")]);
  for (let i = 0; i < 50 && layer.modelCount() === 0; i++)
    await new Promise((r) => setTimeout(r, 20));
  assert.equal(layer.modelCount(), 1);
  const kept = meshes(layer.objectOf("m-2") as import("three").Object3D)[0];
  assert.ok((kept.material as import("three").MeshBasicMaterial).wireframe);
  assert.ok(three);
  layer.dispose();
});
