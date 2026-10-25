// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The 3D editor's props in the scene: low-poly pieces built from primitives, plus imported GLB models (GLTFLoader is
// only fetched when a world has one). Part of the lazily loaded 3D chunk.

import * as three from "three";

import type { WorldProp3d, WorldPropKind } from "../../types";

type Part = [three.BufferGeometry, string, [number, number, number], number?];

/** Release every geometry, material and texture under `root`. */
export function disposeTree(root: three.Object3D): void {
  root.traverse((o) => {
    const mesh = o as three.Mesh;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material)
      ? mesh.material
      : mesh.material
        ? [mesh.material]
        : [];
    for (const m of mats) {
      for (const v of Object.values(m))
        if (v instanceof three.Texture) v.dispose();
      m.dispose();
    }
  });
}

export interface PropLayer {
  /** Every prop's object; exported with the scene and raycast by the editor. */
  root: three.Group;
  sync(props: readonly WorldProp3d[]): void;
  objectOf(id: string): three.Object3D | undefined;
  /** The prop an intersected object belongs to. */
  idOf(obj: three.Object3D | null): string | null;
  /** Hand over a model's file (an import, or one fetched for a saved world); loads it and redraws its props. */
  addModel(ref: string, data: ArrayBuffer): Promise<void>;
  /** Models currently held in memory (diagnostics and tests). */
  modelCount(): number;
  dispose(): void;
}

export function createPropLayer(
  groundAt: (x: number, y: number) => number,
  models: Record<string, ArrayBuffer> = {},
): PropLayer {
  const root = new three.Group();
  root.name = "props";
  // Geometry and materials are shared by every prop of a kind and released with the layer.
  const shared = new Map<string, { dispose(): void }>();
  const cached = <T extends { dispose(): void }>(
    key: string,
    make: () => T,
  ): T => {
    if (!shared.has(key)) shared.set(key, make());
    return shared.get(key) as T;
  };
  const geo = cached;
  const mat = (color: string, emissive?: string): three.Material =>
    cached(
      `m:${color}:${emissive ?? ""}`,
      () =>
        new three.MeshLambertMaterial({
          color,
          flatShading: true,
          ...(emissive ? { emissive, emissiveIntensity: 0.9 } : {}),
        }),
    );
  const box = (w: number, h: number, d: number) =>
    geo(`b:${w}:${h}:${d}`, () => new three.BoxGeometry(w, h, d));
  const cyl = (r0: number, r1: number, h: number, seg = 8) =>
    geo(
      `c:${r0}:${r1}:${h}:${seg}`,
      () => new three.CylinderGeometry(r0, r1, h, seg),
    );
  const cone = (r: number, h: number, seg: number) =>
    geo(`k:${r}:${h}:${seg}`, () => new three.ConeGeometry(r, h, seg));

  const parts: Record<Exclude<WorldPropKind, "model">, () => Part[]> = {
    house: () => [
      [box(0.9, 0.75, 0.9), "#f0e2c4", [0, 0.375, 0]],
      [cone(0.78, 0.6, 4), "#b5583f", [0, 1.05, 0], Math.PI / 4],
      [box(0.22, 0.38, 0.04), "#6b4a2f", [0, 0.19, 0.46]],
    ],
    tree: () => [
      [cyl(0.07, 0.11, 0.7, 6), "#6b4a2f", [0, 0.35, 0]],
      [
        geo("tree", () => new three.IcosahedronGeometry(0.55, 0)),
        "#3f8a3a",
        [0, 1.15, 0],
      ],
    ],
    lamp: () => [
      [cyl(0.035, 0.05, 1.4, 6), "#3b3b44", [0, 0.7, 0]],
      [box(0.16, 0.2, 0.16), "#ffd77a", [0, 1.48, 0]],
      [cone(0.14, 0.1, 4), "#3b3b44", [0, 1.63, 0], Math.PI / 4],
    ],
    fence: () => [
      [box(0.08, 0.55, 0.08), "#9a6b42", [-0.42, 0.275, 0]],
      [box(0.08, 0.55, 0.08), "#9a6b42", [0.42, 0.275, 0]],
      [box(0.96, 0.07, 0.05), "#b7845a", [0, 0.42, 0]],
      [box(0.96, 0.07, 0.05), "#b7845a", [0, 0.22, 0]],
    ],
    well: () => [
      [cyl(0.36, 0.38, 0.45, 10), "#8d8a83", [0, 0.225, 0]],
      [cyl(0.29, 0.29, 0.02, 10), "#2f5f8f", [0, 0.4, 0]],
      [box(0.06, 0.8, 0.06), "#6b4a2f", [-0.32, 0.6, 0]],
      [box(0.06, 0.8, 0.06), "#6b4a2f", [0.32, 0.6, 0]],
      [cone(0.5, 0.32, 4), "#a2533c", [0, 1.12, 0], Math.PI / 4],
    ],
    crate: () => [
      [box(0.5, 0.5, 0.5), "#b5834a", [0, 0.25, 0]],
      [box(0.52, 0.06, 0.52), "#8a5f33", [0, 0.48, 0]],
    ],
    rock: () => [
      [
        geo("rock", () => new three.DodecahedronGeometry(0.36, 0)),
        "#8d8a83",
        [0, 0.18, 0],
      ],
    ],
    bench: () => [
      [box(0.8, 0.06, 0.28), "#9a6b42", [0, 0.32, 0]],
      [box(0.8, 0.22, 0.05), "#9a6b42", [0, 0.5, -0.12]],
      [box(0.06, 0.3, 0.26), "#5a4030", [-0.34, 0.15, 0]],
      [box(0.06, 0.3, 0.26), "#5a4030", [0.34, 0.15, 0]],
    ],
    stall: () => [
      [box(0.9, 0.45, 0.45), "#b7845a", [0, 0.225, 0.15]],
      [box(0.05, 1.1, 0.05), "#6b4a2f", [-0.42, 0.55, -0.25]],
      [box(0.05, 1.1, 0.05), "#6b4a2f", [0.42, 0.55, -0.25]],
      [box(0.05, 0.95, 0.05), "#6b4a2f", [-0.42, 0.475, 0.35]],
      [box(0.05, 0.95, 0.05), "#6b4a2f", [0.42, 0.475, 0.35]],
      [box(1.0, 0.05, 0.75), "#d9534f", [0, 1.05, 0.05]],
      [box(0.3, 0.12, 0.25), "#f2c84b", [-0.2, 0.51, 0.15]],
      [box(0.25, 0.1, 0.22), "#79b85a", [0.2, 0.5, 0.15]],
    ],
  };

  const build = (kind: Exclude<WorldPropKind, "model">) => {
    const g = new three.Group();
    for (const [gg, color, [x, y, z], ry] of parts[kind]()) {
      const m = new three.Mesh(
        gg,
        mat(
          color,
          kind === "lamp" && color === "#ffd77a" ? "#ffb43a" : undefined,
        ),
      );
      m.position.set(x, y, z);
      if (ry) m.rotation.y = ry;
      g.add(m);
    }
    return g;
  };

  // Imported models: one parsed template per file, normalised to stand on the ground inside one tile; each prop is a
  // clone that shares the template's geometry and materials.
  const templates = new Map<string, three.Object3D>();
  const loading = new Map<string, Promise<void>>();
  // Model files stay here after their GPU copy is released, so undoing a delete can draw the model again.
  const pending: Record<string, ArrayBuffer> = { ...models };
  // Imported but not placed yet: kept even though no prop uses them so far.
  const fresh = new Set<string>();
  let last: readonly WorldProp3d[] = [];
  let disposed = false;
  const placeholder = () => {
    const g = new three.Group();
    const m = new three.Mesh(
      box(0.8, 0.8, 0.8),
      cached(
        "m:placeholder",
        () =>
          new three.MeshBasicMaterial({ color: "#8f7ae6", wireframe: true }),
      ),
    );
    m.position.y = 0.4;
    g.add(m);
    return g;
  };

  const load = (ref: string, data: ArrayBuffer): Promise<void> => {
    const known = loading.get(ref);
    if (known) return known;
    const job = (async () => {
      const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
      const gltf = await new GLTFLoader().parseAsync(data, "");
      if (disposed) {
        disposeTree(gltf.scene);
        return;
      }
      const model = gltf.scene;
      const bounds = new three.Box3().setFromObject(model);
      const size = bounds.getSize(new three.Vector3());
      const fit = 1 / Math.max(1e-6, size.x, size.z, size.y * 0.5);
      const holder = new three.Group();
      model.position.set(
        -(bounds.min.x + size.x / 2) * fit,
        -bounds.min.y * fit,
        -(bounds.min.z + size.z / 2) * fit,
      );
      model.scale.setScalar(fit);
      holder.add(model);
      templates.set(ref, holder);
      layer.sync(last);
    })();
    loading.set(ref, job);
    job.catch(() => loading.delete(ref));
    return job;
  };

  const objects = new Map<string, { obj: three.Object3D; key: string }>();
  const keyOf = (p: WorldProp3d) =>
    p.kind === "model"
      ? `model:${p.asset?.ref}:${templates.has(p.asset?.ref ?? "") ? 1 : 0}`
      : p.kind;

  const layer: PropLayer = {
    root,
    sync(props) {
      last = props;
      const live = new Set<string>();
      for (const p of props) {
        live.add(p.id);
        const key = keyOf(p);
        let cur = objects.get(p.id);
        if (!cur || cur.key !== key) {
          let shape: three.Object3D;
          if (p.kind === "model") {
            const ref = p.asset?.ref ?? "";
            const tpl = templates.get(ref);
            shape = tpl ? tpl.clone(true) : placeholder();
            if (!tpl && pending[ref])
              void load(ref, pending[ref]).catch(() => undefined);
          } else shape = build(p.kind);
          // Each prop keeps one outer object for its life; only what it holds is swapped (a model replacing its
          // placeholder), so the gizmo stays attached to what the player sees.
          let obj = cur?.obj;
          if (obj) obj.clear();
          else {
            obj = new three.Group();
            obj.userData.propId = p.id;
            root.add(obj);
          }
          obj.add(shape);
          cur = { obj, key };
          objects.set(p.id, cur);
        }
        const s = p.scale ?? 1;
        cur.obj.position.set(p.x, groundAt(p.x, p.y), p.y);
        cur.obj.rotation.set(0, ((p.rot ?? 0) * Math.PI) / 180, 0);
        cur.obj.scale.setScalar(s);
      }
      for (const [id, cur] of objects) {
        if (!live.has(id)) {
          cur.obj.removeFromParent();
          objects.delete(id);
        }
      }
      // A model no prop uses any more is released (its clones only shared these buffers).
      const used = new Set(
        props.filter((p) => p.kind === "model").map((p) => p.asset?.ref),
      );
      for (const ref of used) if (ref) fresh.delete(ref);
      for (const [ref, tpl] of templates) {
        if (!used.has(ref) && !fresh.has(ref)) {
          disposeTree(tpl);
          templates.delete(ref);
          loading.delete(ref);
        }
      }
    },
    objectOf(id) {
      return objects.get(id)?.obj;
    },
    idOf(obj) {
      for (let o: three.Object3D | null = obj; o; o = o.parent) {
        if (typeof o.userData.propId === "string") return o.userData.propId;
        if (o === root) break;
      }
      return null;
    },
    addModel(ref, data) {
      pending[ref] = data;
      fresh.add(ref);
      return load(ref, data);
    },
    modelCount() {
      return templates.size;
    },
    dispose() {
      disposed = true;
      root.removeFromParent();
      for (const tpl of templates.values()) disposeTree(tpl);
      templates.clear();
      for (const ref of Object.keys(pending)) delete pending[ref];
      for (const r of shared.values()) r.dispose();
      shared.clear();
      objects.clear();
    },
  };
  return layer;
}
