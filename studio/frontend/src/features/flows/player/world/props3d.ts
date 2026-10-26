// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { ModelAsset, WorldProp3d, WorldPropKind } from "../../types";
import type { Point, WorldMap } from "./procgen";

export const PROP_KINDS: readonly WorldPropKind[] = [
  "house",
  "tree",
  "lamp",
  "fence",
  "well",
  "crate",
  "rock",
  "bench",
  "stall",
];

export const PROP_LABELS: Record<WorldPropKind, string> = {
  house: "House",
  tree: "Tree",
  lamp: "Lamp post",
  fence: "Fence",
  well: "Well",
  crate: "Crate",
  rock: "Rock",
  bench: "Bench",
  stall: "Market stall",
  model: "Model",
};

export const OPEN_PROPS: ReadonlySet<WorldPropKind> = new Set([
  "lamp",
  "bench",
]);
export const MAX_PROPS = 400;
export const MAX_GLB_BYTES = 20 * 1024 * 1024;
export const MAX_TRIANGLES = 150_000;

export const propTile = (p: Pick<WorldProp3d, "x" | "y">): Point => ({
  x: Math.floor(p.x),
  y: Math.floor(p.y),
});

export function snapToTile(
  x: number,
  y: number,
  width: number,
  height: number,
): { x: number; y: number } {
  const tx = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const ty = Math.max(0, Math.min(height - 1, Math.floor(y)));
  return { x: tx + 0.5, y: ty + 0.5 };
}

export const snapRotation = (deg: number): number =>
  (((Math.round(deg / 15) * 15) % 360) + 360) % 360;

export const clampScale = (s: number): number =>
  Number.isFinite(s)
    ? Math.max(0.2, Math.min(5, Math.round(s * 100) / 100))
    : 1;

export function newPropId(taken: Iterable<string>, kind: string): string {
  const used = new Set(taken);
  for (let i = 1; ; i++) {
    const id = `${kind}-${i}`;
    if (!used.has(id)) return id;
  }
}

export function makeProp(
  kind: WorldPropKind,
  at: Point,
  taken: Iterable<string>,
  extra: Partial<WorldProp3d> = {},
): WorldProp3d {
  return {
    id: newPropId(taken, kind),
    kind,
    x: at.x + 0.5,
    y: at.y + 0.5,
    rot: 0,
    scale: 1,
    solid: !OPEN_PROPS.has(kind),
    ...extra,
  };
}

function reserved(map: WorldMap): Set<number> {
  const out = new Set<number>();
  const add = (p: Point | undefined) => {
    if (p) out.add(p.y * map.width + p.x);
  };
  add(map.spawn);
  for (const p of Object.values(map.npcs)) add(p);
  for (const p of Object.values(map.items)) add(p);
  return out;
}

export function blockedCells(
  map: WorldMap,
  props: readonly WorldProp3d[],
  open: readonly Point[] = [],
): Uint8Array {
  const out = new Uint8Array(map.width * map.height);
  const keep = reserved(map);
  for (const p of open) keep.add(p.y * map.width + p.x);
  for (const p of props) {
    if (p.solid === false) continue;
    const t = propTile(p);
    if (t.x < 0 || t.y < 0 || t.x >= map.width || t.y >= map.height) continue;
    const i = t.y * map.width + t.x;
    if (!keep.has(i)) out[i] = 1;
  }
  return out;
}

export function setMapProps(
  map: WorldMap,
  props: readonly WorldProp3d[],
  open: readonly Point[] = [],
): void {
  map.props = props.slice();
  map.blocked = props.length ? blockedCells(map, props, open) : undefined;
}

export function freeTileNear(
  map: WorldMap,
  from: Point,
  isFree: (x: number, y: number) => boolean,
): Point | null {
  const taken = new Set(
    (map.props ?? []).map((p) => {
      const t = propTile(p);
      return t.y * map.width + t.x;
    }),
  );
  const keep = reserved(map);
  for (let r = 0; r <= 12; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = from.x + dx;
        const y = from.y + dy;
        if (x < 1 || y < 1 || x >= map.width - 1 || y >= map.height - 1)
          continue;
        const i = y * map.width + x;
        if (!taken.has(i) && !keep.has(i) && isFree(x, y)) return { x, y };
      }
    }
  }
  return null;
}


export interface PlaceRequest {
  text: string;
  rows: string[];
  spawn: Point;
  focus: Point;
  taken: string[];
}

/** The map as the place solver reads it: `#` blocked, `R` walkable but kept clear, `.` free. */
export function placeRequest(
  map: WorldMap,
  props: readonly WorldProp3d[],
  isFree: (x: number, y: number) => boolean,
  text: string,
  focus: Point,
): PlaceRequest {
  const blocked = blockedCells(map, props);
  const keep = reserved(map);
  for (const p of props) {
    const t = propTile(p);
    keep.add(t.y * map.width + t.x);
  }
  const rows: string[] = [];
  for (let y = 0; y < map.height; y++) {
    let row = "";
    for (let x = 0; x < map.width; x++) {
      const i = y * map.width + x;
      row += blocked[i] || !isFree(x, y) ? "#" : keep.has(i) ? "R" : ".";
    }
    rows.push(row);
  }
  return {
    text: text.trim().slice(0, 600),
    rows,
    spawn: map.spawn,
    focus,
    taken: props.map((p) => p.id),
  };
}

export function mergePlaced(
  props: readonly WorldProp3d[],
  placed: unknown,
  map: Pick<WorldMap, "width" | "height">,
): WorldProp3d[] {
  const out = props.slice();
  const ids = new Set(props.map((p) => p.id));
  for (const raw of Array.isArray(placed) ? placed : []) {
    if (out.length >= MAX_PROPS) break;
    const p = raw as Partial<WorldProp3d> | null;
    if (
      !p ||
      !PROP_KINDS.includes(p.kind as WorldPropKind) ||
      typeof p.x !== "number" ||
      typeof p.y !== "number" ||
      !Number.isFinite(p.x) ||
      !Number.isFinite(p.y)
    )
      continue;
    const kind = p.kind as Exclude<WorldPropKind, "model">;
    const at = {
      x: Math.max(0, Math.min(map.width - 1, Math.floor(p.x))),
      y: Math.max(0, Math.min(map.height - 1, Math.floor(p.y))),
    };
    const prop = makeProp(kind, at, ids, {
      rot: snapRotation(Number(p.rot) || 0),
      ...(typeof p.solid === "boolean" ? { solid: p.solid } : {}),
      ...(typeof p.name === "string" && p.name
        ? { name: p.name.slice(0, 60) }
        : {}),
    });
    ids.add(prop.id);
    out.push(prop);
  }
  return out;
}

export interface PropHistory {
  props: WorldProp3d[];
  past: WorldProp3d[][];
  future: WorldProp3d[][];
}

const HISTORY = 100;

export const startHistory = (props: readonly WorldProp3d[]): PropHistory => ({
  props: props.slice(),
  past: [],
  future: [],
});

export function commit(h: PropHistory, next: WorldProp3d[]): PropHistory {
  if (JSON.stringify(next) === JSON.stringify(h.props)) return h;
  return {
    props: next.slice(0, MAX_PROPS),
    past: [...h.past, h.props].slice(-HISTORY),
    future: [],
  };
}

export function undo(h: PropHistory): PropHistory {
  const prev = h.past[h.past.length - 1];
  if (!prev) return h;
  return {
    props: prev,
    past: h.past.slice(0, -1),
    future: [h.props, ...h.future],
  };
}

export function redo(h: PropHistory): PropHistory {
  const next = h.future[0];
  if (!next) return h;
  return { props: next, past: [...h.past, h.props], future: h.future.slice(1) };
}

export function updateProp(
  props: readonly WorldProp3d[],
  id: string,
  patch: Partial<WorldProp3d>,
): WorldProp3d[] {
  return props.map((p) => (p.id === id ? { ...p, ...patch, id } : p));
}

export const removeProp = (
  props: readonly WorldProp3d[],
  id: string,
): WorldProp3d[] => props.filter((p) => p.id !== id);

// Binary glTF checks, the same as the backend's core/flows/glb.py: everything embedded, glTF 2.0, under the size
// and triangle budgets, no geometry or texture codec the player does not ship.

const UNSUPPORTED: Record<string, string> = {
  KHR_draco_mesh_compression: "Draco compression",
  EXT_meshopt_compression: "meshopt compression",
  KHR_texture_basisu: "KTX2 textures",
  EXT_mesh_gpu_instancing: "GPU instancing",
};

export class GlbError extends Error {}

interface GltfJson {
  buffers?: { uri?: unknown }[];
  images?: { uri?: unknown }[];
  accessors?: { count?: unknown }[];
  meshes?: {
    primitives?: {
      mode?: number;
      indices?: number;
      attributes?: Record<string, number>;
    }[];
  }[];
  nodes?: { mesh?: unknown }[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

export function countTriangles(doc: GltfJson): number {
  const acc = doc.accessors ?? [];
  const count = (i: unknown) => {
    const c = typeof i === "number" ? acc[i]?.count : undefined;
    return typeof c === "number" && c > 0 ? c : 0;
  };
  const perMesh = (doc.meshes ?? []).map((mesh) => {
    let total = 0;
    for (const prim of mesh?.primitives ?? []) {
      const mode = prim.mode ?? 4;
      const n =
        prim.indices !== undefined
          ? count(prim.indices)
          : count(prim.attributes?.POSITION);
      if (mode === 4) total += Math.floor(n / 3);
      else if (mode === 5 || mode === 6) total += Math.max(0, n - 2);
    }
    return total;
  });
  let placed = 0;
  for (const node of doc.nodes ?? []) {
    const m = node?.mesh;
    if (typeof m === "number" && m >= 0 && m < perMesh.length)
      placed += perMesh[m];
  }
  return Math.max(
    placed,
    perMesh.reduce((a, b) => a + b, 0),
  );
}

export function inspectGlb(buf: ArrayBuffer): {
  bytes: number;
  triangles: number;
} {
  const n = buf.byteLength;
  if (n > MAX_GLB_BYTES)
    throw new GlbError(
      `The model is ${(n / 2 ** 20).toFixed(1)} MB; models can be at most 20 MB`,
    );
  const dv = new DataView(buf);
  if (n < 20 || dv.getUint32(0, true) !== 0x46546c67)
    throw new GlbError("Not a binary glTF (.glb) model");
  const version = dv.getUint32(4, true);
  if (version !== 2)
    throw new GlbError(
      `glTF version ${version} is not supported; export glTF 2.0`,
    );
  if (dv.getUint32(8, true) !== n)
    throw new GlbError("The model file is cut short or padded");
  const len = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== 0x4e4f534a || 20 + len > n)
    throw new GlbError("The model has no glTF JSON chunk");
  let doc: GltfJson;
  try {
    doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, len)));
  } catch {
    throw new GlbError("The model's glTF JSON is not valid");
  }
  if (!doc || typeof doc !== "object")
    throw new GlbError("The model's glTF JSON is not valid");
  for (const entry of [...(doc.buffers ?? []), ...(doc.images ?? [])]) {
    if (entry && typeof entry === "object" && "uri" in entry)
      throw new GlbError(
        "The model points at outside files; export it as a single .glb with everything embedded",
      );
  }
  const used = new Set([
    ...(doc.extensionsRequired ?? []),
    ...(doc.extensionsUsed ?? []),
  ]);
  for (const [ext, what] of Object.entries(UNSUPPORTED))
    if (used.has(ext))
      throw new GlbError(
        `The model uses ${what}, which the 3D player cannot read; export without it`,
      );
  const triangles = countTriangles(doc);
  if (triangles > MAX_TRIANGLES)
    throw new GlbError(
      `The model has ${triangles.toLocaleString("en-US")} triangles; props can have at most ${MAX_TRIANGLES.toLocaleString("en-US")} (decimate it in Blender)`,
    );
  return { bytes: n, triangles };
}


export type GizmoMode = "translate" | "rotate" | "scale";

export interface EditorState {
  props: WorldProp3d[];
  selected: string | null;
  mode: GizmoMode;
  canUndo: boolean;
  canRedo: boolean;
}

export interface WorldEditor {
  state(): EditorState;
  subscribe(cb: () => void): () => void;
  add(kind: Exclude<WorldPropKind, "model">): void;
  addModel(asset: ModelAsset, data: ArrayBuffer, name?: string): Promise<void>;
  placeRequest(text: string): PlaceRequest;
  /** Props added, or -1 when the scene changed since `placeRequest`. */
  addPlaced(placed: unknown): number;
  select(id: string | null): void;
  setMode(mode: GizmoMode): void;
  update(id: string, patch: Partial<WorldProp3d>): void;
  remove(id?: string | null): void;
  undo(): void;
  redo(): void;
  exportGlb(): Promise<Blob>;
  close(): void;
}

export async function saveUntilCurrent(
  current: () => string,
  saved: string,
  save: () => Promise<boolean>,
): Promise<boolean> {
  for (let i = 0; i < 5; i++) {
    const now = current();
    if (now === saved) return true;
    if (!(await save())) return false;
    saved = now;
  }
  return current() === saved;
}

export class SaveQueue {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(job: () => Promise<T>): Promise<T> {
    const next = this.tail.then(job, job);
    this.tail = next.catch(() => undefined);
    return next;
  }

  idle(): Promise<unknown> {
    return this.tail;
  }
}
