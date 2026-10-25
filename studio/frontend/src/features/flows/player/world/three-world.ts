// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Loaded only via import() so three.js stays out of the startup bundle; same WorldGameHandle / WorldBridge as Phaser.

import * as three from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

import type { WorldItem, WorldNpc, WorldSpec } from "../../types";
import type {
  Area,
  Interactable,
  WorldArt,
  WorldBridge,
  WorldGameHandle,
} from "./phaser-game";
import {
  type Point,
  SOLID,
  type TileKind,
  type WorldMap,
  findPath,
  isSolid,
  kindAt,
} from "./procgen";
import { createPropLayer } from "./props-scene3d";
import { type WorldEditor, setMapProps } from "./props3d";
import { canvas, fitImage, paintItem } from "./sprites";
import type { WorldEditorRuntime } from "./world-editor3d";
import {
  TILE_COLORS,
  TIMES,
  type Terrain,
  buildTerrain,
  decorCells,
  fogRange,
  heightAt,
  houseCells,
  resolveView3d,
  slide,
  standTiles,
  waterLevel,
} from "./world3d";

const NPC_TILES_PER_S = 1.5;
const REACH = 1.35;
const TURN_RAD_PER_S = 2.2;
const ACTOR_H = 1.15;

interface Npc {
  id: string;
  name: string;
  group: three.Group;
  label: three.Sprite;
  status: string;
  pos: { x: number; y: number };
  home: Point;
  path: Point[];
  rest: number;
  phase: number;
  area: Area | null;
  busy: boolean;
  wander: boolean;
  speech: { sprite: three.Sprite; until: number } | null;
}

interface Item {
  id: string;
  name: string;
  sprite: three.Sprite;
  at: Point;
  base: number;
  gone: number;
}

function hash2(x: number, y: number, seed: number): number {
  let h = (x * 374761393 + y * 668265263 + seed * 2246822519) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function textCanvas(
  text: string,
  opts: { bubble?: boolean; px?: number; maxW?: number } = {},
): HTMLCanvasElement {
  const px = opts.px ?? 44;
  const maxW = opts.maxW ?? 900;
  const font = `600 ${px}px Inter, system-ui, sans-serif`;
  const [, probe] = canvas(4, 4);
  probe.font = font;
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (probe.measureText(next).width > maxW && line) {
      lines.push(line);
      line = w;
    } else line = next;
  }
  if (line) lines.push(line);
  const pad = px * 0.5;
  const w = Math.min(
    maxW,
    Math.max(...lines.map((l) => probe.measureText(l).width)),
  );
  const lh = px * 1.25;
  const tail = opts.bubble ? px * 0.5 : 0;
  const [c, ctx] = canvas(w + pad * 2, lines.length * lh + pad * 2 + tail);
  ctx.font = font;
  ctx.fillStyle = opts.bubble
    ? "rgba(255,255,255,0.95)"
    : "rgba(17,19,28,0.72)";
  const r = px * 0.5;
  const bh = c.height - tail;
  ctx.beginPath();
  ctx.roundRect(0, 0, c.width, bh, r);
  ctx.fill();
  if (opts.bubble) {
    ctx.beginPath();
    ctx.moveTo(c.width / 2 - tail, bh);
    ctx.lineTo(c.width / 2 + tail, bh);
    ctx.lineTo(c.width / 2, c.height);
    ctx.fill();
  }
  ctx.fillStyle = opts.bubble ? "#1b1b28" : "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  lines.forEach((l, i) => ctx.fillText(l, c.width / 2, pad + lh * (i + 0.5)));
  return c;
}

function meanColor(img: HTMLImageElement): three.Color | null {
  try {
    const [, ctx] = canvas(8, 8);
    ctx.drawImage(img, 0, 0, 8, 8);
    const d = ctx.getImageData(0, 0, 8, 8).data;
    let r = 0;
    let g = 0;
    let b = 0;
    for (let i = 0; i < d.length; i += 4) {
      r += d[i];
      g += d[i + 1];
      b += d[i + 2];
    }
    const n = d.length / 4;
    return new three.Color().setRGB(
      r / n / 255,
      g / n / 255,
      b / n / 255,
      three.SRGBColorSpace,
    );
  } catch {
    return null;
  }
}

export function createWorld3d(
  parent: HTMLElement,
  spec: WorldSpec,
  map: WorldMap,
  art: WorldArt,
  picked: string[],
  start: Point | null,
  bridge: WorldBridge,
): WorldGameHandle {
  const view = resolveView3d(spec);
  const light = TIMES[view.time_of_day];
  const terrain: Terrain = buildTerrain(map, spec.seed, view.terrain);
  const disposables: { dispose(): void }[] = [];
  const keep = <T extends { dispose(): void }>(x: T): T => {
    disposables.push(x);
    return x;
  };
  const drop = (x: { dispose(): void } | null | undefined) => {
    if (!x) return;
    x.dispose();
    const i = disposables.indexOf(x);
    if (i >= 0) disposables.splice(i, 1);
  };
  const dropSprite = (sprite: three.Sprite) => {
    sprite.removeFromParent();
    const mat = sprite.material as three.SpriteMaterial;
    drop(mat.map);
    drop(mat);
  };

  const renderer = new three.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.dataset.testid = "world3d-canvas";
  parent.appendChild(renderer.domElement);

  const scene = new three.Scene();
  const sky = new three.Color(view.sky);
  scene.background = sky;
  const [fogNear, fogFar] = fogRange(view.fog);
  scene.fog = new three.Fog(sky, fogNear, fogFar);
  scene.add(
    new three.HemisphereLight(light.hemi[0], light.hemi[1], light.hemi[2]),
  );
  const sun = new three.DirectionalLight(light.sun[0], light.sun[1]);
  const elev = (light.sun[2] * Math.PI) / 180;
  sun.position.set(Math.cos(elev) * 0.6, Math.sin(elev), Math.cos(elev) * 0.8);
  scene.add(sun);
  let skyDome: three.Mesh | null = null;
  {
    const dome = keep(new three.SphereGeometry(300, 24, 12));
    const zenith = sky.clone().multiplyScalar(0.62);
    const horizon = sky
      .clone()
      .lerp(
        new three.Color("#ffffff"),
        view.time_of_day === "night" ? 0.05 : 0.22,
      );
    const p = dome.getAttribute("position");
    const c = new Float32Array(p.count * 3);
    const tmp = new three.Color();
    for (let i = 0; i < p.count; i++) {
      const t = Math.max(0, p.getY(i) / 300);
      tmp.copy(horizon).lerp(zenith, Math.sqrt(t));
      c.set([tmp.r, tmp.g, tmp.b], i * 3);
    }
    dome.setAttribute("color", new three.BufferAttribute(c, 3));
    skyDome = new three.Mesh(
      dome,
      keep(
        new three.MeshBasicMaterial({
          vertexColors: true,
          side: three.BackSide,
          fog: false,
          depthWrite: false,
        }),
      ),
    );
    skyDome.renderOrder = -1;
    (scene.fog as three.Fog).color.copy(horizon);
    scene.add(skyDome);
  }

  const world = new three.Group();
  world.name = "world";
  scene.add(world);

  const camera = new three.PerspectiveCamera(72, 1, 0.05, 400);
  camera.rotation.order = "YXZ";

  const tint: Partial<Record<TileKind, three.Color>> = {};
  for (const [kind, img] of Object.entries(art.tiles)) {
    const c = img && meanColor(img);
    if (c) tint[kind as TileKind] = c;
  }
  const W = map.width;
  const H = map.height;
  const R = W + 1;
  const pos = new Float32Array(W * H * 18);
  const col = new Float32Array(W * H * 18);
  const base = new three.Color();
  const waterTiles: Point[] = [];
  let v = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const kind = kindAt(map, x, y);
      if (kind === "water") waterTiles.push({ x, y });
      const groundKind: TileKind =
        kind === "tree" || kind === "rock" || kind === "flowers"
          ? spec.biome === "desert" || spec.biome === "beach"
            ? "sand"
            : spec.biome === "snow"
              ? "snow"
              : "grass"
          : kind === "wall"
            ? spec.biome === "dungeon"
              ? "floor"
              : "path"
            : kind;
      base.set(TILE_COLORS[groundKind]);
      const t = tint[groundKind];
      if (t) base.lerp(t, 0.55);
      const j = 0.94 + hash2(x, y, spec.seed) * 0.12;
      const corners: [number, number][] = [
        [x, y],
        [x, y + 1],
        [x + 1, y],
        [x + 1, y],
        [x, y + 1],
        [x + 1, y + 1],
      ];
      for (const [cx, cy] of corners) {
        pos[v] = cx;
        pos[v + 1] = terrain.corners[cy * R + cx];
        pos[v + 2] = cy;
        col[v] = base.r * j;
        col[v + 1] = base.g * j;
        col[v + 2] = base.b * j;
        v += 3;
      }
    }
  }
  const groundGeo = keep(new three.BufferGeometry());
  groundGeo.setAttribute("position", new three.BufferAttribute(pos, 3));
  groundGeo.setAttribute("color", new three.BufferAttribute(col, 3));
  groundGeo.computeVertexNormals();
  const groundMat = keep(new three.MeshLambertMaterial({ vertexColors: true }));
  world.add(new three.Mesh(groundGeo, groundMat));

  if (waterTiles.length) {
    const wpos = new Float32Array(waterTiles.length * 18);
    let i = 0;
    for (const p of waterTiles) {
      const h = waterLevel(terrain, p.x, p.y);
      for (const [cx, cy] of [
        [p.x, p.y],
        [p.x, p.y + 1],
        [p.x + 1, p.y],
        [p.x + 1, p.y],
        [p.x, p.y + 1],
        [p.x + 1, p.y + 1],
      ]) {
        wpos[i++] = cx;
        wpos[i++] = h;
        wpos[i++] = cy;
      }
    }
    const wg = keep(new three.BufferGeometry());
    wg.setAttribute("position", new three.BufferAttribute(wpos, 3));
    wg.computeVertexNormals();
    const wc = new three.Color(TILE_COLORS.water);
    if (tint.water) wc.lerp(tint.water, 0.5);
    const wm = keep(
      new three.MeshLambertMaterial({
        color: wc,
        transparent: true,
        opacity: 0.85,
      }),
    );
    world.add(new three.Mesh(wg, wm));
  }

  const decor = decorCells(map);
  const houses = houseCells(spec, map);
  const houseAt = new Set(houses.map((p) => `${p.x},${p.y}`));
  const m4 = new three.Matrix4();
  const q = new three.Quaternion();
  const up = new three.Vector3(0, 1, 0);
  const instanced = (
    geo: three.BufferGeometry,
    mat: three.Material,
    cells: Point[],
    place: (p: Point, r: number) => [number, number, number, number], // y offset, sx, sy, rot
  ) => {
    if (!cells.length) return null;
    const mesh = new three.InstancedMesh(keep(geo), keep(mat), cells.length);
    cells.forEach((p, i) => {
      const r = hash2(p.x, p.y, spec.seed + 7);
      const [dy, sx, sy, rot] = place(p, r);
      const gx = p.x + 0.5 + (r - 0.5) * 0.25;
      const gz = p.y + 0.5 + (hash2(p.y, p.x, spec.seed) - 0.5) * 0.25;
      q.setFromAxisAngle(up, rot);
      m4.compose(
        new three.Vector3(gx, heightAt(terrain, gx, gz) + dy * sy, gz),
        q,
        new three.Vector3(sx, sy, sx),
      );
      mesh.setMatrixAt(i, m4);
    });
    mesh.instanceMatrix.needsUpdate = true;
    world.add(mesh);
    return mesh;
  };
  const trees = decor.tree.filter((p) => !houseAt.has(`${p.x},${p.y}`));
  const rocks = decor.rock.filter((p) => !houseAt.has(`${p.x},${p.y}`));
  const conifer = spec.biome === "snow" || spec.biome === "forest";
  const treeTint = tint.tree ?? null;
  const leaf = new three.Color(
    spec.biome === "snow"
      ? "#3f7a5a"
      : spec.biome === "desert"
        ? "#7c9a3e"
        : "#3f8a3a",
  );
  if (treeTint) leaf.lerp(treeTint, 0.5);
  const treeSize = (r: number) => 0.85 + r * 0.45;
  instanced(
    new three.CylinderGeometry(0.07, 0.11, 0.7, 6),
    new three.MeshLambertMaterial({ color: "#6b4a2f" }),
    trees,
    (_p, r) => [0.35, treeSize(r), treeSize(r), r * 6],
  );
  instanced(
    conifer
      ? new three.ConeGeometry(0.48, 1.4, 7)
      : new three.IcosahedronGeometry(0.55, 0),
    new three.MeshLambertMaterial({ color: leaf, flatShading: true }),
    trees,
    (_p, r) => [conifer ? 1.25 : 1.15, treeSize(r), treeSize(r), r * 6],
  );
  const rockColor = new three.Color(TILE_COLORS.rock);
  if (tint.rock) rockColor.lerp(tint.rock, 0.5);
  instanced(
    new three.DodecahedronGeometry(0.36, 0),
    new three.MeshLambertMaterial({ color: rockColor, flatShading: true }),
    rocks,
    (_p, r) => [0.12, 0.8 + r * 0.6, 0.55 + r * 0.4, r * 6],
  );
  const dungeon = spec.biome === "dungeon";
  const wallColor = new three.Color(dungeon ? "#4d4957" : "#e2d3b6");
  if (tint.wall) wallColor.lerp(tint.wall, 0.5);
  const wallH = dungeon ? 2.6 : 1.7;
  instanced(
    new three.BoxGeometry(1, 1, 1),
    new three.MeshLambertMaterial({ color: wallColor }),
    decor.wall,
    () => [0.5, 1.001, wallH, 0],
  );
  if (!dungeon) {
    const seen = new Set<string>();
    const roofMat = keep(
      new three.MeshLambertMaterial({ color: "#a2533c", flatShading: true }),
    );
    for (const s of decor.wall) {
      if (seen.has(`${s.x},${s.y}`)) continue;
      let x0 = s.x;
      let x1 = s.x;
      let y0 = s.y;
      let y1 = s.y;
      const stack = [s];
      seen.add(`${s.x},${s.y}`);
      while (stack.length) {
        const p = stack.pop() as Point;
        x0 = Math.min(x0, p.x);
        x1 = Math.max(x1, p.x);
        y0 = Math.min(y0, p.y);
        y1 = Math.max(y1, p.y);
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const n = { x: p.x + dx, y: p.y + dy };
          const k = `${n.x},${n.y}`;
          if (
            !seen.has(k) &&
            n.x >= 0 &&
            n.y >= 0 &&
            n.x < W &&
            n.y < H &&
            kindAt(map, n.x, n.y) === "wall"
          ) {
            seen.add(k);
            stack.push(n);
          }
        }
      }
      const bw = x1 - x0 + 1;
      const bd = y1 - y0 + 1;
      if (bw > 10 || bd > 10) continue;
      const roof = new three.Mesh(
        keep(new three.ConeGeometry(Math.SQRT1_2, 1, 4)),
        roofMat,
      );
      const cx = x0 + bw / 2;
      const cz = y0 + bd / 2;
      roof.rotation.y = Math.PI / 4;
      roof.scale.set(bw + 0.4, 1.3, bd + 0.4);
      roof.position.set(cx, heightAt(terrain, cx, cz) + wallH + 0.65, cz);
      world.add(roof);
    }
  }
  if (houses.length) {
    const body = keep(new three.BoxGeometry(0.9, 0.75, 0.9));
    const top = keep(new three.ConeGeometry(0.78, 0.6, 4));
    const bodyMat = keep(new three.MeshLambertMaterial({ color: "#f0e2c4" }));
    const topMat = keep(
      new three.MeshLambertMaterial({ color: "#b5583f", flatShading: true }),
    );
    for (const p of houses) {
      const g = new three.Group();
      const b = new three.Mesh(body, bodyMat);
      b.position.y = 0.375;
      const t = new three.Mesh(top, topMat);
      t.position.y = 1.05;
      t.rotation.y = Math.PI / 4;
      g.add(b, t);
      g.position.set(
        p.x + 0.5,
        heightAt(terrain, p.x + 0.5, p.y + 0.5),
        p.y + 0.5,
      );
      g.rotation.y = Math.floor(hash2(p.x, p.y, spec.seed) * 4) * (Math.PI / 2);
      world.add(g);
    }
  }
  if (decor.flowers.length) {
    const petals = ["#f2c84b", "#ef7aa8", "#ffffff", "#b58ce8"];
    const fl = instanced(
      new three.IcosahedronGeometry(0.07, 0),
      new three.MeshLambertMaterial({ color: "#ffffff" }),
      decor.flowers,
      (_p, r) => [0.08, 1 + r, 1 + r, 0],
    );
    if (fl)
      decor.flowers.forEach((p, i) =>
        fl.setColorAt(
          i,
          new three.Color(
            petals[Math.floor(hash2(p.x, p.y, 3) * petals.length)],
          ),
        ),
      );
  }

  const props = createPropLayer(
    (x, y) => heightAt(terrain, x, y),
    art.models ?? {},
  );
  world.add(props.root);
  props.sync(map.props ?? []);
  let editor: WorldEditorRuntime | null = null;
  let opening: Promise<WorldEditor> | null = null;
  let destroyed = false;

  const texOf = (c: HTMLCanvasElement) => {
    const t = keep(new three.CanvasTexture(c));
    t.colorSpace = three.SRGBColorSpace;
    return t;
  };
  const tagSprite = (text: string, height: number, bubble = false) => {
    const c = textCanvas(text, { bubble });
    const s = new three.Sprite(
      keep(
        new three.SpriteMaterial({
          map: texOf(c),
          depthTest: !bubble,
          transparent: true,
        }),
      ),
    );
    s.scale.set((height * c.width) / c.height, height, 1);
    s.center.set(0.5, 0);
    s.renderOrder = bubble ? 10 : 5;
    return s;
  };
  const setTag = (
    s: three.Sprite,
    text: string,
    height: number,
    bubble = false,
  ) => {
    const c = textCanvas(text, { bubble });
    const mat = s.material as three.SpriteMaterial;
    drop(mat.map);
    mat.map = texOf(c);
    mat.needsUpdate = true;
    s.scale.set((height * c.width) / c.height, height, 1);
  };

  const npcs: Npc[] = [];
  const spawnNpc = (n: WorldNpc, at: Point) => {
    const g = new three.Group();
    const img = art.sprites[n.id];
    if (img) {
      const c = fitImage(img, 192, 256);
      const s = new three.Sprite(
        keep(
          new three.SpriteMaterial({
            map: texOf(c),
            transparent: true,
            alphaTest: 0.1,
          }),
        ),
      );
      s.center.set(0.5, 0);
      s.scale.set(ACTOR_H * 0.75, ACTOR_H, 1);
      g.add(s);
    } else {
      const hue = hash2(
        n.id.length,
        n.id.charCodeAt(0),
        n.id.charCodeAt(n.id.length - 1),
      );
      const m = keep(
        new three.MeshLambertMaterial({
          color: new three.Color().setHSL(hue, 0.55, 0.55),
        }),
      );
      const cap = new three.Mesh(
        keep(new three.CapsuleGeometry(0.2, 0.5, 4, 10)),
        m,
      );
      cap.position.y = 0.45;
      const head = new three.Mesh(
        keep(new three.SphereGeometry(0.15, 12, 10)),
        keep(new three.MeshLambertMaterial({ color: "#f1c9a5" })),
      );
      head.position.y = 0.98;
      g.add(cap, head);
    }
    const label = tagSprite(n.name, 0.16);
    label.position.y = ACTOR_H + 0.12;
    g.add(label);
    scene.add(g);
    npcs.push({
      id: n.id,
      name: n.name,
      group: g,
      label,
      status: "",
      pos: { x: at.x + 0.5, y: at.y + 0.5 },
      home: at,
      path: [],
      rest: 1 + Math.random() * 3,
      phase: Math.random() * 10,
      area: null,
      busy: false,
      wander: n.wander !== false,
      speech: null,
    });
  };
  for (const n of spec.npcs) {
    const at = map.npcs[n.id];
    if (at) spawnNpc(n, at);
  }
  const items = new Map<string, Item>();
  const spawnItem = (it: WorldItem, at: Point) => {
    const img = art.sprites[it.id];
    const c = img ? fitImage(img, 128, 128) : paintItem(it.id, 128);
    const s = new three.Sprite(
      keep(
        new three.SpriteMaterial({
          map: texOf(c),
          transparent: true,
          alphaTest: 0.1,
        }),
      ),
    );
    s.center.set(0.5, 0);
    s.scale.set(0.42, 0.42, 1);
    const b = heightAt(terrain, at.x + 0.5, at.y + 0.5) + 0.15;
    s.position.set(at.x + 0.5, b, at.y + 0.5);
    scene.add(s);
    items.set(it.id, {
      id: it.id,
      name: it.name,
      sprite: s,
      at,
      base: b,
      gone: 0,
    });
  };
  for (const it of spec.items) {
    const at = map.items[it.id];
    if (at && !picked.includes(it.id)) spawnItem(it, at);
  }
  // Signs nothing walks into or picks up (a home a villager built).
  const markers = new Map<string, three.Sprite>();
  const prompt = tagSprite("E", 0.14);
  prompt.visible = false;
  scene.add(prompt);

  const spawn = start && !isSolid(map, start.x, start.y) ? start : map.spawn;
  const me = { x: spawn.x + 0.5, y: spawn.y + 0.5 };
  {
    let best: Npc | null = null;
    for (const n of npcs)
      if (
        !best ||
        Math.hypot(n.pos.x - me.x, n.pos.y - me.y) <
          Math.hypot(best.pos.x - me.x, best.pos.y - me.y)
      )
        best = n;
    camera.rotation.y = best
      ? Math.atan2(-(best.pos.x - me.x), -(best.pos.y - me.y))
      : 0;
  }
  const controls = new PointerLockControls(camera, renderer.domElement);
  controls.minPolarAngle = 0.4;
  controls.maxPolarAngle = Math.PI - 0.4;

  const overlay = document.createElement("button");
  overlay.type = "button";
  overlay.dataset.testid = "world3d-lock";
  overlay.textContent =
    "Click to look around · WASD to walk · E to talk · Esc for the mouse";
  Object.assign(overlay.style, {
    position: "absolute",
    left: "50%",
    bottom: "84px",
    transform: "translateX(-50%)",
    padding: "10px 18px",
    borderRadius: "999px",
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(10,12,20,0.66)",
    color: "white",
    font: "500 13px Inter, system-ui, sans-serif",
    cursor: "pointer",
    zIndex: "5",
    backdropFilter: "blur(6px)",
  });
  const cross = document.createElement("div");
  Object.assign(cross.style, {
    position: "absolute",
    left: "50%",
    top: "50%",
    width: "6px",
    height: "6px",
    margin: "-3px 0 0 -3px",
    borderRadius: "50%",
    background: "rgba(255,255,255,0.85)",
    boxShadow: "0 0 0 1.5px rgba(0,0,0,0.35)",
    pointerEvents: "none",
    zIndex: "5",
    display: "none",
  });
  if (getComputedStyle(parent).position === "static")
    parent.style.position = "relative";
  parent.append(overlay, cross);
  let dragLook = false;
  const requestLock = () => {
    const el = renderer.domElement as HTMLCanvasElement & {
      requestPointerLock(): Promise<void> | void;
    };
    try {
      const r = el.requestPointerLock();
      if (r && typeof (r as Promise<void>).catch === "function")
        (r as Promise<void>).catch(() => {
          dragLook = true;
        });
    } catch {
      dragLook = true;
    }
    // No lock a moment later (headless browsers, iframes without the permission): drag to look instead.
    window.setTimeout(() => {
      if (!controls.isLocked) dragLook = true;
    }, 400);
  };
  overlay.addEventListener("click", requestLock);

  const keys = new Set<string>();
  let route: Point[] = [];
  let pendingTalk: string | null = null;
  let talking: string | null = null;
  let nearby: Interactable | null = null;
  let region: string | null = null;
  let regionTimer = 0;
  let walkPhase = 0;
  let faceYaw: number | null = null;
  let drag: { x: number; y: number } | null = null;

  const typing = (e: Event) => {
    const t = e.target as HTMLElement | null;
    return Boolean(
      t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable),
    );
  };
  const interact = (target: Interactable) => {
    route = [];
    pendingTalk = null;
    keys.clear();
    bridge.onInteract(target);
  };
  const onKeyDown = (e: KeyboardEvent) => {
    if (typing(e) || bridge.locked()) return;
    if (e.code === "KeyE" || e.code === "Space") {
      if (nearby) {
        e.preventDefault();
        interact(nearby);
      }
      return;
    }
    if (/^(Key[WASD]|Arrow(Up|Down|Left|Right))$/.test(e.code)) {
      keys.add(e.code);
      route = [];
      pendingTalk = null;
      e.preventDefault();
    }
  };
  const onKeyUp = (e: KeyboardEvent) => {
    keys.delete(e.code);
  };
  const onBlur = () => keys.clear();
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", onBlur);

  const raycaster = new three.Raycaster();
  const onPointerDown = (e: PointerEvent) => {
    if (controls.isLocked || bridge.locked()) return;
    drag = { x: e.clientX, y: e.clientY };
    const rect = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(
      new three.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      ),
      camera,
    );
    const hit = raycaster.intersectObjects(
      npcs.map((n) => n.group),
      true,
    )[0];
    const npc =
      hit &&
      npcs.find((n) => n.group === hit.object.parent || n.group === hit.object);
    if (npc) {
      pendingTalk = npc.id;
      if (Math.hypot(npc.pos.x - me.x, npc.pos.y - me.y) <= REACH)
        interact({ kind: "npc", id: npc.id, name: npc.name });
      else route = findPath(map, tileOf(me), tileOf(npc.pos));
    }
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!drag || controls.isLocked || !dragLook) return;
    camera.rotation.y -= (e.clientX - drag.x) * 0.005;
    camera.rotation.x = Math.max(
      -1.1,
      Math.min(1.1, camera.rotation.x - (e.clientY - drag.y) * 0.004),
    );
    drag = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = () => {
    drag = null;
  };
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);

  const resize = () => {
    const w = Math.max(1, parent.clientWidth);
    const h = Math.max(1, parent.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(parent);

  function tileOf(p: { x: number; y: number }): Point {
    return { x: Math.floor(p.x), y: Math.floor(p.y) };
  }
  function steer(
    p: { x: number; y: number },
    path: Point[],
    speed: number,
    dt: number,
    solid: boolean,
  ): [number, number] | null {
    while (path.length) {
      const tx = path[0].x + 0.5;
      const ty = path[0].y + 0.5;
      const dx = tx - p.x;
      const dy = ty - p.y;
      const d = Math.hypot(dx, dy);
      if (d < 0.08) {
        path.shift();
        continue;
      }
      const step = Math.min(d, speed * dt);
      const mx = (dx / d) * step;
      const my = (dy / d) * step;
      if (solid) [p.x, p.y] = slide(map, p.x, p.y, mx, my);
      else {
        p.x += mx;
        p.y += my;
      }
      return [dx / d, dy / d];
    }
    return null;
  }
  const blockedByNpc = (x: number, y: number) =>
    npcs.some(
      (n) =>
        Math.hypot(n.pos.x - x, n.pos.y - y) < 0.42 &&
        Math.hypot(n.pos.x - me.x, n.pos.y - me.y) >= 0.42,
    );

  let last = performance.now();
  let raf = 0;
  const frame = (now: number) => {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(now - last, 50) / 1000;
    last = now;
    const locked = bridge.locked();
    if (locked && controls.isLocked) controls.unlock();
    overlay.style.display =
      !locked && !controls.isLocked && !dragLook ? "block" : "none";
    cross.style.display = controls.isLocked ? "block" : "none";

    if (!locked) {
      const turn =
        (keys.has("ArrowLeft") ? 1 : 0) - (keys.has("ArrowRight") ? 1 : 0);
      if (turn) {
        camera.rotation.y += turn * TURN_RAD_PER_S * dt;
        faceYaw = null;
      }
    }
    if (faceYaw !== null) {
      let d = faceYaw - camera.rotation.y;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      camera.rotation.y += d * Math.min(1, dt * 6);
      if (Math.abs(d) < 0.01) faceYaw = null;
    }
    const yaw = camera.rotation.y;
    const fx = -Math.sin(yaw);
    const fz = -Math.cos(yaw);
    let mx = 0;
    let mz = 0;
    if (!locked) {
      const fwd =
        (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) -
        (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0);
      const side = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0);
      mx = fx * fwd + -fz * side;
      mz = fz * fwd + fx * side;
    }
    let moving = false;
    const speed = view.walk_speed;
    if (locked) route = [];
    else if (mx || mz) {
      const len = Math.hypot(mx, mz);
      const nx = me.x + (mx / len) * speed * dt;
      const ny = me.y + (mz / len) * speed * dt;
      if (!blockedByNpc(nx, ny))
        [me.x, me.y] = slide(map, me.x, me.y, nx - me.x, ny - me.y);
      moving = true;
    } else if (route.length) {
      const dir = steer(me, route, speed, dt, true);
      if (dir) {
        moving = true;
        faceYaw = Math.atan2(-dir[0], -dir[1]);
      }
    }
    walkPhase += moving ? dt * 9 : 0;
    const ground = heightAt(terrain, me.x, me.y);
    if (editor) editor.frame();
    else
      camera.position.set(
        me.x,
        ground + view.camera_height + (moving ? Math.sin(walkPhase) * 0.03 : 0),
        me.y,
      );
    skyDome?.position.copy(camera.position);

    for (const n of npcs) {
      const close = Math.hypot(n.pos.x - me.x, n.pos.y - me.y) < 1.8;
      let walking = false;
      if (talking === n.id || n.busy || close || !n.wander) {
        n.path = [];
      } else if (steer(n.pos, n.path, NPC_TILES_PER_S, dt, false)) {
        walking = true;
      } else {
        n.rest -= dt;
        if (n.rest <= 0) {
          n.rest = 2 + Math.random() * 4;
          const here = tileOf(n.pos);
          const a = n.area;
          const inside =
            a &&
            here.x >= a.x &&
            here.x < a.x + a.w &&
            here.y >= a.y &&
            here.y < a.y + a.h;
          const target = a
            ? {
                x: a.x + Math.floor(Math.random() * a.w),
                y: a.y + Math.floor(Math.random() * a.h),
              }
            : {
                x: n.home.x + Math.round((Math.random() - 0.5) * 6),
                y: n.home.y + Math.round((Math.random() - 0.5) * 6),
              };
          if (a && !inside) {
            n.rest = 0.5;
            n.path = findPath(map, here, target);
          } else if (!isSolid(map, target.x, target.y))
            n.path = findPath(map, here, target, 800).slice(0, 8);
        }
      }
      n.phase += walking ? dt * 10 : 0;
      const bob = walking
        ? Math.abs(Math.sin(n.phase)) * 0.07
        : Math.sin(now / 600 + n.home.x) * 0.012;
      n.group.position.set(
        n.pos.x,
        heightAt(terrain, n.pos.x, n.pos.y) + bob,
        n.pos.y,
      );
      const look =
        close || talking === n.id
          ? Math.atan2(me.x - n.pos.x, me.y - n.pos.y)
          : null;
      if (look !== null) n.group.rotation.y = look;
      if (talking === n.id) {
        const d = Math.hypot(me.x - n.pos.x, me.y - n.pos.y);
        if (d > 1e-3 && d < 1.7) {
          const step = Math.min(1.7 - d, 2.5 * dt);
          [me.x, me.y] = slide(
            map,
            me.x,
            me.y,
            ((me.x - n.pos.x) / d) * step,
            ((me.y - n.pos.y) / d) * step,
          );
        }
      }
      const dist = Math.hypot(n.pos.x - me.x, n.pos.y - me.y);
      n.label.visible = dist < 12 && nearby?.id !== n.id;
      if (n.speech) {
        n.speech.sprite.position.copy(n.group.position);
        n.speech.sprite.position.y += ACTOR_H + 0.45;
        if (now > n.speech.until) {
          dropSprite(n.speech.sprite);
          n.speech = null;
        }
      }
      if (pendingTalk === n.id && dist <= REACH && !locked)
        interact({ kind: "npc", id: n.id, name: n.name });
    }
    for (const it of items.values()) {
      if (it.gone) {
        const t = Math.min(1, (now - it.gone) / 380);
        it.sprite.position.y = it.base + t * 0.8;
        (it.sprite.material as three.SpriteMaterial).opacity = 1 - t;
        if (t >= 1) {
          scene.remove(it.sprite);
          items.delete(it.id);
        }
      } else
        it.sprite.position.y = it.base + Math.sin(now / 450 + it.at.x) * 0.06;
    }

    let best: Interactable | null = null;
    let bestD = REACH;
    let bp: three.Vector3 | null = null;
    for (const n of npcs) {
      const d = Math.hypot(n.pos.x - me.x, n.pos.y - me.y);
      if (d < bestD) {
        bestD = d;
        best = { kind: "npc", id: n.id, name: n.name };
        bp = n.group.position.clone().setY(n.group.position.y + ACTOR_H + 0.12);
      }
    }
    for (const it of items.values()) {
      if (it.gone) continue;
      const d = Math.hypot(it.at.x + 0.5 - me.x, it.at.y + 0.5 - me.y);
      if (d < bestD) {
        bestD = d;
        best = { kind: "item", id: it.id, name: it.name };
        bp = it.sprite.position.clone().setY(it.base + 0.7);
      }
    }
    if ((best?.id ?? null) !== (nearby?.id ?? null)) {
      nearby = best;
      bridge.onNearby(best);
    }
    prompt.visible = Boolean(best && bp && !locked);
    if (bp) prompt.position.copy(bp);

    regionTimer -= dt * 1000;
    if (regionTimer <= 0) {
      regionTimer = 200;
      const p = tileOf(me);
      const r = spec.regions.find(
        (g) => p.x >= g.x && p.x < g.x + g.w && p.y >= g.y && p.y < g.y + g.h,
      );
      if ((r?.id ?? null) !== region) {
        region = r?.id ?? null;
        bridge.onRegion(region);
      }
    }
    renderer.render(scene, camera);
  };
  raf = requestAnimationFrame(frame);

  const npcById = (id: string) => npcs.find((n) => n.id === id);
  return {
    destroy() {
      destroyed = true;
      editor?.close();
      editor = null;
      props.dispose();
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      if (controls.isLocked) controls.unlock();
      controls.dispose();
      for (const d of disposables) d.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      overlay.remove();
      cross.remove();
    },
    playerTile() {
      return tileOf(me);
    },
    addNpc(npc: WorldNpc, at: Point) {
      if (!npcs.some((n) => n.id === npc.id)) spawnNpc(npc, at);
    },
    addItem(item: WorldItem, at: Point) {
      if (!items.has(item.id)) spawnItem(item, at);
    },
    addMarker(id: string, at: Point, icon: string, label: string) {
      if (markers.has(id)) return;
      const sign = tagSprite(`${icon} ${label}`, 0.22);
      sign.position.set(
        at.x + 0.5,
        heightAt(terrain, at.x + 0.5, at.y + 0.5) + 0.2,
        at.y + 0.5,
      );
      scene.add(sign);
      markers.set(id, sign);
    },
    openEditor() {
      if (editor) return Promise.resolve(editor);
      opening ??= import("./world-editor3d").then(({ createWorldEditor }) => {
        opening = null;
        if (destroyed) throw new Error("The 3D view was closed");
        if (controls.isLocked) controls.unlock();
        const ed = createWorldEditor({
          scene,
          camera,
          dom: renderer.domElement,
          layer: props,
          exportRoot: world,
          map,
          groundAt: (x, y) => heightAt(terrain, x, y),
          player: () => ({ x: me.x, y: me.y, yaw: camera.rotation.y }),
          openTile: (x, y) => !SOLID.has(kindAt(map, x, y)),
          onProps: (next) => setMapProps(map, next, standTiles(me.x, me.y)),
        });
        const close = ed.close;
        ed.close = () => {
          close();
          if (editor === ed) editor = null;
        };
        editor = ed;
        return ed;
      });
      return opening;
    },
    removeItem(id: string) {
      const it = items.get(id);
      if (it && !it.gone) it.gone = performance.now();
    },
    faceNpc(id: string | null) {
      talking = id;
      const n = id ? npcById(id) : null;
      if (n) faceYaw = Math.atan2(-(n.pos.x - me.x), -(n.pos.y - me.y));
    },
    walkTo(tile: Point) {
      route = findPath(map, tileOf(me), tile);
    },
    npcTile(id: string) {
      const n = npcById(id);
      return n ? tileOf(n.pos) : null;
    },
    npcTiles() {
      const out: Record<string, Point> = {};
      for (const n of npcs) out[n.id] = tileOf(n.pos);
      return out;
    },
    setNpcPlace(id: string, area: Area | null) {
      const n = npcById(id);
      if (!n) return;
      const same =
        n.area === area ||
        (n.area &&
          area &&
          n.area.x === area.x &&
          n.area.y === area.y &&
          n.area.w === area.w &&
          n.area.h === area.h);
      if (same) return;
      n.area = area;
      n.path = [];
      n.rest = 0;
    },
    setNpcBusy(id: string, busy: boolean) {
      const n = npcById(id);
      if (n) n.busy = busy;
    },
    setNpcStatus(id: string, icon: string) {
      const n = npcById(id);
      const text = icon ? `${icon} ${n?.name}` : (n?.name ?? "");
      if (n && n.status !== text) {
        n.status = text;
        setTag(n.label, text, 0.16);
      }
    },
    say(id: string, text: string, ms = 4000) {
      const n = npcById(id);
      if (!n) return;
      if (n.speech) dropSprite(n.speech.sprite);
      const s = tagSprite(
        text,
        Math.min(0.9, 0.22 * Math.ceil(text.length / 34)) + 0.12,
        true,
      );
      scene.add(s);
      n.speech = { sprite: s, until: performance.now() + ms };
    },
  };
}
