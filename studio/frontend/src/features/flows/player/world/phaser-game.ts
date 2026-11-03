// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The world renderer and simulation. Only world-player.tsx imports this, and only with import(),
// so Phaser stays in its own chunk and never loads for pipelines or stories.

import * as Phaser from "phaser";

import { Prng, seedFrom } from "../../expr";
import type { WorldItem, WorldNpc, WorldProp3d, WorldSpec } from "../../types";
import { type PropStates, closedProps, isOpen, isUsable } from "./action-verbs";
import {
  type Point,
  SOLID,
  TILE_KINDS,
  type TileKind,
  type WorldMap,
  findPath,
  isSolid,
} from "./procgen";
import type { Painter } from "./city-draw";
import { PROP_LABELS, type WorldEditor, blockedCells } from "./props3d";
import {
  canvas,
  coverImage,
  fitImage,
  paintChibi,
  paintItem,
  paintTile,
} from "./sprites";

export interface Interactable {
  kind: "npc" | "item" | "prop";
  id: string;
  name: string;
  prop?: WorldProp3d;
}

export interface WorldBridge {
  onNearby(target: Interactable | null): void;
  onInteract(target: Interactable): void;
  onRegion(regionId: string | null): void;
  locked(): boolean;
  action?: unknown;
  props?: PropStates;
  onEvent?(text: string): void;
  safeAt?(p: Point): boolean;
  onTile?(p: Point, drag: boolean): boolean;
}

export interface WorldArt {
  tiles: Partial<Record<TileKind, HTMLImageElement>>;
  sprites: Record<string, HTMLImageElement>;
  models?: Record<string, ArrayBuffer>;
}

export interface WorldGameHandle {
  destroy(): void;
  playerTile(): Point;
  /** The player's body overlaps this prop (3D action movement); tile checks miss a capsule reaching across. */
  blocksProp?(p: WorldProp3d): boolean;
  removeItem(id: string): void;
  addNpc(npc: WorldNpc, at: Point): void;
  addItem(item: WorldItem, at: Point): void;
  addMarker(id: string, at: Point, icon: string, label: string): void;
  faceNpc(id: string | null): void;
  walkTo(tile: Point): void;
  npcTile(id: string): Point | null;
  npcTiles(): Record<string, Point>;
  setNpcPlace(id: string, area: Area | null): void;
  setNpcBusy(id: string, busy: boolean): void;
  setNpcStatus(id: string, icon: string): void;
  say(id: string, text: string, ms?: number): void;
  openEditor?(): Promise<WorldEditor>;
  actionState?(): unknown;
  paintCity?(paint: (g: Painter, tile: number) => void): void;
  setPropStates?(
    states: PropStates,
    changed: readonly string[],
    loud: boolean,
  ): void;
}

export interface Area {
  x: number;
  y: number;
  w: number;
  h: number;
}

const PROP_MARKS: Record<string, number> = {
  house: 0xe9d7b5,
  tree: 0x3f8a3a,
  lamp: 0xf2c84b,
  fence: 0x9a6b42,
  well: 0x8d8a83,
  crate: 0xb5834a,
  rock: 0x8d8a83,
  bench: 0x9a6b42,
  stall: 0xd9534f,
  door: 0x7a5233,
  chest: 0xc89b3c,
  lever: 0x6d7a8a,
  model: 0x8f7ae6,
};

const VARIANTS = 4;
const WALK_TILES_PER_S = 4.4;
const NPC_TILES_PER_S = 1.5;
const REACH = 1.35;

interface Actor {
  id: string;
  name: string;
  body: Phaser.GameObjects.Zone;
  sprite: Phaser.GameObjects.Image;
  shadow: Phaser.GameObjects.Image;
  home: Point;
  path: Point[];
  rest: number;
  phase: number;
  area: Area | null;
  busy: boolean;
  speech: Phaser.GameObjects.Container | null;
  rng: Prng;
}

export function createWorldGame(
  parent: HTMLElement,
  spec: WorldSpec,
  map: WorldMap,
  art: WorldArt,
  picked: string[],
  start: Point | null,
  bridge: WorldBridge,
): WorldGameHandle {
  const T = spec.tile_size;
  const ground: TileKind =
    spec.biome === "dungeon"
      ? "floor"
      : spec.biome === "desert" || spec.biome === "beach"
        ? "sand"
        : spec.biome === "snow"
          ? "snow"
          : "grass";
  let sceneRef: WorldScene | null = null;
  const npcSpecs = new Map<string, WorldNpc>();
  for (const n of spec.npcs) if (!npcSpecs.has(n.id)) npcSpecs.set(n.id, n);
  const pendingMarkers = new Map<string, [Point, string, string]>();
  let propState: PropStates = bridge.props ?? {};
  const usable = (map.props ?? []).filter(isUsable);
  const propMarks = new Map<string, Phaser.GameObjects.Graphics>();
  const reblock = () => {
    if (map.props?.length)
      map.blocked = blockedCells(map, closedProps(map.props, propState));
  };
  // The map may come from an earlier game where a door stood open: always derive blocking from this state.
  if (usable.length) reblock();
  const propTarget = (p: WorldProp3d): Interactable => ({
    kind: "prop",
    id: p.id,
    name:
      (p.name ?? PROP_LABELS[p.kind]) +
      (p.kind === "door" && p.locked_by && !propState[p.id]?.unlocked
        ? " (locked)"
        : ""),
    prop: p,
  });

  class WorldScene extends Phaser.Scene {
    player!: Actor;
    npcs: Actor[] = [];
    items = new Map<
      string,
      {
        sprite: Phaser.GameObjects.Image;
        shadow: Phaser.GameObjects.Image;
        at: Point;
        name: string;
      }
    >();
    markers = new Map<string, Phaser.GameObjects.Text[]>();
    keys!: Record<string, Phaser.Input.Keyboard.Key>;
    bubble!: Phaser.GameObjects.Container;
    nearby: Interactable | null = null;
    region: string | null = null;
    regionTimer = 0;
    talking: string | null = null;
    pendingTalk: string | null = null;
    route: Point[] = [];
    layer!: Phaser.Tilemaps.TilemapLayer;
    city: Phaser.GameObjects.Graphics | null = null;

    constructor() {
      super("world");
    }

    create() {
      sceneRef = this;
      this.buildTextures();
      const data: number[][] = [];
      for (let y = 0; y < map.height; y++) {
        const row: number[] = [];
        for (let x = 0; x < map.width; x++) {
          row.push(
            map.tiles[y * map.width + x] * VARIANTS * VARIANTS +
              (x % VARIANTS) +
              (y % VARIANTS) * VARIANTS,
          );
        }
        data.push(row);
      }
      const tm = this.make.tilemap({ data, tileWidth: T, tileHeight: T });
      const tileset = tm.addTilesetImage(
        "flows-tiles",
        "flows-tiles",
        T,
        T,
        1,
        2,
      );
      const layer = tileset ? tm.createLayer(0, tileset, 0, 0) : null;
      if (!layer) throw new Error("Could not build the tile layer");
      const solid: number[] = [];
      TILE_KINDS.forEach((k, i) => {
        if (SOLID.has(k))
          for (let v = 0; v < VARIANTS * VARIANTS; v++)
            solid.push(i * VARIANTS * VARIANTS + v);
      });
      layer.setCollision(solid);
      layer.setDepth(-10);
      if (map.blocked) {
        map.blocked.forEach((b, i) => {
          if (b)
            layer
              .getTileAt(i % map.width, Math.floor(i / map.width))
              ?.setCollision(true);
        });
      }
      for (const p of map.props ?? []) {
        const cx = Math.floor(p.x) * T + T / 2;
        const cy = Math.floor(p.y) * T + T / 2;
        const mark = this.add.graphics();
        mark.fillStyle(0x1b1b28, 0.35);
        mark.fillEllipse(cx, cy + T * 0.28, T * 0.7, T * 0.22);
        mark.fillStyle(PROP_MARKS[p.kind] ?? 0xc8b48a, 1);
        mark.lineStyle(Math.max(1, T / 24), 0x1b1b28, 0.6);
        mark.fillRoundedRect(
          cx - T * 0.28,
          cy - T * 0.36,
          T * 0.56,
          T * 0.56,
          T * 0.12,
        );
        mark.strokeRoundedRect(
          cx - T * 0.28,
          cy - T * 0.36,
          T * 0.56,
          T * 0.56,
          T * 0.12,
        );
        mark.setDepth(cy);
        if (isUsable(p)) {
          propMarks.set(p.id, mark);
          if (isOpen(p, propState)) mark.setAlpha(0.35);
        }
        // Graphics are re-tessellated every frame; skip the ones the camera cannot see.
        const drawable = mark.willRender.bind(mark);
        mark.willRender = (cam) =>
          drawable(cam) &&
          cx + T > cam.worldView.x &&
          cx - T < cam.worldView.right &&
          cy + T > cam.worldView.y &&
          cy - T < cam.worldView.bottom;
      }
      this.physics.world.setBounds(0, 0, map.width * T, map.height * T);

      const spawn =
        start && !isSolid(map, start.x, start.y) ? start : map.spawn;
      this.player = this.makeActor(
        "player",
        spec.player.name,
        spawn,
        "flows-player",
        layer,
      );
      this.layer = layer;
      for (const n of spec.npcs) {
        const at = map.npcs[n.id];
        if (at) this.spawnNpc(n, at);
      }
      for (const it of spec.items) {
        const at = map.items[it.id];
        if (at && !picked.includes(it.id)) this.spawnItem(it, at);
      }

      const g = this.add.graphics();
      g.fillStyle(0xffffff, 0.95);
      g.fillRoundedRect(-11, -24, 22, 20, 6);
      g.fillTriangle(-5, -4, 5, -4, 0, 2);
      const t = this.add
        .text(0, -14, "E", {
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: "13px",
          color: "#1b1b28",
          fontStyle: "bold",
        })
        .setOrigin(0.5);
      this.bubble = this.add
        .container(0, 0, [g, t])
        .setDepth(1e6)
        .setVisible(false);
      this.tweens.add({
        targets: this.bubble,
        scale: 1.08,
        duration: 500,
        yoyo: true,
        repeat: -1,
      });

      const cam = this.cameras.main;
      cam.setBounds(0, 0, map.width * T, map.height * T);
      cam.startFollow(this.player.sprite, true, 0.12, 0.12);
      cam.setRoundPixels(true);
      const fitZoom = () =>
        cam.setZoom(
          Phaser.Math.Clamp(Math.round(this.scale.width / (T * 18)), 1, 3),
        );
      fitZoom();
      this.scale.on("resize", fitZoom);
      cam.fadeIn(600, 0, 0, 0);

      const kb = this.input.keyboard;
      if (kb) {
        kb.disableGlobalCapture();
        this.keys = kb.addKeys(
          "W,A,S,D,UP,LEFT,DOWN,RIGHT,E,SPACE",
          false,
        ) as Record<string, Phaser.Input.Keyboard.Key>;
      }
      this.input.on("pointerdown", (p: Phaser.Input.Pointer) => {
        if (bridge.locked()) return;
        const tx = Math.floor(p.worldX / T);
        const ty = Math.floor(p.worldY / T);
        if (bridge.onTile?.({ x: tx, y: ty }, false)) return;
        const npc = this.npcs.find((n) => {
          const at = this.tileOf(n);
          return Math.abs(at.x - tx) <= 0 && Math.abs(at.y - ty) <= 1;
        });
        this.pendingTalk = npc ? npc.id : null;
        if (npc && this.distTo(npc.sprite.x, npc.sprite.y) <= REACH * T) {
          this.interact({ kind: "npc", id: npc.id, name: npc.name });
          return;
        }
        this.route = findPath(map, this.tileOf(this.player), { x: tx, y: ty });
      });
      this.input.on("pointermove", (p: Phaser.Input.Pointer) => {
        if (p.isDown && !bridge.locked())
          bridge.onTile?.(
            { x: Math.floor(p.worldX / T), y: Math.floor(p.worldY / T) },
            true,
          );
      });
      for (const [id, m] of pendingMarkers) drawMarker(this, id, ...m);
      pendingMarkers.clear();
    }

    buildTextures() {
      // Each tile is extruded by 1px (margin 1, spacing 2) so filtering never samples a neighbour: no seams.
      const P = T + 2;
      const [atlas, actx] = canvas(
        VARIANTS * VARIANTS * P,
        TILE_KINDS.length * P,
      );
      TILE_KINDS.forEach((kind, i) => {
        const img = art.tiles[kind];
        let src = img
          ? coverImage(img, VARIANTS * T)
          : paintTile(kind, VARIANTS * T, ground, spec.seed);
        if (img && (kind === "tree" || kind === "rock")) {
          const [c, cx] = canvas(VARIANTS * T, VARIANTS * T);
          cx.drawImage(
            paintTile(ground, VARIANTS * T, ground, spec.seed),
            0,
            0,
          );
          cx.globalAlpha = 0.92;
          cx.drawImage(src, 0, 0);
          src = c;
        }
        for (let v = 0; v < VARIANTS * VARIANTS; v++) {
          const sx = (v % VARIANTS) * T;
          const sy = Math.floor(v / VARIANTS) * T;
          const dx = v * P + 1;
          const dy = i * P + 1;
          actx.drawImage(src, sx, sy, T, T, dx - 1, dy - 1, T + 2, T + 2);
          actx.drawImage(src, sx, sy, T, T, dx, dy, T, T);
        }
      });
      this.textures.addCanvas("flows-tiles", atlas);
      const [sh, sctx] = canvas(64, 24);
      const grad = sctx.createRadialGradient(32, 12, 2, 32, 12, 30);
      grad.addColorStop(0, "rgba(0,0,0,0.55)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      sctx.fillStyle = grad;
      sctx.fillRect(0, 0, 64, 24);
      this.textures.addCanvas("flows-shadow", sh);
      this.actorTexture("flows-player", "player");
      for (const n of spec.npcs) this.actorTexture(`flows-npc-${n.id}`, n.id);
      for (const it of spec.items) this.itemTexture(it.id);
    }

    actorTexture(key: string, id: string): void {
      const actorH = Math.round(T * 1.6);
      const img = art.sprites[id];
      if (this.textures.exists(key)) return;
      this.textures.addCanvas(
        key,
        img ? fitImage(img, actorH * 0.85, actorH) : paintChibi(id, actorH),
      );
    }

    itemTexture(id: string): void {
      const img = art.sprites[id];
      const size = Math.round(T * 0.9);
      if (this.textures.exists(`flows-item-${id}`)) return;
      this.textures.addCanvas(
        `flows-item-${id}`,
        img ? fitImage(img, size, size) : paintItem(id, size),
      );
    }

    spawnNpc(n: WorldNpc, at: Point): void {
      const a = this.makeActor(
        n.id,
        n.name,
        at,
        `flows-npc-${n.id}`,
        this.layer,
      );
      (a.body.body as Phaser.Physics.Arcade.Body).setImmovable(true);
      this.physics.add.collider(this.player.body, a.body);
      a.rest = 1 + a.rng.random() * 3;
      this.npcs.push(a);
      const label = this.add
        .text(0, 0, n.name, {
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: `${Math.max(10, Math.round(T * 0.34))}px`,
          color: "#ffffff",
          stroke: "#1b1b28",
          strokeThickness: 3,
        })
        .setOrigin(0.5, 1)
        .setAlpha(0.9);
      a.sprite.setData("label", label);
    }

    spawnItem(it: WorldItem, at: Point): void {
      const cx = at.x * T + T / 2;
      const cy = at.y * T + T / 2;
      const shadow = this.add
        .image(cx, cy + T * 0.32, "flows-shadow")
        .setDisplaySize(T * 0.6, T * 0.2)
        .setAlpha(0.5);
      const sprite = this.add
        .image(cx, cy, `flows-item-${it.id}`)
        .setOrigin(0.5, 0.8);
      sprite.setDepth(cy);
      this.tweens.add({
        targets: sprite,
        y: cy - T * 0.12,
        duration: 900 + Math.random() * 400,
        yoyo: true,
        repeat: -1,
        ease: "Sine.inOut",
      });
      this.items.set(it.id, { sprite, shadow, at, name: it.name });
    }

    makeActor(
      id: string,
      name: string,
      at: Point,
      texture: string,
      layer: Phaser.Tilemaps.TilemapLayer,
    ): Actor {
      const x = at.x * T + T / 2;
      const y = at.y * T + T / 2;
      const body = this.add.zone(x, y, T * 0.55, T * 0.4);
      this.physics.add.existing(body);
      const b = body.body as Phaser.Physics.Arcade.Body;
      b.setCollideWorldBounds(true);
      this.physics.add.collider(body, layer);
      const shadow = this.add
        .image(x, y + T * 0.18, "flows-shadow")
        .setDisplaySize(T * 0.85, T * 0.3);
      const sprite = this.add.image(x, y, texture).setOrigin(0.5, 0.92);
      const rng = new Prng(seedFrom(spec.seed, "wander", id));
      return {
        id,
        name,
        body,
        sprite,
        shadow,
        home: at,
        path: [],
        rest: 0,
        phase: rng.random() * 10,
        area: null,
        busy: false,
        speech: null,
        rng,
      };
    }

    tileOf(a: Actor): Point {
      return { x: Math.floor(a.body.x / T), y: Math.floor(a.body.y / T) };
    }

    distTo(x: number, y: number): number {
      return Phaser.Math.Distance.Between(
        this.player.body.x,
        this.player.body.y,
        x,
        y,
      );
    }

    interact(target: Interactable) {
      this.route = [];
      this.pendingTalk = null;
      (this.player.body.body as Phaser.Physics.Arcade.Body).setVelocity(0, 0);
      bridge.onInteract(target);
    }

    steer(a: Actor, path: Point[], speed: number): boolean {
      const b = a.body.body as Phaser.Physics.Arcade.Body;
      while (path.length) {
        const wp = path[0];
        const tx = wp.x * T + T / 2;
        const ty = wp.y * T + T / 2;
        const d = Phaser.Math.Distance.Between(a.body.x, a.body.y, tx, ty);
        if (d < 3) {
          path.shift();
          continue;
        }
        const ang = Math.atan2(ty - a.body.y, tx - a.body.x);
        b.setVelocity(Math.cos(ang) * speed, Math.sin(ang) * speed);
        return true;
      }
      b.setVelocity(0, 0);
      return false;
    }

    animate(a: Actor, moving: boolean, vx: number, time: number) {
      a.phase += moving ? 0.016 * 12 : 0;
      const bob = moving
        ? Math.abs(Math.sin(a.phase)) * T * 0.09
        : Math.sin(time / 600 + a.home.x) * T * 0.015;
      a.sprite.setPosition(a.body.x, a.body.y + T * 0.2 - bob);
      a.sprite.setRotation(moving ? Math.sin(a.phase) * 0.06 : 0);
      if (vx < -1) a.sprite.setFlipX(true);
      else if (vx > 1) a.sprite.setFlipX(false);
      a.shadow
        .setPosition(a.body.x, a.body.y + T * 0.22)
        .setScale(moving ? 0.9 + bob / T : 1, 1);
      a.sprite.setDepth(a.body.y);
      a.shadow.setDepth(a.body.y - 1);
      const label = a.sprite.getData("label") as
        | Phaser.GameObjects.Text
        | undefined;
      if (label) {
        label.setPosition(a.body.x, a.body.y - T * 1.35);
        label.setDepth(a.body.y + 0.5);
        label.setVisible(
          this.nearby?.id !== a.id && this.distTo(a.body.x, a.body.y) < T * 5,
        );
      }
      a.speech?.setPosition(a.body.x, a.body.y - T * 1.5).setDepth(1e6 - 1);
    }

    update(time: number, delta: number) {
      const dt = Math.min(delta, 50) / 1000;
      const locked = bridge.locked();
      const pb = this.player.body.body as Phaser.Physics.Arcade.Body;
      let vx = 0;
      let vy = 0;
      if (!locked && this.keys) {
        const k = this.keys;
        if (k.A.isDown || k.LEFT.isDown) vx -= 1;
        if (k.D.isDown || k.RIGHT.isDown) vx += 1;
        if (k.W.isDown || k.UP.isDown) vy -= 1;
        if (k.S.isDown || k.DOWN.isDown) vy += 1;
      }
      const speed = WALK_TILES_PER_S * T;
      if (locked) {
        pb.setVelocity(0, 0);
        this.route = [];
      } else if (vx || vy) {
        this.route = [];
        this.pendingTalk = null;
        const len = Math.hypot(vx, vy);
        pb.setVelocity((vx / len) * speed, (vy / len) * speed);
      } else if (!this.steer(this.player, this.route, speed)) {
        pb.setVelocity(0, 0);
      }
      const moving = Math.hypot(pb.velocity.x, pb.velocity.y) > 4;
      this.animate(this.player, moving, pb.velocity.x, time);

      for (const n of this.npcs) {
        const nb = n.body.body as Phaser.Physics.Arcade.Body;
        const close = this.distTo(n.body.x, n.body.y) < T * 1.8;
        if (
          this.talking === n.id ||
          n.busy ||
          close ||
          npcSpecs.get(n.id)?.wander === false
        ) {
          nb.setVelocity(0, 0);
          n.path = [];
          if (close) n.sprite.setFlipX(this.player.body.x < n.body.x);
        } else if (!this.steer(n, n.path, NPC_TILES_PER_S * T)) {
          n.rest -= dt;
          if (n.rest <= 0) {
            n.rest = 2 + n.rng.random() * 4;
            const here = this.tileOf(n);
            const a = n.area;
            const inside =
              a &&
              here.x >= a.x &&
              here.x < a.x + a.w &&
              here.y >= a.y &&
              here.y < a.y + a.h;
            const target = a
              ? {
                  x: a.x + Math.floor(n.rng.random() * a.w),
                  y: a.y + Math.floor(n.rng.random() * a.h),
                }
              : {
                  x: n.home.x + Math.round((n.rng.random() - 0.5) * 6),
                  y: n.home.y + Math.round((n.rng.random() - 0.5) * 6),
                };
            if (a && !inside) {
              n.rest = 0.5;
              n.path = findPath(map, here, target);
            } else if (!isSolid(map, target.x, target.y))
              n.path = findPath(map, here, target, 800).slice(0, 8);
          }
        }
        this.animate(
          n,
          Math.hypot(nb.velocity.x, nb.velocity.y) > 4,
          nb.velocity.x,
          time,
        );
        if (
          this.pendingTalk === n.id &&
          this.distTo(n.body.x, n.body.y) <= REACH * T &&
          !locked
        ) {
          this.interact({ kind: "npc", id: n.id, name: n.name });
        }
      }

      let best: Interactable | null = null;
      let bestD = REACH * T;
      let bx = 0;
      let by = 0;
      for (const n of this.npcs) {
        const d = this.distTo(n.body.x, n.body.y);
        if (d < bestD) {
          bestD = d;
          best = { kind: "npc", id: n.id, name: n.name };
          bx = n.sprite.x;
          by = n.sprite.y - T * 1.45;
        }
      }
      for (const [id, it] of this.items) {
        const d = this.distTo(it.sprite.x, it.at.y * T + T / 2);
        if (d < bestD) {
          bestD = d;
          best = { kind: "item", id, name: it.name };
          bx = it.sprite.x;
          by = it.at.y * T - T * 0.25;
        }
      }
      for (const p of usable) {
        const cx = Math.floor(p.x) * T + T / 2;
        const cy = Math.floor(p.y) * T + T / 2;
        const d = this.distTo(cx, cy);
        if (d < bestD) {
          bestD = d;
          best = propTarget(p);
          bx = cx;
          by = cy - T * 0.7;
        }
      }
      if (
        (best?.id ?? null) !== (this.nearby?.id ?? null) ||
        best?.name !== this.nearby?.name
      ) {
        this.nearby = best;
        bridge.onNearby(best);
      }
      this.bubble.setVisible(Boolean(best) && !locked);
      if (best) this.bubble.setPosition(bx, by);
      if (
        best &&
        !locked &&
        this.keys &&
        (Phaser.Input.Keyboard.JustDown(this.keys.E) ||
          Phaser.Input.Keyboard.JustDown(this.keys.SPACE))
      ) {
        this.interact(best);
      }

      this.regionTimer -= delta;
      if (this.regionTimer <= 0) {
        this.regionTimer = 200;
        const p = this.tileOf(this.player);
        const r = spec.regions.find(
          (g) => p.x >= g.x && p.x < g.x + g.w && p.y >= g.y && p.y < g.y + g.h,
        );
        if ((r?.id ?? null) !== this.region) {
          this.region = r?.id ?? null;
          bridge.onRegion(this.region);
        }
      }
    }
  }

  const game = new Phaser.Game({
    type: Phaser.AUTO,
    parent,
    backgroundColor: "#0b0d12",
    banner: false,
    audio: { noAudio: true },
    scale: {
      mode: Phaser.Scale.RESIZE,
      width: parent.clientWidth || 960,
      height: parent.clientHeight || 600,
    },
    physics: {
      default: "arcade",
      arcade: { debug: false, fps: 60, fixedStep: true },
    },
    render: {
      antialias: true,
      roundPixels: true,
      powerPreference: "high-performance",
    },
    scene: [WorldScene],
  });

  const onVisibility = () => {
    if (document.hidden) game.loop.sleep();
    else game.loop.wake();
  };
  document.addEventListener("visibilitychange", onVisibility);

  function drawMarker(
    scene: WorldScene,
    id: string,
    at: Point,
    icon: string,
    label: string,
  ): void {
    if (scene.markers.has(id)) return;
    const cx = at.x * T + T / 2;
    const cy = at.y * T + T / 2;
    const sign = scene.add
      .text(cx, cy, icon, {
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: `${Math.round(T * 0.8)}px`,
      })
      .setOrigin(0.5, 0.75)
      .setDepth(cy);
    const name = scene.add
      .text(cx, cy - T * 0.55, label, {
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: `${Math.max(9, Math.round(T * 0.28))}px`,
        color: "#fff7d6",
        stroke: "#1b1b28",
        strokeThickness: 3,
      })
      .setOrigin(0.5, 1)
      .setDepth(cy + 1)
      .setAlpha(0.85);
    scene.markers.set(id, [sign, name]);
  }

  return {
    setPropStates(states, changed) {
      propState = states;
      reblock();
      for (const id of changed) {
        const p = usable.find((x) => x.id === id);
        if (!p) continue;
        propMarks.get(id)?.setAlpha(isOpen(p, propState) ? 0.35 : 1);
        const tx = Math.floor(p.x);
        const ty = Math.floor(p.y);
        if (p.kind === "door")
          sceneRef?.layer.getTileAt(tx, ty)?.setCollision(isSolid(map, tx, ty));
      }
    },
    destroy() {
      document.removeEventListener("visibilitychange", onVisibility);
      game.destroy(true);
      sceneRef = null;
    },
    playerTile() {
      return sceneRef ? sceneRef.tileOf(sceneRef.player) : map.spawn;
    },
    removeItem(id: string) {
      const it = sceneRef?.items.get(id);
      if (!it || !sceneRef) return;
      sceneRef.tweens.killTweensOf(it.sprite);
      sceneRef.tweens.add({
        targets: it.sprite,
        y: it.sprite.y - T * 0.8,
        alpha: 0,
        scale: it.sprite.scale * 1.4,
        duration: 380,
        ease: "Quad.out",
        onComplete: () => it.sprite.destroy(),
      });
      it.shadow.destroy();
      sceneRef.items.delete(id);
    },
    addNpc(npc: WorldNpc, at: Point) {
      if (!sceneRef || sceneRef.npcs.some((n) => n.id === npc.id)) return;
      sceneRef.actorTexture(`flows-npc-${npc.id}`, npc.id);
      sceneRef.spawnNpc(npc, at);
    },
    addItem(item: WorldItem, at: Point) {
      if (!sceneRef || sceneRef.items.has(item.id)) return;
      sceneRef.itemTexture(item.id);
      sceneRef.spawnItem(item, at);
    },
    addMarker(id: string, at: Point, icon: string, label: string) {
      if (sceneRef) drawMarker(sceneRef, id, at, icon, label);
      else pendingMarkers.set(id, [at, icon, label]);
    },
    paintCity(paint: (g: Painter, tile: number) => void) {
      if (!sceneRef) return;
      sceneRef.city ??= sceneRef.add.graphics().setDepth(-5);
      paint(sceneRef.city, T);
    },
    faceNpc(id: string | null) {
      if (sceneRef) {
        sceneRef.talking = id;
      }
    },
    walkTo(tile: Point) {
      if (sceneRef) {
        sceneRef.route = findPath(map, sceneRef.tileOf(sceneRef.player), tile);
      }
    },
    npcTile(id: string) {
      const npc = sceneRef?.npcs.find((n) => n.id === id);
      return npc && sceneRef ? sceneRef.tileOf(npc) : null;
    },
    npcTiles() {
      const out: Record<string, Point> = {};
      const scene = sceneRef;
      for (const n of scene?.npcs ?? []) if (scene) out[n.id] = scene.tileOf(n);
      return out;
    },
    setNpcPlace(id: string, area: Area | null) {
      const npc = sceneRef?.npcs.find((n) => n.id === id);
      if (!npc) return;
      const same =
        npc.area === area ||
        (npc.area &&
          area &&
          npc.area.x === area.x &&
          npc.area.y === area.y &&
          npc.area.w === area.w &&
          npc.area.h === area.h);
      if (same) return;
      npc.area = area;
      npc.path = [];
      npc.rest = 0;
    },
    setNpcBusy(id: string, busy: boolean) {
      const npc = sceneRef?.npcs.find((n) => n.id === id);
      if (npc) npc.busy = busy;
    },
    setNpcStatus(id: string, icon: string) {
      const npc = sceneRef?.npcs.find((n) => n.id === id);
      const label = npc?.sprite.getData("label") as
        | Phaser.GameObjects.Text
        | undefined;
      if (npc && label) label.setText(icon ? `${icon} ${npc.name}` : npc.name);
    },
    say(id: string, text: string, ms = 4000) {
      const scene = sceneRef;
      const npc = scene?.npcs.find((n) => n.id === id);
      if (!scene || !npc) return;
      npc.speech?.destroy();
      const t = scene.add
        .text(0, 0, text, {
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: `${Math.max(10, Math.round(T * 0.32))}px`,
          color: "#1b1b28",
          wordWrap: { width: T * 5.5 },
          align: "center",
        })
        .setOrigin(0.5, 1);
      const pad = 6;
      const g = scene.add.graphics();
      g.fillStyle(0xffffff, 0.94);
      g.fillRoundedRect(
        -t.width / 2 - pad,
        -t.height - pad,
        t.width + pad * 2,
        t.height + pad * 2,
        8,
      );
      g.fillTriangle(-5, pad, 5, pad, 0, pad + 6);
      const bubble = scene.add.container(npc.body.x, npc.body.y - T * 1.5, [
        g,
        t,
      ]);
      npc.speech = bubble;
      scene.time.delayedCall(ms, () => {
        bubble.destroy();
        if (npc.speech === bubble) npc.speech = null;
      });
    },
  };
}
