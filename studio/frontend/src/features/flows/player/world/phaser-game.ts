// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The world renderer and simulation. Only world-player.tsx imports this, and only with import(),
// so Phaser stays in its own chunk and never loads for pipelines or stories.

import * as Phaser from "phaser";

import type { WorldSpec } from "../../types";
import {
  type Point,
  SOLID,
  TILE_KINDS,
  type TileKind,
  type WorldMap,
  findPath,
  isSolid,
} from "./procgen";
import {
  canvas,
  coverImage,
  fitImage,
  paintChibi,
  paintItem,
  paintTile,
} from "./sprites";

export interface Interactable {
  kind: "npc" | "item";
  id: string;
  name: string;
}

export interface WorldBridge {
  onNearby(target: Interactable | null): void;
  onInteract(target: Interactable): void;
  onRegion(regionId: string | null): void;
  /** True while a dialogue or menu has the keyboard. */
  locked(): boolean;
}

export interface WorldArt {
  tiles: Partial<Record<TileKind, HTMLImageElement>>;
  /** Keyed by "player", npc id or item id. */
  sprites: Record<string, HTMLImageElement>;
}

export interface WorldGameHandle {
  destroy(): void;
  playerTile(): Point;
  removeItem(id: string): void;
  faceNpc(id: string | null): void;
  /** Walk the player to a tile along the shortest path (click-to-move does the same). */
  walkTo(tile: Point): void;
  /** Where an NPC stands now (they wander). */
  npcTile(id: string): Point | null;
  npcTiles(): Record<string, Point>;
  /** Send an NPC to an area to spend time in; null goes back to wandering near home. */
  setNpcPlace(id: string, area: Area | null): void;
  setNpcBusy(id: string, busy: boolean): void;
  setNpcStatus(id: string, icon: string): void;
  say(id: string, text: string, ms?: number): void;
}

export interface Area {
  x: number;
  y: number;
  w: number;
  h: number;
}

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
    keys!: Record<string, Phaser.Input.Keyboard.Key>;
    bubble!: Phaser.GameObjects.Container;
    nearby: Interactable | null = null;
    region: string | null = null;
    regionTimer = 0;
    talking: string | null = null;
    pendingTalk: string | null = null;
    route: Point[] = [];

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
      for (const n of spec.npcs) {
        const at = map.npcs[n.id];
        if (!at) continue;
        const a = this.makeActor(n.id, n.name, at, `flows-npc-${n.id}`, layer);
        (a.body.body as Phaser.Physics.Arcade.Body).setImmovable(true);
        this.physics.add.collider(this.player.body, a.body);
        a.rest = 1 + Math.random() * 3;
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
      for (const it of spec.items) {
        const at = map.items[it.id];
        if (!at || picked.includes(it.id)) continue;
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
      // Whole-number zoom keeps tile edges on pixel boundaries.
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
          // A generated canopy/rock texture sits on painted ground so it still reads as an obstacle.
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
      const actorH = Math.round(T * 1.6);
      const actor = (key: string, id: string) => {
        const img = art.sprites[id];
        this.textures.addCanvas(
          key,
          img ? fitImage(img, actorH * 0.85, actorH) : paintChibi(id, actorH),
        );
      };
      actor("flows-player", "player");
      for (const n of spec.npcs) actor(`flows-npc-${n.id}`, n.id);
      for (const it of spec.items) {
        const img = art.sprites[it.id];
        const size = Math.round(T * 0.9);
        this.textures.addCanvas(
          `flows-item-${it.id}`,
          img ? fitImage(img, size, size) : paintItem(it.id, size),
        );
      }
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
      return {
        id,
        name,
        body,
        sprite,
        shadow,
        home: at,
        path: [],
        rest: 0,
        phase: Math.random() * 10,
        area: null,
        busy: false,
        speech: null,
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
        const spec_ = spec.npcs.find((x) => x.id === n.id);
        if (
          this.talking === n.id ||
          n.busy ||
          close ||
          spec_?.wander === false
        ) {
          nb.setVelocity(0, 0);
          n.path = [];
          if (close) n.sprite.setFlipX(this.player.body.x < n.body.x);
        } else if (!this.steer(n, n.path, NPC_TILES_PER_S * T)) {
          n.rest -= dt;
          if (n.rest <= 0) {
            n.rest = 2 + Math.random() * 4;
            const here = this.tileOf(n);
            const a = n.area;
            const inside =
              a &&
              here.x >= a.x &&
              here.x < a.x + a.w &&
              here.y >= a.y &&
              here.y < a.y + a.h;
            // Planned NPCs walk the whole way to their place, then mill about inside it.
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

      // Nearest thing in reach.
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
      if ((best?.id ?? null) !== (this.nearby?.id ?? null)) {
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

  return {
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
