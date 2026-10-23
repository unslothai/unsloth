// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Procedural art for worlds without generated images: tile textures, chibi characters and item gems.
// Everything is seeded, so a world looks the same every time it loads.

import { Prng, seedFrom } from "../../expr";
import type { TileKind } from "./procgen";

const BASE: Record<TileKind, [string, string, string]> = {
  grass: ["#78c45d", "#67b04f", "#8fd56f"],
  path: ["#d5b27a", "#c29c63", "#e3c391"],
  water: ["#4aa6e0", "#3a92cc", "#7cc5f0"],
  sand: ["#ecd69d", "#dcc285", "#f6e4b5"],
  snow: ["#eef3fb", "#dbe5f3", "#ffffff"],
  floor: ["#b07d4f", "#9a6a40", "#c48e5e"],
  wall: ["#80808f", "#6b6b79", "#9a9aa8"],
  tree: ["#78c45d", "#67b04f", "#8fd56f"],
  rock: ["#78c45d", "#67b04f", "#8fd56f"],
  flowers: ["#78c45d", "#67b04f", "#8fd56f"],
};

export function canvas(
  w: number,
  h: number,
): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  const ctx = c.getContext("2d") as CanvasRenderingContext2D;
  return [c, ctx];
}

function speckle(
  ctx: CanvasRenderingContext2D,
  s: number,
  colors: string[],
  n: number,
  size: number,
  rng: Prng,
) {
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = colors[Math.floor(rng.random() * colors.length)];
    const r = size * (0.5 + rng.random());
    ctx.fillRect(rng.random() * s, rng.random() * s, r, r);
  }
}

/** A `size`x`size` tileable texture for one tile kind (four tiles across at the usual 4x atlas size). */
export function paintTile(
  kind: TileKind,
  size: number,
  ground: TileKind,
  seed: number,
): HTMLCanvasElement {
  const rng = new Prng(seedFrom("tile", kind, seed));
  const base =
    kind === "tree" || kind === "rock" || kind === "flowers"
      ? BASE[ground === "sand" || ground === "snow" ? ground : "grass"]
      : BASE[kind];
  const [c, ctx] = canvas(size, size);
  ctx.fillStyle = base[0];
  ctx.fillRect(0, 0, size, size);
  const cell = size / 4;
  speckle(
    ctx,
    size,
    [base[1], base[2]],
    Math.round(size * 1.6),
    cell / 14,
    rng,
  );
  if (
    kind === "grass" ||
    kind === "flowers" ||
    kind === "tree" ||
    kind === "rock"
  ) {
    ctx.strokeStyle = base[1];
    ctx.lineWidth = Math.max(1, cell / 24);
    for (let i = 0; i < size / 3; i++) {
      const x = rng.random() * size;
      const y = rng.random() * size;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (rng.random() - 0.5) * cell * 0.15, y - cell * 0.14);
      ctx.stroke();
    }
  }
  if (kind === "path") {
    for (let i = 0; i < size / 5; i++) {
      ctx.fillStyle = rng.random() < 0.5 ? "#b48c58" : "#efd6a8";
      ctx.beginPath();
      ctx.ellipse(
        rng.random() * size,
        rng.random() * size,
        cell * 0.05,
        cell * 0.035,
        0,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }
  if (kind === "water") {
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = Math.max(1, cell / 20);
    for (let i = 0; i < size / 10; i++) {
      const x = rng.random() * size;
      const y = rng.random() * size;
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.16, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();
    }
  }
  if (kind === "floor") {
    ctx.strokeStyle = "rgba(60,30,10,0.45)";
    ctx.lineWidth = Math.max(1, cell / 18);
    for (let y = 0; y < size; y += cell / 3) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size, y);
      ctx.stroke();
      const off = rng.random() * cell;
      for (let x = off; x < size; x += cell * 1.2) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + cell / 3);
        ctx.stroke();
      }
    }
  }
  if (kind === "wall") {
    const bh = cell / 2.5;
    ctx.strokeStyle = "rgba(40,40,50,0.6)";
    ctx.lineWidth = Math.max(1, cell / 16);
    for (let row = 0, y = 0; y < size; y += bh, row++) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size, y);
      ctx.stroke();
      for (let x = (row % 2) * (cell / 2); x < size; x += cell) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + bh);
        ctx.stroke();
      }
    }
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(0, 0, size, bh / 3);
  }
  if (kind === "flowers") {
    const petals = ["#ffd1e8", "#fff2a8", "#ffffff", "#c9b6ff", "#ff9fb2"];
    for (let i = 0; i < size / 4; i++) {
      const x = rng.random() * size;
      const y = rng.random() * size;
      ctx.fillStyle = petals[Math.floor(rng.random() * petals.length)];
      for (let p = 0; p < 4; p++) {
        ctx.beginPath();
        ctx.arc(
          x + Math.cos((p * Math.PI) / 2) * cell * 0.05,
          y + Math.sin((p * Math.PI) / 2) * cell * 0.05,
          cell * 0.045,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
      ctx.fillStyle = "#f7b733";
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.03, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (kind === "tree" || kind === "rock") {
    for (let ty = 0; ty < 4; ty++) {
      for (let tx = 0; tx < 4; tx++) {
        const cx = tx * cell + cell / 2 + (rng.random() - 0.5) * cell * 0.1;
        const cy = ty * cell + cell / 2 + (rng.random() - 0.5) * cell * 0.1;
        ctx.fillStyle = "rgba(0,0,0,0.22)";
        ctx.beginPath();
        ctx.ellipse(
          cx + cell * 0.06,
          cy + cell * 0.3,
          cell * 0.38,
          cell * 0.14,
          0,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        if (kind === "tree") {
          const greens =
            ground === "snow"
              ? ["#2f6b4f", "#3f8a66", "#e8f1f8"]
              : ["#2f7d3a", "#3f9a48", "#6cc46a"];
          for (const [dx, dy, r, col] of [
            [0, 0.02, 0.44, greens[0]],
            [-0.1, -0.06, 0.3, greens[1]],
            [0.08, -0.12, 0.2, greens[2]],
          ] as [number, number, number, string][]) {
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(cx + dx * cell, cy + dy * cell, r * cell, 0, Math.PI * 2);
            ctx.fill();
          }
        } else {
          ctx.fillStyle = "#8a8f99";
          ctx.beginPath();
          ctx.ellipse(
            cx,
            cy + cell * 0.08,
            cell * 0.34,
            cell * 0.26,
            0,
            0,
            Math.PI * 2,
          );
          ctx.fill();
          ctx.fillStyle = "#b3b8c2";
          ctx.beginPath();
          ctx.ellipse(
            cx - cell * 0.08,
            cy - cell * 0.02,
            cell * 0.16,
            cell * 0.1,
            -0.4,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
      }
    }
  }
  return c;
}

/** A front-facing chibi character, `size` pixels tall, coloured from its id. */
export function paintChibi(id: string, size: number): HTMLCanvasElement {
  const rng = new Prng(seedFrom("chibi", id));
  const hue = Math.floor(rng.random() * 360);
  const skins = ["#ffdcb8", "#f1c27d", "#c68642", "#8d5524", "#ffe0d0"];
  const skin = skins[Math.floor(rng.random() * skins.length)];
  const hair = `hsl(${Math.floor(rng.random() * 360)} ${40 + rng.random() * 40}% ${20 + rng.random() * 40}%)`;
  const [c, ctx] = canvas(size * 0.8, size);
  const w = c.width;
  const u = size / 16;
  ctx.lineWidth = Math.max(1, u * 0.45);
  ctx.strokeStyle = "rgba(30,20,30,0.85)";
  // body
  ctx.fillStyle = `hsl(${hue} 60% 55%)`;
  ctx.beginPath();
  ctx.roundRect(w / 2 - 3.6 * u, 8.6 * u, 7.2 * u, 6.2 * u, 2.4 * u);
  ctx.fill();
  ctx.stroke();
  // feet
  ctx.fillStyle = `hsl(${hue} 35% 30%)`;
  for (const dx of [-1.8, 1.8]) {
    ctx.beginPath();
    ctx.ellipse(w / 2 + dx * u, 15 * u, 1.5 * u, 0.9 * u, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  // head
  ctx.fillStyle = skin;
  ctx.beginPath();
  ctx.arc(w / 2, 5.6 * u, 4.6 * u, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  // hair
  ctx.fillStyle = hair;
  ctx.beginPath();
  ctx.arc(w / 2, 4.8 * u, 4.7 * u, Math.PI * 1.02, Math.PI * 1.98);
  ctx.fill();
  // eyes and blush
  ctx.fillStyle = "#1f1724";
  for (const dx of [-1.6, 1.6]) {
    ctx.beginPath();
    ctx.ellipse(w / 2 + dx * u, 6.2 * u, 0.55 * u, 0.8 * u, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "rgba(255,120,140,0.45)";
  for (const dx of [-2.7, 2.7]) {
    ctx.beginPath();
    ctx.ellipse(w / 2 + dx * u, 7.5 * u, 0.9 * u, 0.5 * u, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  return c;
}

/** A sparkling gem for items without art. */
export function paintItem(id: string, size: number): HTMLCanvasElement {
  const rng = new Prng(seedFrom("item", id));
  const hue = Math.floor(rng.random() * 360);
  const [c, ctx] = canvas(size, size);
  const m = size / 2;
  const grad = ctx.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, `hsl(${hue} 90% 80%)`);
  grad.addColorStop(1, `hsl(${hue} 70% 45%)`);
  ctx.fillStyle = grad;
  ctx.strokeStyle = "rgba(30,20,40,0.8)";
  ctx.lineWidth = Math.max(1, size / 20);
  ctx.beginPath();
  ctx.moveTo(m, size * 0.1);
  ctx.lineTo(size * 0.85, size * 0.42);
  ctx.lineTo(m, size * 0.9);
  ctx.lineTo(size * 0.15, size * 0.42);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.beginPath();
  ctx.moveTo(m, size * 0.18);
  ctx.lineTo(size * 0.62, size * 0.4);
  ctx.lineTo(m, size * 0.45);
  ctx.closePath();
  ctx.fill();
  return c;
}

/** Draw `img` scaled to fit a `w`x`h` canvas (aspect kept), bottom-centred. */
export function fitImage(
  img: CanvasImageSource & { width: number; height: number },
  w: number,
  h: number,
): HTMLCanvasElement {
  const [c, ctx] = canvas(w, h);
  const s = Math.min(w / img.width, h / img.height);
  const dw = img.width * s;
  const dh = img.height * s;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, (w - dw) / 2, h - dh, dw, dh);
  return c;
}

/** Scale an image to cover a `size` square (a tileable texture source). */
export function coverImage(
  img: CanvasImageSource & { width: number; height: number },
  size: number,
): HTMLCanvasElement {
  const [c, ctx] = canvas(size, size);
  const s = Math.max(size / img.width, size / img.height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    img,
    (size - img.width * s) / 2,
    (size - img.height * s) / 2,
    img.width * s,
    img.height * s,
  );
  return c;
}
