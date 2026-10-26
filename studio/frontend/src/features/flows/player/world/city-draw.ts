// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Draws the Mayor mode layer (zones, roads, power lines, grown buildings, overlays) onto a Phaser Graphics.
// Called after a sim cycle or a placement, never per frame.

import { type CitySim, type Overlay, ROAD, ROADWIRE, WIRE } from "./city-sim";

export interface Painter {
  clear(): unknown;
  fillStyle(color: number, alpha?: number): unknown;
  fillRect(x: number, y: number, w: number, h: number): unknown;
  fillTriangle(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): unknown;
  lineStyle(width: number, color: number, alpha?: number): unknown;
  strokeRect(x: number, y: number, w: number, h: number): unknown;
  lineBetween(x0: number, y0: number, x1: number, y1: number): unknown;
}

const ZONE_TINT: Record<string, number> = {
  R: 0x5cb85c,
  C: 0x4a90d9,
  I: 0xe0b040,
  P: 0x8a8a8a,
};
// Same wall colour as the 2D house prop marks; roofs from the house generator palette.
const HOUSE = 0xe9d7b5;
const ROOF = 0xb5543c;
const RING = [
  [-1, -1],
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
];

function house(g: Painter, x: number, y: number, T: number): void {
  g.fillStyle(HOUSE, 1);
  g.fillRect(x + T * 0.22, y + T * 0.42, T * 0.56, T * 0.4);
  g.fillStyle(ROOF, 1);
  g.fillTriangle(
    x + T * 0.14,
    y + T * 0.44,
    x + T * 0.86,
    y + T * 0.44,
    x + T * 0.5,
    y + T * 0.12,
  );
  g.fillStyle(0x6b4a2b, 1);
  g.fillRect(x + T * 0.44, y + T * 0.6, T * 0.12, T * 0.22);
}

function bolt(g: Painter, cx: number, cy: number, T: number): void {
  g.fillStyle(0xd9534f, 0.9);
  g.fillRect(cx - T * 0.3, cy - T * 0.3, T * 0.6, T * 0.6);
  g.fillStyle(0xffe14d, 1);
  g.fillTriangle(
    cx + T * 0.06,
    cy - T * 0.26,
    cx - T * 0.16,
    cy + T * 0.04,
    cx + T * 0.02,
    cy + T * 0.04,
  );
  g.fillTriangle(
    cx - T * 0.04,
    cy - T * 0.02,
    cx + T * 0.16,
    cy - T * 0.02,
    cx - T * 0.06,
    cy + T * 0.26,
  );
}

function overlayValue(sim: CitySim, overlay: Overlay, c: number): number {
  if (overlay === "traffic") return sim.traffic[c];
  if (overlay === "pollution") return sim.pollution[c];
  if (overlay === "land_value") return sim.land[c];
  return sim.crime[c];
}

export function drawCity(
  g: Painter,
  sim: CitySim,
  T: number,
  overlay: Overlay | null,
): void {
  g.clear();
  const w = sim.w;
  for (let y = 0; y < sim.h; y++)
    for (let x = 0; x < w; x++) {
      const k = sim.kind[y * w + x];
      const px = x * T;
      const py = y * T;
      if (k === ROAD || k === ROADWIRE) {
        g.fillStyle(0x4a4a52, 1);
        g.fillRect(px, py, T, T);
        g.fillStyle(0xf2e6a0, 0.8);
        g.fillRect(px + T * 0.45, py + T * 0.2, T * 0.1, T * 0.2);
        g.fillRect(px + T * 0.45, py + T * 0.6, T * 0.1, T * 0.2);
      }
      if (k === WIRE || k === ROADWIRE) {
        g.lineStyle(Math.max(1, T / 16), 0x2b2b2b, 0.9);
        g.lineBetween(px, py + T * 0.3, px + T, py + T * 0.3);
        g.fillStyle(0x8b5a2b, 1);
        g.fillRect(px + T * 0.44, py + T * 0.1, T * 0.12, T * 0.5);
      }
    }
  for (const z of sim.zones) {
    const x0 = (z.x - 1) * T;
    const y0 = (z.y - 1) * T;
    g.fillStyle(ZONE_TINT[z.kind], 0.3);
    g.fillRect(x0, y0, T * 3, T * 3);
    g.lineStyle(Math.max(1, T / 16), ZONE_TINT[z.kind], 0.9);
    g.strokeRect(x0 + 1, y0 + 1, T * 3 - 2, T * 3 - 2);
    const cx = z.x * T + T / 2;
    const cy = z.y * T + T / 2;
    if (z.kind === "R") {
      if (z.pop <= 8)
        for (let k = 0; k < z.pop; k++)
          house(g, (z.x + RING[k][0]) * T, (z.y + RING[k][1]) * T, T);
      else {
        const floors = (z.pop - 8) / 8;
        g.fillStyle(0xd8c7a0, 1);
        g.fillRect(x0 + T * 0.3, y0 + T * 0.3, T * 2.4, T * 2.4);
        g.fillStyle(0x5a6b8a, 1);
        for (let f = 0; f < floors; f++)
          g.fillRect(x0 + T * 0.5, y0 + T * (0.5 + f * 0.5), T * 2, T * 0.25);
      }
    } else if (z.kind === "C") {
      for (let k = 0; k < z.pop; k++) {
        const [dx, dy] = RING[[0, 2, 4, 6, 1][k]];
        const bx = (z.x + dx) * T;
        const by = (z.y + dy) * T;
        g.fillStyle(0x6f8fb8, 1);
        g.fillRect(bx + T * 0.15, by + T * 0.1, T * 0.7, T * 0.8);
        g.fillStyle(0xcfe3ff, 1);
        g.fillRect(bx + T * 0.25, by + T * 0.2, T * 0.5, T * 0.12);
      }
    } else if (z.kind === "I") {
      for (let k = 0; k < z.pop; k++) {
        const bx = (z.x + RING[k * 2 + 1][0]) * T;
        const by = (z.y + RING[k * 2 + 1][1]) * T;
        g.fillStyle(0x9a7b4f, 1);
        g.fillRect(bx + T * 0.1, by + T * 0.4, T * 0.8, T * 0.5);
        g.fillStyle(0x5a5a5a, 1);
        g.fillRect(bx + T * 0.65, by + T * 0.1, T * 0.15, T * 0.35);
      }
    } else {
      g.fillStyle(0x6b6b6b, 1);
      g.fillRect(x0 + T * 0.3, y0 + T * 0.6, T * 2.4, T * 2);
      g.fillStyle(0x444444, 1);
      g.fillRect(x0 + T * 2, y0 + T * 0.1, T * 0.4, T * 1);
      g.fillStyle(0xffe14d, 1);
      g.fillTriangle(
        cx,
        cy - T * 0.2,
        cx - T * 0.2,
        cy + T * 0.3,
        cx + T * 0.1,
        cy + T * 0.3,
      );
    }
    if (z.kind !== "P" && !sim.isPowered(z)) bolt(g, cx, cy, T);
  }
  if (!overlay) return;
  if (overlay === "power") {
    for (let i = 0; i < sim.powered.length; i++)
      if (sim.powered[i]) {
        g.fillStyle(0xffe14d, 0.35);
        g.fillRect((i % w) * T, Math.floor(i / w) * T, T, T);
      }
    return;
  }
  const color =
    overlay === "land_value"
      ? 0x3fbf5f
      : overlay === "traffic"
        ? 0xff8c1a
        : overlay === "pollution"
          ? 0x8a5a2b
          : 0xd9534f;
  for (let y = 0; y < sim.hh; y++)
    for (let x = 0; x < sim.hw; x++) {
      const v = overlayValue(sim, overlay, y * sim.hw + x);
      if (!v) continue;
      g.fillStyle(color, Math.min(0.7, v / 300));
      g.fillRect(x * 2 * T, y * 2 * T, T * 2, T * 2);
    }
}
