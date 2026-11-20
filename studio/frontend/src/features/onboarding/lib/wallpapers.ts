// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Painted, not shipped: a few soft gradients cost no bundle and become an ordinary uploaded wallpaper.

type Glow = { x: number; y: number; r: number; color: string };

export type WallpaperPreset = {
  id: string;
  name: string;
  base: string;
  glows: Glow[];
};

export const WALLPAPER_PRESETS: WallpaperPreset[] = [
  {
    id: "dawn",
    name: "Dawn",
    base: "#f5d9cc",
    glows: [
      { x: 0.15, y: 0.25, r: 0.65, color: "#ffb38a" },
      { x: 0.85, y: 0.2, r: 0.6, color: "#f39bbd" },
      { x: 0.55, y: 0.95, r: 0.7, color: "#c3b1f0" },
    ],
  },
  {
    id: "lagoon",
    name: "Lagoon",
    base: "#0b2747",
    glows: [
      { x: 0.2, y: 0.8, r: 0.7, color: "#1fa2c7" },
      { x: 0.8, y: 0.2, r: 0.6, color: "#5ee2c2" },
      { x: 0.55, y: 0.45, r: 0.5, color: "#2a5bd7" },
    ],
  },
  {
    id: "dusk",
    name: "Dusk",
    base: "#1a1031",
    glows: [
      { x: 0.2, y: 0.25, r: 0.65, color: "#6d3fd1" },
      { x: 0.85, y: 0.75, r: 0.6, color: "#ff7a9a" },
      { x: 0.5, y: 1, r: 0.6, color: "#2b2f8a" },
    ],
  },
  {
    id: "graphite",
    name: "Graphite",
    base: "#17181c",
    glows: [
      { x: 0.25, y: 0.15, r: 0.7, color: "#3a3d46" },
      { x: 0.85, y: 0.85, r: 0.6, color: "#2a2c33" },
      { x: 0.7, y: 0.25, r: 0.4, color: "#4b5160" },
    ],
  },
  {
    id: "mist",
    name: "Mist",
    base: "#dfe5ec",
    glows: [
      { x: 0.2, y: 0.2, r: 0.7, color: "#f4f6f9" },
      { x: 0.85, y: 0.75, r: 0.6, color: "#b9c6d6" },
      { x: 0.6, y: 0.1, r: 0.45, color: "#d3d9f0" },
    ],
  },
  {
    id: "sand",
    name: "Sand",
    base: "#efe6d8",
    glows: [
      { x: 0.2, y: 0.2, r: 0.7, color: "#f7f1e7" },
      { x: 0.8, y: 0.7, r: 0.65, color: "#d9bf98" },
      { x: 0.4, y: 0.95, r: 0.5, color: "#c79f74" },
    ],
  },
];

/** Hex #rrggbb at alpha 0, so a glow fades to its own colour rather than through grey. */
function clear(color: string): string {
  return `${color}00`;
}

export function paintWallpaper(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  preset: WallpaperPreset,
): void {
  ctx.fillStyle = preset.base;
  ctx.fillRect(0, 0, width, height);
  const reach = Math.hypot(width, height);
  for (const glow of preset.glows) {
    const x = glow.x * width;
    const y = glow.y * height;
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, glow.r * reach);
    gradient.addColorStop(0, glow.color);
    gradient.addColorStop(1, clear(glow.color));
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
  }
}
