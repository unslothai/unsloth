// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Painted, not shipped: a flat tone costs no bundle and becomes an ordinary uploaded wallpaper.

export type WallpaperPreset = { id: string; name: string; color: string };

export const WALLPAPER_PRESETS: WallpaperPreset[] = [
  { id: "sand", name: "Sand", color: "#e6ddce" },
  { id: "sage", name: "Sage", color: "#c5d1c2" },
  { id: "mist", name: "Mist", color: "#d3dbe5" },
  { id: "clay", name: "Clay", color: "#c4a08b" },
  { id: "slate", name: "Slate", color: "#3b4049" },
  { id: "ink", name: "Ink", color: "#22304a" },
];

export function paintWallpaper(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  preset: WallpaperPreset,
): void {
  ctx.fillStyle = preset.color;
  ctx.fillRect(0, 0, width, height);
}
