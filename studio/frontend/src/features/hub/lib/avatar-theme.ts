// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const PALETTE = [
  "hsl(214 42% 42%)",
  "hsl(172 32% 36%)",
  "hsl(30 42% 42%)",
  "hsl(348 36% 44%)",
  "hsl(198 38% 40%)",
  "hsl(140 26% 38%)",
  "hsl(222 24% 46%)",
  "hsl(16 34% 44%)",
  "hsl(260 22% 48%)",
  "hsl(44 44% 40%)",
];

function hashOwner(owner: string): number {
  let h = 0;
  for (let i = 0; i < owner.length; i++) {
    h = (h * 31 + owner.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function ownerPaletteColor(owner: string): string {
  const owned = owner.trim() || "?";
  return PALETTE[hashOwner(owned) % PALETTE.length];
}
