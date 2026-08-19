// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const PALETTE = [
  "hsl(214 66% 52%)",
  "hsl(172 54% 44%)",
  "hsl(30 70% 52%)",
  "hsl(348 62% 56%)",
  "hsl(198 64% 50%)",
  "hsl(140 48% 46%)",
  "hsl(222 50% 56%)",
  "hsl(16 64% 54%)",
  "hsl(260 46% 58%)",
  "hsl(44 72% 50%)",
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
