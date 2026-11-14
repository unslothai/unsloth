// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FadeLayer = { g: string; id: number; fading: boolean };

export const MAX_FADE_LAYERS = 6;

export function pushGradient(layers: FadeLayer[], g: string): FadeLayer[] {
  const top = layers[layers.length - 1];
  if (top?.g === g) return layers;
  const next = [...layers, { g, id: (top?.id ?? -1) + 1, fading: top !== undefined }];
  return next.length > MAX_FADE_LAYERS ? next.slice(-MAX_FADE_LAYERS) : next;
}

// A layer drops those under it only once its own fade ends: cutting mid-fade made the base jump (a blink).
export function settleGradient(layers: FadeLayer[], id: number): FadeLayer[] {
  const i = layers.findIndex((l) => l.id === id);
  if (i < 0) return layers;
  return [{ ...layers[i], fading: false }, ...layers.slice(i + 1)];
}
