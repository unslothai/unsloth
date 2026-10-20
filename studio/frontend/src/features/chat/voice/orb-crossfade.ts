// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb body's colour layers. A new state's gradient fades in on top; a layer
 * drops the ones under it only once its own fade has finished. Swapping the stack
 * down to two on every change made a layer still mid-fade the base, which lost its
 * fade and jumped to full colour: the blink on entering "hearing", which often
 * lands within a fade of the state before it.
 */
export type FadeLayer = { g: string; id: number; fading: boolean };

/** More would mean a state change every few frames; the oldest go first. */
export const MAX_FADE_LAYERS = 6;

export function pushGradient(layers: FadeLayer[], g: string): FadeLayer[] {
  const top = layers[layers.length - 1];
  if (top?.g === g) return layers;
  const next = [...layers, { g, id: (top?.id ?? -1) + 1, fading: top !== undefined }];
  return next.length > MAX_FADE_LAYERS ? next.slice(-MAX_FADE_LAYERS) : next;
}

/** Layer `id` is fully shown: everything under it is hidden and can go. */
export function settleGradient(layers: FadeLayer[], id: number): FadeLayer[] {
  const i = layers.findIndex((l) => l.id === id);
  if (i < 0) return layers;
  return [{ ...layers[i], fading: false }, ...layers.slice(i + 1)];
}
