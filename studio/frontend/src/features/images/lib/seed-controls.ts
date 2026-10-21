// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

interface SeededImage {
  seed: number;
  batch_seed?: number | null;
  created_at: number;
  model_kind?: string | null;
}

const seedOf = (image: SeededImage): number | null => {
  // Connection runs never sent a seed, so theirs reproduces nothing.
  if (image.model_kind === "connection") return null;
  const seed = image.batch_seed ?? image.seed;
  return Number.isSafeInteger(seed) && seed >= 0 ? seed : null;
};

/** The base seed one run used: its batch's shared seed, so replaying the batch gives the same images. */
export function lastSeedOf(run: readonly SeededImage[]): number | null {
  for (const image of run) {
    const seed = seedOf(image);
    if (seed !== null) return seed;
  }
  return null;
}

/** `hasMore`: the gallery is only partly loaded. Pages follow the pin and manual order, not creation
 *  time, so the newest record may be unseen and no gallery seed is offered. */
export function reusableSeed(
  sessionSeed: number | null,
  gallery: readonly SeededImage[],
  hasMore = false,
): number | null {
  if (sessionSeed !== null) return sessionSeed;
  if (hasMore) return null;
  let newest: SeededImage | null = null;
  for (const image of gallery) {
    if (
      seedOf(image) !== null &&
      (newest === null || image.created_at > newest.created_at)
    )
      newest = image;
  }
  return newest ? seedOf(newest) : null;
}
