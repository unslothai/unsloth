// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Two decisions the download manager makes from a single progress reading.
 *
 * A leaf module on purpose: both are pure, both used to be wrong in the same way (reading a zero
 * byte count as evidence of something it is not), and living here they can be exercised without
 * dragging the poll loop's or the hydrator's import graph into a test.
 */

/**
 * Whether an adopted job keeps the persisted byte counters.
 *
 * Adopting a DIFFERENT generation means the run those bytes described is over and a new one is in
 * flight. Seeding them anyway, while `serverGeneration` jumps straight to the new value, leaves
 * the first poll seeing no generation change: the new run's legitimate zero reads as "could not
 * measure" and the card stays pinned to the previous run's bytes until a positive count arrives.
 */
export function carriesOverSeed(
  adopt: boolean,
  persistedGeneration: number | undefined,
  probedGeneration: number | undefined,
): boolean {
  if (!adopt) return false;
  // An unknown generation on either side is not evidence of a new run, so the seed is kept:
  // that is the adopt-after-reload path, where the counters are all we have.
  if (!Number.isSafeInteger(probedGeneration) || persistedGeneration === undefined) return true;
  return persistedGeneration === probedGeneration;
}

/**
 * "active" (the job is adoptable) or "gone" (its cache was wiped), from one raw reading.
 *
 * A zero alone is not proof of a wipe: a transient measurement failure comes back as a perfectly
 * successful all-zero response, and calling that "gone" drops a job whose partial cache is sitting
 * right there. `cache_path` is the discriminator -- the backend answers null only when no cache dir
 * for this repo exists at all, and a dir it scanned and measured at zero still names itself. An
 * older backend that omits the field entirely is unknown rather than absent, so the job survives.
 */
export function idleProbeVerdict(
  downloadedBytes: number,
  cachePath: string | null | undefined,
): "active" | "gone" {
  if (downloadedBytes > 0) return "active";
  return cachePath === null ? "gone" : "active";
}
