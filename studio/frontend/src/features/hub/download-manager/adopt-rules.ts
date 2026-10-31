// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether an adopted job keeps persisted byte counters. Not for a different generation:
 * the new run's zero would read as "could not measure" and pin the old bytes.
 */
export function carriesOverSeed(
  adopt: boolean,
  persistedGeneration: number | undefined,
  probedGeneration: number | undefined,
): boolean {
  if (!adopt) return false;
  // Unknown generation is not evidence of a new run (adopt-after-reload path), so keep the seed.
  if (!Number.isSafeInteger(probedGeneration) || persistedGeneration === undefined) return true;
  return persistedGeneration === probedGeneration;
}

/** The held-transfer marker travels with the counters it describes. */
export function seededMeasuredTransfer(
  carryOverSeed: boolean,
  persistedMeasuredTransfer: boolean | undefined,
): boolean | undefined {
  return carryOverSeed ? persistedMeasuredTransfer : undefined;
}

/**
 * "active" or "gone" from one reading. Zero alone is not a wipe; `cache_path` null,
 * `cache_measured`, and per-variant `target_present` decide. Unknown keeps the job.
 */
export function idleProbeVerdict(
  downloadedBytes: number,
  cachePath: string | null | undefined,
  targetPresent?: boolean | null,
  cacheMeasured?: boolean,
): "active" | "gone" {
  if (cacheMeasured === false) return "active";
  // Explicit target verdict beats bytes, which may count a shared companion that outlived the shard.
  if (targetPresent === false) return "gone";
  if (downloadedBytes > 0) return "active";
  // The GGUF route excludes None, so a measured-empty answer omits cache_path: treat as gone.
  if (cacheMeasured === true) return cachePath == null ? "gone" : "active";
  return cachePath === null ? "gone" : "active";
}
