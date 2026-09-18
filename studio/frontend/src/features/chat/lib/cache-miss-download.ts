// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Telling "loading a cached model" apart from "downloading it again" (#9094).
 *
 * Only a count that GREW between two readings is a transfer. Below the expected total proves
 * nothing (that is an ordinary partial revision), a complete reading resets the watch, and an
 * unmeasurable cache leaves the watch alone rather than forfeiting the comparison.
 */

/** Previous reading's bytes; null when there is nothing to compare against. */
export interface CacheMissWatch {
  readonly bytes: number | null;
}

export interface CacheMissReading {
  readonly downloaded_bytes?: number | null;
  readonly expected_bytes?: number | null;
  readonly progress?: number | null;
  readonly cache_measured?: boolean | null;
}

export const EMPTY_CACHE_MISS_WATCH: CacheMissWatch = { bytes: null };

/** What the load toast says once a cached load is seen to be downloading after all. */
export const CACHE_MISS_DOWNLOAD_DESCRIPTION =
  "The cached copy is incomplete, so the missing files are downloading from Hugging Face.";

export interface CacheMissVerdict {
  readonly started: boolean;
  readonly watch: CacheMissWatch;
  /** Whole percent, or null when the total is unknown and only bytes are countable. */
  readonly percent: number | null;
}

export function watchCacheMissDownload(
  watch: CacheMissWatch,
  reading: CacheMissReading | null | undefined,
): CacheMissVerdict {
  if (!reading || reading.cache_measured === false) {
    return { started: false, watch, percent: null };
  }
  const progress = reading.progress ?? 0;
  if (progress >= 1) {
    return { started: false, watch: EMPTY_CACHE_MISS_WATCH, percent: null };
  }
  const bytes = reading.downloaded_bytes ?? 0;
  if (!Number.isFinite(bytes) || bytes < 0) {
    return { started: false, watch, percent: null };
  }
  const previous = watch.bytes;
  if (previous != null && bytes > previous) {
    const expected = reading.expected_bytes ?? 0;
    return {
      started: true,
      watch: { bytes },
      percent: expected > 0 ? Math.round(progress * 100) : null,
    };
  }
  return { started: false, watch: { bytes }, percent: null };
}
