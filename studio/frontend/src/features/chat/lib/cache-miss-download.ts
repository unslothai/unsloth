// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Telling "loading a cached model" apart from "downloading it again" (#9094).
 *
 * Kept out of the load hook so the rule can be tested on its own:
 *
 *  - A byte count BELOW the expected total proves nothing: that is the ordinary state of a
 *    partially fetched revision.
 *  - A count that GREW between two readings is a transfer, and nothing else is.
 *  - A complete reading (progress at or past 1) resets the watch.
 *  - An unreadable measurement (`cache_measured` false) is not evidence either way and leaves
 *    the watch as it was, rather than forfeiting the comparison.
 */

/** The bytes seen by the previous reading, or null when there is nothing to compare against. */
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
/** Whole percent to show, or null when the total is unknown and only bytes are countable. */
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
