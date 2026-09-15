// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Telling "loading a cached model" apart from "downloading it again".
 *
 * A load that starts believing the weights are on disk can still turn into a transfer: the
 * backend re-fetches a blob it judged unsafe to resume, or a revision whose cached copy it could
 * not confirm. Reported in #9094, where the toast said "Loading cached model into memory." for
 * fourteen minutes while bytes arrived from Hugging Face and the load then failed.
 *
 * The rule, kept here rather than inside the load hook so it can be tested on its own:
 *
 *  - A byte count BELOW the expected total proves nothing. That is the ordinary state of a
 *    partially fetched revision, and a load reading one is not a transfer.
 *  - A count that GREW between two readings is a transfer, and nothing else is.
 *  - A complete reading (progress at or past 1) resets the watch, so a later transfer still
 *    needs two readings of its own rather than inheriting a stale one.
 *  - An unreadable or absent measurement (`cache_measured` false) is not evidence either way and
 *    leaves the watch as it was, rather than resetting it and forfeiting the comparison.
 */

/** The bytes seen by the previous reading, or null when there is nothing to compare against. */
export interface CacheMissWatch {
  readonly bytes: number | null;
}

/** The fields of the download-progress response this decision reads. */
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
  /** Whether this cached load has been proven to be downloading. */
  readonly started: boolean;
  /** The watch to carry into the next reading. */
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
