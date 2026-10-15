// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The Hub's dataset modalities, as its `modality:` tags spell them. */
export const DATASET_MODALITIES = [
  "3d",
  "audio",
  "document",
  "geospatial",
  "image",
  "tabular",
  "text",
  "timeseries",
  "video",
] as const;
export type DatasetModality = (typeof DATASET_MODALITIES)[number];

// The Hub's size_categories buckets, smallest first: bucket i lies between stop i - 1 and stop i.
const SIZE_BUCKETS = [
  "n<1k",
  "1k<n<10k",
  "10k<n<100k",
  "100k<n<1m",
  "1m<n<10m",
  "10m<n<100m",
  "100m<n<1b",
  "1b<n<10b",
  "10b<n<100b",
  "100b<n<1t",
  "n>1t",
];
export const DATASET_ROW_STOPS = [
  1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
] as const;

export function datasetModalities(tags: readonly string[]): string[] {
  return tags
    .filter((tag) => tag.startsWith("modality:"))
    .map((tag) => tag.slice("modality:".length).toLowerCase());
}

export function datasetSizeBucket(tags: readonly string[]): number | undefined {
  for (const tag of tags) {
    if (tag.startsWith("size_categories:")) {
      const index = SIZE_BUCKETS.indexOf(
        tag.slice("size_categories:".length).toLowerCase(),
      );
      if (index >= 0) {
        return index;
      }
    }
  }
  return undefined;
}

/** A dataset with any picked modality (none picked: all) whose size bucket meets the row range.
 *  Datasets that don't publish what a filter asks for are left out of it. */
export function matchesDatasetFilters(
  dataset: { modalities: readonly string[]; sizeBucket?: number },
  modalities: readonly string[],
  minRows: number | undefined,
  maxRows: number | undefined,
): boolean {
  if (
    modalities.length > 0 &&
    !dataset.modalities.some((modality) => modalities.includes(modality))
  ) {
    return false;
  }
  if (minRows === undefined && maxRows === undefined) {
    return true;
  }
  const bucket = dataset.sizeBucket;
  if (bucket === undefined) {
    return false;
  }
  const low = bucket === 0 ? 0 : DATASET_ROW_STOPS[bucket - 1];
  const high =
    bucket === DATASET_ROW_STOPS.length
      ? Number.POSITIVE_INFINITY
      : DATASET_ROW_STOPS[bucket];
  return (
    (minRows === undefined || high > minRows) &&
    (maxRows === undefined || low < maxRows)
  );
}
