// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface HfSplitEntry {
  dataset: string;
  config: string;
  split: string;
}

export type LoadHfDatasetSplitsArgs = {
  datasetName: string;
  accessToken?: string;
  localPath?: string | null;
  online: boolean;
  preferLocalCache: boolean;
  signal: AbortSignal;
};

export type DatasetSplitLoadResult = {
  entries: HfSplitEntry[];
  error: string | null;
  source: "local" | "remote" | "manual";
};

export type DatasetSplitFetchers = {
  local: (args: LoadHfDatasetSplitsArgs) => Promise<HfSplitEntry[]>;
  remote: (args: LoadHfDatasetSplitsArgs) => Promise<HfSplitEntry[]>;
};

// Matched verbatim below, so the message survives normalisation unchanged.
export const MIRROR_SPLITS_UNAVAILABLE =
  "Subset and split options come from the public Hugging Face datasets-server, which is not used when a custom Hugging Face endpoint is configured.";

export function normalizeDatasetSplitsError(message: string): string {
  if (message === MIRROR_SPLITS_UNAVAILABLE) {
    return message;
  }
  const normalized = message.toLowerCase();
  if (
    normalized.includes("dataset scripts are no longer supported") ||
    normalized.includes("runs arbitrary python code")
  ) {
    return "We can’t load subset/split options for this Hub dataset because it relies on a legacy custom script.";
  }
  if (
    normalized.includes("unauthorized") ||
    normalized.includes("forbidden") ||
    normalized.includes("access token") ||
    normalized.includes("private") ||
    normalized.includes("gated") ||
    normalized.includes("401") ||
    normalized.includes("403")
  ) {
    return "Unable to load dataset splits. This dataset may be private or gated. Add a Hugging Face token with access and try again.";
  }
  if (normalized.includes("not found") || normalized.includes("404")) {
    return "Dataset not found. Check the dataset name and try again.";
  }
  return "Unable to load dataset split options for this dataset.";
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new DOMException("The operation was aborted", "AbortError");
  }
}

export async function loadHfDatasetSplits(
  args: LoadHfDatasetSplitsArgs,
  fetchers: DatasetSplitFetchers,
): Promise<DatasetSplitLoadResult> {
  if (args.preferLocalCache) {
    try {
      const entries = await fetchers.local(args);
      throwIfAborted(args.signal);
      if (entries.length > 0) {
        return { entries, error: null, source: "local" };
      }
    } catch {
      throwIfAborted(args.signal);
    }
  }

  if (args.online) {
    try {
      const entries = await fetchers.remote(args);
      throwIfAborted(args.signal);
      if (entries.length > 0) {
        return { entries, error: null, source: "remote" };
      }
    } catch (error) {
      throwIfAborted(args.signal);
      const message =
        error instanceof Error
          ? error.message
          : "Failed to fetch dataset splits";
      return {
        entries: [],
        error: normalizeDatasetSplitsError(message),
        source: "manual",
      };
    }
  }

  return {
    entries: [],
    error:
      "Dataset config and split metadata is unavailable. Enter the values manually.",
    source: "manual",
  };
}
