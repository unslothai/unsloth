// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HfSplitEntry {
  dataset: string;
  config: string;
  split: string;
}

export interface HfSplitsResponse {
  splits: HfSplitEntry[];
  pending: unknown[];
  failed: unknown[];
}

export interface HfDatasetSplitsResult {
  /** All unique subset names found in the dataset */
  subsets: string[];
  /** All split names available for the currently selected subset */
  splits: string[];
  /** Raw split entries from the API */
  entries: HfSplitEntry[];
  /** Whether the dataset has more than one subset */
  hasMultipleSubsets: boolean;
  /** Whether the selected subset has more than one split */
  hasMultipleSplits: boolean;
  /** True while the request is in-flight */
  isLoading: boolean;
  /** Error message if the fetch failed */
  error: string | null;
}

const HF_SPLITS_API = "https://datasets-server.huggingface.co/splits";

function normalizeDatasetSplitsError(message: string): string {
  const normalized = message.toLowerCase();

  // datasets-server returns technical script/runtime details for legacy datasets.
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

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Fetches the available configs (subsets) and splits for a HuggingFace dataset
 * using the datasets-server API.
 *
 * @param datasetName - HF dataset id (e.g. "ibm/duorc"), or null to skip.
 * @param selectedSubset - Currently selected subset, used to filter splits.
 * @param options.accessToken - Optional HF access token for gated datasets.
 */
export function useHfDatasetSplits(
  datasetName: string | null,
  selectedSubset: string | null,
  options?: { accessToken?: string },
): HfDatasetSplitsResult {
  const [entries, setEntries] = useState<HfSplitEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  
  const [prevDatasetName, setPrevDatasetName] = useState(datasetName);
  if (datasetName !== prevDatasetName) {
    setPrevDatasetName(datasetName);
    setEntries([]);
    setError(null);
  }

  const accessToken = options?.accessToken;

  const fetchSplits = useCallback(
    async (dataset: string, signal: AbortSignal) => {
      const url = `${HF_SPLITS_API}?dataset=${encodeURIComponent(dataset)}`;
      const headers: Record<string, string> = {};
      if (accessToken) {
        headers.Authorization = `Bearer ${accessToken}`;
      }

      const res = await fetch(url, { headers, signal });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(
          body?.error || `Failed to fetch splits (${res.status})`,
        );
      }

      const data: HfSplitsResponse = await res.json();
      return data.splits ?? [];
    },
    [accessToken],
  );

  useEffect(() => {
    if (!datasetName) {
      setEntries([]);
      setError(null);
      setIsLoading(false);
      return;
    }

    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    fetchSplits(datasetName, controller.signal)
      .then((splits) => {
        if (!controller.signal.aborted) {
          setEntries(splits);
          setError(null);
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          const rawErrorMessage =
            err instanceof Error
              ? err.message
              : typeof err === "string"
                ? err
                : "Failed to fetch dataset splits";
          console.warn("[useHfDatasetSplits] Failed to fetch dataset splits", {
            datasetName,
            message: rawErrorMessage,
            error: err,
          });
          setError(normalizeDatasetSplitsError(rawErrorMessage));
          setEntries([]);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => controller.abort();
  }, [datasetName, fetchSplits]);

  // Derive unique subsets
  const subsets = Array.from(new Set(entries.map((e) => e.config)));

  // Derive splits for the active subset.
  // If dataset has >1 subset and none is selected yet, return no splits so UI
  // doesn't auto-pick/show a split before subset is chosen.
  const activeSubset =
    selectedSubset ?? (subsets.length === 1 ? subsets[0] : null);
  const filteredEntries = activeSubset
    ? entries.filter((e) => e.config === activeSubset)
    : [];
  const splits = Array.from(new Set(filteredEntries.map((e) => e.split)));

  return {
    subsets,
    splits,
    entries,
    hasMultipleSubsets: subsets.length > 1,
    hasMultipleSplits: activeSubset ? splits.length > 1 : false,
    isLoading,
    error,
  };
}
