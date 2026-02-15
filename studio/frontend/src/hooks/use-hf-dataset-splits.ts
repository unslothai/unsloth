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
  /** All unique config (subset) names found in the dataset */
  configs: string[];
  /** All split names available for the currently selected config */
  splits: string[];
  /** Raw split entries from the API */
  entries: HfSplitEntry[];
  /** Whether the dataset has more than one config */
  hasMultipleConfigs: boolean;
  /** Whether the selected config has more than one split */
  hasMultipleSplits: boolean;
  /** True while the request is in-flight */
  isLoading: boolean;
  /** Error message if the fetch failed */
  error: string | null;
}

const HF_SPLITS_API = "https://datasets-server.huggingface.co/splits";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Fetches the available configs (subsets) and splits for a HuggingFace dataset
 * using the datasets-server API.
 *
 * @param datasetName - HF dataset id (e.g. "ibm/duorc"), or null to skip.
 * @param selectedConfig - Currently selected config, used to filter splits.
 * @param options.accessToken - Optional HF access token for gated datasets.
 */
export function useHfDatasetSplits(
  datasetName: string | null,
  selectedConfig: string | null,
  options?: { accessToken?: string },
): HfDatasetSplitsResult {
  const [entries, setEntries] = useState<HfSplitEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          setError(err.message || "Failed to fetch dataset splits");
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

  // Derive unique configs
  const configs = Array.from(new Set(entries.map((e) => e.config)));

  // Derive splits for the selected config (or all splits if no config selected)
  const filteredEntries = selectedConfig
    ? entries.filter((e) => e.config === selectedConfig)
    : entries;
  const splits = Array.from(new Set(filteredEntries.map((e) => e.split)));

  return {
    configs,
    splits,
    entries,
    hasMultipleConfigs: configs.length > 1,
    hasMultipleSplits: splits.length > 1,
    isLoading,
    error,
  };
}
