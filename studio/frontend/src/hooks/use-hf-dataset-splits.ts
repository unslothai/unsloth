


import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import {
  type DatasetSplitFetchers,
  type HfSplitEntry,
  type LoadHfDatasetSplitsArgs,
  loadHfDatasetSplits,
  normalizeDatasetSplitsError,
} from "./hf-dataset-split-sources";

export type { HfSplitEntry } from "./hf-dataset-split-sources";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  requiresManualEntry: boolean;
}

const HF_SPLITS_API = "https://datasets-server.huggingface.co/splits";
const MAX_SPLIT_ENTRIES = 2048;

function validatedEntries(value: unknown): HfSplitEntry[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const entries: HfSplitEntry[] = [];
  for (const item of value.slice(0, MAX_SPLIT_ENTRIES)) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const candidate = item as Partial<HfSplitEntry>;
    if (
      typeof candidate.dataset !== "string" ||
      typeof candidate.config !== "string" ||
      typeof candidate.split !== "string" ||
      !candidate.config.trim() ||
      !candidate.split.trim()
    ) {
      continue;
    }
    entries.push({
      dataset: candidate.dataset,
      config: candidate.config,
      split: candidate.split,
    });
  }
  return entries;
}

async function fetchLocalSplits({
  datasetName,
  localPath,
  signal,
}: LoadHfDatasetSplitsArgs): Promise<HfSplitEntry[]> {
  const response = await authFetch("/api/hub/datasets/local-options", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: datasetName,
      local_path: localPath ?? null,
    }),
    signal,
  });
  if (!response.ok) {
    throw new Error(
      `Failed to read cached dataset metadata (${response.status})`,
    );
  }
  const payload = (await response.json()) as { splits?: unknown };
  return validatedEntries(payload.splits);
}

async function fetchRemoteSplits({
  accessToken,
  datasetName,
  signal,
}: LoadHfDatasetSplitsArgs): Promise<HfSplitEntry[]> {
  const url = `${HF_SPLITS_API}?dataset=${encodeURIComponent(datasetName)}`;
  const headers: Record<string, string> = {};
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }
  const response = await fetch(url, { headers, signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.error || `Failed to fetch splits (${response.status})`,
    );
  }
  const payload = (await response.json()) as HfSplitsResponse;
  return validatedEntries(payload.splits);
}

const DEFAULT_FETCHERS: DatasetSplitFetchers = {
  local: fetchLocalSplits,
  remote: fetchRemoteSplits,
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useHfDatasetSplits(
  datasetName: string | null,
  selectedSubset: string | null,
  options?: {
    accessToken?: string;
    localPath?: string | null;
    online?: boolean;
    preferLocalCache?: boolean;
  },
): HfDatasetSplitsResult {
  const [entries, setEntries] = useState<HfSplitEntry[]>([]);
  const [isLoading, setIsLoading] = useState(datasetName !== null);
  const [error, setError] = useState<string | null>(null);
  const requestKey = JSON.stringify([
    datasetName,
    options?.preferLocalCache === true,
    options?.localPath ?? null,
    options?.online !== false,
  ]);
  const [previousRequestKey, setPreviousRequestKey] = useState(requestKey);
  if (requestKey !== previousRequestKey) {
    setPreviousRequestKey(requestKey);
    setEntries([]);
    setError(null);
    setIsLoading(datasetName !== null);
  }

  const accessToken = options?.accessToken;
  const localPath = options?.localPath;
  const online = options?.online ?? true;
  const preferLocalCache = options?.preferLocalCache ?? false;

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

    loadHfDatasetSplits(
      {
        datasetName,
        accessToken,
        localPath,
        online,
        preferLocalCache,
        signal: controller.signal,
      },
      DEFAULT_FETCHERS,
    )
      .then((result) => {
        if (!controller.signal.aborted) {
          setEntries(result.entries);
          setError(result.error);
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
  }, [accessToken, datasetName, localPath, online, preferLocalCache]);

  // Derive unique subsets
  const subsets = Array.from(new Set(entries.map((e) => e.config)));

  // Splits for the active subset. With >1 subset and none selected, return no
  // splits so the UI doesn't auto-pick before a subset is chosen.
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
    requiresManualEntry:
      datasetName !== null && !isLoading && entries.length === 0,
  };
}
