// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";
import {
  getHfDatasetsServerBase,
  useHfDatasetsServer,
  useHfEndpoint,
  useHubSource,
} from "@/lib/hf-endpoint";
import { hubFetch } from "@/lib/hub-fetch";
import { useEffect, useState } from "react";
import {
  type DatasetSplitFetchers,
  type HfSplitEntry,
  type LoadHfDatasetSplitsArgs,
  loadHfDatasetSplits,
  normalizeDatasetSplitsError,
} from "./hf-dataset-split-sources";

export type { HfSplitEntry } from "./hf-dataset-split-sources";

export interface HfSplitsResponse {
  splits: HfSplitEntry[];
  pending: unknown[];
  failed: unknown[];
}

export interface HfDatasetSplitsResult {
  subsets: string[];
  splits: string[];
  entries: HfSplitEntry[];
  hasMultipleSubsets: boolean;
  hasMultipleSplits: boolean;
  isLoading: boolean;
  error: string | null;
  requiresManualEntry: boolean;
}

// HF_ENDPOINT does not redirect datasets-server; only HF_DATASETS_SERVER (via /api/health) does.
function getHfSplitsApi(): string {
  return `${getHfDatasetsServerBase()}/splits`;
}
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
  const url = `${getHfSplitsApi()}?dataset=${encodeURIComponent(datasetName)}`;
  const headers: Record<string, string> = {};
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }
  const response = await hubFetch(url, { headers, signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.error || `Failed to fetch splits (${response.status})`,
    );
  }
  const payload = (await response.json()) as HfSplitsResponse;
  return validatedEntries(payload.splits);
}

async function fetchHubSplits({
  accessToken,
  datasetName,
  signal,
}: LoadHfDatasetSplitsArgs): Promise<HfSplitEntry[]> {
  const response = await authFetch("/api/hub/datasets/hub-options", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(accessToken),
    },
    body: JSON.stringify({ dataset_name: datasetName }),
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Failed to read dataset options (${response.status})`,
    );
  }
  const payload = (await response.json()) as { splits?: unknown };
  return validatedEntries(payload.splits);
}

const DEFAULT_FETCHERS: DatasetSplitFetchers = {
  local: fetchLocalSplits,
  remote: fetchRemoteSplits,
  hub: fetchHubSplits,
};

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
  // In the request identity, so a server arriving later triggers a refetch.
  const hfDatasetsServer = useHfDatasetsServer();
  const hubSource = useHubSource();
  const hfEndpoint = useHfEndpoint();
  const requestKey = JSON.stringify([
    datasetName,
    options?.preferLocalCache === true,
    options?.localPath ?? null,
    options?.online !== false,
    hfDatasetsServer,
    hubSource,
    hfEndpoint,
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
  }, [accessToken, datasetName, localPath, online, preferLocalCache, hfDatasetsServer, hubSource, hfEndpoint]);

  const subsets = Array.from(new Set(entries.map((e) => e.config)));

  // With >1 subset and none selected, return no splits so the UI does not auto-pick.
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
