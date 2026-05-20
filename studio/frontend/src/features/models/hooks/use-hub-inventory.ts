// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type CachedGgufRepo,
  type CachedModelRepo,
  listCachedGguf,
  listCachedModels,
  listLocalModels,
} from "@/features/chat/api/chat-api";
import {
  type CachedDatasetRepo,
  listCachedDatasets,
  listLocalDatasets,
} from "@/features/training/api/datasets-api";
import type { LocalDatasetInfo } from "@/features/training/types/datasets";
import { getHfToken } from "@/stores/hf-token-store";
import type { InventoryHint } from "../components/download-section";
import { bumpInventoryVersion } from "../inventory-events";
import { buildLocalInventoryRows } from "../lib/view-models";
import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export interface HubInventory {
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  availableSet: Set<string>;
  partialSet: Set<string>;
  downloadedReady: boolean;
  inventoryError: boolean;
  refreshInventory: (hint?: InventoryHint) => void;
}

type InventoryRowLike = {
  repo_id?: string;
  id?: string;
  size_bytes?: number;
  partial?: boolean;
  updated_at?: number | null;
};

function sourceSignature(rows: ReadonlyArray<InventoryRowLike>): string {
  return rows
    .map(
      (r) =>
        `${r.repo_id ?? r.id ?? ""}:${r.size_bytes ?? ""}:${
          r.partial ? 1 : 0
        }:${r.updated_at ?? ""}`,
    )
    .sort()
    .join(",");
}

function toCachedRow(
  row: {
    repo_id: string;
    size_bytes: number;
    cache_path?: string;
    partial?: boolean;
    pipeline_tag?: string | null;
    tags?: string[];
    library_name?: string | null;
  },
  isGguf: boolean,
): CachedInventoryRow {
  return {
    kind: "cache",
    id: row.repo_id,
    repoId: row.repo_id,
    owner: row.repo_id.includes("/") ? row.repo_id.split("/")[0] : "Hub",
    repo: row.repo_id.includes("/")
      ? row.repo_id.split("/").slice(1).join("/")
      : row.repo_id,
    isGguf,
    bytes: row.size_bytes,
    cachePath: row.cache_path ?? null,
    partial: row.partial ?? false,
    pipelineTag: row.pipeline_tag ?? null,
    tags: row.tags,
    libraryName: row.library_name ?? null,
  };
}

export function useHubInventory(isDatasetMode: boolean): HubInventory {
  const [cachedGguf, setCachedGguf] = useState<CachedGgufRepo[]>([]);
  const [cachedModels, setCachedModels] = useState<CachedModelRepo[]>([]);
  const [cachedDatasets, setCachedDatasets] = useState<CachedDatasetRepo[]>([]);
  const [localRows, setLocalRows] = useState<LocalInventoryRow[]>([]);
  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [downloadedReady, setDownloadedReady] = useState(false);
  const [loadFailed, setLoadFailed] = useState({
    models: false,
    datasets: false,
  });

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const lastSignatureRef = useRef<string | null>(null);
  const loadEpochRef = useRef(0);
  const loadInventory = useCallback((onSettled?: () => void) => {
    const epoch = ++loadEpochRef.current;
    const fresh = () => mountedRef.current && loadEpochRef.current === epoch;
    const hfToken = getHfToken() || undefined;
    const fetchers: Array<Promise<string | null>> = [
      listCachedGguf(hfToken)
        .then((rows) => {
          if (fresh()) setCachedGguf(rows);
          return sourceSignature(rows);
        })
        .catch(() => null),
      listCachedModels(hfToken)
        .then((rows) => {
          if (fresh()) setCachedModels(rows);
          return sourceSignature(rows);
        })
        .catch(() => null),
      listCachedDatasets()
        .then((rows) => {
          if (fresh()) setCachedDatasets(rows);
          return sourceSignature(rows);
        })
        .catch(() => null),
      listLocalModels()
        .then((response) => {
          if (fresh()) {
            setLocalRows(buildLocalInventoryRows(response.models));
          }
          return sourceSignature(response.models);
        })
        .catch(() => null),
      listLocalDatasets()
        .then((response) => {
          if (fresh()) setLocalDatasets(response.datasets);
          return sourceSignature(response.datasets);
        })
        .catch(() => null),
    ];
    void Promise.all(fetchers).then((parts) => {
      if (!fresh()) return;
      // parts order matches `fetchers`: gguf, models, datasets, localModels,
      // localDatasets. `null` means that fetch threw.
      setLoadFailed({
        models: parts[0] === null || parts[1] === null || parts[3] === null,
        datasets: parts[2] === null || parts[4] === null,
      });
      const signature = parts.includes(null) ? null : parts.join("|");
      if (signature === null || signature !== lastSignatureRef.current) {
        lastSignatureRef.current = signature;
        bumpInventoryVersion();
      }
      onSettled?.();
    });
  }, []);

  const refreshInventory = useCallback(
    (hint?: InventoryHint) => {
      if (hint && mountedRef.current) {
        const baseEntry = {
          repo_id: hint.repoId,
          size_bytes: hint.bytes ?? 0,
          partial: false,
        };
        const merge = <T extends { repo_id: string; size_bytes: number }>(
          prev: T[],
          seed: T,
        ): T[] => {
          const lower = seed.repo_id.toLowerCase();
          const idx = prev.findIndex((r) => r.repo_id.toLowerCase() === lower);
          if (idx === -1) return [...prev, seed];
          const merged = {
            ...prev[idx],
            ...seed,
            size_bytes: Math.max(prev[idx].size_bytes, seed.size_bytes),
          };
          return [...prev.slice(0, idx), merged, ...prev.slice(idx + 1)];
        };
        if (hint.kind === "gguf") {
          setCachedGguf((prev) =>
            merge(prev, { ...baseEntry, cache_path: "" }),
          );
        } else if (hint.kind === "model") {
          setCachedModels((prev) => merge(prev, baseEntry));
        } else if (hint.kind === "dataset") {
          setCachedDatasets((prev) => merge(prev, baseEntry));
        }
      }
      loadInventory();
    },
    [loadInventory],
  );

  useEffect(() => {
    loadInventory(() => setDownloadedReady(true));
  }, [loadInventory]);

  const cachedModelRows = useMemo<CachedInventoryRow[]>(
    () =>
      [
        ...cachedGguf.map((row) => toCachedRow(row, true)),
        ...cachedModels.map((row) => toCachedRow(row, false)),
      ].sort((a, b) => a.repoId.localeCompare(b.repoId)),
    [cachedGguf, cachedModels],
  );

  const cachedDatasetRows = useMemo<CachedInventoryRow[]>(
    () =>
      cachedDatasets
        .map((row) => toCachedRow(row, false))
        .sort((a, b) => a.repoId.localeCompare(b.repoId)),
    [cachedDatasets],
  );

  const localDatasetRows = useMemo<LocalInventoryRow[]>(() => {
    return localDatasets
      .map((ds) => {
        const repoId = ds.id.includes("/") ? ds.id : null;
        const owner = repoId ? ds.id.split("/")[0] : "Local";
        return {
          kind: "local" as const,
          id: ds.id,
          repoId,
          owner,
          title: ds.label || ds.id,
          source: "custom" as const,
          sourceLabel: "Local dataset",
          path: ds.path,
          isGguf: false,
          updatedAt: ds.updated_at ?? null,
        };
      })
      .sort((a, b) => a.title.localeCompare(b.title));
  }, [localDatasets]);

  const dedupedLocalRows = useMemo<LocalInventoryRow[]>(() => {
    if (localRows.length === 0) return localRows;
    const cachedSet = new Set<string>();
    for (const row of cachedModelRows) {
      cachedSet.add(row.repoId.toLowerCase());
    }
    if (cachedSet.size === 0) return localRows;
    return localRows.filter((row) => {
      if (row.source !== "hf_cache") return true;
      if (!row.repoId) return true;
      return !cachedSet.has(row.repoId.toLowerCase());
    });
  }, [localRows, cachedModelRows]);

  const cachedRows = isDatasetMode ? cachedDatasetRows : cachedModelRows;
  const effectiveLocalRows = isDatasetMode ? localDatasetRows : dedupedLocalRows;

  const availableSet = useMemo(() => {
    const set = new Set<string>();
    for (const row of cachedRows) set.add(row.repoId.toLowerCase());
    for (const row of effectiveLocalRows) {
      if (row.repoId) set.add(row.repoId.toLowerCase());
    }
    return set;
  }, [cachedRows, effectiveLocalRows]);

  const partialSet = useMemo(() => {
    const set = new Set<string>();
    for (const row of cachedRows) {
      if (row.partial) set.add(row.repoId.toLowerCase());
    }
    for (const row of effectiveLocalRows) {
      if (row.partial && row.repoId) set.add(row.repoId.toLowerCase());
    }
    return set;
  }, [cachedRows, effectiveLocalRows]);

  const inventoryError = isDatasetMode ? loadFailed.datasets : loadFailed.models;

  return {
    cachedRows,
    localRows: effectiveLocalRows,
    availableSet,
    partialSet,
    downloadedReady,
    inventoryError,
    refreshInventory,
  };
}
