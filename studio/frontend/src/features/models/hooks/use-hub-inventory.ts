// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type CachedGgufRepo,
  type CachedModelRepo,
  listCachedGguf,
  listCachedModels,
  listLocalModels,
} from "@/features/chat";
import {
  type CachedDatasetRepo,
  listCachedDatasets,
  listLocalDatasets,
  type LocalDatasetInfo,
} from "@/features/training";
import { getHfToken } from "@/stores/hf-token-store";
import type { InventoryHint } from "../components/download-section";
import {
  bumpInventoryVersion,
  getInventoryVersion,
  useInventoryVersion,
} from "@/stores/inventory-events";
import { buildLocalInventoryRows, normalizeTimestamp } from "../lib/view-models";
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
    quant_method?: string | null;
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
    quantMethod: row.quant_method ?? null,
  };
}

function compareCachedRows(a: CachedInventoryRow, b: CachedInventoryRow): number {
  if (Boolean(a.partial) !== Boolean(b.partial)) return a.partial ? 1 : -1;
  return a.repoId.localeCompare(b.repoId);
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
  // Version this hook itself last published. Lets the subscription below
  // ignore its own bumps so an external mutation triggers exactly one
  // refetch and a self-bump triggers none.
  const selfVersionRef = useRef(getInventoryVersion());
  const loadInventory = useCallback((onSettled?: () => void) => {
    const epoch = ++loadEpochRef.current;
    const fresh = () => mountedRef.current && loadEpochRef.current === epoch;
    const hfToken = getHfToken() || undefined;
    const onFetchError = (source: string) => (err: unknown) => {
      console.warn(`Hub inventory: ${source} fetch failed`, err);
      return null;
    };
    const fetchers: Array<Promise<string | null>> = [
      listCachedGguf(hfToken)
        .then((rows) => {
          if (fresh()) setCachedGguf(rows);
          return sourceSignature(rows);
        })
        .catch(onFetchError("cached-gguf")),
      listCachedModels(hfToken)
        .then((rows) => {
          if (fresh()) setCachedModels(rows);
          return sourceSignature(rows);
        })
        .catch(onFetchError("cached-models")),
      listCachedDatasets()
        .then((rows) => {
          if (fresh()) setCachedDatasets(rows);
          return sourceSignature(rows);
        })
        .catch(onFetchError("cached-datasets")),
      listLocalModels()
        .then((response) => {
          if (fresh()) {
            setLocalRows(buildLocalInventoryRows(response.models));
          }
          return sourceSignature(response.models);
        })
        .catch(onFetchError("local-models")),
      listLocalDatasets()
        .then((response) => {
          if (fresh()) setLocalDatasets(response.datasets);
          return sourceSignature(response.datasets);
        })
        .catch(onFetchError("local-datasets")),
    ];
    void Promise.all(fetchers).then((parts) => {
      if (!fresh()) return;
      // parts order matches `fetchers`: gguf, models, datasets, localModels,
      // localDatasets. `null` means that fetch threw.
      setLoadFailed({
        models: parts[0] === null || parts[1] === null || parts[3] === null,
        datasets: parts[2] === null || parts[4] === null,
      });
      // Build the signature with a per-slot sentinel for failed fetches rather
      // than discarding it. A single flaky endpoint then bumps the global
      // version once (when its slot flips to/from failed) instead of on every
      // refresh, which would otherwise fan a down endpoint into an app-wide
      // refetch storm across all useInventoryVersion consumers.
      const signature = parts
        .map((part, i) => (part === null ? ` fail:${i}` : part))
        .join("|");
      if (signature !== lastSignatureRef.current) {
        lastSignatureRef.current = signature;
        bumpInventoryVersion();
      }
      // Sync to the current global version on every load, not only when we
      // bumped. Otherwise a load triggered by an external bump that returned
      // unchanged data would leave selfVersionRef stale, and the next
      // unrelated bump would be misread as our own.
      selfVersionRef.current = getInventoryVersion();
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
          setCachedGguf((prev) => merge(prev, baseEntry));
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

  // Refetch when any other surface (a download completing in the global
  // panel, a fine-tune delete, a dataset upload) mutates on-disk inventory.
  // Bumps this hook published itself are filtered via selfVersionRef.
  const inventoryVersion = useInventoryVersion();
  useEffect(() => {
    if (inventoryVersion === selfVersionRef.current) return;
    selfVersionRef.current = inventoryVersion;
    loadInventory();
  }, [inventoryVersion, loadInventory]);

  const cachedModelRows = useMemo<CachedInventoryRow[]>(
    () =>
      [
        ...cachedGguf.map((row) => toCachedRow(row, true)),
        ...cachedModels.map((row) => toCachedRow(row, false)),
      ].sort(compareCachedRows),
    [cachedGguf, cachedModels],
  );

  const cachedDatasetRows = useMemo<CachedInventoryRow[]>(
    () =>
      cachedDatasets
        .map((row) => toCachedRow(row, false))
        .sort(compareCachedRows),
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
          updatedAt: normalizeTimestamp(ds.updated_at),
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
