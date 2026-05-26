// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearCompletedInventoryHint,
  useDownloadManagerStore,
} from "@/features/download-jobs";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { LocalDatasetInfo } from "./api";
import { dedupeSameSourceHubCacheRows } from "./inventory-dedupe";
import {
  type PendingHintReconciliationCommit,
  pendingWithInventoryHints,
  reconcileInventoryHints,
  useInventoryHintStore,
} from "./inventory-hint-store";
import {
  INVENTORY_HINT_TTL_MS,
  type PendingInventoryHints,
} from "./inventory-hints";
import type {
  CachedInventoryRow,
  InventoryHint,
  LocalInventoryRow,
} from "./types";
import {
  type DeviceInventorySource,
  useDeviceInventorySources,
} from "./use-device-inventory";
import {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
  defaultCapabilities,
  normalizeTimestamp,
} from "./view-models";

export interface HubInventory {
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  availableSet: Set<string>;
  partialSet: Set<string>;
  downloadedReady: boolean;
  inventoryError: boolean;
  inventoryWarning: boolean;
  refreshInventory: (hint?: InventoryHint) => Promise<void>;
}

export interface HubPartialSets {
  modelPartialSet: Set<string>;
  datasetPartialSet: Set<string>;
}

export type HubInventoryKind = "models" | "datasets";

type HubInventoryOptions = {
  kind?: HubInventoryKind;
  enabled?: boolean;
  includeLocal?: boolean;
};

type HubPartialSetOptions = {
  modelEnabled?: boolean;
  datasetEnabled?: boolean;
};

const MODEL_HUB_CACHE_SOURCES = [
  "cachedGguf",
  "cachedModels",
] as const satisfies readonly DeviceInventorySource[];

const DATASET_HUB_CACHE_SOURCES = [
  "cachedDatasets",
] as const satisfies readonly DeviceInventorySource[];

const LOCAL_MODEL_INVENTORY_SOURCE = [
  "localModels",
] as const satisfies readonly DeviceInventorySource[];

function nextInventoryHintExpiryDelay(
  pending: PendingInventoryHints,
): number | null {
  let nextExpiry = Number.POSITIVE_INFINITY;
  for (const hints of Object.values(pending)) {
    for (const hint of hints.values()) {
      if (typeof hint.createdAt !== "number") {
        continue;
      }
      nextExpiry = Math.min(nextExpiry, hint.createdAt + INVENTORY_HINT_TTL_MS);
    }
  }
  return Number.isFinite(nextExpiry)
    ? Math.max(0, nextExpiry - Date.now())
    : null;
}

const LOCAL_DATASET_INVENTORY_SOURCE = [
  "localDatasets",
] as const satisfies readonly DeviceInventorySource[];

const EMPTY_PARTIAL_SET = new Set<string>();

function compareCachedRows(
  a: CachedInventoryRow,
  b: CachedInventoryRow,
): number {
  if (Boolean(a.partial) !== Boolean(b.partial)) return a.partial ? 1 : -1;
  return a.repoId.localeCompare(b.repoId);
}

function partialSetFromRows<T extends { partial?: boolean }>(
  rows: readonly T[],
  getRepoId: (row: T) => string | null | undefined,
): Set<string> {
  const complete = new Set<string>();
  for (const row of rows) {
    const repoId = getRepoId(row);
    if (repoId && !row.partial) complete.add(repoId.toLowerCase());
  }
  const partial = new Set<string>();
  for (const row of rows) {
    const repoId = getRepoId(row);
    if (!repoId) continue;
    const key = repoId.toLowerCase();
    if (row.partial && !complete.has(key)) partial.add(key);
  }
  return partial;
}

function localDatasetSourceLabel(source?: string | null): string {
  switch (source) {
    case "recipe":
      return "Recipe dataset";
    case "upload":
      return "Uploaded dataset";
    default:
      return "Local dataset";
  }
}

function isReadyForDisplay(source: {
  error: string | null;
  loading: boolean;
  ready: boolean;
  rows: readonly unknown[];
}): boolean {
  if (source.error !== null) return true;
  if (!source.ready) return false;
  return !(source.loading && source.rows.length === 0);
}

function inventoryEmptyRevalidationSignature(
  sources: readonly {
    error: string | null;
    key: string | null;
    ready: boolean;
  }[],
): string {
  return sources
    .map((source) =>
      [
        source.key ?? "pending",
        source.ready ? "ready" : "pending",
        source.error ?? "",
      ].join("\u0001"),
    )
    .join("\u0002");
}

export function useHubPartialSets(
  options: HubPartialSetOptions = {},
): HubPartialSets {
  const hfToken = useHfTokenStore((state) => state.token) || undefined;
  const modelEnabled = options.modelEnabled ?? true;
  const datasetEnabled = options.datasetEnabled ?? true;
  const modelInventory = useDeviceInventorySources(MODEL_HUB_CACHE_SOURCES, {
    hfToken,
    enabled: modelEnabled,
  });
  const datasetInventory = useDeviceInventorySources(
    DATASET_HUB_CACHE_SOURCES,
    {
      hfToken,
      enabled: datasetEnabled,
    },
  );

  const modelPartialSet = useMemo(() => {
    if (!modelEnabled) return EMPTY_PARTIAL_SET;
    return partialSetFromRows(
      [...modelInventory.cachedGguf.rows, ...modelInventory.cachedModels.rows],
      (row) => row.repo_id,
    );
  }, [
    modelEnabled,
    modelInventory.cachedGguf.rows,
    modelInventory.cachedModels.rows,
  ]);

  const datasetPartialSet = useMemo(() => {
    if (!datasetEnabled) return EMPTY_PARTIAL_SET;
    return partialSetFromRows(
      datasetInventory.cachedDatasets.rows,
      (row) => row.repo_id,
    );
  }, [datasetEnabled, datasetInventory.cachedDatasets.rows]);

  return { modelPartialSet, datasetPartialSet };
}

export function useHubInventory(
  options: HubInventoryOptions = {},
): HubInventory {
  const hfToken = useHfTokenStore((state) => state.token) || undefined;
  const isDatasetMode = (options.kind ?? "models") === "datasets";
  const enabled = options.enabled ?? true;
  const includeLocal = options.includeLocal ?? true;
  const modelInventory = useDeviceInventorySources(MODEL_HUB_CACHE_SOURCES, {
    hfToken,
    enabled: enabled && !isDatasetMode,
  });
  const datasetInventory = useDeviceInventorySources(
    DATASET_HUB_CACHE_SOURCES,
    {
      hfToken,
      enabled: enabled && isDatasetMode,
    },
  );
  const localModelInventory = useDeviceInventorySources(
    LOCAL_MODEL_INVENTORY_SOURCE,
    {
      hfToken,
      enabled: enabled && !isDatasetMode && includeLocal,
    },
  );
  const localDatasetInventory = useDeviceInventorySources(
    LOCAL_DATASET_INVENTORY_SOURCE,
    {
      hfToken,
      enabled: enabled && isDatasetMode && includeLocal,
    },
  );
  const cachedGgufSource = modelInventory.cachedGguf;
  const cachedModelsSource = modelInventory.cachedModels;
  const cachedDatasetsSource = datasetInventory.cachedDatasets;
  const localModelsSource = localModelInventory.localModels;
  const localDatasetsSource = localDatasetInventory.localDatasets;
  const refreshModelInventory = modelInventory.refresh;
  const refreshDatasetInventory = datasetInventory.refresh;
  const refreshLocalModelInventory = localModelInventory.refresh;
  const refreshLocalDatasetInventory = localDatasetInventory.refresh;
  const pendingHints = useInventoryHintStore((state) => state.pending);
  const observedInventoryKeys = useInventoryHintStore(
    (state) => state.observedKeys,
  );
  const rememberPendingInventoryHint = useInventoryHintStore(
    (state) => state.rememberHint,
  );
  const pruneExpiredPendingHints = useInventoryHintStore(
    (state) => state.pruneExpiredHints,
  );
  const commitPendingHintReconciliations = useInventoryHintStore(
    (state) => state.commitReconciliations,
  );
  const completedHints = useDownloadManagerStore(
    (state) => state.completedInventoryHints,
  );
  const emptyRevalidationSignatureRef = useRef<string | null>(null);

  const pendingForRender = useMemo(() => {
    return pendingWithInventoryHints(pendingHints, completedHints);
  }, [completedHints, pendingHints]);

  useEffect(() => {
    const delay = nextInventoryHintExpiryDelay(pendingHints);
    if (delay === null) {
      return;
    }
    if (typeof window === "undefined") {
      pruneExpiredPendingHints();
      return;
    }
    const timer = window.setTimeout(pruneExpiredPendingHints, delay + 1);
    return () => window.clearTimeout(timer);
  }, [pendingHints, pruneExpiredPendingHints]);

  const cachedGgufReconciliation = useMemo(
    () =>
      reconcileInventoryHints({
        pending: pendingForRender,
        kind: "gguf",
        rows: cachedGgufSource.rows,
        previouslyObserved: observedInventoryKeys.gguf,
      }),
    [cachedGgufSource.rows, observedInventoryKeys.gguf, pendingForRender],
  );
  const cachedModelsReconciliation = useMemo(
    () =>
      reconcileInventoryHints({
        pending: pendingForRender,
        kind: "model",
        rows: cachedModelsSource.rows,
        previouslyObserved: observedInventoryKeys.model,
      }),
    [cachedModelsSource.rows, observedInventoryKeys.model, pendingForRender],
  );
  const cachedDatasetsReconciliation = useMemo(
    () =>
      reconcileInventoryHints({
        pending: pendingForRender,
        kind: "dataset",
        rows: cachedDatasetsSource.rows,
        previouslyObserved: observedInventoryKeys.dataset,
      }),
    [
      cachedDatasetsSource.rows,
      observedInventoryKeys.dataset,
      pendingForRender,
    ],
  );

  useEffect(() => {
    const reconciliations: PendingHintReconciliationCommit[] = [];
    if (!isDatasetMode) {
      reconciliations.push(
        {
          kind: "gguf",
          reconciliation: cachedGgufReconciliation,
        },
        {
          kind: "model",
          reconciliation: cachedModelsReconciliation,
        },
      );
    }
    if (isDatasetMode) {
      reconciliations.push({
        kind: "dataset",
        reconciliation: cachedDatasetsReconciliation,
      });
    }
    const completedHintsToSuppress = commitPendingHintReconciliations(
      completedHints,
      pendingForRender,
      reconciliations,
    );
    for (const hint of completedHintsToSuppress) {
      clearCompletedInventoryHint(hint);
    }
  }, [
    completedHints,
    commitPendingHintReconciliations,
    pendingForRender,
    isDatasetMode,
    cachedGgufReconciliation,
    cachedModelsReconciliation,
    cachedDatasetsReconciliation,
  ]);

  const cachedGguf = cachedGgufReconciliation.rows;
  const cachedModels = cachedModelsReconciliation.rows;
  const cachedDatasets = cachedDatasetsReconciliation.rows;

  const cachedModelRows = useMemo<CachedInventoryRow[]>(
    () =>
      [
        ...cachedGguf.map((row) => buildCachedInventoryRow(row, "gguf")),
        ...cachedModels.map((row) =>
          buildCachedInventoryRow(row, "safetensors"),
        ),
      ].sort(compareCachedRows),
    [cachedGguf, cachedModels],
  );

  const cachedDatasetRows = useMemo<CachedInventoryRow[]>(
    () =>
      cachedDatasets
        .map((row) => buildCachedInventoryRow(row, "unknown"))
        .sort(compareCachedRows),
    [cachedDatasets],
  );

  const localRows = useMemo(
    () => buildLocalInventoryRows(localModelsSource.rows),
    [localModelsSource.rows],
  );

  const localDatasetRows = useMemo<LocalInventoryRow[]>(() => {
    return localDatasetsSource.rows
      .map((ds: LocalDatasetInfo) => {
        const repoId = ds.id.includes("/") ? ds.id : null;
        const owner = repoId ? ds.id.split("/")[0] : "Local";
        return {
          kind: "local" as const,
          id: ds.id,
          repoId,
          owner,
          title: ds.label || ds.id,
          source: "custom" as const,
          sourceLabel: localDatasetSourceLabel(ds.source),
          path: ds.path,
          isGguf: false,
          loadId: ds.id,
          modelFormat: "unknown" as const,
          runtime: "unknown" as const,
          formatVariant: null,
          capabilities: defaultCapabilities("unknown"),
          updatedAt: normalizeTimestamp(ds.updated_at),
        };
      })
      .sort((a, b) => a.title.localeCompare(b.title));
  }, [localDatasetsSource.rows]);

  const dedupedModelInventory = useMemo(
    () =>
      dedupeSameSourceHubCacheRows({
        cachedRows: cachedModelRows,
        localRows,
      }),
    [cachedModelRows, localRows],
  );

  const dedupedDatasetInventory = useMemo(
    () =>
      dedupeSameSourceHubCacheRows({
        cachedRows: cachedDatasetRows,
        localRows: localDatasetRows,
      }),
    [cachedDatasetRows, localDatasetRows],
  );

  const cachedRows = isDatasetMode
    ? dedupedDatasetInventory.cachedRows
    : dedupedModelInventory.cachedRows;
  const effectiveLocalRows = isDatasetMode
    ? dedupedDatasetInventory.localRows
    : dedupedModelInventory.localRows;

  const availableSet = useMemo(() => {
    const set = new Set<string>();
    for (const row of cachedRows) set.add(row.repoId.toLowerCase());
    for (const row of effectiveLocalRows) {
      if (row.repoId) set.add(row.repoId.toLowerCase());
    }
    return set;
  }, [cachedRows, effectiveLocalRows]);

  const partialSet = useMemo(() => {
    return partialSetFromRows(
      [...cachedRows, ...effectiveLocalRows],
      (row) => row.repoId,
    );
  }, [cachedRows, effectiveLocalRows]);

  const modelReady =
    isReadyForDisplay(cachedGgufSource) &&
    isReadyForDisplay(cachedModelsSource) &&
    (!includeLocal || isReadyForDisplay(localModelsSource));
  const datasetReady =
    isReadyForDisplay(cachedDatasetsSource) &&
    (!includeLocal || isReadyForDisplay(localDatasetsSource));
  const inventoryFailed = isDatasetMode
    ? Boolean(
        cachedDatasetsSource.error ||
          (includeLocal && localDatasetsSource.error),
      )
    : Boolean(
        cachedGgufSource.error ||
          cachedModelsSource.error ||
          (includeLocal && localModelsSource.error),
      );
  const inventoryWarning =
    inventoryFailed && (cachedRows.length > 0 || effectiveLocalRows.length > 0);

  const refreshDeviceInventory = useCallback(async () => {
    if (isDatasetMode) {
      await Promise.all([
        refreshDatasetInventory(),
        ...(includeLocal ? [refreshLocalDatasetInventory()] : []),
      ]);
      return;
    }
    await Promise.all([
      refreshModelInventory(),
      ...(includeLocal ? [refreshLocalModelInventory()] : []),
    ]);
  }, [
    includeLocal,
    isDatasetMode,
    refreshDatasetInventory,
    refreshLocalDatasetInventory,
    refreshLocalModelInventory,
    refreshModelInventory,
  ]);
  const refreshInventory = useCallback(
    async (hint?: InventoryHint) => {
      if (hint) rememberPendingInventoryHint(hint);
      await refreshDeviceInventory();
    },
    [refreshDeviceInventory, rememberPendingInventoryHint],
  );
  const downloadedReady = isDatasetMode ? datasetReady : modelReady;
  const hasInventoryRows =
    cachedRows.length > 0 || effectiveLocalRows.length > 0;
  const hasActiveEmptyRefresh = isDatasetMode
    ? cachedDatasetsSource.loading ||
      (includeLocal && localDatasetsSource.loading)
    : cachedGgufSource.loading ||
      cachedModelsSource.loading ||
      (includeLocal && localModelsSource.loading);
  const emptyRevalidationSignature = useMemo(
    () =>
      isDatasetMode
        ? inventoryEmptyRevalidationSignature([
            cachedDatasetsSource,
            ...(includeLocal ? [localDatasetsSource] : []),
          ])
        : inventoryEmptyRevalidationSignature([
            cachedGgufSource,
            cachedModelsSource,
            ...(includeLocal ? [localModelsSource] : []),
          ]),
    [
      cachedDatasetsSource,
      cachedGgufSource,
      cachedModelsSource,
      includeLocal,
      isDatasetMode,
      localDatasetsSource,
      localModelsSource,
    ],
  );

  useEffect(() => {
    if (
      !enabled ||
      !downloadedReady ||
      inventoryFailed ||
      hasInventoryRows ||
      hasActiveEmptyRefresh
    ) {
      return;
    }
    if (emptyRevalidationSignatureRef.current === emptyRevalidationSignature) {
      return;
    }
    const timer = window.setTimeout(() => {
      emptyRevalidationSignatureRef.current = emptyRevalidationSignature;
      void refreshDeviceInventory();
    }, 500);
    return () => window.clearTimeout(timer);
  }, [
    downloadedReady,
    emptyRevalidationSignature,
    enabled,
    hasActiveEmptyRefresh,
    hasInventoryRows,
    inventoryFailed,
    refreshDeviceInventory,
  ]);

  return {
    cachedRows,
    localRows: effectiveLocalRows,
    availableSet,
    partialSet,
    downloadedReady,
    inventoryError: inventoryFailed,
    inventoryWarning,
    refreshInventory,
  };
}
