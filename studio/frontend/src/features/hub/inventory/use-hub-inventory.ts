// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearCompletedInventoryHint,
  useDownloadManagerStore,
  type ManagedDownload,
} from "@/features/hub/download-manager";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { LocalDatasetInfo } from "./api";
import {
  dedupeSameSourceHubCacheRows,
  partialSetFromRows,
} from "./inventory-dedupe";
import {
  type PendingHintReconciliationCommit,
  pendingHintReconciliationNeedsCommit,
  pendingInventoryHintsEqual,
  pendingWithInventoryHints,
  reconcileInventoryHints,
  useInventoryHintStore,
} from "./inventory-hint-store";
import {
  type PendingInventoryHints,
  nextInventoryHintExpiryDelay,
} from "./inventory-hints";
import type { CachedInventoryRow, LocalInventoryRow } from "./types";
import {
  type DeviceInventorySource,
  inventoryEmptyRevalidationSignature,
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
  refreshInventory: () => Promise<void>;
}

export type HubInventoryKind = "models" | "datasets";

type HubInventoryOptions = {
  kind?: HubInventoryKind;
  enabled?: boolean;
  includeLocal?: boolean;
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

const LOCAL_DATASET_INVENTORY_SOURCE = [
  "localDatasets",
] as const satisfies readonly DeviceInventorySource[];

type LiveInventoryJob = Pick<
  ManagedDownload,
  "kind" | "repoId" | "variant" | "state" | "startedAt"
> & {
  displayBytes: number;
};

function compareCachedRows(
  a: CachedInventoryRow,
  b: CachedInventoryRow,
): number {
  if (Boolean(a.partial) !== Boolean(b.partial)) return a.partial ? 1 : -1;
  return a.repoId.localeCompare(b.repoId);
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

function liveInventoryRank(job: ManagedDownload): number {
  if (job.state === "running" || job.state === "cancelling") return 2;
  if (job.state === "cancelled" || job.state === "error") return 1;
  return 0;
}

function shouldSurfaceLiveJob(
  job: ManagedDownload,
  isDatasetMode: boolean,
): boolean {
  if (isDatasetMode !== (job.kind === "dataset")) return false;
  if (job.state === "running" || job.state === "cancelling") return true;
  if (job.state !== "cancelled" && job.state !== "error") return false;
  return Math.max(job.downloadedBytes, job.completedBytes) > 0;
}

function liveJobDisplayBytes(job: ManagedDownload): number {
  if (job.expectedBytes > 0) return job.expectedBytes;
  if (job.state === "running" || job.state === "cancelling") return 0;
  return Math.max(job.downloadedBytes, job.completedBytes, 0);
}

function createLiveInventoryJobsSelector(isDatasetMode: boolean): (state: {
  jobs: Record<string, ManagedDownload>;
}) => LiveInventoryJob[] {
  let cache: { signature: string; jobs: LiveInventoryJob[] } = {
    signature: "",
    jobs: [],
  };
  return (state) => {
    const selectedByRepo = new Map<string, ManagedDownload>();
    for (const job of Object.values(state.jobs)) {
      if (!shouldSurfaceLiveJob(job, isDatasetMode)) continue;
      const repoKey = job.repoId.trim().toLowerCase();
      if (!repoKey) continue;
      const current = selectedByRepo.get(repoKey);
      if (
        !current ||
        liveInventoryRank(job) > liveInventoryRank(current) ||
        (liveInventoryRank(job) === liveInventoryRank(current) &&
          job.startedAt > current.startedAt)
      ) {
        selectedByRepo.set(repoKey, job);
      }
    }
    const jobs = [...selectedByRepo.values()]
      .map((job) => ({
        kind: job.kind,
        repoId: job.repoId,
        variant: job.variant,
        state: job.state,
        startedAt: job.startedAt,
        displayBytes: liveJobDisplayBytes(job),
      }))
      .sort((a, b) => a.repoId.localeCompare(b.repoId));
    const signature = jobs
      .map(
        (job) =>
          `${job.kind}\u0001${job.repoId.toLowerCase()}\u0001${job.variant ?? ""}\u0001${job.state}\u0001${job.startedAt}\u0001${job.displayBytes}`,
      )
      .join("\u0002");
    if (signature === cache.signature) return cache.jobs;
    cache = { signature, jobs };
    return jobs;
  };
}

function liveDownloadInventoryRows(
  jobs: readonly LiveInventoryJob[],
  isDatasetMode: boolean,
): CachedInventoryRow[] {
  return jobs.map((job) => {
    const modelFormat = isDatasetMode
      ? "unknown"
      : job.variant
        ? "gguf"
        : "safetensors";
    return {
      ...buildCachedInventoryRow(
        {
          repo_id: job.repoId,
          inventory_id: `cache:${modelFormat}:${job.repoId}`,
          load_id: job.repoId,
          model_format: modelFormat,
          runtime:
            modelFormat === "gguf"
              ? "llama_cpp"
              : modelFormat === "safetensors"
                ? "transformers"
                : "unknown",
          size_bytes: job.displayBytes,
          partial: true,
          partial_transport: null,
          optimistic: true,
        },
        modelFormat,
      ),
      liveDownload: true,
    };
  });
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
  const pruneExpiredPendingHints = useInventoryHintStore(
    (state) => state.pruneExpiredHints,
  );
  const commitPendingHintReconciliations = useInventoryHintStore(
    (state) => state.commitReconciliations,
  );
  const completedHints = useDownloadManagerStore(
    (state) => state.completedInventoryHints,
  );
  const selectLiveInventoryJobs = useMemo(
    () => createLiveInventoryJobsSelector(isDatasetMode),
    [isDatasetMode],
  );
  const liveInventoryJobs = useDownloadManagerStore(selectLiveInventoryJobs);
  const emptyRevalidationSignatureRef = useRef<string | null>(null);
  const pendingForRenderRef = useRef<PendingInventoryHints | null>(null);

  const pendingForRenderNext = useMemo(
    () => pendingWithInventoryHints(pendingHints, completedHints),
    [completedHints, pendingHints],
  );
  const pendingForRender = useMemo(() => {
    const current = pendingForRenderRef.current;
    if (current && pendingInventoryHintsEqual(current, pendingForRenderNext)) {
      return current;
    }
    return pendingForRenderNext;
  }, [pendingForRenderNext]);

  useEffect(() => {
    const current = pendingForRenderRef.current;
    if (!current || !pendingInventoryHintsEqual(current, pendingForRender)) {
      pendingForRenderRef.current = pendingForRender;
    }
  }, [pendingForRender]);

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
    if (
      completedHints.length === 0 &&
      !reconciliations.some((commit) =>
        pendingHintReconciliationNeedsCommit(
          pendingForRender,
          observedInventoryKeys,
          commit,
        ),
      )
    ) {
      return;
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
    observedInventoryKeys,
    isDatasetMode,
    cachedGgufReconciliation,
    cachedModelsReconciliation,
    cachedDatasetsReconciliation,
  ]);

  const cachedGguf = cachedGgufReconciliation.rows;
  const cachedModels = cachedModelsReconciliation.rows;
  const cachedDatasets = cachedDatasetsReconciliation.rows;
  const liveDownloadRows = useMemo(
    () => liveDownloadInventoryRows(liveInventoryJobs, isDatasetMode),
    [isDatasetMode, liveInventoryJobs],
  );

  const cachedModelRows = useMemo<CachedInventoryRow[]>(
    () =>
      [
        ...(isDatasetMode ? [] : liveDownloadRows),
        ...cachedGguf.map((row) => buildCachedInventoryRow(row, "gguf")),
        ...cachedModels.map((row) =>
          buildCachedInventoryRow(row, "safetensors"),
        ),
      ].sort(compareCachedRows),
    [cachedGguf, cachedModels, isDatasetMode, liveDownloadRows],
  );

  const cachedDatasetRows = useMemo<CachedInventoryRow[]>(
    () =>
      [
        ...(isDatasetMode ? liveDownloadRows : []),
        ...cachedDatasets.map((row) => buildCachedInventoryRow(row, "unknown")),
      ]
        .sort(compareCachedRows),
    [cachedDatasets, isDatasetMode, liveDownloadRows],
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
  const refreshInventory = useCallback(async () => {
    await refreshDeviceInventory();
  }, [refreshDeviceInventory]);
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
