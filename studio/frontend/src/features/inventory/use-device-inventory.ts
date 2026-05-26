// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CachedDatasetRepo,
  type CachedGgufRepo,
  type CachedModelRepo,
  type LocalDatasetInfo,
  type LocalModelInfo,
  listCachedDatasets,
  listCachedGguf,
  listCachedModels,
  listLocalDatasets,
  listLocalModels,
} from "./api";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { fingerprintToken } from "@/lib/token-fingerprint";
import { useInventoryVersion } from "@/stores/inventory-events";
import { useCallback, useEffect, useMemo } from "react";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

export type DeviceInventorySource =
  | "cachedGguf"
  | "cachedModels"
  | "cachedDatasets"
  | "localModels"
  | "localDatasets";

export type DeviceInventoryRows = {
  cachedGguf: CachedGgufRepo[];
  cachedModels: CachedModelRepo[];
  cachedDatasets: CachedDatasetRepo[];
  localModels: LocalModelInfo[];
  localDatasets: LocalDatasetInfo[];
};

export type DeviceInventorySourceState<Rows extends readonly unknown[]> = {
  rows: Rows;
  loading: boolean;
  ready: boolean;
  error: string | null;
  key: string | null;
};

type DeviceInventoryState = {
  [K in DeviceInventorySource]: DeviceInventorySourceState<
    DeviceInventoryRows[K]
  >;
};

type FetchInventorySourceOptions = {
  hfToken?: string | null;
  tokenFingerprint?: string;
  inventoryVersion: number;
  force?: boolean;
};

type MarkInventoryFreshOptions = {
  hfToken?: string | null;
  tokenFingerprint?: string;
  inventoryVersion: number;
};

type UseDeviceInventoryOptions = {
  hfToken?: string | null;
  enabled?: boolean;
};

type UseDeviceInventoryResult<
  Sources extends readonly DeviceInventorySource[],
> = Pick<DeviceInventoryState, Sources[number]> & {
  refresh: () => Promise<void>;
};

const TOKEN_SCOPED_SOURCES = new Set<DeviceInventorySource>([
  "cachedGguf",
  "cachedModels",
]);
const TOKEN_SCOPED_INVENTORY_DEBOUNCE_MS = 500;

const inFlight = new Map<string, Promise<unknown>>();

function emptySource<
  Rows extends readonly unknown[],
>(): DeviceInventorySourceState<Rows> {
  return {
    rows: [] as unknown as Rows,
    loading: false,
    ready: false,
    error: null,
    key: null,
  };
}

export const useDeviceInventoryStore = create<DeviceInventoryState>(() => ({
  cachedGguf: emptySource<CachedGgufRepo[]>(),
  cachedModels: emptySource<CachedModelRepo[]>(),
  cachedDatasets: emptySource<CachedDatasetRepo[]>(),
  localModels: emptySource<LocalModelInfo[]>(),
  localDatasets: emptySource<LocalDatasetInfo[]>(),
}));

function sourceRequestKey(
  source: DeviceInventorySource,
  inventoryVersion: number,
  tokenFingerprint: string,
): string {
  const tokenPart = TOKEN_SCOPED_SOURCES.has(source)
    ? tokenFingerprint
    : "local";
  return `${source}:${inventoryVersion}:${tokenPart}`;
}

function fetchErrorMessage(
  source: DeviceInventorySource,
  error: unknown,
): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  switch (source) {
    case "cachedGguf":
      return "Couldn't scan cached GGUF models";
    case "cachedModels":
      return "Couldn't scan cached Hugging Face models";
    case "cachedDatasets":
      return "Couldn't scan cached datasets";
    case "localModels":
      return "Couldn't scan local models";
    case "localDatasets":
      return "Couldn't scan local datasets";
    default:
      return "Couldn't scan inventory";
  }
}

function logInventorySourceFailure(
  source: DeviceInventorySource,
  error: unknown,
): void {
  if (!import.meta.env?.DEV) {
    return;
  }
  console.warn(`Inventory source "${source}" failed to refresh:`, error);
}

async function runSourceFetch<K extends DeviceInventorySource>(
  source: K,
  hfToken?: string | null,
): Promise<DeviceInventoryRows[K]> {
  switch (source) {
    case "cachedGguf":
      return (await listCachedGguf(hfToken)) as DeviceInventoryRows[K];
    case "cachedModels":
      return (await listCachedModels(hfToken)) as DeviceInventoryRows[K];
    case "cachedDatasets":
      return (await listCachedDatasets()) as DeviceInventoryRows[K];
    case "localModels":
      return (await listLocalModels()).models as DeviceInventoryRows[K];
    case "localDatasets":
      return (await listLocalDatasets()).datasets as DeviceInventoryRows[K];
    default:
      throw new Error("Unsupported inventory source");
  }
}

function updateSourceState<K extends DeviceInventorySource>(
  source: K,
  patch: Partial<DeviceInventorySourceState<DeviceInventoryRows[K]>>,
): void {
  useDeviceInventoryStore.setState((state) => ({
    [source]: {
      ...state[source],
      ...patch,
    },
  }));
}

export function fetchInventorySource<K extends DeviceInventorySource>(
  source: K,
  options: FetchInventorySourceOptions,
): Promise<DeviceInventoryRows[K]> {
  const tokenFingerprint =
    options.tokenFingerprint ?? fingerprintToken(options.hfToken);
  const key = sourceRequestKey(
    source,
    options.inventoryVersion,
    tokenFingerprint,
  );
  const current = useDeviceInventoryStore.getState()[
    source
  ] as DeviceInventorySourceState<DeviceInventoryRows[K]>;
  const pending = inFlight.get(key) as
    | Promise<DeviceInventoryRows[K]>
    | undefined;
  if (pending) {
    if (options.force) {
      return pending
        .catch(() => undefined)
        .then(() => fetchInventorySource(source, options));
    }
    return pending;
  }
  if (!options.force && current.ready && current.key === key) {
    return Promise.resolve(current.rows);
  }

  // Keep whatever `ready` was set last time across refetches. A stale-but-known
  // state should NOT briefly flip to "loading" on a bump — that's what caused
  // every consumer (Hub catalog, chat picker, train picker) to re-render with a
  // spinner the instant a download completed elsewhere, producing the laggy
  // feel users reported. The forthcoming success path resolves with fresh
  // data; we just keep showing the previous rows until then.
  updateSourceState(source, {
    loading: true,
    ready: current.ready,
    error: null,
    key,
  });

  const request = runSourceFetch(source, options.hfToken)
    .then((rows) => {
      if (useDeviceInventoryStore.getState()[source].key === key) {
        updateSourceState(source, {
          rows,
          loading: false,
          ready: true,
          error: null,
          key,
        });
      }
      return rows;
    })
    .catch((error) => {
      if (useDeviceInventoryStore.getState()[source].key === key) {
        updateSourceState(source, {
          loading: false,
          ready: false,
          error: fetchErrorMessage(source, error),
          key,
        });
      }
      throw error;
    })
    .finally(() => {
      if (inFlight.get(key) === request) {
        inFlight.delete(key);
      }
    });

  inFlight.set(key, request);
  return request;
}

export function markInventorySourcesFresh(
  sources: readonly DeviceInventorySource[],
  options: MarkInventoryFreshOptions,
): void {
  const tokenFingerprint =
    options.tokenFingerprint ?? fingerprintToken(options.hfToken);
  useDeviceInventoryStore.setState((state) => {
    const patch: Partial<DeviceInventoryState> = {};
    let changed = false;
    for (const source of sources) {
      const current = state[source];
      if (!current.ready) {
        continue;
      }
      const key = sourceRequestKey(
        source,
        options.inventoryVersion,
        tokenFingerprint,
      );
      if (current.key === key) {
        continue;
      }
      patch[source] = { ...current, key } as never;
      changed = true;
    }
    return changed ? patch : state;
  });
}

export function useDeviceInventorySources<
  const Sources extends readonly DeviceInventorySource[],
>(
  sources: Sources,
  options: UseDeviceInventoryOptions = {},
): UseDeviceInventoryResult<Sources> {
  const rawInventoryVersion = useInventoryVersion();
  const rawHfToken = options.hfToken ?? undefined;
  const debouncedInventoryVersion = useDebouncedValue(
    rawInventoryVersion,
    TOKEN_SCOPED_INVENTORY_DEBOUNCE_MS,
  );
  const debouncedHfToken = useDebouncedValue(
    rawHfToken,
    TOKEN_SCOPED_INVENTORY_DEBOUNCE_MS,
  );
  const enabled = options.enabled ?? true;
  const sourceKey = sources.join("|");
  const sourceList = useMemo(
    () => sourceKey.split("|").filter(Boolean) as DeviceInventorySource[],
    [sourceKey],
  );
  const hasTokenScopedSources = useMemo(
    () => sourceList.some((source) => TOKEN_SCOPED_SOURCES.has(source)),
    [sourceList],
  );
  const inventoryVersion = hasTokenScopedSources
    ? debouncedInventoryVersion
    : rawInventoryVersion;
  const hfToken = hasTokenScopedSources ? debouncedHfToken : rawHfToken;
  const tokenFingerprint = useMemo(() => fingerprintToken(hfToken), [hfToken]);
  const state = useDeviceInventoryStore(
    useShallow((s) => {
      const selected = {} as Pick<DeviceInventoryState, Sources[number]>;
      for (const source of sourceList) {
        (selected as Partial<DeviceInventoryState>)[source] = s[
          source
        ] as never;
      }
      return selected;
    }),
  );

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled(
      sourceList.map((source) =>
        fetchInventorySource(source, {
          hfToken,
          tokenFingerprint,
          inventoryVersion,
          force: true,
        }),
      ),
    );
    for (const [index, result] of results.entries()) {
      const source = sourceList[index];
      if (source && result.status === "rejected") {
        logInventorySourceFailure(source, result.reason);
      }
    }
  }, [sourceList, hfToken, tokenFingerprint, inventoryVersion]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    for (const source of sourceList) {
      fetchInventorySource(source, {
        hfToken,
        tokenFingerprint,
        inventoryVersion,
      }).catch((error) => logInventorySourceFailure(source, error));
    }
  }, [enabled, sourceList, hfToken, tokenFingerprint, inventoryVersion]);

  return {
    ...state,
    refresh,
  } as UseDeviceInventoryResult<Sources>;
}
