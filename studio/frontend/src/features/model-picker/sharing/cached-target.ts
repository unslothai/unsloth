// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { ggufVariantsQuery, runBoundedVariantsRequest } from "@/features/chat";
import {
  type GgufVariantsResponse,
  INVENTORY_FRESHNESS_WINDOW_MS,
  isInventoryStampFresh,
  withAbort,
  buildLocalInventoryRows,
  fetchInventorySource,
  ggufVariantsMatch,
  hubTokenHeader,
  residentModelIdMatches,
  useDeviceInventoryStore,
} from "@/features/hub";
import type { resolveRunConfigTarget } from "./target";

type RunConfigTarget = NonNullable<ReturnType<typeof resolveRunConfigTarget>>;

export async function resolveCachedRunConfigTarget(
  target: RunConfigTarget,
  options: {
    hfToken?: string;
    inventoryVersion: number;
    signal: AbortSignal;
  },
): Promise<RunConfigTarget> {
  options.signal.throwIfAborted();
  if (target.meta.source !== "hub" || target.meta.isDownloaded) {
    return target;
  }
  const readInventory = <
    K extends "cachedGguf" | "cachedModels" | "localModels",
  >(
    source: K,
  ) => {
    const current = useDeviceInventoryStore.getState()[source];
    return fetchInventorySource(source, {
      ...options,
      force:
        current.ready &&
        !current.loading &&
        !isInventoryStampFresh(
          current.refreshedAt,
          Date.now(),
          INVENTORY_FRESHNESS_WINDOW_MS,
        ),
    });
  };
  const [cached, local] = await withAbort(
    Promise.allSettled([
      readInventory(target.meta.isGguf ? "cachedGguf" : "cachedModels"),
      readInventory("localModels"),
    ]),
    options.signal,
  );
  options.signal.throwIfAborted();
  let failure: unknown = [cached, local].find(
    (result) => result.status === "rejected",
  )?.reason;
  const candidates = [
    ...(cached.status === "fulfilled" ? cached.value : [])
      .filter(
        (row) => !row.partial && residentModelIdMatches(target.id, row.repo_id),
      )
      .map((row) => ({
        loadId: row.load_id || row.repo_id,
        localPath:
          row.load_id && row.load_id !== row.repo_id
            ? row.load_id
            : row.cache_path,
      })),
    ...buildLocalInventoryRows(local.status === "fulfilled" ? local.value : [])
      .filter(
        (row) =>
          !row.partial &&
          row.isGguf === target.meta.isGguf &&
          row.repoId !== null &&
          residentModelIdMatches(target.id, row.repoId),
      )
      .map((row) => ({
        loadId: row.loadId,
        localPath: row.loadId === row.repoId ? row.path : row.loadId,
      })),
  ];
  for (const candidate of candidates) {
    if (!target.meta.isGguf) {
      return {
        ...target,
        meta: {
          ...target.meta,
          loadId: candidate.loadId,
          isDownloaded: true,
        },
      };
    }
    let listing: GgufVariantsResponse;
    try {
      listing = await runBoundedVariantsRequest(
        options.signal,
        async (signal) => {
          const query = ggufVariantsQuery(candidate.loadId, candidate, true);
          const response = await authFetch(`/api/hub/gguf-variants?${query}`, {
            headers: hubTokenHeader(options.hfToken),
            signal,
          });
          if (!response.ok) {
            throw new Error("Could not check cached GGUF variants.");
          }
          return response.json();
        },
      );
    } catch (error) {
      options.signal.throwIfAborted();
      failure = error;
      continue;
    }
    const requested = target.meta.ggufVariant ?? listing.default_variant;
    const variant = listing.variants.find(
      (entry) =>
        entry.downloaded === true &&
        !entry.partial &&
        (ggufVariantsMatch(requested, entry.quant) ||
          requested === entry.filename),
    );
    if (variant) {
      return {
        ...target,
        meta: {
          ...target.meta,
          loadId: candidate.loadId,
          isDownloaded: true,
          ggufVariant: variant.quant,
          ggufFilename: variant.filename,
        },
      };
    }
  }
  if (failure) {
    throw failure;
  }
  return target;
}
