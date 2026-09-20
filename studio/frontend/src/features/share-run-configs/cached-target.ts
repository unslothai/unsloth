// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  type GgufVariantsResponse,
  buildLocalInventoryRows,
  fetchInventorySource,
  hubTokenHeader,
  useDeviceInventoryStore,
} from "@/features/hub";
import {
  ggufVariantsQuery,
  runBoundedVariantsRequest,
} from "../chat/api/gguf-variants-request";
import {
  INVENTORY_FRESHNESS_WINDOW_MS,
  isInventoryStampFresh,
} from "../hub/inventory/inventory-freshness";
import {
  ggufVariantsMatch,
  residentModelIdMatches,
} from "../model-picker/model-config/model-identity";
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
  return runBoundedVariantsRequest(options.signal, async (signal) => {
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
    const [cached, local] = await Promise.all([
      readInventory(target.meta.isGguf ? "cachedGguf" : "cachedModels"),
      readInventory("localModels"),
    ]);
    signal.throwIfAborted();
    const candidates = [
      ...cached
        .filter(
          (row) =>
            !row.partial && residentModelIdMatches(target.id, row.repo_id),
        )
        .map((row) => ({
          loadId: row.load_id || row.repo_id,
          localPath:
            row.load_id && row.load_id !== row.repo_id
              ? row.load_id
              : row.cache_path,
        })),
      ...buildLocalInventoryRows(local)
        .filter(
          (row) =>
            !row.partial &&
            row.isGguf === target.meta.isGguf &&
            residentModelIdMatches(target.id, row.modelId, row.loadId),
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
      const query = ggufVariantsQuery(candidate.loadId, candidate, true);
      const response = await authFetch(`/api/hub/gguf-variants?${query}`, {
        headers: hubTokenHeader(options.hfToken),
        signal,
      });
      if (!response.ok) {
        throw new Error("Could not check cached GGUF variants.");
      }
      const listing: GgufVariantsResponse = await response.json();
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
            ggufVariant: target.meta.ggufVariant ?? variant.quant,
            ggufFilename: variant.filename,
          },
        };
      }
    }
    return target;
  });
}
