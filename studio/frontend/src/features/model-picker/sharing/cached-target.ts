// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { ggufVariantsQuery, runBoundedVariantsRequest } from "@/features/chat";
import {
  type GgufVariantsResponse,
  INVENTORY_FRESHNESS_WINDOW_MS,
  buildLocalInventoryRows,
  fetchInventorySource,
  ggufVariantsMatch,
  hubTokenHeader,
  isInventoryStampFresh,
  listGgufVariants,
  residentModelIdMatches,
  useDeviceInventoryStore,
  withAbort,
} from "@/features/hub";
import {
  isOllamaModelId,
  isStandaloneGgufPath,
} from "../model-config/model-identity";
import type { resolveRunConfigTarget } from "./target";

type RunConfigTarget = NonNullable<ReturnType<typeof resolveRunConfigTarget>>;
type ResolutionOptions = {
  hfToken?: string;
  inventoryVersion: number;
  signal: AbortSignal;
  checkLocalPath?: boolean;
};
type CachedVariantsResponse = Pick<
  GgufVariantsResponse,
  "variants" | "default_variant" | "dependencies_resolved"
> & {
  resolved_locally?: boolean;
};

export class RunConfigResolutionError extends Error {}

function readVariantsResponse(value: unknown): CachedVariantsResponse {
  if (!value || typeof value !== "object") {
    throw new Error("Invalid GGUF variants response.");
  }
  const response = value as Partial<CachedVariantsResponse>;
  if (
    !Array.isArray(response.variants) ||
    !response.variants.every(
      (variant) =>
        variant &&
        typeof variant === "object" &&
        typeof variant.filename === "string" &&
        variant.filename.length > 0 &&
        typeof variant.quant === "string" &&
        variant.quant.length > 0 &&
        typeof variant.size_bytes === "number" &&
        Number.isFinite(variant.size_bytes) &&
        variant.size_bytes >= 0 &&
        (variant.downloaded === undefined ||
          typeof variant.downloaded === "boolean") &&
        (variant.partial === undefined || typeof variant.partial === "boolean"),
    ) ||
    (response.default_variant != null &&
      typeof response.default_variant !== "string") ||
    (response.resolved_locally !== undefined &&
      typeof response.resolved_locally !== "boolean") ||
    (response.dependencies_resolved !== undefined &&
      typeof response.dependencies_resolved !== "boolean")
  ) {
    throw new Error("Invalid GGUF variants response.");
  }
  return {
    variants: response.variants,
    default_variant: response.default_variant ?? null,
    resolved_locally: response.resolved_locally === true,
    dependencies_resolved: response.dependencies_resolved === true,
  };
}

function listCachedVariants(
  id: string,
  localPath: string | undefined,
  options: { hfToken?: string; signal: AbortSignal },
): Promise<CachedVariantsResponse> {
  return runBoundedVariantsRequest(options.signal, async (signal) => {
    const query = ggufVariantsQuery(id, { localPath }, true);
    const response = await authFetch(`/api/hub/gguf-variants?${query}`, {
      headers: hubTokenHeader(options.hfToken),
      signal,
    });
    if (!response.ok) {
      throw new Error("Could not check cached GGUF variants.");
    }
    return readVariantsResponse(await response.json());
  });
}

function findCachedVariant(
  target: RunConfigTarget,
  listing: CachedVariantsResponse,
) {
  const requested = target.meta.ggufVariant ?? listing.default_variant;
  return listing.variants.find(
    (entry) =>
      entry.downloaded === true &&
      !entry.partial &&
      (ggufVariantsMatch(requested, entry.quant) ||
        requested === entry.filename),
  );
}

function resolveLocalTarget(
  target: RunConfigTarget,
  listing: CachedVariantsResponse,
): RunConfigTarget {
  if (isStandaloneGgufPath(target.id)) {
    return {
      ...target,
      meta: { ...target.meta, isGguf: true, ggufVariant: undefined },
    };
  }
  if (listing.variants.length === 0) {
    return {
      ...target,
      meta: { ...target.meta, isGguf: false, ggufVariant: undefined },
    };
  }
  const variant = findCachedVariant(target, listing);
  if (!variant) {
    throw new RunConfigResolutionError(
      "The selected folder does not contain a complete copy of the requested GGUF variant.",
    );
  }
  return {
    ...target,
    meta: {
      ...target.meta,
      isGguf: true,
      isDownloaded: true,
      ggufVariant: variant.quant,
      ggufFilename: variant.filename,
    },
  };
}

export async function resolveCachedRunConfigTarget(
  target: RunConfigTarget,
  options: ResolutionOptions,
): Promise<RunConfigTarget> {
  options.signal.throwIfAborted();
  if (target.meta.isDownloaded) {
    return target;
  }
  if (target.meta.source !== "hub") {
    if (isStandaloneGgufPath(target.id) || isOllamaModelId(target.id)) {
      return target;
    }
    const listing = await listCachedVariants(target.id, target.id, options);
    options.signal.throwIfAborted();
    return resolveLocalTarget(target, listing);
  }
  if (options.checkLocalPath) {
    const listing = await listCachedVariants(target.id, target.id, options);
    options.signal.throwIfAborted();
    if (listing.resolved_locally) {
      return resolveLocalTarget(
        {
          ...target,
          id: `./${target.id}`,
          meta: { ...target.meta, source: "local" },
        },
        listing,
      );
    }
  }
  return resolveHubRunConfigTarget(target, options);
}

async function resolveHubRunConfigTarget(
  target: RunConfigTarget,
  options: ResolutionOptions,
): Promise<RunConfigTarget> {
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
    let listing: CachedVariantsResponse;
    try {
      listing = await listCachedVariants(
        candidate.loadId,
        candidate.localPath,
        options,
      );
    } catch {
      options.signal.throwIfAborted();
      continue;
    }
    const variant = findCachedVariant(target, listing);
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
  return resolveHubVariantTarget(target, options);
}

async function resolveHubVariantTarget(
  target: RunConfigTarget,
  options: ResolutionOptions,
): Promise<RunConfigTarget> {
  if (
    !target.meta.isGguf ||
    (target.meta.ggufVariant &&
      !target.meta.ggufVariant.toLowerCase().endsWith(".gguf"))
  ) {
    return target;
  }
  let listing: CachedVariantsResponse;
  try {
    listing = readVariantsResponse(
      await listGgufVariants(target.id, options.hfToken, {
        signal: options.signal,
      }),
    );
  } catch {
    options.signal.throwIfAborted();
    return target;
  }
  options.signal.throwIfAborted();
  const requested = target.meta.ggufVariant ?? listing.default_variant;
  const variant = listing.variants.find((entry) =>
    target.meta.ggufVariant
      ? requested === entry.filename
      : ggufVariantsMatch(requested, entry.quant),
  );
  if (!variant) {
    if (!listing.dependencies_resolved) {
      return target;
    }
    throw new RunConfigResolutionError(
      "The shared GGUF variant is unavailable for this model. Ask the sender for an updated link.",
    );
  }
  return {
    ...target,
    meta: {
      ...target.meta,
      ggufVariant: variant.quant,
      ggufFilename: variant.filename,
      isDownloaded: variant.downloaded === true && !variant.partial,
    },
  };
}
