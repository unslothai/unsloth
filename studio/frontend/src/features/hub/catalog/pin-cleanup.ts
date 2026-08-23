// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cachePathKey } from "@/lib/cache-path-key";
import {
  type ParsedPinKey,
  hasPinnedArtifact,
  removePinnedArtifactIfPresent,
} from "@/stores/pinned-models";

interface CacheCopyEvidence {
  cache_path: string;
  partial: boolean;
}

interface CachedRepoEvidence {
  repo_id: string;
  cache_path?: string | null;
  cache_copies?: readonly CacheCopyEvidence[];
  partial?: boolean;
}

interface LocalRepoEvidence {
  source: string;
  model_id?: string | null;
  model_format?: string | null;
  path: string;
  partial?: boolean;
}

export interface LocalPinCleanupEvidence {
  plainPinMayRemain: boolean;
  ggufState: "absent" | "represented" | "uncertain";
  ggufCacheCopies: readonly { cachePath: string }[];
}

const NO_GGUF_QUANTS = new Set<string>();

export function localPinInventoryNeeds(pins: readonly ParsedPinKey[]): {
  gguf: boolean;
  models: boolean;
} {
  return {
    gguf: pins.some((pin) => pin.quant !== null),
    models: pins.some((pin) => pin.quant === null),
  };
}

function hasCompleteCopy(row: CachedRepoEvidence): boolean {
  return row.cache_copies && row.cache_copies.length > 0
    ? row.cache_copies.some((copy) => !copy.partial)
    : row.partial !== true;
}

export function buildLocalPinCleanupEvidence(
  repoId: string,
  ggufRows: readonly CachedRepoEvidence[],
  modelRows: readonly CachedRepoEvidence[],
  localRows: readonly LocalRepoEvidence[],
): LocalPinCleanupEvidence {
  const normalizedRepoId = repoId.trim().toLowerCase();
  const matchesRepo = (candidate: string) =>
    candidate.trim().toLowerCase() === normalizedRepoId;
  const matchingGgufRows = ggufRows.filter((row) => matchesRepo(row.repo_id));
  const matchingModelRows = modelRows.filter((row) => matchesRepo(row.repo_id));
  const matchingLocalRows = localRows.filter(
    (row) =>
      row.source === "hf_cache" && matchesRepo(row.model_id?.trim() ?? ""),
  );
  const completeLocalRows = matchingLocalRows.filter(
    (row) => row.partial !== true,
  );
  const plainPinMayRemain =
    matchingGgufRows.some(hasCompleteCopy) ||
    matchingModelRows.some(hasCompleteCopy) ||
    completeLocalRows.length > 0;
  const matchingQuantLocalRows = completeLocalRows.filter(
    (row) =>
      !row.model_format ||
      row.model_format === "gguf" ||
      row.model_format === "unknown",
  );
  const ggufCacheCopies = matchingGgufRows.flatMap((row) => [
    ...(row.cache_path ? [{ cachePath: row.cache_path }] : []),
    ...(row.cache_copies ?? []).map((copy) => ({
      cachePath: copy.cache_path,
    })),
  ]);
  const remainingCachePathKeys = ggufCacheCopies
    .map((copy) => cachePathKey(copy.cachePath))
    .filter(Boolean);
  const hasUnrepresentedLocalCopy = matchingQuantLocalRows.some((row) => {
    const localPath = cachePathKey(row.path);
    return !remainingCachePathKeys.some(
      (cachePath) =>
        localPath === cachePath || localPath.startsWith(`${cachePath}/`),
    );
  });
  const ggufState =
    matchingGgufRows.length === 0 && matchingQuantLocalRows.length === 0
      ? "absent"
      : ggufCacheCopies.length === 0 || hasUnrepresentedLocalCopy
        ? "uncertain"
        : "represented";
  return { plainPinMayRemain, ggufState, ggufCacheCopies };
}

export function pinsToRemoveAfterLocalCacheDelete(
  pins: readonly ParsedPinKey[],
  evidence: Pick<LocalPinCleanupEvidence, "plainPinMayRemain" | "ggufState">,
  representedGgufQuants: ReadonlySet<string> | null = null,
): ParsedPinKey[] {
  const remainingGgufQuants =
    evidence.ggufState === "absent"
      ? NO_GGUF_QUANTS
      : evidence.ggufState === "uncertain"
        ? null
        : representedGgufQuants;
  return pins.filter((pin) =>
    pin.quant === null
      ? !evidence.plainPinMayRemain
      : remainingGgufQuants !== null &&
        !remainingGgufQuants.has(pin.quant.toLowerCase()),
  );
}

export async function removeQuantPinIfNoCopyRemains(
  repoId: string,
  quant: string,
  getRemainingQuants: () => Promise<ReadonlySet<string> | null>,
): Promise<boolean> {
  if (!hasPinnedArtifact(repoId, quant)) {
    return false;
  }
  const remainingQuants = await getRemainingQuants();
  return remainingQuants !== null &&
    !remainingQuants.has(quant.trim().toLowerCase())
    ? removePinnedArtifactIfPresent(repoId, quant)
    : false;
}
