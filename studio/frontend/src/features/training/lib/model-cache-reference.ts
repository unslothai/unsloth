// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedModelRepo,
  LocalModelInfo,
  ModelInventoryFormat,
} from "@/features/hub";
import { normalizeModelIdentity } from "@/features/hub/lib/model-identity.ts";
import { cachedInventoryPathMatchesSelection } from "./cache-reference.ts";
import { isTrainableModelFormat } from "./model-support.ts";

export type ModelCacheReference = {
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
};

export function modelInventoryIdentityMatches(
  row: CachedModelRepo | LocalModelInfo,
  key: string,
): boolean {
  return (
    ("repo_id" in row && normalizeModelIdentity(row.repo_id) === key) ||
    ("model_id" in row &&
      row.model_id != null &&
      normalizeModelIdentity(row.model_id) === key) ||
    (row.load_id != null && normalizeModelIdentity(row.load_id) === key) ||
    ("id" in row && normalizeModelIdentity(row.id) === key)
  );
}

/**
 * Resolve the cache copy that a training selection should point at.
 *
 * The same repo can be present under more than one HF cache root, and the
 * inventory returns every copy. Taking whichever usable row happens to come
 * first would silently retarget an explicit selection at a different copy, so a
 * selected path that is still usable wins; promotion to another copy only
 * happens once the selected one is gone from the inventory. With no selection
 * (`preferredPath` null/empty) every row matches and the first usable one wins,
 * which is the behaviour a fresh selection wants.
 */
export function selectModelCacheReference(
  cachedRows: readonly CachedModelRepo[],
  localRows: readonly LocalModelInfo[],
  model: string,
  preferredPath: string | null = null,
): ModelCacheReference | null {
  const key = normalizeModelIdentity(model);
  const usableCached = cachedRows.filter(
    (row) =>
      !row.partial &&
      isTrainableModelFormat(row.model_format) &&
      modelInventoryIdentityMatches(row, key),
  );
  const cachedMatch =
    usableCached.find((row) =>
      cachedInventoryPathMatchesSelection(
        row.cache_path ?? null,
        preferredPath,
      ),
    ) ?? usableCached[0];
  if (cachedMatch) {
    return {
      localPath: cachedMatch.cache_path ?? null,
      modelFormat: cachedMatch.model_format ?? null,
    };
  }
  const usableLocal = localRows.filter(
    (row) =>
      row.source === "hf_cache" &&
      !row.partial &&
      isTrainableModelFormat(row.model_format) &&
      modelInventoryIdentityMatches(row, key),
  );
  const localMatch =
    usableLocal.find((row) =>
      cachedInventoryPathMatchesSelection(row.path, preferredPath),
    ) ?? usableLocal[0];
  if (!localMatch) {
    return null;
  }
  return {
    localPath: localMatch.path,
    modelFormat: localMatch.model_format ?? null,
  };
}
