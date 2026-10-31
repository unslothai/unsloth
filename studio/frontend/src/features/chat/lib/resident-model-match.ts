// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  ggufVariantsMatch,
  isHfCacheSnapshotPath,
  isStandaloneGgufPath,
  modelIdsMatch,
  residentModelIdMatches,
} from "@/features/hub/lib/model-identity";

/** Catalog id and load path; they differ when a cached row pins a snapshot dir. */
export type ModelPickNames = {
  id: string;
  loadPath?: string | null;
  ggufVariant?: string | null;
};

export type ResidentModelStatus = {
  active_model?: string | null;
  model_identifier?: string | null;
  gguf_variant?: string | null;
};

/** Each side has two names for one model; matching them avoids a needless reload. */
export function residentModelMatchesPick(
  status: ResidentModelStatus,
  pick: ModelPickNames,
): boolean {
  if (!status.active_model) {
    return false;
  }
  // A standalone file has no quant choice; the backend derives a variant from its name.
  const picksItsOwnVariant = !(
    !pick.ggufVariant && isStandaloneGgufPath(pick.loadPath ?? pick.id)
  );
  if (
    picksItsOwnVariant &&
    !ggufVariantsMatch(status.gguf_variant, pick.ggufVariant)
  ) {
    return false;
  }
  // Only the raw identifier says which snapshot is resident.
  if (status.model_identifier) {
    return modelIdsMatch(status.model_identifier, pick.loadPath ?? pick.id);
  }
  // An older backend put the raw path in active_model.
  const loadName = pick.loadPath ?? pick.id;
  if (modelIdsMatch(status.active_model, loadName)) {
    return true;
  }
  // A pinned snapshot reloads rather than risk keeping the old weights.
  if (isHfCacheSnapshotPath(loadName)) {
    return false;
  }
  return residentModelIdMatches(status.active_model, pick.id, pick.loadPath);
}
