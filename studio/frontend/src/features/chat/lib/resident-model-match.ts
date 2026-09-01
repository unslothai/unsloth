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

/** The two names one pick answers to: its picker row's catalog id, and the `model_path` its load
 *  sends. They differ whenever a cached row pins a snapshot dir. */
export type ModelPickNames = {
  id: string;
  loadPath?: string | null;
  ggufVariant?: string | null;
};

/** The resident model as `/api/inference/status` reports it. */
export type ResidentModelStatus = {
  active_model?: string | null;
  model_identifier?: string | null;
  gguf_variant?: string | null;
};

/** Whether the resident model is the one this pick names. Each side holds two strings for one
 *  model and they need not be equal: a pinned cached row loads by snapshot path while its row
 *  keeps the repo id, and the status publishes the public id beside the raw one. Recognising
 *  that skips a reload `/load` would have answered `already_loaded`. */
export function residentModelMatchesPick(
  status: ResidentModelStatus,
  pick: ModelPickNames,
): boolean {
  if (!status.active_model) {
    return false;
  }
  // A standalone file has no quant to choose between, and its row carries no label
  // (settingsGgufVariantForRow) while the backend derives one from the filename.
  const picksItsOwnVariant = !(
    !pick.ggufVariant && isStandaloneGgufPath(pick.loadPath ?? pick.id)
  );
  // a quant switch within one repo is a real reload
  if (
    picksItsOwnVariant &&
    !ggufVariantsMatch(status.gguf_variant, pick.ggufVariant)
  ) {
    return false;
  }
  // Only the raw identifier says WHICH weights are resident, as matches_load_source does: every
  // snapshot of a repo publishes the same public id.
  if (status.model_identifier) {
    return modelIdsMatch(status.model_identifier, pick.loadPath ?? pick.id);
  }
  // No raw identifier. The name this pick loads by still names one revision, so try it literally
  // first: an older backend put the raw path in active_model.
  const loadName = pick.loadPath ?? pick.id;
  if (modelIdsMatch(status.active_model, loadName)) {
    return true;
  }
  // What is left collapses onto a public id, which cannot tell one snapshot of a repo from
  // another, so a pick pinned to one reloads rather than risk keeping the old weights.
  if (isHfCacheSnapshotPath(loadName)) {
    return false;
  }
  // a native lease withholds the raw path and reports its display label alone
  return residentModelIdMatches(status.active_model, pick.id, pick.loadPath);
}
