// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Import only non-React identity helpers and avoid the model-picker barrel's Chat cycle.
import {
  isHfCacheSnapshotPath,
  isStandaloneGgufPath,
  looksLikeModelPath,
  publicModelId,
} from "@/features/model-picker/model-config/model-identity";
import {
  ggufVariantsMatch,
  modelIdsMatch,
} from "@/lib/model-identity";

/**
 * Whether the public model id reported by the inference status names one of the
 * candidate picker identities. Only a namespaced public collapse is unambiguous.
 */
export function residentModelIdMatches(
  activeModelId: string | null | undefined,
  ...candidates: (string | null | undefined)[]
): boolean {
  if (candidates.some((candidate) => modelIdsMatch(activeModelId, candidate))) {
    return true;
  }
  const active = activeModelId?.trim();
  // A path-shaped active id is the raw identifier, which the literal pass covered.
  if (!active || looksLikeModelPath(active)) {
    return false;
  }
  return candidates.some((candidate) => {
    const trimmed = candidate?.trim();
    if (!trimmed) {
      return false;
    }
    const publicId = publicModelId(trimmed);
    return publicId.includes("/") && modelIdsMatch(active, publicId);
  });
}

/** The two names one pick answers to: its picker row's catalog id, and the `model_path`
 * its load sends. They differ whenever a cached row pins a snapshot dir. */
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

/**
 * Whether the resident model is the one this pick names.
 *
 * Each side holds two strings for one model and they need not be equal: a pinned cached row
 * loads by snapshot path while its row keeps the repo id, and the status publishes the public
 * id beside the raw one. Recognising that here skips a reload `/load` would have answered
 * `already_loaded`, and the confirmation that goes with it.
 */
export function residentModelMatchesPick(
  status: ResidentModelStatus,
  pick: ModelPickNames,
): boolean {
  if (!status.active_model) {
    return false;
  }
  // A standalone file has no quant to choose between, and its row carries no label
  // while the backend derives a variant from the filename.
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
  // Only the raw identifier says WHICH weights are resident, as matches_load_source does:
  // every snapshot of a repo publishes the same public id.
  if (status.model_identifier) {
    return modelIdsMatch(status.model_identifier, pick.loadPath ?? pick.id);
  }
  // No raw identifier. The name this pick loads by still names one revision, so try it
  // literally first: an older backend put the raw path in active_model.
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
