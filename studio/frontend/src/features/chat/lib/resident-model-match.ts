// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  ggufVariantsMatch,
  isStandaloneGgufPath,
  modelIdsMatch,
  residentModelIdMatches,
} from "@/features/hub/lib/model-identity";

/** The names one model pick answers to: the catalog id its picker row shows, and the
 * `model_path` its load sends, which differ whenever a cached row pins a snapshot dir. */
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
 * Whether the model already resident on the inference server is the one this pick names.
 *
 * Both sides hold two strings for one model and they need not be the same one: a pinned
 * cached row loads by snapshot path while its picker row keeps the repo id, and the status
 * publishes the clean public id next to the raw load identifier. Loading a model that is
 * already resident returns `already_loaded` without touching llama-server, so a caller that
 * recognises it here can skip both the reload and the confirmation that goes with it.
 *
 * A standalone `.gguf` is exempt from the variant comparison: it is one file with no quant to
 * choose between, and the backend labels the resident copy from its filename while the picker
 * row deliberately carries none.
 *
 * The public id is not enough on its own. Two snapshots of one cached repo report the same
 * `active_model`, and the inventory repoints `load_id` at the newest, so a repo that advanced
 * under a resident older snapshot would read as resident and keep serving stale weights.
 */
export function residentModelMatchesPick(
  status: ResidentModelStatus,
  pick: ModelPickNames,
): boolean {
  if (!status.active_model) {
    return false;
  }
  // see settingsGgufVariantForRow: the row for a standalone file carries no quant label
  const picksItsOwnVariant = !(
    !pick.ggufVariant && isStandaloneGgufPath(pick.loadPath ?? pick.id)
  );
  // a quant switch within one repo is a real reload, so the variant has to agree too
  if (
    picksItsOwnVariant &&
    !ggufVariantsMatch(status.gguf_variant, pick.ggufVariant)
  ) {
    return false;
  }
  // the public id names every snapshot of a repo, so only the raw identifier settles which weights are resident, as LlamaCppBackend.matches_load_source
  if (status.model_identifier) {
    return modelIdsMatch(status.model_identifier, pick.loadPath ?? pick.id);
  }
  // a native-lease load withholds the raw path and reports its display label alone
  return residentModelIdMatches(status.active_model, pick.id, pick.loadPath);
}
