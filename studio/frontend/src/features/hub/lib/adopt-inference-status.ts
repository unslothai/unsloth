// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Adopting the resident model into the chat runtime store from the Hub.
//
// Landing straight on /hub is the one entry point where nothing has applied
// /api/inference/status yet. Pinning only the checkpoint leaves every other field
// useActiveModelConfig reads at its default, so the settings page would pass those
// defaults on as the live config and Apply would reload with them. Adoption therefore
// applies the whole status, exactly as the chat runtime's refresh does.

import { ggufVariantsMatch, modelIdsMatch } from "./model-identity.ts";

/** The parts of the chat runtime store adoption has to look at. */
export interface ResidentAdoptionState {
  /** ``params.checkpoint``. */
  checkpoint: string | null;
  /** Whether that checkpoint names an external provider's model. */
  checkpointIsExternal: boolean;
  /** ``activeGgufVariant``. */
  activeGgufVariant: string | null;
  /** ``modelLoading``: a load this tab started still owns the store. */
  modelLoading: boolean;
}

/** What ``/api/inference/status`` says is resident, already resolved. */
export interface ResidentStatusFacts {
  /** ``resolveInferenceCheckpointId(status)``; null when nothing is loaded. */
  checkpointId: string | null;
  /** ``status.gguf_variant``. */
  ggufVariant: string | null;
}

export interface ResidentAdoptionActions {
  /** Re-pin ``params.checkpoint`` onto the resident model. */
  setCheckpoint: (checkpointId: string, ggufVariant: string | null) => void;
  /**
   * Apply the rest of the status. Receives the store values from BEFORE
   * ``setCheckpoint`` ran, which is how applyActiveModelStatusToStore tells a
   * hydration from steady state.
   */
  applyStatus: (previous: {
    checkpoint: string | null;
    ggufVariant: string | null;
  }) => void;
}

/**
 * Adopt the resident model reported by ``/api/inference/status``.
 *
 * Returns whether anything was adopted. Never loads or unloads a model: it only
 * mirrors what the server already has.
 */
export function adoptResidentModelStatus(
  status: ResidentStatusFacts,
  state: ResidentAdoptionState,
  actions: ResidentAdoptionActions,
): boolean {
  const { checkpointId } = status;
  // An external selection has no local mirror, so the resident GGUF's settings would
  // describe a model the user is not talking to.
  if (state.checkpointIsExternal) {
    return false;
  }
  // A load this tab started applies its own status when it settles, and owns the params
  // meanwhile. Adopting underneath it would fight both.
  if (state.modelLoading) {
    return false;
  }
  if (!checkpointId) {
    // An empty status is not the same as a model going away: an idle unload frees the
    // model but keeps a stash the next request reloads, and /status carries no field
    // telling the two apart. The store names the model meanwhile, so leave it pinned;
    // clearing belongs to whatever performed the unload, not to an observation of one.
    return false;
  }
  const previous = {
    checkpoint: state.checkpoint,
    ggufVariant: state.activeGgufVariant,
  };
  const alreadyPinned =
    modelIdsMatch(previous.checkpoint, checkpointId) &&
    ggufVariantsMatch(previous.ggufVariant, status.ggufVariant);
  if (!alreadyPinned) {
    actions.setCheckpoint(checkpointId, status.ggufVariant);
  }
  // Unconditional, even when the checkpoint matched: a persisted checkpoint rehydrates
  // from localStorage without the fields saying how the model was launched.
  actions.applyStatus(previous);
  return true;
}
