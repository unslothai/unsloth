// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Adopting the resident model into the chat runtime store from the Hub. Landing straight on /hub
// is the one entry point where nothing has applied /api/inference/status yet, and pinning only
// the checkpoint leaves every other useActiveModelConfig field at its default, which the settings
// page would pass on as the live config. So adoption applies the whole status, as chat does.

import { ggufVariantsMatch, modelIdsMatch } from "./model-identity.ts";

/** The parts of the chat runtime store adoption has to look at. */
export interface ResidentAdoptionState {
  checkpoint: string | null;
  /** Whether that checkpoint names an external provider's model. */
  checkpointIsExternal: boolean;
  activeGgufVariant: string | null;
  /** ``modelLoading``: a load this tab started still owns the store. */
  modelLoading: boolean;
  /** Whether the idle-unload loop will actually free the model. From
   * ``/openai-auto-switch``; ``/status`` says nothing about it. */
  idleUnloadArmed: boolean;
}

/** What ``/api/inference/status`` says is resident, already resolved. */
export interface ResidentStatusFacts {
  /** ``resolveInferenceCheckpointId(status)``; null when nothing is loaded. */
  checkpointId: string | null;
  ggufVariant: string | null;
}

export interface ResidentAdoptionActions {
  /** Re-pin ``params.checkpoint`` onto the resident model. */
  setCheckpoint: (checkpointId: string, ggufVariant: string | null) => void;
  /** Drop a checkpoint the server has genuinely unloaded. Optional, so a caller
   * that only wants the pinning half can leave it out. */
  clearCheckpoint?: () => void;
  /**
   * Apply the rest of the status. Receives the store values from BEFORE ``setCheckpoint`` ran,
   * which is how applyActiveModelStatusToStore tells a hydration from steady state.
   */
  applyStatus: (previous: {
    checkpoint: string | null;
    ggufVariant: string | null;
  }) => void;
}

/**
 * Adopt the resident model reported by ``/api/inference/status``. Returns whether anything was
 * adopted. Never loads or unloads: it only mirrors what the server already has.
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
    // An empty status means one of two things and /status cannot say which. Armed, the idle
    // loop frees the model but keeps a stash the next request reloads, so clearing would drop
    // a selection that is coming back. Disarmed, nothing brings it back, and leaving the row
    // resident seeds the settings editor from a launch config nothing is running.
    if (state.idleUnloadArmed) {
      return false;
    }
    if (state.checkpoint) {
      actions.clearCheckpoint?.();
      return true;
    }
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
