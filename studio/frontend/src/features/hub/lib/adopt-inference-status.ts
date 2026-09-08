// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Landing straight on /hub is the one entry point where nothing else applies
// /api/inference/status. Cache mutation guards need the same resident state as Chat.

import { ggufVariantsMatch, modelIdsMatch } from "./model-identity.ts";

/** The parts of the chat runtime store adoption has to look at. */
export interface ResidentAdoptionState {
  checkpoint: string | null;
  /** Whether that checkpoint names an external provider's model. */
  checkpointIsExternal: boolean;
  activeGgufVariant: string | null;
  /** ``modelLoading``: a model load still owns the store. */
  modelLoading: boolean;
  /** Whether the idle-unload loop will actually free the model. From
   * ``/openai-auto-switch``; ``/status`` says nothing about it. */
  idleUnloadArmed: boolean;
}

/** What ``/api/inference/status`` says is resident, already resolved. */
export interface ResidentStatusFacts {
  /** ``resolveInferenceCheckpointId(status)``; null when nothing is loaded, and null
   * for a speech model, which chat cannot adopt. ``speechOnly`` tells the two apart. */
  checkpointId: string | null;
  ggufVariant: string | null;
  /** The slot holds a speech model rather than being empty. Chat cannot adopt one, but
   * it is not the idle eviction the rule below is about either: an Audio load took the
   * slot outright and no stash brings the chat model back. */
  speechOnly?: boolean;
}

export interface ResidentAdoptionActions {
  /** Re-pin ``params.checkpoint`` onto the resident model. */
  setCheckpoint: (checkpointId: string, ggufVariant: string | null) => void;
  /** Drop a checkpoint the server has genuinely unloaded. */
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
  // An external selection has no local mirror.
  if (state.checkpointIsExternal) {
    return false;
  }
  // A load applies its own status when it settles and owns the store meanwhile.
  if (state.modelLoading) {
    return false;
  }
  if (!checkpointId) {
    // An empty status means one of two things and /status cannot say which. Armed, the idle
    // loop frees the model but keeps a stash the next request reloads, so clearing would drop
    // a selection that is coming back. Disarmed, nothing brings it back.
    // A speech model is neither: an Audio load took the slot, not the idle loop, and no
    // stash reloads the chat model. Keeping the prior pick would leave stale Hub residency.
    if (state.idleUnloadArmed && !status.speechOnly) {
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
