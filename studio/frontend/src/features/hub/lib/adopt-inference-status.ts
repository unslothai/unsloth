// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Landing straight on /hub is the only entry point where nothing else applies inference status.

import { ggufVariantsMatch, modelIdsMatch } from "./model-identity.ts";

export interface ResidentAdoptionState {
  checkpoint: string | null;
  checkpointIsExternal: boolean;
  activeGgufVariant: string | null;
  modelLoading: boolean;
  /** From `/openai-auto-switch`; `/status` says nothing about it. */
  idleUnloadArmed: boolean;
}

export interface ResidentStatusFacts {
  /** Null when nothing is loaded and for a speech model; `speechOnly` tells them apart. */
  checkpointId: string | null;
  ggufVariant: string | null;
  /** The slot holds a speech model rather than being empty. */
  speechOnly?: boolean;
}

export interface ResidentAdoptionActions {
  setCheckpoint: (checkpointId: string, ggufVariant: string | null) => void;
  clearCheckpoint?: () => void;
  /** Receives store values from before `setCheckpoint`, to tell hydration from steady state. */
  applyStatus: (previous: {
    checkpoint: string | null;
    ggufVariant: string | null;
  }) => void;
}

/** Never loads or unloads: only mirrors what the server has. */
export function adoptResidentModelStatus(
  status: ResidentStatusFacts,
  state: ResidentAdoptionState,
  actions: ResidentAdoptionActions,
): boolean {
  const { checkpointId } = status;
  if (state.checkpointIsExternal) {
    return false;
  }
  // A load applies its own status when it settles and owns the store meanwhile.
  if (state.modelLoading) {
    return false;
  }
  if (!checkpointId) {
    // Empty status is ambiguous: when idle unload is armed the stash reloads the model, so keep it.
    // A speech model took the slot outright, so it does not count.
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
  // Unconditional: a rehydrated checkpoint lacks the fields saying how the model was launched.
  actions.applyStatus(previous);
  return true;
}
