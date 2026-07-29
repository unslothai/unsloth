// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Adopting the resident model into the chat runtime store from the Hub.
//
// Landing straight on /hub (a reload, or a deep link after an OpenAI-compatible
// auto-switch loaded something else) is the one entry point where nothing has
// applied /api/inference/status yet: useChatModelRuntime has no mount sync and
// the chat page is a different route. Pinning only the checkpoint leaves every
// other field useActiveModelConfig reads at its default, so the Hub's settings
// page passes those defaults on as the resident model's live config and Apply
// reloads the model with them. Adoption therefore has to apply the whole status,
// exactly as the chat runtime's refresh does.

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
   * Drop a local checkpoint the server no longer has.
   *
   * Optional so a caller that only wants the pinning half can leave it out.
   */
  clearCheckpoint?: () => void;
  /**
   * Apply the rest of the status. Receives the store values from BEFORE
   * ``setCheckpoint`` ran, which is what applyActiveModelStatusToStore needs to
   * tell a hydration from steady state.
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
  // An external-provider selection has no local mirror, so stamping the resident
  // GGUF's capabilities and launch settings onto it would describe a model the
  // user is not talking to. It also owns the store, so an empty status must not
  // clear it: clearCheckpoint drops the persisted external pick as well.
  if (state.checkpointIsExternal) {
    return false;
  }
  // A load this tab started applies its own status when it settles, and the load
  // dialog owns the params meanwhile. Adopting underneath it would fight both.
  if (state.modelLoading) {
    return false;
  }
  if (!checkpointId) {
    // The server has nothing loaded, so neither should we. Unloading from another
    // tab, from the monitor or over the API leaves this store pinned otherwise,
    // and the settings page goes on treating that row as resident and seeding the
    // editor from a launch config nothing is running.
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
  // Unconditional, even when the checkpoint already matched: a persisted
  // checkpoint rehydrates from localStorage on its own, with none of the fields
  // that say how the model was actually launched.
  actions.applyStatus(previous);
  return true;
}
