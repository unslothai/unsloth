// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain module so the node suite can drive it.

export type ChatModelLoadedInput = {
  checkpoint: string;
  modelLoading?: boolean;
  isExternalModel: boolean;
  residentCheckpoint: string | null | undefined;
};

/** Resident, not merely picked: other loads evict chat. `undefined` (no status yet) = loaded. */
export function chatModelLoaded({
  checkpoint,
  modelLoading = false,
  isExternalModel,
  residentCheckpoint,
}: ChatModelLoadedInput): boolean {
  if (!checkpoint || modelLoading) return false;
  return isExternalModel || residentCheckpoint !== null;
}
