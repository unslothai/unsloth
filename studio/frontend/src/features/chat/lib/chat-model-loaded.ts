// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the header may call the picked model loaded. Its own plain module so the node suite can
// drive it: runtime-provider.tsx pulls in the whole chat runtime.

export type ChatModelLoadedInput = {
  /** The picker selection, "" when nothing is picked. */
  checkpoint: string;
  /** Callers that show their own loading state can leave this out. */
  modelLoading?: boolean;
  /** An API model: served remotely, so nothing is resident here. */
  isExternalModel: boolean;
  /** What /api/inference/status holds; undefined before the first read. */
  residentCheckpoint: string | null | undefined;
};

/** Resident, not merely picked. Loading an image or video model evicts the chat model (the GPU
 *  arbiter allows a single owner) and leaves the selection alone, so a check on the selection
 *  reported a model the backend had released and the next prompt failed with a bare 400.
 *  `undefined` means the first status read has not landed: assume loaded, or the header flashes
 *  "not loaded" on every startup. */
export function chatModelLoaded({
  checkpoint,
  modelLoading = false,
  isExternalModel,
  residentCheckpoint,
}: ChatModelLoadedInput): boolean {
  if (!checkpoint || modelLoading) return false;
  return isExternalModel || residentCheckpoint !== null;
}
