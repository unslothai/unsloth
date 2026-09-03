// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// its own plain module so the node suite can drive it: chat-page.tsx pulls in the whole chat runtime

export type KnownContextWindowInput = {
  // the window the resident backend serves; null where none sizes one
  loadedContextLength: number | null;
  // a load in flight still carries the outgoing model's window
  modelLoading: boolean;
  isExternalModel: boolean;
  // what /api/inference/status holds; undefined before the first read
  residentCheckpoint: string | null | undefined;
};

// the recount that fills the bar stands down for images, audio, Deep Research and a busy backend
export function hasKnownContextWindow({
  loadedContextLength,
  modelLoading,
  isExternalModel,
  residentCheckpoint,
}: KnownContextWindowInput): boolean {
  if (modelLoading || isExternalModel) return false;
  if (loadedContextLength == null || loadedContextLength <= 0) return false;
  // matches chatModelLoaded: undefined is "status not read yet", not "evicted"
  return residentCheckpoint !== null;
}
