// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain module so the node test suite can import it without the chat runtime.

export type KnownContextWindowInput = {
  loadedContextLength: number | null;
  // A load in flight still carries the outgoing model's window.
  modelLoading: boolean;
  isExternalModel: boolean;
  residentCheckpoint: string | null | undefined;
};

export function hasKnownContextWindow({
  loadedContextLength,
  modelLoading,
  isExternalModel,
  residentCheckpoint,
}: KnownContextWindowInput): boolean {
  if (modelLoading || isExternalModel) return false;
  if (loadedContextLength == null || loadedContextLength <= 0) return false;
  // undefined means status not read yet, not evicted.
  return residentCheckpoint !== null;
}
