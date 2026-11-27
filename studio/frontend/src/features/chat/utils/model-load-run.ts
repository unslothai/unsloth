// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelLoadRunToken = {
  attemptId: number;
};

/**
 * Async loads may finish after a replacement starts; only the run owning the slot may update state.
 */
export function ownsModelLoadRun(
  active: ModelLoadRunToken | null,
  candidate: ModelLoadRunToken,
): boolean {
  return active?.attemptId === candidate.attemptId;
}

export function releaseOwnedModelLoadRun<T extends ModelLoadRunToken>(
  active: T | null,
  candidate: ModelLoadRunToken,
): T | null {
  return ownsModelLoadRun(active, candidate) ? null : active;
}
