// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelLoadRunToken = {
  attemptId: number;
};

/**
 * Async load callbacks may finish after a replacement run has started. Only
 * the run that still owns the slot may update or clear shared loading state.
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
