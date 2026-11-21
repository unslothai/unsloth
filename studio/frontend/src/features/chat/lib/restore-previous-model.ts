// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * True when the load got no answer (transport failure): the backend may still be loading,
 * so a rollback would race it. HTTP or body errors are answers and keep the rollback.
 */
export function loadOutcomeUnknown(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { unslothTransportFailure?: unknown }).unslothTransportFailure ===
      true
  );
}

export function shouldRestorePreviousModel(error: unknown): boolean {
  return !loadOutcomeUnknown(error);
}
