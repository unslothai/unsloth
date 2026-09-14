// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A settings route this build of the UI knows about and the running backend does not.
 *
 * Distinct from a failed read on purpose. A caller deciding whether it may skip work
 * treats the two oppositely: an absent route means the backend has no such state to
 * disagree about, while a read that could not be made says nothing at all, and assuming
 * it said "no" is how a saved setting goes missing.
 */
export class SettingsRouteAbsentError extends Error {
  constructor(route: string) {
    super(`Settings route not served by this backend: ${route}`);
    this.name = "SettingsRouteAbsentError";
  }
}

export function isSettingsRouteAbsent(error: unknown): boolean {
  return error instanceof SettingsRouteAbsentError;
}
