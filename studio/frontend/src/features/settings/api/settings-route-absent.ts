// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A route the UI knows and the backend does not. Distinct from a failed read: treating
 * "could not ask" as "no" is how a saved setting goes missing.
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
