// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What a settled status does to `loadedLlamaExtraArgs`: `{}` leaves it alone. */
export function resolveLlamaExtraArgsSeed({
  incoming,
  isGguf,
  hydratingExistingModel,
  seedLoadParams,
}: {
  incoming: string[] | null | undefined;
  isGguf: boolean;
  hydratingExistingModel: boolean;
  seedLoadParams: boolean;
}): { loadedLlamaExtraArgs?: string[] | null } {
  if (!seedLoadParams) {
    return {};
  }
  if (incoming !== undefined) {
    return isGguf ? { loadedLlamaExtraArgs: incoming ?? null } : {};
  }
  // A backend that omits the field: keep this tab's first-hand record for the same model, but
  // never hand the previous model's arguments to a new one.
  return hydratingExistingModel ? { loadedLlamaExtraArgs: null } : {};
}
