// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type InventoryFetchCoalesceInput = {
  force?: boolean;
  ready: boolean;
  inFlight: boolean;
};

/** True when a new request should join the in-flight fetch instead of queueing another scan. */
export function shouldCoalesceInFlightInventoryFetch(
  input: InventoryFetchCoalesceInput,
): boolean {
  return input.inFlight && (!input.force || !input.ready);
}
