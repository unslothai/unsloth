// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RefreshSupersession {
  latest: { seq: number; settled: Promise<void> } | null;
}

/** Call in start order, synchronously with the sequence number, so `latest` is the newest read. */
export function registerRefresh(
  supersession: RefreshSupersession,
  seq: number,
  settled: Promise<void>,
): void {
  supersession.latest = { seq, settled };
}

/**
 * The refresh that superseded `seq`, or `undefined` when nothing newer is registered, so an
 * unmount bump ends the chain instead of a promise waiting on itself forever.
 */
export function supersedingRefresh(
  supersession: RefreshSupersession,
  seq: number,
): Promise<void> | undefined {
  const { latest } = supersession;
  return latest && latest.seq > seq ? latest.settled : undefined;
}
