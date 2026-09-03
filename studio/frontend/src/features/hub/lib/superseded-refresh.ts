// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A caller waiting for resident state must wait for the newest refresh, not only
// for an older response that was dropped after a focus or visibility refresh.

/** The newest refresh started, so a superseded one can hand its caller that promise. */
export interface RefreshSupersession {
  latest: { seq: number; settled: Promise<void> } | null;
}

/**
 * Record `settled` as what callers of refresh `seq` are holding. Call it in start order,
 * synchronously with the sequence number, so `latest` always names the newest read.
 */
export function registerRefresh(
  supersession: RefreshSupersession,
  seq: number,
  settled: Promise<void>,
): void {
  supersession.latest = { seq, settled };
}

/**
 * What the dropped response of refresh `seq` should resolve with: the refresh that superseded
 * it, so its caller still comes back to a settled store. `undefined` once nothing newer is
 * registered, which is how a sequence bump that starts no read (unmount) ends the chain rather
 * than handing a promise back to itself, where it would wait forever.
 */
export function supersedingRefresh(
  supersession: RefreshSupersession,
  seq: number,
): Promise<void> | undefined {
  const { latest } = supersession;
  return latest && latest.seq > seq ? latest.settled : undefined;
}
