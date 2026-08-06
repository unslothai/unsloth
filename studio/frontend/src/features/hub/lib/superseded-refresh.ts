


// A caller that awaits a status refresh is waiting for the store to be settled, not for one
// particular HTTP response. Refreshes are sequenced so the newest read owns the store, and a
// response that lands after a newer read started writes nothing. Resolving that dropped
// response's promise anyway releases its caller onto the store the newer read has not written
// yet, which is exactly the pre-read store the await was there to replace: the Hub's settings
// handlers would then resolve a quant from the model an API switch already displaced and bake
// it into the settings target, and Apply reloads and persists under whatever the target names.
// A settings open is guarded against a newer settings open, but nothing bumps that guard for a
// focus or visibilitychange refresh, so those are the reads that can supersede one unnoticed.
//
// So a dropped response resolves with the refresh that superseded it instead of with nothing.

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
