// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * When to say the disk is filling up, and how often. A level is announced on CROSSING and
 * re-armed only once free space clears it by REARM_MARGIN_GB, so a disk on the line does not
 * toast every reading. Thresholds are absolute: a percentage would nag on a 4 TB disk with
 * 400 GB free.
 */

export const LOW_DISK_FREE_GB = 20;
export const CRITICAL_DISK_FREE_GB = 5;
export const REARM_MARGIN_GB = 5;

export type DiskPressure = "ok" | "low" | "critical";

export type DiskReading = {
  /** Decimal GB, as /api/system reports it. */
  free_gb: number | null | undefined;
  total_gb: number | null | undefined;
};

export type LowDiskState = {
  /** Highest level already announced, until free space re-arms it. */
  notified: DiskPressure;
};

export const INITIAL_LOW_DISK_STATE: LowDiskState = { notified: "ok" };

const RANK: Record<DiskPressure, number> = { ok: 0, low: 1, critical: 2 };

const BY_RANK: DiskPressure[] = ["ok", "low", "critical"];

const THRESHOLD_GB: Record<DiskPressure, number> = {
  ok: Number.POSITIVE_INFINITY,
  low: LOW_DISK_FREE_GB,
  critical: CRITICAL_DISK_FREE_GB,
};

function isReadable(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/** null when the host did not report a usable disk reading. */
export function diskPressure(disk: DiskReading): DiskPressure | null {
  if (!isReadable(disk.free_gb) || !isReadable(disk.total_gb)) return null;
  // A zero total is psutil having failed, not a full disk.
  if (disk.total_gb <= 0) return null;
  if (disk.free_gb <= CRITICAL_DISK_FREE_GB) return "critical";
  if (disk.free_gb <= LOW_DISK_FREE_GB) return "low";
  return "ok";
}

export type LowDiskDecision = {
  state: LowDiskState;
  notify: Exclude<DiskPressure, "ok"> | null;
};

/** Announce only an escalation; forget a level only past its re-arm margin. */
export function nextLowDiskNotice(
  state: LowDiskState,
  disk: DiskReading,
): LowDiskDecision {
  const pressure = diskPressure(disk);
  // An unreadable disk is not a recovery: leave the state exactly as it was.
  if (pressure === null) return { state, notify: null };
  if (pressure !== "ok" && RANK[pressure] > RANK[state.notified]) {
    return { state: { notified: pressure }, notify: pressure };
  }
  if (RANK[pressure] < RANK[state.notified]) {
    const forgotten = forgetRearmedLevels(
      state.notified,
      disk.free_gb as number,
    );
    if (forgotten !== state.notified) {
      return { state: { notified: forgotten }, notify: null };
    }
  }
  return { state, notify: null };
}

/** The highest level still armed at *free*. Instantaneous pressure would drop a level the disk
 * has not cleared: 21 GB clears critical's re-arm point but not low's, earning a second low
 * warning on the next dip. */
function forgetRearmedLevels(
  notified: DiskPressure,
  free: number,
): DiskPressure {
  let level = notified;
  while (level !== "ok" && free >= THRESHOLD_GB[level] + REARM_MARGIN_GB) {
    level = BY_RANK[RANK[level] - 1];
  }
  return level;
}

// Per browser session, not per mount: reopening Settings must not re-warn.
let sessionState: LowDiskState = INITIAL_LOW_DISK_STATE;

export function observeDiskPressure(
  disk: DiskReading,
): Exclude<DiskPressure, "ok"> | null {
  const decision = nextLowDiskNotice(sessionState, disk);
  sessionState = decision.state;
  return decision.notify;
}

export function resetLowDiskNotices(): void {
  sessionState = INITIAL_LOW_DISK_STATE;
}
