// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { observeDiskPressure, type DiskPressure } from "./low-disk";

/**
 * Ask the host how much room is left, at the moments room matters: when a download is about to
 * request bytes, plus once at app mount. No interval; /api/system/disk is one syscall.
 */

/** Registered by the mounted hook: the wording needs i18n and the store, neither reachable here. */
type Notifier = (level: Exclude<DiskPressure, "ok">, disk: DiskReadingResponse) => void;

let notifier: Notifier | null = null;

export function setLowDiskNotifier(next: Notifier | null): void {
  notifier = next;
}

export interface DiskReadingResponse {
  path?: string | null;
  total_gb: number | null;
  free_gb: number | null;
  percent_used?: number | null;
}

/** Long enough to collapse a burst of queued downloads, short enough to catch one that filled the disk. */
const MIN_INTERVAL_MS = 30_000;

let lastCheckedAt = 0;
let inFlight: Promise<void> | null = null;
/** At most one reading waiting behind the current one. See `force` below. */
let queued: Promise<void> | null = null;

export function __resetLowDiskCheckForTests(): void {
  lastCheckedAt = 0;
  inFlight = null;
  queued = null;
  notifier = null;
}

async function readDisk(): Promise<DiskReadingResponse | null> {
  try {
    const response = await authFetch("/api/system/disk");
    if (!response.ok) return null;
    return (await response.json()) as DiskReadingResponse;
  } catch {
    // A disk reading is advice, never a gate: a host that cannot answer must not stop a
    // download or surface an error the user cannot act on.
    return null;
  }
}

function runCheck(): Promise<void> {
  lastCheckedAt = Date.now();
  inFlight = (async () => {
    const disk = await readDisk();
    if (!disk) return;
    // BEFORE observing: observeDiskPressure records the level it returns, so running it unheard
    // spends the crossing and a later owner login hears nothing until free space re-arms.
    if (!notifier) return;
    const level = observeDiskPressure(disk);
    if (level === null) return;
    notifier(level, disk);
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

/**
 * Read the disk and warn if a threshold was crossed. Never rejects, never blocks the caller.
 *
 * `force` needs a reading taken AFTER it asked, so it skips the interval AND declines to share
 * an in-flight request, which may be the pre-download reading it is correcting; it chains
 * behind instead. Bounded at one in flight plus one waiting.
 */
export function checkDiskSpace(options: { force?: boolean } = {}): Promise<void> {
  if (!options.force && Date.now() - lastCheckedAt < MIN_INTERVAL_MS) {
    return Promise.resolve();
  }
  if (inFlight) {
    // Unforced callers arrive together; any reading answers them all.
    if (!options.force) return inFlight;
    if (!queued) {
      queued = inFlight
        .then(() => {
          queued = null;
          return runCheck();
        })
        .catch(() => {
          queued = null;
        });
    }
    return queued;
  }
  return runCheck();
}
