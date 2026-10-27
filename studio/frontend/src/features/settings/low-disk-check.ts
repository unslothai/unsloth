// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports
import { disposableTimeoutSignal } from "@/features/hub/lib/abort-signals";
import { observeDiskPressure, type DiskPressure } from "./low-disk";

/** Read before a download requests bytes and once at mount; no polling. */

/** Registered by the mounted hook, which has i18n and the store. */
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

/** Collapses a burst of queued downloads but still catches one that filled the disk. */
const MIN_INTERVAL_MS = 30_000;

let lastCheckedAt = 0;
let inFlight: Promise<void> | null = null;
/** At most one reading waiting behind the current one. */
let queued: Promise<void> | null = null;

export function __resetLowDiskCheckForTests(): void {
  lastCheckedAt = 0;
  inFlight = null;
  queued = null;
  notifier = null;
}

const READ_TIMEOUT_MS = 10_000;

async function readDisk(): Promise<DiskReadingResponse | null> {
  // Bounded: `inFlight` is the only slot, and a hung read would silence the warning for the session.
  const timeout = disposableTimeoutSignal(READ_TIMEOUT_MS);
  try {
    const response = await authFetch("/api/system/disk", { signal: timeout.signal });
    if (!response.ok) return null;
    return (await response.json()) as DiskReadingResponse;
  } catch {
    // Advice, never a gate: a failed or timed-out read must not stop a download.
    return null;
  } finally {
    // Dispose once settled, or abort listeners pile up.
    timeout.dispose();
  }
}

function runCheck(): Promise<void> {
  lastCheckedAt = Date.now();
  inFlight = (async () => {
    const disk = await readDisk();
    if (!disk) return;
    // Before observing: observeDiskPressure records the level, so an unheard run spends the crossing.
    if (!notifier) return;
    const level = observeDiskPressure(disk);
    if (level === null) return;
    try {
      notifier(level, disk);
    } catch {
      // The crossing is already spent; swallow so a throwing notifier is not an unhandled rejection.
    }
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

/** Never rejects. `force` needs a reading taken after it asked, so it skips the interval and
 * chains behind any in-flight read; at most one in flight plus one waiting. */
export function checkDiskSpace(options: { force?: boolean } = {}): Promise<void> {
  if (!options.force && Date.now() - lastCheckedAt < MIN_INTERVAL_MS) {
    return Promise.resolve();
  }
  if (inFlight) {
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
