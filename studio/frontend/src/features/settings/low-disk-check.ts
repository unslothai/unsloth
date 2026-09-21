// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { observeDiskPressure, type DiskPressure } from "./low-disk";

/**
 * Ask the host how much room is left, at the moments that room matters.
 *
 * This replaced a 60 s interval on /api/system. That route enumerates GPUs, reads package
 * metadata and samples CPU, so running it forever in every open tab on the chance that a disk
 * is filling was the wrong trade twice over: too expensive for what it asks, and still no help
 * to the person who has not opened Studio. /api/system/disk is one syscall.
 *
 * A disk fills because something writes to it, and in Studio that something is nearly always a
 * download. So the check runs where the bytes are about to be requested, plus once when the app
 * mounts to catch a disk that was already full before the user did anything.
 */

/** How the reading is turned into a message. Registered by the mounted hook, because the
 * wording needs i18n and the action needs the settings dialog store, and neither is reachable
 * from a plain module function. Null until the app shell mounts. */
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

/** Two downloads queued back to back are one disk, so the second re-read tells nobody
 * anything. Long enough to collapse a burst, short enough that a download that filled the disk
 * is noticed before the next one starts. */
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
    const level = observeDiskPressure(disk);
    if (level === null) return;
    notifier?.(level, disk);
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

/**
 * Read the disk and warn if a threshold was crossed. Never rejects, never blocks the caller.
 *
 * `force` means the caller needs a reading taken AFTER it asked: the app mount, which is the
 * first of the session and has nothing to collapse against, and a finished download, which is
 * asking precisely because the number from before it started writing is now wrong. So force
 * skips the interval AND declines to share an in-flight request, since that request may well be
 * the pre-download reading it is trying to correct. It chains behind it instead.
 *
 * Still bounded: one in flight and at most one waiting, so a queue of files finishing together
 * costs two readings rather than one per file.
 */
export function checkDiskSpace(options: { force?: boolean } = {}): Promise<void> {
  if (!options.force && Date.now() - lastCheckedAt < MIN_INTERVAL_MS) {
    return Promise.resolve();
  }
  if (inFlight) {
    // Unforced callers arrive together when a page starts several downloads at once, and any
    // reading answers them.
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
