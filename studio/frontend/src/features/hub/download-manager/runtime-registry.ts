// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { JobListeners, JobRuntime } from "./download-manager-types";
import { evictOldestUnprotected } from "../lib/lru-map";

const MAX_SUPPRESSED_COMPLETED_INVENTORY_HINTS = 64;

class DownloadManagerRuntimeRegistry {
  readonly runtimes = new Map<string, JobRuntime>();
  readonly listeners = new Map<string, Set<JobListeners>>();
  readonly removalTimers = new Map<string, number>();
  readonly hydrationRetryTimers = new Set<number>();
  readonly suppressedCompletedInventoryHints = new Set<string>();
  readonly pendingStartRepoKeys = new Set<string>();
  inventoryBumpTimer: number | null = null;

  clearRemovalTimer(key: string): void {
    const timer = this.removalTimers.get(key);
    if (timer === undefined) return;
    clearRuntimeTimer(timer);
    this.removalTimers.delete(key);
  }

  reset(): void {
    for (const key of Array.from(this.runtimes.keys())) {
      teardownRuntime(key);
    }
    for (const timer of this.removalTimers.values()) {
      clearRuntimeTimer(timer);
    }
    for (const timer of this.hydrationRetryTimers) {
      clearRuntimeTimer(timer);
    }
    clearRuntimeTimer(this.inventoryBumpTimer);
    this.runtimes.clear();
    this.listeners.clear();
    this.removalTimers.clear();
    this.hydrationRetryTimers.clear();
    this.suppressedCompletedInventoryHints.clear();
    this.pendingStartRepoKeys.clear();
    this.inventoryBumpTimer = null;
  }
}

export const runtimeRegistry = new DownloadManagerRuntimeRegistry();

export function clearRuntimeTimer(timer: number | null): void {
  if (timer === null) return;
  if (typeof window !== "undefined") {
    window.clearTimeout(timer);
  } else {
    globalThis.clearTimeout(timer);
  }
}

export function clearWatchdog(rt: JobRuntime | undefined): void {
  if (rt?.watchdog == null) return;
  clearRuntimeTimer(rt.watchdog);
  rt.watchdog = null;
}

export function teardownRuntime(key: string): void {
  const rt = runtimeRegistry.runtimes.get(key);
  if (!rt) return;
  clearRuntimeTimer(rt.pollTimer);
  if (rt.visibilityListener != null && typeof document !== "undefined") {
    document.removeEventListener("visibilitychange", rt.visibilityListener);
  }
  rt.abort?.abort();
  clearWatchdog(rt);
  runtimeRegistry.runtimes.delete(key);
}

export function pruneSuppressedCompletedInventoryHints(
  liveCompletedKeys: Set<string>,
): void {
  evictOldestUnprotected(
    runtimeRegistry.suppressedCompletedInventoryHints,
    MAX_SUPPRESSED_COMPLETED_INVENTORY_HINTS,
    (key) => liveCompletedKeys.has(key),
  );
}
