// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getBrowserOfflineRetryDelayMs,
  getHubPhase,
  getLastHubFailure,
  isDirectHubOffline,
  shouldPreferLocalCache,
  isHubProxyServing,
  type HubFailure,
  type HubPhase,
  isHuggingFaceOffline,
  subscribeNetworkStatus,
} from "@/features/hub/lib/network";
import { useCallback, useSyncExternalStore } from "react";

function getOnlineSnapshot(): boolean {
  return !isHuggingFaceOffline();
}

function getServerOnlineSnapshot(): boolean {
  return true;
}

function subscribeOnlineStatus(onStoreChange: () => void): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const clearRetryTimer = () => {
    if (timer === null) {
      return;
    }
    clearTimeout(timer);
    timer = null;
  };
  const scheduleRetry = () => {
    clearRetryTimer();
    const retryDelay = getBrowserOfflineRetryDelayMs();
    if (retryDelay > 0) {
      timer = setTimeout(handleChange, retryDelay + 50);
    }
  };
  const handleChange = () => {
    onStoreChange();
    scheduleRetry();
  };
  const unsubscribe = subscribeNetworkStatus(handleChange);
  scheduleRetry();
  return () => {
    clearRetryTimer();
    unsubscribe();
  };
}

/**
 * Legacy boolean view. Prefer useHubAvailability() in UI: this collapses
 * "backing off" and "proven reachable" into one bit, so every failure looked
 * identical on screen.
 */
export function useOnlineStatus(): boolean {
  return useSyncExternalStore(
    subscribeOnlineStatus,
    getOnlineSnapshot,
    getServerOnlineSnapshot,
  );
}

function getDirectOnlineSnapshot(): boolean {
  return !isDirectHubOffline();
}

/**
 * For components that fetch repo assets from the Hub themselves: cards, owner
 * avatars, dataset sizes. A blocked catalog listing must not stop them, since a
 * block can be per-path, and their own failures are what suppress them, which
 * is also what lets their own success bring them back.
 */
export function useDirectHubOnline(origin?: string): boolean {
  // Pass the origin the caller actually fetches: a client gated on a window its
  // own failures never arm can neither back off nor recover.
  const getSnapshot = useCallback(
    () => (origin === undefined ? getDirectOnlineSnapshot() : !isDirectHubOffline(origin)),
    [origin],
  );
  return useSyncExternalStore(
    subscribeOnlineStatus,
    getSnapshot,
    getServerOnlineSnapshot,
  );
}

function getBackendOnlineSnapshot(): boolean {
  return !shouldPreferLocalCache();
}

/**
 * For actions the backend carries out: GGUF variant lookups and updates. What
 * decides those is the server's reach, not this browser's, and a proxy serving
 * discovery is proof it has one. Gating them on browser reachability hid the
 * update action exactly when the server could have applied it.
 */
export function useBackendHubOnline(): boolean {
  return useSyncExternalStore(
    subscribeOnlineStatus,
    getBackendOnlineSnapshot,
    getServerOnlineSnapshot,
  );
}

export interface HubAvailability {
  phase: HubPhase;
  failure: HubFailure | null;
  /** True when availability comes from the backend, not this browser. */
  proxyServing: boolean;
}

function getPhaseSnapshot(): HubPhase {
  return getHubPhase();
}

function getServerPhaseSnapshot(): HubPhase {
  return "available";
}

function getServerProxySnapshot(): boolean {
  return false;
}

function getFailureSnapshot(): HubFailure | null {
  return getLastHubFailure();
}

function getServerFailureSnapshot(): HubFailure | null {
  return null;
}

/**
 * Availability plus the reason for the last failure. The failure outlives the
 * backoff and clears only on success, so the UI keeps naming the real cause.
 */
export function useHubAvailability(): HubAvailability {
  const phase = useSyncExternalStore(
    subscribeOnlineStatus,
    getPhaseSnapshot,
    getServerPhaseSnapshot,
  );
  const failure = useSyncExternalStore(
    subscribeOnlineStatus,
    getFailureSnapshot,
    getServerFailureSnapshot,
  );
  const proxyServing = useSyncExternalStore(
    subscribeOnlineStatus,
    isHubProxyServing,
    getServerProxySnapshot,
  );
  return { phase, failure, proxyServing };
}
