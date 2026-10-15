// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getBrowserOfflineRetryDelayMs,
  getHubPhase,
  getLastHubFailure,
  type HubFailure,
  type HubPhase,
  isHuggingFaceOffline,
  subscribeNetworkStatus,
} from "@/features/hub/lib/network";
import { useSyncExternalStore } from "react";

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

/** Legacy boolean view; prefer useHubAvailability(), which distinguishes backoff from reachable. */
export function useOnlineStatus(): boolean {
  return useSyncExternalStore(
    subscribeOnlineStatus,
    getOnlineSnapshot,
    getServerOnlineSnapshot,
  );
}

export interface HubAvailability {
  phase: HubPhase;
  failure: HubFailure | null;
}

function getPhaseSnapshot(): HubPhase {
  return getHubPhase();
}

function getServerPhaseSnapshot(): HubPhase {
  return "available";
}

function getFailureSnapshot(): HubFailure | null {
  return getLastHubFailure();
}

function getServerFailureSnapshot(): HubFailure | null {
  return null;
}

/** The failure outlives the backoff and clears only on success. */
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
  return { phase, failure };
}
