// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getBrowserOfflineRetryDelayMs,
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

export function useOnlineStatus(): boolean {
  return useSyncExternalStore(
    subscribeOnlineStatus,
    getOnlineSnapshot,
    getServerOnlineSnapshot,
  );
}
