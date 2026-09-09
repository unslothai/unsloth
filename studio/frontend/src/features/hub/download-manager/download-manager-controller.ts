// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_SESSION_STORED_EVENT,
  AUTH_TOKEN_KEY,
} from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { resetDownloadApiAdapterState } from "./download-api-adapter";
import {
  createDownloadManagerInitialState,
  removeJob,
  setState,
  useDownloadManagerStore,
} from "./download-manager-state";
import { hydrateDownloadManager, resetHydrationState } from "./hydration";
import { cancelJob, probeAndAdopt, setExpected } from "./poll-loop";
import { runtimeRegistry } from "./runtime-registry";
import {
  cancelConflict,
  requestStart,
  restartConflict,
  resumeConflict,
} from "./transport-conflict";

export type { DownloadKind } from "./constants";
export {
  clearCompletedInventoryHint,
  jobKeyOf,
  repoKeyOf,
  selectActiveJob,
  subscribeJobListeners,
  useDownloadManagerStore,
} from "./download-manager-state";
export type {
  DownloadRequest,
  JobListeners,
  ManagedDownload,
} from "./download-manager-types";
export { hydrateDownloadManager };

function resetDownloadManagerState(): void {
  runtimeRegistry.reset();
  resetDownloadApiAdapterState();
  resetHydrationState();
  setState(createDownloadManagerInitialState());
}

export function __resetDownloadManagerForTests(): void {
  resetDownloadManagerState();
}

function clearWebSessionDownloads(): void {
  if (isTauri) {
    return;
  }
  resetDownloadManagerState();
  // Clearing after the state reset also cancels the throttled empty-state write.
  void useDownloadManagerStore.persist.clearStorage();
}

function handleAuthStorageChange(event: StorageEvent): void {
  if (
    event.key === AUTH_SESSION_MARK_KEY ||
    (event.key === AUTH_TOKEN_KEY && event.newValue === null)
  ) {
    clearWebSessionDownloads();
  }
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, clearWebSessionDownloads);
  window.addEventListener(AUTH_SESSION_STORED_EVENT, clearWebSessionDownloads);
  window.addEventListener("storage", handleAuthStorageChange);
}

export interface DownloadManagerController {
  requestStart: typeof requestStart;
  cancel: typeof cancelJob;
  probeAndAdopt: typeof probeAndAdopt;
  setExpected: typeof setExpected;
  resumeConflict: typeof resumeConflict;
  restartConflict: typeof restartConflict;
  cancelConflict: typeof cancelConflict;
  dismiss: typeof removeJob;
}

export const downloadManager: DownloadManagerController = {
  requestStart,
  cancel: cancelJob,
  probeAndAdopt,
  setExpected,
  resumeConflict,
  restartConflict,
  cancelConflict,
  dismiss: removeJob,
};

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    window.removeEventListener(
      AUTH_SESSION_CLEARED_EVENT,
      clearWebSessionDownloads,
    );
    window.removeEventListener(
      AUTH_SESSION_STORED_EVENT,
      clearWebSessionDownloads,
    );
    window.removeEventListener("storage", handleAuthStorageChange);
    __resetDownloadManagerForTests();
  });
}
