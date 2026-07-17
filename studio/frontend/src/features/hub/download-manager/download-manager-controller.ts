// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DOWNLOAD_KIND } from "./constants";
import {
  createDownloadManagerInitialState,
  jobKeyOf,
  removeJob,
  selectActiveJob,
  setState,
  useDownloadManagerStore,
} from "./download-manager-state";
import { resetDownloadApiAdapterState } from "./download-api-adapter";
import {
  hydrateDownloadManager,
  resetHydrationState,
} from "./hydration";
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

export function __resetDownloadManagerForTests(): void {
  runtimeRegistry.reset();
  resetDownloadApiAdapterState();
  resetHydrationState();
  setState(createDownloadManagerInitialState());
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

/** Cancel the in-flight download for a staged model pick. No-op when nothing is
 *  downloading (e.g. a native/local file that was never fetched). Lets non-React
 *  callers (the chat store's abandon paths) stop a staged transfer without the
 *  useRepoDownload hook. */
export function cancelStagedModelDownload(
  pending: { id: string; ggufVariant?: string | null } | null,
): void {
  if (!pending) return;
  const variant = pending.ggufVariant ?? null;
  const activeJob = selectActiveJob(
    useDownloadManagerStore.getState(),
    DOWNLOAD_KIND.MODEL,
    pending.id,
    variant,
  );
  void downloadManager.cancel(
    activeJob?.key ?? jobKeyOf(DOWNLOAD_KIND.MODEL, pending.id, variant),
  );
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    __resetDownloadManagerForTests();
  });
}
