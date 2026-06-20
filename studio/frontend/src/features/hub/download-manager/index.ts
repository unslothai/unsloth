// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export * from "./api";
export { DownloadManagerPanel } from "./download-manager-panel";
export {
  DownloadProgressBar,
  type DownloadProgress,
} from "./download-progress-bar";
export {
  DEFAULT_TRANSPORT_MODE,
  DOWNLOAD_KIND,
  DOWNLOAD_KINDS,
  TRANSPORT,
  TRANSPORT_MODES,
  isDownloadKind,
  isTransportMode,
  type DownloadKind,
  type TransportMode,
} from "./constants";
export {
  __resetDownloadManagerForTests,
  cancelStagedModelDownload,
  clearCompletedInventoryHint,
  downloadManager,
  hydrateDownloadManager,
  jobKeyOf,
  repoKeyOf,
  selectActiveJob,
  subscribeJobListeners,
  useDownloadManagerStore,
  type DownloadManagerController,
  type DownloadRequest,
  type JobListeners,
  type ManagedDownload,
} from "./download-manager-controller";
export {
  useRepoDownload,
  type DownloadJob,
  type DownloadJobProgress,
  type RepoDownloadConfig,
} from "./use-repo-download";
export {
  getTransportMode,
  useDownloadTransportCapabilities,
  useTransportMode,
} from "./transport-preference";
export type { TransportConflictInfo } from "./types";
