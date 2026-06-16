// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InventoryHint } from "../inventory/types";
import type { DownloadJobState } from "./api";
import type { DownloadKind } from "./constants";
import type { TransportConflictInfo } from "./types";

export interface ManagedDownload {
  key: string;
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  state: DownloadJobState;
  downloadedBytes: number;
  // Finalized bytes on disk (excludes the in-progress `.incomplete` portion
  // that `downloadedBytes` includes for a smooth bar). The idle/registry-loss
  // completion fallback keys off this so a partial can't be marked complete.
  completedBytes: number;
  // Server-verified snapshot/variant completion. This is intentionally not
  // trusted across reloads; hydration re-probes before using it.
  completeOnDisk: boolean;
  expectedBytes: number;
  fraction: number;
  bytesPerSec: number;
  error: string | null;
  startedAt: number;
  serverGeneration?: number;
}

export interface DownloadRequest {
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  expectedBytes: number;
}

export interface JobListeners {
  onComplete?: (variant: string | null, bytes: number) => unknown;
  onCancelled?: (variant: string | null) => unknown;
  onError?: (variant: string | null) => unknown;
}

export interface ConflictEntry {
  info: TransportConflictInfo;
  pending: DownloadRequest;
}

export interface DownloadManagerState {
  jobs: Record<string, ManagedDownload>;
  conflicts: Record<string, ConflictEntry>;
  completedHintSignature: string;
  completedInventoryHints: InventoryHint[];
}

export interface JobRuntime {
  kind: DownloadKind;
  repoId: string;
  epoch: number;
  pollTimer: number | null;
  pollStartedAt: number;
  pollingStarted: boolean;
  abort: AbortController | null;
  inFlight: boolean;
  cancelRequested: boolean;
  watchdog: number | null;
  speedSample: { bytes: number; tMs: number } | null;
  idleSinceMs: number | null;
  lastProgressPollAt: number | null;
  pollFailureStartedAt: number | null;
  visibilityListener: (() => void) | null;
}

export interface ProgressLike {
  downloaded_bytes: number;
  completed_bytes?: number;
  complete_on_disk?: boolean;
  expected_bytes: number;
  progress: number;
}

export type Terminal = "complete" | "cancelled" | "error" | "gone";
