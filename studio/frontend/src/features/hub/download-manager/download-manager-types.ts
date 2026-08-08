// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransferSample } from "@/lib/transfer-stats";
import type { InventoryHint } from "../inventory/types";
import type { DownloadJobState } from "./api";
import type { DownloadKind, ResolvedTransport } from "./constants";
import type { TransportConflictInfo } from "./types";

export interface ManagedDownload {
  key: string;
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  state: DownloadJobState;
  downloadedBytes: number;
  // Finalized bytes on disk (excludes the in-progress `.incomplete` portion in
  // `downloadedBytes`). The completion fallback keys off this so a partial can't
  // be marked complete.
  completedBytes: number;
  // Server-verified completion; not trusted across reloads (hydration re-probes).
  completeOnDisk: boolean;
  expectedBytes: number;
  fraction: number;
  bytesPerSec: number;
  error: string | null;
  startedAt: number;
  serverGeneration?: number;
  /** Files a scoped job is fetching, when known. Every file set of one repo rides the same scope slot (see `scopedVariant`), so this separates "my transfer is already running" from "a different quant of this repo is running": adopting the latter would report ready for files nobody fetched. Unknown stays adoptable only for an UNSCOPED job. */
  scopedFiles?: string[];
  /** Set by the page that staged a multi-repo plan: true for the entry that IS the model the user picked, false for the companion repos it needs. Only the stager can tell them apart, since a checkpoint may be a single `.safetensors` and companions carry `.safetensors` too. Undefined on a job persisted before this field existed. */
  checkpoint?: boolean;
  // The transport this run resolved to, so a surface can say whether stopping
  // keeps a resumable partial. Absent only when neither the backend nor
  // persisted state identifies an adopted job's transport.
  transport?: ResolvedTransport;
  /** A Xet run that fell back to HTTP keeps its original cancel marker, so
   * stopping it still leaves a restart-only partial. Set only in that case,
   * and it, not `transport`, decides the stop control. */
  cancelTransport?: ResolvedTransport;
  // Driven by another subsystem (see external-jobs.ts), not the poll loop. Such
  // a job is never persisted, probed against the hub API, or published as a
  // chat-inventory hint when it completes.
  external?: boolean;
}

export interface DownloadRequest {
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  expectedBytes: number;
  /** Marks a partial-by-design download of `files` only, for a consumer that reads a deliberate subset of a repo (the diffusion loader skips the packaged root single, transformer/ shards and fp16 twins). Set `variant` to `scopedVariant(scopeId)` so this surface keys the job the way the backend does. */
  scopeId?: string | null;
  files?: string[];
  /** See `ManagedDownload.checkpoint`. Carried on the request so the job records the role its stager already knew, rather than a surface re-deriving it from filenames. */
  checkpoint?: boolean;
}

/** The variant slot a scoped job occupies. Mirrors the backend's `_scope_variant`: no GGUF quant label starts with "@", so a scope collides with neither a real variant nor the repo's full snapshot. */
export function scopedVariant(scopeId: string): string {
  return `@${scopeId}`;
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
  /** Rolling byte samples behind the stability-gated rate/ETA. */
  speedSamples: TransferSample[];
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
