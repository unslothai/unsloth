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
  inventoryKind?: Exclude<InventoryHint["kind"], "dataset">;
  state: DownloadJobState;
  downloadedBytes: number;
  /** False when the last poll held `downloadedBytes` instead of measuring it, so it belongs to
   * the previous total. */
  measuredTransfer?: boolean;
  // Excludes `.incomplete` bytes so a partial cannot be marked complete.
  completedBytes: number;
  completeOnDisk: boolean;
  expectedBytes: number;
  /** Display scope when a plan transfers only one missing companion; counters stay plan-wide. */
  presentation?: DownloadPresentation;
  fraction: number;
  bytesPerSec: number;
  etaSeconds: number;
  error: string | null;
  startedAt: number;
  completedAt?: number;
  serverGeneration?: number;
  serverAttempt?: number;
  /** Files a scoped job fetches; separates this transfer from a sibling quant in the same slot. */
  scopedFiles?: string[];
  /** True for the picked model, false for companions; only the stager can tell them apart. */
  checkpoint?: boolean;
  transport?: ResolvedTransport;
  /** A Xet run that fell back to HTTP keeps its cancel marker, which decides the stop control. */
  cancelTransport?: ResolvedTransport;
  external?: boolean;
  activity?: string;
  details?: string[];
}

export interface DownloadRequest {
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  inventoryKind?: Exclude<InventoryHint["kind"], "dataset">;
  expectedBytes: number;
  presentation?: DownloadPresentation;
  scopeId?: string | null;
  files?: string[];
  checkpoint?: boolean;
  callerToast?: CallerToast;
  skipXetNotice?: boolean;
}

export interface DownloadPresentation {
  label: string;
  filename: string;
  expectedBytes: number;
  /** Frozen when attached, because later metadata may grow the plan. */
  cachedPlanPrefixBytes?: number;
}

export interface CallerToast {
  title: string;
  description: string;
  /** Fold into a granted Xet notice, never raise alone. */
  noticeOnly?: boolean;
  /** Re-checked before raising; false drops this line but keeps the notice. Absent = always valid. */
  stillValid?: () => boolean;
}

/** Mirrors the backend's `_scope_variant`: no GGUF quant label starts with "@". */
export function scopedVariant(scopeId: string): string {
  return `@${scopeId}`;
}

export function downloadInventoryHintKind(
  kind: DownloadKind,
  variant: string | null,
  inventoryKind?: Exclude<InventoryHint["kind"], "dataset">,
): InventoryHint["kind"] {
  if (kind === "dataset") return "dataset";
  if (inventoryKind) return inventoryKind;
  return variant && !variant.startsWith("@") ? "gguf" : "model";
}

export function scopedDownloadInventoryKind(
  files: readonly string[] | null | undefined,
): Exclude<InventoryHint["kind"], "dataset"> {
  return files?.some((file) => file.trim().toLowerCase().endsWith(".gguf"))
    ? "gguf"
    : "model";
}

export function downloadRequestInventoryKind(
  request: Pick<
    DownloadRequest,
    "kind" | "variant" | "inventoryKind" | "files"
  >,
): DownloadRequest["inventoryKind"] {
  if (request.inventoryKind) {
    return request.inventoryKind;
  }
  if (
    request.kind !== "model" ||
    !request.variant?.startsWith("@") ||
    !request.files?.length
  ) {
    return undefined;
  }
  return scopedDownloadInventoryKind(request.files);
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

export interface FloorHold {
  attempt: number;
  remainingBytes: number;
  until: number;
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
  speedSamples: TransferSample[];
  /** Generation change seen on a status-only tick, held until a progress poll consumes it. */
  pendingGenerationChange?: boolean;
  /** GGUF floor stays off until the killed run's partial is purged (see floorHoldEnded). */
  floorHold?: FloorHold | null;
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
  /** Null when no cache exists; absent (older backend) is treated as unknown. */
  cache_path?: string | null;
  /** Whether THIS target was found; null/absent leaves the repo-level cache_path rule in charge. */
  target_present?: boolean | null;
  /** False when the cache could not be scanned: unknown, not empty. */
  cache_measured?: boolean;
}

export type Terminal = "complete" | "cancelled" | "error" | "gone";
