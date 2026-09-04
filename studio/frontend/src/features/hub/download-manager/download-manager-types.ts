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
  /** False when the last poll HELD `downloadedBytes` instead of measuring it (see `resolveProgressUpdate`): a held
   * figure belongs to the previous reading's total, so subtracting it from the current `expectedBytes` mixes two plans. */
  measuredTransfer?: boolean;
  // Finalized bytes on disk, excluding the in-progress `.incomplete` portion; the completion fallback keys off this so a partial cannot be marked complete.
  completedBytes: number;
  completeOnDisk: boolean;
  expectedBytes: number;
  /** Optional display scope when an atomic model plan is transferring only one
   *  missing companion. Counters remain plan-wide; the panel subtracts the
   *  already-cached prefix for an honest artifact-sized progress bar. */
  presentation?: DownloadPresentation;
  fraction: number;
  bytesPerSec: number;
  etaSeconds: number;
  error: string | null;
  startedAt: number;
  completedAt?: number;
  serverGeneration?: number;
  /** Files a scoped job is fetching. Every file set of one repo rides the same scope slot, so this separates "my transfer is running" from "a different quant of this repo is running": adopting the latter reports ready for files nobody fetched. Unknown stays adoptable only for an UNSCOPED job. */
  scopedFiles?: string[];
  /** True for the entry that IS the model the user picked, false for companion repos. Only the stager can tell them apart, since a checkpoint may be a single `.safetensors` and companions carry `.safetensors` too. */
  checkpoint?: boolean;
  transport?: ResolvedTransport;
  /** A Xet run that fell back to HTTP keeps its original cancel marker, so it, not `transport`, decides the stop control. */
  cancelTransport?: ResolvedTransport;
  external?: boolean;
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
}

export interface DownloadPresentation {
  label: string;
  filename: string;
  expectedBytes: number;
}

export interface CallerToast {
  title: string;
  description: string;
  /** Fold into a granted Xet notice, never raise alone: #9663 removed chat's auto-load toast as a duplicate of the download panel. */
  noticeOnly?: boolean;
  /** Re-asked just before the raise: false drops this line but keeps the notice, since chat may have moved on while the transfer runs. Absent means always valid. */
  stillValid?: () => boolean;
}

/** Mirrors the backend's `_scope_variant`: no GGUF quant label starts with "@", so a scope collides with neither a real variant nor the repo's full snapshot. */
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

export type ConflictOwner = "caller" | "downloads";

export interface ConflictEntry {
  owner: ConflictOwner;
  info: TransportConflictInfo;
  pending: DownloadRequest;
}

export function conflictInfoForOwner(
  entry: ConflictEntry | undefined,
  owner: ConflictOwner,
): TransportConflictInfo | null {
  return entry?.owner === owner ? entry.info : null;
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
  speedSamples: TransferSample[];
  /**
   * A generation change seen on a status-only tick, held until a progress poll consumes it: status polls twice as often. */
  pendingGenerationChange?: boolean;
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
  /** The scanned cache dir, or null when no cache for this repo exists. Absent from an older backend, which hydration treats as unknown. */
  cache_path?: string | null;
  /** Whether the backend found anything for THIS target rather than the shared repo cache dir. Null or absent leaves the repo-level cache_path rule in charge. */
  target_present?: boolean | null;
  /** False when the cache could not be scanned at all: unknown, not empty. Absent from an older backend is also unknown. */
  cache_measured?: boolean;
}

export type Terminal = "complete" | "cancelled" | "error" | "gone";
