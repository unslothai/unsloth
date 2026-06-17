// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { INVENTORY_HINT_KIND } from "../inventory/constants";
import { inventoryHintKey } from "../inventory/inventory-hints";
import type { InventoryHint } from "../inventory/types";
import { createThrottledStorage, noopStorage } from "../stores/persist-storage";
import type { DownloadJobState } from "./api";
import {
  DOWNLOAD_KIND,
  type DownloadKind,
  isDownloadKind,
} from "./constants";
import {
  ACTIVE_STATES,
  MAX_PROGRESS_FRACTION,
} from "./download-manager-config";
import {
  type DownloadManagerState,
  type JobListeners,
  type ManagedDownload,
} from "./download-manager-types";
import {
  clearRuntimeTimer,
  pruneSuppressedCompletedInventoryHints as pruneRuntimeSuppressedHints,
  runtimeRegistry,
  teardownRuntime,
} from "./runtime-registry";

const PERSIST_KEY = "unsloth.studio.downloads";
const PERSIST_VERSION = 1;
const PERSIST_THROTTLE_MS = 1_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nonNegativeNumber(value: unknown, fallback = 0): number {
  return Math.max(0, finiteNumber(value, fallback));
}

function sanitizePersistedJob(value: unknown): ManagedDownload | null {
  if (!isRecord(value)) return null;
  const kind = isDownloadKind(value.kind) ? value.kind : null;
  const repoId = typeof value.repoId === "string" ? value.repoId : null;
  const state = value.state as DownloadJobState;
  if (!kind || !repoId || !ACTIVE_STATES.has(state)) return null;

  const variant = typeof value.variant === "string" ? value.variant : null;
  const key = jobKeyOf(kind, repoId, variant);
  return {
    key,
    kind,
    repoId,
    variant,
    state,
    downloadedBytes: nonNegativeNumber(value.downloadedBytes),
    completedBytes: nonNegativeNumber(value.completedBytes),
    completeOnDisk: false,
    expectedBytes: nonNegativeNumber(value.expectedBytes),
    fraction: Math.min(Math.max(finiteNumber(value.fraction, 0), 0), 1),
    bytesPerSec: 0,
    error: typeof value.error === "string" ? value.error : null,
    startedAt: nonNegativeNumber(value.startedAt, Date.now()),
    ...(Number.isSafeInteger(value.serverGeneration)
      ? { serverGeneration: Number(value.serverGeneration) }
      : {}),
  };
}

function sanitizePersistedState(
  persisted: unknown,
): Partial<DownloadManagerState> {
  if (!isRecord(persisted) || !isRecord(persisted.jobs)) {
    return { jobs: {}, conflicts: {} };
  }
  const jobs: Record<string, ManagedDownload> = {};
  for (const value of Object.values(persisted.jobs)) {
    const job = sanitizePersistedJob(value);
    if (job) jobs[job.key] = job;
  }
  return {
    jobs,
    conflicts: {},
  };
}

function toPersistedJob(
  job: ManagedDownload,
): Omit<ManagedDownload, "bytesPerSec" | "completeOnDisk"> {
  return {
    key: job.key,
    kind: job.kind,
    repoId: job.repoId,
    variant: job.variant,
    state: job.state,
    downloadedBytes: job.downloadedBytes,
    completedBytes: job.completedBytes,
    expectedBytes: job.expectedBytes,
    fraction: job.fraction,
    error: job.error,
    startedAt: job.startedAt,
    ...(job.serverGeneration !== undefined
      ? { serverGeneration: job.serverGeneration }
      : {}),
  };
}

// Mirrors the backend's normalize_repo_key (strip().lower()) so two casings of
// one repo share a key (else duplicate jobs / mismatched listeners). Keys only;
// `repoId` keeps original casing for display and API calls.
function normalizeRepoIdentity(repoId: string): string {
  return repoId.trim().toLowerCase();
}

function normalizeVariantIdentity(variant: string | null | undefined): string {
  return variant?.trim().toLowerCase() ?? "";
}

export function repoKeyOf(kind: DownloadKind, repoId: string): string {
  return `${kind}:${normalizeRepoIdentity(repoId)}`;
}

export function jobKeyOf(
  kind: DownloadKind,
  repoId: string,
  variant: string | null,
): string {
  const base = repoKeyOf(kind, repoId);
  const variantKey = normalizeVariantIdentity(variant);
  return variantKey ? `${base}#${variantKey}` : base;
}

function completedInventoryHintKind(
  kind: DownloadKind,
  variant: string | null,
): InventoryHint["kind"] {
  return kind === DOWNLOAD_KIND.DATASET
    ? INVENTORY_HINT_KIND.DATASET
    : variant
      ? INVENTORY_HINT_KIND.GGUF
      : INVENTORY_HINT_KIND.MODEL;
}

function liveCompletedInventoryHintKeys(): Set<string> {
  const keys = new Set<string>();
  for (const job of Object.values(getState().jobs)) {
    if (job.state !== "complete") continue;
    keys.add(
      inventoryHintKey(
        completedInventoryHintKind(job.kind, job.variant),
        job.repoId,
      ),
    );
  }
  return keys;
}

function collectCompletedInventoryHints(
  jobs: Record<string, ManagedDownload>,
): InventoryHint[] {
  return Object.values(jobs).flatMap((job) => {
    if (job.state !== "complete") return [];
    const kind = completedInventoryHintKind(job.kind, job.variant);
    if (
      runtimeRegistry.suppressedCompletedInventoryHints.has(
        inventoryHintKey(kind, job.repoId),
      )
    ) {
      return [];
    }
    const bytes = Math.max(job.downloadedBytes, job.expectedBytes, 0);
    return [
      {
        kind,
        repoId: job.repoId,
        ...(bytes > 0 ? { bytes } : {}),
      },
    ];
  });
}

function buildCompletedHintSignature(hints: readonly InventoryHint[]): string {
  return hints
    .map((hint) => `${hint.kind}:${hint.repoId}:${hint.bytes ?? ""}`)
    .sort()
    .join("|");
}

export function createDownloadManagerInitialState(): DownloadManagerState {
  return {
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  };
}

export const useDownloadManagerStore = create<DownloadManagerState>()(
  persist(createDownloadManagerInitialState, {
    name: PERSIST_KEY,
    version: PERSIST_VERSION,
    storage: createJSONStorage(() =>
      typeof window === "undefined"
        ? noopStorage
        : createThrottledStorage(window.localStorage, PERSIST_THROTTLE_MS),
    ),
    migrate: (persisted) => sanitizePersistedState(persisted),
    merge: (persisted, current) => ({
      ...current,
      ...sanitizePersistedState(persisted),
    }),
    partialize: (state) => ({
      jobs: Object.fromEntries(
        Object.entries(state.jobs)
          .filter(([, job]) => ACTIVE_STATES.has(job.state))
          .map(([key, job]) => [key, toPersistedJob(job)] as const),
      ),
      conflicts: {},
    }),
  }),
);

export const setState = useDownloadManagerStore.setState;
export const getState = useDownloadManagerStore.getState;

function withCompletedHintSignature(
  state: DownloadManagerState,
): DownloadManagerState {
  const completedInventoryHints = collectCompletedInventoryHints(state.jobs);
  const completedHintSignature = buildCompletedHintSignature(
    completedInventoryHints,
  );
  if (state.completedHintSignature === completedHintSignature) return state;
  return { ...state, completedHintSignature, completedInventoryHints };
}

function isPreferredRepoActiveJob(
  candidate: ManagedDownload,
  current: ManagedDownload | null | undefined,
): boolean {
  if (!current || !ACTIVE_STATES.has(current.state)) return true;
  if (candidate.startedAt !== current.startedAt) {
    return candidate.startedAt > current.startedAt;
  }
  return candidate.key < current.key;
}

export function findActiveJobForRepo(
  jobs: Record<string, ManagedDownload>,
  kind: DownloadKind,
  repoId: string,
): ManagedDownload | null {
  let selected: ManagedDownload | null = null;
  const repoIdentity = normalizeRepoIdentity(repoId);
  for (const job of Object.values(jobs)) {
    if (job.kind !== kind || normalizeRepoIdentity(job.repoId) !== repoIdentity)
      continue;
    if (!ACTIVE_STATES.has(job.state)) continue;
    if (isPreferredRepoActiveJob(job, selected)) {
      selected = job;
    }
  }
  return selected;
}

function hasRuntimePeerForRepo(
  kind: DownloadKind,
  repoId: string,
  key: string,
): boolean {
  const repoIdentity = normalizeRepoIdentity(repoId);
  for (const [runtimeKey, runtime] of runtimeRegistry.runtimes) {
    if (runtimeKey === key) continue;
    if (runtime.kind === kind && normalizeRepoIdentity(runtime.repoId) === repoIdentity)
      return true;
  }
  return false;
}

// Shared rule for what blocks a fresh GGUF variant start. Peer guard passes
// includeOwnRuntime:false (runs after this start made its own runtime);
// requestStart passes both true (runs before any runtime or job exists).
export function hasVariantRepoActivity(
  kind: DownloadKind,
  repoId: string,
  key: string,
  opts: { includeOwnRuntime: boolean; includePending: boolean },
): boolean {
  const state = getState();
  const own = state.jobs[key];
  if (own && ACTIVE_STATES.has(own.state)) return true;
  if (selectActiveJob(state, kind, repoId, null)) return true;
  const snapshotKey = jobKeyOf(kind, repoId, null);
  if (runtimeRegistry.runtimes.has(snapshotKey)) return true;
  if (opts.includeOwnRuntime && runtimeRegistry.runtimes.has(key)) return true;
  if (opts.includePending) {
    if (runtimeRegistry.pendingStartRepoKeys.has(key)) return true;
    if (runtimeRegistry.pendingStartRepoKeys.has(snapshotKey)) return true;
  }
  return false;
}

export function hasActiveRepoPeer(
  kind: DownloadKind,
  repoId: string,
  key: string,
  variant?: string | null,
): boolean {
  if (kind === DOWNLOAD_KIND.MODEL && variant) {
    return hasVariantRepoActivity(kind, repoId, key, {
      includeOwnRuntime: false,
      includePending: false,
    });
  }
  const active = findActiveJobForRepo(getState().jobs, kind, repoId);
  if (active && active.key !== key) return true;
  return hasRuntimePeerForRepo(kind, repoId, key);
}

function refreshCompletedHintSignature(): void {
  setState((state) => withCompletedHintSignature(state));
}

export function patchJob(key: string, patch: Partial<ManagedDownload>): void {
  setState((state) => {
    const job = state.jobs[key];
    if (!job) return state;
    const nextJob = { ...job, ...patch };
    const nextState = {
      ...state,
      jobs: { ...state.jobs, [key]: nextJob },
    };
    if (job.state !== "complete" && nextJob.state !== "complete") {
      return nextState;
    }
    return withCompletedHintSignature(nextState);
  });
}

export function putJob(job: ManagedDownload): void {
  runtimeRegistry.clearRemovalTimer(job.key);
  const suppressionChanged =
    runtimeRegistry.suppressedCompletedInventoryHints.delete(
      inventoryHintKey(
        completedInventoryHintKind(job.kind, job.variant),
        job.repoId,
      ),
    );
  setState((state) => {
    const existing = state.jobs[job.key];
    const nextState = {
      ...state,
      jobs: { ...state.jobs, [job.key]: job },
    };
    if (
      !suppressionChanged &&
      existing?.state !== "complete" &&
      job.state !== "complete"
    ) {
      return nextState;
    }
    return withCompletedHintSignature(nextState);
  });
}

export function removeJob(key: string): void {
  const job = getState().jobs[key];
  teardownRuntime(key);
  let suppressionChanged = false;
  if (job) {
    suppressionChanged =
      runtimeRegistry.suppressedCompletedInventoryHints.delete(
        inventoryHintKey(
          completedInventoryHintKind(job.kind, job.variant),
          job.repoId,
        ),
      );
  }
  runtimeRegistry.clearRemovalTimer(key);
  setState((state) => {
    if (!(key in state.jobs)) return state;
    const next = { ...state.jobs };
    delete next[key];
    const nextState = { ...state, jobs: next };
    if (!suppressionChanged && job?.state !== "complete") return nextState;
    return withCompletedHintSignature(nextState);
  });
}

export function setConflict(
  key: string,
  entry: DownloadManagerState["conflicts"][string] | null,
): void {
  setState((state) => {
    const next = { ...state.conflicts };
    if (entry) next[key] = entry;
    else delete next[key];
    return { ...state, conflicts: next };
  });
}

export function isCurrent(key: string, epoch: number): boolean {
  const rt = runtimeRegistry.runtimes.get(key);
  return rt != null && rt.epoch === epoch && key in getState().jobs;
}

export function scheduleRemoval(key: string, delayMs: number): void {
  const existing = runtimeRegistry.removalTimers.get(key);
  if (existing != null) clearRuntimeTimer(existing);
  const startedAt = getState().jobs[key]?.startedAt;
  const timer = window.setTimeout(() => {
    runtimeRegistry.removalTimers.delete(key);
    if (getState().jobs[key]?.startedAt === startedAt) removeJob(key);
  }, delayMs);
  runtimeRegistry.removalTimers.set(key, timer);
}

export function selectActiveJob(
  state: DownloadManagerState,
  kind: DownloadKind,
  repoId: string,
  variant?: string | null,
): ManagedDownload | null {
  if (variant !== undefined) {
    const job = state.jobs[jobKeyOf(kind, repoId, variant)];
    return job && ACTIVE_STATES.has(job.state) ? job : null;
  }
  return findActiveJobForRepo(state.jobs, kind, repoId);
}

export function clearCompletedInventoryHint(hint: InventoryHint): void {
  const key = inventoryHintKey(hint.kind, hint.repoId);
  runtimeRegistry.suppressedCompletedInventoryHints.delete(key);
  runtimeRegistry.suppressedCompletedInventoryHints.add(key);
  pruneRuntimeSuppressedHints(liveCompletedInventoryHintKeys());
  refreshCompletedHintSignature();
}

export function subscribeJobListeners(
  kind: DownloadKind,
  repoId: string,
  handlers: JobListeners,
): () => void {
  const key = repoKeyOf(kind, repoId);
  let set = runtimeRegistry.listeners.get(key);
  if (!set) {
    set = new Set();
    runtimeRegistry.listeners.set(key, set);
  }
  set.add(handlers);
  return () => {
    const current = runtimeRegistry.listeners.get(key);
    if (!current) return;
    current.delete(handlers);
    if (current.size === 0) runtimeRegistry.listeners.delete(key);
  };
}

export function setExpectedBytesForJob(
  kind: DownloadKind,
  repoId: string,
  variant: string | null,
  bytes: number,
): void {
  const job = selectActiveJob(getState(), kind, repoId, variant);
  if (!job || job.state !== "running" || bytes <= job.expectedBytes) return;
  patchJob(job.key, {
    expectedBytes: bytes,
    fraction:
      job.fraction > 0
        ? job.fraction
        : Math.min(job.downloadedBytes / bytes, MAX_PROGRESS_FRACTION),
  });
}
