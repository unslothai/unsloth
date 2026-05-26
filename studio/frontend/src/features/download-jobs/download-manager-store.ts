// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { invalidateGgufVariantsCache } from "@/features/inventory/api";
import { INVENTORY_HINT_KIND } from "@/features/inventory/constants";
import { inventoryHintKey } from "@/features/inventory/inventory-hints";
import type { InventoryHint } from "@/features/inventory/types";
import { toast } from "@/lib/toast";
import { getHfToken } from "@/stores/hf-token-store";
import { bumpInventoryVersion } from "@/stores/inventory-events";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  type DownloadJobState,
  type DownloadStartResult,
  type DownloadStartState,
  cancelDatasetDownload,
  cancelModelDownload,
  getActiveModelDownloads,
  getDatasetDownloadProgress,
  getDatasetDownloadStatus,
  getDatasetTransportStatus,
  getDownloadTransportCapabilities,
  getDownloadProgress,
  getGgufDownloadProgress,
  getModelDownloadStatus,
  getModelTransportStatus,
  startDatasetDownload,
  startModelDownload,
} from "./api";
import {
  DOWNLOAD_KIND,
  type DownloadKind,
  TRANSPORT,
  type TransportMode,
  isDownloadKind,
} from "./constants";
import { getTransportMode } from "./transport-preference";
import type { TransportConflictInfo } from "./types";
export type { DownloadKind } from "./constants";

export interface ManagedDownload {
  key: string;
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  state: DownloadJobState;
  downloadedBytes: number;
  expectedBytes: number;
  fraction: number;
  bytesPerSec: number;
  error: string | null;
  startedAt: number;
  serverGeneration?: number;
}

export interface ActiveModelDownloadRef {
  key: string;
  repoId: string;
  variant: string | null;
  state: DownloadJobState;
  startedAt: number;
}

export interface DownloadRequest {
  kind: DownloadKind;
  repoId: string;
  variant: string | null;
  expectedBytes: number;
}

export interface JobListeners {
  // Fire-and-forget side effects (e.g. an inventory refresh); the result is
  // always discarded, so the return type stays permissive.
  onComplete?: (variant: string | null, bytes: number) => unknown;
  onCancelled?: (variant: string | null) => unknown;
  onError?: (variant: string | null) => unknown;
}

interface ConflictEntry {
  info: TransportConflictInfo;
  pending: DownloadRequest;
}

interface DownloadManagerState {
  jobs: Record<string, ManagedDownload>;
  activeJobKeyByRepoKey: Record<string, string>;
  conflicts: Record<string, ConflictEntry>;
  completedHintSignature: string;
  completedInventoryHints: InventoryHint[];
}

// Polls fail transiently; only give up after this many in a row so a persistent
// 5xx surfaces a terminal error instead of freezing the bar forever.
const MAX_CONSECUTIVE_POLL_FAILURES = 6;
const POLL_INTERVAL_MS = 500;
const POLL_BACKOFF_AFTER_MS = 60_000;
const POLL_BACKOFF_INTERVAL_MS = 1_500;
const POLL_JITTER_MS = 50;
const PROGRESS_POLL_INTERVAL_MS = 1_000;
const PROGRESS_POLL_BACKOFF_INTERVAL_MS = 2_000;
// Bound each request so a hung socket aborts and counts as a transient failure
// instead of blocking every later tick.
const POLL_REQUEST_TIMEOUT_MS = 15_000;
const SPEED_EMA_WEIGHT = 0.7;
// Never let the bar read 100% before the job actually reports done.
const MAX_PROGRESS_FRACTION = 0.99;
// After a cancel, recover the UI even if no terminal status is ever observed.
const CANCEL_WATCHDOG_MS = 20_000;
// A running job whose backend record stays idle this long with no cache
// progress has been cleared out from under us; drop it silently.
const IDLE_EVICT_GRACE_MS = 60_000;
// How long a finished job stays visible in the manager before it auto-clears.
const COMPLETE_LINGER_MS = 6_000;
const CANCELLED_LINGER_MS = 6_000;
const ERROR_LINGER_MS = 12_000;
const INVENTORY_BUMP_DEBOUNCE_MS = 250;
const TRANSPORT_STATUS_RETRY_DELAY_MS = 300;

const PERSIST_KEY = "unsloth.studio.downloads";
const PERSIST_VERSION = 1;

// Fallback when there is no DOM (tests, non-browser tooling). The getter must
// return a real store, not `undefined`: createJSONStorage wraps whatever it
// gets, so `undefined` would make hydration throw on the first write.
const noopStorage: StateStorage = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};
const ACTIVE_STATES: ReadonlySet<DownloadJobState> = new Set([
  "running",
  "cancelling",
]);

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

  const variant =
    typeof value.variant === "string"
      ? value.variant
      : value.variant === null
        ? null
        : null;
  const key = jobKeyOf(kind, repoId, variant);
  return {
    key,
    kind,
    repoId,
    variant,
    state,
    downloadedBytes: nonNegativeNumber(value.downloadedBytes),
    expectedBytes: nonNegativeNumber(value.expectedBytes),
    fraction: Math.min(Math.max(finiteNumber(value.fraction, 0), 0), 1),
    bytesPerSec: nonNegativeNumber(value.bytesPerSec),
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
    activeJobKeyByRepoKey: buildActiveJobKeyByRepoKey(jobs),
    conflicts: {},
  };
}

export function repoKeyOf(kind: DownloadKind, repoId: string): string {
  return `${kind}:${repoId}`;
}

export function jobKeyOf(
  kind: DownloadKind,
  repoId: string,
  variant: string | null,
): string {
  const base = repoKeyOf(kind, repoId);
  return variant ? `${base}#${variant}` : base;
}

interface JobRuntime {
  kind: DownloadKind;
  repoId: string;
  epoch: number;
  pollTimer: number | null;
  pollStartedAt: number;
  pollingStarted: boolean;
  abort: AbortController | null;
  inFlight: boolean;
  consecutiveFailures: number;
  lastPollError: string | null;
  cancelRequested: boolean;
  watchdog: number | null;
  speedSample: { bytes: number; tMs: number } | null;
  idleSinceMs: number | null;
  lastProgressPollAt: number | null;
}

const MAX_SUPPRESSED_COMPLETED_INVENTORY_HINTS = 64;

class DownloadManagerRuntimeRegistry {
  readonly runtimes = new Map<string, JobRuntime>();
  readonly listeners = new Map<string, Set<JobListeners>>();
  readonly removalTimers = new Map<string, number>();
  readonly hydrationRetryTimers = new Set<number>();
  readonly suppressedCompletedInventoryHints = new Set<string>();
  inventoryBumpTimer: number | null = null;

  private clearTimer(timer: number | null): void {
    if (timer === null) return;
    if (typeof window !== "undefined") {
      window.clearTimeout(timer);
    } else {
      globalThis.clearTimeout(timer);
    }
  }

  reset(): void {
    for (const runtime of this.runtimes.values()) {
      this.clearTimer(runtime.pollTimer);
      runtime.abort?.abort();
      this.clearTimer(runtime.watchdog);
    }
    for (const timer of this.removalTimers.values()) {
      this.clearTimer(timer);
    }
    for (const timer of this.hydrationRetryTimers) {
      this.clearTimer(timer);
    }
    this.clearTimer(this.inventoryBumpTimer);
    this.runtimes.clear();
    this.listeners.clear();
    this.removalTimers.clear();
    this.hydrationRetryTimers.clear();
    this.suppressedCompletedInventoryHints.clear();
    this.inventoryBumpTimer = null;
  }
}

const runtimeRegistry = new DownloadManagerRuntimeRegistry();
let xetUnavailableWarningShown = false;

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    runtimeRegistry.reset();
  });
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

function pruneSuppressedCompletedInventoryHints(): void {
  const liveCompleted = liveCompletedInventoryHintKeys();
  let protectedScans = 0;
  while (
    runtimeRegistry.suppressedCompletedInventoryHints.size >
      MAX_SUPPRESSED_COMPLETED_INVENTORY_HINTS &&
    protectedScans < runtimeRegistry.suppressedCompletedInventoryHints.size
  ) {
    const oldest =
      runtimeRegistry.suppressedCompletedInventoryHints.values().next().value;
    if (!oldest) break;
    if (liveCompleted.has(oldest)) {
      runtimeRegistry.suppressedCompletedInventoryHints.delete(oldest);
      runtimeRegistry.suppressedCompletedInventoryHints.add(oldest);
      protectedScans += 1;
      continue;
    }
    runtimeRegistry.suppressedCompletedInventoryHints.delete(oldest);
    protectedScans = 0;
  }
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

interface ProgressLike {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
}

interface PollSignal {
  signal: AbortSignal;
  dispose: () => void;
}

// AbortSignal.any is unavailable on older WebView engines (WebKitGTK < 2.44,
// Safari < 17.4) while AbortSignal.timeout is not, so combine the abort signal
// with the request deadline by hand there; without this every poll throws and
// downloads fail with a misleading "lost contact" error.
// Callers MUST invoke dispose() once the request settles so the parent-listener
// + timeout pair don't pile up for the timeout's lifetime on the fallback path.
function pollSignal(parent: AbortSignal, timeoutMs: number): PollSignal {
  if (typeof AbortSignal.any === "function") {
    return {
      signal: AbortSignal.any([parent, AbortSignal.timeout(timeoutMs)]),
      dispose: () => {},
    };
  }
  const controller = new AbortController();
  let cleanedUp = false;
  let parentListenerAttached = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function onParentAbort() {
    abortWithCleanup(parent.reason);
  }

  function cleanup() {
    if (cleanedUp) return;
    cleanedUp = true;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (parentListenerAttached) {
      parentListenerAttached = false;
      try {
        parent.removeEventListener("abort", onParentAbort);
      } catch (error) {
        void error;
      }
    }
  }

  function abortWithCleanup(reason?: unknown) {
    try {
      if (!controller.signal.aborted) {
        controller.abort(reason);
      }
    } catch (error) {
      void error;
    } finally {
      cleanup();
    }
  }

  timer = setTimeout(
    () =>
      abortWithCleanup(new DOMException("Request timed out.", "TimeoutError")),
    timeoutMs,
  );
  controller.signal.addEventListener("abort", cleanup, { once: true });
  if (parent.aborted) {
    abortWithCleanup(parent.reason);
  } else {
    try {
      parent.addEventListener("abort", onParentAbort, { once: true });
      parentListenerAttached = true;
    } catch (error) {
      cleanup();
      throw error;
    }
  }
  return {
    signal: controller.signal,
    dispose: () => abortWithCleanup(),
  };
}

async function withPollRequestTimeout<T>(
  parent: AbortController | null,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const poll: PollSignal = parent
    ? pollSignal(parent.signal, POLL_REQUEST_TIMEOUT_MS)
    : {
        signal: AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
        dispose: () => {},
      };
  try {
    return await request(poll.signal);
  } finally {
    poll.dispose();
  }
}

function apiGetStatus(job: ManagedDownload, signal: AbortSignal) {
  return job.kind === DOWNLOAD_KIND.DATASET
    ? getDatasetDownloadStatus(job.repoId, signal)
    : getModelDownloadStatus(job.repoId, job.variant, signal);
}

function apiGetProgress(
  job: ManagedDownload,
  signal: AbortSignal,
): Promise<ProgressLike> {
  const token = getHfToken() || null;
  if (job.kind === DOWNLOAD_KIND.DATASET) {
    return getDatasetDownloadProgress(job.repoId, {
      signal,
      hfToken: token,
      expectedBytes: job.expectedBytes,
    });
  }
  if (job.variant) {
    return getGgufDownloadProgress(job.repoId, {
      variant: job.variant,
      expectedBytes: job.expectedBytes,
      hfToken: token,
      signal,
    });
  }
  return getDownloadProgress(job.repoId, {
    signal,
    hfToken: token,
    expectedBytes: job.expectedBytes,
  });
}

function apiStart(
  req: DownloadRequest,
  useXet: boolean,
  hfToken: string | null,
): Promise<DownloadStartResult> {
  return req.kind === DOWNLOAD_KIND.DATASET
    ? startDatasetDownload({
        repo_id: req.repoId,
        hf_token: hfToken,
        use_xet: useXet,
      })
    : startModelDownload({
        repo_id: req.repoId,
        gguf_variant: req.variant,
        hf_token: hfToken,
        use_xet: useXet,
      });
}

function apiCancel(job: ManagedDownload, signal?: AbortSignal) {
  const generation = Number.isSafeInteger(job.serverGeneration)
    ? job.serverGeneration
    : undefined;
  return job.kind === DOWNLOAD_KIND.DATASET
    ? cancelDatasetDownload({ repo_id: job.repoId, generation, signal })
    : cancelModelDownload({
        repo_id: job.repoId,
        gguf_variant: job.variant,
        generation,
        signal,
      });
}

function apiTransportStatus(req: DownloadRequest) {
  return req.kind === DOWNLOAD_KIND.DATASET
    ? getDatasetTransportStatus(req.repoId)
    : getModelTransportStatus(req.repoId);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });
}

async function apiTransportStatusWithRetry(
  req: DownloadRequest,
): Promise<Awaited<ReturnType<typeof apiTransportStatus>>> {
  try {
    return await apiTransportStatus(req);
  } catch {
    await wait(TRANSPORT_STATUS_RETRY_DELAY_MS);
    return apiTransportStatus(req);
  }
}

async function effectiveTransportMode(
  preferred: TransportMode,
): Promise<TransportMode> {
  if (preferred !== TRANSPORT.XET) {
    return preferred;
  }
  const capabilities = await getDownloadTransportCapabilities();
  if (capabilities.xet.available) {
    return preferred;
  }
  if (!xetUnavailableWarningShown) {
    xetUnavailableWarningShown = true;
    toast.warning("Xet download transport unavailable", {
      description:
        capabilities.xet.reason ?? "Studio will use HTTP downloads instead.",
    });
  }
  return TRANSPORT.HTTP;
}

function describeUnacceptedStart(state: DownloadStartState): string {
  switch (state) {
    case "deleting":
      return "This repository is being removed. Try again once it finishes.";
    case "running":
    case "cancelling":
      return "Another download for this repository is already in progress. Wait for it to finish or cancel it first.";
    default:
      return "The download could not be started. Try again in a moment.";
  }
}

function accessErrorMessage(raw: string): string | null {
  const lower = raw.toLowerCase();
  const hasStatus =
    lower.includes("401") || lower.includes("403") || lower.includes("404");
  const hasAccessSignal =
    lower.includes("unauthorized") ||
    lower.includes("forbidden") ||
    lower.includes("gated") ||
    lower.includes("private") ||
    lower.includes("repository not found") ||
    lower.includes("not found");
  if (!(hasStatus || hasAccessSignal)) return null;
  return "Couldn't access this Hugging Face repo with the token used for this download. Update the HF token and restart the download, or delete the partial download if you no longer need it.";
}

function normalizeDownloadError(
  error: unknown,
  fallback = "Failed to start download",
): string {
  const raw =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : fallback;
  return accessErrorMessage(raw) ?? raw;
}

function normalizePollError(error: unknown): string {
  return (
    normalizeDownloadError(error, "Download polling failed").trim() ||
    "Download polling failed"
  );
}

function lostContactPollError(lastError: string | null): string {
  const detail = lastError?.trim();
  return detail
    ? `Lost contact with the download. Check your connection and try again. Last error: ${detail}`
    : "Lost contact with the download. Check your connection and try again.";
}

function clearPollFailures(rt: JobRuntime): void {
  rt.consecutiveFailures = 0;
  rt.lastPollError = null;
}

function createDownloadManagerInitialState(): DownloadManagerState {
  return {
    jobs: {},
    activeJobKeyByRepoKey: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  };
}

export const useDownloadManagerStore = create<DownloadManagerState>()(
  persist(
    createDownloadManagerInitialState,
    {
      name: PERSIST_KEY,
      version: PERSIST_VERSION,
      storage: createJSONStorage(() =>
        typeof window === "undefined" ? noopStorage : window.localStorage,
      ),
      migrate: (persisted) => sanitizePersistedState(persisted),
      merge: (persisted, current) => ({
        ...current,
        ...sanitizePersistedState(persisted),
      }),
      // Only in-flight jobs are worth restoring; everything else is transient
      // UI that should not survive a reload. Conflicts are never persisted.
      partialize: (state) => ({
        jobs: Object.fromEntries(
          Object.entries(state.jobs).filter(([, job]) =>
            ACTIVE_STATES.has(job.state),
          ),
        ),
        conflicts: {},
      }),
    },
  ),
);

const setState = useDownloadManagerStore.setState;
const getState = useDownloadManagerStore.getState;

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

function buildActiveJobKeyByRepoKey(
  jobs: Record<string, ManagedDownload>,
): Record<string, string> {
  const index: Record<string, string> = {};
  for (const job of Object.values(jobs)) {
    if (!ACTIVE_STATES.has(job.state)) continue;
    const repoKey = repoKeyOf(job.kind, job.repoId);
    const currentKey = index[repoKey];
    const current = currentKey ? jobs[currentKey] : null;
    if (isPreferredRepoActiveJob(job, current)) {
      index[repoKey] = job.key;
    }
  }
  return index;
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

function findActiveJobForRepo(
  jobs: Record<string, ManagedDownload>,
  kind: DownloadKind,
  repoId: string,
): ManagedDownload | null {
  let selected: ManagedDownload | null = null;
  for (const job of Object.values(jobs)) {
    if (job.kind !== kind || job.repoId !== repoId) continue;
    if (!ACTIVE_STATES.has(job.state)) continue;
    if (isPreferredRepoActiveJob(job, selected)) {
      selected = job;
    }
  }
  return selected;
}

function findRuntimeKeyForRepo(
  kind: DownloadKind,
  repoId: string,
): string | null {
  for (const [key, runtime] of runtimeRegistry.runtimes) {
    if (runtime.kind === kind && runtime.repoId === repoId) return key;
  }
  return null;
}

function hasRuntimePeerForRepo(
  kind: DownloadKind,
  repoId: string,
  key: string,
): boolean {
  for (const [runtimeKey, runtime] of runtimeRegistry.runtimes) {
    if (runtimeKey === key) continue;
    if (runtime.kind === kind && runtime.repoId === repoId) return true;
  }
  return false;
}

function hasActiveRepoPeer(req: DownloadRequest, key: string): boolean {
  const active = findActiveJobForRepo(getState().jobs, req.kind, req.repoId);
  if (active && active.key !== key) return true;
  return hasRuntimePeerForRepo(req.kind, req.repoId, key);
}

function activeJobKeyIndexEqual(
  a: Record<string, string>,
  b: Record<string, string>,
): boolean {
  const aKeys = Object.keys(a);
  if (aKeys.length !== Object.keys(b).length) return false;
  for (const key of aKeys) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

function withActiveJobKeyIndex(
  state: DownloadManagerState,
): DownloadManagerState {
  const activeJobKeyByRepoKey = buildActiveJobKeyByRepoKey(state.jobs);
  if (activeJobKeyIndexEqual(state.activeJobKeyByRepoKey, activeJobKeyByRepoKey)) {
    return state;
  }
  return { ...state, activeJobKeyByRepoKey };
}

function refreshCompletedHintSignature(): void {
  setState((state) => withCompletedHintSignature(state));
}

function patchJob(key: string, patch: Partial<ManagedDownload>): void {
  setState((state) => {
    const job = state.jobs[key];
    if (!job) return state;
    const nextJob = { ...job, ...patch };
    const nextState = {
      ...state,
      jobs: { ...state.jobs, [key]: nextJob },
    };
    const activeChanged =
      ACTIVE_STATES.has(job.state) !== ACTIVE_STATES.has(nextJob.state);
    const indexedState = activeChanged
      ? withActiveJobKeyIndex(nextState)
      : nextState;
    if (job.state !== "complete" && nextJob.state !== "complete") {
      return indexedState;
    }
    return withCompletedHintSignature(indexedState);
  });
}

function putJob(job: ManagedDownload): void {
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
    const existingActive =
      existing !== undefined && ACTIVE_STATES.has(existing.state);
    const nextActive = ACTIVE_STATES.has(job.state);
    const indexedState =
      existingActive !== nextActive || nextActive
        ? withActiveJobKeyIndex(nextState)
        : nextState;
    if (
      !suppressionChanged &&
      existing?.state !== "complete" &&
      job.state !== "complete"
    ) {
      return indexedState;
    }
    return withCompletedHintSignature(indexedState);
  });
}

function removeJob(key: string): void {
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
  const timer = runtimeRegistry.removalTimers.get(key);
  if (timer != null) {
    window.clearTimeout(timer);
    runtimeRegistry.removalTimers.delete(key);
  }
  setState((state) => {
    if (!(key in state.jobs)) return state;
    const next = { ...state.jobs };
    delete next[key];
    const nextState = { ...state, jobs: next };
    const indexedState =
      job && ACTIVE_STATES.has(job.state)
        ? withActiveJobKeyIndex(nextState)
        : nextState;
    if (!suppressionChanged && job?.state !== "complete") return indexedState;
    return withCompletedHintSignature(indexedState);
  });
}

function setConflict(key: string, entry: ConflictEntry | null): void {
  setState((state) => {
    const next = { ...state.conflicts };
    if (entry) next[key] = entry;
    else delete next[key];
    return { ...state, conflicts: next };
  });
}

function isCurrent(key: string, epoch: number): boolean {
  const rt = runtimeRegistry.runtimes.get(key);
  return rt != null && rt.epoch === epoch && key in getState().jobs;
}

function clearWatchdog(rt: JobRuntime | undefined): void {
  if (rt?.watchdog != null) {
    window.clearTimeout(rt.watchdog);
    rt.watchdog = null;
  }
}

function teardownRuntime(key: string): void {
  const rt = runtimeRegistry.runtimes.get(key);
  if (!rt) return;
  if (rt.pollTimer != null) window.clearTimeout(rt.pollTimer);
  // Abort any in-flight poll so a superseded/torn-down job stops hitting the
  // backend instead of running until its 15s request timeout.
  rt.abort?.abort();
  clearWatchdog(rt);
  runtimeRegistry.runtimes.delete(key);
}

function scheduleRemoval(key: string, delayMs: number): void {
  const existing = runtimeRegistry.removalTimers.get(key);
  if (existing != null) window.clearTimeout(existing);
  const startedAt = getState().jobs[key]?.startedAt;
  const timer = window.setTimeout(() => {
    runtimeRegistry.removalTimers.delete(key);
    // Only clear if a new download for this key hasn't replaced it meanwhile.
    if (getState().jobs[key]?.startedAt === startedAt) removeJob(key);
  }, delayMs);
  runtimeRegistry.removalTimers.set(key, timer);
}

function notify(
  job: ManagedDownload,
  event: keyof JobListeners,
  bytes: number,
): void {
  const set = runtimeRegistry.listeners.get(repoKeyOf(job.kind, job.repoId));
  if (!set) return;
  for (const handlers of [...set]) {
    try {
      if (event === "onComplete")
        void handlers.onComplete?.(job.variant, bytes);
      else if (event === "onCancelled")
        void handlers.onCancelled?.(job.variant);
      else void handlers.onError?.(job.variant);
    } catch (error) {
      console.warn("Download job listener failed", {
        kind: job.kind,
        repoId: job.repoId,
        variant: job.variant,
        event,
        error,
      });
    }
  }
}

type Terminal = "complete" | "cancelled" | "error" | "gone";

function scheduleInventoryBump(): void {
  if (typeof window === "undefined") {
    bumpInventoryVersion();
    return;
  }
  if (runtimeRegistry.inventoryBumpTimer !== null) {
    window.clearTimeout(runtimeRegistry.inventoryBumpTimer);
  }
  runtimeRegistry.inventoryBumpTimer = window.setTimeout(() => {
    runtimeRegistry.inventoryBumpTimer = null;
    bumpInventoryVersion();
  }, INVENTORY_BUMP_DEBOUNCE_MS);
}

function hasObservedExpectedBytes(job: ManagedDownload): boolean {
  return job.expectedBytes > 0 && job.downloadedBytes >= job.expectedBytes;
}

function resolveProgressUpdate(
  job: ManagedDownload,
  progressResp: ProgressLike,
): {
  expected: number;
  downloadedBytes: number;
  fraction: number;
  madeProgress: boolean;
} {
  const reported = progressResp.expected_bytes;
  const expected = Math.max(
    reported > 0 ? reported : job.expectedBytes,
    job.expectedBytes,
  );
  const previousDownloadedBytes = job.downloadedBytes;
  const downloadedBytes = Math.max(
    previousDownloadedBytes,
    progressResp.downloaded_bytes,
  );
  const madeProgress =
    downloadedBytes > previousDownloadedBytes || expected > job.expectedBytes;
  const rawFraction =
    progressResp.progress > 0
      ? progressResp.progress
      : expected > 0
        ? downloadedBytes / expected
        : 0;
  const fraction = Math.min(rawFraction, MAX_PROGRESS_FRACTION);
  return { expected, downloadedBytes, fraction, madeProgress };
}

function progressPollDelayMs(rt: JobRuntime): number {
  return Date.now() - rt.pollStartedAt >= POLL_BACKOFF_AFTER_MS
    ? PROGRESS_POLL_BACKOFF_INTERVAL_MS
    : PROGRESS_POLL_INTERVAL_MS;
}

function shouldPollProgress(rt: JobRuntime, force: boolean): boolean {
  if (force || rt.lastProgressPollAt === null) {
    return true;
  }
  return Date.now() - rt.lastProgressPollAt >= progressPollDelayMs(rt);
}

function finalize(
  key: string,
  outcome: Terminal,
  opts: { bytes?: number; error?: string | null } = {},
): void {
  const job = getState().jobs[key];
  teardownRuntime(key);
  if (!job) return;
  if (job.kind === DOWNLOAD_KIND.MODEL) {
    invalidateGgufVariantsCache(job.repoId);
  }
  if (outcome === "gone") {
    // Record evicted from under us (backend restart, cancel watchdog, or
    // idle-evict). On-disk bytes may still have changed and per-card listeners
    // (variant cache, "On device" badge) need a chance to refresh just like
    // any other terminal outcome.
    notify(job, "onCancelled", 0);
    removeJob(key);
  } else if (outcome === "complete") {
    const bytes = opts.bytes || job.downloadedBytes || job.expectedBytes || 0;
    patchJob(key, {
      state: "complete",
      fraction: 1,
      downloadedBytes: Math.max(job.downloadedBytes, bytes),
      bytesPerSec: 0,
    });
    notify(job, "onComplete", bytes);
    scheduleRemoval(key, COMPLETE_LINGER_MS);
  } else if (outcome === "cancelled") {
    patchJob(key, { state: "cancelled", bytesPerSec: 0 });
    notify(job, "onCancelled", 0);
    scheduleRemoval(key, CANCELLED_LINGER_MS);
  } else {
    patchJob(key, {
      state: "error",
      error:
        opts.error === null
          ? null
          : normalizeDownloadError(opts.error, "Download failed"),
      bytesPerSec: 0,
    });
    notify(job, "onError", 0);
    scheduleRemoval(key, ERROR_LINGER_MS);
  }
  // Every terminal can change on-disk bytes (a completed snapshot, a kept
  // partial after cancel/error, or an evicted "gone" record), so refresh
  // inventory consumers that aren't this download's own card listener (pickers,
  // the Hub page when a background download finishes there).
  scheduleInventoryBump();
}

async function tick(key: string): Promise<void> {
  const rt = runtimeRegistry.runtimes.get(key);
  if (!rt) return;
  const job = getState().jobs[key];
  if (!job) {
    teardownRuntime(key);
    return;
  }
  // Pause network polling while the tab is hidden; the download continues
  // server-side and the next visible tick resumes it. Drop the speed baseline
  // so the first tick back measures a fresh interval, not the whole hidden gap.
  if (typeof document !== "undefined" && document.hidden) {
    rt.speedSample = null;
    return;
  }
  if (rt.inFlight) return;
  rt.inFlight = true;
  const epoch = rt.epoch;
  const abort = rt.abort;
  try {
    const status = await withPollRequestTimeout(abort, (signal) =>
      apiGetStatus(job, signal),
    );
    if (!isCurrent(key, epoch)) {
      return;
    }

    const terminalKind: Terminal | null =
      status.state === "complete"
        ? "complete"
        : status.state === "error"
          ? "error"
          : status.state === "cancelled" ||
              (status.state === "idle" && rt.cancelRequested)
            ? "cancelled"
            : null;

    if (terminalKind !== null) {
      clearPollFailures(rt);
      const current = getState().jobs[key];
      if (!current) return;
      if (terminalKind === "complete") {
        finalize(key, "complete", { bytes: current.downloadedBytes });
      } else if (terminalKind === "error") {
        finalize(key, "error", { error: status.error ?? null });
      } else {
        finalize(key, "cancelled");
      }
      return;
    }

    if (status.state === "idle") {
      const observedJob = getState().jobs[key];
      if (observedJob && hasObservedExpectedBytes(observedJob)) {
        clearPollFailures(rt);
        finalize(key, "complete", { bytes: observedJob.downloadedBytes });
        return;
      }
    }

    const forceProgress = status.state === "idle";
    if (!shouldPollProgress(rt, forceProgress)) {
      clearPollFailures(rt);
      rt.idleSinceMs = null;
      return;
    }
    rt.lastProgressPollAt = Date.now();
    const progressResp = await withPollRequestTimeout(abort, (signal) =>
      apiGetProgress(job, signal),
    );
    if (!isCurrent(key, epoch)) return;
    // Re-read after the awaits: setExpected (or another tick path) may have
    // bumped the monotonic fields on the live record while these requests were
    // in flight. Deriving them from the pre-await snapshot would regress them.
    const current = getState().jobs[key];
    if (!current) return;
    clearPollFailures(rt);

    const { expected, downloadedBytes, fraction, madeProgress } =
      resolveProgressUpdate(current, progressResp);

    const nowMs = Date.now();
    const last = rt.speedSample;
    let bytesPerSec = last ? current.bytesPerSec : 0;
    if (last) {
      const dt = (nowMs - last.tMs) / 1000;
      const db = Math.max(0, downloadedBytes - last.bytes);
      if (dt > 0) {
        const sample = db / dt;
        bytesPerSec =
          bytesPerSec > 0
            ? bytesPerSec * SPEED_EMA_WEIGHT + sample * (1 - SPEED_EMA_WEIGHT)
            : sample;
      }
    }
    rt.speedSample = { bytes: downloadedBytes, tMs: nowMs };

    patchJob(key, {
      expectedBytes: expected,
      downloadedBytes,
      fraction,
      bytesPerSec,
    });

    if (status.state === "idle") {
      const updatedJob = getState().jobs[key];
      if (updatedJob && hasObservedExpectedBytes(updatedJob)) {
        finalize(key, "complete", { bytes: updatedJob.downloadedBytes });
      } else if (madeProgress) {
        rt.idleSinceMs = null;
      } else {
        rt.idleSinceMs ??= Date.now();
        if (Date.now() - rt.idleSinceMs >= IDLE_EVICT_GRACE_MS) {
          finalize(key, "gone");
        }
      }
    } else {
      rt.idleSinceMs = null;
    }
  } catch (error) {
    if (!isCurrent(key, epoch)) return;
    const accessMessage = accessErrorMessage(
      error instanceof Error ? error.message : String(error ?? ""),
    );
    if (accessMessage) {
      finalize(key, "error", { error: accessMessage });
      return;
    }
    rt.lastPollError = normalizePollError(error);
    rt.consecutiveFailures += 1;
    if (rt.consecutiveFailures >= MAX_CONSECUTIVE_POLL_FAILURES) {
      finalize(key, "error", {
        error: lostContactPollError(rt.lastPollError),
      });
    }
  } finally {
    rt.inFlight = false;
  }
}

function pollDelayMs(rt: JobRuntime): number {
  const elapsedMs = Date.now() - rt.pollStartedAt;
  const base =
    elapsedMs >= POLL_BACKOFF_AFTER_MS
      ? POLL_BACKOFF_INTERVAL_MS
      : POLL_INTERVAL_MS;
  const jitter = Math.round((Math.random() * 2 - 1) * POLL_JITTER_MS);
  return Math.max(100, base + jitter);
}

async function runPollTick(key: string): Promise<void> {
  try {
    await tick(key);
  } finally {
    const rt = runtimeRegistry.runtimes.get(key);
    if (rt && key in getState().jobs) {
      rt.pollTimer = window.setTimeout(
        () => void runPollTick(key),
        pollDelayMs(rt),
      );
    }
  }
}

function beginPolling(key: string, rt: JobRuntime): void {
  rt.pollingStarted = true;
  void runPollTick(key);
}

async function startJob(
  req: DownloadRequest,
  opts: { adopt?: boolean; useXet?: boolean } = {},
): Promise<void> {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  if (hasActiveRepoPeer(req, key)) return;
  // Capture the epoch before teardown so it stays monotonic per key: a stale
  // in-flight tick from the superseded runtime must never match isCurrent().
  const nextEpoch = (runtimeRegistry.runtimes.get(key)?.epoch ?? 0) + 1;
  teardownRuntime(key);
  const existing = getState().jobs[key];
  // Adopting a job that was mid-cancel (e.g. reloaded during a cancel) must keep
  // its cancelling state so the poller still settles it as cancelled.
  const adoptingCancel =
    opts.adopt === true && existing?.state === "cancelling";
  const rt: JobRuntime = {
    kind: req.kind,
    repoId: req.repoId,
    epoch: nextEpoch,
    pollTimer: null,
    pollStartedAt: Date.now(),
    pollingStarted: false,
    abort: new AbortController(),
    inFlight: false,
    consecutiveFailures: 0,
    lastPollError: null,
    cancelRequested: adoptingCancel,
    watchdog: null,
    speedSample: null,
    idleSinceMs: null,
    lastProgressPollAt: null,
  };
  runtimeRegistry.runtimes.set(key, rt);
  const epoch = rt.epoch;

  // Never let a stale 0 (e.g. start fired before the size fetch resolved)
  // clobber a known total and blank the bar's denominator.
  const expected = Math.max(existing?.expectedBytes ?? 0, req.expectedBytes);
  const hfToken = getHfToken() || null;
  const requestedUseXet = opts.useXet ?? (getTransportMode() === TRANSPORT.XET);
  const requestedMode: TransportMode = requestedUseXet
    ? TRANSPORT.XET
    : TRANSPORT.HTTP;
  const mode = opts.adopt
    ? TRANSPORT.HTTP
    : await effectiveTransportMode(requestedMode);
  // When re-adopting a running job (e.g. after a reload), keep the bytes we
  // already knew so the bar doesn't flash back to 0% before the first poll.
  const seedDownloaded = opts.adopt ? (existing?.downloadedBytes ?? 0) : 0;
  const seedFraction = opts.adopt ? (existing?.fraction ?? 0) : 0;
  if (hasActiveRepoPeer(req, key)) {
    teardownRuntime(key);
    return;
  }
  putJob({
    key,
    kind: req.kind,
    repoId: req.repoId,
    variant: req.variant,
    state: adoptingCancel ? "cancelling" : "running",
    downloadedBytes: seedDownloaded,
    expectedBytes: expected,
    fraction: seedFraction,
    bytesPerSec: 0,
    error: null,
    startedAt: Date.now(),
  });

  if (!opts.adopt) {
    let result: DownloadStartResult;
    try {
      result = await apiStart(
        req,
        mode === TRANSPORT.XET,
        hfToken,
      );
    } catch (err) {
      if (!isCurrent(key, epoch)) return;
      finalize(key, "error", {
        error: normalizeDownloadError(err),
      });
      return;
    }
    if (!isCurrent(key, epoch)) return;
    // The backend answers 202 even when it refuses to spawn a worker; only poll
    // when a worker for this key is actually live.
    if (!result.accepted) {
      finalize(key, "error", { error: describeUnacceptedStart(result.state) });
      return;
    }
    if (Number.isSafeInteger(result.generation)) {
      patchJob(key, { serverGeneration: result.generation });
    }
  }

  beginPolling(key, rt);
}

async function cancelJob(key: string): Promise<void> {
  const job = getState().jobs[key];
  if (!job) return;
  const rt = runtimeRegistry.runtimes.get(key);
  const cancelEpoch = rt?.epoch ?? 0;
  if (rt) rt.cancelRequested = true;
  patchJob(key, { state: "cancelling" });
  clearWatchdog(rt);
  // Fallback: if the poller never sees a terminal state (wedged backend, or the
  // record evicted before a tick reads it), recover after a bounded wait.
  if (rt) {
    rt.watchdog = window.setTimeout(() => {
      const live = runtimeRegistry.runtimes.get(key);
      if (!live || live.epoch !== cancelEpoch || !live.cancelRequested) return;
      finalize(key, "gone");
    }, CANCEL_WATCHDOG_MS);
  }
  try {
    await apiCancel(job, rt?.abort?.signal);
    const live = runtimeRegistry.runtimes.get(key);
    // A newer job for this key superseded us; the cancel is stale.
    if (rt && live && live.epoch !== cancelEpoch) return;
    // With no live poller to observe the terminal state (no runtime, or it was
    // never polling), settle here instead.
    if (!live || !live.pollingStarted) finalize(key, "cancelled");
  } catch (err) {
    const liveAtError = runtimeRegistry.runtimes.get(key);
    if (rt && liveAtError && liveAtError.epoch !== cancelEpoch) return;
    if (rt?.abort?.signal.aborted) return;

    // The cancel request failed at the transport, but the worker may have
    // already died server-side. Re-probe status before flipping the CTA back
    // to "running"; otherwise the user waits IDLE_EVICT_GRACE_MS for the
    // idle-eviction rescue while the backend is already silent.
    let probedTerminal: Terminal | null = null;
    let probedError: string | null = null;
    try {
      const probe = await apiGetStatus(
        job,
        AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
      );
      if (probe.state === "complete") {
        probedTerminal = "complete";
      } else if (probe.state === "error") {
        probedTerminal = "error";
        probedError = probe.error ?? null;
      } else if (probe.state === "cancelled" || probe.state === "idle") {
        probedTerminal = "cancelled";
      }
    } catch {
      // Probe also failed; fall through to revert.
    }

    const live = runtimeRegistry.runtimes.get(key);
    if (rt && live && live.epoch !== cancelEpoch) return;

    if (probedTerminal !== null) {
      if (live) clearWatchdog(live);
      if (probedTerminal === "complete") {
        const current = getState().jobs[key];
        finalize(key, "complete", { bytes: current?.downloadedBytes ?? 0 });
      } else if (probedTerminal === "error") {
        finalize(key, "error", { error: probedError });
      } else {
        finalize(key, "cancelled");
      }
      return;
    }

    if (live) {
      live.cancelRequested = false;
      clearWatchdog(live);
    }
    patchJob(key, { state: "running" });
    toast.error("Couldn't cancel the download. It's still running.");
    console.warn("Failed to cancel download", err);
  }
}

async function requestStart(req: DownloadRequest): Promise<void> {
  if (findActiveJobForRepo(getState().jobs, req.kind, req.repoId)) return;
  if (findRuntimeKeyForRepo(req.kind, req.repoId)) return;
  let mode: TransportMode;
  try {
    const status = await apiTransportStatusWithRetry(req);
    mode = await effectiveTransportMode(getTransportMode());
    if (
      status.has_partial &&
      status.last_transport &&
      status.last_transport !== mode
    ) {
      setConflict(repoKeyOf(req.kind, req.repoId), {
        info: {
          previous: status.last_transport,
          next: mode,
          resumable: status.resumable,
        },
        pending: req,
      });
      return;
    }
  } catch (err) {
    console.warn(
      "Transport status check failed; download was not started.",
      err,
    );
    toast.error("Couldn't verify existing partial download", {
      description:
        "No download was started. Check your connection and try again.",
    });
    return;
  }
  void startJob(req, { useXet: mode === TRANSPORT.XET });
}

function adoptJob(req: DownloadRequest): void {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  if (runtimeRegistry.runtimes.get(key)?.pollingStarted) return;
  void startJob(req, { adopt: true });
}

async function probeAndAdopt(
  kind: DownloadKind,
  repoId: string,
  signal: AbortSignal,
): Promise<void> {
  try {
    if (kind === DOWNLOAD_KIND.MODEL) {
      const downloads = await getActiveModelDownloads(repoId, signal);
      if (signal.aborted) return;
      const active = downloads.find(
        (download) =>
          download.variant === null &&
          (download.state === "running" || download.state === "cancelling"),
      );
      if (active) {
        adoptJob({ kind, repoId, variant: null, expectedBytes: 0 });
      }
      return;
    }

    const status = await getDatasetDownloadStatus(repoId, signal);
    if (signal.aborted) return;
    if (status.state === "running" || status.state === "cancelling") {
      adoptJob({ kind, repoId, variant: null, expectedBytes: 0 });
    }
  } catch (error) {
    if (signal.aborted) {
      return;
    }
    if (import.meta.env.DEV) {
      console.debug("Download adoption probe failed", { kind, repoId, error });
    }
  }
}

function setExpected(
  kind: DownloadKind,
  repoId: string,
  variant: string | null,
  bytes: number,
): void {
  const job = selectActiveJob(getState(), kind, repoId, variant);
  if (!job || job.state !== "running" || bytes <= job.expectedBytes) return;
  // Reflect a late-resolving total on the live bar immediately rather than
  // waiting for the next poll tick, so the denominator stops reading unknown.
  patchJob(job.key, {
    expectedBytes: bytes,
    fraction:
      job.fraction > 0
        ? job.fraction
        : Math.min(job.downloadedBytes / bytes, MAX_PROGRESS_FRACTION),
  });
}

function reportConflictStartError(error: unknown): void {
  const description =
    error instanceof Error ? error.message : String(error || "Unknown error");
  toast.error("Couldn't start download", { description });
}

function resumeConflict(repoKey: string): void {
  const entry = getState().conflicts[repoKey];
  if (!entry) return;
  setConflict(repoKey, null);
  void startJob(entry.pending, {
    useXet: entry.info.previous === TRANSPORT.XET,
  }).catch(reportConflictStartError);
}

function restartConflict(repoKey: string): void {
  const entry = getState().conflicts[repoKey];
  if (!entry) return;
  setConflict(repoKey, null);
  void startJob(entry.pending, {
    useXet: entry.info.next === TRANSPORT.XET,
  }).catch(reportConflictStartError);
}

function cancelConflict(repoKey: string): void {
  setConflict(repoKey, null);
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

function buildActiveModelDownloadRefs(
  state: DownloadManagerState,
): ActiveModelDownloadRef[] {
  return Object.values(state.jobs)
    .filter(
      (job) => job.kind === DOWNLOAD_KIND.MODEL && ACTIVE_STATES.has(job.state),
    )
    .map((job) => ({
      key: job.key,
      repoId: job.repoId,
      variant: job.variant,
      state: job.state,
      startedAt: job.startedAt,
    }))
    .sort((a, b) => b.startedAt - a.startedAt);
}

export function selectActiveModelDownloadRefs(
  state: DownloadManagerState,
): ActiveModelDownloadRef[] {
  return buildActiveModelDownloadRefs(state);
}

export function createActiveModelDownloadRefsSelector(): (
  state: DownloadManagerState,
) => ActiveModelDownloadRef[] {
  let cache: { signature: string; refs: ActiveModelDownloadRef[] } = {
    signature: "",
    refs: [],
  };
  return (state) => {
    const refs = buildActiveModelDownloadRefs(state);
    const signature = refs
      .map(
        (ref) =>
          `${ref.key}\u0001${ref.repoId}\u0001${ref.variant ?? ""}\u0001${ref.state}\u0001${ref.startedAt}`,
      )
      .join("\u0002");
    if (signature === cache.signature) {
      return cache.refs;
    }
    cache = { signature, refs };
    return refs;
  };
}

export function getCompletedInventoryHints(): InventoryHint[] {
  return getState().completedInventoryHints;
}

export function clearCompletedInventoryHint(hint: InventoryHint): void {
  const key = inventoryHintKey(hint.kind, hint.repoId);
  runtimeRegistry.suppressedCompletedInventoryHints.delete(key);
  runtimeRegistry.suppressedCompletedInventoryHints.add(key);
  pruneSuppressedCompletedInventoryHints();
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

let hydrated = false;
const HYDRATE_STATUS_TIMEOUT_RETRIES = 4;
const HYDRATE_STATUS_RETRY_MS = 2_500;

function isRequestTimeout(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "TimeoutError")
  );
}

type HydratedIdleProbeResult = "active" | "gone" | "settled";

async function probeHydratedIdleProgress(
  key: string,
  job: ManagedDownload,
): Promise<HydratedIdleProbeResult> {
  try {
    const progressResp = await apiGetProgress(
      job,
      AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
    );
    const current = getState().jobs[key];
    if (!current || !ACTIVE_STATES.has(current.state)) return "settled";
    const { expected, downloadedBytes, fraction } = resolveProgressUpdate(
      current,
      progressResp,
    );
    patchJob(key, {
      expectedBytes: expected,
      downloadedBytes,
      fraction,
    });
    const updated = getState().jobs[key];
    if (!updated || !ACTIVE_STATES.has(updated.state)) return "settled";
    if (hasObservedExpectedBytes(updated)) {
      finalize(key, "complete", { bytes: updated.downloadedBytes });
      return "settled";
    }
    return downloadedBytes > 0 ? "active" : "gone";
  } catch {
    return "active";
  }
}

async function settleHydratedJob(
  key: string,
  req: DownloadRequest,
  status: { state: DownloadJobState; error?: string | null },
): Promise<void> {
  if (status.state === "running" || status.state === "cancelling") {
    adoptJob(req);
  } else if (status.state === "complete") {
    finalize(key, "complete");
  } else if (status.state === "cancelled") {
    finalize(key, "cancelled");
  } else if (status.state === "error") {
    finalize(key, "error", { error: status.error ?? null });
  } else {
    const job = getState().jobs[key];
    if (job && hasObservedExpectedBytes(job)) {
      finalize(key, "complete", { bytes: job.downloadedBytes });
    } else if (job?.state === "running") {
      const probeResult = await probeHydratedIdleProgress(key, job);
      if (probeResult === "settled") return;
      if (probeResult === "active") {
        adoptJob(req);
        return;
      }
      const latest = getState().jobs[key];
      if (latest && hasObservedExpectedBytes(latest)) {
        finalize(key, "complete", { bytes: latest.downloadedBytes });
      } else {
        finalize(key, "gone");
      }
    } else {
      finalize(key, "gone");
    }
  }
}

function scheduleHydrationProbeRetry(
  key: string,
  req: DownloadRequest,
  attempt: number,
): void {
  const timer = window.setTimeout(() => {
    runtimeRegistry.hydrationRetryTimers.delete(timer);
    const job = getState().jobs[key];
    if (!job || !ACTIVE_STATES.has(job.state)) return;
    void probeHydratedJob(key, req, attempt);
  }, HYDRATE_STATUS_RETRY_MS);
  runtimeRegistry.hydrationRetryTimers.add(timer);
}

async function probeHydratedJob(
  key: string,
  req: DownloadRequest,
  attempt: number,
): Promise<void> {
  const job = getState().jobs[key];
  if (!job || !ACTIVE_STATES.has(job.state)) return;
  try {
    const status = await apiGetStatus(
      job,
      AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
    );
    const current = getState().jobs[key];
    if (!current || !ACTIVE_STATES.has(current.state)) return;
    await settleHydratedJob(key, req, status);
  } catch (error) {
    if (isRequestTimeout(error) && attempt < HYDRATE_STATUS_TIMEOUT_RETRIES) {
      scheduleHydrationProbeRetry(key, req, attempt + 1);
      return;
    }
    adoptJob(req);
  }
}

// Restore in-flight downloads after a full page reload: re-verify each persisted
// job against the backend and resume polling only for those still running.
// Callers must ensure the backend + auth are ready before invoking; otherwise
// every probe fails and persisted jobs would be discarded while the worker
// keeps writing to disk with no UI to observe it.
export function hydrateDownloadManager(): void {
  if (hydrated) return;
  hydrated = true;
  const jobs = Object.values(getState().jobs);
  for (const job of jobs) {
    if (!ACTIVE_STATES.has(job.state)) {
      removeJob(job.key);
      continue;
    }
    const req: DownloadRequest = {
      kind: job.kind,
      repoId: job.repoId,
      variant: job.variant,
      expectedBytes: job.expectedBytes,
    };
    void probeHydratedJob(job.key, req, 0);
  }
}

export function __resetDownloadManagerForTests(): void {
  runtimeRegistry.reset();
  xetUnavailableWarningShown = false;
  hydrated = false;
  setState(createDownloadManagerInitialState());
}

export interface DownloadManagerController {
  start: typeof startJob;
  requestStart: typeof requestStart;
  cancel: typeof cancelJob;
  adopt: typeof adoptJob;
  probeAndAdopt: typeof probeAndAdopt;
  setExpected: typeof setExpected;
  resumeConflict: typeof resumeConflict;
  restartConflict: typeof restartConflict;
  cancelConflict: typeof cancelConflict;
  dismiss: typeof removeJob;
}

export const downloadManager: DownloadManagerController = {
  start: startJob,
  requestStart,
  cancel: cancelJob,
  adopt: adoptJob,
  probeAndAdopt,
  setExpected,
  resumeConflict,
  restartConflict,
  cancelConflict,
  dismiss: removeJob,
};
