// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type DownloadJobState,
  type DownloadStartResult,
  type DownloadStartState,
  cancelDatasetDownload,
  cancelModelDownload,
  getDatasetDownloadProgress,
  getDatasetDownloadStatus,
  getDatasetTransportStatus,
  getDownloadProgress,
  getGgufDownloadProgress,
  getModelDownloadStatus,
  getModelTransportStatus,
  startDatasetDownload,
  startModelDownload,
} from "@/features/chat";
import { getHfToken } from "@/stores/hf-token-store";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import type { TransportConflictInfo } from "../components/transport-conflict-dialog";
import { bumpInventoryVersion } from "@/stores/inventory-events";
import { getTransportMode } from "../lib/transport-preference";

// Backend jobs live under model and dataset registries; the on-disk progress
// shape and the start/cancel endpoints are chosen by kind plus, for models,
// whether a GGUF variant is set.
export type DownloadKind = "model" | "dataset";

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
  conflicts: Record<string, ConflictEntry>;
}

// Polls fail transiently; only give up after this many in a row so a persistent
// 5xx surfaces a terminal error instead of freezing the bar forever.
const MAX_CONSECUTIVE_POLL_FAILURES = 6;
const POLL_INTERVAL_MS = 500;
// Bound each request so a hung socket aborts and counts as a transient failure
// instead of blocking every later tick.
const POLL_REQUEST_TIMEOUT_MS = 15_000;
const SPEED_EMA_WEIGHT = 0.7;
// Never let the bar read 100% before the job actually reports done.
const MAX_PROGRESS_FRACTION = 0.99;
// After a cancel, recover the UI even if no terminal status is ever observed.
const CANCEL_WATCHDOG_MS = 20_000;
// A running job whose backend record reports idle this many polls in a row has
// been cleared out from under us (record evicted); drop it silently.
const MAX_IDLE_READS = 3;
// How long a finished job stays visible in the manager before it auto-clears.
const COMPLETE_LINGER_MS = 6_000;
const CANCELLED_LINGER_MS = 6_000;
const ERROR_LINGER_MS = 12_000;

const PERSIST_KEY = "unsloth.studio.downloads";

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
  epoch: number;
  interval: number | null;
  abort: AbortController | null;
  inFlight: boolean;
  consecutiveFailures: number;
  cancelRequested: boolean;
  watchdog: number | null;
  speedSample: { bytes: number; tMs: number } | null;
  idleReads: number;
}

// Timers, abort controllers, and polling bookkeeping live outside the store so
// the persisted state stays plain serializable data.
const runtimes = new Map<string, JobRuntime>();
const listeners = new Map<string, Set<JobListeners>>();
const removalTimers = new Map<string, number>();

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
  const onParentAbort = () => controller.abort(parent.reason);
  if (parent.aborted) {
    controller.abort(parent.reason);
  } else {
    parent.addEventListener("abort", onParentAbort, { once: true });
  }
  const timer = setTimeout(
    () => controller.abort(new DOMException("Request timed out.", "TimeoutError")),
    timeoutMs,
  );
  controller.signal.addEventListener(
    "abort",
    () => {
      clearTimeout(timer);
      parent.removeEventListener("abort", onParentAbort);
    },
    { once: true },
  );
  return {
    signal: controller.signal,
    dispose: () => {
      if (!controller.signal.aborted) {
        controller.abort();
      }
    },
  };
}

function apiGetStatus(job: ManagedDownload, signal: AbortSignal) {
  return job.kind === "dataset"
    ? getDatasetDownloadStatus(job.repoId, signal)
    : getModelDownloadStatus(job.repoId, job.variant, signal);
}

function apiGetProgress(
  job: ManagedDownload,
  signal: AbortSignal,
): Promise<ProgressLike> {
  const token = getHfToken() || undefined;
  if (job.kind === "dataset") {
    return getDatasetDownloadProgress(job.repoId, signal, token);
  }
  if (job.variant) {
    return getGgufDownloadProgress(
      job.repoId,
      job.variant,
      job.expectedBytes,
      token,
      signal,
    );
  }
  return getDownloadProgress(job.repoId, signal, token);
}

function apiStart(
  req: DownloadRequest,
  useXet: boolean,
): Promise<DownloadStartResult> {
  const token = getHfToken() || undefined;
  return req.kind === "dataset"
    ? startDatasetDownload({
        repo_id: req.repoId,
        hf_token: token,
        use_xet: useXet,
      })
    : startModelDownload({
        repo_id: req.repoId,
        gguf_variant: req.variant,
        hf_token: token,
        use_xet: useXet,
      });
}

function apiCancel(job: ManagedDownload) {
  return job.kind === "dataset"
    ? cancelDatasetDownload({ repo_id: job.repoId })
    : cancelModelDownload({ repo_id: job.repoId, gguf_variant: job.variant });
}

function apiTransportStatus(req: DownloadRequest) {
  return req.kind === "dataset"
    ? getDatasetTransportStatus(req.repoId)
    : getModelTransportStatus(req.repoId);
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

export const useDownloadManagerStore = create<DownloadManagerState>()(
  persist(
    (): DownloadManagerState => ({
      jobs: {},
      conflicts: {},
    }),
    {
      name: PERSIST_KEY,
      storage: createJSONStorage(() =>
        typeof window === "undefined" ? noopStorage : window.localStorage,
      ),
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

function patchJob(key: string, patch: Partial<ManagedDownload>): void {
  setState((state) => {
    const job = state.jobs[key];
    if (!job) return state;
    return { ...state, jobs: { ...state.jobs, [key]: { ...job, ...patch } } };
  });
}

function putJob(job: ManagedDownload): void {
  setState((state) => ({ ...state, jobs: { ...state.jobs, [job.key]: job } }));
}

function removeJob(key: string): void {
  const timer = removalTimers.get(key);
  if (timer != null) {
    window.clearTimeout(timer);
    removalTimers.delete(key);
  }
  setState((state) => {
    if (!(key in state.jobs)) return state;
    const next = { ...state.jobs };
    delete next[key];
    return { ...state, jobs: next };
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
  const rt = runtimes.get(key);
  return rt != null && rt.epoch === epoch && key in getState().jobs;
}

function clearWatchdog(rt: JobRuntime | undefined): void {
  if (rt?.watchdog != null) {
    window.clearTimeout(rt.watchdog);
    rt.watchdog = null;
  }
}

function teardownRuntime(key: string): void {
  const rt = runtimes.get(key);
  if (!rt) return;
  if (rt.interval != null) window.clearInterval(rt.interval);
  // Abort any in-flight poll so a superseded/torn-down job stops hitting the
  // backend instead of running until its 15s request timeout.
  rt.abort?.abort();
  clearWatchdog(rt);
  runtimes.delete(key);
}

function scheduleRemoval(key: string, delayMs: number): void {
  const existing = removalTimers.get(key);
  if (existing != null) window.clearTimeout(existing);
  const startedAt = getState().jobs[key]?.startedAt;
  const timer = window.setTimeout(() => {
    removalTimers.delete(key);
    // Only clear if a new download for this key hasn't replaced it meanwhile.
    if (getState().jobs[key]?.startedAt === startedAt) removeJob(key);
  }, delayMs);
  removalTimers.set(key, timer);
}

function notify(
  job: ManagedDownload,
  event: keyof JobListeners,
  bytes: number,
): void {
  const set = listeners.get(repoKeyOf(job.kind, job.repoId));
  if (!set) return;
  for (const handlers of set) {
    try {
      if (event === "onComplete")
        void handlers.onComplete?.(job.variant, bytes);
      else if (event === "onCancelled")
        void handlers.onCancelled?.(job.variant);
      else void handlers.onError?.(job.variant);
    } catch {
      // A listener side effect (e.g. an inventory refresh) must never wedge the
      // poller teardown.
    }
  }
}

type Terminal = "complete" | "cancelled" | "error" | "gone";

function finalize(
  key: string,
  outcome: Terminal,
  opts: { bytes?: number; error?: string | null } = {},
): void {
  const job = getState().jobs[key];
  teardownRuntime(key);
  if (!job) return;
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
      error: opts.error ?? null,
      bytesPerSec: 0,
    });
    notify(job, "onError", 0);
    scheduleRemoval(key, ERROR_LINGER_MS);
  }
  // Every terminal can change on-disk bytes (a completed snapshot, a kept
  // partial after cancel/error, or an evicted "gone" record), so refresh
  // inventory consumers that aren't this download's own card listener (pickers,
  // the Hub page when a background download finishes there).
  bumpInventoryVersion();
}

async function tick(key: string): Promise<void> {
  const rt = runtimes.get(key);
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
  const poll: PollSignal = abort
    ? pollSignal(abort.signal, POLL_REQUEST_TIMEOUT_MS)
    : { signal: AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS), dispose: () => {} };
  try {
    const [progressResp, status] = await Promise.all([
      apiGetProgress(job, poll.signal),
      apiGetStatus(job, poll.signal),
    ]);
    if (!isCurrent(key, epoch)) return;
    // Re-read after the awaits: setExpected (or another tick path) may have
    // bumped the monotonic fields on the live record while these requests were
    // in flight. Deriving them from the pre-await snapshot would regress them.
    const current = getState().jobs[key];
    if (!current) return;
    rt.consecutiveFailures = 0;

    const reported = progressResp.expected_bytes;
    const expected = Math.max(
      reported > 0 ? reported : current.expectedBytes,
      current.expectedBytes,
    );
    const fraction =
      progressResp.progress > 0
        ? progressResp.progress
        : expected > 0
          ? Math.min(
              progressResp.downloaded_bytes / expected,
              MAX_PROGRESS_FRACTION,
            )
          : 0;

    const nowMs = Date.now();
    const last = rt.speedSample;
    let bytesPerSec = current.bytesPerSec;
    if (last) {
      const dt = (nowMs - last.tMs) / 1000;
      const db = progressResp.downloaded_bytes - last.bytes;
      if (dt > 0 && db >= 0) {
        const sample = db / dt;
        bytesPerSec =
          bytesPerSec > 0
            ? bytesPerSec * SPEED_EMA_WEIGHT + sample * (1 - SPEED_EMA_WEIGHT)
            : sample;
      }
    }
    rt.speedSample = { bytes: progressResp.downloaded_bytes, tMs: nowMs };

    patchJob(key, {
      expectedBytes: expected,
      downloadedBytes: progressResp.downloaded_bytes,
      fraction,
      bytesPerSec,
    });

    if (status.state === "complete") {
      finalize(key, "complete", { bytes: progressResp.downloaded_bytes });
    } else if (
      status.state === "cancelled" ||
      (status.state === "idle" && rt.cancelRequested)
    ) {
      finalize(key, "cancelled");
    } else if (status.state === "error") {
      finalize(key, "error", { error: status.error ?? null });
    } else if (status.state === "idle") {
      rt.idleReads += 1;
      if (rt.idleReads >= MAX_IDLE_READS) finalize(key, "gone");
    } else {
      rt.idleReads = 0;
    }
  } catch {
    // A supersede, teardown, repo change, or unmount aborts this epoch's
    // requests; that is not a download failure, so don't count it.
    if (!isCurrent(key, epoch)) return;
    rt.consecutiveFailures += 1;
    if (rt.consecutiveFailures >= MAX_CONSECUTIVE_POLL_FAILURES) {
      finalize(key, "error", {
        error:
          "Lost contact with the download. Check your connection and try again.",
      });
    }
  } finally {
    poll.dispose();
    rt.inFlight = false;
  }
}

function beginPolling(key: string, rt: JobRuntime): void {
  void tick(key);
  rt.interval = window.setInterval(() => void tick(key), POLL_INTERVAL_MS);
}

async function startJob(
  req: DownloadRequest,
  opts: { adopt?: boolean; useXet?: boolean } = {},
): Promise<void> {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  // Capture the epoch before teardown so it stays monotonic per key: a stale
  // in-flight tick from the superseded runtime must never match isCurrent().
  const nextEpoch = (runtimes.get(key)?.epoch ?? 0) + 1;
  teardownRuntime(key);
  const existing = getState().jobs[key];
  // Adopting a job that was mid-cancel (e.g. reloaded during a cancel) must keep
  // its cancelling state so the poller still settles it as cancelled.
  const adoptingCancel =
    opts.adopt === true && existing?.state === "cancelling";
  const rt: JobRuntime = {
    epoch: nextEpoch,
    interval: null,
    abort: new AbortController(),
    inFlight: false,
    consecutiveFailures: 0,
    cancelRequested: adoptingCancel,
    watchdog: null,
    speedSample: null,
    idleReads: 0,
  };
  runtimes.set(key, rt);
  const epoch = rt.epoch;

  // Never let a stale 0 (e.g. start fired before the size fetch resolved)
  // clobber a known total and blank the bar's denominator.
  const expected = Math.max(existing?.expectedBytes ?? 0, req.expectedBytes);
  // When re-adopting a running job (e.g. after a reload), keep the bytes we
  // already knew so the bar doesn't flash back to 0% before the first poll.
  const seedDownloaded = opts.adopt ? (existing?.downloadedBytes ?? 0) : 0;
  const seedFraction = opts.adopt ? (existing?.fraction ?? 0) : 0;
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
      result = await apiStart(req, opts.useXet ?? getTransportMode() === "xet");
    } catch (err) {
      if (!isCurrent(key, epoch)) return;
      finalize(key, "error", {
        error: err instanceof Error ? err.message : "Failed to start download",
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
  }

  beginPolling(key, rt);
}

async function cancelJob(key: string): Promise<void> {
  const job = getState().jobs[key];
  if (!job) return;
  const rt = runtimes.get(key);
  const cancelEpoch = rt?.epoch ?? 0;
  if (rt) rt.cancelRequested = true;
  patchJob(key, { state: "cancelling" });
  clearWatchdog(rt);
  // Fallback: if the poller never sees a terminal state (wedged backend, or the
  // record evicted before a tick reads it), recover after a bounded wait.
  if (rt) {
    rt.watchdog = window.setTimeout(() => {
      const live = runtimes.get(key);
      if (!live || live.epoch !== cancelEpoch || !live.cancelRequested) return;
      finalize(key, "gone");
    }, CANCEL_WATCHDOG_MS);
  }
  try {
    await apiCancel(job);
    const live = runtimes.get(key);
    // A newer job for this key superseded us; the cancel is stale.
    if (rt && live && live.epoch !== cancelEpoch) return;
    // With no live poller to observe the terminal state (no runtime, or it was
    // never polling), settle here instead.
    if (!live || live.interval == null) finalize(key, "cancelled");
  } catch (err) {
    const live = runtimes.get(key);
    if (rt && live && live.epoch !== cancelEpoch) return;
    if (live) {
      live.cancelRequested = false;
      clearWatchdog(live);
    }
    // The download may still be running; revert the CTA and let polling continue.
    patchJob(key, { state: "running" });
    toast.error("Couldn't cancel the download. It's still running.");
    console.warn("Failed to cancel download", err);
  }
}

async function requestStart(req: DownloadRequest): Promise<void> {
  const mode = getTransportMode();
  try {
    const status = await apiTransportStatus(req);
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
      "Transport status check failed; starting download without the conflict prompt.",
      err,
    );
  }
  void startJob(req, { useXet: mode === "xet" });
}

function adoptJob(req: DownloadRequest): void {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  if (runtimes.get(key)?.interval != null) return;
  void startJob(req, { adopt: true });
}

async function probeAndAdopt(
  kind: DownloadKind,
  repoId: string,
  signal: AbortSignal,
): Promise<void> {
  try {
    const status =
      kind === "dataset"
        ? await getDatasetDownloadStatus(repoId, signal)
        : await getModelDownloadStatus(repoId, null, signal);
    if (signal.aborted) return;
    if (status.state === "running" || status.state === "cancelling") {
      adoptJob({ kind, repoId, variant: null, expectedBytes: 0 });
    }
  } catch {
    // Best-effort: a failed probe just skips adoption until the next mount.
  }
}

function setExpected(kind: DownloadKind, repoId: string, bytes: number): void {
  const job = selectActiveJob(getState(), kind, repoId);
  if (!job || bytes <= job.expectedBytes) return;
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

function resumeConflict(repoKey: string): void {
  const entry = getState().conflicts[repoKey];
  if (!entry) return;
  setConflict(repoKey, null);
  void startJob(entry.pending, { useXet: entry.info.previous === "xet" });
}

function restartConflict(repoKey: string): void {
  const entry = getState().conflicts[repoKey];
  if (!entry) return;
  setConflict(repoKey, null);
  void startJob(entry.pending, { useXet: entry.info.next === "xet" });
}

function cancelConflict(repoKey: string): void {
  setConflict(repoKey, null);
}

export function selectActiveJob(
  state: DownloadManagerState,
  kind: DownloadKind,
  repoId: string,
): ManagedDownload | null {
  for (const job of Object.values(state.jobs)) {
    if (job.kind !== kind || job.repoId !== repoId) continue;
    if (ACTIVE_STATES.has(job.state)) return job;
  }
  return null;
}

export function subscribeJobListeners(
  kind: DownloadKind,
  repoId: string,
  handlers: JobListeners,
): () => void {
  const key = repoKeyOf(kind, repoId);
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(handlers);
  return () => {
    const current = listeners.get(key);
    if (!current) return;
    current.delete(handlers);
    if (current.size === 0) listeners.delete(key);
  };
}

let hydrated = false;

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
    void apiGetStatus(job, AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS))
      .then((status) => {
        if (status.state === "running" || status.state === "cancelling") {
          adoptJob(req);
        } else {
          removeJob(job.key);
        }
      })
      // Transient probe failures (5xx, network blip) must not orphan a worker
      // still writing to disk: adopt and let polling settle the real state.
      .catch(() => adoptJob(req));
  }
}

export const downloadManager = {
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
