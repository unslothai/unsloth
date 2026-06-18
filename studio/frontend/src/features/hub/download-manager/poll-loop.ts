// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { invalidateGgufVariantsCache } from "../inventory/api";
import { getHfToken } from "../stores/hf-token-store";
import { bumpInventoryVersion } from "../stores/inventory-events";
import { toast } from "@/lib/toast";
import {
  getActiveModelDownloads,
  getDatasetDownloadStatus,
  type ActiveModelDownload,
  type DownloadJobState,
} from "./api";
import {
  CANCELLED_LINGER_MS,
  CANCEL_WATCHDOG_MS,
  COMPLETE_LINGER_MS,
  ERROR_LINGER_MS,
  HIDDEN_POLL_INTERVAL_MS,
  IDLE_EVICT_GRACE_MS,
  INVENTORY_BUMP_DEBOUNCE_MS,
  MAX_PROGRESS_FRACTION,
  POLL_BACKOFF_AFTER_MS,
  POLL_BACKOFF_INTERVAL_MS,
  POLL_DEGRADED_AFTER_MS,
  POLL_DEGRADED_MESSAGE,
  POLL_INTERVAL_MS,
  POLL_JITTER_MS,
  PROGRESS_POLL_BACKOFF_INTERVAL_MS,
  PROGRESS_POLL_INTERVAL_MS,
  SPEED_EMA_WEIGHT,
  ACTIVE_STATES,
  TERMINAL_DISPLAY_STATES,
} from "./download-manager-config";
import {
  DOWNLOAD_KIND,
  TRANSPORT,
  type DownloadKind,
  type TransportMode,
} from "./constants";
import {
  apiCancel,
  apiCancelRequest,
  apiGetProgress,
  apiGetStatus,
  apiStart,
  describeUnacceptedStart,
  effectiveTransportMode,
  normalizeDownloadError,
  pollAccessErrorMessage,
  withPollRequestTimeout,
} from "./download-api-adapter";
import type {
  DownloadRequest,
  JobListeners,
  JobRuntime,
  ManagedDownload,
  ProgressLike,
  Terminal,
} from "./download-manager-types";
import {
  getState,
  hasActiveRepoPeer,
  isCurrent,
  jobKeyOf,
  patchJob,
  putJob,
  removeJob,
  repoKeyOf,
  scheduleRemoval,
  setExpectedBytesForJob,
} from "./download-manager-state";
import {
  clearWatchdog,
  runtimeRegistry,
  teardownRuntime,
} from "./runtime-registry";
import { getTransportMode } from "./transport-preference";

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

function withDownloadTimeout<T>(
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  return withPollRequestTimeout(null, request);
}

export function hasObservedExpectedBytes(job: ManagedDownload): boolean {
  // Finalized bytes only: an `.incomplete` blob hitting expected size isn't
  // finished until the backend verifies it's usable on disk.
  return (
    job.expectedBytes > 0 &&
    job.completedBytes >= job.expectedBytes &&
    job.completeOnDisk
  );
}

export function resolveProgressUpdate(
  job: ManagedDownload,
  progressResp: ProgressLike,
  opts: { resetMonotonic?: boolean } = {},
): {
  expected: number;
  downloadedBytes: number;
  completedBytes: number;
  completeOnDisk: boolean;
  fraction: number;
  madeProgress: boolean;
} {
  const reported = progressResp.expected_bytes;
  const isGgufVariantJob =
    job.kind === DOWNLOAD_KIND.MODEL && job.variant !== null;
  const backendOwnsGgufProgress = isGgufVariantJob && reported > 0;
  // GGUF totals are backend-owned (non-monotonic); snapshots stay monotonic to
  // absorb jitter, but a generation bump (XET redownload, restart, re-adoption)
  // must drop the stale high-water mark and snap to the new run's bytes.
  const trustBackend = backendOwnsGgufProgress || opts.resetMonotonic === true;
  const expected = trustBackend
    ? reported > 0
      ? reported
      : job.expectedBytes
    : Math.max(reported > 0 ? reported : job.expectedBytes, job.expectedBytes);
  const previousDownloadedBytes = job.downloadedBytes;
  const downloadedBytes = trustBackend
    ? Math.max(0, progressResp.downloaded_bytes)
    : Math.max(previousDownloadedBytes, progressResp.downloaded_bytes);
  const completedBytes = trustBackend
    ? Math.max(0, progressResp.completed_bytes ?? 0)
    : Math.max(job.completedBytes, progressResp.completed_bytes ?? 0);
  const completeOnDisk = progressResp.complete_on_disk === true;
  const madeProgress =
    opts.resetMonotonic === true ||
    downloadedBytes > previousDownloadedBytes ||
    expected !== job.expectedBytes;
  const rawFraction =
    progressResp.progress > 0
      ? progressResp.progress
      : expected > 0
        ? downloadedBytes / expected
        : 0;
  const cappedFraction = Math.min(rawFraction, MAX_PROGRESS_FRACTION);
  // Keep the GGUF variant bar monotonic: backend progress is recomputed from the
  // shared per-repo blobs/ dir, so a sibling quant, generation bump, or
  // no-metadata poll can dip one reading. Resets via startJob's seed fraction.
  const fraction = isGgufVariantJob
    ? Math.max(cappedFraction, job.fraction)
    : cappedFraction;
  return {
    expected,
    downloadedBytes,
    completedBytes,
    completeOnDisk,
    fraction,
    madeProgress,
  };
}

export function applyProgressUpdate(
  key: string,
  job: ManagedDownload,
  progressResp: ProgressLike,
): ReturnType<typeof resolveProgressUpdate> {
  const resolved = resolveProgressUpdate(job, progressResp);
  patchJob(key, {
    expectedBytes: resolved.expected,
    downloadedBytes: resolved.downloadedBytes,
    completedBytes: resolved.completedBytes,
    completeOnDisk: resolved.completeOnDisk,
    fraction: resolved.fraction,
  });
  return resolved;
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

function markPollSuccess(key: string, rt: JobRuntime): void {
  rt.pollFailureStartedAt = null;
  const job = getState().jobs[key];
  if (job?.error === POLL_DEGRADED_MESSAGE && ACTIVE_STATES.has(job.state)) {
    patchJob(key, { error: null });
  }
}

function markPollFailure(key: string, rt: JobRuntime): void {
  const now = Date.now();
  rt.pollFailureStartedAt ??= now;
  if (now - rt.pollFailureStartedAt < POLL_DEGRADED_AFTER_MS) return;
  rt.speedSample = null;
  patchJob(key, {
    error: POLL_DEGRADED_MESSAGE,
    bytesPerSec: 0,
  });
}

export function finalize(
  key: string,
  outcome: Terminal,
  opts: { bytes?: number; error?: string | null } = {},
): void {
  const job = getState().jobs[key];
  teardownRuntime(key);
  if (!job) return;
  if (TERMINAL_DISPLAY_STATES.has(job.state)) return;
  if (job.kind === DOWNLOAD_KIND.MODEL) {
    invalidateGgufVariantsCache(job.repoId);
  }
  if (outcome === "gone") {
    notify(job, "onCancelled", 0);
    removeJob(key);
  } else if (outcome === "complete") {
    // A terminal "complete" arriving before the final progress poll must not
    // leave a stale sub-total. Reconcile to the largest known figure so
    // downloaded == completed >= expected, fraction 1, and report that.
    const bytes = Math.max(
      opts.bytes ?? 0,
      job.downloadedBytes,
      job.completedBytes,
      job.expectedBytes,
    );
    patchJob(key, {
      state: "complete",
      fraction: 1,
      downloadedBytes: bytes,
      completedBytes: bytes,
      completeOnDisk: true,
      bytesPerSec: 0,
      error: null,
    });
    notify(job, "onComplete", bytes);
    scheduleRemoval(key, COMPLETE_LINGER_MS);
  } else if (outcome === "cancelled") {
    patchJob(key, { state: "cancelled", bytesPerSec: 0, error: null });
    notify(job, "onCancelled", 0);
    scheduleRemoval(key, CANCELLED_LINGER_MS);
  } else {
    const rawError =
      typeof opts.error === "string" && opts.error
        ? opts.error
        : "Download failed";
    patchJob(key, {
      state: "error",
      error:
        opts.error === null ? null : (pollAccessErrorMessage(rawError) ?? rawError),
      bytesPerSec: 0,
    });
    notify(job, "onError", 0);
    scheduleRemoval(key, ERROR_LINGER_MS);
  }
  scheduleInventoryBump();
}

type PollStatus = Awaited<ReturnType<typeof apiGetStatus>>;

function terminalKindFromState(state: DownloadJobState): Terminal | null {
  return state === "complete"
    ? "complete"
    : state === "error"
      ? "error"
      : state === "cancelled"
        ? "cancelled"
        : null;
}

function syncServerGeneration(
  key: string,
  job: ManagedDownload,
  status: PollStatus,
): boolean {
  const statusGeneration = status.generation;
  const previousGeneration = job.serverGeneration;
  const generationChanged =
    typeof statusGeneration === "number" &&
    Number.isSafeInteger(statusGeneration) &&
    typeof previousGeneration === "number" &&
    Number.isSafeInteger(previousGeneration) &&
    statusGeneration !== previousGeneration;
  if (
    typeof statusGeneration === "number" &&
    Number.isSafeInteger(statusGeneration)
  ) {
    patchJob(key, { serverGeneration: statusGeneration });
  }
  return generationChanged;
}

async function finalizeTerminalStatus(
  key: string,
  terminalKind: Terminal,
  status: PollStatus,
  abort: AbortController | null,
  epoch: number,
): Promise<void> {
  const current = getState().jobs[key];
  if (!current) return;
  if (terminalKind === "complete") {
    let finalBytes = current.downloadedBytes;
    try {
      const progressResp = await withPollRequestTimeout(abort, (signal) =>
        apiGetProgress(current, signal),
      );
      if (!isCurrent(key, epoch)) return;
      const latest = getState().jobs[key];
      if (latest) {
        const { downloadedBytes } = applyProgressUpdate(key, latest, progressResp);
        finalBytes = downloadedBytes;
      }
    } catch {
      // Terminal status is authoritative; progress reconciliation is best-effort.
    }
    finalize(key, "complete", { bytes: finalBytes });
  } else if (terminalKind === "error") {
    finalize(key, "error", { error: status.error ?? null });
  } else {
    finalize(key, "cancelled");
  }
}

function applySpeedSample(
  rt: JobRuntime,
  current: ManagedDownload,
  downloadedBytes: number,
  nowMs: number,
): number {
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
  return bytesPerSec;
}

function reconcileProgressAndSpeed(
  rt: JobRuntime,
  key: string,
  current: ManagedDownload,
  progressResp: ProgressLike,
  generationChanged: boolean,
): { madeProgress: boolean } {
  const { expected, downloadedBytes, completedBytes, completeOnDisk, fraction, madeProgress } =
    resolveProgressUpdate(current, progressResp, {
      resetMonotonic: generationChanged,
    });
  const bytesPerSec = applySpeedSample(rt, current, downloadedBytes, Date.now());
  patchJob(key, {
    expectedBytes: expected,
    downloadedBytes,
    completedBytes,
    completeOnDisk,
    fraction,
    bytesPerSec,
  });
  markPollSuccess(key, rt);
  return { madeProgress };
}

function handleIdleAfterProgress(
  rt: JobRuntime,
  key: string,
  madeProgress: boolean,
): void {
  const updatedJob = getState().jobs[key];
  if (updatedJob && hasObservedExpectedBytes(updatedJob)) {
    finalize(key, "complete", { bytes: updatedJob.downloadedBytes });
  } else if (rt.cancelRequested) {
    finalize(key, "cancelled");
  } else if (madeProgress) {
    rt.idleSinceMs = null;
  } else {
    rt.idleSinceMs ??= Date.now();
    if (Date.now() - rt.idleSinceMs >= IDLE_EVICT_GRACE_MS) {
      finalize(key, "gone");
    }
  }
}

function handleTickError(
  key: string,
  rt: JobRuntime,
  epoch: number,
  error: unknown,
): void {
  if (!isCurrent(key, epoch)) return;
  const accessMessage = pollAccessErrorMessage(
    error instanceof Error ? error.message : String(error ?? ""),
  );
  if (accessMessage) {
    finalize(key, "error", { error: accessMessage });
    return;
  }
  markPollFailure(key, rt);
}

async function tick(key: string): Promise<void> {
  const rt = runtimeRegistry.runtimes.get(key);
  if (!rt) return;
  const job = getState().jobs[key];
  if (!job) {
    teardownRuntime(key);
    return;
  }
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
    if (!isCurrent(key, epoch)) return;

    const generationChanged = syncServerGeneration(key, job, status);

    const terminalKind = terminalKindFromState(status.state);
    if (terminalKind !== null) {
      await finalizeTerminalStatus(key, terminalKind, status, abort, epoch);
      return;
    }

    if (status.state === "idle") {
      const observedJob = getState().jobs[key];
      if (observedJob && hasObservedExpectedBytes(observedJob)) {
        finalize(key, "complete", { bytes: observedJob.downloadedBytes });
        return;
      }
    }

    const forceProgress = status.state === "idle";
    if (!shouldPollProgress(rt, forceProgress)) {
      rt.idleSinceMs = null;
      markPollSuccess(key, rt);
      return;
    }
    rt.lastProgressPollAt = Date.now();
    const jobForProgress = getState().jobs[key] ?? job;
    const progressResp = await withPollRequestTimeout(abort, (signal) =>
      apiGetProgress(jobForProgress, signal),
    );
    if (!isCurrent(key, epoch)) return;
    const current = getState().jobs[key];
    if (!current) return;

    const { madeProgress } = reconcileProgressAndSpeed(
      rt,
      key,
      current,
      progressResp,
      generationChanged,
    );

    if (status.state === "idle") {
      handleIdleAfterProgress(rt, key, madeProgress);
    } else {
      rt.idleSinceMs = null;
    }
  } catch (error) {
    handleTickError(key, rt, epoch, error);
  } finally {
    rt.inFlight = false;
  }
}

function pollDelayMs(rt: JobRuntime): number {
  if (typeof document !== "undefined" && document.hidden) {
    return HIDDEN_POLL_INTERVAL_MS;
  }
  const elapsedMs = Date.now() - rt.pollStartedAt;
  const base =
    elapsedMs >= POLL_BACKOFF_AFTER_MS
      ? POLL_BACKOFF_INTERVAL_MS
      : POLL_INTERVAL_MS;
  const jitter = Math.round((Math.random() * 2 - 1) * POLL_JITTER_MS);
  return Math.max(100, base + jitter);
}

async function runPollTick(key: string): Promise<void> {
  const startedRuntime = runtimeRegistry.runtimes.get(key);
  const epoch = startedRuntime?.epoch;
  try {
    await tick(key);
  } finally {
    const rt = runtimeRegistry.runtimes.get(key);
    if (
      startedRuntime !== undefined &&
      epoch !== undefined &&
      rt === startedRuntime &&
      isCurrent(key, epoch)
    ) {
      rt.pollTimer = window.setTimeout(
        () => void runPollTick(key),
        pollDelayMs(rt),
      );
    }
  }
}

function beginPolling(key: string, rt: JobRuntime): void {
  rt.pollingStarted = true;
  if (typeof document !== "undefined") {
    const epoch = rt.epoch;
    rt.visibilityListener = () => {
      if (document.hidden || !isCurrent(key, epoch)) return;
      const live = runtimeRegistry.runtimes.get(key);
      if (live !== rt) return;
      if (live.inFlight) return;
      if (live.pollTimer != null) {
        window.clearTimeout(live.pollTimer);
        live.pollTimer = null;
      }
      void runPollTick(key);
    };
    document.addEventListener("visibilitychange", rt.visibilityListener);
  }
  void runPollTick(key);
}

function reissueDroppedStartCancel(
  req: DownloadRequest,
  generation: number | undefined,
): void {
  if (!Number.isSafeInteger(generation)) {
    return;
  }
  void withDownloadTimeout(async (signal) => {
    await apiCancelRequest(req, generation, signal);
  }).catch(() => {});
}

export async function startJob(
  req: DownloadRequest,
  opts: {
    adopt?: boolean;
    useXet?: boolean;
    generation?: number;
    state?: DownloadJobState;
  } = {},
): Promise<void> {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  // Peer guard stops a FRESH start from double-starting a variant already
  // downloading (or colliding with a no-variant snapshot). Skipped when ADOPTING:
  // the restored own entry would look like a peer and freeze the bar; adoptJob's
  // `pollingStarted` guard already prevents double-polling the same key.
  if (!opts.adopt && hasActiveRepoPeer(req.kind, req.repoId, key, req.variant)) {
    return;
  }
  const nextEpoch = (runtimeRegistry.runtimes.get(key)?.epoch ?? 0) + 1;
  teardownRuntime(key);
  const existing = getState().jobs[key];
  const adoptingCancel =
    opts.adopt === true &&
    (opts.state === "cancelling" ||
      (opts.state === undefined && existing?.state === "cancelling"));
  const rt: JobRuntime = {
    kind: req.kind,
    repoId: req.repoId,
    epoch: nextEpoch,
    pollTimer: null,
    pollStartedAt: Date.now(),
    pollingStarted: false,
    abort: new AbortController(),
    inFlight: false,
    cancelRequested: adoptingCancel,
    watchdog: null,
    speedSample: null,
    idleSinceMs: null,
    lastProgressPollAt: null,
    pollFailureStartedAt: null,
    visibilityListener: null,
  };
  runtimeRegistry.runtimes.set(key, rt);
  const epoch = rt.epoch;

  const expected = Math.max(existing?.expectedBytes ?? 0, req.expectedBytes);
  const hfToken = getHfToken() || null;
  const requestedUseXet = opts.useXet ?? (getTransportMode() === TRANSPORT.XET);
  const requestedMode: TransportMode = requestedUseXet
    ? TRANSPORT.XET
    : TRANSPORT.HTTP;
  let mode: TransportMode;
  try {
    mode = opts.adopt
      ? TRANSPORT.HTTP
      : await effectiveTransportMode(requestedMode);
  } catch (error) {
    teardownRuntime(key);
    throw error;
  }
  const seedDownloaded = opts.adopt ? (existing?.downloadedBytes ?? 0) : 0;
  const seedCompleted = opts.adopt ? (existing?.completedBytes ?? 0) : 0;
  const seedFraction = opts.adopt ? (existing?.fraction ?? 0) : 0;
  // An adopted job never called apiStart, so it learns the run's generation from
  // the probe (or persisted value) to scope a later cancel to this exact run.
  const seedGeneration = opts.adopt
    ? Number.isSafeInteger(opts.generation)
      ? opts.generation
      : existing?.serverGeneration
    : undefined;
  if (!opts.adopt && hasActiveRepoPeer(req.kind, req.repoId, key, req.variant)) {
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
    completedBytes: seedCompleted,
    completeOnDisk: false,
    expectedBytes: expected,
    fraction: seedFraction,
    bytesPerSec: 0,
    error: null,
    startedAt: opts.adopt ? (existing?.startedAt ?? Date.now()) : Date.now(),
    ...(Number.isSafeInteger(seedGeneration)
      ? { serverGeneration: seedGeneration }
      : {}),
  });

  if (!opts.adopt) {
    let result;
    try {
      result = await apiStart(req, mode === TRANSPORT.XET, hfToken);
    } catch (err) {
      if (!isCurrent(key, epoch)) return;
      finalize(key, "error", {
        error: normalizeDownloadError(err),
      });
      return;
    }
    // A cancel during this apiStart round-trip can land before the job is
    // claimable; re-issue against the accepted generation.
    if (rt.cancelRequested && result.accepted) {
      reissueDroppedStartCancel(req, result.generation);
    }
    if (!isCurrent(key, epoch)) return;
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

function armCancelWatchdog(
  key: string,
  rt: JobRuntime,
  cancelEpoch: number,
): void {
  rt.watchdog = window.setTimeout(() => {
    void resolveCancelWatchdog(key, cancelEpoch);
  }, CANCEL_WATCHDOG_MS);
}

async function resolveCancelWatchdog(
  key: string,
  cancelEpoch: number,
): Promise<void> {
  const rt = runtimeRegistry.runtimes.get(key);
  if (!rt || rt.epoch !== cancelEpoch || !rt.cancelRequested) return;
  const job = getState().jobs[key];
  if (!job) return;
  const probe = await probeCancelOutcome(key, job, rt, cancelEpoch);
  if (probe === "stale") return;
  const live = runtimeRegistry.runtimes.get(key);
  if (!live || live.epoch !== cancelEpoch || !live.cancelRequested) return;
  if (probe.terminal === "complete") {
    finalize(key, "complete", { bytes: getState().jobs[key]?.downloadedBytes ?? 0 });
  } else if (probe.terminal === "error") {
    finalize(key, "error", { error: probe.error });
  } else {
    finalize(key, "cancelled");
  }
}

function applyCancelResult(
  key: string,
  cancelEpoch: number,
  result: { state: DownloadJobState },
): void {
  const live = runtimeRegistry.runtimes.get(key);
  if (live && live.epoch !== cancelEpoch) return;
  if (result.state === "cancelling" || result.state === "cancelled") {
    if (!live || !live.pollingStarted) finalize(key, "cancelled");
    return;
  }
  if (live) {
    if (result.state !== "idle") {
      live.cancelRequested = false;
    }
    clearWatchdog(live);
  }
  if (result.state === "complete") {
    finalize(key, "complete", {
      bytes: getState().jobs[key]?.downloadedBytes ?? 0,
    });
  } else if (result.state === "error") {
    finalize(key, "error");
  } else if (result.state === "idle") {
    const current = getState().jobs[key];
    finalize(
      key,
      current && hasObservedExpectedBytes(current) ? "complete" : "cancelled",
    );
  } else if (live?.pollingStarted) {
    patchJob(key, { state: "running" });
    toast.error("Couldn't cancel the download. It's still running.");
  } else {
    finalize(key, "cancelled");
  }
}

type CancelProbeResult = { terminal: Terminal | null; error: string | null };

async function probeCancelOutcome(
  key: string,
  job: ManagedDownload,
  rt: JobRuntime | undefined,
  cancelEpoch: number,
): Promise<CancelProbeResult | "stale"> {
  try {
    const probe = await withDownloadTimeout((signal) => apiGetStatus(job, signal));
    if (probe.state === "complete") {
      return { terminal: "complete", error: null };
    }
    if (probe.state === "error") {
      return { terminal: "error", error: probe.error ?? null };
    }
    if (probe.state === "cancelled") {
      return { terminal: "cancelled", error: null };
    }
    if (probe.state === "idle") {
      const current = getState().jobs[key];
      if (current && hasObservedExpectedBytes(current)) {
        return { terminal: "complete", error: null };
      }
      if (current) {
        try {
          const progressResp = await withDownloadTimeout((signal) =>
            apiGetProgress(current, signal),
          );
          const liveAfterProgress = runtimeRegistry.runtimes.get(key);
          if (rt && liveAfterProgress && liveAfterProgress.epoch !== cancelEpoch) {
            return "stale";
          }
          const latest = getState().jobs[key];
          if (latest) {
            applyProgressUpdate(key, latest, progressResp);
            const updated = getState().jobs[key];
            return {
              terminal:
                updated && hasObservedExpectedBytes(updated)
                  ? "complete"
                  : "cancelled",
              error: null,
            };
          }
          return { terminal: "cancelled", error: null };
        } catch {
          return { terminal: "cancelled", error: null };
        }
      }
      return { terminal: "cancelled", error: null };
    }
    return { terminal: null, error: null };
  } catch {
    return { terminal: null, error: null };
  }
}

export async function cancelJob(key: string): Promise<void> {
  const job = getState().jobs[key];
  if (!job) return;
  const rt = runtimeRegistry.runtimes.get(key);
  const cancelEpoch = rt?.epoch ?? 0;
  if (rt) rt.cancelRequested = true;
  patchJob(key, { state: "cancelling" });
  clearWatchdog(rt);
  if (rt) armCancelWatchdog(key, rt, cancelEpoch);
  try {
    const result = await withDownloadTimeout<{ state: DownloadJobState }>(
      (signal) => apiCancel(job, signal),
    );
    applyCancelResult(key, cancelEpoch, result);
  } catch (err) {
    const liveAtError = runtimeRegistry.runtimes.get(key);
    if (rt && liveAtError && liveAtError.epoch !== cancelEpoch) return;
    // apiCancel failed; the probe below is authoritative. Disarm the watchdog so
    // it can't finalize "cancelled" mid-probe and tear down a still-running worker.
    clearWatchdog(liveAtError);

    const probe = await probeCancelOutcome(key, job, rt, cancelEpoch);
    if (probe === "stale") return;

    const live = runtimeRegistry.runtimes.get(key);
    if (rt && (!live || live.epoch !== cancelEpoch)) return;

    if (probe.terminal !== null) {
      if (probe.terminal === "complete") {
        const current = getState().jobs[key];
        finalize(key, "complete", { bytes: current?.downloadedBytes ?? 0 });
      } else if (probe.terminal === "error") {
        finalize(key, "error", { error: probe.error });
      } else {
        finalize(key, "cancelled");
      }
      return;
    }

    if (live) {
      live.cancelRequested = false;
    }
    patchJob(key, { state: "running" });
    toast.error("Couldn't cancel the download. It's still running.");
    console.warn("Failed to cancel download", err);
  }
}

export function adoptJob(
  req: DownloadRequest,
  generation?: number,
  state?: DownloadJobState,
): void {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  if (runtimeRegistry.runtimes.get(key)?.pollingStarted) return;
  void startJob(req, { adopt: true, generation, state });
}

type ProbeAndAdoptOptions = {
  includeVariants?: boolean;
  fresh?: boolean;
  onModelAdopt?: (download: ActiveModelDownload) => void;
};

export async function probeAndAdopt(
  kind: DownloadKind,
  repoId: string,
  signal: AbortSignal,
  options: ProbeAndAdoptOptions = {},
): Promise<void> {
  try {
    if (kind === DOWNLOAD_KIND.MODEL) {
      const downloads = await getActiveModelDownloads(repoId, signal, {
        fresh: options.fresh,
      });
      if (signal.aborted) return;
      const activeDownloads = downloads.filter(
        (download) =>
          (options.includeVariants || download.variant === null) &&
          (download.state === "running" || download.state === "cancelling"),
      );
      for (const active of activeDownloads) {
        options.onModelAdopt?.(active);
        adoptJob(
          { kind, repoId, variant: active.variant, expectedBytes: 0 },
          active.generation,
          active.state,
        );
      }
      return;
    }

    const status = await getDatasetDownloadStatus(repoId, signal);
    if (signal.aborted) return;
    if (status.state === "running" || status.state === "cancelling") {
      adoptJob(
        { kind, repoId, variant: null, expectedBytes: 0 },
        status.generation,
        status.state,
      );
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

export function setExpected(
  kind: DownloadKind,
  repoId: string,
  variant: string | null,
  bytes: number,
): void {
  setExpectedBytesForJob(kind, repoId, variant, bytes);
}
