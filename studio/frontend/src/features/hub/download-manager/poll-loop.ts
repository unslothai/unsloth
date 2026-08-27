// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { carriesOverSeed, seededMeasuredTransfer } from "./adopt-rules";
import { invalidateGgufVariantsCache } from "../inventory/api";
import { getHfToken } from "../stores/hf-token-store";
import { bumpInventoryVersion } from "../stores/inventory-events";
import { toast } from "@/lib/toast";
import { appendSample, computeTransferStats } from "@/lib/transfer-stats";
import {
  getActiveDatasetDownloads,
  getActiveModelDownloads,
  type ActiveModelDownload,
  type DownloadJobState,
} from "./api";
import { cancelExternalJob, isExternalJob } from "./external-jobs";
import {
  CANCELLED_LINGER_MS,
  CANCEL_WATCHDOG_MS,
  COMPLETE_LINGER_MS,
  ERROR_LINGER_MS,
  HIDDEN_POLL_INTERVAL_MS,
  IDLE_EVICT_GRACE_MS,
  INVENTORY_BUMP_DEBOUNCE_MS,
  POLL_BACKOFF_AFTER_MS,
  POLL_BACKOFF_INTERVAL_MS,
  POLL_DEGRADED_AFTER_MS,
  POLL_DEGRADED_MESSAGE,
  POLL_INTERVAL_MS,
  POLL_JITTER_MS,
  PROGRESS_POLL_BACKOFF_INTERVAL_MS,
  PROGRESS_POLL_INTERVAL_MS,
  ACTIVE_STATES,
  TERMINAL_DISPLAY_STATES,
} from "./download-manager-config";
import {
  DOWNLOAD_KIND,
  TRANSPORT,
  type DownloadKind,
  type ResolvedTransport,
  type TransportMode,
  adoptedTransports,
  isResolvedTransport,
  probeDescribesCurrentRun,
  transportAfterStart,
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
  XET_NOTICE_TITLE,
  composeNoticeDescription,
  shouldShowXetNotice,
} from "./xet-progress-notice";
import { reserveXetNoticeFromServer } from "@/features/settings/api/xet-notice";
import {
  currentRoute,
  dismissStartToast,
  liveCallerToast,
  showCallerToast,
  showStartToast,
} from "./start-toast";
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
  hasObservedExpectedBytes,
  resolveProgressUpdate,
} from "./progress-reconcile";
import {
  clearWatchdog,
  runtimeRegistry,
  teardownRuntime,
} from "./runtime-registry";
import { resolveTransportMode } from "./transport-preference";

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

export { hasObservedExpectedBytes, resolveProgressUpdate };

export function applyProgressUpdate(
  key: string,
  job: ManagedDownload,
  progressResp: ProgressLike,
): ReturnType<typeof resolveProgressUpdate> {
  const resolved = resolveProgressUpdate(job, progressResp);
  patchJob(key, {
    expectedBytes: resolved.expected,
    downloadedBytes: resolved.downloadedBytes,
    measuredTransfer: resolved.measuredTransfer,
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
  rt.speedSamples.length = 0;
  patchJob(key, {
    error: POLL_DEGRADED_MESSAGE,
    bytesPerSec: 0,
    etaSeconds: 0,
  });
}

export function finalize(
  key: string,
  outcome: Terminal,
  opts: { bytes?: number; error?: string | null } = {},
): void {
  const job = getState().jobs[key];
  teardownRuntime(key);
  // The 8s duration is a cap, not a description of the transfer: a finished download
  // left the toast still claiming it was running. Before the early returns below, so
  // a job already in a terminal display state still clears it.
  dismissStartToast(key);
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
      etaSeconds: 0,
      error: null,
    });
    notify(job, "onComplete", bytes);
    scheduleRemoval(key, COMPLETE_LINGER_MS);
  } else if (outcome === "cancelled") {
    patchJob(key, {
      state: "cancelled",
      bytesPerSec: 0,
      etaSeconds: 0,
      error: null,
    });
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
      etaSeconds: 0,
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

// Rolling-window rate, withheld until the window is trustworthy. The old EMA
// published its first sample verbatim and decayed toward -- never reaching --
// zero while stalled, so ramp-up and idle ticks produced "753d 5h left" (#7667).
// 0 hides both labels, as the training-start overlay already does.
function applySpeedSample(
  rt: JobRuntime,
  downloadedBytes: number,
  expectedBytes: number,
  nowMs: number,
): { bytesPerSec: number; etaSeconds: number } {
  appendSample(rt.speedSamples, nowMs / 1000, downloadedBytes);
  const stats = computeTransferStats(rt.speedSamples, expectedBytes);
  return {
    bytesPerSec: stats.stable ? stats.rateBytesPerSecond : 0,
    etaSeconds: stats.stable ? stats.etaSeconds : 0,
  };
}

function reconcileProgressAndSpeed(
  rt: JobRuntime,
  key: string,
  current: ManagedDownload,
  progressResp: ProgressLike,
  generationChanged: boolean,
): { madeProgress: boolean } {
  const {
    expected,
    downloadedBytes,
    measuredTransfer,
    completedBytes,
    completeOnDisk,
    fraction,
    madeProgress,
  } = resolveProgressUpdate(current, progressResp, {
    resetMonotonic: generationChanged,
  });
  if (generationChanged) {
    // Another server owns this transfer now, so the old samples describe a
    // different run. The counter cannot say so: a restart resumes from the
    // same cache and never goes backwards for appendSample to catch.
    rt.speedSamples.length = 0;
  }
  const speed = applySpeedSample(rt, downloadedBytes, expected, Date.now());
  patchJob(key, {
    expectedBytes: expected,
    downloadedBytes,
    measuredTransfer,
    completedBytes,
    completeOnDisk,
    fraction,
    bytesPerSec: speed.bytesPerSec,
    etaSeconds: speed.etaSeconds,
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
    rt.speedSamples.length = 0;
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

    // syncServerGeneration persists the new generation immediately, so a change
    // seen on a tick that returns before the progress path would look unchanged
    // on the next one. Hold it until a progress poll actually consumes it.
    if (syncServerGeneration(key, job, status)) {
      rt.pendingGenerationChange = true;
    }

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

    const generationChanged = rt.pendingGenerationChange === true;
    rt.pendingGenerationChange = false;
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
    transport?: ResolvedTransport;
    cancelTransport?: ResolvedTransport | null;
    /** The surface this start was asked for, to hold its toast against. Passed by
     * `requestStart`, whose preflight is itself navigable; else taken here. */
    originRoute?: string;
  } = {},
): Promise<void> {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  const startRoute = opts.originRoute ?? currentRoute();
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
    speedSamples: [],
    idleSinceMs: null,
    lastProgressPollAt: null,
    pollFailureStartedAt: null,
    visibilityListener: null,
  };
  runtimeRegistry.runtimes.set(key, rt);
  const epoch = rt.epoch;

  const expected = Math.max(existing?.expectedBytes ?? 0, req.expectedBytes);
  const hfToken = getHfToken() || null;
  // An explicit opts.useXet (a retry pinning a transport) wins; otherwise carry the stored
  // preference UNRESOLVED so "auto" survives to effectiveTransportMode(). Collapsing it to a
  // boolean here would read "auto" as "not xet" and send every download over HTTP.
  // Never for an adopted job: that branch ignores requestedMode, and resolving it is now a
  // settings round trip. Suspending here left pollingStarted=false long enough for the other
  // concurrent adoptJob caller to replace this runtime, and both continuations then reached
  // beginPolling, leaving duplicate timers and a leaked visibility listener.
  const requestedMode: TransportMode = opts.adopt
    ? TRANSPORT.HTTP
    : opts.useXet === undefined
      ? await resolveTransportMode()
      : opts.useXet
        ? TRANSPORT.XET
        : TRANSPORT.HTTP;
  let mode: ResolvedTransport;
  try {
    mode = opts.adopt
      ? TRANSPORT.HTTP
      : await effectiveTransportMode(requestedMode);
  } catch (error) {
    teardownRuntime(key);
    throw error;
  }
  const carryOverSeed = carriesOverSeed(
    opts.adopt === true,
    existing?.serverGeneration,
    opts.generation,
  );
  const seedDownloaded = carryOverSeed ? (existing?.downloadedBytes ?? 0) : 0;
  const seedCompleted = carryOverSeed ? (existing?.completedBytes ?? 0) : 0;
  const seedFraction = carryOverSeed ? (existing?.fraction ?? 0) : 0;
  // Whatever the counters mean, they keep meaning it. Seeding the bytes without
  // this said "measured" for a figure the poll only held, which is the
  // "0 B left" the guard exists to stop.
  const seedMeasuredTransfer = seededMeasuredTransfer(
    carryOverSeed,
    existing?.measuredTransfer,
  );
  // An adopted job never called apiStart, so it learns the run's generation from
  // the probe (or persisted value) to scope a later cancel to this exact run.
  const seedGeneration = opts.adopt
    ? Number.isSafeInteger(opts.generation)
      ? opts.generation
      : existing?.serverGeneration
    : undefined;
  // An adopted job prefers what the backend just reported and falls back to
  // the persisted value, for the transport and its cancel marker alike.
  const adopted = opts.adopt
    ? adoptedTransports(
        { transport: opts.transport, cancelTransport: opts.cancelTransport },
        existing,
      )
    : { transport: mode, cancelTransport: undefined };
  const activeTransport = adopted.transport;
  if (!opts.adopt && hasActiveRepoPeer(req.kind, req.repoId, key, req.variant)) {
    teardownRuntime(key);
    return;
  }
  putJob({
    key,
    kind: req.kind,
    repoId: req.repoId,
    variant: req.variant,
    etaSeconds: 0,
    state: adoptingCancel ? "cancelling" : "running",
    downloadedBytes: seedDownloaded,
    completedBytes: seedCompleted,
    completeOnDisk: false,
    expectedBytes: expected,
    fraction: seedFraction,
    bytesPerSec: 0,
    error: null,
    startedAt: opts.adopt ? (existing?.startedAt ?? Date.now()) : Date.now(),
    // An adopted job prefers the backend's live transport, then a persisted
    // value. It never claims the HTTP placeholder used to skip resolution.
    ...(activeTransport ? { transport: activeTransport } : {}),
    // A fallback run's cancel marker. Only an adopted job can have one: the
    // fallback happens long after a start.
    ...(adopted.cancelTransport
      ? { cancelTransport: adopted.cancelTransport }
      : {}),
    ...(seedMeasuredTransfer !== undefined
      ? { measuredTransfer: seedMeasuredTransfer }
      : {}),
    ...(Number.isSafeInteger(seedGeneration)
      ? { serverGeneration: seedGeneration }
      : {}),
    // Recorded so a later start for the same scope slot can tell whether this running job is fetching its files or a different quant's. An adopted job keeps the existing record.
    ...(req.files && req.files.length > 0
      ? { scopedFiles: [...req.files] }
      : opts.adopt && existing?.scopedFiles
        ? { scopedFiles: existing.scopedFiles }
        : {}),
    // The staged plan's own verdict on which entry is the picked model, so the panel labels it without guessing from file extensions. An adopted job keeps the existing record, since only the stager can supply it.
    ...(req.checkpoint !== undefined
      ? { checkpoint: req.checkpoint }
      : opts.adopt && existing?.checkpoint !== undefined
        ? { checkpoint: existing.checkpoint }
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
    const started = transportAfterStart(mode, result.transport);
    if (started !== activeTransport) patchJob(key, { transport: started });
    // One start, one toast: the only place a start is announced. A cancel can land
    // mid-flight (the reissue above is the giveaway), and neither message is true
    // of a start that is already stopping.
    const stopping = rt.cancelRequested;
    // Everything above was round trips the user could navigate during. Checked BEFORE
    // reserving, since a reservation is one of three for the life of the install:
    // spending one on a toast that will be discarded on arrival is how starting a
    // download and going to watch it burns all three unseen.
    const onOriginRoute = currentRoute() === startRoute;
    if (
      onOriginRoute &&
      shouldShowXetNotice({
        kind: req.kind,
        transport: started,
        attached: result.attached === true,
        live: result.state === "running" && !stopping,
      })
    ) {
      // Async (it asks the backend for one of the three) and nothing below waits on
      // a toast. A lost reservation still leaves the caller owed its message.
      void reserveXetNoticeFromServer().then(({ granted }) => {
        // This round trip can outlive the transfer: finalize() dismisses by id before
        // it resolves, so raising here would leave a finished or cancelled job claiming
        // to run for another 8s, or hand a restart on the same key a stale message.
        if (!isCurrent(key, epoch) || rt.cancelRequested) return;
        // The caller's line can go stale while the notice stays true: chat moved
        // thread, so nothing auto-loads, but the 0% still needs explaining.
        const caller = liveCallerToast(req.callerToast);
        if (granted) {
          showStartToast(
            key,
            {
              title: XET_NOTICE_TITLE,
              description: composeNoticeDescription(caller),
            },
            startRoute,
          );
          return;
        }
        showCallerToast(key, caller, startRoute);
      });
    } else if (!stopping && onOriginRoute) {
      showCallerToast(key, liveCallerToast(req.callerToast), startRoute);
    }
    // An adopted job can already have fallen back from Xet to HTTP, which
    // keeps its original cancel marker and so its stop control.
    if (isResolvedTransport(result.cancel_transport)) {
      patchJob(key, { cancelTransport: result.cancel_transport });
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
  // Another subsystem owns this transfer; it does the cancelling.
  if (isExternalJob(key)) {
    await cancelExternalJob(key);
    return;
  }
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
  transport?: ResolvedTransport,
  // null is the backend reporting no marker, which must clear a stored one;
  // undefined is a caller that cannot report one at all.
  cancelTransport?: ResolvedTransport | null,
): void {
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  if (runtimeRegistry.runtimes.get(key)?.pollingStarted) {
    // Persistence hydration and the backend-active probe run concurrently, so a
    // late backend response must still replace a missing or stale stored value.
    // Only for the run it described, though: a cancel and restart in between
    // makes this a different job, possibly on the other transport.
    const known = getState().jobs[key]?.serverGeneration;
    if (transport && probeDescribesCurrentRun(known, generation)) {
      patchJob(key, {
        transport,
        ...(cancelTransport === undefined
          ? {}
          : { cancelTransport: cancelTransport ?? undefined }),
        ...(Number.isSafeInteger(known) ? {} : { serverGeneration: generation }),
      });
    }
    return;
  }
  void startJob(req, {
    adopt: true,
    generation,
    state,
    ...(transport ? { transport } : {}),
    ...(cancelTransport === undefined ? {} : { cancelTransport }),
  });
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
          {
            kind,
            repoId,
            variant: active.variant,
            expectedBytes: 0,
            // Carry the live job's own file list so the adopted record can be matched against a later start for the same slot.
            // Without it the adopted job had an unknown set and any sibling checkpoint's request read as "already started".
            ...(active.files && active.files.length > 0 ? { files: [...active.files] } : {}),
          },
          active.generation,
          active.state,
          isResolvedTransport(active.transport) ? active.transport : undefined,
          isResolvedTransport(active.cancel_transport)
            ? active.cancel_transport
            : null,
        );
      }
      return;
    }

    // The active-downloads list, not download-status: only the list reports the
    // transport, without which an adopted HTTP dataset shows Cancel for a
    // transfer that would have resumed.
    const datasets = await getActiveDatasetDownloads(signal, repoId);
    if (signal.aborted) return;
    // No repo compare here: the endpoint resolves the cached casing before it
    // filters, so an exact match against the card's spelling would drop the
    // very row it just asked for.
    for (const active of datasets) {
      if (active.state !== "running" && active.state !== "cancelling") continue;
      adoptJob(
        { kind, repoId, variant: null, expectedBytes: 0 },
        active.generation,
        active.state,
        isResolvedTransport(active.transport) ? active.transport : undefined,
        isResolvedTransport(active.cancel_transport)
          ? active.cancel_transport
          : null,
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
