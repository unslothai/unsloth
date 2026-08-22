// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import {
  signalRunningLlamaJob,
  subscribeToLlamaJobStarted,
} from "@/lib/llama-job-events";
import {
  boundedLlamaStatusRequest,
  llamaStatusRequestIsStale,
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
} from "@/lib/llama-job-lifecycle";
import { useCallback, useEffect, useRef, useState } from "react";

// Initial check plus hourly reminders until dismissed or applied.
const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
// Snooze checks sooner than the hourly reminder.
const SNOOZE_DELAY_MS = 15 * 60 * 1000; // ~15 minutes
// Poll fast enough to catch installer progress milestones.
const JOB_POLL_INTERVAL_MS = 500;

export interface LlamaUpdateJob {
  job_id: string | null;
  state: "idle" | "running" | "success" | "error";
  operation: "update" | "switch" | null;
  requested_backend: "auto" | "cpu" | "cuda" | "rocm" | "vulkan" | null;
  message: string;
  from_tag: string | null;
  to_tag: string | null;
  reload_required: boolean | null;
  error: string | null;
  // Download fraction while running, 1 on success.
  progress: number | null;
  // Identifies the accepted job when notifying other surfaces and tabs.
  started_at: string | null;
  // Set once the job leaves "running"; identifies a completed job so a
  // repeated fetch of the same success can be told apart from the next one.
  finished_at: string | null;
}

export interface LlamaUpdateStatus {
  supported: boolean;
  update_available: boolean;
  component: "llama.cpp" | "whisper.cpp";
  installed_tag: string | null;
  latest_tag: string | null;
  // Prebuilt download size in bytes, if known.
  update_size_bytes: number | null;
  job: LlamaUpdateJob;
}

function parseJob(value: unknown): LlamaUpdateJob {
  const job = (value ?? {}) as Record<string, unknown>;
  return {
    job_id: typeof job.job_id === "string" ? job.job_id : null,
    state: (job.state as LlamaUpdateJob["state"]) ?? "idle",
    operation:
      job.operation === "update" || job.operation === "switch"
        ? job.operation
        : null,
    requested_backend:
      job.requested_backend === "auto" ||
      job.requested_backend === "cpu" ||
      job.requested_backend === "cuda" ||
      job.requested_backend === "rocm" ||
      job.requested_backend === "vulkan"
        ? job.requested_backend
        : null,
    message: typeof job.message === "string" ? job.message : "",
    from_tag: typeof job.from_tag === "string" ? job.from_tag : null,
    to_tag: typeof job.to_tag === "string" ? job.to_tag : null,
    reload_required:
      typeof job.reload_required === "boolean" ? job.reload_required : null,
    error: typeof job.error === "string" ? job.error : null,
    progress: typeof job.progress === "number" ? job.progress : null,
    started_at: typeof job.started_at === "string" ? job.started_at : null,
    finished_at: typeof job.finished_at === "string" ? job.finished_at : null,
  };
}

function llamaJobMarker(job: LlamaUpdateJob): string {
  // Fall back for backends predating job_id.
  return (
    job.job_id ??
    JSON.stringify([
      job.operation,
      job.requested_backend,
      job.started_at,
      job.finished_at,
      job.from_tag,
      job.to_tag,
    ])
  );
}

function parseStatus(value: unknown): LlamaUpdateStatus | null {
  if (!value || typeof value !== "object") return null;
  const s = value as Record<string, unknown>;
  const component =
    s.update_component === "whisper" ? "whisper.cpp" : "llama.cpp";
  const whisper =
    s.whisper && typeof s.whisper === "object"
      ? (s.whisper as Record<string, unknown>)
      : null;
  // Legacy top-level version fields intentionally retain their llama meaning.
  // A whisper-only update must display the nested whisper release instead of
  // presenting equal llama tags as a new llama update.
  const details = component === "whisper.cpp" && whisper ? whisper : s;
  return {
    supported: s.supported === true,
    update_available: s.update_available === true,
    component,
    installed_tag:
      typeof details.installed_tag === "string" ? details.installed_tag : null,
    latest_tag:
      typeof details.latest_tag === "string" ? details.latest_tag : null,
    update_size_bytes:
      typeof details.update_size_bytes === "number"
        ? details.update_size_bytes
        : null,
    job: parseJob(s.job),
  };
}

// The backend job persists as "success" until the next update starts (it's a
// single in-memory record, not per-tab), so a fresh mount -- a new tab, or a
// page reload of a tab that already resynced -- would otherwise replay the
// same completed job forever. Persist the handled marker outside React state
// so it survives both, and is shared across tabs in this browser.
const HANDLED_RELOAD_STORAGE_KEY = "unsloth_llama_update_reload_handled_at";

function getHandledReloadAt(): string | null {
  try {
    return localStorage.getItem(HANDLED_RELOAD_STORAGE_KEY);
  } catch {
    return null;
  }
}

function setHandledReloadAt(finishedAt: string | null): void {
  if (!finishedAt) return;
  try {
    localStorage.setItem(HANDLED_RELOAD_STORAGE_KEY, finishedAt);
  } catch {
    // storage unavailable
  }
}

async function fetchStatus(
  forceRefresh = false,
  signal?: AbortSignal,
): Promise<LlamaUpdateStatus | null> {
  if (!getAuthToken()) return null;
  const res = await authFetch(
    `/api/llama/update-status${forceRefresh ? "?force_refresh=true" : ""}`,
    { signal },
  );
  if (!res.ok) return null;
  return parseStatus(await res.json());
}

async function fetchJobStatus(
  signal?: AbortSignal,
): Promise<LlamaUpdateJob | null> {
  if (!getAuthToken()) return null;
  const res = await authFetch("/api/llama/update-job-status", { signal });
  if (!res.ok) return null;
  return parseJob(await res.json());
}

interface UseLlamaUpdateCheckOptions {
  enabled?: boolean;
  /**
   * Called when a completed update reports `reload_required` (i.e. it unloaded
   * the active model server-side). Consumers use it to resync the chat runtime
   * so the model selector drops to "select model" instead of pointing at a
   * model that now 400s on send. Fires for both this tab's own apply() and a
   * cross-tab update mirrored through the background poll.
   */
  onReloadRequired?: () => void;
}

export interface LlamaApplyResult {
  ok: boolean;
  tag?: string | null;
  reloadRequired?: boolean | null;
  error?: string | null;
}

interface SequencedLlamaUpdateStatus {
  requestId: number;
  status: LlamaUpdateStatus | null;
}

interface SequencedLlamaUpdateJob {
  requestId: number;
  job: LlamaUpdateJob | null;
}

/** Tracks llama.cpp update visibility and apply progress. */
export function useLlamaUpdateCheck({
  enabled = true,
  onReloadRequired,
}: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const snoozeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Serialize job polls within the active generation.
  const pollInFlightGeneration = useRef<number | null>(null);
  const pollGeneration = useRef(0);
  const terminalRecheckJob = useRef<string | null>(null);
  const statusRequestSequence = useRef(0);
  const latestAppliedStatusRequest = useRef(0);
  const surfaceIfAvailableRef = useRef<
    ((next: SequencedLlamaUpdateStatus) => void) | null
  >(null);
  // Read through a ref so startJobPoll stays stable (apply/surfaceIfAvailable
  // depend on it) while still calling the latest callback.
  const onReloadRequiredRef = useRef(onReloadRequired);
  useEffect(() => {
    onReloadRequiredRef.current = onReloadRequired;
  }, [onReloadRequired]);
  // Fires the callback once per completed job, whether this tab watched it run
  // or only saw the persisted "success" after the fact (e.g. another tab
  // applied it). Keyed by finished_at and seeded from localStorage so a fresh
  // mount (new tab, or a page reload of a tab that already resynced) doesn't
  // replay a job some tab already handled.
  const reloadNotifiedForRef = useRef<string | null>(getHandledReloadAt());

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Sequence all reads so older responses cannot overwrite newer ones.
  const requestStatus = useCallback(
    async (forceRefresh = false): Promise<SequencedLlamaUpdateStatus> => {
      const requestId = ++statusRequestSequence.current;
      const status = await fetchStatus(forceRefresh).catch(() => null);
      return { requestId, status };
    },
    [],
  );

  const requestJob = useCallback(async (): Promise<SequencedLlamaUpdateJob> => {
    const requestId = ++statusRequestSequence.current;
    const job = await boundedLlamaStatusRequest((signal) =>
      fetchJobStatus(signal),
    );
    return { requestId, job };
  }, []);

  // Shared by the poll path (this tab watched the job run), the surface path
  // (this tab only saw the persisted success), and apply()'s stale-click path
  // (the job came back embedded in a "not started" response) so none of them
  // can drop or double-fire the notification.
  const notifyReloadIfNeeded = useCallback(
    (
      job: Pick<LlamaUpdateJob, "state" | "reload_required" | "finished_at">,
    ) => {
      // "error" is included for partial chained updates: the llama phase can
      // land (and unload the server) before a later phase fails, and the
      // backend keeps reload_required set in exactly that case. Without the
      // resync the chat UI would keep pointing at the unloaded model.
      if (
        (job.state === "success" || job.state === "error") &&
        job.reload_required &&
        job.finished_at !== reloadNotifiedForRef.current
      ) {
        reloadNotifiedForRef.current = job.finished_at;
        setHandledReloadAt(job.finished_at);
        onReloadRequiredRef.current?.();
      }
    },
    [],
  );

  // Used by apply() and another-tab job tracking.
  const startJobPoll = useCallback(
    (onDone?: (result: LlamaApplyResult) => void) => {
      clearPollTimer();
      const generation = ++pollGeneration.current;
      terminalRecheckJob.current = null;
      pollTimer.current = setInterval(async () => {
        if (pollInFlightGeneration.current === generation) return;
        pollInFlightGeneration.current = generation;
        try {
          const next = await requestJob();
          if (
            !next.job ||
            generation !== pollGeneration.current ||
            llamaStatusRequestIsStale(
              latestAppliedStatusRequest.current,
              next.requestId,
            )
          ) {
            return;
          }
          latestAppliedStatusRequest.current = next.requestId;
          const job = next.job;
          setStatus((current) => (current ? { ...current, job } : current));
          if (job.state === "running") {
            const switching = job.operation === "switch";
            setApplying(!switching);
            setVisible(!switching);
            terminalRecheckJob.current = null;
            return;
          }
          setApplying(false);
          if (job.state === "error") setVisible(true);
          const terminalJob = llamaJobMarker(job);
          const terminalAlreadyHandled =
            terminalRecheckJob.current === terminalJob;
          terminalRecheckJob.current = terminalJob;
          if (!terminalAlreadyHandled && job.state === "success") {
            void refreshHardwareInfo();
            // Resync chat after this or another tab unloads the active model.
            notifyReloadIfNeeded(job);
            onDone?.({
              ok: true,
              tag: job.to_tag,
              reloadRequired: job.reload_required,
            });
          } else if (!terminalAlreadyHandled && job.state === "error") {
            // Keep retry visible if a partial update unloaded the server.
            notifyReloadIfNeeded(job);
            onDone?.({ ok: false, error: job.error });
          } else if (!terminalAlreadyHandled) {
            onDone?.({ ok: false, error: "update did not complete" });
          }
          // Recompute availability; keep polling if reconciliation fails.
          const reconciled = await requestStatus();
          const surfaceReconciled = surfaceIfAvailableRef.current;
          if (
            !reconciled.status ||
            !surfaceReconciled ||
            generation !== pollGeneration.current ||
            llamaStatusRequestIsStale(
              latestAppliedStatusRequest.current,
              reconciled.requestId,
            )
          ) {
            return;
          }
          const reconciledPresentation = llamaUpdatePresentation(
            reconciled.status.update_available,
            reconciled.status.job,
          );
          surfaceReconciled(reconciled);
          if (generation !== pollGeneration.current) return;
          if (reconciledPresentation.running) {
            terminalRecheckJob.current = null;
            return;
          }
          if (
            llamaJobMarker(reconciled.status.job) !== terminalRecheckJob.current
          ) {
            // Let the next tick process a different terminal job.
            return;
          }
          pollGeneration.current += 1;
          clearPollTimer();
        } finally {
          if (pollInFlightGeneration.current === generation) {
            pollInFlightGeneration.current = null;
          }
        }
      }, JOB_POLL_INTERVAL_MS);
    },
    [clearPollTimer, notifyReloadIfNeeded, requestJob, requestStatus],
  );

  const surfaceIfAvailable = useCallback(
    (next: SequencedLlamaUpdateStatus) => {
      if (
        !next.status ||
        llamaStatusRequestIsStale(
          latestAppliedStatusRequest.current,
          next.requestId,
        )
      ) {
        return;
      }
      latestAppliedStatusRequest.current = next.requestId;
      const status = next.status;
      setStatus(status);
      const presentation = llamaUpdatePresentation(
        status.update_available,
        status.job,
      );
      setApplying(presentation.applying);
      setVisible(presentation.visible);
      if (presentation.running) {
        if (!pollTimer.current) startJobPoll();
        return;
      }
      // A completed job persists as "success" until the next update starts, so
      // a tab that missed the running window entirely (mounted, or only checks
      // hourly and misses both the running and just-finished moments) still
      // needs to resync here, not just from the poll path above.
      notifyReloadIfNeeded(status.job);
    },
    [startJobPoll, notifyReloadIfNeeded],
  );

  useEffect(() => {
    surfaceIfAvailableRef.current = surfaceIfAvailable;
  }, [surfaceIfAvailable]);

  useEffect(() => {
    if (!enabled) {
      // Re-enabling will rediscover any still-running job.
      return;
    }
    let canceled = false;

    const firstTimer = setTimeout(() => {
      requestStatus(true).then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, FIRST_CHECK_DELAY_MS);

    const reminder = setInterval(() => {
      requestStatus(true).then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, REMINDER_INTERVAL_MS);

    return () => {
      canceled = true;
      clearTimeout(firstTimer);
      clearInterval(reminder);
      clearPollTimer();
      if (snoozeTimer.current) {
        clearTimeout(snoozeTimer.current);
        snoozeTimer.current = null;
      }
    };
  }, [enabled, surfaceIfAvailable, clearPollTimer, requestStatus]);

  // Cross-tab nudge: a tab that only checks hourly would otherwise stay
  // pointed at a server-unloaded model for up to an hour after a DIFFERENT
  // open tab applies an update. The storage event only fires in other tabs
  // (never the one that wrote it), so this recheck fires promptly there
  // without this tab redundantly re-triggering itself.
  useEffect(() => {
    if (!enabled) return;
    const onStorage = (event: StorageEvent) => {
      if (
        event.key === HANDLED_RELOAD_STORAGE_KEY &&
        event.newValue &&
        event.newValue !== reloadNotifiedForRef.current
      ) {
        requestStatus(true).then(surfaceIfAvailable);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [enabled, surfaceIfAvailable, requestStatus]);

  useEffect(() => {
    if (!enabled) return;
    return subscribeToLlamaJobStarted(() => {
      requestStatus().then(surfaceIfAvailable);
    });
  }, [enabled, surfaceIfAvailable, requestStatus]);

  const dismiss = useCallback(() => {
    setVisible(false);
  }, []);

  const snooze = useCallback(() => {
    setVisible(false);
    if (snoozeTimer.current) clearTimeout(snoozeTimer.current);
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      requestStatus(true).then(surfaceIfAvailable);
    }, SNOOZE_DELAY_MS);
  }, [surfaceIfAvailable, requestStatus]);

  const apply = useCallback(async (): Promise<LlamaApplyResult> => {
    if (applying) return { ok: false, error: "already running" };
    setApplying(true);
    setVisible(true);
    let action: {
      started?: boolean;
      reason?: string | null;
      message?: string | null;
      job?: unknown;
    } | null = null;
    try {
      const res = await authFetch("/api/llama/update", { method: "POST" });
      if (!res.ok) {
        setApplying(false);
        return { ok: false, error: `HTTP ${res.status}` };
      }
      try {
        action = await res.json();
      } catch {
        action = null;
      }
    } catch (e) {
      setApplying(false);
      return { ok: false, error: String(e) };
    }

    const actionJob = parseJob(action?.job);
    // The response job is authoritative. Signal both a newly accepted update
    // and an already-running job this tab discovered through the POST, so every
    // open Settings surface disables and follows the same install immediately.
    signalRunningLlamaJob(actionJob);

    // Non-started jobs stay idle; an already-running update is tracked below.
    // A backend switch is not: it shares this job but installs no new release,
    // so following it here would toast an update that never happened. The
    // shared background listener still follows the switch itself.
    if (
      action &&
      action.started === false &&
      !llamaUpdateAdoptsRunningJob(action.reason, actionJob)
    ) {
      // A stale banner's click can land after another tab already applied the
      // update (e.g. "up_to_date"): the response still carries that tab's
      // completed job, so process reload_required here too, not just from the
      // poll path -- otherwise this rejection silently drops it.
      notifyReloadIfNeeded(actionJob);
      setApplying(false);
      return {
        ok: false,
        error: action.message ?? action.reason ?? "update was not started",
      };
    }

    return await new Promise<LlamaApplyResult>((resolve) =>
      startJobPoll(resolve),
    );
  }, [applying, startJobPoll, notifyReloadIfNeeded]);

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying: enabled && applying,
    apply,
    dismiss,
    snooze,
  };
}
