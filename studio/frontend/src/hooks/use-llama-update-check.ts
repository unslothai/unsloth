// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import {
  signalRunningLlamaJob,
  subscribeToLlamaJobStarted,
} from "@/lib/llama-job-events";
import {
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
} from "@/lib/llama-job-lifecycle";
import { useCallback, useEffect, useRef, useState } from "react";

const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000;
const SNOOZE_DELAY_MS = 15 * 60 * 1000;
const JOB_POLL_INTERVAL_MS = 500;

export interface LlamaUpdateJob {
  state: "idle" | "running" | "success" | "error";
  operation: "update" | "switch" | null;
  requested_backend: "auto" | "cpu" | "cuda" | "rocm" | "vulkan" | null;
  message: string;
  from_tag: string | null;
  to_tag: string | null;
  reload_required: boolean | null;
  error: string | null;
  progress: number | null;
  started_at: string | null;
  // Tells a repeated fetch of the same success apart from the next one.
  finished_at: string | null;
}

export interface ComponentOffer {
  update_available: boolean;
  installed_tag: string | null;
  latest_tag: string | null;
  update_size_bytes: number | null;
}

export interface LlamaUpdateStatus {
  supported: boolean;
  update_available: boolean;
  source_build: boolean;
  component: "llama.cpp" | "whisper.cpp";
  // Both components can be behind at once.
  llama: ComponentOffer;
  whisper: ComponentOffer | null;
  installed_tag: string | null;
  latest_tag: string | null;
  update_size_bytes: number | null;
  // Recorded "auto" now resolves elsewhere; reported only when the release is current.
  backend_migration_available: boolean;
  from_backend: string | null;
  to_backend: string | null;
  job: LlamaUpdateJob;
}

export function llamaUpdateOffered(status: LlamaUpdateStatus): boolean {
  return status.update_available || status.backend_migration_available;
}

function parseJob(value: unknown): LlamaUpdateJob {
  const job = (value ?? {}) as Record<string, unknown>;
  return {
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

function parseStatus(value: unknown): LlamaUpdateStatus | null {
  if (!value || typeof value !== "object") return null;
  const s = value as Record<string, unknown>;
  const component =
    s.update_component === "whisper" ? "whisper.cpp" : "llama.cpp";
  const whisper =
    s.whisper && typeof s.whisper === "object"
      ? (s.whisper as Record<string, unknown>)
      : null;
  // Legacy top-level fields keep their llama meaning; a whisper-only update shows the nested release.
  const details = component === "whisper.cpp" && whisper ? whisper : s;
  return {
    supported: s.supported === true,
    update_available: s.update_available === true,
    source_build: s.source_build === true,
    component,
    installed_tag:
      typeof details.installed_tag === "string" ? details.installed_tag : null,
    latest_tag:
      typeof details.latest_tag === "string" ? details.latest_tag : null,
    update_size_bytes:
      typeof details.update_size_bytes === "number"
        ? details.update_size_bytes
        : null,
    llama: {
      // Absent on backends before the whisper piggyback, where the legacy union is llama's answer.
      update_available:
        typeof s.llama_update_available === "boolean"
          ? s.llama_update_available
          : s.update_available === true && component === "llama.cpp",
      installed_tag: typeof s.installed_tag === "string" ? s.installed_tag : null,
      latest_tag: typeof s.latest_tag === "string" ? s.latest_tag : null,
      update_size_bytes:
        typeof s.update_size_bytes === "number" ? s.update_size_bytes : null,
    },
    whisper: whisper
      ? {
          update_available: whisper.update_available === true,
          installed_tag:
            typeof whisper.installed_tag === "string"
              ? whisper.installed_tag
              : null,
          latest_tag:
            typeof whisper.latest_tag === "string" ? whisper.latest_tag : null,
          update_size_bytes:
            typeof whisper.update_size_bytes === "number"
              ? whisper.update_size_bytes
              : null,
        }
      : null,
    // The backend belongs to the llama.cpp install whatever component the versions describe.
    backend_migration_available: s.backend_migration_available === true,
    from_backend: typeof s.from_backend === "string" ? s.from_backend : null,
    to_backend: typeof s.to_backend === "string" ? s.to_backend : null,
    job: parseJob(s.job),
  };
}

// The backend keeps "success" until the next update, so persist the handled marker across mounts
// and tabs, or a fresh mount replays it.
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
): Promise<LlamaUpdateStatus | null> {
  if (!getAuthToken()) return null;
  try {
    const res = await authFetch(
      `/api/llama/update-status${forceRefresh ? "?force_refresh=true" : ""}`,
    );
    if (!res.ok) return null;
    return parseStatus(await res.json());
  } catch {
    return null;
  }
}

// Manual checks bypass the 24h release cache; job polls read local state.
const recheckStatus = () => fetchStatus(true);

interface UseLlamaUpdateCheckOptions {
  enabled?: boolean;
  /** The update unloaded the model server-side; resync chat. Also fires for cross-tab updates. */
  onReloadRequired?: () => void;
}

export interface LlamaApplyResult {
  ok: boolean;
  tag?: string | null;
  reloadRequired?: boolean | null;
  error?: string | null;
  // A migration can finish at the release and on the old backend; "updated to <tag>" fits neither.
  message?: string;
}

export function useLlamaUpdateCheck({
  enabled = true,
  onReloadRequired,
}: UseLlamaUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<LlamaUpdateStatus | null>(null);
  const [visible, setVisible] = useState(false);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const snoozeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // A ref keeps startJobPoll stable while calling the latest callback.
  const onReloadRequiredRef = useRef(onReloadRequired);
  useEffect(() => {
    onReloadRequiredRef.current = onReloadRequired;
  }, [onReloadRequired]);
  // Once per completed job, keyed by finished_at and seeded from localStorage.
  const reloadNotifiedForRef = useRef<string | null>(getHandledReloadAt());

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Shared by the poll, surface and stale-click paths so none drop or double-fire.
  const notifyReloadIfNeeded = useCallback(
    (
      job: Pick<LlamaUpdateJob, "state" | "reload_required" | "finished_at">,
    ) => {
      // "error" too: a chained update can unload the server before a later phase fails.
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

  const startJobPoll = useCallback(
    (onDone?: (result: LlamaApplyResult) => void) => {
      clearPollTimer();
      const timer = setInterval(async () => {
        const s = await fetchStatus();
        // Polls overlap: a "running" answer landing after a later poll saw the job
        // finish would re-set applying with no timer left to clear it.
        if (!s || pollTimer.current !== timer) return;
        setStatus(s);
        const presentation = llamaUpdatePresentation(llamaUpdateOffered(s), s.job);
        setApplying(presentation.applying);
        setVisible(presentation.visible);
        if (presentation.running) return;
        clearPollTimer();
        if (s.job.state === "success") {
          void refreshHardwareInfo();
          // Fires here, not only in apply's onDone, so cross-tab updates are covered.
          notifyReloadIfNeeded(s.job);
          onDone?.({
            ok: true,
            tag: s.job.to_tag,
            reloadRequired: s.job.reload_required,
            message: s.job.message,
          });
        } else if (s.job.state === "error") {
          // Keep the banner for retry; a partial update may still have unloaded the server.
          notifyReloadIfNeeded(s.job);
          onDone?.({ ok: false, error: s.job.error });
        } else {
          onDone?.({ ok: false, error: "update did not complete" });
        }
      }, JOB_POLL_INTERVAL_MS);
      pollTimer.current = timer;
    },
    [clearPollTimer, notifyReloadIfNeeded],
  );

  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      setStatus(next);
      const presentation = llamaUpdatePresentation(
        llamaUpdateOffered(next),
        next.job,
      );
      setApplying(presentation.applying);
      setVisible(presentation.visible);
      if (presentation.running) {
        if (!pollTimer.current) startJobPoll();
        return;
      }
      // A tab that missed the running window still needs to resync.
      notifyReloadIfNeeded(next.job);
    },
    [startJobPoll, notifyReloadIfNeeded],
  );

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let canceled = false;

    const firstTimer = setTimeout(() => {
      recheckStatus().then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, FIRST_CHECK_DELAY_MS);

    const reminder = setInterval(() => {
      recheckStatus().then((s) => {
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
  }, [enabled, surfaceIfAvailable, clearPollTimer]);

  // The storage event fires only in other tabs, so they recheck promptly after another tab applies.
  useEffect(() => {
    if (!enabled) return;
    const onStorage = (event: StorageEvent) => {
      if (
        event.key === HANDLED_RELOAD_STORAGE_KEY &&
        event.newValue &&
        event.newValue !== reloadNotifiedForRef.current
      ) {
        recheckStatus().then(surfaceIfAvailable);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [enabled, surfaceIfAvailable]);

  useEffect(() => {
    if (!enabled) return;
    return subscribeToLlamaJobStarted(() => {
      fetchStatus().then(surfaceIfAvailable);
    });
  }, [enabled, surfaceIfAvailable]);

  const dismiss = useCallback(() => {
    setVisible(false);
  }, []);

  const snooze = useCallback(() => {
    setVisible(false);
    if (snoozeTimer.current) clearTimeout(snoozeTimer.current);
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      recheckStatus().then(surfaceIfAvailable);
    }, SNOOZE_DELAY_MS);
  }, [surfaceIfAvailable]);

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
    // Signal both a new and a discovered running job so every Settings surface follows it.
    signalRunningLlamaJob(actionJob);

    // Skip backend switches: they share this job but install no release.
    if (
      action &&
      action.started === false &&
      !llamaUpdateAdoptsRunningJob(action.reason, actionJob)
    ) {
      // A stale click's response still carries another tab's completed job.
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
