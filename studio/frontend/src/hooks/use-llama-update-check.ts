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

// Initial check plus hourly reminders until dismissed or applied.
const FIRST_CHECK_DELAY_MS = 1000;
const REMINDER_INTERVAL_MS = 60 * 60 * 1000; // ~1 hour
// Snooze checks sooner than the hourly reminder.
const SNOOZE_DELAY_MS = 15 * 60 * 1000; // ~15 minutes
// Poll fast enough to catch installer progress milestones.
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
  // Download fraction while running, 1 on success.
  progress: number | null;
  // Identifies the accepted job when notifying other surfaces and tabs.
  started_at: string | null;
  // Set once the job leaves "running"; identifies a completed job so a
  // repeated fetch of the same success can be told apart from the next one.
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
  component: "llama.cpp" | "whisper.cpp" | "audio.cpp";
  // Carried per component whatever the card names: several can be behind at once,
  // and the card shows whichever one the notification switches allow.
  llama: ComponentOffer;
  whisper: ComponentOffer | null;
  // The release this Studio pins, when the managed audio.cpp runtime is another one.
  audio: ComponentOffer | null;
  installed_tag: string | null;
  latest_tag: string | null;
  // Prebuilt download size in bytes, if known.
  update_size_bytes: number | null;
  // The install recorded "auto" and detection now resolves elsewhere, so Update would move
  // it. Independent of update_available: reported only when the release is current.
  backend_migration_available: boolean;
  from_backend: string | null;
  to_backend: string | null;
  job: LlamaUpdateJob;
}

/** Whether the banner has anything to offer: a newer release, or a backend the
 *  install's own recorded "auto" would resolve to today. */
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

function parseOffer(offer: Record<string, unknown>): ComponentOffer {
  return {
    update_available: offer.update_available === true,
    installed_tag:
      typeof offer.installed_tag === "string" ? offer.installed_tag : null,
    latest_tag: typeof offer.latest_tag === "string" ? offer.latest_tag : null,
    update_size_bytes:
      typeof offer.update_size_bytes === "number"
        ? offer.update_size_bytes
        : null,
  };
}

function nestedOffer(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function parseStatus(value: unknown): LlamaUpdateStatus | null {
  if (!value || typeof value !== "object") return null;
  const s = value as Record<string, unknown>;
  const component =
    s.update_component === "whisper"
      ? "whisper.cpp"
      : s.update_component === "audio"
        ? "audio.cpp"
        : "llama.cpp";
  const whisper = nestedOffer(s.whisper);
  const audio = nestedOffer(s.audio);
  // Legacy top-level version fields intentionally retain their llama meaning.
  // A whisper-only or audio-only update must display the nested release instead
  // of presenting equal llama tags as a new llama update.
  const details =
    component === "whisper.cpp" && whisper
      ? whisper
      : component === "audio.cpp" && audio
        ? audio
        : s;
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
    // The top-level version fields keep their llama meaning whatever `details` is.
    llama: {
      // Absent from a backend older than the whisper piggyback: there the legacy
      // union is llama's own answer.
      update_available:
        typeof s.llama_update_available === "boolean"
          ? s.llama_update_available
          : s.update_available === true && component === "llama.cpp",
      installed_tag: typeof s.installed_tag === "string" ? s.installed_tag : null,
      latest_tag: typeof s.latest_tag === "string" ? s.latest_tag : null,
      update_size_bytes:
        typeof s.update_size_bytes === "number" ? s.update_size_bytes : null,
    },
    whisper: whisper ? parseOffer(whisper) : null,
    audio: audio ? parseOffer(audio) : null,
    // Always from the top level: the backend belongs to the llama.cpp install whatever
    // component the version fields describe.
    backend_migration_available: s.backend_migration_available === true,
    from_backend: typeof s.from_backend === "string" ? s.from_backend : null,
    to_backend: typeof s.to_backend === "string" ? s.to_backend : null,
    job: parseJob(s.job),
  };
}

// The backend job persists as "success" until the next update starts (it's a single in-memory
// record, not per-tab), so a fresh mount -- a new tab, or a page reload of a tab that already
// resynced -- would otherwise replay the same completed job forever. Persist the handled marker
// outside React state so it survives both, and is shared across tabs in this browser.
const HANDLED_RELOAD_STORAGE_KEY = "unsloth_llama_update_reload_handled_at";
const OFFER_SUPPRESSION_STORAGE_KEY = "unsloth_llama_update_offer_suppression";
const OFFER_SUPPRESSION_EVENT = "unsloth:llama-update-offer-suppression";

type OfferSuppression =
  | { kind: "dismissed"; offerKey: string }
  | { kind: "snoozed"; offerKey: string; until: number };

let retainedSuppression: {
  value: OfferSuppression | null;
  persisted: boolean;
} = { value: null, persisted: true };

export function offerKey(status: LlamaUpdateStatus): string {
  return JSON.stringify({
    llama: status.llama.update_available
      ? [status.llama.installed_tag, status.llama.latest_tag]
      : null,
    whisper: status.whisper?.update_available
      ? [status.whisper.installed_tag, status.whisper.latest_tag]
      : null,
    migration: status.backend_migration_available
      ? [
          status.llama.installed_tag,
          status.llama.latest_tag,
          status.from_backend,
          status.to_backend,
        ]
      : null,
    // Only when offered, so keys stored before audio.cpp joined the card still match.
    ...(status.audio?.update_available
      ? { audio: [status.audio.installed_tag, status.audio.latest_tag] }
      : {}),
  });
}

function parseOfferSuppression(raw: string | null): OfferSuppression | null {
  if (!raw) return null;
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object") return null;
    const record = value as Record<string, unknown>;
    if (record.version !== 1 || typeof record.offerKey !== "string") {
      return null;
    }
    if (record.kind === "dismissed") {
      return { kind: "dismissed", offerKey: record.offerKey };
    }
    if (
      record.kind === "snoozed" &&
      typeof record.until === "number" &&
      Number.isFinite(record.until)
    ) {
      return {
        kind: "snoozed",
        offerKey: record.offerKey,
        until: record.until,
      };
    }
  } catch {
    return null;
  }
  return null;
}

function getOfferSuppression(): OfferSuppression | null {
  if (!retainedSuppression.persisted) return retainedSuppression.value;
  try {
    const value = parseOfferSuppression(
      localStorage.getItem(OFFER_SUPPRESSION_STORAGE_KEY),
    );
    retainedSuppression = { value, persisted: true };
    return value;
  } catch {
    return retainedSuppression.value;
  }
}

function publishOfferSuppression(suppression: OfferSuppression | null): void {
  retainedSuppression = { value: suppression, persisted: false };
  try {
    if (suppression) {
      localStorage.setItem(
        OFFER_SUPPRESSION_STORAGE_KEY,
        JSON.stringify({ version: 1, ...suppression }),
      );
    } else {
      localStorage.removeItem(OFFER_SUPPRESSION_STORAGE_KEY);
    }
    retainedSuppression.persisted = true;
  } catch {
    // Storage can be unavailable in restricted browser contexts.
  }
  window.dispatchEvent(
    new CustomEvent<OfferSuppression | null>(OFFER_SUPPRESSION_EVENT, {
      detail: suppression,
    }),
  );
}

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
  // What the job says it did: a migration can finish at the release and on the backend it
  // started from, so "updated to <tag>" fits neither.
  message?: string;
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
  const suppressionRef = useRef<OfferSuppression | null>(getOfferSuppression());
  const statusRef = useRef<LlamaUpdateStatus | null>(null);
  const activeRef = useRef(enabled);
  const surfaceRef = useRef<(next: LlamaUpdateStatus | null) => void>(() => {});
  const statusReadRef = useRef({ issued: 0, accepted: 0 });

  const readStatus = useCallback(async (forceRefresh = false) => {
    const sequence = ++statusReadRef.current.issued;
    const suppressionAtRequest = suppressionRef.current;
    const next = await fetchStatus(forceRefresh);
    if (!next || sequence < statusReadRef.current.accepted) return null;
    if (
      next.job.state === "running" &&
      next.job.operation !== "switch" &&
      suppressionRef.current !== null &&
      suppressionRef.current !== suppressionAtRequest
    ) {
      return null;
    }
    statusReadRef.current.accepted = sequence;
    return next;
  }, []);

  const armSnoozeTimer = useCallback(() => {
    if (snoozeTimer.current) clearTimeout(snoozeTimer.current);
    snoozeTimer.current = null;
    const suppression = suppressionRef.current;
    if (!activeRef.current || suppression?.kind !== "snoozed") return;
    const remaining = suppression.until - Date.now();
    if (remaining <= 0) return;
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      readStatus(true).then((next) => {
        if (activeRef.current) surfaceRef.current(next);
      });
    }, remaining);
  }, [readStatus]);

  const clearSuppression = useCallback(() => {
    if (!suppressionRef.current) return;
    suppressionRef.current = null;
    publishOfferSuppression(null);
    armSnoozeTimer();
  }, [armSnoozeTimer]);

  const presentStatus = useCallback(
    (next: LlamaUpdateStatus) => {
      const presentation = llamaUpdatePresentation(
        llamaUpdateOffered(next),
        next.job,
      );
      const suppression = suppressionRef.current;
      const suppressed =
        suppression?.offerKey === offerKey(next) &&
        (suppression.kind === "dismissed" || suppression.until > Date.now());
      setApplying(presentation.applying);
      setVisible(presentation.visible && (presentation.running || !suppressed));
      armSnoozeTimer();
      return presentation;
    },
    [armSnoozeTimer],
  );

  const commitStatus = useCallback(
    (next: LlamaUpdateStatus) => {
      statusRef.current = next;
      setStatus(next);
      const presentation = presentStatus(next);
      if (presentation.applying) clearSuppression();
      return presentation;
    },
    [presentStatus, clearSuppression],
  );

  // Read through a ref so startJobPoll stays stable (apply/surfaceIfAvailable
  // depend on it) while still calling the latest callback.
  const onReloadRequiredRef = useRef(onReloadRequired);
  useEffect(() => {
    onReloadRequiredRef.current = onReloadRequired;
  }, [onReloadRequired]);
  // Fires the callback once per completed job, whether this tab watched it run or only saw the
  // persisted "success" after the fact (e.g. another tab applied it). Keyed by finished_at and
  // seeded from localStorage so a fresh mount (new tab, or a page reload of a tab that already
  // resynced) doesn't replay a job some tab already handled.
  const reloadNotifiedForRef = useRef<string | null>(getHandledReloadAt());

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Shared by the poll path (this tab watched the job run), the surface path (this tab only saw the
  // persisted success), and apply()'s stale-click path (the job came back embedded in a "not
  // started" response) so none of them can drop or double-fire the notification.
  const notifyReloadIfNeeded = useCallback(
    (
      job: Pick<LlamaUpdateJob, "state" | "reload_required" | "finished_at">,
    ) => {
      // "error" is included for partial chained updates: the llama phase can land (and unload the
      // server) before a later phase fails, and the backend keeps reload_required set in exactly
      // that case. Without the resync the chat UI would keep pointing at the unloaded model.
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
      const timer = setInterval(async () => {
        const s = await readStatus();
        // Polls overlap: a "running" answer landing after a later poll saw the job
        // finish would re-set applying with no timer left to clear it.
        if (!s || pollTimer.current !== timer) return;
        const presentation = commitStatus(s);
        if (presentation.running) return;
        clearPollTimer();
        if (s.job.state === "success") {
          void refreshHardwareInfo();
          // The update unloads the running model server-side, so the chat runtime still points at a
          // model that now 400s on send. Let the consumer drop the selector to "select model"
          // instead of waiting for a page reload. Fires here (not just from apply's onDone) so a
          // cross-tab update mirrored through this poll is covered too.
          notifyReloadIfNeeded(s.job);
          onDone?.({
            ok: true,
            tag: s.job.to_tag,
            reloadRequired: s.job.reload_required,
            message: s.job.message,
          });
        } else if (s.job.state === "error") {
          // Keep the banner visible so retry is available. A partial chained
          // update can still have unloaded the llama server before failing.
          notifyReloadIfNeeded(s.job);
          onDone?.({ ok: false, error: s.job.error });
        } else {
          onDone?.({ ok: false, error: "update did not complete" });
        }
      }, JOB_POLL_INTERVAL_MS);
      pollTimer.current = timer;
    },
    [clearPollTimer, readStatus, commitStatus, notifyReloadIfNeeded],
  );

  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      const presentation = commitStatus(next);
      if (presentation.running) {
        if (!pollTimer.current) startJobPoll();
        return;
      }
      // A completed job persists as "success" until the next update starts, so a tab that missed
      // the running window entirely (mounted, or only checks hourly and misses both the running and
      // just-finished moments) still needs to resync here, not just from the poll path above.
      notifyReloadIfNeeded(next.job);
    },
    [commitStatus, startJobPoll, notifyReloadIfNeeded],
  );
  useEffect(() => {
    surfaceRef.current = surfaceIfAvailable;
  }, [surfaceIfAvailable]);

  useEffect(() => {
    activeRef.current = enabled;
    if (!enabled) {
      // Re-enabling will rediscover any still-running job.
      return;
    }
    let canceled = false;
    if (statusRef.current) presentStatus(statusRef.current);
    else armSnoozeTimer();

    const firstTimer = setTimeout(() => {
      readStatus(true).then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, FIRST_CHECK_DELAY_MS);

    const reminder = setInterval(() => {
      readStatus(true).then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    }, REMINDER_INTERVAL_MS);

    return () => {
      canceled = true;
      activeRef.current = false;
      clearTimeout(firstTimer);
      clearInterval(reminder);
      clearPollTimer();
      if (snoozeTimer.current) {
        clearTimeout(snoozeTimer.current);
        snoozeTimer.current = null;
      }
    };
  }, [
    enabled,
    readStatus,
    surfaceIfAvailable,
    clearPollTimer,
    armSnoozeTimer,
    presentStatus,
  ]);

  // Cross-tab nudge: a tab that only checks hourly would otherwise stay pointed at a
  // server-unloaded model for up to an hour after a DIFFERENT open tab applies an update. The
  // storage event only fires in other tabs (never the one that wrote it), so this recheck fires
  // promptly there without this tab redundantly re-triggering itself.
  useEffect(() => {
    const acceptSuppression = (suppression: OfferSuppression | null) => {
      if (suppressionRef.current === suppression) return;
      suppressionRef.current = suppression;
      if (statusRef.current) presentStatus(statusRef.current);
      else armSnoozeTimer();
    };
    const onSuppression = (event: Event) => {
      acceptSuppression((event as CustomEvent<OfferSuppression | null>).detail);
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key === OFFER_SUPPRESSION_STORAGE_KEY) {
        const value = parseOfferSuppression(event.newValue);
        retainedSuppression = { value, persisted: true };
        acceptSuppression(value);
        return;
      }
      if (
        enabled &&
        event.key === HANDLED_RELOAD_STORAGE_KEY &&
        event.newValue &&
        event.newValue !== reloadNotifiedForRef.current
      ) {
        readStatus(true).then(surfaceIfAvailable);
      }
    };
    window.addEventListener(OFFER_SUPPRESSION_EVENT, onSuppression);
    window.addEventListener("storage", onStorage);
    acceptSuppression(getOfferSuppression());
    return () => {
      window.removeEventListener(OFFER_SUPPRESSION_EVENT, onSuppression);
      window.removeEventListener("storage", onStorage);
    };
  }, [enabled, readStatus, surfaceIfAvailable, presentStatus, armSnoozeTimer]);

  useEffect(() => {
    if (!enabled) return;
    return subscribeToLlamaJobStarted(() => {
      readStatus().then(surfaceIfAvailable);
    });
  }, [enabled, readStatus, surfaceIfAvailable]);

  const dismiss = useCallback(() => {
    const current = statusRef.current;
    if (!current) return;
    const suppression: OfferSuppression = {
      kind: "dismissed",
      offerKey: offerKey(current),
    };
    suppressionRef.current = suppression;
    publishOfferSuppression(suppression);
    presentStatus(current);
  }, [presentStatus]);

  const snooze = useCallback(() => {
    const current = statusRef.current;
    if (!current) return;
    const suppression: OfferSuppression = {
      kind: "snoozed",
      offerKey: offerKey(current),
      until: Date.now() + SNOOZE_DELAY_MS,
    };
    suppressionRef.current = suppression;
    publishOfferSuppression(suppression);
    presentStatus(current);
  }, [presentStatus]);

  const apply = useCallback(async (): Promise<LlamaApplyResult> => {
    if (applying) return { ok: false, error: "already running" };
    const suppressionAtRequest = suppressionRef.current;
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

    // Non-started jobs stay idle; an already-running update is tracked below. A backend switch is
    // not: it shares this job but installs no new release, so following it here would toast an
    // update that never happened. The shared background listener still follows the switch itself.
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

    if (
      actionJob.operation !== "switch" &&
      (action?.started === true || actionJob.state === "running") &&
      (suppressionRef.current === null ||
        suppressionRef.current === suppressionAtRequest)
    ) {
      clearSuppression();
    }
    return await new Promise<LlamaApplyResult>((resolve) =>
      startJobPoll(resolve),
    );
  }, [applying, startJobPoll, notifyReloadIfNeeded, clearSuppression]);

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying: enabled && applying,
    apply,
    dismiss,
    snooze,
  };
}
