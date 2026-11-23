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
import { notifyModelLifecycle } from "@/lib/model-lifecycle-events";
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
  // The chained phase installing right now, when the backend reports phases.
  runningPhase: "llama" | "whisper" | "audio" | null;
  whisperFailed: boolean;
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
  // and the card lists every one the notification settings allow.
  llama: ComponentOffer;
  whisper: ComponentOffer | null;
  // The newest audio.cpp release, when the managed runtime is behind it.
  audio: ComponentOffer | null;
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
    runningPhase: parseRunningPhase(job.phases),
    whisperFailed:
      (job.phases as Record<string, { state?: unknown } | null> | null)?.whisper
        ?.state === "error",
  };
}

function parseRunningPhase(value: unknown): LlamaUpdateJob["runningPhase"] {
  if (!value || typeof value !== "object") {
    return null;
  }
  const phases = value as Record<
    string,
    { state?: unknown } | null | undefined
  >;
  return (
    (["llama", "whisper", "audio"] as const).find(
      (name) => phases[name]?.state === "running",
    ) ?? null
  );
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
  // Legacy top-level version fields keep their llama meaning; a whisper-only or audio-only update
  // must show the nested release, not equal llama tags as a new llama update.
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
    llama: {
      // Absent on backends before the whisper piggyback, where the legacy union is llama's answer.
      update_available:
        typeof s.llama_update_available === "boolean"
          ? s.llama_update_available
          : s.update_available === true && component === "llama.cpp",
      installed_tag:
        typeof s.installed_tag === "string" ? s.installed_tag : null,
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

// The backend keeps "success" until the next update, so persist the handled marker across mounts
// and tabs, or a fresh mount replays it.
const HANDLED_RELOAD_STORAGE_KEY = "unsloth_llama_update_reload_handled_at";
// sessionStorage, not localStorage: a skip kept for good stranded the install on that release.
const OFFER_SUPPRESSION_STORAGE_KEY = "unsloth_llama_update_offer_suppression";
const OFFER_SUPPRESSION_EVENT = "unsloth:llama-update-offer-suppression";

// Retires the permanent record older builds wrote.
try {
  localStorage.removeItem(OFFER_SUPPRESSION_STORAGE_KEY);
} catch {
  // Storage can be unavailable in restricted browser contexts.
}

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
    // whisper.cpp rides along; null matches keys stored before it joined.
    whisper: null,
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

/** The previous format recorded the whisper.cpp release in the key; ignore it so a new whisper.cpp release keeps the suppression. */
export function offerKeyMatches(
  key: string,
  status: LlamaUpdateStatus,
): boolean {
  const current = offerKey(status);
  if (key === current) return true;
  try {
    const stored: unknown = JSON.parse(key);
    if (!stored || typeof stored !== "object" || !("whisper" in stored)) {
      return false;
    }
    return JSON.stringify({ ...stored, whisper: null }) === current;
  } catch {
    return false;
  }
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
      sessionStorage.getItem(OFFER_SUPPRESSION_STORAGE_KEY),
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
      sessionStorage.setItem(
        OFFER_SUPPRESSION_STORAGE_KEY,
        JSON.stringify({ version: 1, ...suppression }),
      );
    } else {
      sessionStorage.removeItem(OFFER_SUPPRESSION_STORAGE_KEY);
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
  /** The update unloaded the model server-side; resync chat. Also fires for cross-tab updates. */
  onReloadRequired?: () => void;
  /** Reads the cached status once on mount; only refresh() asks GitHub. For Settings, which remounts on every open. */
  passive?: boolean;
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
  passive = false,
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
    if (passive || !activeRef.current || suppression?.kind !== "snoozed") {
      return;
    }
    const remaining = suppression.until - Date.now();
    if (remaining <= 0) return;
    snoozeTimer.current = setTimeout(() => {
      snoozeTimer.current = null;
      readStatus(true).then((next) => {
        if (activeRef.current) surfaceRef.current(next);
      });
    }, remaining);
  }, [passive, readStatus]);

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
        suppression != null &&
        offerKeyMatches(suppression.offerKey, next) &&
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

  // The update stops a loaded dictation sidecar and may swap the audio.cpp runtime without asking for a
  // reload, so the Audio page re-reads STT and runtime status once per finished job, whichever path saw it.
  const settledNotifiedForRef = useRef<string | null>(null);
  const notifySettledOnce = useCallback(
    (job: Pick<LlamaUpdateJob, "state" | "finished_at">) => {
      if (
        (job.state === "success" || job.state === "error") &&
        job.finished_at !== settledNotifiedForRef.current
      ) {
        settledNotifiedForRef.current = job.finished_at;
        notifyModelLifecycle({ runtime: "stt", loading: false, model: null });
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
        notifySettledOnce(s.job);
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
    [
      clearPollTimer,
      readStatus,
      commitStatus,
      notifyReloadIfNeeded,
      notifySettledOnce,
    ],
  );

  const surfaceIfAvailable = useCallback(
    (next: LlamaUpdateStatus | null) => {
      if (!next) return;
      const presentation = commitStatus(next);
      if (presentation.running) {
        if (!pollTimer.current) startJobPoll();
        return;
      }
      // A tab that missed the running window still needs to resync.
      notifyReloadIfNeeded(next.job);
      notifySettledOnce(next.job);
    },
    [commitStatus, startJobPoll, notifyReloadIfNeeded, notifySettledOnce],
  );
  useEffect(() => {
    surfaceRef.current = surfaceIfAvailable;
  }, [surfaceIfAvailable]);

  useEffect(() => {
    activeRef.current = enabled;
    if (!enabled) {
      return;
    }
    let canceled = false;
    if (statusRef.current) presentStatus(statusRef.current);
    else armSnoozeTimer();

    const check = (forceRefresh: boolean) => () => {
      readStatus(forceRefresh).then((s) => {
        if (!canceled) surfaceIfAvailable(s);
      });
    };
    let firstTimer: ReturnType<typeof setTimeout> | undefined;
    let reminder: ReturnType<typeof setInterval> | undefined;
    if (passive) {
      check(false)();
    } else {
      firstTimer = setTimeout(check(true), FIRST_CHECK_DELAY_MS);
      reminder = setInterval(check(true), REMINDER_INTERVAL_MS);
    }

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
    passive,
    readStatus,
    surfaceIfAvailable,
    clearPollTimer,
    armSnoozeTimer,
    presentStatus,
  ]);

  // The storage event fires only in other tabs, so they recheck promptly after another tab applies.
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
        readStatus(!passive).then(surfaceIfAvailable);
      }
    };
    window.addEventListener(OFFER_SUPPRESSION_EVENT, onSuppression);
    window.addEventListener("storage", onStorage);
    acceptSuppression(getOfferSuppression());
    return () => {
      window.removeEventListener(OFFER_SUPPRESSION_EVENT, onSuppression);
      window.removeEventListener("storage", onStorage);
    };
  }, [
    enabled,
    passive,
    readStatus,
    surfaceIfAvailable,
    presentStatus,
    armSnoozeTimer,
  ]);

  useEffect(() => {
    if (!enabled) return;
    return subscribeToLlamaJobStarted(() => {
      readStatus().then(surfaceIfAvailable);
    });
  }, [enabled, readStatus, surfaceIfAvailable]);

  const dismiss = useCallback((): string | null => {
    const current = statusRef.current;
    if (!current) return null;
    const suppression: OfferSuppression = {
      kind: "dismissed",
      offerKey: offerKey(current),
    };
    suppressionRef.current = suppression;
    publishOfferSuppression(suppression);
    presentStatus(current);
    return suppression.offerKey;
  }, [presentStatus]);

  // a later snooze or a skip of another offer is left alone.
  const undoDismiss = useCallback(
    (key: string) => {
      const suppression = suppressionRef.current;
      if (suppression?.kind !== "dismissed" || suppression.offerKey !== key) {
        return;
      }
      clearSuppression();
      const current = statusRef.current;
      if (current) presentStatus(current);
    },
    [clearSuppression, presentStatus],
  );

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
    // Signal both a new and a discovered running job so every Settings surface follows it.
    signalRunningLlamaJob(actionJob);

    // Skip backend switches: they share this job but install no release.
    if (
      action &&
      action.started === false &&
      !llamaUpdateAdoptsRunningJob(action.reason, actionJob)
    ) {
      // A stale click's response carries another tab's completed job; process its reload_required here.
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

  const refresh = useCallback(async (): Promise<LlamaUpdateStatus | null> => {
    const next = await readStatus(true);
    if (activeRef.current) surfaceIfAvailable(next);
    return next;
  }, [readStatus, surfaceIfAvailable]);

  return {
    status: enabled ? status : null,
    visible: enabled && visible,
    applying: enabled && applying,
    apply,
    refresh,
    dismiss,
    undoDismiss,
    snooze,
  };
}
