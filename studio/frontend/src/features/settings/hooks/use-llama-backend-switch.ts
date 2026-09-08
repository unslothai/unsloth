// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { invalidateLlamaFlagCatalog } from "@/features/model-picker/api/llama-flags";
import { useT } from "@/i18n";
import {
  signalLlamaJobStarted,
  subscribeToLlamaJobStarted,
} from "@/lib/llama-job-events";
import {
  type OwnedLlamaSwitchOutcome,
  ownedLlamaSwitchOutcome,
} from "@/lib/llama-job-lifecycle";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type LlamaBackend,
  type LlamaBackendStatus,
  loadLlamaBackendStatus,
  switchLlamaBackend,
} from "../api/llama-backend";
import { backendDisplayName } from "../lib/llama-backend-labels";

// Fast enough to track the installer's progress milestones without hammering.
const JOB_POLL_MS = 700;

/** Manage backend selection and the shared llama.cpp install job. */
export function useLlamaBackendSwitch() {
  const t = useT();
  const [status, setStatus] = useState<LlamaBackendStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<LlamaBackend | null>(null);
  const [applying, setApplying] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const mounted = useRef(false);
  // Prevent overlapping resolver polls.
  const polling = useRef(false);
  // Updates share this job, so completion belongs to this surface only when
  // it has the identity of the switch accepted by apply().
  const ownedJob = useRef<{ startedAt: string | null } | null>(null);

  const refresh = useCallback(async (forceRefresh = false) => {
    try {
      const next = await loadLlamaBackendStatus(forceRefresh);
      if (mounted.current) {
        setStatus(next);
        setLoadError(null);
      }
      return next;
    } catch (error) {
      if (mounted.current) {
        setLoadError(error instanceof Error ? error.message : String(error));
      }
      return null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    // Opening the picker is an explicit host-capability recheck. If an install
    // is running, the poll repeats it once the backend can resolve options.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh(true);
    return () => {
      mounted.current = false;
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [refresh]);

  const finish = useCallback(
    (next: LlamaBackendStatus, outcome: OwnedLlamaSwitchOutcome | null) => {
      if (!mounted.current) {
        return;
      }
      setApplying(false);
      // The install marker is authoritative after completion.
      setSelected(next.backendRequest);
      // Here rather than where the switch is requested: that call only STARTS the
      // install, so the binary whose --help the flag catalogue describes is still
      // the old one, and a panel opened during the job would cache it again. Before
      // the owned-job check, so a tab that merely watched the switch drops it too.
      invalidateLlamaFlagCatalog();
      if (!outcome) {
        return;
      }
      ownedJob.current = null;
      if (outcome === "error") {
        toast.error(t("settings.resources.llamaBackend.switchFailed"), {
          description: next.job.error ?? undefined,
        });
        return;
      }
      if (outcome !== "success") {
        toast.error(t("settings.resources.llamaBackend.switchFailed"), {
          description: t("settings.resources.llamaBackend.switchInterrupted"),
        });
        return;
      }
      // The job detail includes reload or dictation repair information.
      toast.success(
        t("settings.resources.llamaBackend.switchedTo", {
          backend: backendDisplayName(next.backend, t),
        }),
        { description: next.job.message || undefined },
      );
    },
    [t],
  );

  const startPolling = useCallback(() => {
    if (!mounted.current) {
      return;
    }
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
    }
    pollTimer.current = setInterval(async () => {
      if (polling.current) {
        return;
      }
      polling.current = true;
      try {
        // The backend skips option resolution while a job is running, so this
        // becomes one forced host probe on the first terminal status.
        const next = await refresh(true);
        if (!next) {
          return;
        }
        const owned = ownedJob.current;
        const outcome = owned
          ? ownedLlamaSwitchOutcome(next.job, owned.startedAt)
          : null;
        if (next.job.state === "running" && (!owned || outcome === "running")) {
          return;
        }
        if (pollTimer.current) {
          clearInterval(pollTimer.current);
          pollTimer.current = null;
        }
        finish(next, outcome);
      } finally {
        polling.current = false;
      }
    }, JOB_POLL_MS);
  }, [refresh, finish]);

  // Follow jobs started by another surface or browser tab.
  useEffect(
    () =>
      subscribeToLlamaJobStarted(() => {
        void refresh().then((next) => {
          if (next?.job.state === "running") {
            startPolling();
          }
        });
      }),
    [refresh, startPolling],
  );

  const apply = useCallback(() => {
    const requested = selected ?? status?.backendRequest;
    if (!requested) {
      return;
    }
    setApplying(true);
    void (async () => {
      try {
        const started = await switchLlamaBackend(requested);
        if (!started.started) {
          if (!mounted.current) {
            return;
          }
          setApplying(false);
          toast.error(
            started.message ??
              t("settings.resources.llamaBackend.switchFailed"),
          );
          await refresh();
          return;
        }
        // App-wide recovery continues if Settings closes.
        signalLlamaJobStarted(started.job.startedAt);
        if (!mounted.current) {
          return;
        }
        ownedJob.current = { startedAt: started.job.startedAt };
        setStatus((current) =>
          current ? { ...current, job: started.job } : current,
        );
        startPolling();
      } catch (error) {
        if (!mounted.current) {
          return;
        }
        setApplying(false);
        ownedJob.current = null;
        toast.error(error instanceof Error ? error.message : String(error));
      }
    })();
  }, [selected, status?.backendRequest, refresh, startPolling, t]);

  // The component derives the untouched value from status.backendRequest.
  const running = applying || status?.job.state === "running";

  // Follow any shared job already present in status.
  useEffect(() => {
    if (status?.job.state === "running" && !pollTimer.current) {
      startPolling();
    }
  }, [status, startPolling]);

  return {
    status,
    selected,
    setSelected,
    running: Boolean(running),
    apply,
    refresh,
    loadError,
  };
}
