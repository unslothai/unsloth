// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import {
  signalLlamaJobStarted,
  subscribeToLlamaJobStarted,
} from "@/lib/llama-job-events";
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
  // Prevent duplicate completion handling.
  const handledFor = useRef<string | null>(null);
  // Updates share this job but should not produce switch notifications.
  const ourJob = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const next = await loadLlamaBackendStatus();
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
    // refresh updates state only after its network request settles.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    return () => {
      mounted.current = false;
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [refresh]);

  const finish = useCallback(
    (next: LlamaBackendStatus) => {
      if (!mounted.current) {
        return;
      }
      setApplying(false);
      // The install marker is authoritative after completion.
      setSelected(next.backendRequest);
      if (next.job.finishedAt && next.job.finishedAt === handledFor.current) {
        return;
      }
      handledFor.current = next.job.finishedAt;
      if (!ourJob.current) {
        return;
      }
      ourJob.current = false;
      if (next.job.state === "error") {
        toast.error(t("settings.resources.llamaBackend.switchFailed"), {
          description: next.job.error ?? undefined,
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
        const next = await refresh();
        if (!next || next.job.state === "running") {
          return;
        }
        if (pollTimer.current) {
          clearInterval(pollTimer.current);
          pollTimer.current = null;
        }
        finish(next);
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
    ourJob.current = true;
    void (async () => {
      try {
        const started = await switchLlamaBackend(requested);
        if (!started.started) {
          if (!mounted.current) {
            ourJob.current = false;
            return;
          }
          setApplying(false);
          ourJob.current = false;
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
          ourJob.current = false;
          return;
        }
        setStatus((current) =>
          current ? { ...current, job: started.job } : current,
        );
        startPolling();
      } catch (error) {
        if (!mounted.current) {
          ourJob.current = false;
          return;
        }
        setApplying(false);
        ourJob.current = false;
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
    loadError,
  };
}
