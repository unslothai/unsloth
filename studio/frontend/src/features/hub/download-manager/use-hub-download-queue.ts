// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { checkpointFirst } from "./required-assets";
import { useEffect, useRef } from "react";
import { create } from "zustand";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
} from "@/features/auth";
import { useDownloadManagerStore } from "./download-manager-controller";
import { toast } from "@/lib/toast";
import {
  useStagedDownload,
  type StagedDownloadEntry,
} from "./use-staged-download";
export type Plan = {
  id: number;
  origin?: { repoId: string; filename?: string };
  entries: StagedDownloadEntry[];
  remaining: StagedDownloadEntry[];
};
let nextId = 0;
export const useHubQueue = create<{ plans: Plan[] }>(() => ({ plans: [] }));
export function enqueueHubDownload(
  entries: StagedDownloadEntry[],
  origin?: { repoId: string; filename?: string },
) {
  if (!entries.length) return;
  entries = checkpointFirst(entries);
  const key = (rows: StagedDownloadEntry[]) =>
    JSON.stringify(rows.map((e) => [e.repoId, [...e.files].sort()]));
  useHubQueue.setState((state) =>
    state.plans.some((p) => key(p.entries) === key(entries))
      ? state
      : {
          plans: [
            ...state.plans,
            { id: ++nextId, origin, entries, remaining: entries },
          ],
        },
  );
}
export function useHubDownloadQueue() {
  const plan = useHubQueue((s) => s.plans[0]);
  const started = useRef<number | null>(null);
  const finish = () =>
    useHubQueue.setState((s) => ({
      plans: s.plans.filter((p) => p.id !== started.current),
    }));
  const { stage, remaining } = useStagedDownload({
    scopeId: "hub-assets",
    onReady: () => {
      finish();
      toast.success("Download complete", {
        description: "Select the model when you are ready to load it.",
      });
    },
    onCancelled: finish,
  });
  useEffect(() => {
    if (!remaining) return;
    useHubQueue.setState((s) => ({
      plans: s.plans.map((p) =>
        p.id === started.current ? { ...p, remaining } : p,
      ),
    }));
  }, [remaining]);
  useEffect(() => {
    const reset = () => {
      useHubQueue.setState({ plans: [] });
      started.current = null;
      stage([]);
    };
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, reset);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, reset);
    return () => {
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, reset);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, reset);
    };
  }, [stage]);
  useEffect(() => {
    if (!plan || started.current === plan.id) return;
    started.current = plan.id;
    stage(plan.entries);
  }, [plan, stage]);
}

export function useHubDownloadPlan(repoId: string, filename?: string) {
  return useHubQueue((s) =>
    s.plans.find((p) =>
      p.origin
        ? p.origin.repoId === repoId && p.origin.filename === filename
        : p.entries.some(
            (e) =>
              e.repoId === repoId &&
              e.checkpoint !== false &&
              (!filename || e.files.includes(filename)),
          ),
    ),
  );
}
export function useQueuedHubEntries() {
  const plans = useHubQueue((s) => s.plans);
  const jobs = useDownloadManagerStore((s) => s.jobs);
  return plans.flatMap((p) =>
    p.remaining
      .filter(
        (e) =>
          !Object.values(jobs).some(
            (j) =>
              j.variant === "@hub-assets" &&
              j.repoId === e.repoId &&
              (j.state === "running" || j.state === "cancelling") &&
              e.files.length === j.scopedFiles?.length &&
              e.files.every((f) => j.scopedFiles?.includes(f)),
          ),
      )
      .map((e) => ({ ...e, planId: p.id })),
  );
}
