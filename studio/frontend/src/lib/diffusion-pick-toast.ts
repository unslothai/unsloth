// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One toast per Hub pick, from planning through download to model loading.

import { toast } from "@/lib/toast";

export type PickToastPhase = "preparing" | "queued" | "downloading" | "waiting" | "ready";

export interface PickToastProgress {
  downloadedBytes: number;
  totalBytes: number;
}

/** Progress tagged with the plan ID returned by `stage()`. */
export interface StagedPlanProgress extends PickToastProgress {
  plan: number;
}

export interface PickToastOptions {
  describe: (phase: PickToastPhase, progress: PickToastProgress | null) => unknown;
  classNames?: Record<string, string>;
}

export interface PickToast {
  /** Replace the current toast with a new pick. */
  show: () => string;
  /** Dismiss only if `id` still owns the toast. */
  dismiss: (id: string | undefined) => void;
  /** Dismiss the current pick toast. */
  dismissAll: () => void;
  /** Bind download progress to `plan`; other phases clear that binding. */
  setPhase: (id: string | undefined, phase: PickToastPhase, plan?: number) => void;
  /** Update progress only for the bound download plan. */
  progress: (progress: StagedPlanProgress | null) => void;
  /** Transfer ownership to the load; return undefined if the toast is gone. */
  take: (id: string | undefined) => string | undefined;
}

// Images and Video share Sonner's store, so IDs must be unique across controllers.
let seq = 0;

export function createPickToast({ describe, classNames }: PickToastOptions): PickToast {
  let live: {
    id: string;
    phase: PickToastPhase;
    plan?: number;
    progress: PickToastProgress | null;
  } | null = null;

  const render = () => {
    if (!live) return;
    const id = live.id;
    toast(null, {
      id,
      description: describe(live.phase, live.progress) as never,
      duration: Infinity,
      closeButton: true,
      ...(classNames ? { classNames } : {}),
      // Prevent updates from reopening a user-dismissed toast.
      onDismiss: () => {
        if (live?.id === id) live = null;
      },
    });
  };

  return {
    show: () => {
      if (live) toast.dismiss(live.id);
      seq += 1;
      live = { id: `diffusion-pick:${seq}`, phase: "preparing", progress: null };
      render();
      return live.id;
    },
    dismiss: (id) => {
      if (id === undefined || live?.id !== id) return;
      live = null;
      toast.dismiss(id);
    },
    dismissAll: () => {
      if (!live) return;
      const gone = live.id;
      live = null;
      toast.dismiss(gone);
    },
    setPhase: (id, phase, plan) => {
      if (id === undefined || live?.id !== id) return;
      live = { id, phase, plan: phase === "downloading" ? plan : undefined, progress: null };
      render();
    },
    progress: (progress) => {
      // Ignore progress from unrelated download-only plans.
      if (!progress || live?.phase !== "downloading" || live.plan !== progress.plan) return;
      if (
        live.progress?.downloadedBytes === progress.downloadedBytes &&
        live.progress?.totalBytes === progress.totalBytes
      ) {
        return;
      }
      live = {
        ...live,
        progress: { downloadedBytes: progress.downloadedBytes, totalBytes: progress.totalBytes },
      };
      render();
    },
    take: (id) => {
      if (id === undefined || live?.id !== id) return undefined;
      live = null;
      return id;
    },
  };
}
