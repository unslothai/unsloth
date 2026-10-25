// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import {
  isTrainingRunActive,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";
import type { TrainingRuntimeState } from "../types/runtime";

type TaskbarProgress = {
  status: "None" | "Normal" | "Indeterminate" | "Error";
  progress?: number;
};

/** What the taskbar (Windows) or dock (macOS) icon shows for a training state. */
export function taskbarProgressFor(
  state: Pick<
    TrainingRuntimeState,
    "phase" | "isTrainingRunning" | "progressPercent"
  >,
): TaskbarProgress {
  if (state.phase === "error") {
    // Full, so the red bar still shows when a run fails before its first step.
    return { status: "Error", progress: 100 };
  }
  if (state.phase === "training" && state.progressPercent > 0) {
    return { status: "Normal", progress: Math.round(state.progressPercent) };
  }
  // Downloading, loading, and finalizing after the last step have no percent to show.
  if (isTrainingRunActive(state)) {
    return { status: "Indeterminate" };
  }
  return { status: "None" };
}

function setTaskbarProgress({ status, progress }: TaskbarProgress): void {
  void import("@tauri-apps/api/window")
    .then(({ getCurrentWindow, ProgressBarStatus }) =>
      getCurrentWindow().setProgressBar({
        status: ProgressBarStatus[status],
        progress,
      }),
    )
    .catch(() => {});
}

/** Mirrors training progress onto the app icon. Desktop only; call once at the app root. */
export function useTrainingTaskbarProgress() {
  useEffect(() => {
    if (!isTauri) return;
    let shown = "";
    const publish = (state: TrainingRuntimeState) => {
      const next = taskbarProgressFor(state);
      // Progress events arrive many times a second; only cross IPC when the icon changes.
      const key = `${next.status}:${next.progress ?? ""}`;
      if (key === shown) return;
      shown = key;
      setTaskbarProgress(next);
    };
    publish(useTrainingRuntimeStore.getState());
    const unsubscribe = useTrainingRuntimeStore.subscribe(publish);
    return () => {
      unsubscribe();
      setTaskbarProgress({ status: "None" });
    };
  }, []);
}
