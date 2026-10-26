// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainingStatus } from "../api.ts";

type Status = Pick<
  DiffusionTrainingStatus,
  | "active"
  | "status"
  | "step"
  | "total_steps"
  | "in_model_load"
  | "message"
  | "preparation"
>;

/** Preparation and optimization have different units; each bar reports its own stage. */
export function diffusionTrainingProgress(status: Status | null) {
  const running = status?.active === true || status?.status === "running";
  const preparing =
    running && (status?.in_model_load === true || (status?.step ?? 0) <= 0);
  if (preparing) {
    // Older backends only carry these counts in the status text.
    const legacy = status?.message.match(
      /Preparing \(([^()]+?) (\d+)\/(\d+)\)/,
    );
    const prep =
      status?.preparation ??
      (legacy
        ? {
            stage: legacy[1].replaceAll(" ", "_"),
            done: Number(legacy[2]),
            total: Number(legacy[3]),
          }
        : null);
    if (prep) {
      const { done, total } = prep;
      const known =
        Number.isFinite(done) &&
        Number.isFinite(total) &&
        (done as number) >= 0 &&
        (total as number) > 0;
      const finished = known && (done as number) >= (total as number);
      return {
        preparing: true,
        title: finished ? "Training data prepared" : "Preparing training data",
        detail: finished
          ? "Finishing model setup before training starts."
          : "Preparing images and captions so the model can train on them.",
        count: known
          ? `${Math.min(done as number, total as number)} / ${total} prepared`
          : null,
        percent: known
          ? Math.min(100, ((done as number) / (total as number)) * 100)
          : null,
      };
    }
    return {
      preparing: true,
      title: status?.in_model_load
        ? "Loading training model"
        : "Starting training",
      detail: status?.in_model_load
        ? "Getting the model ready before training starts."
        : "Preparing the first training step. Compilation can take longer on the first run.",
      count: null,
      percent: null,
    };
  }
  return {
    preparing: false,
    title:
      status?.status === "completed"
        ? "Training complete"
        : running
          ? "Training"
          : (status?.status ?? ""),
    detail: status?.message ?? null,
    count:
      status && status.total_steps > 0
        ? `${status.step} / ${status.total_steps} steps`
        : null,
    percent:
      status && status.total_steps > 0
        ? Math.max(0, Math.min(100, (status.step / status.total_steps) * 100))
        : 0,
  };
}
