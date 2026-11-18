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
  | "download_progress"
  | "model_load_stage"
  | "model_load_progress"
>;

const QUANTIZATION_NOTE =
  "Downloaded files are reused; 4-bit conversion runs in memory each time.";

const PHASE_MESSAGE =
  /^(?:Loading base model\.\.\.|Training\.\.\.|Preparing \(.*\)\.\.\.)$/;
const TERMINAL_STATUSES = new Set(["completed", "stopped", "error"]);
// Older backends only carry the preparation counts in the status text.
const LEGACY_PREPARING = /Preparing \(([^()]+?) (\d+)\/(\d+)\)/;

// The heading renders this as-is: "stopped" -> "Stopped".
function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function backendNote(message: string | null | undefined): string | null {
  return message && !PHASE_MESSAGE.test(message) ? message : null;
}

export function diffusionTrainingProgress(status: Status | null) {
  const running = status?.active === true || status?.status === "running";
  // A stopped / failed run can still report active while its worker exits; not a restart.
  const terminal = TERMINAL_STATUSES.has(status?.status ?? "");
  const preparing =
    running &&
    !terminal &&
    (status?.in_model_load === true || (status?.step ?? 0) <= 0);
  if (preparing) {
    const note = backendNote(status?.message);
    const legacy = status?.message?.match(LEGACY_PREPARING);
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
      // MiniMax-H3 encodes every caption in one go (n/n) and only then caches latents; that is not done yet.
      const finished =
        known &&
        (done as number) >= (total as number) &&
        prep.stage !== "encode_prompts";
      return {
        preparing: true,
        title: finished ? "Training data prepared" : "Preparing training data",
        detail:
          note ??
          (finished
            ? "Finishing model setup before training starts."
            : "Preparing images and captions so the model can train on them."),
        count: known
          ? `${Math.min(done as number, total as number)} / ${total} prepared`
          : null,
        percent: known
          ? Math.min(100, ((done as number) / (total as number)) * 100)
          : null,
      };
    }
    const load = status?.model_load_progress;
    const stage = status?.model_load_stage;
    // A retained tensor count is stale while the download manager shows a transfer.
    const downloading = status?.download_progress != null;
    const measured = status?.in_model_load && !downloading && load &&
      Number.isFinite(load.done) && Number.isFinite(load.total) &&
      load.done >= 0 && load.total > 0 && load.done <= load.total;
    if (status?.in_model_load && !downloading && (measured || stage)) {
      return {
        preparing: true,
        title: stage === "quantizing" ? "Loading model and converting to 4-bit"
          : stage === "conditioning" ? "Loading text encoders and image encoder"
          : "Loading training model weights",
        detail: note ?? (measured
          ? stage === "quantizing"
            ? `${load.done.toLocaleString()} of ${load.total.toLocaleString()} ${load.unit === "tensors" ? "weight tensors" : load.unit} processed. ${QUANTIZATION_NOTE}`
            : `${load.label} · ${load.done.toLocaleString()} / ${load.total.toLocaleString()} ${load.unit} loaded.`
          : stage === "quantizing"
            ? QUANTIZATION_NOTE
            : "Waiting for the model loader to report progress. Training has not started yet."),
        count: measured ? `${Math.floor(load.done / load.total * 100)}%` : null,
        percent: measured ? load.done / load.total * 100 : null,
      };
    }
    return {
      preparing: true,
      title: status?.in_model_load
        ? "Loading training model"
        : "Starting training",
      detail:
        note ??
        (status?.in_model_load
          ? "Getting the model ready before training starts."
          : "Preparing the first training step. Compilation can take longer on the first run."),
      count: null,
      percent: null,
    };
  }
  return {
    preparing: false,
    title:
      status?.status === "completed"
        ? "Training complete"
        : running && !terminal
          ? "Training"
          : capitalize(status?.status ?? ""),
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
