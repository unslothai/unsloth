// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DiffusionTrainingRunDetail,
  DiffusionTrainingRunSummary,
  DiffusionTrainingStartRequest,
} from "../api";

export const RESUME_UNAVAILABLE_MESSAGE =
  "This run has no checkpoint to continue from. Start a new run instead.";


/**
 * Replay a finished run's stored config with resume_from_checkpoint at its output dir.
 * Throws the backend's resume_blocked_reason when present; the caller owns the toast.
 */
export function buildDiffusionResumePayload(
  detail: DiffusionTrainingRunDetail,
  options: { hfToken?: string | null } = {},
): DiffusionTrainingStartRequest {
  const outputDir = detail.output_dir ?? null;
  if (!(detail.can_resume && outputDir)) {
    throw new Error(detail.resume_blocked_reason || RESUME_UNAVAILABLE_MESSAGE);
  }
  const config = (detail.config ?? {}) as Partial<DiffusionTrainingStartRequest>;
  if (!config.base_model || !config.data_dir || !config.output_dir) {
    // Without a stored path the replay would train the wrong thing.
    throw new Error(
      "This run's saved settings are incomplete, so it cannot be resumed automatically.",
    );
  }

  // Destructured out so the new run never inherits the old run's resume target.
  const {
    hf_token: _replacedToken,
    resume_from_checkpoint: _replacedCheckpoint,
    resumed_from_job_id: _replacedJobId,
    ...inherited
  } = config;
  return {
    ...inherited,
    base_model: config.base_model,
    data_dir: config.data_dir,
    output_dir: config.output_dir,
    // Send the exact advertised bundle: in a shared folder "newest" may be another run's.
    resume_from_checkpoint: detail.checkpoint_path || outputDir,
    resumed_from_job_id: detail.job_id,
    hf_token: options.hfToken || undefined,
  };
}

export function resumeActionLabel(
  run: Pick<DiffusionTrainingRunSummary, "checkpoint_step">,
): string {
  return run.checkpoint_step != null
    ? `Resume from step ${run.checkpoint_step}`
    : "Resume training";
}
