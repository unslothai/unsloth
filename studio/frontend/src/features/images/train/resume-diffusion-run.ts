// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DiffusionTrainingRunDetail,
  DiffusionTrainingRunSummary,
  DiffusionTrainingStartRequest,
} from "../api";

export const RESUME_UNAVAILABLE_MESSAGE =
  "This run has no checkpoint to continue from. Start a new run instead.";


/** Replay a finished diffusion run's stored start config as a resume request, like the LLM tab's
 *  `resumeTrainingRun`: the run's own config is the source of truth, with
 *  `resume_from_checkpoint` pointed at its output directory. The backend picks the newest
 *  complete `checkpoint-<N>` bundle and continues at N+1, up to the SAME `train_steps` target
 *  the original had. Pure and throwing: the caller owns the toast. The thrown message is the
 *  backend's `resume_blocked_reason` whenever it has one, since a generic message would
 *  contradict the server's own explanation. */
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
    // A record from a much older backend, or a hand-edited one, cannot be replayed safely: a missing
    // path would resolve somewhere else entirely and train the wrong thing.
    throw new Error(
      "This run's saved settings are incomplete, so it cannot be resumed automatically.",
    );
  }

  // Destructured out rather than deleted off a widened copy: these three are replaced below, not
  // inherited. Replaying them verbatim would point the new run at whatever the OLD run resumed.
  const {
    hf_token: _replacedToken,
    resume_from_checkpoint: _replacedCheckpoint,
    resumed_from_job_id: _replacedJobId,
    ...inherited
  } = config;
  return {
    ...inherited,
    // Re-stated because the narrowing above proved them present, which `Partial` cannot carry.
    base_model: config.base_model,
    data_dir: config.data_dir,
    output_dir: config.output_dir,
    // The EXACT bundle the backend advertised, when it named one. Sending only the run folder would
    // let the server re-pick "newest", which in a folder two runs share is not necessarily the one
    // whose step this UI is showing.
    resume_from_checkpoint: detail.checkpoint_path || outputDir,
    resumed_from_job_id: detail.job_id,
    hf_token: options.hfToken || undefined,
  };
}

/** The label the Resume action shows, including the step it would continue from. */
export function resumeActionLabel(
  run: Pick<DiffusionTrainingRunSummary, "checkpoint_step">,
): string {
  return run.checkpoint_step != null
    ? `Resume from step ${run.checkpoint_step}`
    : "Resume training";
}
