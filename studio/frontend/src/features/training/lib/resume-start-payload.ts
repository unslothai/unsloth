// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStartRequest } from "../types/api";

/** Select the inspected import as the request's one authoritative resume source. */
export function withImportedResumeCheckpoint(
  payload: TrainingStartRequest,
  checkpointPath: string,
): TrainingStartRequest {
  return {
    ...payload,
    resume_from_checkpoint: null,
    resume_checkpoint_path: null,
    imported_resume_checkpoint: checkpointPath,
    in_place_continuation: false,
  };
}
