// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStatusResponse } from "../types/runtime";

export function statusConfirmsActiveTrainingStart(
  status: Pick<
    TrainingStatusResponse,
    "job_id" | "is_training_running" | "start_request_id"
  >,
  expectedStartRequestId: string,
): boolean {
  return (
    status.is_training_running &&
    status.job_id.trim().length > 0 &&
    status.start_request_id === expectedStartRequestId
  );
}
