// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Mirrors backend models/training.py TrainingQueueItem / TrainingQueueStateResponse. */

export type TrainingQueueItemStatus =
  | "pending"
  | "starting"
  | "running"
  | "done"
  | "skipped";

export interface TrainingQueueItem {
  id: string;
  position: number;
  status: TrainingQueueItemStatus;
  model_name: string;
  dataset_summary: string;
  project_name: string | null;
  job_id: string | null;
  result_status: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingQueueState {
  paused: boolean;
  paused_reason: "restart" | "user" | null;
  pending_count: number;
  max_pending: number;
  active_job_id: string | null;
  items: TrainingQueueItem[];
}
