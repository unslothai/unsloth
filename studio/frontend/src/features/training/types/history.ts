// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TrainingRunSummary {
  id: string;
  status: "running" | "completed" | "stopped" | "error";
  model_name: string;
  dataset_name: string;
  started_at: string;
  ended_at: string | null;
  total_steps: number | null;
  final_step: number | null;
  final_loss: number | null;
  output_dir: string | null;
  duration_seconds: number | null;
  error_message: string | null;
  loss_sparkline: number[] | null;
}

export interface TrainingRunListResponse {
  runs: TrainingRunSummary[];
  total: number;
}

export interface TrainingRunMetrics {
  step_history: number[];
  loss_history: number[];
  loss_step_history: number[];
  lr_history: number[];
  lr_step_history: number[];
  grad_norm_history: number[];
  grad_norm_step_history: number[];
  eval_loss_history: number[];
  eval_step_history: number[];
  final_epoch: number | null;
  final_num_tokens: number | null;
}

export interface TrainingRunDetailResponse {
  run: TrainingRunSummary;
  config: Record<string, unknown>;
  metrics: TrainingRunMetrics;
}

export interface TrainingRunDeleteResponse {
  status: string;
  message: string;
}
