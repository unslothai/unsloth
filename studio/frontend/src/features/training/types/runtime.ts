// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TrainingPhase =
  | "idle"
  | "downloading_model"
  | "downloading_dataset"
  | "loading_model"
  | "loading_dataset"
  | "configuring"
  | "training"
  | "completed"
  | "error"
  | "stopped";

export interface TrainingStatusResponse {
  job_id: string;
  phase: TrainingPhase;
  is_training_running: boolean;
  eval_enabled: boolean;
  message: string;
  error: string | null;
  details?: {
    epoch?: number;
    step?: number;
    total_steps?: number;
    loss?: number;
    learning_rate?: number;
  } | null;
  metric_history?: {
    steps?: number[];
    loss?: number[];
    lr?: number[];
    grad_norm?: number[];
    grad_norm_steps?: number[];
    eval_loss?: number[];
    eval_steps?: number[];
  } | null;
}

export interface TrainingMetricsResponse {
  loss_history: number[];
  lr_history: number[];
  step_history: number[];
  grad_norm_history: number[];
  grad_norm_step_history: number[];
  current_loss: number | null;
  current_lr: number | null;
  current_step: number | null;
}

export interface TrainingProgressPayload {
  job_id: string;
  step: number;
  total_steps: number;
  loss: number | null;
  learning_rate: number | null;
  progress_percent: number;
  epoch: number | null;
  elapsed_seconds: number | null;
  eta_seconds: number | null;
  grad_norm: number | null;
  num_tokens: number | null;
  eval_loss: number | null;
}

export interface TrainingSeriesPoint {
  step: number;
  value: number;
}

export interface TrainingRuntimeState {
  jobId: string | null;
  phase: TrainingPhase;
  isTrainingRunning: boolean;
  evalEnabled: boolean;
  message: string;
  error: string | null;
  isHydrating: boolean;
  hasHydrated: boolean;
  isStarting: boolean;
  startError: string | null;
  sseConnected: boolean;
  firstStepReceived: boolean;
  lastEventId: number | null;
  currentStep: number;
  totalSteps: number;
  currentEpoch: number;
  currentLoss: number;
  currentLearningRate: number;
  progressPercent: number;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  currentGradNorm: number | null;
  currentNumTokens: number | null;
  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
  resetGeneration: number;
  stopRequested: boolean;
  selectedHistoryRunId: string | null;
}

export interface TrainingRuntimeActions {
  setStopRequested: (value: boolean) => void;
  setHydrating: (value: boolean) => void;
  setHasHydrated: (value: boolean) => void;
  setStarting: (value: boolean) => void;
  setStartError: (value: string | null) => void;
  setSseConnected: (value: boolean) => void;
  setLastEventId: (value: number | null) => void;
  resetRuntime: () => void;
  applyStatus: (payload: TrainingStatusResponse) => void;
  applyMetrics: (payload: TrainingMetricsResponse) => void;
  applyProgress: (payload: TrainingProgressPayload, eventId?: number) => void;
  setStartQueued: (jobId: string, message: string) => void;
  setRuntimeError: (message: string) => void;
  setSelectedHistoryRunId: (id: string | null) => void;
}

export type TrainingRuntimeStore = TrainingRuntimeState & TrainingRuntimeActions;

export interface TrainingViewData {
  // Current metrics (for ProgressSection)
  phase: TrainingPhase;
  currentStep: number;
  totalSteps: number;
  currentLoss: number | null;
  currentLearningRate: number | null;
  currentGradNorm: number | null;
  currentEpoch: number | null;
  currentNumTokens: number | null;
  progressPercent: number;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  evalEnabled: boolean;
  message: string;
  error: string | null;
  isTrainingRunning: boolean;

  // Config summary
  modelName: string;
  trainingMethod: string;

  // Time-series (for ChartsSection)
  lossHistory: TrainingSeriesPoint[];
  lrHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
  evalLossHistory: TrainingSeriesPoint[];
}
