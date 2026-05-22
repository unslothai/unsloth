// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type {
  TrainingMetricsResponse,
  TrainingProgressPayload,
  TrainingRuntimeState,
  TrainingRuntimeStore,
  TrainingSeriesPoint,
  TrainingStatusResponse,
} from "../types/runtime";

const initialState: TrainingRuntimeState = {
  jobId: null,
  phase: "idle",
  isTrainingRunning: false,
  evalEnabled: false,
  message: "Ready to train",
  error: null,
  isHydrating: false,
  hasHydrated: false,
  isStarting: false,
  startError: null,
  startModelName: null,
  startDatasetName: null,
  startFromResume: false,
  sseConnected: false,
  firstStepReceived: false,
  lastEventId: null,
  currentStep: 0,
  totalSteps: 0,
  currentEpoch: 0,
  currentLoss: 0,
  currentLearningRate: 0,
  progressPercent: 0,
  elapsedSeconds: null,
  etaSeconds: null,
  currentGradNorm: null,
  currentNumTokens: null,
  outputDir: null,
  lossHistory: [],
  lrHistory: [],
  gradNormHistory: [],
  evalLossHistory: [],
  resetGeneration: 0,
  stopRequested: false,
  selectedHistoryRunId: null,
};

function sortSeries(points: TrainingSeriesPoint[]): TrainingSeriesPoint[] {
  return [...points].sort((a, b) => a.step - b.step);
}

function toSeries(steps: number[], values: number[]): TrainingSeriesPoint[] {
  const points: TrainingSeriesPoint[] = [];
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    const value = values[i];
    if (!Number.isFinite(step) || !Number.isFinite(value)) {
      continue;
    }
    points.push({ step, value });
  }
  return sortSeries(points);
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number") return null;
  return Number.isFinite(value) ? value : null;
}

function upsertPoint(
  points: TrainingSeriesPoint[],
  step: number,
  value: number,
): TrainingSeriesPoint[] {
  const next = points.slice();
  const index = next.findIndex((point) => point.step === step);
  if (index >= 0) {
    next[index] = { step, value };
    return next;
  }
  next.push({ step, value });
  return sortSeries(next);
}

function applyMetricHistoryFromStatus(payload: TrainingStatusResponse): {
  lossHistory: TrainingSeriesPoint[] | null;
  lrHistory: TrainingSeriesPoint[] | null;
  gradNormHistory: TrainingSeriesPoint[] | null;
  evalLossHistory: TrainingSeriesPoint[] | null;
} {
  const history = payload.metric_history;
  if (!history || !history.steps?.length) {
    return {
      lossHistory: null,
      lrHistory: null,
      gradNormHistory: null,
      evalLossHistory: null,
    };
  }

  const steps = history.steps;
  const lossHistory = history.loss ? toSeries(steps, history.loss) : null;
  const lrHistory = history.lr ? toSeries(steps, history.lr) : null;
  const gradNormHistory =
    history.grad_norm && history.grad_norm_steps
      ? toSeries(history.grad_norm_steps, history.grad_norm)
      : null;
  const evalLossHistory =
    history.eval_loss && history.eval_steps
      ? toSeries(history.eval_steps, history.eval_loss)
      : null;

  return { lossHistory, lrHistory, gradNormHistory, evalLossHistory };
}

export const useTrainingRuntimeStore = create<TrainingRuntimeStore>()((set) => ({
  ...initialState,

  setStopRequested: (value) => set({ stopRequested: value }),
  setHydrating: (value) => set({ isHydrating: value }),
  setHasHydrated: (value) => set({ hasHydrated: value }),
  setStarting: (value) => set({ isStarting: value }),
  setStartError: (value) => set({ startError: value }),
  setStartResources: (startModelName, startDatasetName, startFromResume = false) =>
    set({ startModelName, startDatasetName, startFromResume }),
  setSseConnected: (value) => set({ sseConnected: value }),
  setLastEventId: (value) => set({ lastEventId: value }),

  resetRuntime: () =>
    set((state) => ({
      ...initialState,
      hasHydrated: state.hasHydrated,
      lossHistory: [],
      lrHistory: [],
      gradNormHistory: [],
      evalLossHistory: [],
      resetGeneration: state.resetGeneration + 1,
    })),

  setStartQueued: (jobId, message) =>
    set((state) => ({
      ...state,
      jobId,
      message,
      error: null,
      startError: null,
      phase: "configuring",
      isStarting: false,
      sseConnected: false,
      firstStepReceived: false,
      lastEventId: null,
      currentStep: 0,
      totalSteps: 0,
      currentEpoch: 0,
      currentLoss: 0,
      currentLearningRate: 0,
      progressPercent: 0,
      elapsedSeconds: null,
      etaSeconds: null,
      currentGradNorm: null,
      currentNumTokens: null,
      outputDir: null,
      lossHistory: [],
      lrHistory: [],
      gradNormHistory: [],
      evalLossHistory: [],
      resetGeneration: state.resetGeneration + 1,
    })),

  setRuntimeError: (message) =>
    set({
      error: message,
      phase: "error",
      isStarting: false,
      startError: null,
      sseConnected: false,
    }),

  setSelectedHistoryRunId: (selectedHistoryRunId) =>
    set({ selectedHistoryRunId }),

  applyStatus: (payload) =>
    set((state) => {
      const metricHistory = applyMetricHistoryFromStatus(payload);
      const detailStep = payload.details?.step;
      const detailTotal = payload.details?.total_steps;
      const detailLoss = payload.details?.loss;
      const detailLr = payload.details?.learning_rate;
      const detailEpoch = payload.details?.epoch;
      const stopRequested =
        payload.is_training_running ? state.stopRequested : false;

      return {
        ...state,
        jobId: payload.job_id || state.jobId,
        phase: payload.phase,
        isTrainingRunning: payload.is_training_running,
        stopRequested,
        evalEnabled: payload.eval_enabled ?? state.evalEnabled,
        message: payload.message,
        error: payload.error,
        currentStep:
          typeof detailStep === "number" ? Math.max(detailStep, 0) : state.currentStep,
        totalSteps:
          typeof detailTotal === "number"
            ? Math.max(detailTotal, 0)
            : state.totalSteps,
        currentLoss:
          typeof detailLoss === "number" ? detailLoss : state.currentLoss,
        currentLearningRate:
          typeof detailLr === "number" ? detailLr : state.currentLearningRate,
        currentEpoch:
          typeof detailEpoch === "number" ? detailEpoch : state.currentEpoch,
        outputDir: payload.details?.output_dir ?? state.outputDir,
        lossHistory: metricHistory.lossHistory ?? state.lossHistory,
        lrHistory: metricHistory.lrHistory ?? state.lrHistory,
        gradNormHistory: metricHistory.gradNormHistory ?? state.gradNormHistory,
        evalLossHistory: metricHistory.evalLossHistory ?? state.evalLossHistory,
      };
    }),

  applyMetrics: (payload: TrainingMetricsResponse) =>
    set((state) => {
      const lossHistory = toSeries(payload.step_history, payload.loss_history);
      const lrHistory = toSeries(payload.step_history, payload.lr_history);
      const gradNormHistory = toSeries(
        payload.grad_norm_step_history,
        payload.grad_norm_history,
      );
      const latestStep =
        payload.current_step ??
        (payload.step_history.length > 0
          ? payload.step_history[payload.step_history.length - 1]
          : null);

      return {
        ...state,
        lossHistory: lossHistory.length > 0 ? lossHistory : state.lossHistory,
        lrHistory: lrHistory.length > 0 ? lrHistory : state.lrHistory,
        gradNormHistory:
          gradNormHistory.length > 0 ? gradNormHistory : state.gradNormHistory,
        currentStep:
          typeof latestStep === "number"
            ? Math.max(latestStep, state.currentStep)
            : state.currentStep,
        currentLoss:
          typeof payload.current_loss === "number"
            ? payload.current_loss
            : state.currentLoss,
        currentLearningRate:
          typeof payload.current_lr === "number"
            ? payload.current_lr
            : state.currentLearningRate,
      };
    }),

  applyProgress: (payload: TrainingProgressPayload, eventId?: number) =>
    set((state) => {
      const step = Math.max(payload.step, 0);
      const currentLoss = toFiniteNumber(payload.loss);
      const currentLearningRate = toFiniteNumber(payload.learning_rate);
      const currentGradNorm = toFiniteNumber(payload.grad_norm);
      const evalLoss = toFiniteNumber(payload.eval_loss);

      return {
        ...state,
        jobId: payload.job_id || state.jobId,
        currentStep: step,
        totalSteps: Math.max(payload.total_steps, state.totalSteps),
        currentLoss: currentLoss ?? state.currentLoss,
        currentLearningRate: currentLearningRate ?? state.currentLearningRate,
        progressPercent: payload.progress_percent,
        currentEpoch: payload.epoch ?? state.currentEpoch,
        elapsedSeconds: payload.elapsed_seconds,
        etaSeconds: payload.eta_seconds,
        currentGradNorm,
        currentNumTokens: payload.num_tokens,
        firstStepReceived: state.firstStepReceived || step > 0,
        lastEventId: typeof eventId === "number" ? eventId : state.lastEventId,
        lossHistory:
          step > 0 && currentLoss !== null
            ? upsertPoint(state.lossHistory, step, currentLoss)
            : state.lossHistory,
        lrHistory:
          step > 0 && currentLearningRate !== null
            ? upsertPoint(state.lrHistory, step, currentLearningRate)
            : state.lrHistory,
        gradNormHistory:
          step > 0 && currentGradNorm !== null
            ? upsertPoint(state.gradNormHistory, step, currentGradNorm)
            : state.gradNormHistory,
        evalLossHistory:
          step > 0 && evalLoss !== null
            ? upsertPoint(state.evalLossHistory, step, evalLoss)
            : state.evalLossHistory,
      };
    }),
}));

export function shouldShowTrainingView(state: TrainingRuntimeStore): boolean {
  return (
    state.phase !== "idle" ||
    state.isTrainingRunning ||
    state.isStarting ||
    state.lossHistory.length > 0 ||
    state.currentStep > 0
  );
}
