// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { isTrainingProgressForJob } from "../lib/training-stream-scope.ts";
import type {
  TrainingMetricsResponse,
  TrainingPhase,
  TrainingProgressPayload,
  TrainingRuntimeState,
  TrainingRuntimeStore,
  TrainingSeriesPoint,
  TrainingStatusResponse,
} from "../types/runtime";

const ACTIVE_TRAINING_PHASES = new Set<TrainingPhase>([
  "downloading_model",
  "downloading_dataset",
  "loading_model",
  "loading_dataset",
  "configuring",
  "training",
]);

export function isTrainingRunActive(
  state: Pick<TrainingRuntimeState, "phase" | "isTrainingRunning">,
): boolean {
  return state.isTrainingRunning || ACTIVE_TRAINING_PHASES.has(state.phase);
}

export function isTrainingStartPending(
  state: Pick<
    TrainingRuntimeState,
    | "phase"
    | "isStarting"
    | "isTrainingRunning"
    | "stopRequested"
    | "startRequestId"
  >,
): boolean {
  return (
    state.stopRequested ||
    state.isStarting ||
    Boolean(state.startRequestId?.trim()) ||
    isTrainingRunActive(state)
  );
}

const initialState: TrainingRuntimeState = {
  jobId: null,
  phase: "idle",
  isTrainingRunning: false,
  evalEnabled: false,
  message: "Ready to train",
  error: null,
  warnings: [],
  isHydrating: false,
  hasHydrated: false,
  isStarting: false,
  startRequestId: null,
  startError: null,
  startModelName: null,
  startDatasetName: null,
  startProjectName: null,
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
  rewardHistory: [],
  resetGeneration: 0,
  stopRequested: false,
  selectedHistoryRunId: null,
  currentRunViewActive: false,
};

function sortSeries(points: TrainingSeriesPoint[]): TrainingSeriesPoint[] {
  return [...points].sort((a, b) => a.step - b.step);
}

function toSeries(steps: number[], values: number[]): TrainingSeriesPoint[] {
  const points = new Map<number, number>();
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    const value = values[i];
    if (!Number.isFinite(step) || !Number.isFinite(value)) {
      continue;
    }
    points.set(step, value);
  }
  return sortSeries(Array.from(points, ([step, value]) => ({ step, value })));
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number") return null;
  return Number.isFinite(value) ? value : null;
}

function normalizeWarnings(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const warnings: string[] = [];
  const seen = new Set<string>();
  for (const candidate of value) {
    if (typeof candidate !== "string") continue;
    const warning = candidate.trim();
    if (!warning || seen.has(warning)) continue;
    seen.add(warning);
    warnings.push(warning);
  }
  return warnings;
}

function preserveEqualStrings(current: string[], incoming: string[]): string[] {
  if (
    current.length === incoming.length &&
    current.every((value, index) => value === incoming[index])
  ) {
    return current;
  }
  return incoming;
}

function upsertPoint(
  points: TrainingSeriesPoint[],
  step: number,
  value: number,
): TrainingSeriesPoint[] {
  const next = points.slice();
  const index = next.findIndex((point) => point.step === step);
  if (index >= 0) {
    if (next[index].value === value) {
      return points;
    }
    next[index] = { step, value };
    return next;
  }
  next.push({ step, value });
  return sortSeries(next);
}

function mergeSeries(
  current: TrainingSeriesPoint[],
  incoming: TrainingSeriesPoint[],
): TrainingSeriesPoint[] {
  if (incoming.length === 0) {
    return current;
  }
  if (current.length === 0) {
    return incoming;
  }

  const merged: TrainingSeriesPoint[] = [];
  let currentIndex = 0;
  let incomingIndex = 0;
  let added = false;

  while (currentIndex < current.length && incomingIndex < incoming.length) {
    const currentPoint = current[currentIndex];
    const incomingPoint = incoming[incomingIndex];
    if (currentPoint.step === incomingPoint.step) {
      merged.push(currentPoint);
      currentIndex += 1;
      incomingIndex += 1;
    } else if (currentPoint.step < incomingPoint.step) {
      merged.push(currentPoint);
      currentIndex += 1;
    } else {
      merged.push(incomingPoint);
      incomingIndex += 1;
      added = true;
    }
  }

  if (currentIndex < current.length) {
    merged.push(...current.slice(currentIndex));
  }
  if (incomingIndex < incoming.length) {
    merged.push(...incoming.slice(incomingIndex));
    added = true;
  }

  return added ? merged : current;
}

function applyMetricHistoryFromStatus(payload: TrainingStatusResponse): {
  lossHistory: TrainingSeriesPoint[] | null;
  lrHistory: TrainingSeriesPoint[] | null;
  gradNormHistory: TrainingSeriesPoint[] | null;
  evalLossHistory: TrainingSeriesPoint[] | null;
  rewardHistory: TrainingSeriesPoint[] | null;
} {
  const history = payload.metric_history;
  if (!history || !history.steps?.length) {
    return {
      lossHistory: null,
      lrHistory: null,
      gradNormHistory: null,
      evalLossHistory: null,
      rewardHistory: null,
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

  const rewardHistory =
    history.reward && history.reward_steps
      ? toSeries(history.reward_steps, history.reward)
      : null;

  return {
    lossHistory,
    lrHistory,
    gradNormHistory,
    evalLossHistory,
    rewardHistory,
  };
}

export const useTrainingRuntimeStore = create<TrainingRuntimeStore>()(
  (set) => ({
    ...initialState,

    setStopRequested: (value) =>
      set((state) => ({
        stopRequested: value,
        isStarting: value ? false : state.isStarting,
        resetGeneration:
          value && !state.stopRequested
            ? state.resetGeneration + 1
            : state.resetGeneration,
      })),
    setHydrating: (value) => set({ isHydrating: value }),
    setHasHydrated: (value) => set({ hasHydrated: value }),
    tryBeginStarting: (startRequestId) => {
      let acquired = false;
      set((state) => {
        if (!startRequestId || isTrainingStartPending(state)) {
          return state;
        }
        acquired = true;
        return { isStarting: true, startRequestId };
      });
      return acquired;
    },
    setStarting: (value) =>
      set((state) => ({
        isStarting: value,
        startRequestId: value ? state.startRequestId : null,
      })),
    setStartError: (value) => set({ startError: value }),
    setStartResources: (
      startModelName,
      startDatasetName,
      startFromResume = false,
      startProjectName = null,
    ) =>
      set({
        startModelName,
        startDatasetName,
        startProjectName,
        startFromResume,
      }),
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
        rewardHistory: [],
        resetGeneration: state.resetGeneration + 1,
      })),

    setStartPending: (jobId, message, startRequestId = null) =>
      set((state) => {
        if (jobId !== null && state.jobId === jobId) {
          return {
            isStarting: false,
            startRequestId,
            startError: null,
          };
        }
        return {
          ...state,
          jobId,
          message,
          error: null,
          warnings: [],
          startError: null,
          phase: "configuring",
          isStarting: false,
          startRequestId,
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
          rewardHistory: [],
          resetGeneration: state.resetGeneration + 1,
        };
      }),

    setRuntimeError: (message) =>
      set((state) => ({
        error: message,
        phase: "error",
        isStarting: false,
        startRequestId: null,
        startError: null,
        sseConnected: false,
        resetGeneration: state.resetGeneration + 1,
      })),

    setSelectedHistoryRunId: (selectedHistoryRunId) =>
      set({ selectedHistoryRunId }),

    setCurrentRunViewActive: (currentRunViewActive) =>
      set({ currentRunViewActive }),

    applyStatus: (payload) =>
      set((state) => {
        const unmatchedLocalStart =
          !state.isStarting &&
          state.startRequestId !== null &&
          payload.start_request_id !== state.startRequestId;
        const authoritativeDifferentStart =
          payload.is_training_running ||
          ACTIVE_TRAINING_PHASES.has(payload.phase) ||
          Boolean(payload.start_request_id?.trim());
        if (unmatchedLocalStart && !authoritativeDifferentStart) {
          return state;
        }
        const nextJobId = payload.job_id || state.jobId;
        const changedJob =
          payload.job_id.length > 0 && payload.job_id !== state.jobId;
        const localStartStatus =
          state.startRequestId !== null &&
          payload.start_request_id === state.startRequestId;
        const pendingStartRequestId =
          payload.start_request_state === "pending"
            ? payload.start_request_id?.trim() || null
            : null;
        const nextStartRequestId = state.isStarting
          ? state.startRequestId
          : pendingStartRequestId;
        const runtimeState = changedJob
          ? {
              ...state,
              jobId: payload.job_id,
              warnings: [],
              isStarting: state.isStarting,
              startModelName: localStartStatus ? state.startModelName : null,
              startDatasetName: localStartStatus
                ? state.startDatasetName
                : null,
              startProjectName: localStartStatus
                ? state.startProjectName
                : null,
              startFromResume: localStartStatus ? state.startFromResume : false,
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
              rewardHistory: [],
              resetGeneration: state.resetGeneration + 1,
              stopRequested: false,
            }
          : state;
        const metricHistory = applyMetricHistoryFromStatus(payload);
        const warnings = normalizeWarnings(payload.warnings);
        const nextWarnings = warnings
          ? preserveEqualStrings(runtimeState.warnings, warnings)
          : runtimeState.warnings;
        const detailStep = toFiniteNumber(payload.details?.step);
        const detailTotal = toFiniteNumber(payload.details?.total_steps);
        const detailLoss = toFiniteNumber(payload.details?.loss);
        const detailLr = toFiniteNumber(payload.details?.learning_rate);
        const detailEpoch = toFiniteNumber(payload.details?.epoch);
        const canApplyDetailMetrics =
          detailStep !== null && detailStep >= runtimeState.currentStep;
        const stopRequested = payload.is_training_running
          ? runtimeState.stopRequested
          : false;

        return {
          ...runtimeState,
          jobId: nextJobId,
          startRequestId: nextStartRequestId,
          phase: payload.phase,
          isTrainingRunning: payload.is_training_running,
          stopRequested,
          evalEnabled: payload.eval_enabled ?? runtimeState.evalEnabled,
          message: payload.message,
          error: payload.error,
          warnings: nextWarnings,
          currentStep:
            detailStep !== null
              ? Math.max(detailStep, runtimeState.currentStep, 0)
              : runtimeState.currentStep,
          totalSteps:
            detailTotal !== null && detailTotal > 0
              ? Math.max(detailTotal, runtimeState.totalSteps)
              : runtimeState.totalSteps,
          currentLoss:
            canApplyDetailMetrics && detailLoss !== null
              ? detailLoss
              : runtimeState.currentLoss,
          currentLearningRate:
            canApplyDetailMetrics && detailLr !== null
              ? detailLr
              : runtimeState.currentLearningRate,
          currentEpoch:
            canApplyDetailMetrics && detailEpoch !== null
              ? Math.max(detailEpoch, runtimeState.currentEpoch)
              : runtimeState.currentEpoch,
          outputDir:
            payload.details?.output_dir !== undefined
              ? payload.details.output_dir
              : runtimeState.outputDir,
          lossHistory: metricHistory.lossHistory
            ? mergeSeries(runtimeState.lossHistory, metricHistory.lossHistory)
            : runtimeState.lossHistory,
          lrHistory: metricHistory.lrHistory
            ? mergeSeries(runtimeState.lrHistory, metricHistory.lrHistory)
            : runtimeState.lrHistory,
          gradNormHistory: metricHistory.gradNormHistory
            ? mergeSeries(
                runtimeState.gradNormHistory,
                metricHistory.gradNormHistory,
              )
            : runtimeState.gradNormHistory,
          evalLossHistory: metricHistory.evalLossHistory
            ? mergeSeries(
                runtimeState.evalLossHistory,
                metricHistory.evalLossHistory,
              )
            : runtimeState.evalLossHistory,
          rewardHistory: metricHistory.rewardHistory
            ? mergeSeries(runtimeState.rewardHistory, metricHistory.rewardHistory)
            : runtimeState.rewardHistory,
        };
      }),

    applyMetrics: (payload: TrainingMetricsResponse) =>
      set((state) => {
        if (!isTrainingProgressForJob(state.jobId, payload.job_id)) {
          return state;
        }
        const lossHistory = toSeries(
          payload.step_history,
          payload.loss_history,
        );
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
        const normalizedLatestStep = toFiniteNumber(latestStep);
        const latestLoss = toFiniteNumber(payload.current_loss);
        const latestLearningRate = toFiniteNumber(payload.current_lr);
        const canApplyCurrentMetrics =
          normalizedLatestStep !== null &&
          normalizedLatestStep >= state.currentStep;

        return {
          ...state,
          lossHistory: mergeSeries(state.lossHistory, lossHistory),
          lrHistory: mergeSeries(state.lrHistory, lrHistory),
          gradNormHistory: mergeSeries(state.gradNormHistory, gradNormHistory),
          currentStep:
            normalizedLatestStep !== null
              ? Math.max(normalizedLatestStep, state.currentStep)
              : state.currentStep,
          currentLoss:
            canApplyCurrentMetrics && latestLoss !== null
              ? latestLoss
              : state.currentLoss,
          currentLearningRate:
            canApplyCurrentMetrics && latestLearningRate !== null
              ? latestLearningRate
              : state.currentLearningRate,
        };
      }),

    applyProgress: (payload: TrainingProgressPayload, eventId?: number) =>
      set((state) => {
        if (!isTrainingProgressForJob(state.jobId, payload.job_id)) {
          return state;
        }
        const payloadStep = toFiniteNumber(payload.step);
        const normalizedEventId = toFiniteNumber(eventId);
        if (
          payloadStep === null ||
          payloadStep < state.currentStep ||
          (normalizedEventId !== null &&
            state.lastEventId !== null &&
            normalizedEventId < state.lastEventId)
        ) {
          return state;
        }
        const step = Math.max(payloadStep, 0);
        const currentLoss = toFiniteNumber(payload.loss);
        const currentLearningRate = toFiniteNumber(payload.learning_rate);
        const currentGradNorm = toFiniteNumber(payload.grad_norm);
        const evalLoss = toFiniteNumber(payload.eval_loss);
        const reward = toFiniteNumber(payload.reward);
        const totalSteps = toFiniteNumber(payload.total_steps);
        const progressPercent = toFiniteNumber(payload.progress_percent);
        const currentEpoch = toFiniteNumber(payload.epoch);

        return {
          ...state,
          currentStep: step,
          totalSteps:
            totalSteps !== null && totalSteps > 0
              ? Math.max(totalSteps, state.totalSteps)
              : state.totalSteps,
          // A null loss at a new step means non-finite; clear the display rather than keep a stale value.
          currentLoss:
            currentLoss ??
            (step > state.currentStep ? null : state.currentLoss),
          currentLearningRate: currentLearningRate ?? state.currentLearningRate,
          progressPercent:
            progressPercent !== null
              ? Math.max(
                  state.progressPercent,
                  Math.min(Math.max(progressPercent, 0), 100),
                )
              : state.progressPercent,
          currentEpoch:
            currentEpoch !== null
              ? Math.max(currentEpoch, state.currentEpoch)
              : state.currentEpoch,
          elapsedSeconds: payload.elapsed_seconds,
          etaSeconds: payload.eta_seconds,
          currentGradNorm,
          currentNumTokens: payload.num_tokens,
          firstStepReceived: state.firstStepReceived || step > 0,
          lastEventId:
            normalizedEventId !== null
              ? Math.max(
                  normalizedEventId,
                  state.lastEventId ?? normalizedEventId,
                )
              : state.lastEventId,
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
          rewardHistory:
            step > 0 && reward !== null
              ? upsertPoint(state.rewardHistory, step, reward)
              : state.rewardHistory,
        };
      }),
  }),
);

export function shouldShowTrainingView(state: TrainingRuntimeStore): boolean {
  return (
    state.phase !== "idle" ||
    state.isTrainingRunning ||
    state.isStarting ||
    state.lossHistory.length > 0 ||
    state.currentStep > 0
  );
}
