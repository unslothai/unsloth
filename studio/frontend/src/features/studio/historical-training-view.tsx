// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingViewData } from "@/features/training";
import { getTrainingRun } from "@/features/training";
import type { TrainingRunDetailResponse } from "@/features/training";
import { parseBackendTrainingMethod } from "@/features/training/lib/training-methods";
import { type ReactElement, useEffect, useState } from "react";
import { ChartsSection } from "./sections/charts-section";
import { ProgressSection } from "./sections/progress-section";

interface HistoricalTrainingViewProps {
  runId: string;
}

function mapToViewData(detail: TrainingRunDetailResponse): TrainingViewData {
  const { run, metrics } = detail;

  const lossHistory = metrics.loss_step_history
    .map((step, i) => ({ step, value: metrics.loss_history[i] }))
    .filter((p): p is { step: number; value: number } => p.value != null);

  const lrHistory = metrics.lr_step_history
    .map((step, i) => ({ step, value: metrics.lr_history[i] }))
    .filter((p): p is { step: number; value: number } => p.value != null);

  const gradNormHistory = metrics.grad_norm_step_history
    .map((step, i) => ({ step, value: metrics.grad_norm_history[i] }))
    .filter((p): p is { step: number; value: number } => p.value != null);

  const evalLossHistory = metrics.eval_step_history
    .map((step, i) => ({ step, value: metrics.eval_loss_history[i] }))
    .filter((p): p is { step: number; value: number } => p.value != null);

  const phase =
    run.status === "completed"
      ? "completed"
      : run.status === "stopped"
        ? "stopped"
        : run.status === "error"
          ? "error"
          : run.status === "running"
            ? "training"
            : "idle";

  return {
    phase,
    currentStep: run.final_step ?? 0,
    totalSteps: run.total_steps ?? 0,
    currentLoss: run.final_loss,
    currentLearningRate: metrics.lr_history.at(-1) ?? null,
    currentGradNorm: metrics.grad_norm_history.at(-1) ?? null,
    currentEpoch: metrics.final_epoch,
    currentNumTokens: metrics.final_num_tokens ?? null,
    outputDir: run.output_dir ?? null,
    progressPercent:
      run.total_steps && run.final_step
        ? (run.final_step / run.total_steps) * 100
        : 0,
    elapsedSeconds: run.duration_seconds,
    etaSeconds: null,
    evalEnabled: evalLossHistory.length > 0,
    message:
      run.status === "completed"
        ? "Training completed"
        : run.status === "stopped"
          ? "Training stopped"
          : run.status === "running"
            ? "Training in progress"
            : run.error_message ?? "Training errored",
    error: run.status === "error" ? run.error_message : null,
    isTrainingRunning: false,
    modelName: run.model_name,
    trainingMethod: parseBackendTrainingMethod(
      detail.config?.training_type,
      detail.config?.load_in_4bit,
    ),
    lossHistory,
    lrHistory,
    gradNormHistory,
    evalLossHistory,
  };
}

export function HistoricalTrainingView({
  runId,
}: HistoricalTrainingViewProps): ReactElement {
  const [detail, setDetail] = useState<TrainingRunDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Derive loading from detail/error -- no separate state needed
  const loading = detail === null && error === null;

  useEffect(() => {
    const controller = new AbortController();
    getTrainingRun(runId, controller.signal)
      .then((result) => {
        setDetail(result);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Failed to load run");
      });
    return () => {
      controller.abort();
      // Reset on runId change so loading derives correctly for the next fetch
      setDetail(null);
      setError(null);
    };
  }, [runId]);

  if (loading) {
    return (
      <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
        Loading training run...
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-8 text-sm text-red-500">
        {error ?? "Run not found"}
      </div>
    );
  }

  const viewData = mapToViewData(detail);
  const configOverride = detail.config
    ? {
        epochs: detail.config.num_epochs as number | undefined,
        batchSize: detail.config.batch_size as number | undefined,
        learningRate: detail.config.learning_rate as string | undefined,
        maxSteps: detail.config.max_steps as number | undefined,
        contextLength: detail.config.max_seq_length as number | undefined,
        warmupSteps: detail.config.warmup_steps as number | undefined,
        optimizerType: detail.config.optim as string | undefined,
        loraRank: detail.config.lora_r as number | undefined,
        loraAlpha: detail.config.lora_alpha as number | undefined,
        loraDropout: detail.config.lora_dropout as number | undefined,
        loraVariant: detail.config.use_rslora
          ? "rslora"
          : detail.config.use_loftq
            ? "loftq"
            : "lora",
      }
    : undefined;

  return (
    <div className="flex flex-col gap-6">
      <ProgressSection
        data={viewData}
        isHistorical
        configOverride={configOverride}
      />
      <ChartsSection
        currentStep={viewData.currentStep}
        totalSteps={viewData.totalSteps}
        isTraining={false}
        evalEnabled={viewData.evalEnabled}
        lossHistory={viewData.lossHistory}
        lrHistory={viewData.lrHistory}
        gradNormHistory={viewData.gradNormHistory}
        evalLossHistory={viewData.evalLossHistory}
      />
    </div>
  );
}
