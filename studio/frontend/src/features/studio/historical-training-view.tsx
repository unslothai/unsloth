// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  type TrainingRunDetailResponse,
  type TrainingViewData,
  getTrainingRun,
  onTrainingRunUpdated,
  parseBackendTrainingMethod,
  useTrainingActions,
} from "@/features/training";
import { translate, useT } from "@/i18n";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { ChartsSection } from "./sections/charts-section";
import { ProgressSection } from "./sections/progress-section";
import { mapRunConfigToOverride } from "./sections/run-config-override";

type StudioT = ReturnType<typeof useT>;

interface HistoricalTrainingViewProps {
  runId: string;
  onResumeStarted?: () => void;
}

type HistoricalRunResult = {
  runId: string;
  detail: TrainingRunDetailResponse | null;
  error: string | null;
};

function mapToViewData(
  detail: TrainingRunDetailResponse,
  t: StudioT,
): TrainingViewData {
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
    resumedLater: run.resumed_later ?? false,
    progressPercent:
      run.total_steps && run.final_step
        ? (run.final_step / run.total_steps) * 100
        : 0,
    elapsedSeconds: run.duration_seconds,
    etaSeconds: null,
    evalEnabled: evalLossHistory.length > 0,
    message:
      run.status === "completed"
        ? t("studio.history.message.completed")
        : run.status === "stopped"
          ? t("studio.history.message.stopped")
          : run.status === "running"
            ? t("studio.history.message.running")
            : (run.error_message ?? t("studio.history.message.errored")),
    error: run.status === "error" ? run.error_message : null,
    warnings: [],
    isTrainingRunning: false,
    modelName: run.display_name ?? run.model_name,
    projectName: run.project_name,
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
  onResumeStarted,
}: HistoricalTrainingViewProps): ReactElement {
  const t = useT();
  const [result, setResult] = useState<HistoricalRunResult | null>(null);
  const [resuming, setResuming] = useState(false);
  const { resumeTrainingRunFromHistory, startBlocked, stopRequested } =
    useTrainingActions();
  const currentResult = result?.runId === runId ? result : null;
  const detail = currentResult?.detail ?? null;
  const error = currentResult?.error ?? null;

  const handleResume = async () => {
    setResuming(true);
    try {
      const ok = await resumeTrainingRunFromHistory(runId);
      if (ok) onResumeStarted?.();
    } finally {
      setResuming(false);
    }
  };

  const loading = currentResult === null;

  useEffect(() => {
    const controller = new AbortController();
    getTrainingRun(runId, controller.signal)
      .then((detail) => {
        setResult({
          runId,
          detail,
          error: null,
        });
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setResult({
          runId,
          detail: null,
          error:
            err instanceof Error
              ? err.message
              : translate("studio.history.loadingRun"),
        });
      });
    return () => {
      controller.abort();
    };
  }, [runId]);

  useEffect(() => {
    const offUpdated = onTrainingRunUpdated((updated) => {
      if (updated.id !== runId) return;
      setResult((previous) =>
        previous?.runId === runId && previous.detail
          ? {
              ...previous,
              detail: {
                ...previous.detail,
                run: updated,
              },
            }
          : previous,
      );
    });
    return offUpdated;
  }, [runId]);

  if (loading) {
    return (
      <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
        {t("studio.history.loadingRun")}
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-8 text-sm text-red-500">
        {error ?? t("studio.history.runNotFound")}
      </div>
    );
  }

  const viewData = mapToViewData(detail, t);
  const configOverride = mapRunConfigToOverride(detail.config);

  return (
    <div className="flex flex-col gap-6">
      {detail.run.can_resume && (
        <div className="flex justify-end">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="gap-1.5"
            disabled={startBlocked || resuming}
            onClick={() => void handleResume()}
          >
            {resuming ? (
              <Spinner className="size-3.5" />
            ) : (
              <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
            )}
            {stopRequested
              ? t("studio.training.stopping")
              : resuming
                ? t("studio.history.resuming")
                : t("studio.history.resumeTraining")}
          </Button>
        </div>
      )}
      <ProgressSection
        data={viewData}
        isHistorical={true}
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
