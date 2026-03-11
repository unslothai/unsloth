// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { useTrainingRuntimeStore } from "@/features/training";
import { type ReactElement, Suspense, lazy, useMemo } from "react";

const ChartsContent = lazy(() =>
  import("./charts-content").then((module) => ({
    default: module.ChartsContent,
  })),
);
const SKELETON_KEYS = [
  "chart-skeleton-1",
  "chart-skeleton-2",
  "chart-skeleton-3",
  "chart-skeleton-4",
];

export function ChartsSection(): ReactElement | null {
  const currentStep = useTrainingRuntimeStore((state) => state.currentStep);
  const totalSteps = useTrainingRuntimeStore((state) => state.totalSteps);
  const isTraining = useTrainingRuntimeStore((state) => state.isTrainingRunning);
  const evalEnabled = useTrainingRuntimeStore((state) => state.evalEnabled);
  const lossHistoryRaw = useTrainingRuntimeStore((state) => state.lossHistory);
  const lrHistoryRaw = useTrainingRuntimeStore((state) => state.lrHistory);
  const gradNormHistoryRaw = useTrainingRuntimeStore(
    (state) => state.gradNormHistory,
  );
  const evalLossHistoryRaw = useTrainingRuntimeStore(
    (state) => state.evalLossHistory,
  );

  const series = useMemo(
    () => ({
      currentStep,
      totalSteps,
      lossHistory: lossHistoryRaw.map((point) => ({
        step: point.step,
        loss: point.value,
      })),
      lrHistory: lrHistoryRaw.map((point) => ({
        step: point.step,
        lr: point.value,
      })),
      gradNormHistory: gradNormHistoryRaw.map((point) => ({
        step: point.step,
        gradNorm: point.value,
      })),
      evalLossHistory: evalLossHistoryRaw.map((point) => ({
        step: point.step,
        loss: point.value,
      })),
    }),
    [currentStep, evalLossHistoryRaw, gradNormHistoryRaw, lossHistoryRaw, lrHistoryRaw, totalSteps],
  );

  if (
    series.lossHistory.length === 0 &&
    series.lrHistory.length === 0 &&
    series.gradNormHistory.length === 0
  ) {
    return null;
  }

  return (
    <Suspense
      fallback={
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {SKELETON_KEYS.map((key) => (
            <div
              key={key}
              className="h-[280px] rounded-xl border bg-muted/30 animate-pulse"
            />
          ))}
        </div>
      }
    >
      <ChartsContent metrics={series} isTraining={isTraining} evalEnabled={evalEnabled} />
    </Suspense>
  );
}
