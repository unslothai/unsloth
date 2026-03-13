// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useChartPreferencesStore } from "./charts/chart-preferences-store";
import { EvalLossChartCard } from "./charts/eval-loss-chart-card";
import { GradNormChartCard } from "./charts/grad-norm-chart-card";
import { LearningRateChartCard } from "./charts/learning-rate-chart-card";
import { TrainingLossChartCard } from "./charts/training-loss-chart-card";
import type { TrainingChartSeries } from "./charts/types";
import {
  MAX_RENDER_POINTS,
  applyOutlierCap,
  buildStepTicks,
  buildYDomain,
  clamp,
  compressSeries,
  ema,
  toLog1p,
} from "./charts/utils";

type LossDisplayPoint = {
  step: number;
  displayLoss: number;
  displaySmoothed: number;
};

function isStepVisible(step: number, domain: [number, number]): boolean {
  return step >= domain[0] && step <= domain[1];
}

function collectLossValues(
  data: LossDisplayPoint[],
  domain: [number, number],
  options: { includeRaw: boolean; includeSmoothed: boolean },
): number[] {
  const values: number[] = [];

  for (const point of data) {
    if (!isStepVisible(point.step, domain)) {
      continue;
    }

    if (options.includeRaw && Number.isFinite(point.displayLoss)) {
      values.push(point.displayLoss);
    }

    if (options.includeSmoothed && Number.isFinite(point.displaySmoothed)) {
      values.push(point.displaySmoothed);
    }
  }

  return values;
}

export function ChartsContent({
  metrics,
  isTraining,
  evalEnabled,
}: {
  metrics: TrainingChartSeries;
  isTraining: boolean;
  evalEnabled: boolean;
}): ReactElement {
  const {
    windowSize,
    smoothing,
    showRaw,
    showSmoothed,
    showAvgLine,
    lossScale,
    lrScale,
    gradScale,
    lossOutlierMode,
    gradOutlierMode,
    lrOutlierMode,
    setAvailableSteps,
  } = useChartPreferencesStore(
    useShallow((state) => ({
      windowSize: state.windowSize,
      smoothing: state.smoothing,
      showRaw: state.showRaw,
      showSmoothed: state.showSmoothed,
      showAvgLine: state.showAvgLine,
      lossScale: state.lossScale,
      lrScale: state.lrScale,
      gradScale: state.gradScale,
      lossOutlierMode: state.lossOutlierMode,
      gradOutlierMode: state.gradOutlierMode,
      lrOutlierMode: state.lrOutlierMode,
      setAvailableSteps: state.setAvailableSteps,
    })),
  );

  const smoothedData = useMemo(
    () =>
      metrics.lossHistory.length > 0 ? ema(metrics.lossHistory, smoothing) : [],
    [metrics.lossHistory, smoothing],
  );

  const reducedLossData = useMemo(
    () => compressSeries(smoothedData, MAX_RENDER_POINTS),
    [smoothedData],
  );
  const reducedGradNormData = useMemo(
    () => compressSeries(metrics.gradNormHistory, MAX_RENDER_POINTS),
    [metrics.gradNormHistory],
  );
  const reducedLrData = useMemo(
    () => compressSeries(metrics.lrHistory, MAX_RENDER_POINTS),
    [metrics.lrHistory],
  );
  const reducedEvalLossData = useMemo(
    () => compressSeries(metrics.evalLossHistory, MAX_RENDER_POINTS),
    [metrics.evalLossHistory],
  );

  const allSteps = useMemo(() => {
    const set = new Set<number>();
    for (const point of metrics.lossHistory) {
      set.add(point.step);
    }
    for (const point of metrics.gradNormHistory) {
      set.add(point.step);
    }
    for (const point of metrics.lrHistory) {
      set.add(point.step);
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [metrics.gradNormHistory, metrics.lossHistory, metrics.lrHistory]);

  useEffect(() => {
    setAvailableSteps(allSteps.length);
  }, [allSteps.length, setAvailableSteps]);

  const stepCount = Math.max(1, allSteps.length);
  const effectiveWindowSize =
    windowSize == null
      ? stepCount
      : clamp(Math.round(windowSize), 1, stepCount);

  const visibleStepDomain = useMemo<[number, number]>(() => {
    if (allSteps.length === 0) {
      return [0, 1];
    }

    const endIndex = allSteps.length - 1;
    const startIndex = Math.max(0, endIndex - effectiveWindowSize + 1);
    const minStep = allSteps[0] ?? 0;
    const startStep = allSteps[startIndex] ?? minStep;
    const endStep = allSteps[endIndex] ?? startStep;

    if (startStep === endStep) {
      return [startStep, startStep + 4];
    }
    if (endStep - startStep < 6) {
      return [Math.max(minStep, endStep - 6), endStep];
    }
    return [startStep, endStep];
  }, [allSteps, effectiveWindowSize]);

  const xAxisTicks = useMemo(
    () => buildStepTicks(visibleStepDomain[0], visibleStepDomain[1]),
    [visibleStepDomain],
  );

  const displayLossData = useMemo(
    () =>
      reducedLossData.map((point) => ({
        ...point,
        displayLoss: lossScale === "log" ? toLog1p(point.loss) : point.loss,
        displaySmoothed:
          lossScale === "log" ? toLog1p(point.smoothed) : point.smoothed,
      })),
    [lossScale, reducedLossData],
  );

  const displayGradData = useMemo(
    () =>
      reducedGradNormData.map((point) => ({
        ...point,
        displayGradNorm:
          gradScale === "log" ? toLog1p(point.gradNorm) : point.gradNorm,
      })),
    [gradScale, reducedGradNormData],
  );

  const displayLrData = useMemo(
    () =>
      reducedLrData.map((point) => ({
        ...point,
        displayLr: lrScale === "log" ? toLog1p(point.lr) : point.lr,
      })),
    [lrScale, reducedLrData],
  );

  const visibleLossDisplayValues = useMemo(() => {
    const visibleValues = collectLossValues(
      displayLossData,
      visibleStepDomain,
      {
        includeRaw: showRaw,
        includeSmoothed: showSmoothed,
      },
    );

    if (visibleValues.length > 0) {
      return visibleValues;
    }

    return collectLossValues(displayLossData, visibleStepDomain, {
      includeRaw: true,
      includeSmoothed: true,
    });
  }, [displayLossData, showRaw, showSmoothed, visibleStepDomain]);

  const visibleGradDisplayValues = useMemo(
    () =>
      displayGradData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] &&
            point.step <= visibleStepDomain[1],
        )
        .map((point) => point.displayGradNorm)
        .filter((value) => Number.isFinite(value)),
    [displayGradData, visibleStepDomain],
  );

  const visibleLrDisplayValues = useMemo(
    () =>
      displayLrData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] &&
            point.step <= visibleStepDomain[1],
        )
        .map((point) => point.displayLr)
        .filter((value) => Number.isFinite(value)),
    [displayLrData, visibleStepDomain],
  );

  const lossDomain = useMemo(
    () =>
      buildYDomain(applyOutlierCap(visibleLossDisplayValues, lossOutlierMode)),
    [lossOutlierMode, visibleLossDisplayValues],
  );
  const gradDomain = useMemo(
    () =>
      buildYDomain(applyOutlierCap(visibleGradDisplayValues, gradOutlierMode)),
    [gradOutlierMode, visibleGradDisplayValues],
  );
  const lrDomain = useMemo(
    () => buildYDomain(applyOutlierCap(visibleLrDisplayValues, lrOutlierMode)),
    [lrOutlierMode, visibleLrDisplayValues],
  );

  const evalLossDomain = useMemo(() => {
    const vals = reducedEvalLossData.map((point) => point.loss);
    return buildYDomain(vals);
  }, [reducedEvalLossData]);

  const evalLossStepTicks = useMemo(() => {
    if (reducedEvalLossData.length < 2) {
      return undefined;
    }
    const min = reducedEvalLossData[0].step;
    const max = reducedEvalLossData[reducedEvalLossData.length - 1].step;
    return buildStepTicks(min, max);
  }, [reducedEvalLossData]);

  const avgRaw =
    metrics.lossHistory.length > 0
      ? +(
          metrics.lossHistory.reduce((sum, point) => sum + point.loss, 0) /
          metrics.lossHistory.length
        ).toFixed(4)
      : 0;
  const avgDisplay = lossScale === "log" ? toLog1p(avgRaw) : avgRaw;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <TrainingLossChartCard
        data={displayLossData}
        domain={lossDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        avgRaw={avgRaw}
        avgDisplay={avgDisplay}
        showRaw={showRaw}
        showSmoothed={showSmoothed}
        showAvgLine={showAvgLine}
        scale={lossScale}
      />
      <GradNormChartCard
        data={displayGradData}
        domain={gradDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        scale={gradScale}
      />
      <LearningRateChartCard
        data={displayLrData}
        domain={lrDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        scale={lrScale}
      />
      <EvalLossChartCard
        data={reducedEvalLossData}
        domain={evalLossDomain}
        ticks={evalLossStepTicks}
        isTraining={isTraining}
        evalEnabled={evalEnabled}
      />
    </div>
  );
}
