import { type ReactElement, useMemo, useState } from "react";
import { EvalLossChartCard } from "./charts/eval-loss-chart-card";
import { GradNormChartCard } from "./charts/grad-norm-chart-card";
import { LearningRateChartCard } from "./charts/learning-rate-chart-card";
import { TrainingLossChartCard } from "./charts/training-loss-chart-card";
import type { OutlierMode, ScaleMode, TrainingChartSeries, ViewSettingsState } from "./charts/types";
import {
  DEFAULT_VISIBLE_POINTS,
  MAX_RENDER_POINTS,
  applyOutlierCap,
  buildStepTicks,
  buildYDomain,
  clamp,
  compressSeries,
  ema,
  toLog1p,
} from "./charts/utils";

export function ChartsContent({
  metrics,
  isTraining,
  evalEnabled,
}: { metrics: TrainingChartSeries; isTraining: boolean; evalEnabled: boolean }): ReactElement {
  const [smoothing, setSmoothing] = useState(0.75);
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showAvgLine, setShowAvgLine] = useState(true);
  const [windowSize, setWindowSize] = useState<number | null>(
    Math.max(24, Math.floor(DEFAULT_VISIBLE_POINTS / 2)),
  );

  const [lossScale, setLossScale] = useState<ScaleMode>("linear");
  const [lrScale, setLrScale] = useState<ScaleMode>("linear");
  const [gradScale, setGradScale] = useState<ScaleMode>("linear");

  const [lossOutlierMode, setLossOutlierMode] = useState<OutlierMode>("none");
  const [gradOutlierMode, setGradOutlierMode] = useState<OutlierMode>("none");
  const [lrOutlierMode, setLrOutlierMode] = useState<OutlierMode>("none");

  const smoothedData = useMemo(
    () => (metrics.lossHistory.length > 0 ? ema(metrics.lossHistory, 1 - smoothing) : []),
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
    for (const point of reducedLossData) set.add(point.step);
    for (const point of reducedGradNormData) set.add(point.step);
    for (const point of reducedLrData) set.add(point.step);
    return Array.from(set).sort((a, b) => a - b);
  }, [reducedGradNormData, reducedLossData, reducedLrData]);

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
    const values: number[] = [];

    for (const point of displayLossData) {
      if (point.step < visibleStepDomain[0] || point.step > visibleStepDomain[1]) {
        continue;
      }
      if (showRaw && Number.isFinite(point.displayLoss)) {
        values.push(point.displayLoss);
      }
      if (showSmoothed && Number.isFinite(point.displaySmoothed)) {
        values.push(point.displaySmoothed);
      }
    }

    if (values.length === 0) {
      for (const point of displayLossData) {
        if (point.step < visibleStepDomain[0] || point.step > visibleStepDomain[1]) {
          continue;
        }
        if (Number.isFinite(point.displayLoss)) {
          values.push(point.displayLoss);
        }
        if (Number.isFinite(point.displaySmoothed)) {
          values.push(point.displaySmoothed);
        }
      }
    }

    return values;
  }, [displayLossData, showRaw, showSmoothed, visibleStepDomain]);

  const visibleGradDisplayValues = useMemo(
    () =>
      displayGradData
        .filter(
          (point) =>
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
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
            point.step >= visibleStepDomain[0] && point.step <= visibleStepDomain[1],
        )
        .map((point) => point.displayLr)
        .filter((value) => Number.isFinite(value)),
    [displayLrData, visibleStepDomain],
  );

  const lossDomain = useMemo(
    () => buildYDomain(applyOutlierCap(visibleLossDisplayValues, lossOutlierMode)),
    [lossOutlierMode, visibleLossDisplayValues],
  );
  const gradDomain = useMemo(
    () => buildYDomain(applyOutlierCap(visibleGradDisplayValues, gradOutlierMode)),
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
    if (reducedEvalLossData.length < 2) return undefined;
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

  const minWindow = Math.min(10, Math.max(1, allSteps.length));
  const viewSettings: ViewSettingsState = {
    effectiveWindowSize,
    minWindow,
    allStepsLength: allSteps.length,
    setWindowSize: (value) => {
      const clampedWindow = clamp(Math.round(value), 1, Math.max(1, allSteps.length));
      if (clampedWindow >= allSteps.length) {
        setWindowSize(null);
        return;
      }
      setWindowSize(clampedWindow);
    },
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <TrainingLossChartCard
        data={displayLossData}
        domain={lossDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        avgRaw={avgRaw}
        avgDisplay={avgDisplay}
        smoothing={smoothing}
        setSmoothing={setSmoothing}
        showRaw={showRaw}
        setShowRaw={setShowRaw}
        showSmoothed={showSmoothed}
        setShowSmoothed={setShowSmoothed}
        showAvgLine={showAvgLine}
        setShowAvgLine={setShowAvgLine}
        viewSettings={viewSettings}
        scale={lossScale}
        setScale={setLossScale}
        outlierMode={lossOutlierMode}
        setOutlierMode={setLossOutlierMode}
      />
      <GradNormChartCard
        data={displayGradData}
        domain={gradDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        scale={gradScale}
        setScale={setGradScale}
        outlierMode={gradOutlierMode}
        setOutlierMode={setGradOutlierMode}
        viewSettings={viewSettings}
      />
      <LearningRateChartCard
        data={displayLrData}
        domain={lrDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        scale={lrScale}
        setScale={setLrScale}
        outlierMode={lrOutlierMode}
        setOutlierMode={setLrOutlierMode}
        viewSettings={viewSettings}
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
