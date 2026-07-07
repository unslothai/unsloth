// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useMemo } from "react";

import type { TrainingSeriesPoint } from "@/features/training";
// The loss + grad-norm cards are pure presentational (props only), so reuse them directly.
// We do NOT reuse ChartsSection/ChartsContent: those also render an LR and an Eval Loss
// card, which add little for diffusion LoRA training (the LR curve is the deterministic
// schedule the user just picked; eval is not configured). This is a diffusion-only
// two-card layout: Training Loss + Grad Norm (the actual training health signal).
// eslint-disable-next-line no-restricted-imports
import { GradNormChartCard } from "@/features/studio/sections/charts/grad-norm-chart-card";
// eslint-disable-next-line no-restricted-imports
import { TrainingLossChartCard } from "@/features/studio/sections/charts/training-loss-chart-card";
// eslint-disable-next-line no-restricted-imports
import {
  MAX_RENDER_POINTS,
  buildStepTicks,
  buildYDomain,
  compressSeries,
  ema,
} from "@/features/studio/sections/charts/utils";

// Fixed presentation defaults (the LLM tab exposes these via a settings sheet; here we pick
// sensible constants): EMA smoothing on, linear scale, raw + smoothed + average lines shown,
// no outlier trimming (diffusion loss is naturally noisy, not spiky-with-outliers).
const SMOOTHING = 0.8;

function toLossItems(series: TrainingSeriesPoint[]): { step: number; loss: number }[] {
  return series
    .filter((p) => Number.isFinite(p.value))
    .map((p) => ({ step: p.step, loss: p.value }));
}

// The x-domain that spans all points (the LLM tab supports a scrollable window; a training
// run here is short enough to always show the whole thing).
function fullStepDomain(steps: number[]): [number, number] {
  if (steps.length === 0) return [0, 1];
  const min = steps[0];
  const max = steps[steps.length - 1];
  if (min === max) return [min, min + 4];
  if (max - min < 6) return [Math.max(0, max - 6), max];
  return [min, max];
}

// A diffusion-only metrics view: Training Loss and Grad Norm, side by side, with a note
// under the loss card explaining why per-step loss looks noisy. Always renders both cards
// (even with no data) so the parent can decide when to mount them; we never early-return
// null here.
export function DiffusionCharts({
  lossHistory,
  gradNormHistory,
}: {
  lossHistory: TrainingSeriesPoint[];
  gradNormHistory: TrainingSeriesPoint[];
}): ReactElement {
  const lossItems = useMemo(() => toLossItems(lossHistory), [lossHistory]);
  const smoothed = useMemo(
    () => (lossItems.length > 0 ? ema(lossItems, SMOOTHING) : []),
    [lossItems],
  );
  const reducedLoss = useMemo(
    () => compressSeries(smoothed, MAX_RENDER_POINTS),
    [smoothed],
  );
  const lossData = useMemo(
    () =>
      reducedLoss.map((p) => ({
        ...p,
        displayLoss: p.loss,
        displaySmoothed: p.smoothed,
      })),
    [reducedLoss],
  );

  const gradData = useMemo(
    () =>
      compressSeries(
        gradNormHistory
          .filter((p) => Number.isFinite(p.value))
          .map((p) => ({ step: p.step, gradNorm: p.value, displayGradNorm: p.value })),
        MAX_RENDER_POINTS,
      ),
    [gradNormHistory],
  );

  const steps = useMemo(() => {
    const set = new Set<number>();
    for (const p of lossData) set.add(p.step);
    for (const p of gradData) set.add(p.step);
    return Array.from(set).sort((a, b) => a - b);
  }, [lossData, gradData]);

  const stepDomain = useMemo(() => fullStepDomain(steps), [steps]);
  const xAxisTicks = useMemo(
    () => buildStepTicks(stepDomain[0], stepDomain[1]),
    [stepDomain],
  );

  const lossDomain = useMemo(
    () => buildYDomain(lossData.flatMap((p) => [p.displayLoss, p.displaySmoothed])),
    [lossData],
  );
  const gradDomain = useMemo(
    () => buildYDomain(gradData.map((p) => p.displayGradNorm)),
    [gradData],
  );
  const avgRaw =
    lossItems.length > 0
      ? +(lossItems.reduce((s, p) => s + p.loss, 0) / lossItems.length).toFixed(4)
      : 0;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-1">
        <TrainingLossChartCard
          data={lossData}
          domain={lossDomain}
          visibleStepDomain={stepDomain}
          xAxisTicks={xAxisTicks}
          avgRaw={avgRaw}
          avgDisplay={avgRaw}
          showRaw={true}
          showSmoothed={true}
          showAvgLine={true}
          scale="linear"
        />
        <p className="px-1 text-[11px] leading-snug text-muted-foreground">
          Per-step loss is noisy by design: every step samples a random noise level. Watch
          the smoothed line for the trend, not the raw jitter.
        </p>
      </div>
      <GradNormChartCard
        data={gradData}
        domain={gradDomain}
        visibleStepDomain={stepDomain}
        xAxisTicks={xAxisTicks}
        scale="linear"
      />
    </div>
  );
}
