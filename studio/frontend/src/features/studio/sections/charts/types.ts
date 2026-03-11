// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export type ScaleMode = "linear" | "log";
export type OutlierMode = "none" | "p99" | "p95";

export type LossHistoryItem = { step: number; loss: number };
export type SmoothedLossItem = LossHistoryItem & { smoothed: number };

export interface TrainingChartSeries {
  lossHistory: LossHistoryItem[];
  lrHistory: { step: number; lr: number }[];
  gradNormHistory: { step: number; gradNorm: number }[];
  evalLossHistory: { step: number; loss: number }[];
}
