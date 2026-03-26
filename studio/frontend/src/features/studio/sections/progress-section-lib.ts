// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingPhase } from "@/features/training";

export const phaseLabel: Record<TrainingPhase, string> = {
  idle: "Idle",
  downloading_model: "Downloading model",
  downloading_dataset: "Downloading dataset",
  loading_model: "Loading model",
  loading_dataset: "Loading dataset",
  configuring: "Configuring",
  training: "Training",
  completed: "Completed",
  error: "Error",
  stopped: "Stopped",
};

export const phaseColors: Record<TrainingPhase, string> = {
  idle: "bg-muted text-muted-foreground",
  downloading_model:
    "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300",
  downloading_dataset:
    "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300",
  loading_model:
    "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
  loading_dataset:
    "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
  configuring: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
  training:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  completed:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  stopped: "bg-muted text-muted-foreground",
};

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "--";
  const total = Math.floor(seconds);
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatNumber(value: number | null | undefined, digits: number): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toFixed(digits);
}

