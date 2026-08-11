// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads the worker's status message for the window between "downloads finished" and
// "first training step", where tokenizing and mapping a large dataset can run for
// minutes with nothing on screen moving. Kept apart from the overlay so the parsing
// can be exercised without mounting the component.

import type { TrainingPhase } from "@/features/training";

export type PreparationProgress = {
  title: string;
  /** Row counts behind a determinate bar, e.g. `32,000 / 207,865`. */
  detail: string | null;
  /** The worker's own whole-number percent, or null for a bar of unknown length. */
  percent: number | null;
};

const PREPARATION_PHASES = new Set<TrainingPhase>([
  "loading_model",
  "loading_dataset",
  "configuring",
]);

/**
 * Whether the run is setting up rather than stepping.
 *
 * `training` counts while the step is still 0: the worker reports that phase as soon
 * as the trainer is built, and dataset mapping runs inside it.
 */
export function shouldShowPreparationStatus(
  phase: TrainingPhase,
  currentStep: number,
  isStarting: boolean,
): boolean {
  if (isStarting) return true;
  return (
    PREPARATION_PHASES.has(phase) || (phase === "training" && currentStep <= 0)
  );
}

/** The worker's message, or the caller's fallback before it has sent one. */
export function resolvePreparationMessage(
  message: string,
  fallback: string,
): string {
  return message.trim() || fallback;
}

// The worker's tqdm monitor emits `"<desc> <percent>% (<n>/<total>)"` with grouped
// thousands; see `_monitor_tqdm` in studio/backend/core/training/worker.py.
const COUNTED_PREPARATION_RE =
  /^(?<label>.+?)\s+(?<percent>\d{1,3})%\s+\((?<current>[\d,]+)\s*\/\s*(?<total>[\d,]+)\)$/;

function cleanPreparationTitle(label: string): string {
  return label
    .replace(/^Unsloth:\s*/i, "")
    .replace(/\s*\(num_proc\s*=\s*\d+\)$/i, "")
    .replace(/(?:\.\.\.|…)$/, "")
    .trim();
}

function indeterminatePreparation(label: string): PreparationProgress {
  return { title: cleanPreparationTitle(label), detail: null, percent: null };
}

export type PreparationTarget = "model" | "dataset";

// Tried before the model patterns: `tokenizing` is dataset work while `tokenizer` is part of
// loading the model, so neither side keys off a bare `token` stem.
const DATASET_PREPARATION_RE =
  /tokenizing|dataset|standardiz|\bmap\b|\bfilter\b|generating|resolving data|casting|formatting|\bsamples\b|local files|encoding audio|\brows\b|slic/i;

/**
 * Which resource row a preparation step belongs to.
 *
 * The repo ids come first: the worker reports `Loading <repo_id>...`, which carries no word a
 * pattern could key off. Dataset work always names itself, so everything else -- importing,
 * configuring, adapters, trainer setup -- belongs to the model.
 */
export function classifyPreparation(
  title: string,
  resources: {
    modelName?: string | null;
    datasetName?: string | null;
  } = {},
): PreparationTarget {
  const haystack = title.toLowerCase();
  const datasetName = resources.datasetName?.toLowerCase();
  const modelName = resources.modelName?.toLowerCase();
  if (datasetName && haystack.includes(datasetName)) return "dataset";
  if (modelName && haystack.includes(modelName)) return "model";
  return DATASET_PREPARATION_RE.test(title) ? "dataset" : "model";
}

export function parsePreparationProgress(
  message: string,
  fallback: string,
): PreparationProgress {
  const resolved = resolvePreparationMessage(message, fallback);
  const groups = COUNTED_PREPARATION_RE.exec(resolved)?.groups;
  if (!groups) return indeterminatePreparation(resolved);

  // The worker already reports the percent; recomputing it from the counts only lets the
  // bar and the log line above it disagree over rounding.
  const percent = Number(groups.percent);
  const current = Number(groups.current.replaceAll(",", ""));
  const total = Number(groups.total.replaceAll(",", ""));
  if (percent > 100 || total <= 0 || current > total) {
    return indeterminatePreparation(groups.label);
  }

  return {
    title: cleanPreparationTitle(groups.label),
    detail: `${groups.current} / ${groups.total}`,
    percent,
  };
}
