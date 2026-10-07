// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from the overlay so the parsing is testable.

import type { TrainingPhase } from "@/features/training";

export type PreparationProgress = {
  title: string;
  detail: string | null;
  percent: number | null;
};

const PREPARATION_PHASES = new Set<TrainingPhase>([
  "loading_model",
  "loading_dataset",
  "configuring",
]);

// The worker reports `training` once the trainer is built, with dataset mapping still ahead.
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

export function resolvePreparationMessage(
  message: string,
  fallback: string,
): string {
  return message.trim() || fallback;
}

// `_monitor_tqdm` in core/training/worker.py emits `"<desc> <percent>% (<n>/<total>)"`.
const COUNTED_PREPARATION_RE =
  /^(?<label>.+?)\s+(?<percent>\d{1,3})%\s+\((?<current>[\d,]+)\s*\/\s*(?<total>[\d,]+)\)$/;

// Audio loops report bare counts, e.g. `"Encoding audio... 100/1000"`.
const TALLIED_PREPARATION_RE =
  /^(?<label>.+?)\s+(?<current>[\d,]+)\s*\/\s*(?<total>[\d,]+)$/;

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

// A repo id must not match inside another id.
const ID_CHAR = /[a-z0-9._/-]/;

function mentionsResource(haystack: string, name?: string): boolean {
  if (!name) return false;
  let from = 0;
  for (;;) {
    const at = haystack.indexOf(name, from);
    if (at < 0) return false;
    const before = at > 0 ? haystack[at - 1] : "";
    const after = haystack[at + name.length] ?? "";
    if (!ID_CHAR.test(before) && !ID_CHAR.test(after)) return true;
    from = at + 1;
  }
}

// No bare `token`: `tokenizing` is dataset work, `tokenizer` is model loading. Audio codecs only
// preprocess the dataset.
const DATASET_PREPARATION_RE =
  /tokenizing|dataset|standardiz|\bmap\b|\bfilter\b|generating|resolving data|casting|formatting|\bsamples\b|local files|encoding audio|preprocessing|\brows\b|slic|snac|bicodec|outetts|whisper|codec|audio|eval split|chat template|\bconverting\b/i;

// Checked first: `Starting SNAC training...` names a codec only because it names the run.
const MODEL_PREPARATION_RE = /^(?:starting|initializing|queued)\b.*\btraining\b/i;

// Repo ids first, since the worker reports `Loading <repo_id>...`; dataset work names itself, so
// everything else is the model.
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
  const datasetHit = mentionsResource(haystack, datasetName);
  const modelHit = mentionsResource(haystack, modelName);
  // One owner/name can be both repo types; then fall through to the wording.
  const ambiguous = datasetHit && modelHit && datasetName === modelName;
  // The longer id wins, since `org/foo-base` also contains dataset `org/foo`.
  if (!ambiguous) {
    if (datasetHit && (!modelHit || datasetName!.length >= modelName!.length)) {
      return "dataset";
    }
    if (modelHit) return "model";
  }
  if (MODEL_PREPARATION_RE.test(title)) return "model";
  return DATASET_PREPARATION_RE.test(title) ? "dataset" : "model";
}

export function parsePreparationProgress(
  message: string,
  fallback: string,
): PreparationProgress {
  const resolved = resolvePreparationMessage(message, fallback);
  const groups =
    COUNTED_PREPARATION_RE.exec(resolved)?.groups ??
    TALLIED_PREPARATION_RE.exec(resolved)?.groups;
  if (!groups) return indeterminatePreparation(resolved);

  const current = Number(groups.current.replaceAll(",", ""));
  const total = Number(groups.total.replaceAll(",", ""));
  // Use the tqdm percent so it matches the log line.
  const percent =
    groups.percent === undefined
      ? Math.floor((current / total) * 100)
      : Number(groups.percent);
  if (percent > 100 || total <= 0 || current > total) {
    return indeterminatePreparation(groups.label);
  }

  return {
    title: cleanPreparationTitle(groups.label),
    detail: `${groups.current} / ${groups.total}`,
    percent,
  };
}
