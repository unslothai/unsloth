// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// reads the worker's status message for the window between the last download and the first
// training step, kept apart from the overlay so the parsing can be exercised on its own.

import type { TrainingPhase } from "@/features/training";

export type PreparationProgress = {
  title: string;
  // row counts behind a determinate bar, e.g. `32,000 / 207,865`.
  detail: string | null;
  // null for a bar of unknown length.
  percent: number | null;
};

const PREPARATION_PHASES = new Set<TrainingPhase>([
  "loading_model",
  "loading_dataset",
  "configuring",
]);

// `training` counts while the step is still 0: the worker reports that phase as soon as the
// trainer is built, with dataset mapping still ahead of it.
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

// the worker's message, or the caller's fallback before it has sent one.
export function resolvePreparationMessage(
  message: string,
  fallback: string,
): string {
  return message.trim() || fallback;
}

// `_monitor_tqdm` in core/training/worker.py emits `"<desc> <percent>% (<n>/<total>)"`.
const COUNTED_PREPARATION_RE =
  /^(?<label>.+?)\s+(?<percent>\d{1,3})%\s+\((?<current>[\d,]+)\s*\/\s*(?<total>[\d,]+)\)$/;

// the audio loops report bare counts instead, e.g. `"Encoding audio... 100/1000"`.
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

// a repo id must not match inside another: a bare `includes` routed a model status to the
// dataset row whenever one id was a prefix of the other.
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

// no bare `token` stem: `tokenizing` is dataset work, `tokenizer` is part of loading the model.
// the audio codecs are here because they are loaded only to preprocess the dataset.
const DATASET_PREPARATION_RE =
  /tokenizing|dataset|standardiz|\bmap\b|\bfilter\b|generating|resolving data|casting|formatting|\bsamples\b|local files|encoding audio|preprocessing|\brows\b|slic|snac|bicodec|outetts|whisper|codec|audio|eval split|chat template|\bconverting\b/i;

// checked before the dataset patterns: `Starting SNAC training...` names a codec only because
// it names the run, and matching `snac` sent the trainer's own start line to the dataset row.
const MODEL_PREPARATION_RE = /^(?:starting|initializing|queued)\b.*\btraining\b/i;

// repo ids come first because the worker reports `Loading <repo_id>...`, which matches no
// pattern; dataset work always names itself, so everything else belongs to the model.
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
  // the Hub allows one owner/name as both repo types, and then the id decides nothing; fall
  // through to the wording rather than handing every such message to the dataset.
  const ambiguous = datasetHit && modelHit && datasetName === modelName;
  // the longer id wins when both match, because one can contain the other:
  // `Loading org/foo-base` mentions the model AND the dataset `org/foo`.
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
  // the tqdm shape reports its own percent; recomputing it would disagree with the log line above.
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
