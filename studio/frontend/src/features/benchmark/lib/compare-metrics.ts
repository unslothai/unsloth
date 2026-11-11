// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const ACCURACY_LIKE_RE =
  /(em|exact|strict|flexible|acc|f1|match|score)/i;

// Perplexity-style metrics (lm-eval word/byte_perplexity, bits_per_byte, TER) improve downward.
const LOWER_IS_BETTER_RE =
  /(perplexity|bits_per_byte|loss|(^|[^a-z])ter([^a-z]|$))/i;

export function isRatioScore(m: { name: string; score: number }): boolean {
  return ACCURACY_LIKE_RE.test(m.name) && m.score >= 0 && m.score <= 1;
}

export function isLowerBetter(name: string): boolean {
  return LOWER_IS_BETTER_RE.test(name);
}

/** Number only; ratio metrics scaled to percent (caller adds the "%"). */
export function formatMetricValue(m: { name: string; score: number }): string {
  if (isRatioScore(m)) {
    return (m.score * 100).toFixed(1);
  }
  if (Number.isInteger(m.score)) {
    return String(m.score);
  }
  return m.score.toFixed(m.score < 1 ? 4 : 1);
}

export function formatMetricWithUnit(m: {
  name: string;
  score: number;
}): string {
  return `${formatMetricValue(m)}${isRatioScore(m) ? "%" : ""}`;
}

/** Best score across runs for one metric, honouring lower-is-better metrics. */
export function bestMetricScore(
  name: string,
  scores: (number | null)[],
): number | null {
  const present = scores.filter(
    (s): s is number => s != null && Number.isFinite(s),
  );
  if (present.length === 0) return null;
  return isLowerBetter(name) ? Math.min(...present) : Math.max(...present);
}

export function isDisplayMetric(m: { name: string }): boolean {
  const lower = m.name.toLowerCase().trim();
  const skip = new Set([
    "sample_len",
    "num_samples",
    "n_samples",
    "effective_samples",
    "alias",
    "name",
    "bootstrap_iters",
  ]);
  if (skip.has(lower) || lower.startsWith("sample_len")) {
    return false;
  }
  return true;
}

export function pickPrimaryMetric<M extends { name: string }>(
  metrics: M[],
  headline?: string | null,
): M | undefined {
  return (
    (headline ? metrics.find((m) => m.name === headline) : undefined) ??
    metrics.find((m) => isDisplayMetric(m) && ACCURACY_LIKE_RE.test(m.name)) ??
    metrics.find(isDisplayMetric)
  );
}
