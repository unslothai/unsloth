// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const ACCURACY_LIKE_RE = /(em|exact|strict|flexible|acc|f1|match|score)/i;

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

/** The run's headline metric (its `default_metric`, as history and the Flows node show it), else
 * the first accuracy-like one. */
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
