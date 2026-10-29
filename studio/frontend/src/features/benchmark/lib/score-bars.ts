// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Eval runs as rows for the shared Benchmarks bar chart. Type imports only, so it runs in node.

import type { BarGroup } from "../../benchmarks/components/bar-chart";
import type { BenchmarkRunSummary } from "../api/benchmark-api";

export const ACCURACY = "__accuracy__";

export function modelName(model: string): string {
  return model.split(/[\\/]/).filter(Boolean).pop() ?? model;
}

/** The model as a bar label: the quant ("repo:Q8_0") never truncates, so two quants stay told apart. */
export function barLabel(model: string, max = 40): string {
  const cut = model.lastIndexOf(":");
  const name = cut > 0 ? model.slice(0, cut) : model;
  const quant = cut > 0 ? model.slice(cut + 1) : "";
  const room = quant ? max - quant.length - 1 : max;
  const short =
    name.length > room ? `${name.slice(0, Math.max(1, room - 1))}…` : name;
  return quant ? `${short} ${quant}` : short;
}

function parseStderr(s: string | null | undefined): number | null {
  const err = s != null ? Number.parseFloat(s) : Number.NaN;
  return Number.isFinite(err) && err > 0 ? err : null;
}

/** The run's default metric as a 0-1 ratio, or null when it is not one. */
export function runScore(
  run: BenchmarkRunSummary,
): { score: number; stderr: number | null } | null {
  const m = run.metrics.find((x) => x.name === run.default_metric);
  if (!m || m.score < 0 || m.score > 1) return null;
  return { score: m.score, stderr: parseStderr(m.stderr) };
}

export interface ScoreBar {
  model: string;
  score: number;
  stderr: number | null;
  run: BenchmarkRunSummary;
}

/**
 * The latest run of each model on each task, best first. Models take colour slots by first
 * appearance and never cycle, so anything past `maxModels` is left to the history list.
 */
export function latestScores(
  runs: BenchmarkRunSummary[],
  maxModels: number,
): {
  groups: { task: string; bars: ScoreBar[] }[];
  models: string[];
  best: ScoreBar | undefined;
} {
  const newestFirst = [...runs].sort((a, b) =>
    b.created_at.localeCompare(a.created_at),
  );
  const models: string[] = [];
  for (const r of [...newestFirst].reverse()) {
    const name = modelName(r.model);
    if (!models.includes(name)) models.push(name);
  }
  const shown = models.slice(0, maxModels);
  const byTask = new Map<string, ScoreBar[]>();
  for (const run of newestFirst) {
    const s = runScore(run);
    const model = modelName(run.model);
    if (!s || !shown.includes(model)) continue;
    const bars = byTask.get(run.task) ?? [];
    if (bars.some((b) => b.model === model)) continue;
    bars.push({ model, ...s, run });
    byTask.set(run.task, bars);
  }
  const groups = [...byTask.entries()].map(([task, bars]) => ({
    task,
    bars: bars.sort((a, b) => b.score - a.score),
  }));
  const best = groups
    .flatMap((g) => g.bars)
    .sort((a, b) => b.score - a.score)[0];
  return { groups, models: shown, best };
}

export const fmtScore = (v: number): string => `${v.toFixed(1)}%`;

function runDetail(run: BenchmarkRunSummary): string {
  return `${run.n_samples} samples${run.num_fewshot ? ` · ${run.num_fewshot}-shot` : ""}`;
}

/** Scoreboard groups on a 0-100 axis, stderr as the whisker. */
export function scoreboardGroups(
  groups: { task: string; bars: ScoreBar[] }[],
  color: (model: string) => string,
): BarGroup[] {
  return groups.map(({ task, bars }) => ({
    key: task,
    label: task,
    rows: bars.map((b) => {
      const value = b.score * 100;
      const err = b.stderr === null ? null : b.stderr * 100;
      return {
        key: b.model,
        label: barLabel(b.model),
        value,
        low: err === null ? null : value - err,
        high: err === null ? null : value + err,
        color: color(b.model),
        tail: err === null ? undefined : `± ${err.toFixed(1)}`,
        tip: [
          b.run.model,
          `${fmtScore(value)}${err === null ? "" : ` ± ${err.toFixed(1)}`} ${b.run.default_metric}`,
          runDetail(b.run),
          new Date(b.run.created_at).toLocaleString(),
        ],
      };
    }),
  }));
}

/** One metric value per run, read the way the styled export reads it. */
export function metricValue(
  run: BenchmarkRunSummary,
  metric: string,
): { value: number; stderr: number | null; ratio: boolean } | null {
  const name = metric === ACCURACY ? run.default_metric : metric;
  const m =
    run.metrics.find((x) => x.name === name) ??
    (metric === ACCURACY
      ? undefined
      : run.metrics.find((x) =>
          x.name.toLowerCase().includes(metric.toLowerCase()),
        ));
  if (!m) return null;
  const ratio = m.score >= 0 && m.score <= 1;
  const err = parseStderr(m.stderr);
  return {
    value: ratio ? m.score * 100 : m.score,
    stderr: err === null ? null : ratio ? err * 100 : err,
    ratio,
  };
}

/**
 * The comparison chart: one group per task in the order the runs first appear, one bar per run.
 * `percent` is true when every value is a ratio, so the axis can sit on a fixed 0-100.
 */
export function comparisonGroups(
  runs: BenchmarkRunSummary[],
  metric: string,
  color: (model: string) => string,
  format: (v: number) => string,
): { groups: BarGroup[]; percent: boolean; peak: number } {
  const byTask = new Map<string, BarGroup>();
  let percent = true;
  let peak = 0;
  for (const run of runs) {
    const v = metricValue(run, metric);
    if (!v) continue;
    percent &&= v.ratio;
    peak = Math.max(peak, v.value + (v.stderr ?? 0));
    const group = byTask.get(run.task) ?? {
      key: run.task,
      label: run.task,
      rows: [],
    };
    const model = modelName(run.model);
    group.rows.push({
      key: run.id,
      label: barLabel(model),
      value: v.value,
      low: v.stderr === null ? null : v.value - v.stderr,
      high: v.stderr === null ? null : v.value + v.stderr,
      color: color(model),
      tail: v.stderr === null ? undefined : `± ${v.stderr.toFixed(1)}`,
      tip: [
        run.model,
        `${format(v.value)}${v.stderr === null ? "" : ` ± ${v.stderr.toFixed(1)}`}`,
        runDetail(run),
        new Date(run.created_at).toLocaleString(),
      ],
    });
    byTask.set(run.task, group);
  }
  return { groups: [...byTask.values()], percent, peak };
}

/** Models by their oldest run, the scoreboard's colour order, so a model keeps its colour. */
export function modelsOf(runs: BenchmarkRunSummary[]): string[] {
  const out: string[] = [];
  const oldestFirst = [...runs].sort((a, b) =>
    a.created_at.localeCompare(b.created_at),
  );
  for (const r of oldestFirst) {
    const m = modelName(r.model);
    if (!out.includes(m)) out.push(m);
  }
  return out;
}
