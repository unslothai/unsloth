// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeStatus } from "../types";

export type TrainState =
  | "needs_dataset"
  | "setup"
  | "ready"
  | "training"
  | "adapter_ready"
  | "stopped"
  | "failed";

export const TRAIN_STATE_LABEL: Record<TrainState, string> = {
  needs_dataset: "Needs a dataset",
  setup: "Finish setup",
  ready: "Ready to train",
  training: "Training",
  adapter_ready: "Adapter ready",
  stopped: "Stopped",
  failed: "Failed",
};

export interface TrainView {
  state: TrainState;
  detail: string | null;
  history: number[];
  /** True when `history` is the GRPO reward rather than the loss. */
  reward: boolean;
}

interface Metrics {
  loss?: unknown;
  steps?: unknown;
  total?: unknown;
  loss_history?: unknown;
  train_rows?: unknown;
  holdout_rows?: unknown;
  reward?: unknown;
  reward_start?: unknown;
  reward_max?: unknown;
  reward_history?: unknown;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

function lossText(m: Metrics): string | null {
  const loss = num(m.loss);
  return loss === null ? null : `loss ${loss.toFixed(3)}`;
}

/** "1.80 / 2.5": the reward against the best a completion can score, never a bare number. */
export function rewardText(reward: number, max: number | null): string {
  const best = max === null ? "" : ` / ${Number(max.toFixed(2))}`;
  return `reward ${reward.toFixed(2)}${best}`;
}

function numbers(values: unknown): number[] {
  return Array.isArray(values)
    ? values.filter((v): v is number => num(v) !== null)
    : [];
}

function historyOf(m: Metrics): number[] {
  const rewards = numbers(m.reward_history);
  return rewards.length ? rewards : numbers(m.loss_history);
}

function progressText(m: Metrics): string | null {
  const reward = num(m.reward);
  return reward === null ? lossText(m) : rewardText(reward, num(m.reward_max));
}

export function trainView(input: {
  status: NodeStatus;
  params: Record<string, unknown>;
  hasDataset: boolean;
  outputs: Record<string, unknown> | null;
  partial: Record<string, unknown> | null;
  error: string | null;
}): TrainView {
  const { status, params } = input;
  const live = (input.partial?.metrics ?? {}) as Metrics;
  const done = (input.outputs?.metrics ?? {}) as Metrics;
  if (["queued", "waiting", "loading_model", "running"].includes(status)) {
    const step = num(live.steps);
    const total = num(live.total);
    const parts = [
      step !== null && total ? `Step ${step} of ${total}` : "Starting",
      progressText(live),
    ].filter(Boolean);
    return withKind("training", parts.join(" · "), live);
  }
  if (status === "failed") return withKind("failed", input.error, live);
  if (status === "cancelled")
    return withKind("stopped", progressText(live), live);
  if ((status === "done" || status === "cached") && input.outputs?.checkpoint) {
    const train = num(done.train_rows);
    const held = num(done.holdout_rows);
    const start = num(done.reward_start);
    const parts = [
      progressText(done),
      start !== null && num(done.reward) !== null ? `from ${start.toFixed(2)}` : null,
      train !== null && held !== null ? `${train} trained / ${held} held out` : null,
    ].filter(Boolean);
    return withKind("adapter_ready", parts.join(" · ") || null, done);
  }
  if (!input.hasDataset)
    return { state: "needs_dataset", detail: "Connect a dataset", history: [], reward: false };
  if (!String(params.base_model ?? "").trim())
    return { state: "setup", detail: "Name a base model", history: [], reward: false };
  const objective = String(params.objective ?? "sft");
  const prompt = String(params.prompt_column ?? "").trim();
  const response = String(params.response_column ?? "").trim();
  if (objective === "sft" && !!prompt !== !!response)
    return {
      state: "setup",
      detail: "Set both columns, or leave both blank",
      history: [],
      reward: false,
    };
  const rewards = Array.isArray(params.rewards)
    ? params.rewards.filter((r) => String(r?.name ?? "").trim())
    : [];
  if (objective === "grpo" && !rewards.length)
    return { state: "setup", detail: "Pick a reward", history: [], reward: false };
  const pct = num(params.holdout_percent) ?? 0;
  return {
    state: "ready",
    detail: pct > 0 ? `${pct}% held out for scoring` : null,
    history: [],
    reward: false,
  };
}

function withKind(state: TrainState, detail: string | null, m: Metrics): TrainView {
  return {
    state,
    detail,
    history: historyOf(m),
    reward: numbers(m.reward_history).length > 0,
  };
}

/** Best and worst a reward rule (GET /api/rewards `rule`) can score. */
export function ruleRange(rule: Record<string, unknown> | null | undefined): [number, number] | null {
  if (!rule) return null;
  const scores = [
    ...Object.values((rule.score ?? {}) as Record<string, unknown>),
    ...(Array.isArray(rule.bands) ? rule.bands.map((b) => (b as { score?: unknown })?.score) : []),
    rule.missing,
    rule.else,
  ].filter((v): v is number => num(v) !== null);
  return scores.length ? [Math.min(...scores), Math.max(...scores)] : null;
}

/** Best weighted total for these rewards; null when a rule is unknown. */
export function rewardMax(
  picked: { name: string; weight: number }[],
  rules: Record<string, Record<string, unknown> | null | undefined>,
): number | null {
  let total = 0;
  for (const { name, weight } of picked) {
    const range = ruleRange(rules[name]);
    if (!range) return null;
    total += weight * (weight >= 0 ? range[1] : range[0]);
  }
  return total;
}

/** SVG path for a sparkline in a width x height box; flat when every value is the same. */
export function sparklinePath(
  values: number[],
  width: number,
  height: number,
): string {
  if (values.length < 2) return "";
  const lo = Math.min(...values);
  const span = Math.max(...values) - lo || 1;
  const dx = width / (values.length - 1);
  return values
    .map((v, i) => {
      const x = (i * dx).toFixed(1);
      const y = (height - 1 - ((v - lo) / span) * (height - 2)).toFixed(1);
      return `${i ? "L" : "M"}${x},${y}`;
    })
    .join("");
}
