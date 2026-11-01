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
}

interface Metrics {
  loss?: unknown;
  steps?: unknown;
  total?: unknown;
  loss_history?: unknown;
  train_rows?: unknown;
  holdout_rows?: unknown;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

function lossText(m: Metrics): string | null {
  const loss = num(m.loss);
  return loss === null ? null : `loss ${loss.toFixed(3)}`;
}

function historyOf(m: Metrics): number[] {
  return Array.isArray(m.loss_history)
    ? m.loss_history.filter((v): v is number => num(v) !== null)
    : [];
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
      lossText(live),
    ].filter(Boolean);
    return { state: "training", detail: parts.join(" · "), history: historyOf(live) };
  }
  if (status === "failed")
    return { state: "failed", detail: input.error, history: historyOf(live) };
  if (status === "cancelled")
    return { state: "stopped", detail: lossText(live), history: historyOf(live) };
  if ((status === "done" || status === "cached") && input.outputs?.checkpoint) {
    const train = num(done.train_rows);
    const held = num(done.holdout_rows);
    const parts = [
      lossText(done),
      train !== null && held !== null ? `${train} trained / ${held} held out` : null,
    ].filter(Boolean);
    return {
      state: "adapter_ready",
      detail: parts.join(" · ") || null,
      history: historyOf(done),
    };
  }
  if (!input.hasDataset)
    return { state: "needs_dataset", detail: "Connect a dataset", history: [] };
  if (!String(params.base_model ?? "").trim())
    return { state: "setup", detail: "Name a base model", history: [] };
  const prompt = String(params.prompt_column ?? "").trim();
  const response = String(params.response_column ?? "").trim();
  const mapping = params.custom_format_mapping;
  const mapped = !!mapping && typeof mapping === "object" && Object.keys(mapping).length > 0;
  if (!mapped && !!prompt !== !!response)
    return {
      state: "setup",
      detail: "Set both columns, or leave both blank",
      history: [],
    };
  if (!mapped && prompt && prompt === response)
    return { state: "setup", detail: "Use two different columns", history: [] };
  const pct = num(params.holdout_percent) ?? 0;
  return {
    state: "ready",
    detail: pct > 0 ? `${pct}% held out for scoring` : null,
    history: [],
  };
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

/** The adapter path Chat lists for a flow's run folder; the node only knows the folder's name. */
export function adapterPathFor(
  loras: { adapter_path?: unknown }[],
  run: string,
): string | null {
  const found = loras.find(
    (l) =>
      typeof l.adapter_path === "string" &&
      l.adapter_path.split(/[/\\]+/).filter(Boolean).at(-1) === run,
  );
  return found ? (found.adapter_path as string) : null;
}
