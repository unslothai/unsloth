// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeStatus } from "../types";

export interface LoopRound {
  round: number;
  test_score: number | null;
  probe_score: number | null;
  rows_from_failures: number;
  probe_failures: number;
}

export interface LoopView {
  /** "Round 2 of 3", "Reached 92% in round 2", ... */
  headline: string;
  detail: string;
  rounds: LoopRound[];
  /** Test score per finished round, for the sparkline. */
  scores: number[];
  target: number;
  done: boolean;
}

interface LoopInput {
  status: NodeStatus;
  params: Record<string, unknown>;
  outputs: Record<string, unknown> | null;
  partial: Record<string, unknown> | null;
  partialStep: { step: number; total: number } | null;
}

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function roundsOf(value: unknown): LoopRound[] {
  const list = Array.isArray(value)
    ? value
    : value && typeof value === "object"
      ? (value as { rounds?: unknown }).rounds
      : null;
  return Array.isArray(list)
    ? list.filter(
        (r): r is LoopRound =>
          !!r && typeof r === "object" && typeof r.round === "number",
      )
    : [];
}

export function loopView(input: LoopInput): LoopView {
  const total = Math.max(1, num(input.params.rounds, 3));
  const target = num(input.params.target, 90);
  const finished = input.outputs?.history;
  const rounds = roundsOf(finished ?? input.partial?.history);
  const scores = rounds
    .map((r) => r.test_score)
    .filter((s): s is number => typeof s === "number");
  const best = rounds.reduce<LoopRound | null>(
    (top, r) =>
      typeof r.test_score === "number" &&
      (!top || r.test_score > (top.test_score ?? -1))
        ? r
        : top,
    null,
  );
  const feedback = input.params.feedback !== false;
  const detail = `${total} rounds · target ${target}% · ${feedback ? "failures feed back" : "feedback off"}`;
  const stopped =
    finished && typeof finished === "object"
      ? (finished as { stopped?: string }).stopped
      : undefined;
  if (input.status === "done" && best) {
    return {
      headline:
        stopped === "target"
          ? `Reached ${best.test_score}% in round ${best.round}`
          : `Best ${best.test_score}% in round ${best.round}`,
      detail,
      rounds,
      scores,
      target,
      done: true,
    };
  }
  if (input.status === "running" || input.status === "queued") {
    const current = Math.min(total, rounds.length + 1);
    return {
      headline: `Round ${current} of ${total}`,
      detail,
      rounds,
      scores,
      target,
      done: false,
    };
  }
  return {
    headline: input.status === "failed" ? "Stopped on an error" : "Ready",
    detail,
    rounds,
    scores,
    target,
    done: false,
  };
}
