// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import type { FlowNode } from "../types";
import { loopView } from "./loop-status";
import type { NodeLook } from "./node-looks";

const W = 232;
const H = 36;

function y(score: number): number {
  return H - 2 - (Math.max(0, Math.min(100, score)) / 100) * (H - 4);
}

/** Test score per round on a fixed 0-100 scale, with the target as a dashed line. */
function ScoreSparkline({
  scores,
  rounds,
  target,
}: { scores: number[]; rounds: number; target: number }) {
  const dx = W / Math.max(1, rounds - 1);
  const points = scores.map((s, i) => `${(i * dx).toFixed(1)},${y(s).toFixed(1)}`);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-9 w-full overflow-visible"
      preserveAspectRatio="none"
      aria-hidden="true"
      data-loop-sparkline={scores.join(",")}
    >
      <line
        x1={0}
        x2={W}
        y1={y(target)}
        y2={y(target)}
        strokeDasharray="4 3"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
        className="stroke-emerald-500/50"
      />
      {points.length > 1 && (
        <polyline
          points={points.join(" ")}
          fill="none"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
          className="stroke-sky-400"
        />
      )}
      {scores.map((s, i) => (
        <circle
          // biome-ignore lint/suspicious/noArrayIndexKey: one dot per round, in order
          key={i}
          cx={i * dx}
          cy={y(s)}
          r={2.5}
          className="fill-sky-400"
        />
      ))}
    </svg>
  );
}

function LoopNodeBody({ node }: { node: FlowNode }) {
  const runtime = useRuntimeStore((s) => s.nodes[node.id]) ?? IDLE_RUNTIME;
  const view = loopView({
    status: runtime.status,
    params: node.params,
    outputs: runtime.outputs,
    partial: runtime.partial,
    partialStep: runtime.partialStep,
  });
  const total = Math.max(1, Number(node.params.rounds) || 3);
  return (
    <div className="flow-detail flex flex-col gap-1 px-3" data-loop-done={view.done}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-0.5">
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-ui-10 font-medium ${
            view.done
              ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300"
              : "bg-sky-500/15 text-sky-600 dark:text-sky-300"
          }`}
        >
          {view.headline}
        </span>
        <span className="text-ui-11 leading-snug text-muted-foreground">
          {view.detail}
        </span>
      </div>
      <ScoreSparkline scores={view.scores} rounds={total} target={view.target} />
      {view.rounds.length > 0 && (
        <div className="flex flex-wrap gap-x-2 text-ui-11 tabular-nums text-muted-foreground">
          {view.rounds.map((r) => (
            <span key={r.round} title={`probe ${r.probe_score ?? "-"}%, ${r.rows_from_failures} rows from failures`}>
              R{r.round} {r.test_score ?? "-"}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export const LOOP_LOOK: NodeLook = { Body: LoopNodeBody, ownsOutputs: true };
