// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDocStore } from "../stores/doc-store";
import type { NodeRuntime } from "../stores/runtime-store";
import type { FlowNode } from "../types";
import {
  TRAIN_STATE_LABEL,
  type TrainState,
  sparklinePath,
  trainView,
} from "./train-status";

const TONE: Record<TrainState, string> = {
  needs_dataset: "bg-zinc-500/15 text-zinc-300",
  setup: "bg-amber-500/15 text-amber-300",
  ready: "bg-sky-500/15 text-sky-300",
  training: "bg-sky-500/20 text-sky-200",
  adapter_ready: "bg-emerald-500/15 text-emerald-300",
  stopped: "bg-zinc-500/15 text-zinc-300",
  failed: "bg-red-500/15 text-red-300",
};

export function Sparkline({
  values,
  className = "stroke-sky-300",
}: { values: number[]; className?: string }) {
  const path = sparklinePath(values, 232, 32);
  if (!path) return null;
  return (
    <svg
      viewBox="0 0 232 32"
      className="h-8 w-full"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path
        d={path}
        fill="none"
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
        className={className}
      />
    </svg>
  );
}

export function TrainNodeBody({
  node,
  runtime,
}: { node: FlowNode; runtime: NodeRuntime }) {
  const hasDataset = useDocStore((s) =>
    Object.values(s.graph.edges).some(
      (e) => e.target === node.id && e.target_port === "dataset",
    ),
  );
  const view = trainView({
    status: runtime.status,
    params: node.params,
    hasDataset,
    outputs: runtime.outputs,
    partial: runtime.partial,
    error: runtime.error,
  });
  return (
    <div className="flow-detail flex flex-col gap-1 px-3" data-train-state={view.state}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${TONE[view.state]}`}
        >
          {TRAIN_STATE_LABEL[view.state]}
        </span>
        {view.detail && view.state !== "failed" && (
          <span className="truncate text-[11px] text-zinc-400">
            {view.detail}
          </span>
        )}
      </div>
      <Sparkline values={view.history} />
    </div>
  );
}
