// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import type { FlowNode } from "../types";
import {
  BENCHMARK_STATE_LABEL,
  type BenchmarkState,
  RECIPE_TASK,
  benchmarkView,
  percent,
} from "./benchmark-status";
import { type FieldRow, NodeFieldRows } from "./node-card";
import type { NodeLook } from "./node-looks";

const TONE: Record<BenchmarkState, string> = {
  needs_rows: "text-muted-foreground",
  ready: "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
  scoring: "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
  scored:
    "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  stopped: "text-muted-foreground",
  failed: "",
};

function keyFields(node: FlowNode): FieldRow[] {
  const p = node.params;
  const task = String(p.task ?? RECIPE_TASK);
  const rows: FieldRow[] = [{ label: "Task", value: task }];
  if (task === RECIPE_TASK)
    rows.push({ label: "Scorer", value: String(p.scorer ?? "Reference match") });
  if (p.max_rows) rows.push({ label: "Rows", value: `up to ${p.max_rows}` });
  return rows;
}

function BenchmarkNodeBody({ node }: { node: FlowNode }) {
  const runtime = useRuntimeStore((s) => s.nodes[node.id]) ?? IDLE_RUNTIME;
  const hasRows = useDocStore((s) =>
    Object.values(s.graph.edges).some(
      (e) => e.target === node.id && e.target_port === "dataset",
    ),
  );
  const view = benchmarkView({
    status: runtime.status,
    params: node.params,
    hasRows,
    outputs: runtime.outputs,
    phase: runtime.phase,
    error: runtime.error,
  });
  const result = view.result;
  return (
    <>
      <NodeFieldRows rows={keyFields(node)} />
      <div
        className="flow-detail flex flex-col gap-1 px-3"
        data-benchmark-state={view.state}
      >
        {result ? (
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-semibold tabular-nums leading-none text-foreground">
              {percent(result.score)}
            </span>
            <span className="text-ui-11 tabular-nums text-muted-foreground">
              {result.passed} / {result.scored} passed
            </span>
          </div>
        ) : (
          <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-0.5">
            <Badge
              variant={view.state === "failed" ? "destructive" : "outline"}
              className={cn("h-5 shrink-0 px-1.5 text-ui-10", TONE[view.state])}
            >
              {BENCHMARK_STATE_LABEL[view.state]}
            </Badge>
            {view.detail && view.state !== "failed" && (
              <span className="text-ui-11 leading-snug text-muted-foreground">
                {view.detail}
              </span>
            )}
          </div>
        )}
        {result && result.baseScore !== null && (
          <span className="text-ui-11 tabular-nums text-muted-foreground">
            Adapter off {percent(result.baseScore)}
          </span>
        )}
      </div>
    </>
  );
}

export const BENCHMARK_LOOK: NodeLook = { Body: BenchmarkNodeBody, ownsOutputs: true };
