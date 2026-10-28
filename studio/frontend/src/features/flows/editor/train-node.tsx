// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { setTrainingCompareHandoff } from "@/features/chat/lib/training-compare-handoff";
import { cn } from "@/lib/utils";
import { useNavigate } from "@tanstack/react-router";
import { useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import type { FlowNode } from "../types";
import { type FieldRow, NodeFieldRows } from "./node-card";
import type { NodeLook } from "./node-looks";
import {
  TRAIN_STATE_LABEL,
  type TrainState,
  sparklinePath,
  trainView,
} from "./train-status";

const TONE: Record<TrainState, string> = {
  needs_dataset: "text-muted-foreground",
  setup:
    "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  ready: "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
  training: "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
  adapter_ready:
    "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  stopped: "text-muted-foreground",
  failed: "",
};

export function Sparkline({
  values,
  className = "stroke-sky-600 dark:stroke-sky-300",
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

function keyFields(node: FlowNode): FieldRow[] {
  const p = node.params;
  const model = String(p.base_model ?? "").trim();
  const rows: FieldRow[] = [];
  if (model) rows.push({ label: "Model", value: model.split("/").pop() ?? model });
  if (p.max_steps) rows.push({ label: "Steps", value: String(p.max_steps) });
  const held = Number(p.holdout_percent) || 0;
  if (held > 0) rows.push({ label: "Hold out", value: `${held}%` });
  return rows;
}

function TrainNodeBody({ node }: { node: FlowNode }) {
  const runtime = useRuntimeStore((s) => s.nodes[node.id]) ?? IDLE_RUNTIME;
  const hasDataset = useDocStore((s) =>
    Object.values(s.graph.edges).some(
      (e) => e.target === node.id && e.target_port === "dataset",
    ),
  );
  const navigate = useNavigate();
  const view = trainView({
    status: runtime.status,
    params: node.params,
    hasDataset,
    outputs: runtime.outputs,
    partial: runtime.partial,
    error: runtime.error,
  });
  const checkpoint = runtime.outputs?.checkpoint as
    | { base_model?: string; output_dir?: string }
    | undefined;
  return (
    <>
      <NodeFieldRows rows={keyFields(node)} />
      <div
        className="flow-detail flex flex-col gap-1 px-3"
        data-train-state={view.state}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-0.5">
          <Badge
            variant={view.state === "failed" ? "destructive" : "outline"}
            className={cn("h-5 shrink-0 px-1.5 text-ui-10", TONE[view.state])}
          >
            {TRAIN_STATE_LABEL[view.state]}
          </Badge>
          {view.detail && view.state !== "failed" && (
            <span className="text-ui-11 leading-snug text-muted-foreground">
              {view.detail}
            </span>
          )}
        </div>
        <Sparkline values={view.history} />
        {view.state === "adapter_ready" && checkpoint && (
          <Button
            type="button"
            size="xs"
            variant="outline"
            className="nodrag self-start"
            onClick={(event) => {
              event.stopPropagation();
              setTrainingCompareHandoff(
                checkpoint.base_model ?? null,
                checkpoint.output_dir ?? null,
              );
              void navigate({ to: "/chat" });
            }}
          >
            Chat with it
          </Button>
        )}
      </div>
    </>
  );
}

export const TRAIN_LOOK: NodeLook = { Body: TrainNodeBody, ownsOutputs: true };
