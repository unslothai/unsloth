// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  Layers01Icon,
  UngroupLayersIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type Node,
  type NodeProps,
  Position,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { type ReactElement, memo, useEffect } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { NODE_HANDLE_CLASS } from "../utils/handle-layout";
import type {
  SuperCardLastRun,
  SuperCardNodeData,
  SuperCardPort,
} from "../utils/graph/super-cards";
import { BaseHandle } from "./rf-ui/base-handle";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "./rf-ui/base-node";

type SuperCardNode = Node<SuperCardNodeData, "super">;

const RUN_STATUS_LABEL: Record<SuperCardLastRun["status"], string> = {
  pending: "queued",
  running: "running",
  active: "running",
  cancelling: "cancelling",
  cancelled: "cancelled",
  completed: "completed",
  error: "failed",
};

function lastRunLabel(lastRun: SuperCardLastRun | null): string {
  if (!lastRun) {
    return "Not run yet";
  }
  const kind = lastRun.kind === "preview" ? "Preview" : "Full run";
  const columns =
    lastRun.totalColumns > 0
      ? ` · ${lastRun.doneColumns}/${lastRun.totalColumns} columns`
      : "";
  return `${kind} ${RUN_STATUS_LABEL[lastRun.status]}${columns}`;
}

// Inputs on the left edge and outputs on the right in every layout, one
// labelled row per port.
function PortColumn({
  title,
  ports,
  type,
}: {
  title: string;
  ports: SuperCardPort[];
  type: "source" | "target";
}): ReactElement {
  const input = type === "target";
  return (
    <div className={cn("min-w-0 space-y-1", !input && "text-right")}>
      <p
        className={cn("text-xs text-muted-foreground", input ? "pl-4" : "pr-4")}
      >
        {title}
      </p>
      {ports.length === 0 ? (
        <p
          className={cn(
            "text-xs text-muted-foreground/70",
            input ? "pl-4" : "pr-4",
          )}
        >
          None
        </p>
      ) : (
        ports.map((port) => (
          <div
            key={port.handleId}
            className={cn("relative py-0.5", input ? "pl-4" : "pr-4")}
          >
            <span
              className="block truncate font-mono text-xs text-foreground"
              title={port.label}
            >
              {port.label}
            </span>
            <BaseHandle
              id={port.handleId}
              type={type}
              position={input ? Position.Left : Position.Right}
              isConnectable={false}
              title={port.label}
              className={NODE_HANDLE_CLASS}
            />
          </div>
        ))
      )}
    </div>
  );
}

const MAX_CHIPS = 4;

function NameChips({
  label,
  names,
}: {
  label: string;
  names: string[];
}): ReactElement {
  const shown = names.slice(0, MAX_CHIPS);
  const hidden = names.length - shown.length;
  return (
    <div className="flex min-w-0 items-start gap-2">
      <p className="w-16 shrink-0 pt-0.5 text-xs text-muted-foreground">
        {label}
      </p>
      {names.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing</p>
      ) : (
        <div
          className="flex min-w-0 flex-wrap gap-1"
          title={hidden > 0 ? names.join(", ") : undefined}
        >
          {shown.map((name) => (
            <Badge
              key={name}
              variant="outline"
              className="corner-squircle font-mono"
            >
              {name}
            </Badge>
          ))}
          {hidden > 0 && (
            <Badge variant="ghost" className="text-muted-foreground">
              +{hidden} more
            </Badge>
          )}
        </div>
      )}
    </div>
  );
}

function RecipeGraphSuperCardNodeBase({
  id,
  data,
}: NodeProps<SuperCardNode>): ReactElement {
  const openSuperCard = useRecipeStudioStore((state) => state.openSuperCard);
  const ungroupSuperCard = useRecipeStudioStore(
    (state) => state.ungroupSuperCard,
  );
  const updateNodeInternals = useUpdateNodeInternals();
  const inputKey = data.inputs.map((port) => port.handleId).join("|");
  const outputKey = data.outputs.map((port) => port.handleId).join("|");

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, inputKey, outputKey, updateNodeInternals]);

  const lastRunTone =
    data.lastRun?.status === "error"
      ? "text-destructive"
      : data.lastRun?.status === "completed"
        ? "text-emerald-600 dark:text-emerald-400"
        : "text-muted-foreground";

  return (
    <BaseNode
      className={cn(
        "corner-squircle relative w-full min-w-0 overflow-visible rounded-xl border-2 border-primary/30",
        "[box-shadow:5px_5px_0_-1px_var(--card),5px_5px_0_0_var(--border)]",
        data.runtimeState === "running" && "border-ring-strong",
        data.runtimeState === "done" &&
          "border-emerald-500/60 ring-1 ring-emerald-500/20",
      )}
      title="Double-click to open"
    >
      {data.runtimeState === "running" && (
        <div className="pointer-events-none absolute -top-7 right-2 z-20 rounded-full bg-background">
          <Spinner className="size-6 text-primary/90" />
        </div>
      )}
      <BaseNodeHeader className="border-b border-border/50 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="corner-squircle flex size-9 shrink-0 items-center justify-center rounded-md border border-primary/30 bg-primary/10 text-primary">
            <HugeiconsIcon icon={Layers01Icon} className="size-4.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-base font-semibold">
              {data.title}
            </BaseNodeHeaderTitle>
            <p className="truncate text-xs text-muted-foreground">
              Super card · {data.memberCount} blocks
            </p>
          </div>
        </div>
      </BaseNodeHeader>
      <div className="grid grid-cols-2 gap-3 border-b border-border/50 py-2.5">
        <PortColumn title="Inputs" ports={data.inputs} type="target" />
        <PortColumn title="Outputs" ports={data.outputs} type="source" />
      </div>
      <BaseNodeContent className="gap-2.5 px-4 py-3">
        <NameChips label="Produces" names={data.produces} />
        <p className={cn("flex items-center gap-1.5 text-xs", lastRunTone)}>
          <span className="size-1.5 rounded-full bg-current" />
          Last run: {lastRunLabel(data.lastRun)}
        </p>
      </BaseNodeContent>
      <div className="flex items-center justify-between gap-2 border-t border-border/50 px-3 py-2">
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="nodrag corner-squircle text-muted-foreground"
          disabled={data.executionLocked}
          onClick={(event) => {
            event.stopPropagation();
            ungroupSuperCard(data.cardId);
          }}
        >
          <HugeiconsIcon icon={UngroupLayersIcon} className="size-4" />
          Ungroup
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="nodrag corner-squircle"
          onClick={(event) => {
            event.stopPropagation();
            openSuperCard(data.cardId);
          }}
        >
          <HugeiconsIcon icon={ArrowExpand01Icon} className="size-4" />
          Open
        </Button>
      </div>
    </BaseNode>
  );
}

export const RecipeGraphSuperCardNode = memo(RecipeGraphSuperCardNodeBase);
