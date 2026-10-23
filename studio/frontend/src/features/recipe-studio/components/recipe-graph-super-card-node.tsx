// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { Layers01Icon } from "@hugeicons/core-free-icons";
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

function PortHandles({
  ports,
  type,
  position,
}: {
  ports: SuperCardPort[];
  type: "source" | "target";
  position: Position;
}): ReactElement {
  const vertical = position === Position.Left || position === Position.Right;
  return (
    <>
      {ports.map((port, index) => {
        const offset = `${((index + 1) / (ports.length + 1)) * 100}%`;
        return (
          <BaseHandle
            key={port.handleId}
            id={port.handleId}
            type={type}
            position={position}
            isConnectable={false}
            title={port.label}
            className={NODE_HANDLE_CLASS}
            style={vertical ? { top: offset } : { left: offset }}
          />
        );
      })}
    </>
  );
}

function NameChips({
  label,
  names,
}: {
  label: string;
  names: string[];
}): ReactElement {
  return (
    <div className="space-y-1">
      <p className="text-ui-11 font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      {names.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing</p>
      ) : (
        <div className="flex flex-wrap gap-1">
          {names.map((name) => (
            <Badge
              key={name}
              variant="outline"
              className="corner-squircle font-mono text-ui-11"
            >
              {name}
            </Badge>
          ))}
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
  const topBottom = data.layoutDirection === "TB";
  const inputKey = data.inputs.map((port) => port.handleId).join("|");
  const outputKey = data.outputs.map((port) => port.handleId).join("|");

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, inputKey, outputKey, topBottom, updateNodeInternals]);

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
        "[box-shadow:4px_4px_0_-1px_var(--card),4px_4px_0_0_var(--border)]",
        data.runtimeState === "running" && "border-ring-strong",
        data.runtimeState === "done" &&
          "border-emerald-500/60 ring-1 ring-emerald-500/20",
      )}
    >
      {data.runtimeState === "running" && (
        <div className="pointer-events-none absolute -top-7 right-2 z-20 rounded-full bg-background">
          <Spinner className="size-6 text-primary/90" />
        </div>
      )}
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div className="corner-squircle flex size-7 items-center justify-center rounded-sm border border-primary/30 bg-primary/10 text-primary">
            <HugeiconsIcon icon={Layers01Icon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">
              {data.title}
            </BaseNodeHeaderTitle>
            <p className="truncate text-ui-11 text-muted-foreground">
              Super card · {data.memberCount} blocks
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="xs"
            variant="ghost"
            className="nodrag"
            disabled={data.executionLocked}
            onClick={(event) => {
              event.stopPropagation();
              ungroupSuperCard(data.cardId);
            }}
          >
            Ungroup
          </Button>
          <Button
            type="button"
            size="xs"
            variant="ghost"
            className="nodrag"
            onClick={(event) => {
              event.stopPropagation();
              openSuperCard(data.cardId);
            }}
          >
            Open
          </Button>
        </div>
      </BaseNodeHeader>
      <BaseNodeContent className="gap-2 px-3 py-2">
        <NameChips
          label="Takes"
          names={data.inputs.map((port) => port.label)}
        />
        <NameChips label="Produces" names={data.produces} />
        <p className={cn("text-xs", lastRunTone)}>
          Last run: {lastRunLabel(data.lastRun)}
        </p>
      </BaseNodeContent>
      <PortHandles
        ports={data.inputs}
        type="target"
        position={topBottom ? Position.Top : Position.Left}
      />
      <PortHandles
        ports={data.outputs}
        type="source"
        position={topBottom ? Position.Bottom : Position.Right}
      />
    </BaseNode>
  );
}

export const RecipeGraphSuperCardNode = memo(RecipeGraphSuperCardNodeBase);
