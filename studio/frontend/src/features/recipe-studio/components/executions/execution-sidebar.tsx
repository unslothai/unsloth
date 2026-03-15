// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RecipeExecutionRecord } from "../../execution-types";
import {
  executionLabel,
  isExecutionInProgress,
  normalizeRunName,
} from "../../executions/execution-helpers";
import {
  formatStatus,
  formatTimestamp,
  statusRightBorder,
  statusTone,
} from "./executions-view-helpers";

type ExecutionSidebarProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  onSelectExecution: (id: string) => void;
};

export function ExecutionSidebar({
  executions,
  selectedExecutionId,
  onSelectExecution,
}: ExecutionSidebarProps): ReactElement {
  return (
    <aside className="w-72 shrink-0 border-r border-border/60 bg-card/20">
      <div className="flex items-center justify-between  border-border/60 px-3 py-2">
        <p className="text-xs font-semibold uppercase text-muted-foreground">
          Runs
        </p>
      </div>
      <div className="h-[calc(100%-45px)] space-y-2 overflow-auto p-2">
        {executions.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border/60 p-3 text-xs text-muted-foreground">
            No runs yet.
          </div>
        ) : (
          executions.map((execution) => {
            const title =
              execution.kind === "full"
                ? (normalizeRunName(execution.run_name) ??
                  executionLabel(execution.kind))
                : executionLabel(execution.kind);
            return (
              <button
                key={execution.id}
                type="button"
                onClick={() => onSelectExecution(execution.id)}
                className={cn(
                  "w-full rounded-xl corner-squircle border border-r-2 border-border/60 bg-card/60 p-3 text-left transition-colors",
                  selectedExecutionId === execution.id
                    ? "border-primary/35 bg-primary/[0.045]"
                    : "hover:bg-muted/25",
                  statusRightBorder(execution.status),
                )}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-medium">
                    {title}
                  </p>
                  <Badge
                    variant="outline"
                    className={cn("capitalize text-[11px]", statusTone(execution.status))}
                  >
                    {formatStatus(execution.status)}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">{execution.rows} rows</p>
                {isExecutionInProgress(execution.status) &&
                  typeof execution.batch?.total === "number" &&
                  execution.batch.total > 1 && (
                    <p className="text-xs text-muted-foreground">
                      Batch {execution.batch.idx ?? "--"}/{execution.batch.total}
                    </p>
                  )}
                <p className="text-xs text-muted-foreground">
                  {formatTimestamp(execution.createdAt)}
                </p>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
