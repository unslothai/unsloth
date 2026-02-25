import type { ReactElement } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RecipeExecutionRecord } from "../../execution-types";
import { isExecutionInProgress } from "../../executions/execution-helpers";
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
    <aside className="w-72 shrink-0 border-r">
      <div className="flex items-center justify-between border-b px-3 py-2">
        <p className="text-xs font-semibold uppercase text-muted-foreground">
          Executions
        </p>
      </div>
      <div className="h-[calc(100%-45px)] overflow-auto p-2">
        {executions.length === 0 ? (
          <div className="rounded-xl border border-dashed p-3 text-xs text-muted-foreground">
            No executions yet.
          </div>
        ) : (
          executions.map((execution) => (
            <button
              key={execution.id}
              type="button"
              onClick={() => onSelectExecution(execution.id)}
              className={cn(
                "mb-2 w-full rounded-xl corner-squircle border border-r-4 p-3 text-left",
                selectedExecutionId === execution.id
                  ? "border-primary/50 bg-primary/5"
                  : "hover:bg-muted/40",
                statusRightBorder(execution.status),
              )}
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium capitalize">
                  {execution.kind}
                </p>
                <Badge
                  variant="secondary"
                  className={cn("capitalize", statusTone(execution.status))}
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
          ))
        )}
      </div>
    </aside>
  );
}
