import { useMemo, type ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { RecipeExecutionRecord } from "../../execution-types";

type ExecutionsViewProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  currentSignature: string;
  previewLoading: boolean;
  onSelectExecution: (id: string) => void;
  onRunPreview: () => void;
};

function formatTimestamp(value: number): string {
  return new Date(value).toLocaleString();
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function statusTone(status: RecipeExecutionRecord["status"]): string {
  if (status === "completed") {
    return "bg-emerald-100 text-emerald-700";
  }
  if (status === "error") {
    return "bg-red-100 text-red-700";
  }
  return "bg-amber-100 text-amber-700";
}

export function ExecutionsView({
  executions,
  selectedExecutionId,
  currentSignature,
  previewLoading,
  onSelectExecution,
  onRunPreview,
}: ExecutionsViewProps): ReactElement {
  const selectedExecution = useMemo(
    () =>
      executions.find((execution) => execution.id === selectedExecutionId) ??
      null,
    [executions, selectedExecutionId],
  );
  const isStale = Boolean(
    selectedExecution &&
      selectedExecution.recipeSignature.length > 0 &&
      selectedExecution.recipeSignature !== currentSignature,
  );

  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!selectedExecution) {
      return [];
    }
    const names = new Set<string>();
    for (const row of selectedExecution.dataset) {
      for (const key of Object.keys(row)) {
        names.add(key);
      }
    }
    return Array.from(names).map((name) => ({
      accessorKey: name,
      header: name,
      cell: ({ getValue }) => {
        const value = getValue();
        return (
          <p className="max-w-[32rem] whitespace-pre-wrap break-all">
            {formatCellValue(value)}
          </p>
        );
      },
    }));
  }, [selectedExecution]);

  return (
    <div className="flex h-full min-h-0">
      <aside className="w-72 shrink-0 border-r">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Executions
          </p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={onRunPreview}
            disabled={previewLoading}
          >
            {previewLoading ? "Running..." : "Run preview"}
          </Button>
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
                  "mb-2 w-full rounded-xl corner-squircle border p-3 text-left",
                  selectedExecutionId === execution.id
                    ? "border-primary/50 bg-primary/5"
                    : "hover:bg-muted/40",
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
                    {execution.status}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {execution.rows} rows
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatTimestamp(execution.createdAt)}
                </p>
              </button>
            ))
          )}
        </div>
      </aside>
      <section className="min-w-0 flex-1 overflow-auto p-4">
        {!selectedExecution ? (
          <div className="rounded-xl border border-dashed p-4 text-sm text-muted-foreground">
            Select an execution.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="rounded-xl border p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <p className="text-sm font-semibold capitalize">
                  {selectedExecution.kind} execution
                </p>
                <Badge
                  variant="secondary"
                  className={cn("capitalize", statusTone(selectedExecution.status))}
                >
                  {selectedExecution.status}
                </Badge>
                {isStale && (
                  <Badge variant="outline">Recipe changed since this run</Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Started {formatTimestamp(selectedExecution.createdAt)} |{" "}
                {selectedExecution.rows} rows
              </p>
            </div>

            {selectedExecution.status === "running" && (
              <div className="space-y-2 rounded-xl border p-3">
                <Skeleton className="h-5 w-44" />
                <Skeleton className="h-20 w-full" />
              </div>
            )}

            {selectedExecution.status === "error" && (
              <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-3">
                <p className="text-sm font-semibold text-destructive">Preview failed</p>
                <p className="text-xs text-destructive">
                  {selectedExecution.error ?? "Unknown error."}
                </p>
              </div>
            )}

            {selectedExecution.status === "completed" && (
              <>
                <div className="rounded-xl border p-3">
                  <p className="mb-2 text-sm font-semibold">Analysis (full)</p>
                  <pre className="max-h-80 overflow-auto rounded-md bg-muted/40 p-3 text-xs">
                    {JSON.stringify(selectedExecution.analysis ?? {}, null, 2)}
                  </pre>
                </div>
                <div className="rounded-xl border p-3">
                  <p className="mb-2 text-sm font-semibold">Preview data</p>
                  {selectedExecution.dataset.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No rows returned.</p>
                  ) : (
                    <div className="max-h-[55vh] overflow-auto">
                      <DataTable
                        columns={tableColumns}
                        data={selectedExecution.dataset}
                      />
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
