import { useEffect, useMemo, useState, type ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type {
  RecipeExecutionAnalysis,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../../execution-types";

type ExecutionsViewProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  currentSignature: string;
  previewLoading: boolean;
  fullLoading: boolean;
  onSelectExecution: (id: string) => void;
  onRunPreview: () => void;
  onRunFull: () => void;
  onCancelExecution: (id: string) => void;
  onLoadDatasetPage: (id: string, page: number) => void;
};

type AnalysisColumnStat = {
  column_name: string;
  column_type: string;
  simple_dtype: string;
  num_unique: number | null;
  num_null: number | null;
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

function parseNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseString(value: unknown): string {
  return typeof value === "string" && value.length > 0 ? value : "--";
}

function parseAnalysisColumns(analysis: RecipeExecutionAnalysis | null): AnalysisColumnStat[] {
  const items = Array.isArray(analysis?.column_statistics)
    ? analysis.column_statistics
    : [];
  return items
    .map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return null;
      }
      const row = item as Record<string, unknown>;
      return {
        column_name: parseString(row.column_name),
        column_type: parseString(row.column_type),
        simple_dtype: parseString(row.simple_dtype),
        num_unique: parseNumber(row.num_unique),
        num_null: parseNumber(row.num_null),
      };
    })
    .filter((item): item is AnalysisColumnStat => item !== null);
}

function isInProgress(status: RecipeExecutionStatus): boolean {
  return (
    status === "running" ||
    status === "active" ||
    status === "pending" ||
    status === "cancelling"
  );
}

function statusTone(status: RecipeExecutionStatus): string {
  if (status === "completed") {
    return "bg-emerald-100 text-emerald-700";
  }
  if (status === "error" || status === "cancelled") {
    return "bg-red-100 text-red-700";
  }
  if (isInProgress(status)) {
    return "bg-amber-100 text-amber-700";
  }
  return "bg-muted text-muted-foreground";
}

function statusRightBorder(status: RecipeExecutionStatus): string {
  if (status === "completed") {
    return "border-r-emerald-500";
  }
  if (status === "error" || status === "cancelled") {
    return "border-r-red-500";
  }
  if (isInProgress(status)) {
    return "border-r-amber-500";
  }
  return "border-r-border";
}

function formatStatus(status: RecipeExecutionStatus): string {
  if (status === "cancelled") {
    return "cancelled";
  }
  return status;
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(1)}%`;
}

function formatDuration(startedAt: number, finishedAt: number | null): string {
  if (!finishedAt || finishedAt <= startedAt) {
    return "--";
  }
  const seconds = Math.round((finishedAt - startedAt) / 1000);
  return `${seconds}s`;
}

export function ExecutionsView({
  executions,
  selectedExecutionId,
  currentSignature,
  previewLoading,
  fullLoading,
  onSelectExecution,
  onRunPreview,
  onRunFull,
  onCancelExecution,
  onLoadDatasetPage,
}: ExecutionsViewProps): ReactElement {
  const [detailTab, setDetailTab] = useState("overview");
  const [showRaw, setShowRaw] = useState(false);
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

  useEffect(() => {
    if (!showRaw && detailTab === "raw") {
      setDetailTab("overview");
    }
  }, [detailTab, showRaw]);

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

  const analysisColumns = useMemo(
    () => parseAnalysisColumns(selectedExecution?.analysis ?? null),
    [selectedExecution?.analysis],
  );
  const columnTypeCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const column of analysisColumns) {
      map.set(column.column_type, (map.get(column.column_type) ?? 0) + 1);
    }
    return Array.from(map.entries());
  }, [analysisColumns]);
  const sideEffects = useMemo(() => {
    const values = selectedExecution?.analysis?.side_effect_column_names;
    return Array.isArray(values)
      ? values.filter((value): value is string => typeof value === "string")
      : [];
  }, [selectedExecution?.analysis?.side_effect_column_names]);

  const canCancel = Boolean(
    selectedExecution?.jobId && isInProgress(selectedExecution.status),
  );
  const datasetPage = selectedExecution?.datasetPage ?? 1;
  const datasetPageSize = selectedExecution?.datasetPageSize ?? 20;
  const datasetTotal = selectedExecution?.datasetTotal ?? 0;
  const totalPages = Math.max(1, Math.ceil(datasetTotal / datasetPageSize));
  const canPageDataset =
    Boolean(selectedExecution?.jobId) && selectedExecution?.kind === "full";

  return (
    <div className="flex h-full min-h-0">
      <aside className="w-72 shrink-0 border-r">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Executions
          </p>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onRunPreview}
              disabled={previewLoading}
            >
              {previewLoading ? "Running..." : "Preview"}
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={onRunFull}
              disabled={fullLoading}
            >
              {fullLoading ? "Starting..." : "Full run"}
            </Button>
          </div>
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
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold capitalize">
                    {selectedExecution.kind} execution
                  </p>
                  <Badge
                    variant="secondary"
                    className={cn("capitalize", statusTone(selectedExecution.status))}
                  >
                    {formatStatus(selectedExecution.status)}
                  </Badge>
                  {isStale && (
                    <Badge variant="outline">Recipe changed since this run</Badge>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {canCancel && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => onCancelExecution(selectedExecution.id)}
                    >
                      Cancel
                    </Button>
                  )}
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowRaw((value) => !value)}
                  >
                    {showRaw ? "Hide raw" : "View raw"}
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Started {formatTimestamp(selectedExecution.createdAt)} |{" "}
                {selectedExecution.rows} rows | Duration{" "}
                {formatDuration(
                  selectedExecution.createdAt,
                  selectedExecution.finishedAt,
                )}
              </p>
              {selectedExecution.stage && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Stage: {selectedExecution.stage}
                  {selectedExecution.current_column
                    ? ` | Column: ${selectedExecution.current_column}`
                    : ""}
                </p>
              )}
            </div>

            {isInProgress(selectedExecution.status) && (
              <div className="space-y-3 rounded-xl border border-amber-200 bg-amber-50/50 p-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold text-amber-900">
                    Run in progress
                  </p>
                  <p className="text-xs text-amber-800">
                    {formatPercent(selectedExecution.progress?.percent)}
                  </p>
                </div>
                <Progress value={selectedExecution.progress?.percent ?? 0} />
                <div className="grid gap-2 text-xs text-amber-900 md:grid-cols-4">
                  <p>
                    Done: {selectedExecution.progress?.done ?? "--"}
                  </p>
                  <p>
                    Total: {selectedExecution.progress?.total ?? "--"}
                  </p>
                  <p>
                    Rate: {selectedExecution.progress?.rate ?? "--"} rec/s
                  </p>
                  <p>
                    ETA: {selectedExecution.progress?.eta_sec ?? "--"} s
                  </p>
                </div>
                {selectedExecution.current_column && selectedExecution.column_progress && (
                  <p className="text-xs text-amber-900">
                    Column {selectedExecution.current_column}:{" "}
                    {selectedExecution.column_progress.done ?? "--"}/
                    {selectedExecution.column_progress.total ?? "--"} (
                    {formatPercent(selectedExecution.column_progress.percent)})
                  </p>
                )}
              </div>
            )}

            {(selectedExecution.status === "error" ||
              selectedExecution.status === "cancelled") && (
              <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-3">
                <p className="text-sm font-semibold text-destructive">
                  {selectedExecution.status === "cancelled"
                    ? "Execution cancelled"
                    : "Execution failed"}
                </p>
                <p className="text-xs text-destructive">
                  {selectedExecution.error ?? "Unknown error."}
                </p>
              </div>
            )}

            {selectedExecution.status === "running" && (
              <div className="space-y-2 rounded-xl border p-3">
                <Skeleton className="h-5 w-44" />
                <Skeleton className="h-20 w-full" />
              </div>
            )}

            {(selectedExecution.status === "completed" ||
              isInProgress(selectedExecution.status)) && (
              <Tabs value={detailTab} onValueChange={setDetailTab}>
                <TabsList>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="columns">Columns</TabsTrigger>
                  <TabsTrigger value="data">Data</TabsTrigger>
                  {showRaw && <TabsTrigger value="raw">Raw</TabsTrigger>}
                </TabsList>
                <TabsContent value="overview" className="mt-3 space-y-3">
                  <div className="grid gap-3 md:grid-cols-4">
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Records</p>
                      <p className="text-lg font-semibold">
                        {selectedExecution.analysis?.num_records ?? "--"}
                      </p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Target</p>
                      <p className="text-lg font-semibold">
                        {selectedExecution.analysis?.target_num_records ?? "--"}
                      </p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Completion</p>
                      <p className="text-lg font-semibold">
                        {formatPercent(
                          selectedExecution.analysis?.num_records &&
                            selectedExecution.analysis?.target_num_records
                            ? (selectedExecution.analysis.num_records /
                                selectedExecution.analysis.target_num_records) *
                                100
                            : null,
                        )}
                      </p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Columns</p>
                      <p className="text-lg font-semibold">
                        {analysisColumns.length > 0 ? analysisColumns.length : "--"}
                      </p>
                    </div>
                  </div>
                  <div className="rounded-xl border p-3">
                    <p className="mb-2 text-sm font-semibold">Column type breakdown</p>
                    {columnTypeCounts.length === 0 ? (
                      <p className="text-xs text-muted-foreground">No analysis yet.</p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {columnTypeCounts.map(([type, count]) => (
                          <Badge key={type} variant="secondary">
                            {type}: {count}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="rounded-xl border p-3">
                    <p className="mb-2 text-sm font-semibold">Side-effect columns</p>
                    {sideEffects.length === 0 ? (
                      <p className="text-xs text-muted-foreground">None.</p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {sideEffects.map((name) => (
                          <Badge key={name} variant="outline">
                            {name}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="columns" className="mt-3">
                  <div className="rounded-xl border p-3">
                    <p className="mb-2 text-sm font-semibold">Column statistics</p>
                    {analysisColumns.length === 0 ? (
                      <p className="text-xs text-muted-foreground">
                        No column statistics yet.
                      </p>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Data type</TableHead>
                            <TableHead>Unique</TableHead>
                            <TableHead>Nulls</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {analysisColumns.map((column) => (
                            <TableRow key={column.column_name}>
                              <TableCell>{column.column_name}</TableCell>
                              <TableCell>{column.column_type}</TableCell>
                              <TableCell>{column.simple_dtype}</TableCell>
                              <TableCell>{column.num_unique ?? "--"}</TableCell>
                              <TableCell>{column.num_null ?? "--"}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="data" className="mt-3">
                  <div className="rounded-xl border p-3">
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm font-semibold">Dataset sample</p>
                      {canPageDataset && selectedExecution && (
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>
                            Page {datasetPage}/{totalPages}
                          </span>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              isInProgress(selectedExecution.status) || datasetPage <= 1
                            }
                            onClick={() =>
                              onLoadDatasetPage(selectedExecution.id, datasetPage - 1)}
                          >
                            Prev
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              isInProgress(selectedExecution.status) ||
                              datasetPage >= totalPages
                            }
                            onClick={() =>
                              onLoadDatasetPage(selectedExecution.id, datasetPage + 1)}
                          >
                            Next
                          </Button>
                        </div>
                      )}
                    </div>
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
                </TabsContent>
                {showRaw && (
                  <TabsContent value="raw" className="mt-3">
                    <div className="rounded-xl border p-3">
                      <p className="mb-2 text-sm font-semibold">Raw execution</p>
                      <pre className="max-h-96 overflow-auto rounded-md bg-muted/40 p-3 text-xs">
                        {JSON.stringify(selectedExecution, null, 2)}
                      </pre>
                    </div>
                  </TabsContent>
                )}
              </Tabs>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
