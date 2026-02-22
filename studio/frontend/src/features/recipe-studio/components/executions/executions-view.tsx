import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import {
  CheckmarkCircle02Icon,
  Database01Icon,
  Database02Icon,
  Flag02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Progress } from "@/components/ui/progress";
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
import { isExecutionInProgress } from "../../executions/execution-helpers";

type ExecutionsViewProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  currentSignature: string;
  onSelectExecution: (id: string) => void;
  onCancelExecution: (id: string) => void;
  onLoadDatasetPage: (id: string, page: number) => void;
};

type AnalysisColumnStat = {
  column_name: string;
  column_type: string;
  simple_dtype: string;
  num_unique: number | null;
  num_null: number | null;
  input_tokens_mean: number | null;
  output_tokens_mean: number | null;
};

type ModelUsageRow = {
  model: string;
  input: number | null;
  output: number | null;
};

const PREVIEW_DATASET_PAGE_SIZE = 20;
const TERMINAL_STICKY_BOTTOM_THRESHOLD_PX = 24;

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

function isExpandableCellValue(value: string): boolean {
  return value.length > 180;
}

function truncateCellValue(value: string): string {
  if (value.length <= 180) {
    return value;
  }
  return `${value.slice(0, 180).trimEnd()}...`;
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
        input_tokens_mean: parseNumber(row.input_tokens_mean),
        output_tokens_mean: parseNumber(row.output_tokens_mean),
      };
    })
    .filter((item): item is AnalysisColumnStat => item !== null);
}

function statusTone(status: RecipeExecutionStatus): string {
  if (status === "completed") {
    return "bg-emerald-100 text-emerald-700";
  }
  if (status === "error" || status === "cancelled") {
    return "bg-red-100 text-red-700";
  }
  if (isExecutionInProgress(status)) {
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
  if (isExecutionInProgress(status)) {
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

function formatMetricValue(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toLocaleString();
}

function parseModelUsageRows(value: Record<string, unknown> | null): ModelUsageRow[] {
  if (!value) {
    return [];
  }
  return Object.entries(value)
    .map(([name, data]) => {
      if (!data || typeof data !== "object" || Array.isArray(data)) {
        return null;
      }
      const modelObj = data as Record<string, unknown>;
      const tokens =
        modelObj.tokens && typeof modelObj.tokens === "object" && !Array.isArray(modelObj.tokens)
          ? (modelObj.tokens as Record<string, unknown>)
          : null;
      const modelName =
        typeof modelObj.model === "string" && modelObj.model.length > 0
          ? modelObj.model
          : name;
      return {
        model: modelName,
        input: parseNumber(tokens?.input),
        output: parseNumber(tokens?.output),
      };
    })
    .filter((item): item is ModelUsageRow => item !== null);
}

export function ExecutionsView({
  executions,
  selectedExecutionId,
  currentSignature,
  onSelectExecution,
  onCancelExecution,
  onLoadDatasetPage,
}: ExecutionsViewProps): ReactElement {
  const [detailTab, setDetailTab] = useState("overview");
  const [hiddenDatasetColumnsByExecution, setHiddenDatasetColumnsByExecution] = useState<
    Record<string, string[]>
  >({});
  const [expandedDatasetRowsByExecution, setExpandedDatasetRowsByExecution] = useState<
    Record<string, Record<string, boolean>>
  >({});
  const [previewDatasetPageByExecution, setPreviewDatasetPageByExecution] = useState<
    Record<string, number>
  >({});
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const shouldStickTerminalToBottomRef = useRef(true);
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

  const selectedExecutionIdSafe = selectedExecution?.id ?? null;
  const hiddenDatasetColumns = useMemo(() => {
    if (!selectedExecutionIdSafe) {
      return [];
    }
    return hiddenDatasetColumnsByExecution[selectedExecutionIdSafe] ?? [];
  }, [hiddenDatasetColumnsByExecution, selectedExecutionIdSafe]);
  const expandedDatasetRows = useMemo(() => {
    if (!selectedExecutionIdSafe) {
      return {};
    }
    return expandedDatasetRowsByExecution[selectedExecutionIdSafe] ?? {};
  }, [expandedDatasetRowsByExecution, selectedExecutionIdSafe]);

  const datasetColumnNames = useMemo(() => {
    if (!selectedExecution) {
      return [];
    }
    const names = new Set<string>();
    for (const row of selectedExecution.dataset) {
      for (const key of Object.keys(row)) {
        names.add(key);
      }
    }
    return Array.from(names);
  }, [selectedExecution]);

  const visibleDatasetColumnNames = useMemo(
    () =>
      datasetColumnNames.filter(
        (name) => !hiddenDatasetColumns.includes(name),
      ),
    [datasetColumnNames, hiddenDatasetColumns],
  );

  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!selectedExecution) {
      return [];
    }
    return visibleDatasetColumnNames.map((name) => ({
      accessorKey: name,
      header: name,
      cell: ({ getValue, row }) => {
        const rawValue = getValue();
        const value = formatCellValue(rawValue);
        const rowExpanded = Boolean(expandedDatasetRows[row.id]);
        const rowHasExpandableCell = visibleDatasetColumnNames.some((columnName) =>
          isExpandableCellValue(formatCellValue(row.original[columnName])),
        );
        const showTruncated = rowHasExpandableCell && !rowExpanded;

        return (
          <div className="max-w-[32rem]">
            <p className="whitespace-pre-wrap break-all">
              {showTruncated ? truncateCellValue(value) : value}
            </p>
          </div>
        );
      },
    }));
  }, [expandedDatasetRows, selectedExecution, visibleDatasetColumnNames]);

  const analysisColumns = useMemo(
    () => parseAnalysisColumns(selectedExecution?.analysis ?? null),
    [selectedExecution?.analysis],
  );
  const modelUsageRows = useMemo(
    () => parseModelUsageRows(selectedExecution?.model_usage ?? null),
    [selectedExecution?.model_usage],
  );
  const totalInputTokens = useMemo(
    () =>
      modelUsageRows.reduce(
        (acc, item) => acc + (typeof item.input === "number" ? item.input : 0),
        0,
      ),
    [modelUsageRows],
  );
  const totalOutputTokens = useMemo(
    () =>
      modelUsageRows.reduce(
        (acc, item) => acc + (typeof item.output === "number" ? item.output : 0),
        0,
      ),
    [modelUsageRows],
  );
  const sideEffects = useMemo(() => {
    const values = selectedExecution?.analysis?.side_effect_column_names;
    return Array.isArray(values)
      ? values.filter((value): value is string => typeof value === "string")
      : [];
  }, [selectedExecution?.analysis?.side_effect_column_names]);

  const canCancel = Boolean(
    selectedExecution?.jobId && isExecutionInProgress(selectedExecution.status),
  );
  const datasetPage = selectedExecution?.datasetPage ?? 1;
  const datasetPageSize = selectedExecution?.datasetPageSize ?? 20;
  const datasetTotal = selectedExecution?.datasetTotal ?? 0;
  const previewPageRaw = selectedExecutionIdSafe
    ? previewDatasetPageByExecution[selectedExecutionIdSafe] ?? 1
    : 1;
  const previewTotalPages = useMemo(() => {
    if (!selectedExecution || selectedExecution.kind !== "preview") {
      return 1;
    }
    return Math.max(
      1,
      Math.ceil(selectedExecution.dataset.length / PREVIEW_DATASET_PAGE_SIZE),
    );
  }, [selectedExecution]);
  const previewPage = Math.min(previewPageRaw, previewTotalPages);
  const totalPages =
    selectedExecution?.kind === "preview"
      ? previewTotalPages
      : Math.max(1, Math.ceil(datasetTotal / datasetPageSize));
  const canPageDataset =
    selectedExecution?.kind === "preview" ||
    (selectedExecution?.kind === "full" && Boolean(selectedExecution.jobId));
  const datasetRowsForTable = useMemo(() => {
    if (!selectedExecution) {
      return [];
    }
    if (selectedExecution.kind !== "preview") {
      return selectedExecution.dataset;
    }
    const start = (previewPage - 1) * PREVIEW_DATASET_PAGE_SIZE;
    return selectedExecution.dataset.slice(start, start + PREVIEW_DATASET_PAGE_SIZE);
  }, [previewPage, selectedExecution]);
  const currentDatasetPage = selectedExecution?.kind === "preview" ? previewPage : datasetPage;
  const recordsMetric = useMemo(() => {
    if (!selectedExecution || selectedExecution.status !== "completed") {
      return null;
    }
    if (typeof selectedExecution.analysis?.num_records === "number") {
      return selectedExecution.analysis.num_records;
    }
    if (selectedExecution.datasetTotal > 0) {
      return selectedExecution.datasetTotal;
    }
    if (selectedExecution.dataset.length > 0) {
      return selectedExecution.dataset.length;
    }
    return null;
  }, [selectedExecution]);
  const totalMetric = useMemo(() => {
    if (!selectedExecution || selectedExecution.status !== "completed") {
      return null;
    }
    if (typeof selectedExecution.analysis?.target_num_records === "number") {
      return selectedExecution.analysis.target_num_records;
    }
    return selectedExecution.rows > 0 ? selectedExecution.rows : null;
  }, [selectedExecution]);
  const columnCount = analysisColumns.length;
  const llmColumnCount = useMemo(
    () =>
      analysisColumns.reduce(
        (acc, column) => (column.column_type.startsWith("llm") ? acc + 1 : acc),
        0,
      ),
    [analysisColumns],
  );
  const totalNulls = useMemo(
    () =>
      analysisColumns.reduce(
        (acc, column) => acc + (typeof column.num_null === "number" ? column.num_null : 0),
        0,
      ),
    [analysisColumns],
  );
  const nullRate = useMemo(() => {
    if (
      typeof recordsMetric !== "number" ||
      recordsMetric <= 0 ||
      columnCount <= 0
    ) {
      return null;
    }
    return (totalNulls / (recordsMetric * columnCount)) * 100;
  }, [columnCount, recordsMetric, totalNulls]);
  const lowUniquenessColumns = useMemo(() => {
    if (typeof recordsMetric !== "number" || recordsMetric <= 0) {
      return [];
    }
    return analysisColumns
      .filter(
        (column) =>
          typeof column.num_unique === "number" &&
          column.num_unique / recordsMetric < 0.5,
      )
      .map((column) => column.column_name);
  }, [analysisColumns, recordsMetric]);
  const runDuration = useMemo(() => {
    if (!selectedExecution) {
      return "--";
    }
    return formatDuration(selectedExecution.createdAt, selectedExecution.finishedAt);
  }, [selectedExecution]);
  const showSummaryCards = selectedExecution?.status === "completed";
  const showProgressPanel =
    selectedExecution?.status === "completed" ||
    (selectedExecution ? isExecutionInProgress(selectedExecution.status) : false);
  const progressComplete = selectedExecution?.status === "completed";
  const progressPercent = selectedExecution?.progress?.percent ?? (progressComplete ? 100 : 0);
  const terminalLines = selectedExecution?.log_lines ?? [];
  const rawExecution = useMemo(() => {
    if (!selectedExecution) {
      return null;
    }
    const next = { ...selectedExecution } as Record<string, unknown>;
    delete next.dataset;
    delete next.log_lines;
    return next;
  }, [selectedExecution]);

  useEffect(() => {
    if (!terminalRef.current) {
      return;
    }
    shouldStickTerminalToBottomRef.current = true;
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [selectedExecution?.id]);

  useEffect(() => {
    if (!terminalRef.current) {
      return;
    }
    if (!shouldStickTerminalToBottomRef.current) {
      return;
    }
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [terminalLines.length]);

  return (
    <div className="flex h-full min-h-0">
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
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span className="font-medium capitalize">{selectedExecution.kind} execution</span>
              <Badge
                variant="secondary"
                className={cn("capitalize", statusTone(selectedExecution.status))}
              >
                {formatStatus(selectedExecution.status)}
              </Badge>
              <span>{selectedExecution.rows} rows</span>
              <span>Started {formatTimestamp(selectedExecution.createdAt)}</span>
              <span>Duration {formatDuration(selectedExecution.createdAt, selectedExecution.finishedAt)}</span>
              {selectedExecution.stage && (
                <span>
                  Stage: {selectedExecution.stage}
                  {selectedExecution.current_column
                    ? ` | Column: ${selectedExecution.current_column}`
                    : ""}
                </span>
              )}
              {isStale && <Badge variant="outline">Recipe changed since this run</Badge>}
            </div>

            {showProgressPanel && (
              <div
                className={cn(
                  "space-y-3 rounded-xl border p-3",
                  progressComplete
                    ? "border-emerald-200 bg-emerald-50/50"
                    : "border-amber-200 bg-amber-50/50",
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={progressComplete ? CheckmarkCircle02Icon : Flag02Icon}
                      className={cn(
                        "size-4",
                        progressComplete ? "text-emerald-700" : "text-amber-700",
                      )}
                    />
                    <p
                      className={cn(
                        "text-sm font-semibold",
                        progressComplete ? "text-emerald-900" : "text-amber-900",
                      )}
                    >
                      {progressComplete ? "Run completed" : "Run in progress"}
                    </p>
                  </div>
                  <p
                    className={cn(
                      "text-xs",
                      progressComplete ? "text-emerald-800" : "text-amber-800",
                    )}
                  >
                    {formatPercent(progressPercent)}
                  </p>
                </div>
                <Progress value={progressPercent} />
                <div
                  className={cn(
                    "grid gap-2 text-xs md:grid-cols-4",
                    progressComplete ? "text-emerald-900" : "text-amber-900",
                  )}
                >
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
                  <p
                    className={cn(
                      "text-xs",
                      progressComplete ? "text-emerald-900" : "text-amber-900",
                    )}
                  >
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

            {(selectedExecution.status === "completed" ||
              isExecutionInProgress(selectedExecution.status)) && (
              <Tabs value={detailTab} onValueChange={setDetailTab}>
                <div className="flex items-center justify-between gap-2">
                  <TabsList>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="columns">Columns</TabsTrigger>
                    <TabsTrigger value="data">Data</TabsTrigger>
                    <TabsTrigger value="raw">Raw</TabsTrigger>
                  </TabsList>
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
                </div>
                <TabsContent value="overview" className="mt-3 space-y-3">
                  {showSummaryCards && (
                    <div className="space-y-3">
                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="h-full rounded-lg bg-muted/20 p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-xs text-muted-foreground">Run summary</p>
                            <HugeiconsIcon
                              icon={Database01Icon}
                              className="size-4 text-muted-foreground"
                            />
                          </div>
                          <div className="space-y-1 text-xs">
                            <p>
                              Records:{" "}
                              <span className="font-semibold">
                                {formatMetricValue(recordsMetric)} / {formatMetricValue(totalMetric)}
                              </span>
                            </p>
                            <p>
                              Duration: <span className="font-semibold">{runDuration}</span>
                            </p>
                            <p>
                              Columns analyzed:{" "}
                              <span className="font-semibold">
                                {formatMetricValue(columnCount)}
                              </span>
                            </p>
                            <p>
                              Final stage:{" "}
                              <span className="font-semibold">
                                {selectedExecution.stage ?? "--"}
                              </span>
                            </p>
                          </div>
                        </div>
                        <div className="h-full rounded-lg bg-muted/20 p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-xs text-muted-foreground">Insights</p>
                            <HugeiconsIcon
                              icon={Database02Icon}
                              className="size-4 text-muted-foreground"
                            />
                          </div>
                          <div className="space-y-2 text-xs">
                            <p>
                              LLM columns:{" "}
                              <span className="font-semibold">
                                {formatMetricValue(llmColumnCount)}
                              </span>
                            </p>
                            <p>
                              Null rate:{" "}
                              <span className="font-semibold">{formatPercent(nullRate)}</span>
                            </p>
                            <p>
                              Dropped columns:{" "}
                              <span className="font-semibold">
                                {formatMetricValue(sideEffects.length)}
                              </span>
                            </p>
                            {sideEffects.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {sideEffects.map((name) => (
                                  <Badge key={name} variant="outline">
                                    {name}
                                  </Badge>
                                ))}
                              </div>
                            )}
                            <p>
                              Low uniqueness flags:{" "}
                              <span className="font-semibold">
                                {formatMetricValue(lowUniquenessColumns.length)}
                              </span>
                            </p>
                            {lowUniquenessColumns.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {lowUniquenessColumns.slice(0, 3).map((name) => (
                                  <Badge key={name} variant="secondary">
                                    {name}
                                  </Badge>
                                ))}
                                {lowUniquenessColumns.length > 3 && (
                                  <Badge variant="secondary">
                                    +{lowUniquenessColumns.length - 3} more
                                  </Badge>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="rounded-lg bg-muted/20 p-3">
                        <div className="mb-2 flex items-center justify-between">
                          <p className="text-xs text-muted-foreground">Model usage</p>
                          <HugeiconsIcon
                            icon={Flag02Icon}
                            className="size-4 text-muted-foreground"
                          />
                        </div>
                        {modelUsageRows.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No model usage yet.</p>
                        ) : (
                          <div className="space-y-2 text-xs">
                            <div className="grid grid-cols-2 gap-2">
                              <div className="rounded border bg-muted/30 px-2 py-1.5">
                                <p className="text-muted-foreground">Total input</p>
                                <p className="text-sm font-semibold">
                                  {formatMetricValue(totalInputTokens)}
                                </p>
                              </div>
                              <div className="rounded border bg-muted/30 px-2 py-1.5">
                                <p className="text-muted-foreground">Total output</p>
                                <p className="text-sm font-semibold">
                                  {formatMetricValue(totalOutputTokens)}
                                </p>
                              </div>
                            </div>
                            <div className="overflow-hidden rounded-md border">
                              <Table>
                                <TableHeader>
                                  <TableRow>
                                    <TableHead>Model</TableHead>
                                    <TableHead className="text-right">Input</TableHead>
                                    <TableHead className="text-right">Output</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {modelUsageRows.map((usage) => (
                                    <TableRow key={usage.model}>
                                      <TableCell className="max-w-[320px] truncate">
                                        {usage.model}
                                      </TableCell>
                                      <TableCell className="text-right">
                                        {formatMetricValue(usage.input)}
                                      </TableCell>
                                      <TableCell className="text-right">
                                        {formatMetricValue(usage.output)}
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  <div className="overflow-hidden rounded-xl corner-squircle border">
                    <div className="flex items-center justify-between border-b px-3 py-2">
                      <p className="text-sm font-semibold">Terminal output</p>
                      <p className="text-xs text-muted-foreground">
                        {terminalLines.length} lines
                      </p>
                    </div>
                    <div
                      ref={terminalRef}
                      className="max-h-72 overflow-auto bg-zinc-900/80 px-3 py-2 font-mono text-xs text-zinc-200"
                      onScroll={(event) => {
                        const element = event.currentTarget;
                        const distanceFromBottom =
                          element.scrollHeight - element.scrollTop - element.clientHeight;
                        shouldStickTerminalToBottomRef.current =
                          distanceFromBottom <= TERMINAL_STICKY_BOTTOM_THRESHOLD_PX;
                      }}
                    >
                      {terminalLines.length === 0 ? (
                        <p className="text-zinc-400">
                          {isExecutionInProgress(selectedExecution.status)
                            ? "Waiting for logs..."
                            : "No logs captured."}
                        </p>
                      ) : (
                        terminalLines.map((line, index) => (
                          <p
                            key={`${index}-${line.slice(0, 24)}`}
                            className="whitespace-pre-wrap break-words leading-relaxed"
                          >
                            {line}
                          </p>
                        ))
                      )}
                    </div>
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
                            <TableHead>Input tok avg</TableHead>
                            <TableHead>Output tok avg</TableHead>
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
                              <TableCell>{column.input_tokens_mean ?? "--"}</TableCell>
                              <TableCell>{column.output_tokens_mean ?? "--"}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="data" className="mt-3">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm font-semibold">Dataset sample</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {datasetColumnNames.length > 0 && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button type="button" size="sm" variant="outline">
                              Columns
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Visible columns</DropdownMenuLabel>
                            {datasetColumnNames.map((columnName) => (
                              <DropdownMenuCheckboxItem
                                key={columnName}
                                checked={!hiddenDatasetColumns.includes(columnName)}
                                onSelect={(event) => {
                                  event.preventDefault();
                                }}
                                onCheckedChange={(checked) => {
                                  const selectedId = selectedExecution?.id;
                                  if (!selectedId) {
                                    return;
                                  }
                                  setHiddenDatasetColumnsByExecution((current) => {
                                    const currentColumns = current[selectedId] ?? [];
                                    const nextColumns = checked
                                      ? currentColumns.filter((name) => name !== columnName)
                                      : [...currentColumns, columnName];
                                    return {
                                      ...current,
                                      [selectedId]: nextColumns,
                                    };
                                  });
                                }}
                              >
                                {columnName}
                              </DropdownMenuCheckboxItem>
                            ))}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                      {canPageDataset && selectedExecution && (
                        <>
                          <span>
                            Page {currentDatasetPage}/{totalPages}
                          </span>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              isExecutionInProgress(selectedExecution.status) ||
                              currentDatasetPage <= 1
                            }
                            onClick={() => {
                              if (selectedExecution.kind === "preview") {
                                const selectedId = selectedExecution.id;
                                setPreviewDatasetPageByExecution((current) => ({
                                  ...current,
                                  [selectedId]: Math.max(1, currentDatasetPage - 1),
                                }));
                                return;
                              }
                              onLoadDatasetPage(selectedExecution.id, currentDatasetPage - 1);
                            }}
                          >
                            Prev
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              isExecutionInProgress(selectedExecution.status) ||
                              currentDatasetPage >= totalPages
                            }
                            onClick={() => {
                              if (selectedExecution.kind === "preview") {
                                const selectedId = selectedExecution.id;
                                setPreviewDatasetPageByExecution((current) => ({
                                  ...current,
                                  [selectedId]: Math.min(totalPages, currentDatasetPage + 1),
                                }));
                                return;
                              }
                              onLoadDatasetPage(selectedExecution.id, currentDatasetPage + 1);
                            }}
                          >
                            Next
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                  {selectedExecution.dataset.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No rows returned.</p>
                  ) : tableColumns.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      All columns hidden. Use Columns to show at least one.
                    </p>
                  ) : (
                    <div className="max-h-[55vh] overflow-auto">
                      <DataTable
                        columns={tableColumns}
                        data={datasetRowsForTable}
                        getRowClassName={(row, _rowIndex, rowId) => {
                          const canExpand = visibleDatasetColumnNames.some((columnName) =>
                            isExpandableCellValue(formatCellValue(row[columnName])),
                          );
                          if (!canExpand) {
                            return undefined;
                          }
                          return cn(
                            "cursor-pointer",
                            expandedDatasetRows[rowId]
                              ? "bg-primary/[0.05]"
                              : "hover:bg-primary/[0.06]",
                          );
                        }}
                        onRowClick={(row, _rowIndex, rowId) => {
                          const canExpand = visibleDatasetColumnNames.some((columnName) =>
                            isExpandableCellValue(formatCellValue(row[columnName])),
                          );
                          if (!canExpand || !selectedExecutionIdSafe) {
                            return;
                          }
                          setExpandedDatasetRowsByExecution((current) => {
                            const rows = current[selectedExecutionIdSafe] ?? {};
                            return {
                              ...current,
                              [selectedExecutionIdSafe]: {
                                ...rows,
                                [rowId]: !rows[rowId],
                              },
                            };
                          });
                        }}
                      />
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="raw" className="mt-3">
                  <div className="rounded-xl border p-3">
                    <p className="mb-2 text-sm font-semibold">Raw execution</p>
                    <pre className="max-h-96 overflow-auto rounded-md bg-muted/40 p-3 text-xs">
                      {JSON.stringify(rawExecution, null, 2)}
                    </pre>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
