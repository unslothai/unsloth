import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import {
  CheckmarkCircle02Icon,
  Flag02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type {
  RecipeExecutionRecord,
} from "../../execution-types";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import { ExecutionColumnsTab } from "./execution-columns-tab";
import { ExecutionDataTab } from "./execution-data-tab";
import { ExecutionOverviewTab } from "./execution-overview-tab";
import { ExecutionRawTab } from "./execution-raw-tab";
import { ExecutionSidebar } from "./execution-sidebar";
import {
  PREVIEW_DATASET_PAGE_SIZE,
  TERMINAL_STICKY_BOTTOM_THRESHOLD_PX,
  formatCellValue,
  formatDuration,
  formatPercent,
  formatStatus,
  formatTimestamp,
  isExpandableCellValue,
  parseAnalysisColumns,
  parseModelUsageRows,
  statusTone,
  truncateCellValue,
} from "./executions-view-helpers";

type ExecutionsViewProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  currentSignature: string;
  onSelectExecution: (id: string) => void;
  onCancelExecution: (id: string) => void;
  onLoadDatasetPage: (id: string, page: number) => void;
};

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
  const batchTotal = selectedExecution?.batch?.total ?? null;
  const batchIdx = selectedExecution?.batch?.idx ?? null;
  const showBatchProgress = typeof batchTotal === "number" && batchTotal > 1;
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
      <ExecutionSidebar
        executions={executions}
        selectedExecutionId={selectedExecutionId}
        onSelectExecution={onSelectExecution}
      />
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
              <span>
                Duration {formatDuration(selectedExecution.createdAt, selectedExecution.finishedAt)}
              </span>
              {selectedExecution.stage && (
                <span>
                  Stage: {selectedExecution.stage}
                  {selectedExecution.current_column
                    ? ` | Column: ${selectedExecution.current_column}`
                    : ""}
                </span>
              )}
              {showBatchProgress && (
                <span>
                  Batch {batchIdx ?? "--"}/{batchTotal}
                </span>
              )}
              {isStale && <Badge variant="outline">Recipe changed since this run</Badge>}
            </div>

            {showProgressPanel && (
              <div
                className={cn(
                  "space-y-3 rounded-xl border p-3",
                  progressComplete
                    ? "border-emerald-200 bg-emerald-50/50 dark:border-emerald-900/50 dark:bg-emerald-950/25"
                    : "border-amber-200 bg-amber-50/50 dark:border-amber-900/50 dark:bg-amber-950/25",
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={progressComplete ? CheckmarkCircle02Icon : Flag02Icon}
                      className={cn(
                        "size-4",
                        progressComplete
                          ? "text-emerald-700 dark:text-emerald-300"
                          : "text-amber-700 dark:text-amber-300",
                      )}
                    />
                    <p
                      className={cn(
                        "text-sm font-semibold",
                        progressComplete
                          ? "text-emerald-900 dark:text-emerald-100"
                          : "text-amber-900 dark:text-amber-100",
                      )}
                    >
                      {progressComplete ? "Run completed" : "Run in progress"}
                    </p>
                  </div>
                  <p
                    className={cn(
                      "text-xs",
                      progressComplete
                        ? "text-emerald-800 dark:text-emerald-200"
                        : "text-amber-800 dark:text-amber-200",
                    )}
                  >
                    {formatPercent(progressPercent)}
                  </p>
                </div>
                <Progress value={progressPercent} />
                <div
                  className={cn(
                    "grid gap-2 text-xs md:grid-cols-4",
                    progressComplete
                      ? "text-emerald-900 dark:text-emerald-100"
                      : "text-amber-900 dark:text-amber-100",
                  )}
                >
                  <p>Done: {selectedExecution.progress?.done ?? "--"}</p>
                  <p>Total: {selectedExecution.progress?.total ?? "--"}</p>
                  <p>Rate: {selectedExecution.progress?.rate ?? "--"} rec/s</p>
                  <p>ETA: {selectedExecution.progress?.eta_sec ?? "--"} s</p>
                </div>
                {selectedExecution.current_column && selectedExecution.column_progress && (
                  <p
                    className={cn(
                      "text-xs",
                      progressComplete
                        ? "text-emerald-900 dark:text-emerald-100"
                        : "text-amber-900 dark:text-amber-100",
                    )}
                  >
                    Column {selectedExecution.current_column}:{" "}
                    {selectedExecution.column_progress.done ?? "--"}/
                    {selectedExecution.column_progress.total ?? "--"} (
                    {formatPercent(selectedExecution.column_progress.percent)})
                  </p>
                )}
                {showBatchProgress && (
                  <p
                    className={cn(
                      "text-xs",
                      progressComplete
                        ? "text-emerald-900 dark:text-emerald-100"
                        : "text-amber-900 dark:text-amber-100",
                    )}
                  >
                    Processed batch: {batchIdx ?? "--"}/{batchTotal}
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
                <TabsContent value="overview">
                  <ExecutionOverviewTab
                    execution={selectedExecution}
                    showSummaryCards={showSummaryCards}
                    recordsMetric={recordsMetric}
                    totalMetric={totalMetric}
                    runDuration={runDuration}
                    columnCount={columnCount}
                    llmColumnCount={llmColumnCount}
                    nullRate={nullRate}
                    sideEffects={sideEffects}
                    lowUniquenessColumns={lowUniquenessColumns}
                    modelUsageRows={modelUsageRows}
                    terminalLines={terminalLines}
                    terminalRef={terminalRef}
                    onTerminalScroll={(event) => {
                      const element = event.currentTarget;
                      const distanceFromBottom =
                        element.scrollHeight - element.scrollTop - element.clientHeight;
                      shouldStickTerminalToBottomRef.current =
                        distanceFromBottom <= TERMINAL_STICKY_BOTTOM_THRESHOLD_PX;
                    }}
                  />
                </TabsContent>
                <TabsContent value="columns">
                  <ExecutionColumnsTab analysisColumns={analysisColumns} />
                </TabsContent>
                <TabsContent value="data">
                  <ExecutionDataTab
                    execution={selectedExecution}
                    datasetColumnNames={datasetColumnNames}
                    hiddenDatasetColumns={hiddenDatasetColumns}
                    canPageDataset={canPageDataset}
                    currentDatasetPage={currentDatasetPage}
                    totalPages={totalPages}
                    tableColumns={tableColumns}
                    datasetRowsForTable={datasetRowsForTable}
                    visibleDatasetColumnNames={visibleDatasetColumnNames}
                    expandedDatasetRows={expandedDatasetRows}
                    selectedExecutionIdSafe={selectedExecutionIdSafe}
                    onSetHiddenColumns={(updater) => {
                      const selectedId = selectedExecution.id;
                      setHiddenDatasetColumnsByExecution((current) => {
                        const currentColumns = current[selectedId] ?? [];
                        return {
                          ...current,
                          [selectedId]: updater(currentColumns),
                        };
                      });
                    }}
                    onPrevPage={() => {
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
                    onNextPage={() => {
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
                    onToggleRowExpanded={(rowId) => {
                      setExpandedDatasetRowsByExecution((current) => {
                        const rows = current[selectedExecution.id] ?? {};
                        return {
                          ...current,
                          [selectedExecution.id]: {
                            ...rows,
                            [rowId]: !rows[rowId],
                          },
                        };
                      });
                    }}
                  />
                </TabsContent>
                <TabsContent value="raw">
                  <ExecutionRawTab rawExecution={rawExecution} />
                </TabsContent>
              </Tabs>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
