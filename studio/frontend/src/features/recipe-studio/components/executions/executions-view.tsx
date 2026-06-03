// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import {
  CheckmarkCircle02Icon,
  Flag02Icon,
  Share08Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { publishRecipeJob } from "../../api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { resolveImagePreview } from "../../utils/image-preview";
import type {
  RecipeExecutionRecord,
} from "../../execution-types";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import { ExecutionColumnsTab } from "./execution-columns-tab";
import { ExecutionDataTab } from "./execution-data-tab";
import { ExecutionOverviewTab } from "./execution-overview-tab";
import { ExecutionRawTab } from "./execution-raw-tab";
import { ExecutionSidebar } from "./execution-sidebar";
import { PublishExecutionDialog } from "./publish-execution-dialog";
import {
  PREVIEW_DATASET_PAGE_SIZE,
  TERMINAL_STICKY_BOTTOM_THRESHOLD_PX,
  formatCellValue,
  formatDuration,
  formatPercent,
  isExpandableCellValue,
  parseAnalysisColumns,
  parseModelUsageRows,
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
  const formatEta = (value: number | null | undefined): string =>
    typeof value === "number" && Number.isFinite(value)
      ? `${value.toLocaleString()} s`
      : "--";
  const [detailTab, setDetailTab] = useState("data");
  const [hiddenDatasetColumnsByExecution, setHiddenDatasetColumnsByExecution] = useState<
    Record<string, string[]>
  >({});
  const [previewDatasetPageByExecution, setPreviewDatasetPageByExecution] = useState<
    Record<string, number>
  >({});
  const [publishDialogOpen, setPublishDialogOpen] = useState(false);
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

  // Columns where at least one row has text long enough that it would wrap at
  // the default narrow width. We give those columns a wider min-width so the
  // text is readable without clicking anything. The table's wrapper already
  // scrolls horizontally, so a few wide columns just add a horizontal
  // scrollbar instead of squeezing everything into the viewport.
  const wideColumns = useMemo(() => {
    const result = new Set<string>();
    if (!selectedExecution) {
      return result;
    }
    for (const row of selectedExecution.dataset) {
      for (const name of visibleDatasetColumnNames) {
        if (result.has(name)) {
          continue;
        }
        const raw = row[name];
        if (resolveImagePreview(raw)) {
          continue;
        }
        if (isExpandableCellValue(formatCellValue(raw))) {
          result.add(name);
        }
      }
      if (result.size === visibleDatasetColumnNames.length) {
        break;
      }
    }
    return result;
  }, [selectedExecution, visibleDatasetColumnNames]);

  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!selectedExecution) {
      return [];
    }
    return visibleDatasetColumnNames.map((name) => ({
      accessorKey: name,
      header: name,
      cell: ({ getValue }) => {
        const rawValue = getValue();
        const imagePreview = resolveImagePreview(rawValue);
        if (imagePreview?.kind === "ready") {
          return (
            <div>
              <img
                src={imagePreview.src}
                alt={`${name} preview`}
                loading="lazy"
                className="h-24 w-auto max-w-[260px] rounded-md border border-border/60 bg-muted/20 object-contain"
              />
            </div>
          );
        }
        if (imagePreview?.kind === "too_large") {
          return (
            <p className="text-xs text-muted-foreground">
              Image too large to preview
            </p>
          );
        }
        const value = formatCellValue(rawValue);
        const isWide = wideColumns.has(name);
        return (
          <div className={cn(isWide ? "min-w-[48rem]" : "min-w-[12rem]")}>
            <p className="whitespace-pre-wrap break-all">{value}</p>
          </div>
        );
      },
    }));
  }, [selectedExecution, visibleDatasetColumnNames, wideColumns]);

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
  const canPublish = Boolean(
    selectedExecution &&
      selectedExecution.kind === "full" &&
      selectedExecution.status === "completed" &&
      selectedExecution.jobId &&
      selectedExecution.artifact_path,
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
  const hasProgressSnapshot = Boolean(
    selectedExecution?.progress &&
      (typeof selectedExecution.progress.done === "number" ||
        typeof selectedExecution.progress.total === "number" ||
        typeof selectedExecution.progress.percent === "number" ||
        typeof selectedExecution.progress.rate === "number" ||
        typeof selectedExecution.progress.eta_sec === "number"),
  ) || Boolean(
    selectedExecution?.column_progress &&
      (typeof selectedExecution.column_progress.done === "number" ||
        typeof selectedExecution.column_progress.total === "number" ||
        typeof selectedExecution.column_progress.percent === "number"),
  ) || Boolean(
    selectedExecution?.batch &&
      (typeof selectedExecution.batch.idx === "number" ||
        typeof selectedExecution.batch.total === "number"),
  );
  const selectedStatus = selectedExecution?.status ?? null;
  const isSelectedExecutionInProgress = selectedStatus
    ? isExecutionInProgress(selectedStatus)
    : false;
  const showProgressPanel = Boolean(selectedExecution) && (
    selectedStatus === "completed" ||
    isSelectedExecutionInProgress ||
    hasProgressSnapshot
  );
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
    setDetailTab("data");
  }, [selectedExecution?.id]);

  useEffect(() => {
    if (detailTab !== "overview" || !terminalRef.current) {
      return;
    }
    shouldStickTerminalToBottomRef.current = true;
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [detailTab, selectedExecution?.id]);

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
          <div className="rounded-xl border border-dashed border-border/60 p-4 text-sm text-muted-foreground">
            Select an execution.
          </div>
        ) : (
          <div className="space-y-4">
            {showProgressPanel && (
              <div className="space-y-3 rounded-2xl border shadow-border border-border/60 bg-card/55 p-3">
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
                    <p className="text-sm font-semibold text-foreground">
                      Progress
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">{formatPercent(progressPercent)}</p>
                </div>
                <Progress value={progressPercent} className="h-1" />
                <div className="grid gap-2 text-xs md:grid-cols-4">
                  <p className="text-muted-foreground">
                    Done: <span className="text-foreground">{selectedExecution.progress?.done ?? "--"}</span>
                  </p>
                  <p className="text-muted-foreground">
                    Total: <span className="text-foreground">{selectedExecution.progress?.total ?? "--"}</span>
                  </p>
                  <p className="text-muted-foreground">
                    Rate: <span className="text-foreground">{selectedExecution.progress?.rate ?? "--"} rec/s</span>
                  </p>
                  <p className="text-muted-foreground">
                    ETA: <span className="text-foreground">{formatEta(selectedExecution.progress?.eta_sec)}</span>
                  </p>
                </div>
                {selectedExecution.current_column && selectedExecution.column_progress && (
                  <p className="text-xs text-muted-foreground">
                    Column {selectedExecution.current_column}:{" "}
                    {selectedExecution.column_progress.done ?? "--"}/
                    {selectedExecution.column_progress.total ?? "--"} (
                    {formatPercent(selectedExecution.column_progress.percent)})
                  </p>
                )}
                {showBatchProgress && (
                  <p className="text-xs text-muted-foreground">
                    Processed batch: {batchIdx ?? "--"}/{batchTotal}
                  </p>
                )}
                {isStale && <Badge variant="outline">Recipe changed since this run</Badge>}
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

            <Tabs value={detailTab} onValueChange={setDetailTab}>
              <div className="flex items-center justify-between gap-2">
                <TabsList className="border border-border/60 bg-card/40">
                  <TabsTrigger value="data">Data</TabsTrigger>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="columns">Columns</TabsTrigger>
                  <TabsTrigger value="raw">Raw</TabsTrigger>
                </TabsList>
                <div className="flex items-center gap-2">
                  {canPublish && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => setPublishDialogOpen(true)}
                    >
                      <HugeiconsIcon icon={Share08Icon} className="mr-2 size-4" />
                      Publish to Hugging Face
                    </Button>
                  )}
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
                  canPublish={canPublish}
                  onOpenPublish={() => setPublishDialogOpen(true)}
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
                  onOpenOverview={() => setDetailTab("overview")}
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
                />
              </TabsContent>
              <TabsContent value="raw">
                <ExecutionRawTab rawExecution={rawExecution} />
              </TabsContent>
            </Tabs>
          </div>
        )}
      </section>
      <PublishExecutionDialog
        open={publishDialogOpen}
        onOpenChange={setPublishDialogOpen}
        execution={canPublish ? selectedExecution : null}
        onPublish={async (payload) => {
          if (!selectedExecution?.jobId) {
            throw new Error("This run is missing a job id.");
          }
          const response = await publishRecipeJob(selectedExecution.jobId, payload);
          return { url: response.url };
        }}
      />
    </div>
  );
}
