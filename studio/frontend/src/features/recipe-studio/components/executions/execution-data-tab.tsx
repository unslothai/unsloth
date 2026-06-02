// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { GithubIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ColumnDef } from "@tanstack/react-table";
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
import { Spinner } from "@/components/ui/spinner";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import type { RecipeExecutionRecord } from "../../execution-types";
import { formatMetricValue } from "./executions-view-helpers";

function formatSourceResource(value: string | null | undefined): string {
  if (value === "pulls") {
    return "PRs";
  }
  return value ?? "--";
}

function formatFetchedValue(execution: RecipeExecutionRecord): string {
  const source = execution.source_progress;
  if (!source) {
    return "--";
  }
  const fetched = formatMetricValue(source.fetched_items);
  if (typeof source.estimated_total !== "number" || source.estimated_total <= 0) {
    return fetched;
  }
  return `${fetched} / ${formatMetricValue(source.estimated_total)}`;
}

function formatGitHubSourceMessage(execution: RecipeExecutionRecord): string {
  const source = execution.source_progress;
  if (!source) {
    return "Collecting repository threads before rows are available.";
  }
  if (source.status === "rate_limited") {
    return source.message ?? "Waiting for GitHub rate limit. Studio will resume automatically.";
  }
  return source.message ?? "Collecting repository threads before rows are available.";
}

function RunningDatasetEmptyState({
  execution,
  onOpenOverview,
}: {
  execution: RecipeExecutionRecord;
  onOpenOverview: () => void;
}): ReactElement {
  const source = execution.source_progress;
  if (source?.source === "github") {
    const title =
      source.status === "rate_limited"
        ? "Waiting for GitHub rate limit"
        : "Crawling GitHub source";
    const showProgress = typeof source.percent === "number";

    return (
      <div className="rounded-xl border border-border/60 bg-card/55 p-4 shadow-border">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 gap-3">
            <div
              className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-muted/20"
            >
              {showProgress ? (
                <HugeiconsIcon icon={GithubIcon} className="size-4 text-muted-foreground" />
              ) : (
                <Spinner className="size-4 text-muted-foreground" />
              )}
            </div>
            <div className="min-w-0 space-y-1">
              <p className="text-sm font-semibold text-foreground">{title}</p>
              <p className="text-xs text-muted-foreground">
                {formatGitHubSourceMessage(execution)}
              </p>
            </div>
          </div>
          <Button type="button" size="sm" variant="outline" onClick={onOpenOverview}>
            Open Overview
          </Button>
        </div>
        {showProgress && <Progress value={source.percent ?? 0} className="mt-3 h-1" />}
        <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-5">
          <p className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Repo</span>
            <span className="truncate font-semibold">{source.repo ?? "--"}</span>
          </p>
          <p className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Resource</span>
            <span className="font-semibold">
              {formatSourceResource(source.resource)}
              {typeof source.page === "number" ? ` page ${source.page}` : ""}
            </span>
          </p>
          <p className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Fetched</span>
            <span className="font-semibold">{formatFetchedValue(execution)}</span>
          </p>
          <p className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Rate remaining</span>
            <span className="font-semibold">{formatMetricValue(source.rate_remaining)}</span>
          </p>
          <p className="flex items-center justify-between gap-3">
            <span className="text-muted-foreground">Retry wait</span>
            <span className="font-semibold">
              {typeof source.retry_after_sec === "number"
                ? `${formatMetricValue(source.retry_after_sec)}s`
                : "--"}
            </span>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border/60 bg-card/55 p-4 text-center shadow-border">
      <div className="flex flex-col items-center justify-center gap-3 py-6">
        <Spinner className="size-5" />
        <div className="space-y-1">
          <p className="text-sm font-medium text-muted-foreground">
            Generating data…
          </p>
          <p className="text-xs text-muted-foreground">
            {execution.current_column
              ? `Current column: ${execution.current_column}`
              : "Rows will appear here once the run produces a dataset sample."}
          </p>
        </div>
        <Button type="button" size="sm" variant="outline" onClick={onOpenOverview}>
          Open Overview
        </Button>
      </div>
    </div>
  );
}

type ExecutionDataTabProps = {
  execution: RecipeExecutionRecord;
  datasetColumnNames: string[];
  hiddenDatasetColumns: string[];
  canPageDataset: boolean;
  currentDatasetPage: number;
  totalPages: number;
  tableColumns: ColumnDef<Record<string, unknown>>[];
  datasetRowsForTable: Record<string, unknown>[];
  onOpenOverview: () => void;
  onSetHiddenColumns: (updater: (current: string[]) => string[]) => void;
  onPrevPage: () => void;
  onNextPage: () => void;
};

export function ExecutionDataTab({
  execution,
  datasetColumnNames,
  hiddenDatasetColumns,
  canPageDataset,
  currentDatasetPage,
  totalPages,
  tableColumns,
  datasetRowsForTable,
  onOpenOverview,
  onSetHiddenColumns,
  onPrevPage,
  onNextPage,
}: ExecutionDataTabProps): ReactElement {
  return (
    <div className="mt-3">
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
                      onSetHiddenColumns((currentColumns) => {
                        if (checked) {
                          return currentColumns.filter((name) => name !== columnName);
                        }
                        return [...currentColumns, columnName];
                      });
                    }}
                  >
                    {columnName}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          {canPageDataset && (
            <>
              <span>
                Page {currentDatasetPage}/{totalPages}
              </span>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={
                  isExecutionInProgress(execution.status) || currentDatasetPage <= 1
                }
                onClick={onPrevPage}
              >
                Prev
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={
                  isExecutionInProgress(execution.status) ||
                  currentDatasetPage >= totalPages
                }
                onClick={onNextPage}
              >
                Next
              </Button>
            </>
          )}
        </div>
      </div>
      {execution.dataset.length === 0 ? (
        isExecutionInProgress(execution.status) ? (
          <RunningDatasetEmptyState
            execution={execution}
            onOpenOverview={onOpenOverview}
          />
        ) : (
          <p className="text-xs text-muted-foreground">No rows returned.</p>
        )
      ) : tableColumns.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          All columns hidden. Use Columns to show at least one.
        </p>
      ) : (
        <div className="max-h-[55vh] overflow-auto">
          <DataTable columns={tableColumns} data={datasetRowsForTable} />
        </div>
      )}
    </div>
  );
}
