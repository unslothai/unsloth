// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  CheckmarkCircle02Icon,
  Flag02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import type { RecipeExecutionRecord } from "../../execution-types";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import {
  formatMetricValue,
  formatPercent,
} from "../executions/executions-view-helpers";

type ExecutionProgressIslandProps = {
  execution: RecipeExecutionRecord;
  currentColumnIcon: typeof Flag02Icon;
  minimized: boolean;
  onMinimizedChange: (value: boolean) => void;
  onViewExecutions: () => void;
};

function formatEta(value: number | null | undefined): string {
  const metric = formatMetricValue(value);
  if (metric === "--") {
    return "--";
  }
  return `${metric}s`;
}

function statusLabel(input: {
  complete: boolean;
  inProgress: boolean;
}): string {
  if (input.complete) {
    return "Run completed";
  }
  if (input.inProgress) {
    return "Run in progress";
  }
  return "Run status";
}

function formatSourceResource(value: string | null | undefined): string {
  if (value === "pulls") {
    return "PRs";
  }
  return value ?? "source";
}

function formatGitHubSourceSummary(
  execution: RecipeExecutionRecord,
): string | null {
  const source = execution.source_progress;
  if (!source || source.source !== "github") {
    return null;
  }
  if (source.status === "rate_limited") {
    const wait =
      typeof source.retry_after_sec === "number" && source.retry_after_sec > 0
        ? ` ~${formatMetricValue(source.retry_after_sec)}s`
        : "";
    return `Waiting for GitHub rate limit${wait}. Studio will resume automatically.`;
  }
  if (source.status === "retrying") {
    return source.message ?? "GitHub request failed; retrying automatically.";
  }

  const parts = [
    source.repo,
    source.resource
      ? `${formatSourceResource(source.resource)}${
          typeof source.page === "number" ? ` page ${source.page}` : ""
        }`
      : null,
    typeof source.fetched_items === "number"
      ? `${formatMetricValue(source.fetched_items)} fetched`
      : null,
    typeof source.rate_remaining === "number"
      ? `remaining ${formatMetricValue(source.rate_remaining)}`
      : null,
  ].filter(Boolean);

  if (parts.length === 0) {
    return source.message ?? "Crawling GitHub source";
  }
  return parts.join(" · ");
}

export function ExecutionProgressIsland({
  execution,
  currentColumnIcon,
  minimized,
  onMinimizedChange,
  onViewExecutions,
}: ExecutionProgressIslandProps): ReactElement {
  const complete = execution.status === "completed";
  const inProgress = isExecutionInProgress(execution.status);
  const sourceSummary = formatGitHubSourceSummary(execution);
  const showSourceProgress = Boolean(
    sourceSummary &&
      inProgress &&
      (execution.stage === "source" ||
        execution.source_progress?.status === "rate_limited" ||
        execution.source_progress?.status === "retrying"),
  );
  const sourcePercent = showSourceProgress ? execution.source_progress?.percent : null;
  const progressPercent = sourcePercent ?? execution.progress?.percent ?? (complete ? 100 : 0);
  const hasProgressSignal = Boolean(
    (execution.progress &&
      (typeof execution.progress.done === "number" ||
        typeof execution.progress.total === "number" ||
        typeof execution.progress.percent === "number" ||
        typeof execution.progress.rate === "number" ||
        typeof execution.progress.eta_sec === "number")) ||
      typeof sourcePercent === "number",
  );
  const showLoadingSpinner = inProgress && !hasProgressSignal;
  const batchTotal = execution.batch?.total ?? null;
  const showBatch = typeof batchTotal === "number" && batchTotal > 1;

  return (
    <div
      className={cn(
        "w-[clamp(15rem,26vw,20rem)] max-w-[calc(100vw-1rem)] rounded-b-xl border-x border-b bg-card/96 shadow-sm backdrop-blur-sm transition-all",
        minimized ? "min-h-[3rem]" : "min-h-[8.5rem]",
      )}
      aria-live="polite"
    >
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <HugeiconsIcon
            icon={complete ? CheckmarkCircle02Icon : Flag02Icon}
            className={cn(
              "size-3.5",
              complete
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-amber-700 dark:text-amber-300",
            )}
          />
          <p className="truncate text-xs font-medium text-foreground">
            {statusLabel({ complete, inProgress })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {showLoadingSpinner && (
            <Spinner className="size-3.5 text-muted-foreground" />
          )}
          <span className="shrink-0 text-[11px] text-muted-foreground">
            {formatPercent(progressPercent)}
          </span>
          <button
            type="button"
            onClick={() => onMinimizedChange(!minimized)}
            className="inline-flex size-8 shrink-0 items-center justify-center rounded border border-border/70 text-muted-foreground transition hover:bg-muted/50"
            aria-label={minimized ? "Expand progress" : "Minimize progress"}
            title={minimized ? "Expand" : "Minimize"}
          >
            <HugeiconsIcon
              icon={minimized ? ArrowDown01Icon : ArrowUp01Icon}
              className="size-3.5"
            />
          </button>
        </div>
      </div>

      <div className="px-3">
        <Progress value={progressPercent} className="h-1" />
      </div>

      {!minimized && (
        <>
          <div className="grid grid-cols-2 gap-2 px-3 pt-2 text-[11px] text-muted-foreground sm:grid-cols-4">
            <p className="truncate" title={`Done: ${formatMetricValue(execution.progress?.done)}`}>
              Done: {formatMetricValue(execution.progress?.done)}
            </p>
            <p className="truncate" title={`Total: ${formatMetricValue(execution.progress?.total)}`}>
              Total: {formatMetricValue(execution.progress?.total)}
            </p>
            <p className="truncate" title={`Rate: ${formatMetricValue(execution.progress?.rate)}`}>
              Rate: {formatMetricValue(execution.progress?.rate)}
            </p>
            <p className="truncate" title={`ETA: ${formatEta(execution.progress?.eta_sec)}`}>
              ETA: {formatEta(execution.progress?.eta_sec)}
            </p>
          </div>
          {showSourceProgress ? (
            <div className="mt-1 flex items-center gap-1.5 px-3 text-[11px] text-muted-foreground">
              <HugeiconsIcon
                icon={Flag02Icon}
                className="size-3.5 shrink-0 text-amber-700 dark:text-amber-300"
              />
              <p className="truncate" title={sourceSummary ?? undefined}>
                GitHub source: {sourceSummary}
              </p>
            </div>
          ) : (
            <div className="mt-1 flex items-center gap-1.5 px-3 text-[11px] text-muted-foreground">
              <HugeiconsIcon
                icon={currentColumnIcon}
                className="size-3.5 shrink-0"
              />
              <p
                className="truncate"
                title={execution.current_column ?? "--"}
              >
                Column: {execution.current_column ?? "--"}
              </p>
            </div>
          )}
          {showBatch && (
            <div
              className="mt-1 truncate px-3 text-[11px] text-muted-foreground"
              title={`Batch: ${execution.batch?.idx ?? "--"}/${execution.batch?.total ?? "--"}`}
            >
              Batch: {execution.batch?.idx ?? "--"}/{execution.batch?.total ?? "--"}
            </div>
          )}
          <div className="px-3 pb-2 pt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 w-full text-[11px]"
              onClick={onViewExecutions}
            >
              View run details
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
