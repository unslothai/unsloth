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
import { useI18n } from "@/features/i18n";
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

function statusLabel(
  input: {
  complete: boolean;
  inProgress: boolean;
},
  t: (key: "recipe.progress.completed" | "recipe.progress.inProgress" | "recipe.progress.status") => string,
): string {
  if (input.complete) {
    return t("recipe.progress.completed");
  }
  if (input.inProgress) {
    return t("recipe.progress.inProgress");
  }
  return t("recipe.progress.status");
}

export function ExecutionProgressIsland({
  execution,
  currentColumnIcon,
  minimized,
  onMinimizedChange,
  onViewExecutions,
}: ExecutionProgressIslandProps): ReactElement {
  const { t } = useI18n();
  const complete = execution.status === "completed";
  const inProgress = isExecutionInProgress(execution.status);
  const progressPercent = execution.progress?.percent ?? (complete ? 100 : 0);
  const hasProgressSignal = Boolean(
    execution.progress &&
      (typeof execution.progress.done === "number" ||
        typeof execution.progress.total === "number" ||
        typeof execution.progress.percent === "number" ||
        typeof execution.progress.rate === "number" ||
        typeof execution.progress.eta_sec === "number"),
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
            {statusLabel({ complete, inProgress }, t)}
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
            aria-label={
              minimized ? t("recipe.progress.expandProgress") : t("recipe.progress.minimizeProgress")
            }
            title={minimized ? t("recipe.progress.expand") : t("recipe.progress.minimize")}
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
            <p className="truncate" title={`${t("recipe.progress.done")}: ${formatMetricValue(execution.progress?.done)}`}>
              {t("recipe.progress.done")}: {formatMetricValue(execution.progress?.done)}
            </p>
            <p className="truncate" title={`${t("recipe.progress.total")}: ${formatMetricValue(execution.progress?.total)}`}>
              {t("recipe.progress.total")}: {formatMetricValue(execution.progress?.total)}
            </p>
            <p className="truncate" title={`${t("recipe.progress.rate")}: ${formatMetricValue(execution.progress?.rate)}`}>
              {t("recipe.progress.rate")}: {formatMetricValue(execution.progress?.rate)}
            </p>
            <p className="truncate" title={`${t("recipe.progress.eta")}: ${formatEta(execution.progress?.eta_sec)}`}>
              {t("recipe.progress.eta")}: {formatEta(execution.progress?.eta_sec)}
            </p>
          </div>
          <div className="mt-1 flex items-center gap-1.5 px-3 text-[11px] text-muted-foreground">
            <HugeiconsIcon
              icon={currentColumnIcon}
              className="size-3.5 shrink-0"
            />
            <p
              className="truncate"
              title={execution.current_column ?? "--"}
            >
              {t("recipe.progress.column")}: {execution.current_column ?? "--"}
            </p>
          </div>
          {showBatch && (
            <div
              className="mt-1 truncate px-3 text-[11px] text-muted-foreground"
              title={`${t("recipe.progress.batch")}: ${execution.batch?.idx ?? "--"}/${execution.batch?.total ?? "--"}`}
            >
              {t("recipe.progress.batch")}: {execution.batch?.idx ?? "--"}/{execution.batch?.total ?? "--"}
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
              {t("recipe.progress.viewRunDetails")}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
