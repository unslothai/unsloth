// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  useTrainingQueueActions,
  useTrainingQueueStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingQueueItem } from "@/features/training";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  ArrowUp01Icon,
  CheckmarkCircle01Icon,
  Clock01Icon,
  Delete02Icon,
  PauseIcon,
  PlayCircleIcon,
  PlayListAddIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { useT } from "@/i18n";

function statusBadge(
  item: TrainingQueueItem,
  t: ReturnType<typeof useT>,
): ReactElement {
  if (item.status === "pending") {
    return (
      <Badge variant="outline">
        <HugeiconsIcon icon={Clock01Icon} />
        {t("studio.training.queue.statusPending")}
      </Badge>
    );
  }
  if (item.status === "starting" || item.status === "running") {
    return (
      <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <HugeiconsIcon icon={PlayCircleIcon} />
        {t("studio.training.queue.statusRunning")}
      </Badge>
    );
  }
  if (item.status === "skipped") {
    return (
      <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400">
        <HugeiconsIcon icon={AlertCircleIcon} />
        {t("studio.training.queue.statusSkipped")}
      </Badge>
    );
  }
  if (item.result_status === "completed") {
    return (
      <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <HugeiconsIcon icon={CheckmarkCircle01Icon} />
        {t("studio.training.queue.statusCompleted")}
      </Badge>
    );
  }
  if (item.result_status === "error") {
    return (
      <Badge variant="destructive">
        <HugeiconsIcon icon={AlertCircleIcon} />
        {t("studio.training.queue.statusFailed")}
      </Badge>
    );
  }
  return <Badge variant="secondary">{t("studio.training.queue.statusStopped")}</Badge>;
}

function QueueItemRow({
  item,
  isFirstPending,
  canMoveUp,
  canMoveDown,
}: {
  item: TrainingQueueItem;
  isFirstPending: boolean;
  canMoveUp: boolean;
  canMoveDown: boolean;
}): ReactElement {
  const t = useT();
  const { removeItem, moveItem } = useTrainingQueueActions();
  const isPending = item.status === "pending";

  return (
    <li className="flex items-start gap-2 rounded-lg border bg-card px-3 py-2.5">
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-foreground">
            {item.model_name}
          </span>
          {statusBadge(item, t)}
        </div>
        <span className="truncate text-xs text-muted-foreground">
          {item.dataset_summary}
        </span>
        {isFirstPending && (
          <span className="text-xs text-emerald-600 dark:text-emerald-400">
            {t("studio.training.queue.runsNext")}
          </span>
        )}
        {item.error_message && item.status !== "pending" && (
          <span className="text-xs leading-relaxed text-red-500">
            {item.error_message}
          </span>
        )}
      </div>
      {isPending && (
        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon-sm"
            disabled={!canMoveUp}
            onClick={() => void moveItem(item.id, "up")}
            aria-label={t("studio.training.queue.moveUp")}
          >
            <HugeiconsIcon icon={ArrowUp01Icon} className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            disabled={!canMoveDown}
            onClick={() => void moveItem(item.id, "down")}
            aria-label={t("studio.training.queue.moveDown")}
          >
            <HugeiconsIcon icon={ArrowDown01Icon} className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            className="text-muted-foreground hover:text-red-500"
            onClick={() => void removeItem(item.id)}
            aria-label={t("studio.training.queue.remove")}
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-4" />
          </Button>
        </div>
      )}
    </li>
  );
}

export function TrainingQueuePanel(): ReactElement {
  const t = useT();
  const items = useTrainingQueueStore((s) => s.items);
  const paused = useTrainingQueueStore((s) => s.paused);
  const pendingCount = useTrainingQueueStore((s) => s.pendingCount);
  const maxPending = useTrainingQueueStore((s) => s.maxPending);
  const activeJobId = useTrainingQueueStore((s) => s.activeJobId);
  const isTrainingRunning = useTrainingRuntimeStore((s) => s.isTrainingRunning);
  const runtimeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const runtimeModelName = useTrainingRuntimeStore((s) => s.startModelName);
  const runtimeDatasetName = useTrainingRuntimeStore((s) => s.startDatasetName);
  const { pause, resume } = useTrainingQueueActions();

  const active = items.filter(
    (i) => i.status === "pending" || i.status === "starting" || i.status === "running",
  );
  const finished = items.filter((i) => i.status === "done" || i.status === "skipped");
  const pendingIds = active.filter((i) => i.status === "pending").map((i) => i.id);

  // Direct starts never enter the queue; show them here anyway.
  const hasRowForActiveRun = items.some(
    (i) =>
      (i.status === "starting" || i.status === "running") &&
      i.job_id !== null &&
      (i.job_id === activeJobId || i.job_id === runtimeJobId),
  );
  const showDirectRun = isTrainingRunning && !hasRowForActiveRun;

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5">
          <HugeiconsIcon icon={PlayListAddIcon} className="size-4" />
          {t("studio.training.queue.buttonLabel")}
          {pendingCount > 0 && (
            <Badge variant="secondary" className="px-1.5">
              {pendingCount}
            </Badge>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="flex flex-col gap-0 overflow-y-auto">
        <SheetHeader className="pb-3">
          <SheetTitle>{t("studio.training.queue.title")}</SheetTitle>
          <SheetDescription>{t("studio.training.queue.description")}</SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-4 px-6 pb-6">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-muted-foreground">
              {t("studio.training.queue.pendingCount", {
                count: String(pendingCount),
              })}
            </span>
            {paused ? (
              <Button variant="outline" size="sm" onClick={() => void resume()}>
                <HugeiconsIcon icon={PlayCircleIcon} className="size-4" />
                {t("studio.training.queue.resume")}
              </Button>
            ) : (
              <Button variant="outline" size="sm" onClick={() => void pause()}>
                <HugeiconsIcon icon={PauseIcon} className="size-4" />
                {t("studio.training.queue.pause")}
              </Button>
            )}
          </div>

          {paused && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-700 dark:text-amber-400">
              {t("studio.training.queue.pausedNote")}
            </p>
          )}

          {pendingCount >= maxPending && (
            <p className="text-xs text-muted-foreground">
              {t("studio.training.queue.queueFullNote")}
            </p>
          )}

          {showDirectRun && (
            <>
              <p className="text-xs font-medium text-muted-foreground">
                {t("studio.training.queue.runningNow")}
              </p>
              <ul className="flex flex-col gap-2">
                <li className="flex items-start gap-2 rounded-lg border bg-card px-3 py-2.5">
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-foreground">
                        {runtimeModelName ?? t("studio.training.queue.statusRunning")}
                      </span>
                      <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                        <HugeiconsIcon icon={PlayCircleIcon} />
                        {t("studio.training.queue.statusRunning")}
                      </Badge>
                    </div>
                    {runtimeDatasetName && (
                      <span className="truncate text-xs text-muted-foreground">
                        {runtimeDatasetName}
                      </span>
                    )}
                  </div>
                </li>
              </ul>
            </>
          )}

          {active.length === 0 ? (
            !showDirectRun && (
              <p className="rounded-lg border border-dashed px-3 py-6 text-center text-sm text-muted-foreground">
                {t("studio.training.queue.empty")}
              </p>
            )
          ) : (
            <ul className="flex flex-col gap-2">
              {active.map((item) => (
                <QueueItemRow
                  key={item.id}
                  item={item}
                  isFirstPending={!paused && item.id === pendingIds[0]}
                  canMoveUp={pendingIds.indexOf(item.id) > 0}
                  canMoveDown={
                    pendingIds.indexOf(item.id) >= 0 &&
                    pendingIds.indexOf(item.id) < pendingIds.length - 1
                  }
                />
              ))}
            </ul>
          )}

          <p className="text-xs leading-relaxed text-muted-foreground/70">
            {t("studio.training.queue.configureHint")}
          </p>

          {finished.length > 0 && (
            <>
              <p className="pt-2 text-xs font-medium text-muted-foreground">
                {t("studio.training.queue.recentlyFinished")}
              </p>
              <ul className="flex flex-col gap-2">
                {finished.map((item) => (
                  <QueueItemRow
                    key={item.id}
                    item={item}
                    isFirstPending={false}
                    canMoveUp={false}
                    canMoveDown={false}
                  />
                ))}
              </ul>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
