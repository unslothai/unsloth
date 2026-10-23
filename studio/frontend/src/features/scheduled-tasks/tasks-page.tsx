// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Add01Icon,
  Delete02Icon,
  FilterMailIcon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  Search01Icon,
  Clock01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { PauseIcon, PlayIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import {
  type InterpolationValues,
  type TranslationKey,
  formatRelativeTime,
  useLocale,
  useT,
} from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { ScheduledTask } from "./api";
import { runErrorText } from "./results";
import {
  type TaskSchedule,
  formatClock,
  formatRunDay,
  relativeTimeParts,
  summarizeRepeats,
  summarizeSchedule,
  waitForDue,
} from "./schedule";
import { SCHEDULED_TASKS_WRITTEN_KEY, useScheduledTasksStore } from "./store";
import { TaskDialog } from "./task-dialog";
import { useUnseenScheduledRuns } from "./unseen";

const OVERDUE_REFRESH_MS = 5_000;
const OVERDUE_REFRESH_MAX_MS = 60_000;
const REFOCUS_DEDUPE_MS = 1_000;

const ROW_BUTTON =
  "flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50 data-[state=open]:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

type TaskStatus = ScheduledTask["status"];

const STATUSES: TaskStatus[] = ["active", "paused", "completed"];
const STATUS_LABEL = {
  active: "scheduledTasks.active",
  paused: "scheduledTasks.paused",
  completed: "scheduledTasks.completed",
} as const satisfies Record<TaskStatus, TranslationKey>;
const FILTER_KEY = "unsloth_scheduled_tasks_filter";

function loadFilter(): TaskStatus[] {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(FILTER_KEY) ?? "null");
    if (Array.isArray(saved)) return STATUSES.filter((status) => saved.includes(status));
  } catch {
    // Unreadable storage shows every task.
  }
  return STATUSES;
}

function saveFilter(shown: TaskStatus[]): void {
  try {
    localStorage.setItem(FILTER_KEY, JSON.stringify(shown));
  } catch {
    // Without storage the filter resets on the next visit.
  }
}

function compareTasks(a: ScheduledTask, b: ScheduledTask): number {
  const rank = STATUSES.indexOf(a.status) - STATUSES.indexOf(b.status);
  if (rank !== 0) return rank;
  if (a.nextRunAt !== null && b.nextRunAt !== null && a.nextRunAt !== b.nextRunAt) {
    return a.nextRunAt - b.nextRunAt;
  }
  return b.updatedAt - a.updatedAt;
}

function scheduleText(
  schedule: TaskSchedule,
  locale: string,
  t: (key: TranslationKey, values?: InterpolationValues) => string,
): string {
  const summary = summarizeSchedule(schedule, locale);
  return t(`scheduledTasks.schedule.${summary.key}`, summary.values);
}

function repeatsText(
  schedule: TaskSchedule,
  locale: string,
  t: (key: TranslationKey, values?: InterpolationValues) => string,
): string {
  const summary = summarizeRepeats(schedule, locale);
  if (summary.key === "weekly") return summary.days;
  if (summary.key === "monthly") return t("scheduledTasks.monthlyDay", { day: summary.day });
  return t(`scheduledTasks.form.${summary.key}`);
}

function clockText(schedule: TaskSchedule, locale: string): string {
  return formatClock(schedule.kind === "once" ? schedule.at.split("T")[1] : schedule.time, locale);
}

const REPEATS_COLUMN = "hidden w-36 shrink-0 truncate md:block";
const TIME_COLUMN = "hidden w-24 shrink-0 truncate md:block";
const NEXT_RUN_COLUMN = "hidden w-32 shrink-0 truncate md:block";

function errorText(error: unknown): string | undefined {
  return error instanceof Error ? error.message : undefined;
}

function useMinuteClock(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function TaskRow({
  task,
  now,
  unseen,
  onEdit,
  onDelete,
}: {
  task: ScheduledTask;
  now: number;
  unseen: boolean;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const update = useScheduledTasksStore((state) => state.update);
  const runNow = useScheduledTasksStore((state) => state.runNow);
  const clearUnseen = useUnseenScheduledRuns((state) => state.clearTask);
  const [busy, setBusy] = useState(false);
  const run = task.lastRun;
  const running = run?.status === "queued" || run?.status === "running";

  async function act(action: () => Promise<unknown>, failure: "runFailed" | "updateFailed") {
    setBusy(true);
    try {
      await action();
    } catch (error) {
      toast.error(t(`scheduledTasks.errors.${failure}`), { description: errorText(error) });
    } finally {
      setBusy(false);
    }
  }

  // Always past tense: a just-finished run (or a server clock slightly ahead) would read "in 0 sec.".
  const ago = (at: number) =>
    formatRelativeTime(locale, ...relativeTimeParts(Math.min(at, now - 1_000), now));
  const finishedAt = run?.finishedAt ?? null;
  const repeats = repeatsText(task.schedule, locale, t);
  const clock = clockText(task.schedule, locale);
  const nextDay = task.nextRunAt === null ? "—" : formatRunDay(task.nextRunAt, now, locale, task.timezone);

  return (
    <div className="group/task-row flex items-center gap-4 rounded-xl px-5 py-4 transition-colors duration-150 hover:bg-muted/70 dark:hover:bg-[rgb(255_255_255_/_calc(0.055*var(--contrast-wash-gain,1)))]">
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex min-w-0 items-center gap-2">
          <button
            type="button"
            onClick={() => {
              clearUnseen(task.id);
              onEdit();
            }}
            className="min-w-0 cursor-pointer truncate rounded-md text-left text-ui-15 font-semibold text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {task.title}
          </button>
          {unseen && (
            <span
              role="img"
              aria-label={t("scheduledTasks.newResult")}
              className="size-2 shrink-0 rounded-full bg-primary"
            />
          )}
          {task.status !== "active" && (
            <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-ui-11 font-medium text-muted-foreground">
              {task.status === "paused" ? t("scheduledTasks.paused") : t("scheduledTasks.completed")}
            </span>
          )}
        </div>
        <p className="truncate text-ui-13 text-muted-foreground md:hidden">
          {[repeats, clock, task.nextRunAt === null ? null : nextDay]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {running ? (
          <p className="flex items-center gap-1.5 text-ui-13 text-foreground">
            <Spinner className="size-3.5" label={t("scheduledTasks.running")} />
            {t("scheduledTasks.running")}
          </p>
        ) : run?.status === "succeeded" && finishedAt !== null ? (
          <p className="text-ui-13 text-muted-foreground">
            {t("scheduledTasks.lastRun", { when: ago(finishedAt) })}
            {run.threadId && (
              <>
                {" · "}
                <button
                  type="button"
                  onClick={() => {
                    clearUnseen(task.id);
                    void navigate({ to: "/chat", search: { thread: run.threadId ?? undefined } });
                  }}
                  className="cursor-pointer font-medium text-foreground underline-offset-2 hover:underline"
                >
                  {t("scheduledTasks.openChat")}
                </button>
              </>
            )}
          </p>
        ) : run?.status === "failed" && finishedAt !== null ? (
          <p title={runErrorText(run, t)} className="truncate text-ui-13 text-destructive">
            {t("scheduledTasks.lastRunFailed", {
              when: ago(finishedAt),
              reason: runErrorText(run, t),
            })}
          </p>
        ) : null}
      </div>
      <span title={scheduleText(task.schedule, locale, t)} className={cn(REPEATS_COLUMN, "text-ui-15 text-foreground/85")}>
        {repeats}
      </span>
      <span title={task.timezone} className={cn(TIME_COLUMN, "text-ui-15 text-foreground/85")}>
        {clock}
      </span>
      <span className={cn(NEXT_RUN_COLUMN, "text-ui-15 text-muted-foreground")}>{nextDay}</span>
      <div className="flex shrink-0 items-center gap-1">
        <button
          type="button"
          aria-label={t("scheduledTasks.runNow")}
          title={t("scheduledTasks.runNow")}
          disabled={busy || running}
          onClick={() => {
            clearUnseen(task.id);
            void act(() => runNow(task.id), "runFailed");
          }}
          className={cn(
            ROW_BUTTON,
            "opacity-0 group-hover/task-row:opacity-100 group-focus-within/task-row:opacity-100 [@media(pointer:coarse)]:opacity-100",
          )}
        >
          <PlayIcon strokeWidth={1.75} className="size-4.5" />
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button type="button" aria-label={t("scheduledTasks.options")} className={ROW_BUTTON}>
              <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-48 font-heading rounded-[14px] border-0"
          >
            <DropdownMenuItem onSelect={onEdit}>
              <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-icon" />
              {t("scheduledTasks.edit")}
            </DropdownMenuItem>
            {task.status !== "completed" && (
              <DropdownMenuItem
                disabled={busy}
                onSelect={() =>
                  void act(() => update(task.id, { enabled: !task.enabled }), "updateFailed")
                }
              >
                {task.enabled ? (
                  <PauseIcon strokeWidth={1.75} className="size-icon" />
                ) : (
                  <PlayIcon strokeWidth={1.75} className="size-icon" />
                )}
                {task.enabled ? t("scheduledTasks.pause") : t("scheduledTasks.resume")}
              </DropdownMenuItem>
            )}
            <DropdownMenuItem variant="destructive" onSelect={onDelete}>
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
              {t("common.delete")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

export function TasksPage() {
  const t = useT();
  const { tasks, status, error, refresh, remove } = useScheduledTasksStore(
    useShallow((state) => ({
      tasks: state.tasks,
      status: state.status,
      error: state.error,
      refresh: state.refresh,
      remove: state.remove,
    })),
  );
  const serverOffset = useScheduledTasksStore((state) => state.serverOffset);
  const now = useMinuteClock() + serverOffset;
  const [editing, setEditing] = useState<ScheduledTask | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<ScheduledTask | null>(null);
  const [shown, setShown] = useState<TaskStatus[]>(loadFilter);
  const [query, setQuery] = useState("");
  const visibleTasks = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return tasks
      .filter(
        (task) =>
          shown.includes(task.status) &&
          (needle === "" ||
            task.title.toLocaleLowerCase().includes(needle) ||
            task.prompt.toLocaleLowerCase().includes(needle)),
      )
      .sort(compareTasks);
  }, [tasks, shown, query]);
  const unseenTasks = useUnseenScheduledRuns((state) => state.tasks);
  const pageUnseen = useUnseenScheduledRuns((state) => state.page);
  const clearPageUnseen = useUnseenScheduledRuns((state) => state.clearPage);

  useEffect(() => {
    if (pageUnseen) clearPageUnseen();
  }, [pageUnseen, clearPageUnseen]);

  // ?edit=<id>: open that task once listed, then drop the param so close/reload does not reopen it.
  const navigate = useNavigate();
  const { edit: editId, create } = useSearch({ strict: false });
  useEffect(() => {
    if (!create) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the URL asks for the dialog once, then is cleared.
    setEditing(null);
    void navigate({ to: "/tasks", search: {}, replace: true });
  }, [create, navigate]);
  useEffect(() => {
    if (!editId || status !== "ready") return;
    const target = tasks.find((task) => task.id === editId);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- as above, once the list has the task.
    if (target) setEditing(target);
    else toast.info(t("scheduledTasks.taskMissing"));
    void navigate({ to: "/tasks", search: {}, replace: true });
  }, [editId, status, tasks, navigate, t]);

  function toggleStatus(status: TaskStatus, on: boolean) {
    const next = STATUSES.filter((candidate) => (candidate === status ? on : shown.includes(candidate)));
    setShown(next);
    saveFilter(next);
  }

  useEffect(() => {
    void refresh();
    // Other tabs/devices send no event: re-read on storage marks, show and focus.
    const onStorage = (event: StorageEvent) => {
      if (event.key === SCHEDULED_TASKS_WRITTEN_KEY) void refresh();
    };
    let lookedAt = 0;
    const lookAgain = () => {
      const now = Date.now();
      if (now - lookedAt < REFOCUS_DEDUPE_MS) return;
      lookedAt = now;
      void refresh();
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") lookAgain();
    };
    const onFocus = lookAgain;
    window.addEventListener("storage", onStorage);
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const soonest = useMemo(
    () =>
      tasks.reduce<number | null>(
        (earliest, task) =>
          task.nextRunAt !== null && (earliest === null || task.nextRunAt < earliest)
            ? task.nextRunAt
            : earliest,
        null,
      ),
    [tasks],
  );
  useEffect(() => {
    if (soonest === null) return;
    let timer: number | undefined;
    let again = OVERDUE_REFRESH_MS;
    const look = () => {
      if (document.visibilityState !== "hidden") void refresh();
      timer = window.setTimeout(look, again);
      again = Math.min(again * 2, OVERDUE_REFRESH_MAX_MS);
    };
    const wait = () => {
      const serverNow = Date.now() + useScheduledTasksStore.getState().serverOffset;
      const next = waitForDue(soonest, serverNow);
      timer = window.setTimeout(next.due ? look : wait, next.delay);
    };
    wait();
    return () => window.clearTimeout(timer);
  }, [soonest, refresh]);

  async function confirmDelete() {
    const target = deleting;
    if (!target) return;
    setDeleting(null);
    try {
      await remove(target.id);
    } catch (error) {
      toast.error(t("scheduledTasks.errors.deleteFailed"), { description: errorText(error) });
    }
  }

  const loading = status === "idle" || status === "loading";

  return (
    <main className="mx-auto w-full max-w-5xl 4xl:max-w-6xl px-6 pb-10 pt-8 max-sm:px-4 font-heading sm:px-10">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          {t("scheduledTasks.title")}
        </h1>
        <div className="flex w-full min-w-0 items-center justify-end gap-3 sm:w-auto sm:flex-1">
          <div className="relative min-w-0 flex-1 sm:max-w-sm">
            <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
              <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4" />
            </span>
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t("scheduledTasks.search")}
              aria-label={t("scheduledTasks.search")}
              className="h-9 w-full rounded-full border-none bg-muted pl-10 pr-4 shadow-none dark:bg-card"
            />
          </div>
          <Button onClick={() => setEditing(null)}>
            <HugeiconsIcon icon={Add01Icon} strokeWidth={1.75} className="size-icon" />
            {t("scheduledTasks.newTask")}
          </Button>
        </div>
      </div>
      <p className="mt-4 text-ui-15 text-muted-foreground">{t("scheduledTasks.description")}</p>

      {loading && tasks.length === 0 ? (
        <div className="mt-12">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={index} className="flex items-start gap-4 rounded-xl px-5 py-4">
              <div className="flex flex-1 flex-col gap-2">
                <Skeleton className="h-4 w-48 rounded-[8px]" />
                <Skeleton className="h-3.5 w-72 rounded-[8px]" />
              </div>
            </div>
          ))}
        </div>
      ) : status === "error" && tasks.length === 0 ? (
        <div className="mt-16 flex flex-col items-center gap-3 text-center">
          <p className="text-sm text-muted-foreground">{t("scheduledTasks.loadFailed")}</p>
          {error && <p className="max-w-md text-ui-13 text-muted-foreground">{error}</p>}
          <Button variant="outline" onClick={() => void refresh()}>
            {t("scheduledTasks.retry")}
          </Button>
        </div>
      ) : tasks.length === 0 ? (
        <div className="mt-16 flex flex-col items-center gap-2 text-center">
          <span className="mb-2 flex size-12 items-center justify-center rounded-2xl bg-muted text-foreground/70">
            <HugeiconsIcon icon={Clock01Icon} strokeWidth={1.75} className="size-6" />
          </span>
          <p className="text-ui-15 font-semibold text-foreground">{t("scheduledTasks.emptyTitle")}</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            {t("scheduledTasks.emptyDescription")}
          </p>
          <Button
            variant="outline"
            className="mt-3 border-none bg-background shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
            onClick={() => setEditing(null)}
          >
            <HugeiconsIcon icon={Add01Icon} strokeWidth={1.75} className="size-icon" />
            {t("scheduledTasks.createFirst")}
          </Button>
        </div>
      ) : (
        <div className="mt-10 flex flex-col">
          <div className="flex items-center gap-4 px-5 pb-2 text-ui-14 font-medium text-muted-foreground">
            <h2 className="min-w-0 flex-1">{t("scheduledTasks.upcoming")}</h2>
            <span className={REPEATS_COLUMN}>{t("scheduledTasks.columns.repeats")}</span>
            <span className={TIME_COLUMN}>{t("scheduledTasks.columns.time")}</span>
            <span className={NEXT_RUN_COLUMN}>{t("scheduledTasks.columns.nextRun")}</span>
            <span className="flex shrink-0 items-center gap-1">
              <span className="size-9" />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    aria-label={t("scheduledTasks.filter")}
                    title={t("scheduledTasks.filter")}
                    className={cn(ROW_BUTTON, shown.length < STATUSES.length && "text-foreground")}
                  >
                    <HugeiconsIcon icon={FilterMailIcon} strokeWidth={1.75} className="size-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  className="app-user-menu menu-soft-surface ring-0 w-52 font-heading rounded-[14px] border-0"
                >
                  <DropdownMenuLabel className="text-muted-foreground">
                    {t("scheduledTasks.filterBy")}
                  </DropdownMenuLabel>
                  {STATUSES.map((status) => (
                    <DropdownMenuCheckboxItem
                      key={status}
                      checked={shown.includes(status)}
                      onCheckedChange={(on) => toggleStatus(status, on)}
                      onSelect={(event) => event.preventDefault()}
                    >
                      {t(STATUS_LABEL[status])}
                    </DropdownMenuCheckboxItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </span>
          </div>
          {visibleTasks.length === 0 ? (
            <p className="px-5 py-4 text-ui-13 text-muted-foreground">
              {t("scheduledTasks.noMatches")}
            </p>
          ) : (
            visibleTasks.map((task) => (
              <TaskRow
                key={task.id}
                task={task}
                now={now}
                unseen={task.id in unseenTasks}
                onEdit={() => setEditing(task)}
                onDelete={() => setDeleting(task)}
              />
            ))
          )}
        </div>
      )}

      {editing !== undefined && (
        <TaskDialog
          key={editing?.id ?? "new"}
          task={editing}
          onClose={() => setEditing(undefined)}
          onDelete={(task) => setDeleting(task)}
        />
      )}
      <AlertDialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent onOverlayClick={() => setDeleting(null)}>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("scheduledTasks.deleteDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("scheduledTasks.deleteDialog.description", { title: deleting?.title ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => void confirmDelete()}>
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
