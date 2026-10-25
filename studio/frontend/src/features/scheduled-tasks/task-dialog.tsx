// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Folder01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useId, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useChatProjects } from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { ScheduledTask, ScheduledTaskPatch, TaskNotify } from "./api";
import {
  ISO_WEEKDAYS,
  REPEAT_KINDS,
  type RepeatKind,
  type ScheduleForm,
  browserTimeZone,
  defaultScheduleForm,
  formFromSchedule,
  formatClock,
  localDateKey,
  sameSchedule,
  scheduleFormComplete,
  scheduleFromForm,
  timeChoices,
  titleFromPrompt,
  weekdayName,
  zoneLabel,
} from "./schedule";
import { useScheduledTasksStore } from "./store";

const REPEAT_LABELS: Record<RepeatKind, TranslationKey> = {
  once: "scheduledTasks.form.once",
  daily: "scheduledTasks.form.daily",
  weekdays: "scheduledTasks.form.weekdays",
  weekly: "scheduledTasks.form.weekly",
  monthly: "scheduledTasks.form.monthly",
};
const NOTIFY_LABELS: Record<TaskNotify, TranslationKey> = {
  all: "scheduledTasks.form.notifyAll",
  failures: "scheduledTasks.form.notifyFailures",
  none: "scheduledTasks.form.notifyNone",
};
const NOTIFY_CHOICES: readonly TaskNotify[] = ["all", "failures", "none"];
const DAYS_OF_MONTH = Array.from({ length: 31 }, (_, index) => index + 1);

/** Ask for notification permission from the save click: browsers prompt only during a user gesture. */
function askToNotify(): void {
  if (isTauri) {
    void primeNativeNotificationPermission();
    return;
  }
  if ("Notification" in window && Notification.permission === "default") {
    void Notification.requestPermission().catch(() => undefined);
  }
}

function Row({ id, label, children }: { id?: string; label: string; children: ReactNode }) {
  return (
    <div className="flex min-h-14 flex-wrap items-center justify-between gap-x-4 gap-y-2 px-5 py-2">
      <label htmlFor={id} className="text-ui-15 text-foreground">
        {label}
      </label>
      {children}
    </div>
  );
}

// Dark mode separates by fill, not outline: a white wash on the card surface, no borders or dividers.
const CARD =
  "flex flex-col divide-y divide-border overflow-hidden rounded-[20px] border border-border dark:divide-y-0 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]";
const BARE_SELECT =
  "h-9 w-auto gap-1.5 border-none bg-transparent px-2 text-ui-15 text-muted-foreground shadow-none hover:text-foreground focus-visible:ring-1 dark:bg-transparent dark:hover:bg-transparent";
const BARE_INPUT =
  "h-9 w-auto border-none bg-transparent px-2 text-right text-ui-15 text-muted-foreground shadow-none focus-visible:ring-1 dark:bg-transparent";
// Its scroll buttons paint the base popover colour, which shows as bands on the lifted dark menu.
const LONG_LIST =
  "max-h-[min(--spacing(72),var(--radix-select-content-available-height))] [&_[data-slot^=select-scroll]]:bg-transparent";

// Select values cannot be empty, so null (no project) has a stand-in.
const NO_PROJECT = "__none__";

export function TaskDialog({
  task,
  onClose,
  onDelete,
}: {
  task: ScheduledTask | null;
  onClose: () => void;
  onDelete?: (task: ScheduledTask) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const ids = useId();
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const create = useScheduledTasksStore((state) => state.create);
  const update = useScheduledTasksStore((state) => state.update);
  const [now] = useState(() => new Date());
  // A task named after its instructions edits with an empty name, so the name keeps following them.
  const [title, setTitle] = useState(task?.named ? task.title : "");
  const [prompt, setPrompt] = useState(task?.prompt ?? "");
  const [form, setForm] = useState<ScheduleForm>(() =>
    task ? formFromSchedule(task.schedule, now) : defaultScheduleForm(now),
  );
  const [notify, setNotify] = useState<TaskNotify>(task?.notify ?? "all");
  const [archiveSuccessful, setArchiveSuccessful] = useState(task?.archiveSuccessful ?? false);
  const [projectId, setProjectId] = useState<string | null>(task?.projectId ?? null);
  const { projects } = useChatProjects();
  const openProjects = projects.filter((project) => !project.archived || project.id === projectId);
  const [advanced, setAdvanced] = useState(
    () =>
      task !== null &&
      (task.notify !== "all" || task.archiveSuccessful || task.projectId !== null),
  );
  const [busy, setBusy] = useState(false);
  const timezone = task?.timezone ?? browserTimeZone();
  const valid = prompt.trim() !== "" && scheduleFormComplete(form);
  const named = title.trim() !== "";
  const finalTitle = named ? title.trim() : titleFromPrompt(prompt);

  const patchForm = (patch: Partial<ScheduleForm>) => setForm((current) => ({ ...current, ...patch }));
  const toggleDay = (day: number) =>
    setForm((current) => ({
      ...current,
      days: current.days.includes(day)
        ? current.days.filter((entry) => entry !== day)
        : [...current.days, day],
    }));

  async function save() {
    if (!valid || busy) return;
    if (notify !== "none" && (!task || task.notify === "none")) askToNotify();
    setBusy(true);
    const schedule = scheduleFromForm(form);
    try {
      if (task) {
        // Only what changed: a rename must not move a run that is already due.
        const patch: ScheduledTaskPatch = {};
        if (finalTitle !== task.title) patch.title = finalTitle;
        if (prompt.trim() !== task.prompt) patch.prompt = prompt.trim();
        if (!sameSchedule(schedule, task.schedule)) patch.schedule = schedule;
        if (named !== task.named) patch.named = named;
        if (notify !== task.notify) patch.notify = notify;
        if (archiveSuccessful !== task.archiveSuccessful) patch.archiveSuccessful = archiveSuccessful;
        if (projectId !== task.projectId) patch.projectId = projectId;
        if (Object.keys(patch).length > 0) await update(task.id, patch);
      } else {
        await create({
          title: finalTitle,
          prompt: prompt.trim(),
          schedule,
          timezone,
          notify,
          named,
          archiveSuccessful,
          projectId,
        });
      }
      onClose();
    } catch (error) {
      toast.error(t("scheduledTasks.errors.saveFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={true}
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent
        className="corner-squircle dialog-soft-surface gap-4 sm:max-w-xl"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          const field = promptRef.current;
          field?.focus();
          field?.setSelectionRange(field.value.length, field.value.length);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            void save();
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-ui-21">
            {task ? t("scheduledTasks.form.editTitle") : t("scheduledTasks.form.createTitle")}
          </DialogTitle>
          <DialogDescription className="sr-only">{t("scheduledTasks.description")}</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          <div className={CARD}>
            <Input
              id={`${ids}-name`}
              aria-label={t("scheduledTasks.form.name")}
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              maxLength={120}
              disabled={busy}
              placeholder={t("scheduledTasks.form.namePlaceholder")}
              className="h-12 rounded-none border-none bg-transparent px-5 text-ui-15 font-medium shadow-none focus-visible:ring-0 dark:bg-transparent dark:focus-visible:bg-transparent"
            />
          </div>
          <div className={CARD}>
            <Textarea
              ref={promptRef}
              id={`${ids}-prompt`}
              aria-label={t("scheduledTasks.form.instructions")}
              aria-describedby={`${ids}-prompt-hint`}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              maxLength={20_000}
              rows={5}
              disabled={busy}
              placeholder={t("scheduledTasks.form.instructionsPlaceholder")}
              className="max-h-60 min-h-32 resize-none rounded-none border-none bg-transparent px-5 py-3.5 text-ui-15 shadow-none focus-visible:ring-0 dark:bg-transparent dark:focus-visible:bg-transparent"
            />
          </div>
        </div>
        <p id={`${ids}-prompt-hint`} className="sr-only">
          {t("scheduledTasks.form.instructionsHint")}
        </p>
        <div className={CARD}>
          <Row id={`${ids}-repeat`} label={t("scheduledTasks.form.repeat")}>
            <Select
              value={form.repeat}
              onValueChange={(value) => patchForm({ repeat: value as RepeatKind })}
              disabled={busy}
            >
              <SelectTrigger id={`${ids}-repeat`} className={BARE_SELECT}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                {REPEAT_KINDS.map((kind) => (
                  <SelectItem key={kind} value={kind}>
                    {t(REPEAT_LABELS[kind])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
          {form.repeat === "once" && (
            <Row id={`${ids}-date`} label={t("scheduledTasks.form.date")}>
              <Input
                id={`${ids}-date`}
                type="date"
                value={form.date}
                min={localDateKey(now)}
                onChange={(event) => patchForm({ date: event.target.value })}
                disabled={busy}
                className={BARE_INPUT}
              />
            </Row>
          )}
          {form.repeat === "monthly" && (
            <Row id={`${ids}-day`} label={t("scheduledTasks.form.dayOfMonth")}>
              <Select
                value={String(form.dayOfMonth)}
                onValueChange={(value) => patchForm({ dayOfMonth: Number(value) })}
                disabled={busy}
              >
                <SelectTrigger id={`${ids}-day`} className={BARE_SELECT}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="end" className={LONG_LIST}>
                  {DAYS_OF_MONTH.map((day) => (
                    <SelectItem key={day} value={String(day)}>
                      {day}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Row>
          )}
          {form.repeat === "weekly" && (
            <Row label={t("scheduledTasks.form.days")}>
              <div role="group" aria-label={t("scheduledTasks.form.days")} className="flex flex-wrap justify-end gap-1">
                {ISO_WEEKDAYS.map((day) => {
                  const selected = form.days.includes(day);
                  return (
                    <button
                      key={day}
                      type="button"
                      aria-pressed={selected}
                      title={weekdayName(day, locale, "long")}
                      disabled={busy}
                      onClick={() => toggleDay(day)}
                      className={cn(
                        "h-8 min-w-11 cursor-pointer rounded-full px-2.5 text-ui-13 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
                        selected
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover:text-foreground dark:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:text-foreground/75",
                      )}
                    >
                      {weekdayName(day, locale, "short")}
                    </button>
                  );
                })}
              </div>
            </Row>
          )}
          <Row id={`${ids}-time`} label={t("scheduledTasks.form.time")}>
            <Select
              value={form.time}
              onValueChange={(value) => patchForm({ time: value })}
              disabled={busy}
            >
              <SelectTrigger id={`${ids}-time`} title={timezone} className={BARE_SELECT}>
                <SelectValue>
                  {`${formatClock(form.time, locale)} ${zoneLabel(timezone, locale, now.getTime())}`}
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="end" className={LONG_LIST}>
                {timeChoices(form.time).map((time) => (
                  <SelectItem key={time} value={time}>
                    {formatClock(time, locale)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
        </div>
        {form.repeat === "monthly" && form.dayOfMonth > 28 && (
          <p className="-mt-2 px-1 text-ui-13 text-muted-foreground">
            {t("scheduledTasks.form.lastDayNote")}
          </p>
        )}
        <div className="pb-5">
          <button
            type="button"
            aria-expanded={advanced}
            aria-controls={advanced ? `${ids}-advanced` : undefined}
            onClick={() => setAdvanced((open) => !open)}
            className="flex cursor-pointer items-center gap-1 rounded-md px-1 text-ui-14 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {t("scheduledTasks.form.advanced")}
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className={cn("size-4 transition-transform", advanced && "rotate-180")}
            />
          </button>
          {advanced && (
            <div id={`${ids}-advanced`} className={cn(CARD, "mt-3")}>
              <Row id={`${ids}-notify`} label={t("scheduledTasks.form.notifications")}>
                <Select
                  value={notify}
                  onValueChange={(value) => setNotify(value as TaskNotify)}
                  disabled={busy}
                >
                  <SelectTrigger id={`${ids}-notify`} className={BARE_SELECT}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="end">
                    {NOTIFY_CHOICES.map((choice) => (
                      <SelectItem key={choice} value={choice}>
                        {t(NOTIFY_LABELS[choice])}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Row>
              <Row id={`${ids}-archive`} label={t("scheduledTasks.form.archiveSuccessful")}>
                <Switch
                  id={`${ids}-archive`}
                  checked={archiveSuccessful}
                  onCheckedChange={setArchiveSuccessful}
                  disabled={busy}
                  className="mr-2"
                />
              </Row>
              <Row id={`${ids}-project`} label={t("scheduledTasks.form.project")}>
                <Select
                  value={projectId ?? NO_PROJECT}
                  onValueChange={(value) => setProjectId(value === NO_PROJECT ? null : value)}
                  disabled={busy}
                >
                  <SelectTrigger id={`${ids}-project`} className={cn(BARE_SELECT, "max-w-[60%]")}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="end" className={cn(LONG_LIST, "max-w-72")}>
                    {openProjects.map((project) => (
                      <SelectItem key={project.id} value={project.id}>
                        <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                        {project.name}
                      </SelectItem>
                    ))}
                    <SelectItem value={NO_PROJECT}>{t("scheduledTasks.form.projectNone")}</SelectItem>
                  </SelectContent>
                </Select>
              </Row>
            </div>
          )}
        </div>
        <DialogFooter className="flex-wrap gap-2 sm:justify-between">
          {task && onDelete ? (
            <Button
              type="button"
              variant="destructive"
              disabled={busy}
              onClick={() => {
                onClose();
                onDelete(task);
              }}
            >
              {t("common.delete")}
            </Button>
          ) : (
            <span />
          )}
          <span className="flex flex-wrap gap-2">
            <Button type="button" variant="muted" disabled={busy} onClick={onClose}>
              {t("common.cancel")}
            </Button>
            <Button type="button" disabled={!valid || busy} onClick={() => void save()}>
              {busy
                ? t("common.saving")
                : task
                  ? t("common.save")
                  : t("scheduledTasks.form.create")}
            </Button>
          </span>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
