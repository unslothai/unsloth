// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useId, useState } from "react";
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
import { Textarea } from "@/components/ui/textarea";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { ScheduledTask, ScheduledTaskPatch } from "./api";
import {
  ISO_WEEKDAYS,
  REPEAT_KINDS,
  type RepeatKind,
  type ScheduleForm,
  browserTimeZone,
  defaultScheduleForm,
  formFromSchedule,
  localDateKey,
  sameSchedule,
  scheduleFormComplete,
  scheduleFromForm,
  weekdayName,
} from "./schedule";
import { useScheduledTasksStore } from "./store";

const REPEAT_LABELS: Record<RepeatKind, TranslationKey> = {
  once: "scheduledTasks.form.once",
  daily: "scheduledTasks.form.daily",
  weekdays: "scheduledTasks.form.weekdays",
  weekly: "scheduledTasks.form.weekly",
  monthly: "scheduledTasks.form.monthly",
};
const DAYS_OF_MONTH = Array.from({ length: 31 }, (_, index) => index + 1);

/** Results arrive while the user is elsewhere, so ask once, from the click that makes the first
 *  task: browsers only show the prompt during a user gesture. */
function askToNotify(): void {
  if (isTauri) {
    void primeNativeNotificationPermission();
    return;
  }
  if ("Notification" in window && Notification.permission === "default") {
    void Notification.requestPermission().catch(() => undefined);
  }
}

// One line of the schedule card: the label on the left, its control on the right.
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

const CARD =
  "flex flex-col divide-y divide-border/70 overflow-hidden rounded-[20px] border border-border/70 dark:divide-[rgb(255_255_255_/_calc(0.1*var(--contrast-edge-gain,1)))] dark:border-[rgb(255_255_255_/_calc(0.12*var(--contrast-edge-gain,1)))]";
// Controls sit flush in their row and read as its value, like a settings list.
const BARE_SELECT =
  "h-9 w-auto gap-1.5 border-none bg-transparent px-2 text-ui-15 text-muted-foreground shadow-none hover:text-foreground focus-visible:ring-1 dark:bg-transparent dark:hover:bg-transparent";
const BARE_INPUT =
  "h-9 w-auto border-none bg-transparent px-2 text-right text-ui-15 text-muted-foreground shadow-none focus-visible:ring-1 dark:bg-transparent";

/** Creates a task, or edits `task`. Mounted while open, so each opening starts from the task. */
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
  const create = useScheduledTasksStore((state) => state.create);
  const update = useScheduledTasksStore((state) => state.update);
  const [now] = useState(() => new Date());
  const [title, setTitle] = useState(task?.title ?? "");
  const [prompt, setPrompt] = useState(task?.prompt ?? "");
  const [form, setForm] = useState<ScheduleForm>(() =>
    task ? formFromSchedule(task.schedule, now) : defaultScheduleForm(now),
  );
  const [busy, setBusy] = useState(false);
  // An edit keeps the zone the task was made in; a new task takes this browser's.
  const timezone = task?.timezone ?? browserTimeZone();
  const valid = title.trim() !== "" && prompt.trim() !== "" && scheduleFormComplete(form);

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
    if (!task) askToNotify();
    setBusy(true);
    const schedule = scheduleFromForm(form);
    try {
      if (task) {
        // Only what changed: a rename must not move a run that is already due.
        const patch: ScheduledTaskPatch = {};
        if (title.trim() !== task.title) patch.title = title.trim();
        if (prompt.trim() !== task.prompt) patch.prompt = prompt.trim();
        if (!sameSchedule(schedule, task.schedule)) patch.schedule = schedule;
        if (Object.keys(patch).length > 0) await update(task.id, patch);
      } else {
        await create({ title: title.trim(), prompt: prompt.trim(), schedule, timezone });
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
        <div className={CARD}>
          <Input
            id={`${ids}-name`}
            aria-label={t("scheduledTasks.form.name")}
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            maxLength={120}
            autoFocus={true}
            disabled={busy}
            placeholder={t("scheduledTasks.form.namePlaceholder")}
            className="h-12 rounded-none border-none bg-transparent px-5 text-ui-15 font-medium shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
          <Textarea
            id={`${ids}-prompt`}
            aria-label={t("scheduledTasks.form.instructions")}
            aria-describedby={`${ids}-prompt-hint`}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={20_000}
            rows={5}
            disabled={busy}
            placeholder={t("scheduledTasks.form.instructionsPlaceholder")}
            className="max-h-60 min-h-32 resize-none rounded-none border-none bg-transparent px-5 py-3.5 text-ui-15 shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
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
                <SelectContent
                  align="end"
                  className="max-h-[min(--spacing(72),var(--radix-select-content-available-height))]"
                >
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
                          : "bg-muted text-muted-foreground hover:text-foreground",
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
            <Input
              id={`${ids}-time`}
              type="time"
              value={form.time}
              onChange={(event) => patchForm({ time: event.target.value })}
              disabled={busy}
              className={BARE_INPUT}
            />
          </Row>
        </div>
        <p className="-mt-2 px-1 text-ui-13 text-muted-foreground">
          {form.repeat === "monthly" && form.dayOfMonth > 28
            ? `${t("scheduledTasks.form.lastDayNote")} `
            : ""}
          {t("scheduledTasks.form.timezone", { timezone })}
        </p>
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
