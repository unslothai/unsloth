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

function Field({ id, label, children }: { id: string; label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={id} className="text-ui-13 font-medium text-foreground">
        {label}
      </label>
      {children}
    </div>
  );
}

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
        className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg"
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
        <Field id={`${ids}-name`} label={t("scheduledTasks.form.name")}>
          <Input
            id={`${ids}-name`}
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            maxLength={120}
            autoFocus={true}
            disabled={busy}
            placeholder={t("scheduledTasks.form.namePlaceholder")}
          />
        </Field>
        <div className="flex flex-col gap-1.5">
          <label htmlFor={`${ids}-prompt`} className="text-ui-13 font-medium text-foreground">
            {t("scheduledTasks.form.instructions")}
          </label>
          <p className="text-ui-13 text-muted-foreground">
            {t("scheduledTasks.form.instructionsHint")}
          </p>
          <Textarea
            id={`${ids}-prompt`}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={20_000}
            rows={4}
            disabled={busy}
            placeholder={t("scheduledTasks.form.instructionsPlaceholder")}
            className="mt-0.5 max-h-60 rounded-[16px]"
          />
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <Field id={`${ids}-repeat`} label={t("scheduledTasks.form.repeat")}>
            <Select
              value={form.repeat}
              onValueChange={(value) => patchForm({ repeat: value as RepeatKind })}
              disabled={busy}
            >
              <SelectTrigger id={`${ids}-repeat`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {REPEAT_KINDS.map((kind) => (
                  <SelectItem key={kind} value={kind}>
                    {t(REPEAT_LABELS[kind])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          {form.repeat === "once" && (
            <Field id={`${ids}-date`} label={t("scheduledTasks.form.date")}>
              <Input
                id={`${ids}-date`}
                type="date"
                value={form.date}
                min={localDateKey(now)}
                onChange={(event) => patchForm({ date: event.target.value })}
                disabled={busy}
              />
            </Field>
          )}
          {form.repeat === "monthly" && (
            <Field id={`${ids}-day`} label={t("scheduledTasks.form.dayOfMonth")}>
              <Select
                value={String(form.dayOfMonth)}
                onValueChange={(value) => patchForm({ dayOfMonth: Number(value) })}
                disabled={busy}
              >
                <SelectTrigger id={`${ids}-day`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-[min(--spacing(72),var(--radix-select-content-available-height))]">
                  {DAYS_OF_MONTH.map((day) => (
                    <SelectItem key={day} value={String(day)}>
                      {day}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          )}
          <Field id={`${ids}-time`} label={t("scheduledTasks.form.time")}>
            <Input
              id={`${ids}-time`}
              type="time"
              value={form.time}
              onChange={(event) => patchForm({ time: event.target.value })}
              disabled={busy}
            />
          </Field>
        </div>
        {form.repeat === "weekly" && (
          <div className="flex flex-col gap-1.5">
            <span className="text-ui-13 font-medium text-foreground">
              {t("scheduledTasks.form.days")}
            </span>
            <div role="group" aria-label={t("scheduledTasks.form.days")} className="flex flex-wrap gap-1.5">
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
                      "h-8 min-w-12 cursor-pointer rounded-full px-3 text-ui-13 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
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
          </div>
        )}
        <p className="text-ui-13 text-muted-foreground">
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
            <Button type="button" variant="ghost" disabled={busy} onClick={onClose}>
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
