// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Folder01Icon } from "@/lib/folder-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useId, useMemo, useRef, useState } from "react";
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
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  buildExternalModelId,
  chatLocalModelOptions,
  type ExternalProviderConfig,
  externalReasoningTakesEffort,
  getExternalReasoningCapabilities,
  useChatProjects,
  useChatRuntimeStore,
  useExternalProvidersStore,
} from "@/features/chat";
import { useDeviceInventorySources } from "@/features/hub";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { ScheduledTask, ScheduledTaskPatch, TaskNotify, TaskReasoningEffort } from "./api";
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
  hourly: "scheduledTasks.form.hourly",
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

function Row({
  id,
  label,
  hint,
  children,
}: { id?: string; label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="flex min-h-14 flex-wrap items-center justify-between gap-x-4 gap-y-2 px-5 py-2">
      <span className="flex items-center gap-1.5">
        <label htmlFor={id} className="text-ui-15 text-foreground">
          {label}
        </label>
        {hint && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={hint}
                className="inline-flex rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                <HugeiconsIcon icon={InformationCircleIcon} className="size-[var(--ui-icon-size-hint)]" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-[calc(300px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
              {hint}
            </TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </div>
  );
}

// Dark mode separates by fill, not outline: a white wash on the card surface, no borders or dividers.
const CARD =
  "flex flex-col divide-y divide-border overflow-hidden rounded-[20px] border border-border dark:divide-y-0 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]";
const BARE_SELECT =
  "-mr-1.5 h-9 w-auto gap-1.5 border-none bg-transparent px-3.5 text-ui-15 text-muted-foreground shadow-none hover:text-foreground focus-visible:ring-1 dark:bg-transparent";
const BARE_INPUT =
  "-mr-1.5 h-9 w-auto border-none bg-transparent px-3.5 text-right text-ui-15 text-muted-foreground shadow-none focus-visible:ring-1 dark:bg-transparent";
// Its scroll buttons paint the base popover colour, which shows as bands on the lifted dark menu.
const LONG_LIST =
  "max-h-[min(--spacing(72),var(--radix-select-content-available-height))] [&_[data-slot^=select-scroll]]:bg-transparent";

// Select values cannot be empty, so null (the loaded model, the model's default) has a stand-in.
const LOADED_MODEL = "__loaded__";
const NO_PROJECT = "__none__";
const DEFAULT_EFFORT = "__default__";
const RELATIVE_LABELS: Partial<Record<TaskReasoningEffort, TranslationKey>> = {
  off: "scheduledTasks.form.reasoningOff",
  lowest: "scheduledTasks.form.reasoningLowest",
  highest: "scheduledTasks.form.reasoningHighest",
};
const ALL_EFFORTS: readonly TaskReasoningEffort[] = ["off", "lowest", "highest"];
const CONNECTION_LEVELS: readonly TaskReasoningEffort[] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/** A connection's level as the Chat composer names it (formatReasoningEffortLabel). */
function levelLabel(level: TaskReasoningEffort, modelId: string | undefined): string {
  if (level === "max") return "Max";
  if (level === "xhigh") {
    const normalized = modelId?.trim().toLowerCase() ?? "";
    return normalized.startsWith("claude-opus-4-6") || normalized.startsWith("claude-sonnet-4-6")
      ? "Max"
      : "Extra High";
  }
  return level.charAt(0).toUpperCase() + level.slice(1);
}

type LoadedReasoning = { supportsReasoning: boolean; reasoningStyle: string; reasoningAlwaysOn: boolean };

type ConnectionChoice = { id: string; modelId: string; provider: ExternalProviderConfig };

/** Models a task can run on: local and connected providers, never one needing a download. */
function useModelChoices(current: string | null) {
  const inventory = useDeviceInventorySources(["localModels", "cachedGguf"]);
  const providers = useExternalProvidersStore((state) => state.providers);
  const connectionsEnabled = useExternalProvidersStore((state) => state.connectionsEnabled);
  return useMemo(() => {
    const local = chatLocalModelOptions(inventory.localModels.rows).map(({ id, name }) => ({ id, name }));
    // GGUF only: a run switches back afterwards only to a GGUF.
    for (const repo of inventory.cachedGguf.rows) {
      if (!local.some((option) => option.id === repo.repo_id)) {
        local.push({ id: repo.repo_id, name: repo.repo_id });
      }
    }
    const connections = (connectionsEnabled ? providers : []).map((provider) => ({
      provider,
      models: provider.models.map(
        (modelId): ConnectionChoice => ({ id: buildExternalModelId(provider.id, modelId), modelId, provider }),
      ),
    }));
    const known =
      current === null ||
      local.some((option) => option.id === current) ||
      connections.some((group) => group.models.some((option) => option.id === current));
    // A saved choice that is gone (deleted, disconnected) stays visible rather than silently changing.
    const missing = known ? null : current;
    const connection = connections.flatMap((group) => group.models).find((option) => option.id === current);
    return { local, connections, missing, connection };
  }, [inventory.localModels.rows, inventory.cachedGguf.rows, providers, connectionsEnabled, current]);
}

/** Efforts the task's model takes; all of them for an unloaded model (a run resolves it), empty when none. */
function effortChoices(
  connection: ConnectionChoice | undefined,
  loaded: LoadedReasoning | null,
): readonly TaskReasoningEffort[] {
  if (!connection) {
    if (!loaded) return ALL_EFFORTS;
    if (!loaded.supportsReasoning) return [];
    return loaded.reasoningAlwaysOn || loaded.reasoningStyle === "reasoning_effort"
      ? ["lowest", "highest"]
      : ALL_EFFORTS;
  }
  const { provider, modelId } = connection;
  const caps = getExternalReasoningCapabilities(provider.providerType, modelId, {
    isReasoningProvider: provider.isReasoningModel === true,
    baseUrl: provider.baseUrl ?? null,
    apiType: provider.apiType,
    reasoningConfig: provider.reasoningConfig,
  });
  if (!externalReasoningTakesEffort(caps)) return [];
  const levels = CONNECTION_LEVELS.filter((level) =>
    (caps.reasoningEffortLevels as readonly string[]).includes(level),
  );
  return caps.supportsReasoningOff ? ["off", ...levels] : levels;
}

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
  const [model, setModel] = useState<string | null>(task?.model ?? null);
  const [allowLoad, setAllowLoad] = useState(task?.allowModelLoad ?? false);
  const [archiveSuccessful, setArchiveSuccessful] = useState(task?.archiveSuccessful ?? false);
  const [projectId, setProjectId] = useState<string | null>(task?.projectId ?? null);
  const { projects } = useChatProjects();
  const openProjects = projects.filter((project) => !project.archived || project.id === projectId);
  const [chosenEffort, setEffort] = useState<TaskReasoningEffort | null>(task?.reasoningEffort ?? null);
  const models = useModelChoices(model);
  const checkpoint = useChatRuntimeStore((state) => state.params.checkpoint);
  const supportsReasoning = useChatRuntimeStore((state) => state.supportsReasoning);
  const reasoningStyle = useChatRuntimeStore((state) => state.reasoningStyle);
  const reasoningAlwaysOn = useChatRuntimeStore((state) => state.reasoningAlwaysOn);
  const loadedReasoning =
    checkpoint && !models.connection && (model === null || model === checkpoint)
      ? { supportsReasoning, reasoningStyle, reasoningAlwaysOn }
      : null;
  const efforts = effortChoices(models.connection, loadedReasoning);
  const effort = chosenEffort !== null && efforts.includes(chosenEffort) ? chosenEffort : null;
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
        if (model !== task.model) patch.model = model;
        // Only a changed choice: a saved level the current model does not offer is kept, not cleared.
        if (chosenEffort !== task.reasoningEffort) patch.reasoningEffort = effort;
        if (allowLoad !== task.allowModelLoad) patch.allowModelLoad = allowLoad;
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
          model,
          reasoningEffort: effort,
          allowModelLoad: allowLoad,
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
          <div className={cn(CARD, "rounded-full")}>
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
          <Row id={`${ids}-model`} label={t("scheduledTasks.form.model")}>
            <Select
              value={model ?? LOADED_MODEL}
              onValueChange={(value) => setModel(value === LOADED_MODEL ? null : value)}
              disabled={busy}
            >
              <SelectTrigger id={`${ids}-model`} className={cn(BARE_SELECT, "max-w-[60%]")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end" className={cn(LONG_LIST, "max-w-80")}>
                <SelectItem value={LOADED_MODEL}>{t("scheduledTasks.form.modelLoaded")}</SelectItem>
                {models.missing && (
                  <SelectItem value={models.missing}>
                    {t("scheduledTasks.form.modelMissing", { model: models.missing })}
                  </SelectItem>
                )}
                {models.local.length > 0 && (
                  <SelectGroup>
                    <SelectLabel>{t("scheduledTasks.form.modelOnDevice")}</SelectLabel>
                    {models.local.map((option) => (
                      <SelectItem key={option.id} value={option.id}>
                        {option.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
                {models.connections.map(
                  (group) =>
                    group.models.length > 0 && (
                      <SelectGroup key={group.provider.id}>
                        <SelectLabel>{group.provider.name}</SelectLabel>
                        {group.models.map((option) => (
                          <SelectItem key={option.id} value={option.id}>
                            {option.modelId}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    ),
                )}
                {models.local.length === 0 && models.connections.every((group) => group.models.length === 0) && (
                  <p className="max-w-64 px-3 py-2 text-ui-12 text-muted-foreground">
                    {t("scheduledTasks.form.modelEmpty")}
                  </p>
                )}
              </SelectContent>
            </Select>
          </Row>
          {model !== null && !models.connection && (
            <Row
              id={`${ids}-allow-load`}
              label={t("scheduledTasks.form.allowLoad")}
              hint={t("scheduledTasks.form.allowLoadHint")}
            >
              <Switch
                id={`${ids}-allow-load`}
                checked={allowLoad}
                onCheckedChange={setAllowLoad}
                disabled={busy}
                className="mr-2"
              />
            </Row>
          )}
          {efforts.length > 0 && (
            <Row id={`${ids}-effort`} label={t("scheduledTasks.form.reasoning")}>
              <Select
                value={effort ?? DEFAULT_EFFORT}
                onValueChange={(value) =>
                  setEffort(value === DEFAULT_EFFORT ? null : (value as TaskReasoningEffort))
                }
                disabled={busy}
              >
                <SelectTrigger id={`${ids}-effort`} className={BARE_SELECT}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="end">
                  {efforts.includes("off") && (
                    <SelectItem value="off">{t("scheduledTasks.form.reasoningOff")}</SelectItem>
                  )}
                  <SelectItem value={DEFAULT_EFFORT}>{t("scheduledTasks.form.reasoningDefault")}</SelectItem>
                  {efforts.filter((choice) => choice !== "off").map((choice) => (
                    <SelectItem key={choice} value={choice}>
                      {RELATIVE_LABELS[choice]
                        ? t(RELATIVE_LABELS[choice])
                        : levelLabel(choice, models.connection?.modelId)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Row>
          )}
        </div>
        {model !== null && !models.connection && (
          <p className="-mt-2 px-1 text-ui-13 text-muted-foreground">
            {allowLoad ? t("scheduledTasks.form.modelNoteAllowed") : t("scheduledTasks.form.modelNote")}
          </p>
        )}
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
          {form.repeat === "hourly" && (
            <Row id={`${ids}-hours`} label={t("scheduledTasks.form.intervalHours")}>
              <Input
                id={`${ids}-hours`}
                type="number"
                min={1}
                max={24}
                step={1}
                value={Number.isNaN(form.hours) ? "" : form.hours}
                onChange={(event) => patchForm({ hours: event.target.valueAsNumber })}
                disabled={busy}
                className={cn(BARE_INPUT, "w-24 pr-8")}
              />
            </Row>
          )}
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
          {form.repeat !== "hourly" && (
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
          )}
        </div>
        {form.repeat === "hourly" && (
          <p className="-mt-2 px-1 text-ui-13 text-muted-foreground">
            {t("scheduledTasks.form.hourlyNote")}
          </p>
        )}
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
