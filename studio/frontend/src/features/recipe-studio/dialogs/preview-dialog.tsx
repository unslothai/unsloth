// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useI18n } from "@/features/i18n";
import { cn } from "@/lib/utils";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
  CookBookIcon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useState } from "react";
import type { RecipeExecutionKind } from "../execution-types";
import type { RecipeRunSettings } from "../stores/recipe-executions";
import { FieldLabel } from "./shared/field-label";

type RunDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  kind: RecipeExecutionKind;
  onKindChange: (kind: RecipeExecutionKind) => void;
  rows: number;
  fullRunName: string;
  onFullRunNameChange: (name: string) => void;
  onRowsChange: (rows: number) => void;
  settings: RecipeRunSettings;
  onSettingsChange: (patch: Partial<RecipeRunSettings>) => void;
  loading: boolean;
  validateLoading: boolean;
  validateResult: {
    valid: boolean;
    errors: string[];
    rawDetail: string | null;
  } | null;
  errors: string[];
  onRun: () => void;
  onValidate: () => void;
  container?: HTMLDivElement | null;
};

type ValidationResult = RunDialogProps["validateResult"];

const MAX_RECORDS = 200_000;
const MAX_WORKERS = 2_048;
const MAX_SHUTDOWN_WINDOW = 10_000;
const MAX_RETRY_STEPS = 100;

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  const next = Math.floor(value);
  if (next < min) {
    return min;
  }
  if (next > max) {
    return max;
  }
  return next;
}

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function commitInt(
  raw: string,
  current: number,
  min: number,
  max: number,
  apply: (value: number) => void,
  setDraft: (value: string) => void,
): void {
  const trimmed = raw.trim();
  if (!trimmed) {
    setDraft(String(current));
    return;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    setDraft(String(current));
    return;
  }
  const next = clampInt(parsed, min, max);
  apply(next);
  setDraft(String(next));
}

function commitFloat(
  raw: string,
  current: number,
  min: number,
  max: number,
  apply: (value: number) => void,
  setDraft: (value: string) => void,
): void {
  const trimmed = raw.trim();
  if (!trimmed) {
    setDraft(String(current));
    return;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    setDraft(String(current));
    return;
  }
  const next = clampFloat(parsed, min, max);
  apply(next);
  setDraft(String(next));
}

type DraftInputFieldProps = {
  id: string;
  label: string;
  hint: string;
  inputMode: "numeric" | "decimal";
  value: string;
  onChange: (value: string) => void;
  onBlur: () => void;
  placeholder?: string;
};

function DraftInputField({
  id,
  label,
  hint,
  inputMode,
  value,
  onChange,
  onBlur,
  placeholder,
}: DraftInputFieldProps): ReactElement {
  return (
    <div className="grid gap-1.5">
      <FieldLabel label={label} htmlFor={id} hint={hint} />
      <Input
        id={id}
        type="text"
        inputMode={inputMode}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onBlur={onBlur}
        placeholder={placeholder}
      />
    </div>
  );
}

function AdvancedSettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="space-y-3 rounded-2xl border border-border/70 bg-card/60 p-4">
      <div className="space-y-0.5">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      {children}
    </div>
  );
}

function ValidationResultPanel({
  validateResult,
}: {
  validateResult: ValidationResult;
}): ReactElement | null {
  const { t } = useI18n();
  if (!validateResult) {
    return null;
  }

  return (
    <div
      className={cn(
        "space-y-3 rounded-2xl border p-4 shadow-border backdrop-blur-sm",
        validateResult.valid
          ? "border-emerald-300/70 bg-emerald-50/80 dark:border-emerald-900/60 dark:bg-emerald-950/30"
          : "border-destructive/30 bg-destructive/5",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full border",
            validateResult.valid
              ? "border-emerald-300/70 bg-emerald-500/10 text-emerald-700 dark:border-emerald-900/60 dark:text-emerald-300"
              : "border-destructive/30 bg-destructive/10 text-destructive",
          )}
        >
          <HugeiconsIcon
            icon={
              validateResult.valid ? CheckmarkCircle02Icon : AlertCircleIcon
            }
            className="size-4"
          />
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <p
            className={cn(
              "text-sm font-semibold",
              validateResult.valid
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-destructive",
            )}
          >
            {validateResult.valid
              ? t("recipe.run.validation.readyTitle")
              : t("recipe.run.validation.fixTitle")}
          </p>
          <p className="text-xs text-muted-foreground">
            {validateResult.valid
              ? t("recipe.run.validation.readyDescription")
              : t("recipe.run.validation.fixDescription")}
          </p>
        </div>
      </div>
      {!validateResult.valid && validateResult.errors.length > 0 && (
        <div className="space-y-1">
          {validateResult.errors.map((error) => (
            <p key={error} className="break-words text-xs text-destructive">
              {error}
            </p>
          ))}
        </div>
      )}
      {!validateResult.valid && validateResult.rawDetail && (
        <p className="break-words text-xs text-destructive">
          {validateResult.rawDetail}
        </p>
      )}
    </div>
  );
}

type RunDialogBodyProps = Omit<
  RunDialogProps,
  "open" | "onOpenChange" | "container"
> & {
  onClose: () => void;
};

function RunDialogBody({
  kind,
  onKindChange,
  rows,
  fullRunName,
  onFullRunNameChange,
  onRowsChange,
  settings,
  onSettingsChange,
  loading,
  validateLoading,
  validateResult,
  errors,
  onRun,
  onValidate,
  onClose,
}: RunDialogBodyProps): ReactElement {
  const { t } = useI18n();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const kindLabel =
    kind === "preview"
      ? t("recipe.run.kind.test")
      : t("recipe.run.kind.full");
  const startLabel =
    kind === "preview"
      ? t("recipe.run.action.startTest")
      : t("recipe.run.action.startFull");
  const normalizedFullRunName = fullRunName.trim();
  const isFullRunNameMissing =
    kind === "full" && normalizedFullRunName.length === 0;
  const rowHint =
    kind === "preview"
      ? t("recipe.run.records.hintPreview")
      : t("recipe.run.records.hintFull");

  const [rowsDraft, setRowsDraft] = useState(String(rows));
  const [batchSizeDraft, setBatchSizeDraft] = useState(
    String(settings.batchSize),
  );
  const [llmParallelDraft, setLlmParallelDraft] = useState(
    settings.llmParallelRequests === null
      ? ""
      : String(settings.llmParallelRequests),
  );
  const [workersDraft, setWorkersDraft] = useState(
    String(settings.nonInferenceWorkers),
  );
  const [windowDraft, setWindowDraft] = useState(
    String(settings.shutdownErrorWindow),
  );
  const [restartsDraft, setRestartsDraft] = useState(
    String(settings.maxConversationRestarts),
  );
  const [correctionsDraft, setCorrectionsDraft] = useState(
    String(settings.maxConversationCorrectionSteps),
  );
  const [shutdownRateDraft, setShutdownRateDraft] = useState(
    String(settings.shutdownErrorRate),
  );

  return (
    <>
      <DialogHeader className="space-y-2">
        <DialogTitle>{kindLabel}</DialogTitle>
        <p className="text-sm text-muted-foreground">
          {t("recipe.run.dialog.description")}
        </p>
      </DialogHeader>

      <div className="grid gap-1.5">
        <FieldLabel
          label={t("recipe.run.type.label")}
          hint={t("recipe.run.type.hint")}
        />
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant={kind === "preview" ? "default" : "outline"}
            className="corner-squircle min-h-10 justify-center whitespace-normal px-3 text-center"
            aria-pressed={kind === "preview"}
            onClick={() => onKindChange("preview")}
          >
            {t("recipe.run.kind.test")}
          </Button>
          <Button
            type="button"
            variant={kind === "full" ? "default" : "outline"}
            className="corner-squircle min-h-10 justify-center whitespace-normal px-3 text-center"
            aria-pressed={kind === "full"}
            onClick={() => onKindChange("full")}
          >
            {t("recipe.run.kind.full")}
          </Button>
        </div>
      </div>

      {kind === "full" && (
        <div className="grid gap-1.5">
          <FieldLabel
            label={t("recipe.run.name.label")}
            htmlFor="run-name"
            hint={t("recipe.run.name.hint")}
          />
          <Input
            id="run-name"
            type="text"
            value={fullRunName}
            onChange={(event) => onFullRunNameChange(event.target.value)}
            placeholder={t("recipe.run.name.placeholder")}
            aria-invalid={isFullRunNameMissing}
          />
          {isFullRunNameMissing ? (
            <p className="text-xs text-destructive">
              {t("recipe.run.name.required")}
            </p>
          ) : null}
        </div>
      )}

      <div className="grid gap-1.5">
        <FieldLabel label={t("recipe.run.records.label")} htmlFor="run-rows" hint={rowHint} />
        <Input
          id="run-rows"
          type="text"
          inputMode="numeric"
          value={rowsDraft}
          onChange={(event) => setRowsDraft(event.target.value)}
          onBlur={() =>
            commitInt(
              rowsDraft,
              rows,
              1,
              MAX_RECORDS,
              onRowsChange,
              setRowsDraft,
            )
          }
        />
      </div>

      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild={true}>
          <button
            type="button"
            className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              className={cn(
                "size-3.5 transition-transform",
                advancedOpen && "rotate-180",
              )}
            />
            {advancedOpen
              ? t("recipe.run.advanced.hide")
              : t("recipe.run.advanced.show")}
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          {kind === "full" && (
            <AdvancedSettingsSection
              title={t("recipe.run.batching.title")}
              description={t("recipe.run.batching.description")}
            >
              <div className="flex items-center justify-between gap-3 text-sm">
                <div className="space-y-0.5">
                  <span className="font-medium">{t("recipe.run.batching.enableLabel")}</span>
                  <p className="text-xs text-muted-foreground">
                    {t("recipe.run.batching.enableHint")}
                  </p>
                </div>
                <Switch
                  checked={settings.batchEnabled}
                  onCheckedChange={(checked) =>
                    onSettingsChange({ batchEnabled: Boolean(checked) })
                  }
                />
              </div>
              {rows >= 1000 && !settings.batchEnabled ? (
                <p className="text-xs text-muted-foreground">
                  {t("recipe.run.batching.recommendation")}
                </p>
              ) : null}
            </AdvancedSettingsSection>
          )}
          <AdvancedSettingsSection
            title={t("recipe.run.throughput.title")}
            description={t("recipe.run.throughput.description")}
          >
            <div className="grid gap-4 md:grid-cols-2">
              <DraftInputField
                id="run-llm-parallel"
                label={t("recipe.run.throughput.aiRequests")}
                hint={t("recipe.run.throughput.aiRequestsHint")}
                inputMode="numeric"
                value={llmParallelDraft}
                onChange={setLlmParallelDraft}
                onBlur={() => {
                  const trimmed = llmParallelDraft.trim();
                  if (!trimmed) {
                    onSettingsChange({ llmParallelRequests: null });
                    setLlmParallelDraft("");
                    return;
                  }
                  const parsed = Number(trimmed);
                  if (!Number.isFinite(parsed)) {
                    setLlmParallelDraft(
                      settings.llmParallelRequests === null
                        ? ""
                        : String(settings.llmParallelRequests),
                    );
                    return;
                  }
                  const next = clampInt(parsed, 1, MAX_WORKERS);
                  onSettingsChange({ llmParallelRequests: next });
                  setLlmParallelDraft(String(next));
                }}
                placeholder={t("recipe.run.throughput.aiRequestsPlaceholder")}
              />
              <DraftInputField
                id="run-non-inference-workers"
                label={t("recipe.run.throughput.cpuWorkers")}
                hint={t("recipe.run.throughput.cpuWorkersHint")}
                inputMode="numeric"
                value={workersDraft}
                onChange={setWorkersDraft}
                onBlur={() =>
                  commitInt(
                    workersDraft,
                    settings.nonInferenceWorkers,
                    1,
                    MAX_WORKERS,
                    (value) => onSettingsChange({ nonInferenceWorkers: value }),
                    setWorkersDraft,
                  )
                }
              />
              {kind === "full" && settings.batchEnabled && (
                <>
                  <DraftInputField
                    id="run-batch-size"
                    label={t("recipe.run.batching.batchSize")}
                    hint={t("recipe.run.batching.batchSizeHint")}
                    inputMode="numeric"
                    value={batchSizeDraft}
                    onChange={setBatchSizeDraft}
                    onBlur={() =>
                      commitInt(
                        batchSizeDraft,
                        settings.batchSize,
                        1,
                        MAX_RECORDS,
                        (value) => onSettingsChange({ batchSize: value }),
                        setBatchSizeDraft,
                      )
                    }
                  />
                  <div className="flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground">
                    <div className="space-y-0.5">
                      <p className="font-medium">{t("recipe.run.batching.mergeLabel")}</p>
                      <p className="text-xs text-muted-foreground">
                        {t("recipe.run.batching.mergeHint")}
                      </p>
                    </div>
                    <Switch
                      checked={settings.mergeBatches}
                      onCheckedChange={(checked) =>
                        onSettingsChange({ mergeBatches: Boolean(checked) })
                      }
                    />
                  </div>
                </>
              )}
            </div>
          </AdvancedSettingsSection>
          <AdvancedSettingsSection
            title={t("recipe.run.retries.title")}
            description={t("recipe.run.retries.description")}
          >
            <div className="grid gap-4 md:grid-cols-2">
              <DraftInputField
                id="run-shutdown-window"
                label={t("recipe.run.retries.failureWindow")}
                hint={t("recipe.run.retries.failureWindowHint")}
                inputMode="numeric"
                value={windowDraft}
                onChange={setWindowDraft}
                onBlur={() =>
                  commitInt(
                    windowDraft,
                    settings.shutdownErrorWindow,
                    1,
                    MAX_SHUTDOWN_WINDOW,
                    (value) => onSettingsChange({ shutdownErrorWindow: value }),
                    setWindowDraft,
                  )
                }
              />
              <DraftInputField
                id="run-shutdown-rate"
                label={t("recipe.run.retries.stopAfterFailures")}
                hint={t("recipe.run.retries.stopAfterFailuresHint")}
                inputMode="decimal"
                value={shutdownRateDraft}
                onChange={setShutdownRateDraft}
                onBlur={() =>
                  commitFloat(
                    shutdownRateDraft,
                    settings.shutdownErrorRate,
                    0,
                    1,
                    (value) => onSettingsChange({ shutdownErrorRate: value }),
                    setShutdownRateDraft,
                  )
                }
              />
              <DraftInputField
                id="run-max-restarts"
                label={t("recipe.run.retries.fullRetries")}
                hint={t("recipe.run.retries.fullRetriesHint")}
                inputMode="numeric"
                value={restartsDraft}
                onChange={setRestartsDraft}
                onBlur={() =>
                  commitInt(
                    restartsDraft,
                    settings.maxConversationRestarts,
                    0,
                    MAX_RETRY_STEPS,
                    (value) =>
                      onSettingsChange({ maxConversationRestarts: value }),
                    setRestartsDraft,
                  )
                }
              />
              <DraftInputField
                id="run-correction-steps"
                label={t("recipe.run.retries.correctionAttempts")}
                hint={t("recipe.run.retries.correctionAttemptsHint")}
                inputMode="numeric"
                value={correctionsDraft}
                onChange={setCorrectionsDraft}
                onBlur={() =>
                  commitInt(
                    correctionsDraft,
                    settings.maxConversationCorrectionSteps,
                    0,
                    MAX_RETRY_STEPS,
                    (value) =>
                      onSettingsChange({ maxConversationCorrectionSteps: value }),
                    setCorrectionsDraft,
                  )
                }
              />
              <div className="flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground md:col-span-2">
                <div className="space-y-0.5">
                  <p className="font-medium">{t("recipe.run.retries.keepRunningLabel")}</p>
                  <p className="text-xs text-muted-foreground">
                    {t("recipe.run.retries.keepRunningHint")}
                  </p>
                </div>
                <Switch
                  checked={settings.disableEarlyShutdown}
                  onCheckedChange={(checked) =>
                    onSettingsChange({
                      disableEarlyShutdown: Boolean(checked),
                    })
                  }
                />
              </div>
            </div>
          </AdvancedSettingsSection>
        </CollapsibleContent>
      </Collapsible>

      {errors.length > 0 && (
        <div className="max-h-44 space-y-2 overflow-y-auto rounded-2xl border border-destructive/30 bg-destructive/5 p-4 shadow-border">
          <div className="flex items-center gap-2">
            <HugeiconsIcon
              icon={AlertCircleIcon}
              className="size-4 text-destructive"
            />
            <Badge
              variant="outline"
              className="rounded-full text-[10px] text-destructive"
            >
              {t("recipe.run.beforeRun")}
            </Badge>
          </div>
          {errors.map((error) => (
            <p key={error} className="break-words text-xs text-destructive">
              {error}
            </p>
          ))}
        </div>
      )}

      <ValidationResultPanel validateResult={validateResult} />

      <DialogFooter>
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={loading}
          className="corner-squircle border-border/70 bg-card/70"
        >
          {t("recipe.run.action.cancel")}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={onValidate}
          disabled={loading || validateLoading}
          className="corner-squircle border-border/70 bg-card/70"
        >
          <HugeiconsIcon icon={TestTube01Icon} className="size-3.5" />
          {validateLoading
            ? t("recipe.run.action.checking")
            : t("recipe.run.action.checkRecipe")}
        </Button>
        <Button
          type="button"
          onClick={onRun}
          disabled={loading || isFullRunNameMissing}
          className="corner-squircle"
        >
          <HugeiconsIcon icon={CookBookIcon} className="size-3.5" />
          {loading ? t("recipe.run.action.starting") : startLabel}
        </Button>
      </DialogFooter>
    </>
  );
}

export function RunDialog({
  open,
  onOpenChange,
  container,
  ...contentProps
}: RunDialogProps): ReactElement {
  const draftKey = [open ? "open" : "closed", contentProps.kind].join("|");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle max-h-[650px] overflow-y-auto overflow-x-hidden border-border/70 bg-background/95 sm:max-w-2xl shadow-border backdrop-blur-xl"
      >
        <RunDialogBody
          key={draftKey}
          {...contentProps}
          onClose={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
