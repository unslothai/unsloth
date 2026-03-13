// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  CookBookIcon,
  SparklesIcon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
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
    <div className="grid gap-2">
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

function ValidationResultPanel({
  validateResult,
}: {
  validateResult: ValidationResult;
}): ReactElement | null {
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
            icon={validateResult.valid ? CheckmarkCircle02Icon : AlertCircleIcon}
            className="size-4"
          />
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <p
            className={cn(
              "text-sm font-semibold",
              validateResult.valid ? "text-emerald-700 dark:text-emerald-300" : "text-destructive",
            )}
          >
            {validateResult.valid ? "Recipe looks good" : "Recipe needs attention"}
          </p>
          <p className="text-xs text-muted-foreground">
            {validateResult.valid
              ? "Validation passed. You can start the run when ready."
              : "Fix the issues below, then validate again."}
          </p>
        </div>
      </div>
      {!validateResult.valid && validateResult.errors.length > 0 && (
        <div className="space-y-1">
          {validateResult.errors.map((error) => (
            <p key={error} className="text-xs text-destructive">
              {error}
            </p>
          ))}
        </div>
      )}
      {!validateResult.valid && validateResult.rawDetail && (
        <p className="text-xs text-destructive">{validateResult.rawDetail}</p>
      )}
    </div>
  );
}

export function RunDialog({
  open,
  onOpenChange,
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
  container,
}: RunDialogProps): ReactElement {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const kindLabel = kind === "preview" ? "Preview" : "Full run";
  const normalizedFullRunName = fullRunName.trim();
  const isFullRunNameMissing = kind === "full" && normalizedFullRunName.length === 0;
  const rowHint =
    kind === "preview"
      ? "How many sample rows to generate for a quick check."
      : "How many rows to generate in total.";

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
  const showBatchingHint =
    kind === "full" && rows >= 1000 && !settings.batchEnabled;

  useEffect(() => {
    if (!open) {
      return;
    }
    setRowsDraft(String(rows));
    setBatchSizeDraft(String(settings.batchSize));
    setLlmParallelDraft(
      settings.llmParallelRequests === null
        ? ""
        : String(settings.llmParallelRequests),
    );
    setWorkersDraft(String(settings.nonInferenceWorkers));
    setWindowDraft(String(settings.shutdownErrorWindow));
    setRestartsDraft(String(settings.maxConversationRestarts));
    setCorrectionsDraft(String(settings.maxConversationCorrectionSteps));
    setShutdownRateDraft(String(settings.shutdownErrorRate));
  }, [
    rows,
    settings.batchSize,
    settings.llmParallelRequests,
    settings.nonInferenceWorkers,
    settings.shutdownErrorWindow,
    settings.maxConversationRestarts,
    settings.maxConversationCorrectionSteps,
    settings.shutdownErrorRate,
    open,
  ]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle border-border/70 bg-background/95 sm:max-w-2xl shadow-border backdrop-blur-xl"
      >
        <DialogHeader className="space-y-2">
          <DialogTitle>{kindLabel} settings</DialogTitle>
          <p className="text-sm text-muted-foreground">
            Configure run size and performance knobs for this execution.
          </p>
        </DialogHeader>

        {showBatchingHint && (
          <div className="flex items-start gap-3 rounded-2xl border border-amber-300/70 bg-amber-50/80 p-4 shadow-border dark:border-amber-900/60 dark:bg-amber-950/30">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-full border border-amber-300/70 bg-amber-500/10 text-amber-700 dark:border-amber-900/60 dark:text-amber-300">
              <HugeiconsIcon icon={SparklesIcon} className="size-4" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-200">
                Bigger runs usually feel smoother with batching on
              </p>
              <p className="text-xs leading-relaxed text-amber-900/80 dark:text-amber-100/80">
                You&apos;re generating {rows.toLocaleString()} records. Turning on batching
                usually makes larger runs easier to manage and more resilient if
                something goes wrong mid-run.
              </p>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between rounded-2xl border border-border/70 bg-card/60 px-4 py-3 text-sm shadow-border">
          <div className="space-y-0.5">
            <span className="font-medium text-foreground">Preview mode</span>
            <p className="text-xs text-muted-foreground">
              Turn this off for a full dataset run.
            </p>
          </div>
          <Switch
            checked={kind === "preview"}
            onCheckedChange={(checked) =>
              onKindChange(checked ? "preview" : "full")
            }
          />
        </div>

        {kind === "full" && (
          <div className="grid gap-2">
            <FieldLabel
              label="Run name"
              htmlFor="run-name"
              hint="Optional label shown in executions list."
            />
            <Input
              id="run-name"
              type="text"
              value={fullRunName}
              onChange={(event) => onFullRunNameChange(event.target.value)}
              placeholder="Sprint dataset v2"
              aria-invalid={isFullRunNameMissing}
            />
            {isFullRunNameMissing ? (
              <p className="text-xs text-destructive">
                Run name is required before starting a full run.
              </p>
            ) : null}
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-2">
          <div className="grid gap-2">
            <FieldLabel label="Records" htmlFor="run-rows" hint={rowHint} />
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
          <div className="grid gap-2">
            <FieldLabel
              label="LLM parallel"
              htmlFor="run-llm-parallel"
              hint="How many LLM calls run at once. Leave empty to keep each model's own setting."
            />
            <Input
              id="run-llm-parallel"
              type="text"
              inputMode="numeric"
              placeholder="Use model config"
              value={llmParallelDraft}
              onChange={(event) => setLlmParallelDraft(event.target.value)}
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
            />
          </div>
        </div>

        {kind === "full" && (
          <div className="space-y-3 rounded-2xl border border-border/70 bg-card/60 p-4 shadow-border">
            <div className="flex items-center justify-between gap-3 text-sm">
              <div className="space-y-0.5">
                <span className="font-medium">Enable batching</span>
                <p className="text-xs text-muted-foreground">
                  Split big runs into manageable chunks.
                </p>
              </div>
              <Switch
                checked={settings.batchEnabled}
                onCheckedChange={(checked) =>
                  onSettingsChange({ batchEnabled: Boolean(checked) })
                }
              />
            </div>

            {settings.batchEnabled && (
              <div className="space-y-3">
                <DraftInputField
                  id="run-batch-size"
                  label="Batch size"
                  hint="Rows handled per batch during generation."
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
                <div className="flex items-center justify-between gap-3 text-sm">
                  <div className="space-y-0.5">
                    <span className="font-medium">Merge batches to one parquet</span>
                    <p className="text-xs text-muted-foreground">
                      Combine chunk outputs into one final file when done.
                    </p>
                  </div>
                  <Switch
                    checked={settings.mergeBatches}
                    onCheckedChange={(checked) =>
                      onSettingsChange({ mergeBatches: Boolean(checked) })
                    }
                  />
                </div>
              </div>
            )}
          </div>
        )}

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild={true}>
            <button
              type="button"
              className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
            >
              {advancedOpen ? "Hide advanced" : "Show advanced"}
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 space-y-4">
            <div className="grid gap-4 rounded-2xl border border-border/70 bg-card/60 p-4 shadow-border md:grid-cols-2">
              <DraftInputField
                id="run-non-inference-workers"
                label="CPU workers"
                hint="Worker threads for non-LLM steps like samplers and expressions."
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
              <DraftInputField
                id="run-shutdown-window"
                label="Error window"
                hint="How many attempts to observe before early-stop checks kick in."
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
                id="run-max-restarts"
                label="Conversation restarts"
                hint="How many full retries to do if model output fails validation."
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
                label="Correction steps"
                hint="Extra in-chat fix attempts before a full retry."
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
                      onSettingsChange({
                        maxConversationCorrectionSteps: value,
                      }),
                    setCorrectionsDraft,
                  )
                }
              />
              <DraftInputField
                id="run-shutdown-rate"
                label="Shutdown error rate"
                hint="Stop early if failure rate passes this value. Example: 0.5 = 50%."
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
              <div className="flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground md:col-span-2">
                <div className="space-y-0.5">
                  <p className="font-medium">Keep running through failures</p>
                  <p className="text-xs text-muted-foreground">
                    Recommended for longer runs when you want maximum output.
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
          </CollapsibleContent>
        </Collapsible>

        {errors.length > 0 && (
          <div className="max-h-44 space-y-2 overflow-y-auto rounded-2xl border border-destructive/30 bg-destructive/5 p-4 shadow-border">
            <div className="flex items-center gap-2">
              <HugeiconsIcon icon={AlertCircleIcon} className="size-4 text-destructive" />
              <Badge variant="outline" className="rounded-full text-[10px] text-destructive">
                Run checks
              </Badge>
            </div>
            {errors.map((error) => (
              <p key={error} className="text-xs text-destructive">
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
            onClick={() => onOpenChange(false)}
            disabled={loading}
            className="corner-squircle border-border/70 bg-card/70"
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onValidate}
            disabled={loading || validateLoading}
            className="corner-squircle border-border/70 bg-card/70"
          >
            <HugeiconsIcon icon={TestTube01Icon} className="size-3.5" />
            {validateLoading ? "Validating..." : "Validate recipe"}
          </Button>
          <Button
            type="button"
            onClick={onRun}
            disabled={loading || isFullRunNameMissing}
            className="corner-squircle"
          >
            <HugeiconsIcon icon={CookBookIcon} className="size-3.5" />
            {loading ? "Starting..." : `Start ${kindLabel.toLowerCase()}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
