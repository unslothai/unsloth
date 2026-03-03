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
import { CookBookIcon, TestTube01Icon } from "@hugeicons/core-free-icons";
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
      className={
        validateResult.valid
          ? "space-y-1 rounded-xl border border-emerald-300 bg-emerald-50 p-3"
          : "space-y-1 rounded-xl border border-destructive/30 bg-destructive/5 p-3"
      }
    >
      <p
        className={
          validateResult.valid
            ? "text-xs font-semibold uppercase text-emerald-700"
            : "text-xs font-semibold uppercase text-destructive"
        }
      >
        {validateResult.valid ? "Validation passed" : "Validation failed"}
      </p>
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
        className="corner-squircle sm:max-w-2xl shadow-border"
      >
        <DialogHeader>
          <DialogTitle>{kindLabel} settings</DialogTitle>
          <p className="text-sm text-muted-foreground">
            Configure run size and performance knobs for this execution.
          </p>
        </DialogHeader>

        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-foreground">Preview mode</span>
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
            />
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
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-medium">Enable batching</span>
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
                  <span className="font-medium">
                    Merge batches to one parquet
                  </span>
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
            <div className="grid gap-4 md:grid-cols-2">
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
              <div className="flex items-center gap-3 text-sm text-foreground">
                <Switch
                  checked={settings.disableEarlyShutdown}
                  onCheckedChange={(checked) =>
                    onSettingsChange({
                      disableEarlyShutdown: Boolean(checked),
                    })
                  }
                />
                Disable early shutdown
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {errors.length > 0 && (
          <div className="max-h-44 space-y-1 overflow-y-auto rounded-xl border border-destructive/30 bg-destructive/5 p-3">
            <p className="text-xs font-semibold uppercase text-destructive">
              Run checks
            </p>
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
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onValidate}
            disabled={loading || validateLoading}
          >
            <HugeiconsIcon icon={TestTube01Icon} className="size-3.5" />
            {validateLoading ? "Validating..." : "Validate recipe"}
          </Button>
          <Button type="button" onClick={onRun} disabled={loading}>
            <HugeiconsIcon icon={CookBookIcon} className="size-3.5" />
            {loading ? "Starting..." : `Start ${kindLabel.toLowerCase()}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
