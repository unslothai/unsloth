import { type ReactElement, useMemo, useState } from "react";
import {
  CookBookIcon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
import type { RecipeExecutionKind } from "../execution-types";
import type { RecipeRunSettings } from "../stores/recipe-executions";
import { FieldLabel } from "./shared/field-label";

type RunDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  kind: RecipeExecutionKind;
  onKindChange: (kind: RecipeExecutionKind) => void;
  rows: number;
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

export function RunDialog({
  open,
  onOpenChange,
  kind,
  onKindChange,
  rows,
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
  const rowHint = useMemo(
    () =>
      kind === "preview"
        ? "How many sample rows to generate for a quick check."
        : "How many rows to generate in total.",
    [kind],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle sm:max-w-2xl"
      >
        <DialogHeader>
          <DialogTitle>{kindLabel} settings</DialogTitle>
          <p className="text-sm text-muted-foreground">
            Configure run size and performance knobs for this execution.
          </p>
        </DialogHeader>

        <label className="flex items-center justify-between rounded-xl border bg-muted/20 px-3 py-2 text-sm">
          <span className="font-medium text-foreground">Preview mode</span>
          <Switch
            checked={kind === "preview"}
            onCheckedChange={(checked) =>
              onKindChange(checked ? "preview" : "full")
            }
          />
        </label>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="grid gap-2">
            <FieldLabel
              label="Records"
              htmlFor="run-rows"
              hint={rowHint}
            />
            <Input
              id="run-rows"
              type="number"
              min={1}
              max={200000}
              value={String(rows)}
              onChange={(event) => {
                const parsed = Number(event.target.value);
                if (!Number.isFinite(parsed)) {
                  return;
                }
                onRowsChange(clampInt(parsed, 1, 200000));
              }}
            />
          </div>
          <div className="grid gap-2">
            <FieldLabel
              label="Batch size"
              htmlFor="run-buffer-size"
              hint="Rows handled per batch. Bigger can be faster; smaller uses less memory."
            />
            <Input
              id="run-buffer-size"
              type="number"
              min={1}
              max={200000}
              value={String(settings.bufferSize)}
              onChange={(event) => {
                const parsed = Number(event.target.value);
                if (!Number.isFinite(parsed)) {
                  return;
                }
                onSettingsChange({
                  bufferSize: clampInt(parsed, 1, 200000),
                });
              }}
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
              type="number"
              min={1}
              max={2048}
              placeholder="Use model config"
              value={settings.llmParallelRequests ?? ""}
              onChange={(event) => {
                const value = event.target.value.trim();
                if (!value) {
                  onSettingsChange({ llmParallelRequests: null });
                  return;
                }
                const parsed = Number(value);
                if (!Number.isFinite(parsed)) {
                  return;
                }
                onSettingsChange({
                  llmParallelRequests: clampInt(parsed, 1, 2048),
                });
              }}
            />
          </div>
        </div>

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
              <div className="grid gap-2">
                <FieldLabel
                  label="CPU workers"
                  htmlFor="run-non-inference-workers"
                  hint="Worker threads for non-LLM steps like samplers and expressions."
                />
                <Input
                  id="run-non-inference-workers"
                  type="number"
                  min={1}
                  max={2048}
                  value={String(settings.nonInferenceWorkers)}
                  onChange={(event) => {
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) {
                      return;
                    }
                    onSettingsChange({
                      nonInferenceWorkers: clampInt(parsed, 1, 2048),
                    });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="Error window"
                  htmlFor="run-shutdown-window"
                  hint="How many attempts to observe before early-stop checks kick in."
                />
                <Input
                  id="run-shutdown-window"
                  type="number"
                  min={1}
                  max={10000}
                  value={String(settings.shutdownErrorWindow)}
                  onChange={(event) => {
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) {
                      return;
                    }
                    onSettingsChange({
                      shutdownErrorWindow: clampInt(parsed, 1, 10000),
                    });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="Conversation restarts"
                  htmlFor="run-max-restarts"
                  hint="How many full retries to do if model output fails validation."
                />
                <Input
                  id="run-max-restarts"
                  type="number"
                  min={0}
                  max={100}
                  value={String(settings.maxConversationRestarts)}
                  onChange={(event) => {
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) {
                      return;
                    }
                    onSettingsChange({
                      maxConversationRestarts: clampInt(parsed, 0, 100),
                    });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="Correction steps"
                  htmlFor="run-correction-steps"
                  hint="Extra in-chat fix attempts before a full retry."
                />
                <Input
                  id="run-correction-steps"
                  type="number"
                  min={0}
                  max={100}
                  value={String(settings.maxConversationCorrectionSteps)}
                  onChange={(event) => {
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) {
                      return;
                    }
                    onSettingsChange({
                      maxConversationCorrectionSteps: clampInt(parsed, 0, 100),
                    });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="Shutdown error rate"
                  htmlFor="run-shutdown-rate"
                  hint="Stop early if failure rate passes this value. Example: 0.5 = 50%."
                />
                <Input
                  id="run-shutdown-rate"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={String(settings.shutdownErrorRate)}
                  onChange={(event) => {
                    const parsed = Number(event.target.value);
                    if (!Number.isFinite(parsed)) {
                      return;
                    }
                    onSettingsChange({
                      shutdownErrorRate: clampFloat(parsed, 0, 1),
                    });
                  }}
                />
              </div>
              <label className="flex items-center gap-3 text-sm text-foreground">
                <Switch
                  checked={settings.disableEarlyShutdown}
                  onCheckedChange={(checked) =>
                    onSettingsChange({
                      disableEarlyShutdown: Boolean(checked),
                    })
                  }
                />
                Disable early shutdown
              </label>
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

        {validateResult && (
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
        )}

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
