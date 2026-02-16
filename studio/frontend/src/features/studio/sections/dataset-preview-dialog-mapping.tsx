import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import type { CheckFormatResponse } from "@/features/training/types/datasets";
import { cn } from "@/lib/utils";
import { AlertCircleIcon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function HeaderPick({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground cursor-pointer select-none">
      <Checkbox
        checked={checked}
        onCheckedChange={(v) => onCheckedChange(v === true)}
        aria-label={label}
        className="h-3.5 w-3.5"
      />
      <span>{label}</span>
    </label>
  );
}

export function DatasetMappingCard({
  leftLabel,
  rightLabel,
  mappingOk,
  input,
  output,
}: {
  leftLabel: string;
  rightLabel: string;
  mappingOk: boolean;
  input: string | null;
  output: string | null;
}) {
  return (
    <div
      className={cn(
        "rounded-xl corner-squircle ring-1 px-5 py-4 mb-4",
        mappingOk
          ? "ring-emerald-200/70 bg-emerald-50/70 text-emerald-950 dark:ring-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-50"
          : "ring-amber-200/70 bg-amber-50/70 text-amber-950 dark:ring-amber-900/50 dark:bg-amber-950/30 dark:text-amber-50",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "rounded-xl corner-squircle p-2 shrink-0",
            mappingOk ? "bg-emerald-500/15" : "bg-amber-500/15",
          )}
        >
          <HugeiconsIcon
            icon={mappingOk ? CheckmarkCircle02Icon : AlertCircleIcon}
            className={cn(
              "size-4",
              mappingOk
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-amber-700 dark:text-amber-300",
            )}
          />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold tracking-tight">
            {mappingOk ? "Mapping ready" : "Map dataset columns"}
          </p>
          <p
            className={cn(
              "text-xs mt-0.5",
              mappingOk
                ? "text-emerald-800/80 dark:text-emerald-200/80"
                : "text-amber-800/80 dark:text-amber-200/80",
            )}
          >
            {mappingOk
              ? "Looks good. We'll convert this dataset automatically."
              : `We couldn't auto-detect the format. Pick the ${leftLabel.toLowerCase()} column and the ${rightLabel.toLowerCase()} column. We'll convert it to a supported format automatically.`}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Badge
              variant="outline"
              className="h-6 text-[11px] bg-white/60 dark:bg-transparent"
            >
              {leftLabel}: <span className="font-mono">{input ?? "--"}</span>
            </Badge>
            <Badge
              variant="outline"
              className="h-6 text-[11px] bg-white/60 dark:bg-transparent"
            >
              {rightLabel}: <span className="font-mono">{output ?? "--"}</span>
            </Badge>
          </div>
          {!mappingOk && (
            <p className="mt-2 text-xs text-amber-800/80 dark:text-amber-200/80">
              Select exactly 1 {leftLabel.toLowerCase()} and 1{" "}
              {rightLabel.toLowerCase()} column to continue.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export function DatasetMappingFooter({
  leftLabel,
  rightLabel,
  mappingOk,
  isStarting,
  startError,
  onCancel,
  onStartTraining,
}: {
  leftLabel: string;
  rightLabel: string;
  mappingOk: boolean;
  isStarting: boolean;
  startError: string | null;
  onCancel: () => void;
  onStartTraining: () => Promise<void>;
}) {
  return (
    <div className="mt-3 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] text-muted-foreground/70 leading-relaxed">
          Tip: you can click {leftLabel} / {rightLabel} in the headers.
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer"
            onClick={onCancel}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            className="cursor-pointer"
            disabled={!mappingOk || isStarting}
            onClick={() => void onStartTraining()}
          >
            {isStarting ? "Starting..." : "Continue"}
          </Button>
        </div>
      </div>

      {startError && (
        <p className="text-xs text-red-500 leading-relaxed text-center">
          {startError}
        </p>
      )}
    </div>
  );
}

export function deriveDefaultMapping(
  data: CheckFormatResponse,
  isVlm: boolean,
): { input: string | null; output: string | null } {
  const input = isVlm
    ? data.detected_image_column ?? pickRole(data.suggested_mapping, "image")
    : pickRole(data.suggested_mapping, "user");
  const output = isVlm
    ? data.detected_text_column ?? pickRole(data.suggested_mapping, "text")
    : pickRole(data.suggested_mapping, "assistant");

  if (input && output && input === output) {
    return { input, output: null };
  }

  return { input: input ?? null, output: output ?? null };
}

function pickRole(
  mapping: Record<string, string> | null | undefined,
  role: string,
): string | null {
  if (!mapping) return null;
  for (const [col, mappedRole] of Object.entries(mapping)) {
    if (mappedRole === role) return col;
  }
  return null;
}
