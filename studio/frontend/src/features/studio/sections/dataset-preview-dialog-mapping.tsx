import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { CheckFormatResponse } from "@/features/training/types/datasets";

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
    <div className="rounded-xl corner-squircle ring-1 ring-amber-200/70 bg-amber-50/70 px-5 py-4 mb-4 text-amber-950 dark:ring-amber-900/50 dark:bg-amber-950/30 dark:text-amber-50">
      <div className="flex items-start gap-3">
        <div className="rounded-xl corner-squircle bg-amber-500/15 p-2 shrink-0">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="size-4 text-amber-700 dark:text-amber-300"
          />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold tracking-tight">Map dataset columns</p>
          <p className="text-xs text-amber-800/80 dark:text-amber-200/80 mt-0.5">
            We couldn&apos;t auto-detect the format. Pick the{" "}
            {leftLabel.toLowerCase()} column and the {rightLabel.toLowerCase()}{" "}
            column. We&apos;ll convert it to a supported format automatically.
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
  if (isVlm) {
    const input =
      data.detected_image_column ?? pickRole(data.suggested_mapping, "image");
    const output =
      data.detected_text_column ?? pickRole(data.suggested_mapping, "text");
    return { input: input ?? null, output: output ?? null };
  }
  const input = pickRole(data.suggested_mapping, "user");
  const output = pickRole(data.suggested_mapping, "assistant");
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
