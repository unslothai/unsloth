// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** VRAM bar: weights, KV cache, MTP draft reserve. Shaped like the live monitor's meter. */

import {
  type ModelMemorySource,
  useModelMemory,
} from "@/hooks/use-model-memory";
import { useT } from "@/i18n";
import {
  type ModelMemoryPressure,
  type ModelMemorySegments,
  formatKvRate,
  formatMemoryGb,
} from "@/lib/model-memory";
import { cn } from "@/lib/utils";

/** For use inside `.map()`, where calling the hook directly would break the rules of hooks. */
export function ModelMemoryBarFor({
  gpuGb,
  showReadout,
  compact,
  className,
  ...source
}: ModelMemorySource & {
  gpuGb?: number | null;
  showReadout?: boolean;
  compact?: boolean;
  className?: string;
}) {
  const segments = useModelMemory(source, gpuGb);
  if (segments.status === "unknown") return null;
  return (
    <ModelMemoryBar
      segments={segments}
      showReadout={showReadout}
      compact={compact}
      className={className}
    />
  );
}

/** Non-zero segments get a minimum width; the percentages behind them stay exact. */
const MIN_SEGMENT_PX = 3;

const SPEC_COLOR = "color-mix(in oklab, var(--primary) 62%, black)";

const SEGMENT_COLORS: Record<
  ModelMemoryPressure,
  { weights: string; kv: string; spec: string }
> = {
  normal: {
    weights: "var(--primary)",
    kv: "var(--foreground)",
    spec: SPEC_COLOR,
  },
  high: {
    weights: "var(--color-amber-500, #f59e0b)",
    kv: "var(--color-amber-600, #d97706)",
    spec: "var(--color-amber-800, #92400e)",
  },
  critical: {
    weights: "var(--destructive)",
    kv: "color-mix(in oklab, var(--destructive) 78%, black)",
    spec: "color-mix(in oklab, var(--destructive) 55%, black)",
  },
};

export function ModelMemoryBar({
  segments,
  showReadout = false,
  compact = false,
  className,
}: {
  segments: ModelMemorySegments;
  showReadout?: boolean;
  compact?: boolean;
  className?: string;
}) {
  const t = useT();
  if (segments.status === "unknown") return null;

  const {
    modelPct,
    kvPct,
    specPct,
    modelGb,
    kvGb,
    specGb,
    totalGb,
    budgetGb,
    kvBytesPerToken,
    pressure,
  } = segments;
  const colors = SEGMENT_COLORS[pressure];
  // Oversized weights need a smaller quant; context overflow needs shorter context or a quantized KV.
  const warning =
    segments.status === "model-exceeds"
      ? t("modelMemory.tooLarge")
      : segments.status === "context-exceeds"
        ? t("modelMemory.oomLikely")
        : null;

  const readout =
    specGb > 0
      ? t("modelMemory.readoutWithSpec", {
          model: formatMemoryGb(modelGb),
          kv: formatMemoryGb(kvGb),
          spec: formatMemoryGb(specGb),
          total: formatMemoryGb(totalGb),
          budget: formatMemoryGb(budgetGb),
        })
      : t("modelMemory.readout", {
          model: formatMemoryGb(modelGb),
          context: formatMemoryGb(kvGb + specGb),
          total: formatMemoryGb(totalGb),
          budget: formatMemoryGb(budgetGb),
        });

  // llama.cpp reserves the whole KV cache up front, so the bar charts the reservation.
  const perTokenLine =
    kvBytesPerToken > 0
      ? t("modelMemory.kvRate", { rate: formatKvRate(kvBytesPerToken) })
      : null;

  return (
    <div className={cn("mt-1 w-full", className)}>
      {/* Decorative: the row has a Radix tooltip, and an aria-label here would join the button's name. */}
      <div
        aria-hidden="true"
        className={cn(
          "flex w-full overflow-hidden rounded-full bg-muted",
          compact ? "h-[3px]" : "h-1.5",
          compact && pressure === "normal" && "opacity-55",
        )}
      >
        <div
          data-testid="model-memory-weights"
          className="h-full"
          style={{
            width: `${modelPct}%`,
            minWidth: modelGb > 0 ? MIN_SEGMENT_PX : 0,
            backgroundColor: colors.weights,
          }}
        />
        <div
          data-testid="model-memory-context"
          className="h-full"
          style={{
            width: `${kvPct}%`,
            minWidth: kvGb > 0 ? MIN_SEGMENT_PX : 0,
            backgroundColor: colors.kv,
          }}
        />
        <div
          data-testid="model-memory-spec"
          className="h-full"
          style={{
            width: `${specPct}%`,
            minWidth: specGb > 0 ? MIN_SEGMENT_PX : 0,
            backgroundColor: colors.spec,
          }}
        />
      </div>
      {showReadout ? (
        <>
          <p className="mt-1 text-ui-10 text-muted-foreground tabular-nums">
            {readout}
          </p>
          {perTokenLine ? (
            <p className="text-ui-10 text-muted-foreground/80 tabular-nums">
              {perTokenLine}
            </p>
          ) : null}
        </>
      ) : null}
      {warning ? (
        <p
          data-testid="model-memory-warning"
          className={cn(
            "mt-1 text-ui-11",
            segments.status === "model-exceeds"
              ? "text-rose-600 dark:text-rose-400"
              : "text-amber-600 dark:text-amber-500",
          )}
        >
          {warning}
        </p>
      ) : null}
    </div>
  );
}
