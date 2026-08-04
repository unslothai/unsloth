// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * VRAM bar for a downloaded model row: weights, then KV cache, then the MTP
 * draft reserve.
 *
 * Deliberately shaped like the live monitor's meter -- same height, same track
 * -- so "how full is my GPU" looks the same wherever it's asked. Colours come
 * from the user's Appearance settings (`--primary`, `--foreground`), so it
 * re-themes with everything else instead of pinning its own palette.
 */

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

/**
 * Self-sizing bar for one model. Use this inside a `.map()`, where calling the
 * hook directly would break the rules of hooks. Renders nothing if the model
 * can't be sized.
 */
export function ModelMemoryBarFor({
  gpuGb,
  showReadout,
  className,
  ...source
}: ModelMemorySource & {
  gpuGb?: number | null;
  showReadout?: boolean;
  className?: string;
}) {
  const segments = useModelMemory(source, gpuGb);
  if (segments.status === "unknown") return null;
  return (
    <ModelMemoryBar
      segments={segments}
      showReadout={showReadout}
      className={className}
    />
  );
}

/**
 * Minimum width for a segment that isn't zero. On a 128 GB host a 0.7 GB KV
 * cache works out to two pixels, which reads as "nothing here" rather than
 * "small". Presentational only -- the percentages behind it stay exact.
 */
const MIN_SEGMENT_PX = 3;

/**
 * The MTP reserve is a darker shade of the accent rather than its own colour:
 * it's a second set of weights, so it belongs visually with the weights segment
 * while staying distinguishable from it.
 */
const SPEC_COLOR = "color-mix(in oklab, var(--primary) 62%, black)";

/**
 * Segment colours per pressure band. Below 80% the bar keeps the user's accent;
 * above it the hue takes over to signal how tight things are. Each band keeps
 * three distinguishable steps so the weights / KV / draft split survives.
 */
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
  className,
}: {
  segments: ModelMemorySegments;
  /** Print the GB breakdown beside the bar. For roomy surfaces (the hub card),
   *  where a two-pixel segment alone cannot convey the numbers. */
  showReadout?: boolean;
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
  // Two failures, two fixes: oversized weights need a smaller quant, while a
  // total that only overflows with context needs a shorter context or a
  // quantized KV cache. Both get said -- staying silent on the first reads as
  // "this is fine" on a model that can't load at all.
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

  // llama.cpp reserves the whole KV cache up front -- 131072 context allocates
  // 131072 cells before a token arrives -- so the bar charts the reservation.
  // The rate is what says whether a shorter context would help.
  const perTokenLine =
    kvBytesPerToken > 0
      ? t("modelMemory.kvRate", { rate: formatKvRate(kvBytesPerToken) })
      : null;

  return (
    <div className={cn("mt-1 w-full", className)}>
      {/* Decorative: the row already carries a Radix tooltip, so a native
          `title` here would stack a second one on hover, and an aria-label on a
          descendant would be concatenated into the row button's accessible
          name. The numbers stay reachable through `showReadout`. */}
      <div
        aria-hidden="true"
        className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted"
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
