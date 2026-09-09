// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown } from "lucide-react";
import { useId } from "react";
import type { MemoryEstimate } from "../api/memory-estimate";
import {
  type MemoryFitVerdict,
  formatMemoryGb,
  glueNoteItems,
  resolveDraftCacheNote,
  resolveKvNote,
  resolveMemoryFit,
} from "../model-config/memory-fit";

const MEMORY_VALUE_TONE: Record<MemoryFitVerdict, string> = {
  fits: "text-nav-fg",
  tight: "text-amber-500",
  exceeds: "text-red-500",
  unknown: "text-nav-fg",
};

/** Match the size, padding, and type of the surrounding numeric controls. */
function MemoryFigure({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="flex min-h-8 min-w-0 items-center justify-between gap-3">
      <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
        {label}
      </span>
      <span
        title={value}
        className={`inline-flex h-8 w-[92px] shrink-0 items-center justify-end rounded-full border-transparent bg-black/[0.04] pl-3 pr-2 text-ui-13 font-medium tabular-nums dark:bg-white/[0.05] ${tone ?? "text-nav-fg"}`}
      >
        <span className="min-w-0 truncate">{value}</span>
      </span>
    </div>
  );
}

function MemoryBreakdownLine({
  label,
  value,
  note,
  muted,
}: {
  label: string;
  value: string;
  note?: string;
  muted?: boolean;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-1">
      <span className="min-w-0 text-ui-12 text-muted-foreground">{label}</span>
      <span
        className={`whitespace-nowrap pr-2 text-ui-12 tabular-nums ${muted ? "text-muted-foreground" : "text-nav-fg"}`}
      >
        {value}
      </span>
      {note ? (
        <span className="col-span-2 min-w-0 text-ui-11 leading-relaxed text-muted-foreground/80">
          {glueNoteItems(note)}
        </span>
      ) : null}
    </div>
  );
}

/** Show the loader's estimate and an optional breakdown for the requested settings. */
export function MemoryEstimateRow({
  estimate,
  loading,
  stale,
  gpuCapacityGb,
  totalCapacityGb,
  systemRamCapacityGb,
  freeGpuCapacityGb,
  usableSystemRamGb,
  isUnifiedMemory,
  singleMemoryPool,
  expanded,
  onExpandedChange,
}: {
  estimate: MemoryEstimate | null;
  loading: boolean;
  stale: boolean;
  /** GPU or shared capacity in GiB; 0 means unknown. */
  gpuCapacityGb: number;
  /** Combined capacity in GiB; 0 means unknown. */
  totalCapacityGb: number;
  /** Host RAM capacity in GiB. */
  systemRamCapacityGb: number;
  /** Free GPU memory in GiB; warnings only, since replacement can free memory. */
  freeGpuCapacityGb: number;
  /** Available host RAM minus the loader reserve, in GiB. */
  usableSystemRamGb: number;
  isUnifiedMemory: boolean;
  /** Whether GPU and CPU share one memory pool. */
  singleMemoryPool: boolean;
  expanded: boolean;
  onExpandedChange: (next: boolean) => void;
}) {
  const contentId = useId();
  if (!estimate?.available) {
    // Hide unavailable estimates without flickering during loading.
    return null;
  }
  const { gpuFit, totalFit, prefix, advisory } = resolveMemoryFit(estimate, {
    gpuCapacityGb,
    totalCapacityGb,
    systemRamCapacityGb,
    freeGpuCapacityGb,
    usableSystemRamGb,
    singleMemoryPool,
  });
  const kvNote = resolveKvNote(estimate);
  const draftCacheNote = resolveDraftCacheNote(
    estimate.drafterRuntimeGpuBytes,
    estimate.drafterRuntimeBytes,
  );
  return (
    <div className="space-y-4 border-b border-border/60 pb-5">
      <button
        type="button"
        onClick={() => onExpandedChange(!expanded)}
        aria-expanded={expanded}
        aria-controls={contentId}
        aria-label={`Estimated Memory Usage: ${expanded ? "Hide" : "Show"} breakdown`}
        className="group flex min-h-8 w-full items-center justify-between gap-3 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
            Estimated Memory Usage
          </span>
          <span className="shrink-0 rounded-md bg-black/[0.04] px-1.5 py-0.5 text-ui-10 font-medium uppercase tracking-wide text-muted-foreground dark:bg-white/[0.06]">
            Beta
          </span>
        </span>
        <ChevronDown
          aria-hidden="true"
          className={`size-3.5 shrink-0 text-muted-foreground transition-transform duration-200 group-hover:text-nav-fg motion-reduce:transition-none ${expanded ? "rotate-0" : "-rotate-90"}`}
        />
      </button>
      <div
        aria-busy={loading || stale}
        className={`space-y-3 transition-opacity ${stale || loading ? "opacity-50" : ""}`}
      >
        {/* A shared pool uses the total, regardless of CPU offloading. */}
        <MemoryFigure
          label={
            singleMemoryPool ? (isUnifiedMemory ? "Unified" : "Shared") : "GPU"
          }
          value={`${prefix}${formatMemoryGb(
            singleMemoryPool ? estimate.totalBytes : estimate.gpuBytes,
          )}`}
          tone={MEMORY_VALUE_TONE[singleMemoryPool ? totalFit : gpuFit]}
        />
        {singleMemoryPool ? null : (
          <MemoryFigure
            label="Total"
            value={`${prefix}${formatMemoryGb(estimate.totalBytes)}`}
            tone={MEMORY_VALUE_TONE[totalFit]}
          />
        )}
      </div>
      <div id={contentId} hidden={!expanded} className="space-y-3">
        <MemoryBreakdownLine
          label="Weights"
          value={formatMemoryGb(estimate.weightsBytes)}
          note={
            estimate.gpuLayers != null && estimate.layerCount != null
              ? `${estimate.gpuLayers} of ${estimate.layerCount + 1} layers on GPU`
              : undefined
          }
        />
        <MemoryBreakdownLine
          label="KV cache"
          value={
            estimate.kvEstimable ? formatMemoryGb(estimate.kvBytes) : "unknown"
          }
          note={estimate.kvEstimable ? kvNote : undefined}
          muted={!estimate.kvEstimable}
        />
        <MemoryBreakdownLine
          label="Compute buffers"
          value={formatMemoryGb(estimate.computeBytes)}
        />
        {/* Projector weights are already included above. */}
        {estimate.projectorRuntimeBytes > 0 && (
          <MemoryBreakdownLine
            label="Vision encoder"
            value={formatMemoryGb(estimate.projectorRuntimeBytes)}
          />
        )}
        {/* Draft cache is additional to the drafter's weights. */}
        {estimate.drafterRuntimeBytes > 0 && (
          <MemoryBreakdownLine
            label="Draft cache"
            value={formatMemoryGb(estimate.drafterRuntimeBytes)}
            note={draftCacheNote}
          />
        )}
      </div>
      {advisory && (
        <p
          className={`text-pretty text-ui-12 leading-relaxed ${advisory.tone === "warn" ? "text-amber-700 dark:text-amber-400" : "text-muted-foreground"}`}
        >
          {advisory.text}
        </p>
      )}
    </div>
  );
}
