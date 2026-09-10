// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown } from "lucide-react";
import { useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { MemoryEstimate } from "../api/memory-estimate";
import {
  type MemoryFitVerdict,
  formatMemoryGb,
  glueNoteItems,
  memoryFigureCandidates,
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

/** Match the size and type of the surrounding numeric controls. */
function MemoryFigure({
  label,
  bytes,
  bounded,
  tone,
}: {
  label: string;
  bytes: number;
  bounded: boolean;
  tone?: string;
}) {
  const candidates = useMemo(
    () => memoryFigureCandidates(bytes, bounded),
    [bytes, bounded],
  );
  const value = candidates[0];
  const [displayIndex, setDisplayIndex] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const measureRef = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const button = buttonRef.current;
    const measurement = measureRef.current;
    if (!button || !measurement) return;
    let active = true;
    const fit = () => {
      if (!active) return;
      const style = getComputedStyle(button);
      const width =
        button.clientWidth -
        Number.parseFloat(style.paddingLeft) -
        Number.parseFloat(style.paddingRight);
      const index = Array.from(measurement.children).findIndex(
        (child) => child.getBoundingClientRect().width <= width,
      );
      setDisplayIndex(index < 0 ? candidates.length - 1 : index);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(button);
    observer.observe(measurement);
    void document.fonts.ready.then(fit);
    return () => {
      active = false;
      observer.disconnect();
    };
  }, [candidates]);
  return (
    <div className="flex min-h-8 min-w-0 items-center justify-between gap-3">
      <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
        {label}
      </span>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            ref={buttonRef}
            type="button"
            aria-label={`${label}: ${value}`}
            className={`relative inline-flex h-8 w-[80px] shrink-0 cursor-default! items-center justify-center overflow-hidden rounded-full border-transparent bg-black/[0.04] px-2 text-ui-13 font-medium tabular-nums focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 dark:bg-white/[0.05] ${tone ?? "text-nav-fg"}`}
          >
            <span aria-hidden="true" className="min-w-0 truncate">
              {candidates[displayIndex] ?? value}
            </span>
            <span
              ref={measureRef}
              aria-hidden="true"
              className="pointer-events-none invisible absolute left-0 top-0 flex w-max flex-col items-start whitespace-nowrap"
            >
              {candidates.map((candidate) => (
                <span key={candidate}>{candidate}</span>
              ))}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent>{value}</TooltipContent>
      </Tooltip>
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
  freeGpuCapacityKnown,
  freeGpuReserveDeficitGb,
  usableSystemRamGb,
  usableSystemRamKnown,
  systemRamReserveDeficitGb,
  isUnifiedMemory,
  singleMemoryPool,
  reclaimableTotalBytes,
  reclaimableGpuBytes,
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
  freeGpuCapacityKnown?: boolean;
  freeGpuReserveDeficitGb?: number;
  /** Available host RAM minus the loader reserve, in GiB. */
  usableSystemRamGb: number;
  usableSystemRamKnown?: boolean;
  systemRamReserveDeficitGb?: number;
  isUnifiedMemory: boolean;
  /** Whether GPU and CPU share one memory pool. */
  singleMemoryPool: boolean;
  /** What the resident copy of this model hands back when it is unloaded for the reload. */
  reclaimableTotalBytes?: number;
  reclaimableGpuBytes?: number;
  expanded: boolean;
  onExpandedChange: (next: boolean) => void;
}) {
  const contentId = useId();
  if (!estimate?.available) {
    // Hide unavailable estimates without flickering during loading.
    return null;
  }
  const { gpuFit, totalFit, cpuOnly, bounded, advisory } = resolveMemoryFit(
    estimate,
    {
      gpuCapacityGb,
      totalCapacityGb,
      systemRamCapacityGb,
      freeGpuCapacityGb,
      freeGpuCapacityKnown,
      freeGpuReserveDeficitGb,
      usableSystemRamGb,
      usableSystemRamKnown,
      systemRamReserveDeficitGb,
      singleMemoryPool,
      reclaimableTotalBytes,
      reclaimableGpuBytes,
    },
  );
  const kvNote = resolveKvNote(estimate);
  const draftCacheNote = resolveDraftCacheNote(
    estimate.drafterRuntimeGpuBytes,
    estimate.drafterRuntimeBytes,
  );
  return (
    <div className="flex flex-col border-b border-border/60 pb-3.5">
      <button
        type="button"
        onClick={() => onExpandedChange(!expanded)}
        aria-expanded={expanded}
        aria-controls={contentId}
        aria-label={`Estimated Memory Usage: ${expanded ? "Hide" : "Show"} breakdown`}
        className="group flex min-h-7 w-full items-center justify-between gap-3 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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
            cpuOnly
              ? "RAM"
              : singleMemoryPool
                ? isUnifiedMemory
                  ? "Unified"
                  : "Shared"
                : "GPU"
          }
          bytes={
            singleMemoryPool || cpuOnly
              ? estimate.totalBytes
              : estimate.gpuBytes
          }
          bounded={bounded}
          tone={
            MEMORY_VALUE_TONE[singleMemoryPool || cpuOnly ? totalFit : gpuFit]
          }
        />
        {singleMemoryPool || cpuOnly ? null : (
          <MemoryFigure
            label="Total"
            bytes={estimate.totalBytes}
            bounded={bounded}
            tone={MEMORY_VALUE_TONE[totalFit]}
          />
        )}
      </div>
      <div id={contentId} hidden={!expanded} className="mt-3 space-y-3">
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
          className={`mt-1.5 text-pretty text-ui-12 leading-relaxed ${advisory.tone === "warn" ? "text-amber-700 dark:text-amber-400" : "text-muted-foreground"}`}
        >
          {advisory.text}
        </p>
      )}
    </div>
  );
}
